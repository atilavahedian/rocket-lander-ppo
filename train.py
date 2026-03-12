from pathlib import Path
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


class RocketEnv:
    def __init__(self):
        self.dt = 0.05
        self.gravity = 1.6
        self.max_steps = 500

        self.x_bound = 5.0
        self.y_bound = 10.0

        self.pad_x = 0.0
        self.pad_width = 2.0

        self.main_thrust = 2.6
        self.side_torque = 1.6

        self.max_fuel = 150.0
        self.main_fuel_cost = 1.0
        self.side_fuel_cost = 0.12

        self.max_landing_vx = 0.40
        self.max_landing_vy = 0.45
        self.max_landing_angle = 0.18
        self.max_landing_ang_vel = 0.40

        self.reset()

    @staticmethod
    def _sample_range(easy_low, easy_high, hard_low, hard_high, difficulty):
        low = easy_low + (hard_low - easy_low) * difficulty
        high = easy_high + (hard_high - easy_high) * difficulty
        return np.random.uniform(low, high)

    def reset(self, difficulty=1.0):
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        # Start easy and expand toward the original distribution over training.
        self.x = self._sample_range(-0.8, 0.8, -2.5, 2.5, difficulty)
        self.y = self._sample_range(2.5, 3.5, 3.5, 6.0, difficulty)
        self.vx = self._sample_range(-0.08, 0.08, -0.35, 0.35, difficulty)
        self.vy = self._sample_range(-0.12, 0.02, -0.30, 0.08, difficulty)
        self.angle = self._sample_range(-0.08, 0.08, -0.24, 0.24, difficulty)
        self.angular_velocity = self._sample_range(-0.05, 0.05, -0.20, 0.20, difficulty)

        self.fuel = self.max_fuel
        self.steps = 0
        self.done = False

        self.landed = False
        self.crashed = False
        self.out_of_bounds = False
        self.hit_ceiling = False
        self.timeout = False

        self.last_fuel_used = 0.0
        self.prev_potential = self._potential()
        return self._get_state()

    def _decode_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        throttle = np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0)
        torque = np.clip(action[1], -1.0, 1.0)
        return throttle, torque

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        self.steps += 1
        self.last_fuel_used = 0.0

        throttle, torque_cmd = self._decode_action(action)

        ax = 0.0
        ay = -self.gravity
        angular_acc = 0.0

        if self.fuel > 0.0:
            thrust = throttle * self.main_thrust
            fuel_used = 0.0

            main_cost = self.main_fuel_cost * throttle
            if thrust > 0.0 and self.fuel >= main_cost:
                ax += thrust * np.sin(self.angle)
                ay += thrust * np.cos(self.angle)
                fuel_used += main_cost

            if abs(torque_cmd) > 1e-6 and self.fuel >= fuel_used + self.side_fuel_cost:
                angular_acc += torque_cmd * self.side_torque
                fuel_used += self.side_fuel_cost

            self.fuel = max(0.0, self.fuel - fuel_used)
            self.last_fuel_used = fuel_used

        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.angular_velocity += angular_acc * self.dt

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.angle += self.angular_velocity * self.dt
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

        if self.y <= 0.0:
            self.y = 0.0
            self.done = True

            on_pad = abs(self.x - self.pad_x) <= self.pad_width / 2
            soft_vx = abs(self.vx) <= self.max_landing_vx
            soft_vy = abs(self.vy) <= self.max_landing_vy
            upright = abs(self.angle) <= self.max_landing_angle
            stable_spin = abs(self.angular_velocity) <= self.max_landing_ang_vel

            if on_pad and soft_vx and soft_vy and upright and stable_spin:
                self.landed = True
            else:
                self.crashed = True

        if not self.done:
            if self.y > self.y_bound:
                self.done = True
                self.out_of_bounds = True
                self.hit_ceiling = True
            elif abs(self.x) > self.x_bound:
                self.done = True
                self.out_of_bounds = True

        if not self.done and self.steps >= self.max_steps:
            self.done = True
            self.timeout = True

        reward = self._compute_reward()
        return self._get_state(), reward, self.done, {}

    def _potential(self):
        dist_x = abs(self.x - self.pad_x)
        return (
            2.0 * dist_x
            + 0.60 * self.y
            + 1.00 * abs(self.vx)
            + 1.30 * abs(self.vy)
            + 0.80 * abs(self.angle)
            + 0.25 * abs(self.angular_velocity)
        )

    def _compute_reward(self):
        potential = self._potential()
        progress = 4.0 * (self.prev_potential - potential)
        self.prev_potential = potential

        dist_x = abs(self.x - self.pad_x)
        reward = progress

        reward -= 0.05
        reward -= 0.03 * self.last_fuel_used
        reward -= 0.15 * abs(self.angular_velocity)

        if self.vy > 0.20:
            reward -= 2.5 * (self.vy - 0.20)

        if self.y < 2.0:
            reward += 0.8 * max(0.0, 1.0 - dist_x / (self.pad_width / 2 + 1e-6))
            reward += 0.6 * max(0.0, 1.0 - abs(self.vx) / self.max_landing_vx)
            reward += 0.8 * max(0.0, 1.0 - abs(self.vy) / self.max_landing_vy)
            reward += 0.6 * max(0.0, 1.0 - abs(self.angle) / self.max_landing_angle)
            reward += 0.4 * max(0.0, 1.0 - abs(self.angular_velocity) / self.max_landing_ang_vel)

            reward -= 4.0 * max(0.0, abs(self.vy) - self.max_landing_vy)
            reward -= 2.5 * max(0.0, abs(self.vx) - self.max_landing_vx)
            reward -= 2.0 * max(0.0, abs(self.angle) - self.max_landing_angle)

        if self.landed:
            reward += 250.0 + 0.25 * self.fuel
        elif self.crashed:
            impact = (
                abs(self.vx)
                + 1.5 * abs(self.vy)
                + 2.0 * abs(self.angle)
                + 0.5 * abs(self.angular_velocity)
            )
            reward -= 180.0 + 40.0 * impact + 15.0 * dist_x
        elif self.hit_ceiling:
            reward -= 320.0
        elif self.out_of_bounds:
            reward -= 260.0
        elif self.timeout:
            reward -= 200.0

        return float(reward)

    def _get_state(self):
        return np.array(
            [
                self.x / self.x_bound,
                self.y / self.y_bound,
                self.vx / 2.0,
                self.vy / 2.0,
                np.sin(self.angle),
                np.cos(self.angle),
                self.angular_velocity / 2.0,
                self.fuel / self.max_fuel,
            ],
            dtype=np.float32,
        )

    def render(self):
        print(
            f"step={self.steps} | "
            f"x={self.x:.2f}, y={self.y:.2f} | "
            f"vx={self.vx:.2f}, vy={self.vy:.2f} | "
            f"angle={self.angle:.2f}, ang_vel={self.angular_velocity:.2f} | "
            f"fuel={self.fuel:.1f}"
        )


class RunningNorm:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state):
        hidden = self.backbone(state)
        mu = self.mu_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        log_std = torch.clamp(self.log_std, -2.5, 0.5)
        std = torch.exp(log_std).expand_as(mu)
        return mu, std, value


class PPOAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=2,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.995,
        lam=0.97,
        eps_clip=0.2,
        update_epochs=15,
        minibatch_size=512,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        device=device,
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.obs_norm = RunningNorm(state_dim)
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def _state_tensor(self, state, already_normalized=False):
        state = np.asarray(state, dtype=np.float32)
        if not already_normalized:
            state = self.obs_norm.normalize(state)
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _normalized_state(self, state, update_stats=False):
        state = np.asarray(state, dtype=np.float32)
        if update_stats:
            self.obs_norm.update(state)
        return self.obs_norm.normalize(state)

    def select_action(self, state, deterministic=False, update_obs_stats=False):
        norm_state = self._normalized_state(state, update_stats=update_obs_stats)
        state_t = self._state_tensor(norm_state, already_normalized=True)

        with torch.no_grad():
            mu, std, value = self.policy(state_t)
            dist = Normal(mu, std)

            if deterministic:
                squashed_action = torch.tanh(mu)
                log_prob = torch.zeros(1, device=self.device)
            else:
                raw_action = dist.rsample()
                squashed_action = torch.tanh(raw_action)
                log_prob = dist.log_prob(raw_action) - torch.log(
                    1 - squashed_action.pow(2) + 1e-6
                )
                log_prob = log_prob.sum(dim=-1)

        return (
            squashed_action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
            norm_state,
        )

    def get_value(self, state):
        state_t = self._state_tensor(state)
        with torch.no_grad():
            _, _, value = self.policy(state_t)
        return float(value.item())

    def evaluate_actions(self, states, actions):
        mu, std, values = self.policy(states)
        dist = Normal(mu, std)

        actions = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * torch.log((1 + actions) / (1 - actions))

        log_probs = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values

    def compute_gae(self, rewards, dones, values, last_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout, last_value):
        states_np = np.asarray(rollout["norm_states"], dtype=np.float32)
        actions_np = np.asarray(rollout["actions"], dtype=np.float32)

        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions_np, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["dones"], rollout["values"], last_value
        )

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_states = len(states)
        indices = np.arange(num_states)
        stop_early = False

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_states, self.minibatch_size):
                batch_idx = indices[start:start + self.minibatch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_log_probs, entropy, values = self.evaluate_actions(batch_states, batch_actions)

                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.HuberLoss()(values, batch_returns)
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                if approx_kl > 1.5 * self.target_kl:
                    stop_early = True
                    break

            if stop_early:
                break


def make_rollout():
    return {
        "norm_states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "log_probs": [],
        "values": [],
    }


def evaluate_agent(agent, env, num_episodes=50):
    rewards = []
    successes = 0
    crashes = 0
    out_bounds = 0
    timeouts = 0

    for _ in range(num_episodes):
        state = env.reset(difficulty=1.0)
        done = False
        total_reward = 0.0

        while not done:
            action, _, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes += int(env.landed)
        crashes += int(env.crashed)
        out_bounds += int(env.out_of_bounds)
        timeouts += int(env.timeout)

    return {
        "success_rate": successes / num_episodes,
        "crash_rate": crashes / num_episodes,
        "out_of_bounds_rate": out_bounds / num_episodes,
        "timeout_rate": timeouts / num_episodes,
        "avg_reward": float(np.mean(rewards)),
    }


def expert_action(env):
    # Fuel-aware controller found by random search on the exact environment.
    desired_angle = np.clip(
        -1.1773275109747672 * env.x - 2.364154746301627 * env.vx,
        -0.14698488059532808,
        0.14698488059532808,
    )
    torque = np.clip(
        3.5746795356611845 * (desired_angle - env.angle)
        - 2.9039527970109353 * env.angular_velocity,
        -1.0,
        1.0,
    )

    if env.y > 3.6408115835312795:
        target_speed = 1.255917718969772
    elif env.y > 0.5227521441903312:
        target_speed = 0.7328326290250521
    else:
        target_speed = 0.17113364267331846

    max_up_accel = max(1e-3, env.main_thrust * max(np.cos(env.angle), 0.0) - env.gravity)
    braking_distance = max(
        0.0,
        ((max(0.0, -env.vy)) ** 2 - target_speed ** 2) / (2.0 * max_up_accel),
    )

    if env.y <= braking_distance + 0.01713677217781534:
        throttle = 1.0
    else:
        desired_acc = 2.3498962558001324 * ((-target_speed) - env.vy) + env.gravity
        thrust = desired_acc / max(np.cos(env.angle), 0.35)
        throttle = np.clip(thrust / env.main_thrust, 0.0, 1.0)

    return np.array([2.0 * throttle - 1.0, torque], dtype=np.float32)


def evaluate_expert(env, num_episodes=100):
    rewards = []
    successes = 0
    crashes = 0
    out_bounds = 0
    timeouts = 0

    for _ in range(num_episodes):
        state = env.reset(difficulty=1.0)
        done = False
        total_reward = 0.0

        while not done:
            action = expert_action(env)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        successes += int(env.landed)
        crashes += int(env.crashed)
        out_bounds += int(env.out_of_bounds)
        timeouts += int(env.timeout)

    return {
        "success_rate": successes / num_episodes,
        "crash_rate": crashes / num_episodes,
        "out_of_bounds_rate": out_bounds / num_episodes,
        "timeout_rate": timeouts / num_episodes,
        "avg_reward": float(np.mean(rewards)),
    }


def curriculum_difficulty(update, num_updates):
    warmup_updates = max(1, int(0.60 * num_updates))
    return min(1.0, 0.25 + 0.75 * update / warmup_updates)


def default_save_dir():
    if "ROCKET_RL_SAVE_DIR" in os.environ:
        return Path(os.environ["ROCKET_RL_SAVE_DIR"]).expanduser()
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return base_dir / "rocket_rl_runs"


def save_checkpoint(path, agent, extra=None):
    payload = {
        "model": agent.policy.state_dict(),
        "obs_mean": agent.obs_norm.mean,
        "obs_var": agent.obs_norm.var,
        "obs_count": agent.obs_norm.count,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, agent):
    checkpoint = torch.load(path, map_location=agent.device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        agent.policy.load_state_dict(checkpoint["model"])
        if {"obs_mean", "obs_var", "obs_count"} <= checkpoint.keys():
            agent.obs_norm.mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
            agent.obs_norm.var = np.asarray(checkpoint["obs_var"], dtype=np.float32)
            agent.obs_norm.count = float(checkpoint["obs_count"])
    else:
        # Backward compatibility for older checkpoints that only saved the model weights.
        agent.policy.load_state_dict(checkpoint)

    return checkpoint


def smoke_test():
    env = RocketEnv()
    agent = PPOAgent()

    state = env.reset(difficulty=0.25)
    for i in range(20):
        action, log_prob, value, norm_state = agent.select_action(
            state, deterministic=False, update_obs_stats=True
        )
        next_state, reward, done, _ = env.step(action)

        print(
            i,
            "action:", action,
            "log_prob:", log_prob,
            "value:", value,
            "reward:", reward,
            "done:", done,
        )

        assert np.all(np.isfinite(state)), "state has NaN/inf"
        assert np.all(np.isfinite(norm_state)), "normalized state has NaN/inf"
        assert np.all(np.isfinite(action)), "action has NaN/inf"
        assert np.isfinite(log_prob), "log_prob has NaN/inf"
        assert np.isfinite(value), "value has NaN/inf"
        assert np.isfinite(reward), "reward has NaN/inf"

        state = env.reset(difficulty=0.25) if done else next_state

    print("smoke test passed")


def pretrain_from_expert(agent, episodes=400, epochs=8, batch_size=2048, lr=1e-3):
    env = RocketEnv()
    states = []
    actions = []

    for episode in range(episodes):
        difficulty = 0.25 + 0.75 * episode / max(1, episodes - 1)
        state = env.reset(difficulty=difficulty)
        done = False

        while not done:
            action = expert_action(env)
            states.append(state.copy())
            actions.append(action.copy())
            state, _, done, _ = env.step(action)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    agent.obs_norm.update(states)
    norm_states = agent.obs_norm.normalize(states)

    states_t = torch.tensor(norm_states, dtype=torch.float32, device=agent.device)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=agent.device)

    optimizer = optim.Adam(
        list(agent.policy.backbone.parameters()) + list(agent.policy.mu_head.parameters()),
        lr=lr,
    )

    num_samples = len(states_t)
    indices = np.arange(num_samples)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]

            mu, _, _ = agent.policy(batch_states)
            predicted_actions = torch.tanh(mu)
            loss = nn.MSELoss()(predicted_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        mean_loss = epoch_loss / num_samples
        print(f"Expert pretrain epoch {epoch}/{epochs} | MSE: {mean_loss:.6f}")


def train():
    env = RocketEnv()
    agent = PPOAgent(
        lr=3e-4,
        gamma=0.995,
        lam=0.97,
        eps_clip=0.2,
        update_epochs=15,
        minibatch_size=512,
        ent_coef=0.002,
        vf_coef=0.5,
        target_kl=0.015,
    )

    total_timesteps = 524288
    rollout_steps = 2048
    eval_every_updates = 10
    eval_episodes = 100
    num_updates = total_timesteps // rollout_steps
    print("num_updates:", num_updates)

    episode_returns = []
    eval_success_history = []
    eval_reward_history = []

    best_success_rate = -1.0
    best_eval_reward = -float("inf")

    save_dir = default_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / "best_rocket_ppo_continuous.pth"
    final_model_path = save_dir / "final_rocket_ppo_continuous.pth"

    expert_metrics = evaluate_expert(RocketEnv(), num_episodes=100)
    print("expert baseline:", expert_metrics)

    pretrain_from_expert(agent)
    warm_start_metrics = evaluate_agent(agent, RocketEnv(), num_episodes=50)
    print("after expert pretrain:", warm_start_metrics)

    state = env.reset(difficulty=curriculum_difficulty(1, num_updates))
    current_episode_return = 0.0
    initial_lr = 3e-4

    for update in range(1, num_updates + 1):
        rollout = make_rollout()
        train_difficulty = curriculum_difficulty(update, num_updates)

        for _ in range(rollout_steps):
            action, log_prob, value, norm_state = agent.select_action(
                state, deterministic=False, update_obs_stats=True
            )
            next_state, reward, done, _ = env.step(action)

            rollout["norm_states"].append(norm_state)
            rollout["actions"].append(action)
            rollout["rewards"].append(reward)
            rollout["dones"].append(done)
            rollout["log_probs"].append(log_prob)
            rollout["values"].append(value)

            current_episode_return += reward
            state = next_state

            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                state = env.reset(difficulty=train_difficulty)

        last_value = 0.0 if rollout["dones"][-1] else agent.get_value(state)

        frac = 1.0 - (update - 1) / num_updates
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = initial_lr * frac

        agent.update(rollout, last_value)

        if update % eval_every_updates == 0:
            eval_env = RocketEnv()
            metrics = evaluate_agent(agent, eval_env, num_episodes=eval_episodes)

            eval_success_history.append(metrics["success_rate"])
            eval_reward_history.append(metrics["avg_reward"])

            recent_train = np.mean(episode_returns[-20:]) if episode_returns else float("nan")

            print(
                f"Update {update}/{num_updates} | "
                f"Difficulty: {train_difficulty:.2f} | "
                f"Episodes: {len(episode_returns)} | "
                f"RecentTrain(20): {recent_train:.2f} | "
                f"EvalSuccess: {metrics['success_rate']:.2f} | "
                f"Crash: {metrics['crash_rate']:.2f} | "
                f"OOB: {metrics['out_of_bounds_rate']:.2f} | "
                f"Timeout: {metrics['timeout_rate']:.2f} | "
                f"EvalReward: {metrics['avg_reward']:.2f}"
            )

            should_save = False
            if metrics["success_rate"] > best_success_rate:
                should_save = True
            elif metrics["success_rate"] == best_success_rate and metrics["avg_reward"] > best_eval_reward:
                should_save = True

            if should_save:
                best_success_rate = metrics["success_rate"]
                best_eval_reward = metrics["avg_reward"]
                save_checkpoint(
                    best_model_path,
                    agent,
                    extra={"metrics": metrics, "update": update},
                )

    save_checkpoint(
        final_model_path,
        agent,
        extra={
            "best_success_rate": best_success_rate,
            "best_eval_reward": best_eval_reward,
        },
    )

    print("best model:", best_model_path)
    print("final model:", final_model_path)

    return {
        "agent": agent,
        "episode_returns": episode_returns,
        "eval_success_history": eval_success_history,
        "eval_reward_history": eval_reward_history,
        "eval_every_updates": eval_every_updates,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }


def plot_training(history):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["episode_returns"])
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")

    plt.subplot(1, 2, 2)
    eval_x = (
        np.arange(len(history["eval_success_history"])) * history["eval_every_updates"]
        + history["eval_every_updates"]
    )
    plt.plot(eval_x, history["eval_success_history"])
    plt.title("Eval Success Rate")
    plt.xlabel("Update")
    plt.ylabel("Success Rate")

    plt.tight_layout()
    plt.show()


def evaluate_checkpoint(checkpoint_path, episodes=50):
    env = RocketEnv()
    agent = PPOAgent()
    load_checkpoint(checkpoint_path, agent)
    metrics = evaluate_agent(agent, env, num_episodes=episodes)
    print(metrics)
    return metrics


def demo_checkpoint(checkpoint_path):
    env = RocketEnv()
    agent = PPOAgent()
    load_checkpoint(checkpoint_path, agent)

    state = env.reset(difficulty=1.0)
    done = False
    total_reward = 0.0

    while not done:
        action, _, _, _ = agent.select_action(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print("total_reward:", total_reward)
    print("landed:", env.landed, "crashed:", env.crashed, "out_of_bounds:", env.out_of_bounds)


def main():
    smoke_test()
    history = train()
    plot_training(history)
    evaluate_checkpoint(history["best_model_path"], episodes=50)
    demo_checkpoint(history["best_model_path"])


if __name__ == "__main__":
    main()
