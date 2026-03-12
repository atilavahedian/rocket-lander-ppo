# Rocket Lander PPO

Custom reinforcement learning project for training a continuous-control rocket to land on a pad using Proximal Policy Optimization (PPO) in PyTorch.

## What This Project Does

- Simulates a 2D rocket landing environment with thrust, torque, fuel use, and strict landing constraints
- Trains a PPO agent with observation normalization and generalized advantage estimation
- Uses curriculum learning to start from easier states and ramp toward full difficulty
- Uses expert imitation pretraining to bootstrap the policy before PPO fine-tuning
- Saves checkpoints with both model weights and normalization statistics

## Why This Project Exists

This project started as a reinforcement learning experiment and turned into a debugging exercise around:

- reward hacking
- PPO training stability
- observation normalization consistency
- expert warm-start training

The current codebase fixes the main learning bugs and demonstrates real landing behavior, but PPO fine-tuning is still somewhat unstable across long runs. Best checkpoint performance is currently more meaningful than final checkpoint performance.

## Current Status

- Environment works
- Expert controller lands reliably
- PPO learns non-trivial landing behavior
- Training is still being tuned for long-run stability

This is a strong portfolio project for custom environment design, RL debugging, and training-system iteration.

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib

## Project Structure

- `train.py`: environment, PPO agent, expert controller, training loop, checkpointing, evaluation, and plotting

## Running Locally

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python train.py
```

Optional checkpoint output directory:

```bash
export ROCKET_RL_SAVE_DIR=./rocket_rl_runs
python train.py
```

## Running In Google Colab

1. Open a new Colab notebook.
2. Set runtime type to GPU.
3. Upload `train.py`.
4. Install dependencies if needed:

```python
!pip install torch matplotlib numpy
```

5. Run:

```python
!python /content/train.py
```

To persist checkpoints to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["ROCKET_RL_SAVE_DIR"] = "/content/drive/MyDrive/rocket_rl_runs"
!python /content/train.py
```

## Known Limitations

- PPO fine-tuning can regress after early gains
- Best saved checkpoint is often stronger than the final model
- Results can vary by run and hyperparameter choice

## Portfolio Summary

Highlights:

- Designed a custom continuous-control RL environment from scratch
- Diagnosed and fixed reward misalignment and PPO data-consistency bugs
- Added expert imitation pretraining and curriculum learning
- Built checkpoint save/load flow that preserves normalization state for correct evaluation

