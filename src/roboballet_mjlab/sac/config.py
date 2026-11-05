# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Values taken from github.com/leggedrobotics/rsl_rl_sac at e0d243a. Changed for this task as commented.
"""Configuration of the SAC agent, kept apart from the others'. DESIGN section 16.

A value with no comment equals the fork's default, sac.py:37-54 or sac_mlp_model.py:43-46. Those are
the constructor's defaults, which are not what the fork's paper ran: its Table C9 sets nearly all of
them differently. docs/reviews/sac_status.md has both, side by side.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class SACCfg:
  """The fork's update on the RoboBallet GNN, with FastTD3Cfg's worlds, batch, replay and hindsight."""

  seed: int = 1

  # Differ from the fork, so that this compares directly with FastTD3Cfg, DESIGN 16.
  num_worlds: int = 1024  # Fork: whatever the env has. FastTD3Cfg's.
  batch_size: int = 2048  # Fork: mini_batch_size 256. FastTD3Cfg's.
  replay_max_size: int = 2_000_000  # Fork: 1_000_000. FastTD3Cfg's.
  num_mini_batches: int = 2  # Fork: 1. With num_steps_per_env 1 this is FastTD3Cfg's rate: 2 fresh batches per round.
  num_steps_per_env: int = 1  # The fork's runner requires a value and has no default for SAC. One round, as FastTD3Cfg.
  num_learning_epochs: int = 1
  total_env_rounds: int = 1_000_000  # Fork: whatever the caller asks of OffPolicyRunner.learn. FastTD3Cfg's.

  gamma: float = 0.94  # Fork: 0.998. DESIGN 16: tuned for this task by the original paper.
  n_step: int = 1  # Not in the fork's code. Its paper's Table C9 has 5.
  tau: float = 0.005
  lr_actor: float = 1e-3
  lr_critic: float = 1e-3
  lr_decay_steps: int = 0  # Not in the fork. Grad steps of a cosine on actor and critic, as FastTD3Cfg's. 0 keeps them fixed.
  lr_end_fraction: float = 1.0  # End rate over start rate.
  lr_alpha: float = 1e-3
  alpha: float = 0.05  # The initial value of the learned temperature.
  target_entropy_scale: float = 1.0  # Target entropy = -scale * number of action dimensions, 7 per robot.
  max_grad_norm: float = 1.0
  policy_frequency: int = 2

  init_noise_std: float = 1.0  # The form of the paper's first change, with the code's default. Its Table C9 has 0.15.
  log_std_min: float = -20.0
  log_std_max: float = 2.0

  # Not in the fork, which has no hindsight. Same rule, same module and same values as FastTD3Cfg, DESIGN 16.
  hindsight: bool = True
  hindsight_fraction: float = 0.4
  hindsight_zero_dwelling_actions: bool = True
  hindsight_zero_collision_penalty: bool = True
  hindsight_flush_episodes: int = 256

  # Not in the fork, which is f32 and eager. DESIGN 14.1, for the learner call only.
  matmul_precision: Literal['high', 'highest'] = 'high'
  compile: bool = True
  bf16: bool = True

  # Not in the fork's update. FastTD3Cfg's cadences, and the checkpoint rule of DESIGN 14.
  eval_every_rounds: int = 2_000
  eval_episodes: int = 1024
  checkpoint_every_s: float = 1800.0
  policy_snapshot_every_steps: int = 100_000  # Policy only, kept. 0 turns it off.
  resume_min_rows: int = 200_000  # A resumed run's first update waits for this many rows, as TrainCfg's min_size. A fresh run does not.
  log_every_rounds: int = 200
  profile: bool = True
  profile_every_rounds: int = 1500  # One 120-round window per this many rounds.
