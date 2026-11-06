"""Configuration of the FastTD3 agent, kept apart from the original's. DESIGN section 12."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class FastTD3Cfg:
  """FastTD3 on the RoboBallet GNN. Reference and every deviation: DESIGN section 12.1."""

  seed: int = 1
  num_worlds: int = 1024
  batch_size: int = 2048
  num_updates: int = 2  # Per env round, policy on odd i.
  policy_frequency: int = 2
  learning_starts: int = 10  # Env rounds.
  total_env_rounds: int = 1_000_000

  gamma: float = 0.94
  n_step: int = 1  # The reference's num_steps, hyperparams.py:47.
  tau: float = 0.1
  lr_policy: float = 3e-4
  lr_critic: float = 3e-4
  lr_decay_steps: int = 0  # Grad steps of the cosine, train.py:302-313. 0 keeps both rates fixed, as the reference's defaults do.
  lr_end_fraction: float = 1.0  # End rate over start rate, the reference's *_learning_rate_end.
  weight_decay: float = 0.1
  target_noise_sigma: float = 0.001
  target_noise_clip: float = 0.5  # Applied after scaling, unlike TrainCfg.

  num_atoms: int = 101
  v_min: float = -1.0
  v_max: float = 1.0
  use_cdq: bool = True

  std_min: float = 0.001
  std_max: float = 0.4
  # Not in the reference, which keeps std_max fixed. The original's sigma falls 0.8x per 100k steps
  # (TrainCfg.policy_sigma_decay). A cosine on std_max over this many grad steps, 0 keeps it fixed.
  noise_decay_steps: int = 0
  noise_end_fraction: float = 1.0

  # Same rule and same module as TrainCfg. The flush is 256 here and 8 there: at most 30,720 rows, about 30 rounds late at
  # W=1024, where DESIGN 14.3 item 5 sized the 8 to a limiter this agent does not have.
  hindsight: bool = True
  hindsight_fraction: float = 0.4
  hindsight_zero_dwelling_actions: bool = True
  hindsight_zero_collision_penalty: bool = True
  hindsight_flush_episodes: int = 256
  replay_max_size: int = 2_000_000

  matmul_precision: Literal['high', 'highest'] = 'high'
  compile: bool = True
  bf16: bool = True

  eval_every_rounds: int = 2_000
  eval_episodes: int = 1024
  checkpoint_every_s: float = 1800.0  # DESIGN 14.
  policy_snapshot_every_steps: int = 10_000  # Policy only, kept. 0 turns it off. Near TrainCfg's 100,000 a sweep run sees a few.
  resume_min_rows: int = 200_000  # A resumed run's first update waits for this many rows, as TrainCfg's min_size. A fresh run does not.
  log_every_rounds: int = 200
  profile: bool = True
  profile_every_rounds: int = 1500  # One 120-round window per this many rounds.
