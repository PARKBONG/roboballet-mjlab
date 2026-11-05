"""Configuration of the PPO agent, kept apart from the others'. DESIGN section 17."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.structs import RobotState, Scene


@dataclasses.dataclass(frozen=True)
class PPOCfg:
  """rsl_rl.algorithms.PPO on the RoboBallet GNN, by rsl-rl's own argument names. No hindsight."""

  seed: int = 1
  num_worlds: int = 1024
  num_steps_per_env: int = 24
  total_env_rounds: int = 1_000_000

  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.94
  lam: float = 0.95
  entropy_coef: float = 0.01
  # With learning_rate below, the plan's: on the full model the outputs stop depending on the state within about an
  # iteration, docs/reviews/ppo_status.md. Only 'fixed' gets past ppo.py:249's cap of 1e-2. A new default is the lead's, in DESIGN 17.
  schedule: Literal['adaptive', 'fixed'] = 'adaptive'
  desired_kl: float = 0.01
  # The rest are rsl-rl's defaults, ppo.py:44-53, which mjlab/rl/config.py keeps too.
  value_loss_coef: float = 1.0
  learning_rate: float = 1e-3
  max_grad_norm: float = 1.0
  use_clipped_value_loss: bool = True
  normalize_advantage_per_mini_batch: bool = False
  optimizer: str = 'adam'
  init_std: float = 1.0  # mjlab/rl/config.py:135-137.
  std_type: Literal['scalar', 'log'] = 'scalar'

  # The learner call only. Acting and evaluation stay in f32 at highest, DESIGN 14.1. bf16 is an
  # option and not the default: ppo/models.py says why.
  matmul_precision: Literal['high', 'highest'] = 'high'
  bf16: bool = False
  compile: bool = True

  # FastTD3Cfg's protocol and intervals. A round is one env step of every world in both.
  eval_every_rounds: int = 2_000
  eval_episodes: int = 1024
  checkpoint_every_s: float = 900.0
  log_every_rounds: int = 200
  policy_snapshot_every_rounds: int = 10_000  # Policy only, about 36 MB, kept. 0 turns it off.
  profile: bool = True
  profile_every_rounds: int = 1488  # A multiple of num_steps_per_env, so windows hold whole iterations.
  profile_window_rounds: int = 120  # 5 iterations.

  def runner_dict(self, env_cfg: EnvCfg, model_cfg: ModelCfg) -> dict[str, Any]:
    """What OnPolicyRunner reads, with the models by class_name. A new dict, because rsl-rl pops keys from it."""
    models = dict(env_cfg=env_cfg, model_cfg=model_cfg, seed=self.seed, compile=self.compile, bf16=self.bf16)
    groups = [*RobotState._fields, *Scene._fields]  # The keys of BalletVecEnv's TensorDict.
    return {
      'num_steps_per_env': self.num_steps_per_env,
      'save_interval': 10**12,  # The runner saves nothing here: log_dir is None, and ppo/train.py keeps the checkpoints.
      # 19 reads of the GPU per round, on_policy_runner.py:90. ppo/train.py checks the whole rollout at its end, in one.
      'check_for_nan': False,
      'obs_groups': {'actor': groups, 'critic': groups},
      'actor': {
        'class_name': 'roboballet_mjlab.ppo.models:BalletActor',
        'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': self.init_std, 'std_type': self.std_type},
        **models,
      },
      'critic': {'class_name': 'roboballet_mjlab.ppo.models:BalletCritic', **models},
      'algorithm': {
        'class_name': 'roboballet_mjlab.ppo.models:BalletPPO',
        'matmul_precision': self.matmul_precision,
        'num_learning_epochs': self.num_learning_epochs,
        'num_mini_batches': self.num_mini_batches,
        'clip_param': self.clip_param,
        'gamma': self.gamma,
        'lam': self.lam,
        'value_loss_coef': self.value_loss_coef,
        'entropy_coef': self.entropy_coef,
        'learning_rate': self.learning_rate,
        'max_grad_norm': self.max_grad_norm,
        'optimizer': self.optimizer,
        'use_clipped_value_loss': self.use_clipped_value_loss,
        'schedule': self.schedule,
        'desired_kl': self.desired_kl,
        'normalize_advantage_per_mini_batch': self.normalize_advantage_per_mini_batch,
        'rnd_cfg': None,
      },
    }

