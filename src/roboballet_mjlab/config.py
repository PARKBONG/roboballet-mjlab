# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Configuration for the mjlab port of RoboBallet.

Defaults reproduce the reference run `runs/paper_repro/paper_4r10t_s1`, which
matches paper Table S5. See docs/DESIGN.md and docs/reports/ for the evidence
behind each constant. These dataclasses are the contract between modules.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

ArmModel = Literal['spheres_stick', 'mesh_tool']
RollbackMode = Literal['random', 'deterministic']

# Robot order used by every array in the original code (base_env.py:37-46,
# hparams.py:60).
ROBOT_ORDER_4 = ('panda1', 'panda2', 'panda3', 'panda4')
ROBOT_ORDER_8 = (
    'panda1', 'panda2', 'panda5', 'panda6',
    'panda3', 'panda4', 'panda7', 'panda8',
)

JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
HOME_QPOS = (0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0)


@dataclasses.dataclass(frozen=True)
class WorldCfg:
  """What exists in the simulated cell."""

  num_robots: int = 4
  num_obstacles: int = 30
  arm_model: ArmModel = 'spheres_stick'
  # Rendering only. Visual geoms add kinematics cost, so training omits them.
  visual_meshes: bool = False
  # Contact pool per world. Must exceed candidates passing broadphase, not just
  # penetrating contacts, and overflow is silent, so size it generously.
  nconmax: int = 512

  @property
  def robot_names(self) -> tuple[str, ...]:
    return ROBOT_ORDER_8 if self.num_robots > 4 else ROBOT_ORDER_4[: self.num_robots]


@dataclasses.dataclass(frozen=True)
class EnvCfg:
  """Task semantics. See docs/reports/env_spec.md sections 4 to 9."""

  world: WorldCfg = dataclasses.field(default_factory=WorldCfg)
  num_targets: int = 10
  timestep_s: float = 0.1
  max_joint_velocity: float = 2.175
  max_joint_acceleration: float = 8.0
  position_threshold: float = 0.025
  angle_threshold_rad: float = 15.0 / 180.0 * math.pi
  dwell_time_s: float = 0.5
  collision_penalty_scaling: float = 15.0
  # The reference run turns return-to-start off, so these are its values, not
  # the code defaults of 0.25 and True.
  return_to_start_reward_weight: float = 0.0
  return_to_start_required: bool = False
  return_to_start_threshold: float = 0.25
  return_to_start_coeff: float = 3.0
  return_to_start_epsilon: float = 1e-3
  augment_obstacles: bool = True
  collisions_enabled: bool = True  # False only for golden no-collision replays.
  rollback: RollbackMode = 'random'

  @property
  def max_steps(self) -> int:
    return int((1.0 * self.num_targets + 2.0) / self.timestep_s)

  @property
  def dwell_steps(self) -> int:
    return math.ceil(self.dwell_time_s / self.timestep_s)

  @property
  def collision_penalty_per_robot(self) -> float:
    return self.collision_penalty_scaling / self.max_steps / self.world.num_robots


@dataclasses.dataclass(frozen=True)
class ModelCfg:
  """GNN sizes. See docs/reports/algo_spec.md section 3."""

  node_layers: int = 7
  node_size: int = 512
  edge_embed_layers: int = 6
  edge_update_layers: int = 6  # Paper Table S5; the code default is 7.
  edge_size: int = 256
  global_layers: int = 7
  head_sizes: tuple[int, ...] = (64, 64)
  head_last_scale: float = 0.005
  layernorm_eps: float = 1e-6


@dataclasses.dataclass(frozen=True)
class TrainCfg:
  """TD3 plus hindsight. Defaults are parity mode, algo_spec.md sections 1 and 4."""

  seed: int = 1
  num_worlds: int = 64
  batch_size: int = 128
  samples_per_insert: float = 5.0
  fused_steps: int = 20
  total_grad_steps: int = 10_000_000

  gamma: float = 0.94
  tau_critic: float = 8e-5
  tau_policy: float = 4e-5
  lr_policy: float = 5e-5
  lr_policy_decay: float = 0.98
  lr_critic: float = 5e-5
  lr_critic_decay: float = 0.95
  decay_every_steps: float = 1e5
  target_noise_sigma: float = 5e-4
  target_noise_clip: float = 0.4  # Applied to the unit normal, before scaling.

  policy_sigma: float = 0.7
  policy_sigma_decay: float = 0.8
  random_episodes_per_world: int = 32  # actor.py:597-600, latched per world, DESIGN 14.2.

  hindsight_fraction: float = 0.4
  hindsight_zero_dwelling_actions: bool = True
  hindsight_zero_collision_penalty: bool = True
  hindsight_flush_episodes: int = 8  # DESIGN 14.3 item 5. 960 rows at most, inside the limiter's band.

  # Replay capacity in transitions. None means int(2e6 // (T / 10)).
  replay_max_size: int | None = None
  replay_min_divisor: int = 10

  # Throughput mode. With schedule_by_samples, tau and every schedule are
  # rescaled by batch_size * world_size / 128 so that they stay denominated in
  # samples rather than gradient steps. Parity mode leaves this off.
  schedule_by_samples: bool = False
  lr_mult: float = 1.0

  # The learner call only. Everything else in the process stays at highest, DESIGN 14.1.
  matmul_precision: Literal['high', 'highest'] = 'high'
  compile: bool = True
  bf16: bool = True  # Learner, as the original. Acting and evaluation are always f32.

  eval_every_steps: int = 20_000
  eval_episodes: int = 1024
  checkpoint_every_s: float = 1800.0  # One file, overwritten. The original keeps every one, 69 GB by 1.87M steps.
  policy_snapshot_every_steps: int = 100_000  # Policy only, about 36 MB, kept. 0 turns it off.
  log_every_steps: int = 1_000
  profile: bool = True

  def replay_capacity(self, num_targets: int) -> int:
    if self.replay_max_size is not None:
      return self.replay_max_size
    return int(int(2e6) // (num_targets / 10))
