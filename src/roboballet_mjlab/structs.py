# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Tensor containers shared by env, scenes, agent and replay.

Everything lives on the GPU. Shapes use W worlds (or batch), R robots, T targets,
O obstacles. Rotations are 3x3 matrices, world frame, unscaled metres and radians.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class Scene(NamedTuple):
  """Per-episode static data, the batched equivalent of a StartState proto."""

  base_pos: torch.Tensor  # [W, R, 3]
  base_rot: torch.Tensor  # [W, R, 3, 3]
  box_pos: torch.Tensor  # [W, O, 3]
  box_rot: torch.Tensor  # [W, O, 3, 3]
  box_spans: torch.Tensor  # [W, O, 3], full side lengths, not half sizes.
  target_pos: torch.Tensor  # [W, T, 3]
  target_rot: torch.Tensor  # [W, T, 3, 3]
  q0: torch.Tensor  # [W, R, 7]


class RobotState(NamedTuple):
  """Per-step robot data: exactly what the replay stores for each row."""

  q: torch.Tensor  # [W, R, 7]
  qd: torch.Tensor  # [W, R, 7], commanded velocity after clamps, not clipped at limits.
  dwell: torch.Tensor  # [W, R], int, remaining dwell steps in 0..dwell_steps.
  tip_pos: torch.Tensor  # [W, R, 3], TCP.
  tip_rot: torch.Tensor  # [W, R, 3, 3], TCP.
  target_done: torch.Tensor  # [W, T], bool.
  score: torch.Tensor  # [W]
  step: torch.Tensor  # [W], int, 0 at reset.
  # Per-observation obstacle augmentation in 0..3, planning_env.py:264-304 and
  # :941-945. One draw per observation, so s' of step t equals s of step t+1.
  aug_code: torch.Tensor  # [W, O], int.


class Obs(NamedTuple):
  """A full observation: the raw inputs of the graph feature builder."""

  robot: RobotState
  scene: Scene


class StepOut(NamedTuple):
  """Result of BalletEnv.step, before the auto-reset of terminated worlds."""

  next_robot: RobotState  # Post-step state, including the terminal state.
  reward: torch.Tensor  # [W]
  terminal: torch.Tensor  # [W], bool. True on timeout too: no bootstrap in either case.
  collision_robot_steps: torch.Tensor  # [W], int, robots rolled back this step.
  collision_penalty: torch.Tensor  # [W]
  acceleration: torch.Tensor  # [W], mean |dqd/dt|*dt over robots and joints.


class Graph(NamedTuple):
  """Dense graph batch. Topology is fixed for a run and held by the feature builder."""

  nodes: torch.Tensor  # [B, N, 17], N = R + T + O.
  edges: torch.Tensor  # [B, E, 30], E = R(R-1) + RT + RO.
  globals: torch.Tensor  # [B, 2]
