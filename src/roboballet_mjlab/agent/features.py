# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Graph features from raw observations, docs/reports/algo_spec.md section 2.

Node and edge layouts follow agents/compute_features.py exactly. Every relative pose
is composed from scaled poses, so relative translations are in units of 5 m.

Rotations are multiplied with sums of products, never matmul: under matmul precision
`high` a batched 3x3 matmul is off by up to 7e-4 and 80 times slower (0.1 vs 9 ms for
491,520 products on a B200). For the same reason the box augmentation here does not
use env.poses.augment_boxes; tests/test_agent_features.py checks that they agree.
"""

from __future__ import annotations

import torch

from roboballet_mjlab.config import JOINT_LOWER, JOINT_UPPER, EnvCfg
from roboballet_mjlab.structs import Graph, Obs

NODE_DIM = 17
EDGE_DIM = 30
GLOBAL_DIM = 2
ACTION_DIM = 7
# Node columns of the target done flag and the obstacle marker.
TARGET_COLUMN = 15
OBSTACLE_COLUMN = 16
WORLD_HALF_SPAN = 5.0  # base_env.py:52-62, the range of translations and spans.

# planning_env.py:264-304: code k post-multiplies the rotation by +90 degrees about
# local axis k-1 and swaps the two other spans. That permutes the rotation's columns
# the same way as the spans and flips the sign of one column.
_AUG_ORDER = ((0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2))
_AUG_SIGN = ((1, 1, 1), (1, 1, -1), (-1, 1, 1), (1, -1, 1))


class Topology:
  """Everything about the graph that is fixed for a run, computed once."""

  def __init__(self, cfg: EnvCfg, device: torch.device | str):
    self.num_robots = R = cfg.world.num_robots
    self.num_targets = T = cfg.num_targets
    self.num_obstacles = O = cfg.world.num_obstacles
    self.num_nodes = R + T + O
    self.num_robot_edges = R * (R - 1)
    self.num_edges = self.num_robot_edges + R * T + R * O
    self.max_steps = cfg.max_steps
    self.dwell_steps = cfg.dwell_steps
    self.max_joint_velocity = cfg.max_joint_velocity

    # Receiver ascending, then sender ascending, compute_features.py:244-307.
    pairs = [(r, s) for r in range(R) for s in range(R) if r != s]
    pairs += [(r, R + t) for r in range(R) for t in range(T)]
    pairs += [(r, R + T + o) for r in range(R) for o in range(O)]
    self.receivers, self.senders = torch.tensor(pairs, device=device).unbind(-1)
    self.robot_receivers = self.receivers[: self.num_robot_edges]
    self.robot_senders = self.senders[: self.num_robot_edges]
    # Rows of the robot-robot block by sender, [R, R-1].
    self.robot_edges_by_sender = self.robot_senders.argsort(stable=True).view(R, R - 1)

    self.joint_low = torch.tensor(JOINT_LOWER, device=device)
    self.joint_range = torch.tensor(JOINT_UPPER, device=device) - self.joint_low
    self.aug_order = torch.tensor(_AUG_ORDER, device=device)
    self.aug_sign = torch.tensor(_AUG_SIGN, dtype=torch.float32, device=device)
    self.empty_nodes = torch.zeros(1, self.num_nodes, NODE_DIM, device=device)
    self.empty_nodes[:, R + T :, OBSTACLE_COLUMN] = 1.0


def _relative(
  recv_pos: torch.Tensor, recv_rot: torch.Tensor, send_pos: torch.Tensor, send_rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Each sender in each receiver's frame: [B, R, S, 3] and [B, R, S, 3, 3]."""
  inverse = recv_rot.transpose(-1, -2)
  rot = (inverse[:, :, None, :, :, None] * send_rot[:, None, :, None]).sum(-2)
  delta = send_pos[:, None] - recv_pos[:, :, None]
  pos = (inverse[:, :, None] * delta[..., None, :]).sum(-1)
  return pos, rot


def _pose9(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
  """Translation, then the top two rows of the rotation, compute_features.py:97-105."""
  return torch.cat([pos, rot[..., :2, :].flatten(-2)], -1)


def build_graph(obs: Obs, topo: Topology) -> Graph:
  """Builds the dense graph batch on the observation's device."""
  robot, scene = obs
  B = robot.q.shape[0]
  R, T = topo.num_robots, topo.num_targets

  nodes = topo.empty_nodes.repeat(B, 1, 1)
  nodes[:, :R, :7] = 2.0 * (robot.q - topo.joint_low) / topo.joint_range - 1.0
  nodes[:, :R, 7:14] = robot.qd / topo.max_joint_velocity
  # With no dwell configured the raw value is 0, so the column is -1, planning_env.py:978-986.
  nodes[:, :R, 14] = 2.0 * robot.dwell / max(topo.dwell_steps, 1) - 1.0
  nodes[:, R : R + T, TARGET_COLUMN] = 2.0 * robot.target_done - 1.0

  # Tip in the world frame, scaled, equals scaled base composed with scaled tip_relative.
  tip_pos = robot.tip_pos / WORLD_HALF_SPAN
  target_pos = scene.target_pos / WORLD_HALF_SPAN
  box_pos = scene.box_pos / WORLD_HALF_SPAN
  base_pos = scene.base_pos / WORLD_HALF_SPAN

  code = robot.aug_code.long()
  order = topo.aug_order[code]
  box_rot = scene.box_rot.gather(-1, order[:, :, None].expand(-1, -1, 3, -1))
  box_rot = box_rot * topo.aug_sign[code][:, :, None]
  box_spans = scene.box_spans.gather(-1, order) / WORLD_HALF_SPAN

  robot_edges = _pose9(*_relative(base_pos, scene.base_rot, base_pos, scene.base_rot))
  target_edges = _pose9(*_relative(tip_pos, robot.tip_rot, target_pos, scene.target_rot))
  rel_pos, rel_rot = _relative(tip_pos, robot.tip_rot, box_pos, box_rot)
  # Each rotation column scaled by its span, then the centroid: 3x4, row major.
  box_edges = torch.cat([rel_rot * box_spans[:, None, :, None], rel_pos[..., None]], -1).flatten(-2)

  edges = robot.q.new_zeros(B, topo.num_edges, EDGE_DIM)
  a, b = topo.num_robot_edges, topo.num_robot_edges + R * T
  edges[:, :a, :9] = robot_edges[:, topo.robot_receivers, topo.robot_senders]
  edges[:, a:b, 9:18] = target_edges.flatten(1, 2)
  edges[:, b:, 18:] = box_edges.flatten(1, 2)

  globals_ = torch.stack([2.0 * robot.step / topo.max_steps - 1.0, 2.0 * robot.score - 1.0], -1)
  return Graph(nodes, edges, globals_)
