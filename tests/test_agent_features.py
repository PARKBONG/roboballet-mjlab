"""E6: graph features against the original builder, on real rollouts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.config import EnvCfg, WorldCfg
from roboballet_mjlab.env.poses import augment_boxes
from roboballet_mjlab.structs import Graph, Obs
from tests.golden.agent_load import DEVICE, E6, load_e6

E6_TOLERANCE = 1e-4  # paper_baseline 10, E6
OBSERVED_TOLERANCE = 1e-6  # algo_spec 12.2 saw 2.4e-7 on three observations.


@pytest.fixture(scope='module')
def e6_4r() -> E6:
  return load_e6('4r10t', shards=2)


@pytest.fixture(scope='module')
def topo_4r() -> Topology:
  return Topology(EnvCfg(), DEVICE)


def _check_against(graph: Graph, expected: Graph) -> dict[str, float]:
  errors = {}
  for name, got, want in zip(Graph._fields, graph, expected):
    assert got.shape == want.shape, name
    assert got.dtype == torch.float32, name
    errors[name] = (got - want).abs().max().item()
  print('E6 max abs error', errors, 'observations', graph.nodes.shape[0])
  assert max(errors.values()) <= E6_TOLERANCE
  assert max(errors.values()) <= OBSERVED_TOLERANCE
  return errors


def test_e6_4r10t_1000_observations(e6_4r, topo_4r):
  assert e6_4r.graph.nodes.shape == (1000, 44, 17)
  assert e6_4r.graph.edges.shape == (1000, 172, 30)
  _check_against(build_graph(e6_4r.obs, topo_4r), e6_4r.graph)


def test_e6_8r40t():
  e6 = load_e6('8r40t', shards=1)
  cfg = EnvCfg(world=WorldCfg(num_robots=8), num_targets=40)
  assert e6.graph.nodes.shape == (40, 78, 17)
  assert e6.graph.edges.shape == (40, 616, 30)
  topo = Topology(cfg, DEVICE)
  _check_against(build_graph(e6.obs, topo), e6.graph)
  assert np.array_equal(topo.senders.cpu().numpy(), e6.raw['senders'])
  assert np.array_equal(topo.receivers.cpu().numpy(), e6.raw['receivers'])


def test_golden_covers_every_case(e6_4r):
  """The comparison means little unless done flags, dwell and all aug codes occur."""
  raw = e6_4r.raw
  assert raw['target_done'].any() and not raw['target_done'].all()
  assert (raw['dwell'] > 0).any() and (raw['dwell'] == 5).any()
  assert raw['score'].max() > 0.5
  assert set(np.unique(raw['aug_code'])) == {0, 1, 2, 3}
  assert raw['step'].max() > 30


def test_topology_matches_jraph(e6_4r, topo_4r):
  assert np.array_equal(topo_4r.senders.cpu().numpy(), e6_4r.raw['senders'])
  assert np.array_equal(topo_4r.receivers.cpu().numpy(), e6_4r.raw['receivers'])


def test_robot_edges_are_base_to_base(e6_4r, topo_4r):
  """Moving the tips must leave the robot-robot block unchanged, paper notwithstanding."""
  robot, scene = e6_4r.obs
  moved = robot._replace(tip_pos=robot.tip_pos + 0.1)
  a = build_graph(e6_4r.obs, topo_4r).edges
  b = build_graph(Obs(moved, scene), topo_4r).edges
  rows = topo_4r.num_robot_edges
  assert torch.equal(a[:, :rows], b[:, :rows])
  assert not torch.equal(a[:, rows:], b[:, rows:])


def test_independent_of_matmul_precision(e6_4r, topo_4r):
  torch.set_float32_matmul_precision('highest')
  exact = build_graph(e6_4r.obs, topo_4r)
  torch.set_float32_matmul_precision('high')
  fast = build_graph(e6_4r.obs, topo_4r)
  torch.set_float32_matmul_precision('highest')
  for a, b in zip(exact, fast):
    assert torch.equal(a, b)


def test_augmentation_agrees_with_env_poses(e6_4r, topo_4r):
  """Same boxes as env.poses.augment_boxes, the env's copy of planning_env.py:264-304."""
  torch.set_float32_matmul_precision('highest')
  robot, scene = e6_4r.obs
  codes = robot.aug_code
  rot, spans = augment_boxes(scene.box_rot, scene.box_spans, codes)
  order = topo_4r.aug_order[codes]
  mine_rot = scene.box_rot.gather(-1, order[:, :, None].expand(-1, -1, 3, -1)) * topo_4r.aug_sign[codes][:, :, None]
  assert (mine_rot - rot).abs().max() <= 2e-7
  assert torch.equal(scene.box_spans.gather(-1, order), spans)
