"""scenes/sampler.py: distributions, bounds and determinism on small batches."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from roboballet_mjlab.config import WorldCfg
from roboballet_mjlab.scenes import ik, sampler
from roboballet_mjlab.sim.world import build_world, default_bases
from roboballet_mjlab.structs import Scene
from tests.test_scenes_common import OFFSET, face_stats
from tests.test_sim_common import DEVICE

NUM_WORLDS = 512
CFG = sampler.SamplerCfg()
QUICK = ik.IkBudget(restarts=5, iters=20)


@pytest.fixture(scope='module')
def world():
  return build_world(WorldCfg(), NUM_WORLDS, DEVICE)


@pytest.fixture(scope='module')
def world8():
  return build_world(WorldCfg(num_robots=8), 256, DEVICE)


@pytest.fixture(scope='module')
def kin():
  return ik.ArmKinematics('spheres_stick', DEVICE)


def _gen(seed=0):
  return torch.Generator(DEVICE).manual_seed(seed)


def test_random_rotations_are_uniform_rotations():
  rot = sampler.random_rotations(200_000, DEVICE, _gen())
  assert (rot.transpose(1, 2) @ rot - torch.eye(3, device=DEVICE)).abs().max() < 1e-5
  assert (torch.linalg.det(rot) - 1).abs().max() < 1e-5
  assert abs(rot[:, 2, 2].abs().mean() - 0.5) < 0.005  # paper_baseline 7.3: 0.5008 in the data.


def test_rails_ranges_order_and_freedom(world):
  scene = sampler.blank_scene(world, NUM_WORLDS, 1)
  failed, counts = sampler.sample_rails(world, scene, CFG, _gen(1))
  assert not failed.any()
  assert counts['rail_draws'] >= NUM_WORLDS * world.num_robots
  y = scene.base_pos[..., 1]
  assert y[:, :2].max() <= CFG.floor_rail_cut and y[:, :2].min() >= -CFG.rail_limit
  assert y[:, 2:].min() >= -CFG.floor_rail_cut and y[:, 2:].max() <= CFG.rail_limit
  default_pos, _ = default_bases(world.cfg, DEVICE)
  assert (scene.base_pos[..., [0, 2]] == default_pos[:, [0, 2]]).all()
  ids = torch.arange(NUM_WORLDS, device=DEVICE)
  assert not sampler.any_robot_collides(world, scene, ids).any()
  # Same-rail gaps keep the sphere clearance measured in sphere_model.md, verifier row 13.
  assert (y[:, 0] - y[:, 3]).abs().min() > 0.25 and (y[:, 1] - y[:, 2]).abs().min() > 0.25


def test_rails_eight_robots_use_the_released_range_on_the_ceiling(world8):
  scene = sampler.blank_scene(world8, 256, 1)
  failed, _ = sampler.sample_rails(world8, scene, CFG, _gen(2))
  assert not failed.any()
  y = scene.base_pos[..., 1]
  floor = scene.base_pos[0, :, 2] < 1
  # Robot order: panda1, 2, 5, 6, 3, 4, 7, 8. Floor robots stop at the cut.
  assert floor.tolist() == [True, True, False, False, True, True, False, False]
  assert y[:, :2].max() <= CFG.floor_rail_cut and y[:, 4:6].min() >= -CFG.floor_rail_cut
  assert y[:, [2, 3]].max() > CFG.floor_rail_cut + 0.1
  assert y[:, [6, 7]].min() < -CFG.floor_rail_cut - 0.1
  ids = torch.arange(256, device=DEVICE)
  assert not sampler.any_robot_collides(world8, scene, ids).any()


def _placed_scene(world, n, num_targets, seed):
  gen = _gen(seed)
  scene = sampler.blank_scene(world, n, num_targets)
  assert not sampler.sample_rails(world, scene, CFG, gen)[0].any()
  assert not sampler.sample_boxes(world, scene, CFG, gen)[0].any()
  return scene, gen


def test_boxes_ranges_and_freedom(world):
  scene, _ = _placed_scene(world, NUM_WORLDS, 1, 3)
  for i in range(3):
    assert scene.box_spans[..., i].min() >= CFG.spans_min[i]
    assert scene.box_spans[..., i].max() <= CFG.spans_max[i]
    assert scene.box_pos[..., i].min() >= CFG.box_pos_min[i]
    assert scene.box_pos[..., i].max() <= CFG.box_pos_max[i]
  ids = torch.arange(NUM_WORLDS, device=DEVICE)
  assert not sampler.any_robot_collides(world, scene, ids).any()


def test_box_order_does_not_matter_to_acceptance(world):
  """Rejection looks at one box at a time, so box 0 and box 29 share a distribution."""
  scene, _ = _placed_scene(world, NUM_WORLDS, 1, 4)
  first, last = scene.box_pos[:, :10, 2].mean(), scene.box_pos[:, -10:, 2].mean()
  assert abs(first - last) < 0.03


def test_candidates_follow_the_stored_convention(world):
  scene, gen = _placed_scene(world, 64, 1, 5)
  ids = torch.arange(64, device=DEVICE).repeat_interleave(2000)
  pos, rot = sampler.draw_candidates(scene, ids, CFG, gen)
  assert (rot.transpose(1, 2) @ rot - torch.eye(3, device=DEVICE)).abs().max() < 1e-5
  assert (torch.linalg.det(rot) - 1).abs().max() < 1e-5
  sub = Scene(*[f[ids] for f in scene])._replace(target_pos=pos[:, None], target_rot=rot[:, None])
  has_owner, residual, angle = face_stats(sub)
  # A box can match by accident, but a candidate must always have an owner.
  assert has_owner.all()
  assert residual.max() < 1e-4, f'Offset {OFFSET} m to 1e-4'
  assert angle.max() <= 22.5 + 1e-2
  # Uniform on the 22.5 degree cap: theoretical median 15.85 degrees, env_spec 10.3.
  assert abs(float(angle.median()) - 15.85) < 0.3


def test_candidates_are_area_weighted(world):
  """One box dominates the area, and faces split by their share of it."""
  n = 200_000
  scene = sampler.blank_scene(world, 1, 1)
  scene.box_spans[0, :] = 1e-4
  scene.box_spans[0, 7] = torch.as_tensor([0.02, 0.05, 0.5], device=DEVICE)
  scene.box_pos[0, 7] = torch.as_tensor([0.0, 0.0, 0.5], device=DEVICE)
  ids = torch.zeros(n, dtype=torch.long, device=DEVICE)
  pos, _ = sampler.draw_candidates(scene, ids, CFG, _gen(6))
  local = pos - scene.box_pos[0, 7]
  x_face = (local[:, 0].abs() - (0.01 + OFFSET)).abs() < 1e-4
  y_face = (local[:, 1].abs() - (0.025 + OFFSET)).abs() < 1e-4
  z_face = (local[:, 2].abs() - (0.25 + OFFSET)).abs() < 1e-4
  area = torch.as_tensor([0.05 * 0.5, 0.02 * 0.5, 0.02 * 0.05])
  share = area / area.sum()
  for face, expected in zip((x_face, y_face, z_face), share):
    assert abs(face.float().mean().item() - expected.item()) < 0.005
  assert abs((local[x_face, 0] > 0).float().mean().item() - 0.5) < 0.01


def test_rank_within_scene():
  ids = torch.as_tensor([0, 0, 2, 5, 5, 5], device=DEVICE)
  assert sampler._rank_within_scene(ids).tolist() == [0, 1, 0, 0, 1, 2]


def test_targets_fill_every_scene_with_independent_witnesses(world, kin):
  scene, gen = _placed_scene(world, 64, 10, 7)
  witnesses, failed, counts = sampler.sample_targets(world, kin, scene, CFG, QUICK, gen)
  assert not failed.any()
  assert (witnesses.robot[~failed] >= 0).all()
  assert counts['target_accepted'] >= 640
  assert not (scene.target_pos == 0).all(-1).any()
  assert not (witnesses.q == 0).all(-1).any()
  ids = torch.arange(64, device=DEVICE).repeat_interleave(10)
  ok = ik.witness_valid(world, scene, ids, witnesses.robot.flatten(),
                        witnesses.q.flatten(0, 1), scene.target_pos.flatten(0, 1),
                        scene.target_rot.flatten(0, 1), ik.IkBudget())
  assert ok.all()


def test_generate_is_deterministic(world, kin):
  one = sampler.generate(world, kin, 32, 10, CFG, QUICK, _gen(11))
  two = sampler.generate(world, kin, 32, 10, CFG, QUICK, _gen(11))
  other = sampler.generate(world, kin, 32, 10, CFG, QUICK, _gen(12))
  for a, b in zip(one.scene, two.scene):
    assert torch.equal(a, b)
  assert torch.equal(one.witnesses.q, two.witnesses.q)
  assert torch.equal(one.witnesses.robot, two.witnesses.robot)
  assert one.counts == two.counts
  assert not torch.equal(one.scene.box_pos, other.scene.box_pos)
  assert not torch.equal(one.scene.target_pos, other.scene.target_pos)
  assert not torch.equal(one.scene.base_pos, other.scene.base_pos)


def test_stale_world_state_does_not_change_a_pool(world, kin):
  """Resume depends on this: a shard must not depend on what ran before it."""
  one = sampler.generate(world, kin, 16, 10, CFG, QUICK, _gen(13))
  sampler.generate(world, kin, 48, 10, CFG, QUICK, _gen(14))
  two = sampler.generate(world, kin, 16, 10, CFG, QUICK, _gen(13))
  assert torch.equal(one.scene.target_pos, two.scene.target_pos)


def test_rail_failure_is_counted_not_a_hang(world, kin):
  """Every robot forced onto y = 0, so the second on a rail can never fit."""
  cfg = dataclasses.replace(CFG, rail_limit=0.0, floor_rail_cut=0.0, max_rail_draws=3)
  result = sampler.generate(world, kin, 8, 1, cfg, QUICK, _gen(15))
  assert result.counts['rail_failures'] == 8
  assert len(result.scene.q0) == 0


def test_box_failure_is_counted_not_a_hang(world, kin):
  cfg = dataclasses.replace(CFG, spans_min=(4.0, 4.0, 4.0), spans_max=(4.0, 4.0, 4.0), max_box_draws=2)
  result = sampler.generate(world, kin, 8, 1, cfg, QUICK, _gen(16))
  assert result.counts['box_failures'] == 8
  assert result.counts['box_draws'] == 8 * 2 * world.num_obstacles
  assert len(result.scene.q0) == 0


def test_target_failure_is_counted_not_a_hang(world, kin):
  """An offset of 10 m puts every candidate out of reach."""
  cfg = dataclasses.replace(CFG, surface_offset=10.0, max_target_rounds=2)
  result = sampler.generate(world, kin, 8, 2, cfg, QUICK, _gen(17))
  assert result.counts['target_failures'] == 8
  assert result.counts['target_accepted'] == 0
  assert result.counts['target_draws'] == 8 * 2 * cfg.draws_per_missing * cfg.max_target_rounds
  assert len(result.scene.q0) == 0


def test_too_many_scenes_for_the_world_is_an_error(world, kin):
  with pytest.raises(ValueError, match='do not fit'):
    sampler.generate(world, kin, NUM_WORLDS + 1, 1, CFG, QUICK, _gen())
