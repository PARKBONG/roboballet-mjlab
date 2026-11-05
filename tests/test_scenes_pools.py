"""G5 validity of the generated pools, DESIGN section 6. Reads the pools through roboballet_mjlab.paths.scene_pool.

The C MuJoCo recheck covers every scene and witness, not only the random 1,000 the brief asks
for. A pool that was not generated fails its test: it is not a skip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, WorldCfg
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.scenes.io import load_pool, load_taskset
from roboballet_mjlab.scripts.gen_scenes import recheck
from roboballet_mjlab.sim.world import default_bases
from tests.test_scenes_common import (
  POS_TOL, ANG_TOL, check_static_boxes, face_stats, summarise_recheck)
from tests.test_sim_common import DATA, DEVICE

CONFIGS = [(4, 10), (8, 40)]
COUNTS = {'train': 50_000, 'heldout': 5_000, 'static': 10, 'static_paper': 10}
POOLS = [(r, t, split) for r, t in CONFIGS for split in COUNTS]


def _load(robots, targets, split):
  path = scene_pool(f'{robots}r_{targets}t_30o_{split}')
  if not path.exists():
    pytest.fail(f'Missing {path}')
  return load_pool(path, DEVICE)


@pytest.fixture(scope='module', params=POOLS, ids=lambda p: f'{p[0]}r_{p[1]}t_{p[2]}')
def pool_case(request):
  robots, targets, split = request.param
  pool = _load(robots, targets, split)
  summary = summarise_recheck(recheck(pool, 'spheres_stick'))
  print(f'\n{robots}r_{targets}t {split}: C MuJoCo {summary}')
  return robots, targets, split, pool, summary


def test_shapes_sizes_and_types(pool_case):
  robots, targets, split, pool, _ = pool_case
  n = COUNTS[split]
  scene = pool.scene
  assert scene.base_pos.shape == (n, robots, 3) and scene.base_rot.shape == (n, robots, 3, 3)
  assert scene.box_pos.shape == (n, 30, 3) and scene.box_rot.shape == (n, 30, 3, 3)
  assert scene.box_spans.shape == (n, 30, 3)
  assert scene.target_pos.shape == (n, targets, 3) and scene.target_rot.shape == (n, targets, 3, 3)
  assert scene.q0.shape == (n, robots, 7)
  assert pool.witnesses.q.shape == (n, targets, 7) and pool.witnesses.robot.shape == (n, targets)
  for field in scene:
    assert field.dtype == torch.float32 and torch.isfinite(field).all()
  assert (scene.q0 == torch.as_tensor(HOME_QPOS, device=DEVICE)).all()


def test_rotations_are_rotations(pool_case):
  *_, pool, _ = pool_case
  # f32 tolerance. Tested in chunks because the 8r pool holds 2M target frames.
  for rot in (pool.scene.base_rot, pool.scene.box_rot, pool.scene.target_rot):
    for chunk in rot.split(4096):
      assert (chunk.transpose(-1, -2) @ chunk - torch.eye(3, device=DEVICE)).abs().max() < 1e-5
      assert (torch.linalg.det(chunk) - 1).abs().max() < 1e-5


def test_c_mujoco_home_is_free(pool_case):
  """100% home-collision free, on C MuJoCo, over the whole pool."""
  *_, summary = pool_case
  assert summary['scenes'] > 0 and summary['home_collisions'] == 0


def test_c_mujoco_witnesses(pool_case):
  """Each stored witness: strictly within 2.5 cm and 15 degrees, free, other robots at home."""
  robots, targets, split, pool, summary = pool_case
  assert summary['witness_collisions'] == 0 and summary['witness_far'] == 0 and summary['witness_limits'] == 0
  assert summary['worst_pos_m'] < POS_TOL and summary['worst_ang_deg'] < math.degrees(ANG_TOL)
  if split == 'static_paper':
    # The one split that may lack a witness: its targets were not filtered.
    assert summary['witnesses'] == int((pool.witnesses.robot >= 0).sum())
    assert (pool.witnesses.robot >= -1).all() and (pool.witnesses.robot < robots).all()
  else:
    assert ((pool.witnesses.robot >= 0) & (pool.witnesses.robot < robots)).all()
    assert summary['witness_missing'] == 0, 'Reachable by construction, DESIGN G5'
    assert summary['witnesses'] == COUNTS[split] * targets


def test_metadata(pool_case):
  robots, targets, split, pool, summary = pool_case
  meta = pool.meta
  assert meta['robots'] == robots and meta['targets'] == targets and meta['obstacles'] == 30
  assert meta['split'] == split and meta['arm_model'] == 'spheres_stick'
  assert 'restarts' in meta['budget'] and isinstance(meta['seed'], int)
  assert len(meta['git_revision']) == 40 and len(meta['code_sha256']) == 64
  assert meta['c_check']['scenes'] >= COUNTS[split]
  print(f"\n{robots}r_{targets}t {split}: {meta.get('rates')}, {meta['generation_s']:.1f} s, "
        f"generator c_check {meta['c_check']}")
  if split != 'static_paper':
    assert 0 < meta['rates']['target_accept'] < 1


def test_generated_targets_follow_the_stored_convention(pool_case):
  _, _, split, pool, _ = pool_case
  if split == 'static_paper':
    return  # The paper's targets are not on the 3 cm rule (env_spec 11.6).
  scene = pool.scene
  has_owner, residual, angle = face_stats(type(scene)(*[f[:2000] for f in scene]))
  assert has_owner.all() and residual.max() < 1e-5 and angle.max() <= 22.5 + 1e-2


def test_static_cells(pool_case):
  robots, targets, split, pool, _ = pool_case
  if not split.startswith('static'):
    return
  base_pos, base_rot = default_bases(WorldCfg(num_robots=robots), DEVICE)
  assert (pool.scene.base_pos == base_pos).all() and (pool.scene.base_rot == base_rot).all()
  for field in (pool.scene.box_pos, pool.scene.box_rot, pool.scene.box_spans):
    assert (field == field[0]).all(), 'Every static set holds the same boxes'
  check_static_boxes(pool.scene.box_pos, pool.scene.box_rot, pool.scene.box_spans, robots)
  assert 'env_spec D4' in pool.meta['box_rotation']


def test_paper_tasksets_are_the_paper_targets(pool_case):
  robots, targets, split, pool, summary = pool_case
  if split != 'static_paper':
    return
  pos, rot = load_taskset(DATA / f'paper_test_tasksets/test_{targets}targets.pbtxt', DEVICE)
  lift = torch.as_tensor([0, 0, 0.6 if robots == 8 else 0.0], device=DEVICE)
  assert torch.equal(pool.scene.target_pos, pos + lift) and torch.equal(pool.scene.target_rot, rot)
  ceiling = pool.meta['reachability_ceiling']
  assert ceiling == pytest.approx(float((pool.witnesses.robot >= 0).float().mean()))
  print(f'\n{robots}r_{targets}t paper taskset ceiling {ceiling:.3f}, per set {pool.meta["reachable_per_set"]}')
  assert 0.5 < ceiling <= 1.0


@pytest.mark.parametrize('robots,targets', CONFIGS)
def test_train_and_heldout_share_no_scene(robots, targets):
  train, heldout = _load(robots, targets, 'train'), _load(robots, targets, 'heldout')
  a, b = train.scene.box_pos[:, 0].cpu().numpy(), heldout.scene.box_pos[:, 0].cpu().numpy()
  assert len(np.unique(np.concatenate([a, b]), axis=0)) == len(a) + len(b)


@pytest.mark.parametrize('robots,targets', CONFIGS)
def test_static_new_sets_differ_from_each_other(robots, targets):
  static = _load(robots, targets, 'static')
  pos = static.scene.target_pos.flatten(1).cpu().numpy()
  assert len(np.unique(pos, axis=0)) == COUNTS['static']
