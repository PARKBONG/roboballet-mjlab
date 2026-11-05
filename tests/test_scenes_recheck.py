"""The C MuJoCo recheck must be able to fail, or its zeros in test_scenes_pools mean nothing."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import JOINT_UPPER, WorldCfg
from roboballet_mjlab.scenes import ik, sampler
from roboballet_mjlab.scenes.ik import rotation_exp
from roboballet_mjlab.sim.world import build_world
from tests.test_scenes_common import ANG_TOL, POS_TOL, recheck_scene, summarise_recheck
from tests.test_sim_common import DEVICE, CWorld


@pytest.fixture(scope='module')
def case():
  """One real scene with one target, and the TCP pose of its witness on C MuJoCo."""
  world = build_world(WorldCfg(), 16, DEVICE)
  kin = ik.ArmKinematics('spheres_stick', DEVICE)
  gen = torch.Generator(DEVICE).manual_seed(0)
  made = sampler.generate(world, kin, 16, 1, sampler.SamplerCfg(), ik.IkBudget(), gen)
  scene = {k: v[0].cpu().numpy() for k, v in made.scene._asdict().items()}
  q, robot = made.witnesses.q[0].cpu().numpy(), made.witnesses.robot[0].cpu().numpy()
  c = CWorld(4, 30)
  assert not _flags(recheck_scene(c, scene, q, robot))
  all_q = scene['q0'].copy()
  all_q[robot[0]] = q[0]
  c.set_q(all_q)
  c.forward()
  pos, rot = c.tcp()
  return c, scene, q, robot, pos[robot[0]].copy(), rot[robot[0]].copy()


def _flags(row: dict) -> list[str]:
  names = ('witness_collision', 'witness_far', 'witness_limits', 'witness_missing')
  out = [name for name in names if row[name].any()]
  return out + (['home_collision'] if row['home_collision'] else [])


def _with(scene, **changes):
  return {**scene, **changes}


def test_box_on_a_robot_at_home_is_caught(case):
  c, scene, q, robot, _, _ = case
  box_pos = scene['box_pos'].copy()
  box_pos[0] = scene['base_pos'][0] + [0, 0, 0.5]
  assert 'home_collision' in _flags(recheck_scene(c, _with(scene, box_pos=box_pos), q, robot))


def test_box_on_the_witness_tip_is_caught(case):
  c, scene, q, robot, pos, _ = case
  box_pos = scene['box_pos'].copy()
  box_pos[0] = pos
  row = recheck_scene(c, _with(scene, box_pos=box_pos), q, robot)
  assert row['witness_collision'].all() and not row['home_collision']


def test_other_robots_stay_at_q0_in_the_witness_check(case):
  """A box on the home pose of another robot collides in the home check, whichever robot moves."""
  c, scene, q, robot, _, _ = case
  other = (robot[0] + 1) % 4
  box_pos = scene['box_pos'].copy()
  box_pos[0] = scene['base_pos'][other] + [0, 0, 0.5]
  row = recheck_scene(c, _with(scene, box_pos=box_pos), q, robot)
  assert row['witness_collision'].all()


@pytest.mark.parametrize('distance,far', [(POS_TOL - 1e-5, False), (POS_TOL, True), (POS_TOL + 1e-3, True)])
def test_position_tolerance_is_strict(case, distance, far):
  c, scene, q, robot, pos, rot = case
  target_pos = (pos + [distance, 0, 0])[None].astype(np.float64)
  row = recheck_scene(c, _with(scene, target_pos=target_pos, target_rot=rot[None]), q, robot)
  assert bool(row['witness_far'].all()) == far


@pytest.mark.parametrize('angle,far', [(ANG_TOL - 1e-3, False), (ANG_TOL + 1e-3, True)])
def test_angle_tolerance_is_strict(case, angle, far):
  c, scene, q, robot, pos, rot = case
  turn = rotation_exp(torch.as_tensor([[0.0, 0.0, angle]], dtype=torch.float64)).numpy()[0]
  target_rot = (turn @ rot)[None]
  row = recheck_scene(c, _with(scene, target_pos=pos[None], target_rot=target_rot), q, robot)
  assert bool(row['witness_far'].all()) == far


def test_joint_limit_is_caught(case):
  c, scene, q, robot, _, _ = case
  bad = q.copy()
  bad[0, 0] = JOINT_UPPER[0] + 0.01
  assert recheck_scene(c, scene, bad, robot)['witness_limits'].all()


def test_missing_witness_is_reported_and_summarised(case):
  c, scene, q, robot, _, _ = case
  row = recheck_scene(c, scene, q, np.full_like(robot, -1))
  assert row['witness_missing'].all()
  summary = summarise_recheck([row, recheck_scene(c, scene, q, robot)])
  assert summary['scenes'] == 2 and summary['witnesses'] == 1 and summary['witness_missing'] == 1
  assert summary['worst_pos_m'] < POS_TOL and summary['worst_ang_deg'] < math.degrees(ANG_TOL)
