"""scenes/ik.py: FK, the solver, and the witness search that the G5 pools rest on."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, EnvCfg, WorldCfg
from roboballet_mjlab.env.ballet_env import targets_within
from roboballet_mjlab.scenes import ik, sampler
from roboballet_mjlab.sim.world import build_world
from tests.test_scenes_common import angular_distance, recheck_scene
from tests.test_sim_common import DEVICE, CWorld

NUM_WORLDS = 256


@pytest.fixture(scope='module')
def world():
  return build_world(WorldCfg(), NUM_WORLDS, DEVICE)


@pytest.fixture(scope='module')
def kin():
  return ik.ArmKinematics('spheres_stick', DEVICE)


def _gen(seed=0):
  return torch.Generator(DEVICE).manual_seed(seed)


def _random_q(kin, n, gen):
  return kin.lower + torch.rand(n, 7, device=DEVICE, generator=gen) * (kin.upper - kin.lower)


def _random_bases(n, gen):
  pos = torch.rand(n, 3, device=DEVICE, generator=gen) - 0.5
  return pos, sampler.random_rotations(n, DEVICE, gen)


@pytest.mark.parametrize('arm', ['spheres_stick', 'mesh_tool'])
def test_fk_matches_c_mujoco(arm):
  """E1 tolerance: 1e-4 m and 1e-3 rad, against a world built without sim/world.py."""
  kin = ik.ArmKinematics(arm, DEVICE)
  gen = _gen(1)
  n = 500
  q, (base_pos, base_rot) = _random_q(kin, n, gen), _random_bases(n, gen)
  pos, rot, _, _ = kin.fk(q, base_pos, base_rot)
  world = CWorld(1, 1, arm)
  worst_pos, worst_ang = 0.0, 0.0
  for i in range(n):
    world.set_scene(base_pos[i, None].cpu().numpy(), base_rot[i, None].cpu().numpy(),
                    np.array([[0, 0, 50.0]]), np.eye(3)[None], np.array([[0.05, 0.05, 0.05]]))
    world.set_q(q[i, None].cpu().numpy())
    world.forward()
    c_pos, c_rot = world.tcp()
    worst_pos = max(worst_pos, float(np.linalg.norm(c_pos[0] - pos[i].cpu().numpy())))
    worst_ang = max(worst_ang, angular_distance(c_rot[0], rot[i].cpu().numpy()))
  print(f'{arm}: FK vs C MuJoCo, worst pos {worst_pos:.2e} m, worst angle {worst_ang:.2e} rad')
  assert worst_pos <= 1e-4 and worst_ang <= 1e-3


def test_fk_matches_world(world, kin):
  gen = _gen(2)
  scene = sampler.blank_scene(world, NUM_WORLDS, 1)
  w = torch.arange(NUM_WORLDS, device=DEVICE)
  q = _random_q(kin, NUM_WORLDS * world.num_robots, gen).view(NUM_WORLDS, world.num_robots, 7)
  world.set_scene(w, scene)
  world.set_q(q)
  world.forward()
  world_pos, world_rot = world.tcp()
  for r in range(world.num_robots):
    pos, rot, _, _ = kin.fk(q[:, r], scene.base_pos[:, r], scene.base_rot[:, r])
    assert (pos - world_pos[:, r]).abs().max() < 1e-4
    assert ik.rotation_error(rot, world_rot[:, r]).norm(dim=-1).max() < 1e-3


def test_rotation_error_inverts_exp():
  gen = _gen(3)
  n = 2000
  axis = ik._unit_gaussian((n, 3), DEVICE, gen)
  angle = torch.rand(n, 1, device=DEVICE, generator=gen) * 3.1
  w = axis * angle
  rot = sampler.random_rotations(n, DEVICE, gen)
  assert (ik.rotation_error(ik.rotation_exp(w) @ rot, rot) - w).abs().max() < 2e-3


def test_rotation_error_magnitude_is_the_scored_angle():
  gen = _gen(4)
  a, b = sampler.random_rotations(200, DEVICE, gen), sampler.random_rotations(200, DEVICE, gen)
  mine = ik.rotation_error(a, b).norm(dim=-1).cpu().numpy()
  theirs = np.array([angular_distance(x, y) for x, y in zip(a.cpu().numpy(), b.cpu().numpy())])
  assert np.abs(mine - theirs).max() < 1e-3


def _half_turn_about_local_x(rot):
  """rot with its local y and z columns negated."""
  return rot @ torch.diag(torch.as_tensor([1.0, -1.0, -1.0], device=rot.device))


def _world_tcp(world, kin, n):
  """TCP of robot 0 on World, at home for the first world and random joints for the rest."""
  scene = sampler.blank_scene(world, n, 1)
  w = torch.arange(n, device=DEVICE)
  q = scene.q0.clone()
  q[1:, 0] = _random_q(kin, n - 1, _gen(11))
  world.set_scene(w, scene)
  world.set_q(q, w)
  world.forward()
  pos, rot = world.tcp()
  return scene, w, q, pos[:n, 0].clone(), rot[:n, 0].clone()


def test_half_turn_of_a_real_tcp_is_rejected_as_the_env_rejects_it(world, kin):
  """A target at the TCP position with local y and z negated is 180 degrees away."""
  n = 16
  scene, ids, q, pos, rot = _world_tcp(world, kin, n)
  target = _half_turn_about_local_x(rot)
  robot = torch.zeros(n, dtype=torch.long, device=DEVICE)
  budget = ik.IkBudget()
  assert not targets_within(pos[:, None], rot[:, None], pos[:, None], target[:, None], EnvCfg()).any()
  assert not ik.within_tolerance(pos, rot, pos, target, budget).any()
  assert not ik.witness_valid(world, scene, ids, robot, q[:, 0], pos, target, budget).any()
  assert ik.witness_valid(world, scene, ids, robot, q[:, 0], pos, rot, budget)[0]


def test_rotation_error_of_a_real_half_turn_is_pi_about_local_x(world, kin):
  n = 16
  _, _, _, _, rot = _world_tcp(world, kin, n)
  target = _half_turn_about_local_x(rot)
  w = ik.rotation_error(target, rot)
  assert (w.norm(dim=-1) - math.pi).abs().max() < 1e-3
  local_x = rot[:, :, 0]
  assert ((w * local_x).sum(-1).abs() - math.pi).abs().max() < 1e-3
  assert (ik.rotation_exp(w) @ rot - target).abs().max() < 1e-3


# Below the branch the axis is good to f32. Above it, v carries noise of a few 1e-7 over sin.
@pytest.mark.parametrize('off_pi,tol', [(0.0, 2e-3), (1e-6, 2e-3), (1e-5, 2e-3), (5e-5, 2e-3),
                                        (1e-4, 1e-2), (2e-4, 5e-3), (1e-3, 2e-3), (1e-2, 2e-3)])
def test_rotation_error_inverts_exp_near_pi(off_pi, tol):
  gen = _gen(12)
  n = 2000
  axis = ik._unit_gaussian((n, 3), DEVICE, gen)
  rot = sampler.random_rotations(n, DEVICE, gen)
  target = ik.rotation_exp(axis * (math.pi - off_pi)) @ rot
  w = ik.rotation_error(target, rot)
  assert (w.norm(dim=-1) - (math.pi - off_pi)).abs().max() < 2e-3
  assert (ik.rotation_exp(w) @ rot - target).abs().max() < tol
  # At the half turn either sign of the axis is the same rotation.
  assert ((w * axis).sum(-1).abs() - (math.pi - off_pi)).abs().max() < tol


def test_rotation_error_is_unchanged_away_from_pi():
  """The half-turn branch must not touch ordinary rotations, or every pool would move."""
  gen = _gen(13)
  a, b = sampler.random_rotations(5000, DEVICE, gen), sampler.random_rotations(5000, DEVICE, gen)
  e = a @ b.transpose(-1, -2)
  v = 0.5 * torch.stack(
    [e[:, 2, 1] - e[:, 1, 2], e[:, 0, 2] - e[:, 2, 0], e[:, 1, 0] - e[:, 0, 1]], dim=-1)
  sin = v.norm(dim=-1, keepdim=True)
  cos = 0.5 * (e.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1)
  keep = (sin[:, 0] >= 1e-3)
  expected = v * (torch.atan2(sin, cos) / sin.clamp(min=1e-7))
  assert torch.equal(ik.rotation_error(a, b)[keep], expected[keep])


def test_descent_does_not_stall_at_a_half_turn(kin):
  """FK gives a target whose relative rotation is exactly symmetric: no antisymmetric part."""
  gen = _gen(14)
  n = 200
  budget = ik.IkBudget()
  goal, (base_pos, base_rot) = _random_q(kin, n, gen), _random_bases(n, gen)
  pos, rot, _, _ = kin.fk(goal, base_pos, base_rot)
  target = _half_turn_about_local_x(rot)
  e = target @ rot.transpose(-1, -2)
  assert torch.equal(e, e.transpose(-1, -2)), 'The premise of this test'
  one = ik.descend(kin, goal, pos, target, base_pos, base_rot, dataclasses.replace(budget, iters=1))
  assert (one - goal).abs().amax(-1).min() > 0.01, 'The first step must move every arm'
  q = ik.descend(kin, goal, pos, target, base_pos, base_rot, budget)
  _, end_rot, _, _ = kin.fk(q, base_pos, base_rot)
  angle = ik.rotation_error(target, end_rot).norm(dim=-1)
  print(f'half-turn start: median angle after a descent {angle.median():.3f} rad')
  assert angle.median() < 1.0


def test_within_tolerance_is_the_env_predicate():
  """With no margin it equals targets_within across the whole angle range, both sides of 15 degrees."""
  gen = _gen(15)
  cfg = EnvCfg()
  budget = ik.IkBudget(margin=1.0, pos_tol=cfg.position_threshold, ang_tol=cfg.angle_threshold_rad)
  axis = ik._unit_gaussian((6000, 3), DEVICE, gen)
  angle = torch.cat([torch.linspace(0, math.pi, 3000, device=DEVICE),
                     cfg.angle_threshold_rad + torch.linspace(-3e-3, 3e-3, 3000, device=DEVICE)])
  rot = sampler.random_rotations(6000, DEVICE, gen)
  target = ik.rotation_exp(axis * angle[:, None]) @ rot
  pos = torch.zeros(6000, 3, device=DEVICE)
  mine = ik.within_tolerance(pos, rot, pos, target, budget)
  theirs = targets_within(pos[:, None], rot[:, None], pos[:, None], target[:, None], cfg)[:, 0, 0]
  assert torch.equal(mine, theirs)
  assert mine.any() and not mine.all()


def test_within_tolerance_keeps_the_margin_on_the_angle():
  budget = ik.IkBudget()
  rot = torch.eye(3, device=DEVICE)[None]
  pos = torch.zeros(1, 3, device=DEVICE)
  turned = lambda angle: ik.rotation_exp(torch.as_tensor([[0, 0, angle]], device=DEVICE, dtype=torch.float32)) @ rot
  inside = budget.ang_tol * (budget.margin - 2e-4)
  between = budget.ang_tol * (budget.margin + (1 - budget.margin) / 2)
  assert ik.within_tolerance(pos, rot, pos, turned(inside), budget).all()
  assert not ik.within_tolerance(pos, rot, pos, turned(between), budget).any()
  env = targets_within(pos[:, None], rot[:, None], pos[:, None], turned(between)[:, None], EnvCfg())
  assert env.all(), 'The margin sits inside what the env accepts'


def test_aim_stays_inside_the_tolerance():
  gen = _gen(5)
  budget = ik.IkBudget()
  n = 5000
  pos = torch.zeros(n, 3, device=DEVICE)
  rot = sampler.random_rotations(n, DEVICE, gen)
  aim_pos, aim_rot = ik._aim_inside_tolerance(pos, rot, budget, gen)
  pos_err = (aim_pos - pos).norm(dim=-1)
  ang_err = ik.rotation_error(aim_rot, rot).norm(dim=-1)
  assert pos_err.max() <= budget.aim_fraction * budget.pos_tol * 1.0001
  assert ang_err.max() <= budget.aim_fraction * budget.ang_tol * 1.001
  # Uniform in the ball: the median radius is 0.794 of the maximum.
  assert abs(pos_err.median() / (budget.aim_fraction * budget.pos_tol) - 0.794) < 0.02


def test_descend_reaches_reachable_poses(kin):
  gen = _gen(6)
  n = 4000
  budget = ik.IkBudget()
  goal, (base_pos, base_rot) = _random_q(kin, n, gen), _random_bases(n, gen)
  pos, rot, _, _ = kin.fk(goal, base_pos, base_rot)
  q = ik.descend(kin, goal + 0.3 * (2 * torch.rand(n, 7, device=DEVICE, generator=gen) - 1) * 0.5,
                 pos, rot, base_pos, base_rot, budget)
  assert (q >= kin.lower).all() and (q <= kin.upper).all()
  tcp_pos, tcp_rot, _, _ = kin.fk(q, base_pos, base_rot)
  rate = ik.within_tolerance(tcp_pos, tcp_rot, pos, rot, budget).float().mean()
  print(f'near start, one descent: {rate:.4f}')
  assert rate > 0.97


def test_descend_step_is_bounded(kin):
  gen = _gen(7)
  n = 1000
  one = dataclasses.replace(ik.IkBudget(), iters=1)
  q0, (base_pos, base_rot) = _random_q(kin, n, gen), _random_bases(n, gen)
  pos, rot, _, _ = kin.fk(_random_q(kin, n, gen), base_pos, base_rot)
  q1 = ik.descend(kin, q0, pos, rot, base_pos, base_rot, one)
  assert (q1 - q0).abs().max() <= one.max_step_rad * 1.0001


def _scene_with_one_box(world, n, box_pos, spans):
  scene = sampler.blank_scene(world, n, 1)
  scene.box_pos[:, 0] = torch.as_tensor(box_pos, device=DEVICE)
  scene.box_spans[:, 0] = torch.as_tensor(spans, device=DEVICE)
  return scene


def test_out_of_reach_costs_nothing(world, kin):
  """The reach gate of mujoco_simulator.py:522-526: no restart runs, no witness appears."""
  scene = sampler.blank_scene(world, 4, 1)
  ids = torch.arange(4, device=DEVICE)
  pos = torch.full((4, 3), 5.0, device=DEVICE)
  rot = torch.eye(3, device=DEVICE).repeat(4, 1, 1)
  result = ik.find_witnesses(world, kin, scene, ids, pos, rot, ik.IkBudget(restarts=5), _gen())
  assert not result.found.any() and (result.robot == -1).all() and result.restarts_run == 0


def _tcp_of(kin, scene, ids, robot, q):
  return kin.fk(q, scene.base_pos[ids, robot], scene.base_rot[ids, robot])[:2]


def test_search_is_bounded_when_no_pose_is_free(world, kin):
  """A huge box swallows the target and the arm: full budget spent, nothing accepted."""
  n = 8
  scene = _scene_with_one_box(world, n, (0.3, -0.6, 0.4), (0.6, 0.6, 0.6))
  ids = torch.arange(n, device=DEVICE)
  q = torch.as_tensor(HOME_QPOS, device=DEVICE).repeat(n, 1)
  q[:, 3] += 0.3
  pos, rot = _tcp_of(kin, scene, ids, torch.zeros(n, dtype=torch.long, device=DEVICE), q)
  budget = ik.IkBudget(restarts=3, iters=10)
  in_reach = ((pos[:, None] - scene.base_pos).norm(dim=-1) <= budget.reach_m).sum()
  result = ik.find_witnesses(world, kin, scene, ids, pos, rot, budget, _gen())
  assert not result.found.any()
  assert result.restarts_run == in_reach * budget.restarts


def test_witness_valid_rejects_collision_and_distance(world, kin):
  n = 2
  scene = sampler.blank_scene(world, n, 1)
  scene.box_pos[1, 0] = scene.base_pos[1, 0] + torch.as_tensor([0, 0, 0.5], device=DEVICE)
  scene.box_spans[1, 0] = torch.as_tensor([0.3, 0.3, 0.3], device=DEVICE)
  ids = torch.arange(n, device=DEVICE)
  robot = torch.zeros(n, dtype=torch.long, device=DEVICE)
  q = torch.as_tensor(HOME_QPOS, device=DEVICE).repeat(n, 1)
  pos, rot = _tcp_of(kin, scene, ids, robot, q)
  budget = ik.IkBudget()
  ok = ik.witness_valid(world, scene, ids, robot, q, pos, rot, budget)
  assert ok.tolist() == [True, False]
  far = ik.witness_valid(world, scene, ids, robot, q, pos + 0.03, rot, budget)
  assert far.tolist() == [False, False]
  turned = ik.rotation_exp(torch.as_tensor([[0, 0, math.radians(16)]] * n, device=DEVICE)) @ rot
  assert not ik.witness_valid(world, scene, ids, robot, q, pos, turned, budget).any()


def test_witnesses_pass_on_c_mujoco(world, kin):
  """Search on the GPU, then an independent check of what it returned."""
  gen = _gen(8)
  cfg = sampler.SamplerCfg()
  n = 32
  scene = sampler.blank_scene(world, n, 1)
  assert not sampler.sample_rails(world, scene, cfg, gen)[0].any()
  assert not sampler.sample_boxes(world, scene, cfg, gen)[0].any()
  ids = torch.arange(n, device=DEVICE).repeat_interleave(4)
  pos, rot = sampler.draw_candidates(scene, ids, cfg, gen)
  result = ik.find_witnesses(world, kin, scene, ids, pos, rot, ik.IkBudget(restarts=20), gen)
  assert result.found.any()
  c = CWorld(world.num_robots, world.num_obstacles)
  for i in result.found.nonzero()[:, 0].tolist():
    s = int(ids[i])
    arrays = {k: v[s].cpu().numpy() for k, v in scene._asdict().items()}
    arrays['target_pos'] = pos[i, None].cpu().numpy()
    arrays['target_rot'] = rot[i, None].cpu().numpy()
    row = recheck_scene(c, arrays, result.q[i, None].cpu().numpy(), result.robot[i, None].cpu().numpy())
    assert not row['home_collision'] and not row['witness_collision'].any()
    assert not row['witness_far'].any() and not row['witness_limits'].any()


def test_search_is_deterministic(world, kin):
  cfg = sampler.SamplerCfg()
  gen = _gen(9)
  scene = sampler.blank_scene(world, 8, 1)
  sampler.sample_rails(world, scene, cfg, gen)
  sampler.sample_boxes(world, scene, cfg, gen)
  ids = torch.arange(8, device=DEVICE).repeat_interleave(4)
  pos, rot = sampler.draw_candidates(scene, ids, cfg, gen)
  budget = ik.IkBudget(restarts=5)
  one = ik.find_witnesses(world, kin, scene, ids, pos, rot, budget, _gen(10))
  two = ik.find_witnesses(world, kin, scene, ids, pos, rot, budget, _gen(10))
  assert torch.equal(one.q, two.q) and torch.equal(one.robot, two.robot)
