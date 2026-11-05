"""Gate G2: rollback table of env_spec 5.4 and the rows of env_spec 15, in the ported env."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_UPPER, EnvCfg
from roboballet_mjlab.env.ballet_env import BalletEnv, solve_targets
from roboballet_mjlab.env.poses import augment_boxes
from roboballet_mjlab.sim.world import build_world
from roboballet_mjlab.structs import Scene
from tests.test_sim_common import DEVICE, default_scene, env_cfg, stored_scenes

HOME = torch.as_tensor(HOME_QPOS, device=DEVICE)
ARMS = ['mesh_tool', 'spheres_stick']  # env_spec 15: original tool first, then the stick.


def _tip_at(cfg: EnvCfg, scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
  """TCP poses of scene.q0, for placing targets exactly on them."""
  n = len(scene.q0)
  world = build_world(cfg.world, n, DEVICE)
  world.set_scene(torch.arange(n, device=DEVICE), scene)
  world.set_q(scene.q0)
  world.forward()
  return world.tcp()


def _env(cfg: EnvCfg, scene: Scene, seed: int = 0, n: int | None = None) -> BalletEnv:
  n = len(scene.q0) if n is None else n
  env = BalletEnv(cfg, n, scene, DEVICE, seed, debug=True)
  env.reset(scene_ids=torch.arange(n, device=DEVICE) % len(scene.q0))
  return env


def _cfg(arm='spheres_stick', **kw) -> EnvCfg:
  return env_cfg(num_obstacles=1, arm=arm, **kw)


def _vel(cfg: EnvCfg, n: int, values: dict[tuple[int, int], float]) -> torch.Tensor:
  """Action [n, R, 7] from rad/s per (robot, joint)."""
  action = torch.zeros(n, cfg.world.num_robots, 7, device=DEVICE)
  for (robot, joint), v in values.items():
    action[:, robot, joint] = v / cfg.max_joint_velocity
  return action


# env_spec 15, motion rows.


@pytest.mark.parametrize('arm', ARMS)
def test_normal_stepping(arm):
  cfg = _cfg(arm, timestep_s=0.5)
  env = _env(cfg, default_scene(cfg.world, 1))
  action = _vel(cfg, 1, {(r, j): 0.02 for r in (0, 1) for j in range(7)})
  q = env.step(action).next_robot.q[0]
  assert torch.allclose(q[:2], HOME + 0.01, atol=1e-5)
  assert torch.allclose(q[2:], HOME, atol=1e-7)


@pytest.mark.parametrize('arm', ARMS)
def test_acceleration_clip(arm):
  cfg = _cfg(arm, timestep_s=0.01)
  env = _env(cfg, default_scene(cfg.world, 1))
  q0 = HOME[1].item()
  out = env.step(_vel(cfg, 1, {(0, 1): 1.0}))
  assert out.next_robot.qd[0, 0, 1].item() == pytest.approx(0.08, abs=1e-6)
  assert out.next_robot.q[0, 0, 1].item() == pytest.approx(q0 + 0.08 * 0.01, abs=1e-6)
  out = env.step(_vel(cfg, 1, {(0, 1): 0.12}))
  assert out.next_robot.q[0, 0, 1].item() == pytest.approx(q0 + 0.2 * 0.01, abs=1e-6)


def test_velocity_ramp_stop_and_limit_keeps_velocity():
  cfg = _cfg()
  env = _env(cfg, default_scene(cfg.world, 1))
  speeds = [env.step(_vel(cfg, 1, {(1, 6): 2.175})).next_robot.qd[0, 1, 6].item() for _ in range(40)]
  assert speeds[:3] == pytest.approx([0.8, 1.6, 2.175], abs=1e-5)
  # Joint 7 holds at its limit while the observed velocity stays commanded, env_spec 4.
  q = env.observe().robot.q[0, 1, 6].item()
  assert q == pytest.approx(JOINT_UPPER[6], abs=1e-5)
  assert env.observe().robot.qd[0, 1, 6].item() == pytest.approx(2.175, abs=1e-5)
  stops = [env.step(_vel(cfg, 1, {})).next_robot.qd[0, 1, 6].item() for _ in range(3)]
  assert stops == pytest.approx([1.375, 0.575, 0.0], abs=1e-5)


def test_velocity_is_not_clipped_to_the_action_bound():
  cfg = _cfg()
  env = _env(cfg, default_scene(cfg.world, 1))
  action = torch.zeros(1, 4, 7, device=DEVICE)
  action[0, 1, 6] = 1.2  # 2.61 rad/s, env_spec 4: only the acceleration clamp applies.
  for _ in range(5):
    out = env.step(action)
  assert out.next_robot.qd[0, 1, 6].item() == pytest.approx(2.61, abs=1e-5)


def test_action_tensor_is_not_mutated():
  cfg = _cfg(num_targets=1, dwell_time_s=0.5)
  scene = default_scene(cfg.world, 1, 1)
  pos, rot = _tip_at(cfg, scene)
  scene = scene._replace(target_pos=pos[:, :1], target_rot=rot[:, :1])
  env = _env(cfg, scene)
  action = torch.full((1, 4, 7), 0.01, device=DEVICE)
  before = action.clone()
  env.step(action)
  env.step(action)
  assert torch.equal(action, before)


def test_max_steps_and_terminal_at_timeout():
  # base_env_test.py:113-126: LAST on the step that reaches the limit.
  cfg = _cfg(timestep_s=0.5, num_targets=3)
  assert cfg.max_steps == 10
  env = _env(cfg, default_scene(cfg.world, 1, 3))
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  flags = [env.step(zero).terminal.item() for _ in range(10)]
  assert flags == [False] * 9 + [True]
  assert env.observe().robot.step.item() == 0  # Immediate reset, DESIGN section 5.


def test_constants_for_all_cells():
  for num_targets, steps in ((10, 120), (20, 220), (40, 420)):
    assert EnvCfg(num_targets=num_targets).max_steps == steps
  assert EnvCfg().dwell_steps == 5
  assert EnvCfg().collision_penalty_per_robot == pytest.approx(0.03125)


# Table crash, env_spec 15 and planning_env_test.py:229-248.


def test_table_crash_on_mesh_tool():
  cfg = _cfg('mesh_tool', timestep_s=0.5, num_targets=3, collision_penalty_scaling=42.0)
  env = _env(cfg, default_scene(cfg.world, 1, 3))
  action = _vel(cfg, 1, {(0, 1): 0.6})
  rollbacks, penalty = [], 0.0
  for _ in range(9):
    out = env.step(action)
    rollbacks.append(out.collision_robot_steps.item())
    penalty += out.collision_penalty.item()
    last_q = out.next_robot.q[0, 0, 1].item()
    assert out.next_robot.qd[0, 0, 1].item() == (0.0 if rollbacks[-1] else pytest.approx(0.6))
  out = env.step(action)
  rollbacks.append(out.collision_robot_steps.item())
  penalty += out.collision_penalty.item()
  assert out.next_robot.q[0, 0, 1].item() == pytest.approx(0.4146, abs=1e-3)
  assert last_q == pytest.approx(0.4146, abs=1e-3)
  assert rollbacks == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  assert penalty == pytest.approx(42.0 * (1 / 4) * (6 / 10), abs=1e-5)
  assert sum(rollbacks) == 6


def test_table_crash_on_stick_only_hits_that_robot():
  cfg = _cfg(timestep_s=0.5, num_targets=3, collision_penalty_scaling=42.0)
  env = _env(cfg, default_scene(cfg.world, 4, 3))
  action = _vel(cfg, 4, {(0, 1): 0.6})
  rollbacks = []
  for _ in range(10):
    out = env.step(action)
    rollbacks.append(out.collision_robot_steps.cpu().tolist())
    # The other arms are not disturbed and it never terminates early from a collision.
    assert torch.allclose(out.next_robot.q[:, 1:], HOME, atol=1e-7)
  assert out.terminal.all()
  assert all(r == [r[0]] * 4 for r in rollbacks)  # Identical worlds, identical outcome.
  flat = [r[0] for r in rollbacks]
  first = flat.index(1)
  assert first >= 3 and flat[first:] == [1] * (10 - first)


# Rollback table of env_spec 5.4, on many identical worlds in one step, on both arms.


def _boundary_scene(cfg):
  """panda1 and panda4 share a rail, panda2 and panda3 are parked far along the other."""
  scene = default_scene(cfg.world, 1, cfg.num_targets)
  pos = scene.base_pos.clone()
  pos[0, 0, 1], pos[0, 1, 1], pos[0, 2, 1], pos[0, 3, 1] = -0.3, -0.7, 0.7, 0.3
  return scene._replace(base_pos=pos)


class _Pair:
  """Finds the contact boundary of panda1 and panda4 swinging about joint 1 (spec/probe3.py)."""

  def __init__(self, arm):
    self.arm = arm
    self.cfg = self.env_cfg()
    self.scene = _boundary_scene(self.cfg)
    self.n = 2801
    self.grid = torch.arange(self.n, device=DEVICE) * 0.001
    self.world = build_world(self.cfg.world, self.n, DEVICE)
    expanded = Scene(*[x.expand(self.n, *x.shape[1:]) for x in self.scene])
    self.world.set_scene(torch.arange(self.n, device=DEVICE), expanded)

  def env_cfg(self, **kw):
    # 10 targets, so that the rewards are the 4r10t ones of the table. The mesh pool is
    # smaller because CCD memory scales with it, and these states have few candidates.
    return env_cfg(num_obstacles=1, arm=self.arm, nconmax=128, **kw)

  def collide(self, q1, q4):
    """Flags [n, R] with joint 1 of panda1 and panda4 at the given angles."""
    q = HOME.repeat(self.n, 4, 1)
    q[:, 0, 0] = q1
    q[:, 3, 0] = q4
    self.world.set_q(q)
    self.world.forward()
    return self.world.robots_in_collision()

  def first_contact(self, q1=None):
    """First grid angle in contact, panda4 at +t and panda1 at -t or fixed at q1."""
    flags = self.collide(-self.grid if q1 is None else q1, self.grid).any(1)
    assert flags.any()
    return self.grid[flags.int().argmax()].item()

  def both_new_case(self):
    """Start angles and a move with contact only at (new, new), env_spec 5.4 row 1."""
    th = self.first_contact()
    e = 0.002
    for d in (0.002, 0.004, 0.008, 0.016, 0.032, 0.064):
      a, b, a2, b2 = -(th - d), th - d, -(th + e), th + e
      free = not (self.collide(a, b).any() or self.collide(a2, b).any()
                  or self.collide(a, b2).any())
      if free and self.collide(a2, b2)[0, [0, 3]].all():
        return a, b, d + e
    raise AssertionError('no start with contact only at (new, new)')

  def innocent_case(self):
    """panda1 waits further along, panda4 walks into it. env_spec 5.4 row 2."""
    a3 = -(self.first_contact() + 0.15)
    b_free = self.first_contact(q1=a3) - 0.004
    dq = 0.008
    assert not self.collide(a3, b_free).any()
    assert self.collide(a3, b_free + dq)[0, [0, 3]].all()
    return a3, b_free, dq


@pytest.fixture(scope='module', params=ARMS)
def pair(request):
  return _Pair(request.param)


def _one_step_outcomes(cfg, scene, q1, q4, moves, n=4000, seed=1):
  """Runs n copies of one step from joint-1 angles (q1, q4). Returns the StepOut and start q."""
  q0 = HOME.repeat(1, 4, 1)
  q0[0, 0, 0], q0[0, 3, 0] = q1, q4
  env = BalletEnv(cfg, n, scene._replace(q0=q0), DEVICE, seed, debug=True)
  action = _vel(cfg, n, {key: dq / cfg.timestep_s for key, dq in moves.items()})
  start = env.observe().robot.q.clone()
  return env.step(action), start


def test_rollback_both_moving_exactly_one_uniform(pair):
  a, b, dq = pair.both_new_case()
  n = 4000
  out, start = _one_step_outcomes(pair.cfg, pair.scene, a, b, {(0, 0): -dq, (3, 0): dq}, n=n)
  robot = out.next_robot
  assert (out.collision_robot_steps == 1).all()
  assert (out.reward + 0.03125).abs().max() < 1e-6
  moved1 = (robot.q[:, 0, 0] - start[:, 0, 0]).abs() > 1e-5
  moved4 = (robot.q[:, 3, 0] - start[:, 3, 0]).abs() > 1e-5
  assert (moved1 ^ moved4).all()
  assert abs(moved1.sum().item() - n / 2) < 5 * math.sqrt(n) / 2  # Uniform, within 5 sigma.
  # The robot that stayed has zero velocity, the other completes its move and keeps it.
  v = dq / pair.cfg.timestep_s
  assert (robot.qd[moved1, 3] == 0).all() and (robot.qd[moved4, 0] == 0).all()
  assert (robot.qd[moved1, 0, 0] + v).abs().max() < 1e-5
  assert (robot.qd[moved4, 3, 0] - v).abs().max() < 1e-5
  # Whole-robot state: rolled-back q is exactly the old q.
  assert torch.equal(robot.q[moved4, 0], start[moved4, 0])
  assert torch.equal(robot.q[moved1, 3], start[moved1, 3])


def test_rollback_both_moving_deterministic_takes_lexicographic_first(pair):
  a, b, dq = pair.both_new_case()
  cfg = pair.env_cfg(rollback='deterministic')
  out, start = _one_step_outcomes(cfg, pair.scene, a, b, {(0, 0): -dq, (3, 0): dq}, n=8)
  robot = out.next_robot
  assert (out.collision_robot_steps == 1).all()
  assert torch.equal(robot.q[:, 0], start[:, 0]) and (robot.qd[:, 0] == 0).all()
  assert (robot.q[:, 3, 0] - start[:, 3, 0]).abs().min() > 1e-5


def test_rollback_innocent_robot_50_50(pair):
  a3, b_free, dq = pair.innocent_case()
  n = 4000
  # A tiny motion of joint 7 gives the innocent a velocity that can visibly be zeroed.
  moves = {(3, 0): dq, (0, 6): 1e-6}
  out, start = _one_step_outcomes(pair.cfg, pair.scene, a3, b_free, moves, n=n)
  robot = out.next_robot
  count = out.collision_robot_steps
  assert ((count == 1) | (count == 2)).all()
  both, only4 = count == 2, count == 1
  assert abs(both.sum().item() - n / 2) < 5 * math.sqrt(n) / 2
  assert (out.reward[only4] + 0.03125).abs().max() < 1e-6
  assert (out.reward[both] + 0.0625).abs().max() < 1e-6
  # panda4 is always reset. The innocent loses its velocity only when it was picked first.
  assert torch.equal(robot.q[:, 3], start[:, 3]) and (robot.qd[:, 3] == 0).all()
  assert (robot.qd[both, 0] == 0).all()
  assert (robot.qd[only4, 0, 6] > 0).all()
  assert torch.equal(robot.q[both, 0], start[both, 0])


def test_rollback_innocent_robot_deterministic_always_both(pair):
  a3, b_free, dq = pair.innocent_case()
  cfg = pair.env_cfg(rollback='deterministic')
  out, start = _one_step_outcomes(cfg, pair.scene, a3, b_free, {(3, 0): dq}, n=8)
  assert (out.collision_robot_steps == 2).all()
  assert (out.reward + 0.0625).abs().max() < 1e-6
  assert torch.equal(out.next_robot.q, start)


def _rollback_cfg(rollback, arm):
  return env_cfg(num_obstacles=1, arm=arm, nconmax=128, rollback=rollback)


@pytest.mark.parametrize('rollback', ['random', 'deterministic'])
@pytest.mark.parametrize('arm', ARMS)
def test_rollback_self_hit_only_that_robot(arm, rollback):
  cfg = _rollback_cfg(rollback, arm)
  scene = default_scene(cfg.world, 1)
  # Walk from home towards the self-collision pin of mujoco_simulator_test.py:82-86.
  pin = torch.tensor([0.0, -0.7854, -0.156768, -3.059768, 0.0, 1.3962, 0.0], device=DEVICE)
  n = 1001
  world = build_world(cfg.world, n, DEVICE)
  world.set_scene(torch.arange(n, device=DEVICE), Scene(*[x.expand(n, *x.shape[1:]) for x in scene]))
  s = torch.linspace(0, 1, n, device=DEVICE)
  q = HOME.repeat(n, 4, 1)
  q[:, 0] = HOME + s[:, None] * (pin - HOME)
  world.set_q(q)
  world.forward()
  flags = world.robots_in_collision()
  assert flags[-1, 0] and not flags[:, 1:].any()
  first = flags[:, 0].int().argmax().item()
  assert first > 10
  q_old, q_new = q[first - 5:first - 4], q[first + 5, 0]
  env = _env(cfg, scene._replace(q0=q_old), n=64)
  action = torch.zeros(64, 4, 7, device=DEVICE)
  action[:, 0] = (q_new - q_old[0, 0]) / cfg.timestep_s / cfg.max_joint_velocity
  out = env.step(action)
  assert (out.collision_robot_steps == 1).all()
  assert torch.equal(out.next_robot.q[:, 0], env.pool.q0[0, 0].expand(64, 7))
  assert (out.next_robot.qd == 0).all()
  assert (out.reward + 0.03125).abs().max() < 1e-6


def _box_on_new_tip(cfg, scene, q_old, q_new, robot):
  """A small box where the tool of `robot` ends up: free at q_old, in contact at q_new."""
  world = build_world(cfg.world, 1, DEVICE)
  zero = torch.zeros(1, dtype=torch.long, device=DEVICE)
  world.set_scene(zero, scene)
  world.set_q(q_new)
  world.forward()
  pos, rot = world.tcp()
  # Just behind the TCP, along the tool, is inside the last stick sphere and the tool tip.
  box = pos[0, robot] - 0.02 * rot[0, robot, :, 2]
  scene = scene._replace(box_pos=box[None, None], box_spans=torch.full((1, 1, 3), 0.01, device=DEVICE))
  world.set_scene(zero, scene)
  world.forward()
  expected = [r == robot for r in range(4)]
  assert world.robots_in_collision()[0].cpu().tolist() == expected
  world.set_q(q_old)
  world.forward()
  assert not world.robots_in_collision().any()
  return scene


@pytest.mark.parametrize('rollback', ['random', 'deterministic'])
@pytest.mark.parametrize('arm', ARMS)
def test_rollback_obstacle_hit_only_that_robot_and_others_finish(arm, rollback):
  cfg = _rollback_cfg(rollback, arm)
  scene = default_scene(cfg.world, 1)
  q_new = scene.q0.clone()
  q_new[0, 1, :2] += 0.08  # panda2, the largest move allowed from rest.
  scene = _box_on_new_tip(cfg, scene, scene.q0, q_new, robot=1)
  env = _env(cfg, scene, n=64)
  out = env.step(_vel(cfg, 64, {(1, 0): 0.8, (1, 1): 0.8, (0, 0): 0.1, (2, 0): 0.1, (3, 0): 0.1}))
  assert (out.collision_robot_steps == 1).all()
  assert (out.reward + 0.03125).abs().max() < 1e-6
  robot = out.next_robot
  assert torch.allclose(robot.q[:, 1], HOME, atol=1e-7) and (robot.qd[:, 1] == 0).all()
  for r in (0, 2, 3):
    assert (robot.qd[:, r, 0] - 0.1).abs().max() < 1e-6


def test_rollback_into_a_pose_another_robot_just_took():
  """panda1 hits a box and goes back, where panda4 has just moved: panda4 goes back as well."""
  pair = _Pair('spheres_stick')
  a3, b_free, dq = pair.innocent_case()  # (a3, b_free + dq) touches, (a3, b_free) does not.
  away = 0.08  # The largest move from rest. Away from panda4, so (new, new) is free of it.
  assert not pair.collide(a3 + away, b_free + dq).any()
  q_old = HOME.repeat(1, 4, 1)
  q_old[0, 0, 0], q_old[0, 3, 0] = a3, b_free
  q_new = q_old.clone()
  q_new[0, 0, 0], q_new[0, 3, 0] = a3 + away, b_free + dq
  scene = _box_on_new_tip(pair.cfg, pair.scene, q_old, q_new, robot=0)
  for rollback in ('random', 'deterministic'):
    cfg = pair.env_cfg(rollback=rollback)
    out, start = _one_step_outcomes(cfg, scene, a3, b_free, {(0, 0): away, (3, 0): dq}, n=64)
    assert (out.collision_robot_steps == 2).all()
    assert (out.reward + 0.0625).abs().max() < 1e-6
    assert torch.equal(out.next_robot.q, start)
    assert (out.next_robot.qd == 0).all()


def test_collision_never_terminates():
  cfg = _cfg(timestep_s=0.5, num_targets=3)
  env = _env(cfg, default_scene(cfg.world, 1, 3))
  for _ in range(9):
    assert not env.step(_vel(cfg, 1, {(0, 1): 0.6})).terminal.item()


# Targets and dwell, env_spec 9 and 15.


def _targets_on_home_tips(cfg, picks, world_scene=None):
  """A scene whose targets sit exactly on the home TCP of the robots in `picks`."""
  scene = default_scene(cfg.world, 1, len(picks)) if world_scene is None else world_scene
  pos, rot = _tip_at(cfg, scene)
  index = torch.as_tensor(picks, device=DEVICE)
  return scene._replace(target_pos=pos[:, index], target_rot=rot[:, index])


def test_thresholds_are_strict_with_full_so3():
  cfg = _cfg(num_targets=8)
  scene = default_scene(cfg.world, 1, 8)
  pos, rot = _tip_at(cfg, scene)
  # Position offsets just inside and outside of 2.5 cm, angles around 15 degrees about tool z.
  tp = pos[:, [0, 0, 0, 0]]
  tp = torch.cat([tp + torch.tensor([[[0.0249, 0, 0]] * 4], device=DEVICE),
                  tp + torch.tensor([[[0.0251, 0, 0]] * 4], device=DEVICE)], 1)
  tr = rot[:, [0] * 8].clone()
  env = _env(cfg, scene._replace(target_pos=tp, target_rot=tr))
  env.step(torch.zeros(1, 4, 7, device=DEVICE))
  done = env.observe().robot.target_done[0].cpu().tolist()
  assert done == [True] + [False] * 7  # Inside: first one, and panda1 dwells for the rest.

  def spun(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return rot[0, 0] @ torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=DEVICE)

  cases = [(14.9, True), (15.1, False)]
  for deg, expected in cases:
    tr = spun(deg).expand(1, 8, 3, 3)
    env = _env(cfg, scene._replace(target_pos=pos[:, [0] * 8], target_rot=tr))
    env.step(torch.zeros(1, 4, 7, device=DEVICE))
    assert env.observe().robot.target_done[0, 0].item() == expected, deg


def test_one_solve_per_robot_per_step_lowest_indices_win():
  cfg = _cfg(num_targets=4)
  # Targets 0, 1: panda1. Target 2: panda2. Target 3: panda1 and panda2 both near it.
  scene = _targets_on_home_tips(cfg, [0, 0, 1, 0])
  env = _env(cfg, scene)
  out = env.step(torch.zeros(1, 4, 7, device=DEVICE))
  robot = out.next_robot
  assert robot.target_done[0].cpu().tolist() == [True, False, True, False]
  assert robot.dwell[0].cpu().tolist() == [5, 5, 0, 0]
  assert out.reward.item() == pytest.approx(0.5, abs=1e-6)


def test_lowest_index_robot_takes_a_shared_target():
  cfg = _cfg(num_targets=2)
  env = _env(cfg, _targets_on_home_tips(cfg, [0, 3]))
  robot = env.step(torch.zeros(1, 4, 7, device=DEVICE)).next_robot
  assert robot.dwell[0].cpu().tolist() == [5, 0, 0, 5]
  # The same target within reach of two robots: only the first robot in index order pays.
  within = torch.tensor([[[False, True, True, False]]], device=DEVICE)
  done = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)
  dwell = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
  done2, dwell2 = solve_targets(within, done, dwell, 5)
  assert done2.item() and dwell2[0].cpu().tolist() == [0, 5, 0, 0]
  dwell = torch.tensor([[0, 2, 0, 0]], device=DEVICE)
  done2, dwell2 = solve_targets(within, done, dwell, 5)
  assert done2.item() and dwell2[0].cpu().tolist() == [0, 2, 5, 0]
  assert not done.item() and dwell[0, 2] == 0  # Inputs are left alone.


def test_done_is_sticky_and_not_evaluated_at_reset():
  cfg = _cfg(num_targets=2)
  scene = _targets_on_home_tips(cfg, [0, 1])
  env = _env(cfg, scene)
  # A reset on top of a target does not solve it: solving happens at step ends only.
  assert not env.observe().robot.target_done.any()
  assert env.observe().robot.score.item() == 0
  out = env.step(torch.zeros(1, 4, 7, device=DEVICE))
  assert out.next_robot.target_done.all() and out.next_robot.step.item() == 1


def test_sticky_after_arm_moves_away():
  cfg = _cfg(num_targets=2)
  scene = _targets_on_home_tips(cfg, [0, 0])
  scene = scene._replace(target_pos=torch.cat(
    [scene.target_pos[:, :1], torch.full((1, 1, 3), 9.0, device=DEVICE)], 1))
  env = _env(cfg, scene)
  out = env.step(torch.zeros(1, 4, 7, device=DEVICE))
  assert out.next_robot.target_done[0].cpu().tolist() == [True, False]
  tip = out.next_robot.tip_pos[0, 0].clone()
  action = _vel(cfg, 1, {(0, 0): 1.0})
  for _ in range(15):
    out = env.step(action)
    assert out.next_robot.target_done[0].cpu().tolist() == [True, False]
    assert out.next_robot.score.item() == pytest.approx(0.5)
  assert (out.next_robot.tip_pos[0, 0] - tip).norm() > 0.1


def test_second_target_on_same_pose_solves_at_t_plus_5():
  cfg = _cfg(num_targets=2)
  env = _env(cfg, _targets_on_home_tips(cfg, [0, 0]))
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  done = [env.step(zero).next_robot.target_done[0].sum().item() for _ in range(6)]
  assert done == [1, 1, 1, 1, 1, 2]  # Minimum interval between solves of one arm: 5 steps.


@pytest.mark.parametrize('arm', ARMS)
def test_dwell_two_steps(arm):
  # planning_env_test.py:464-553: observed 1.0, 0.5, 0.0, arm frozen, moves after.
  cfg = _cfg(arm, timestep_s=0.5, dwell_time_s=1.0, num_targets=2)
  assert cfg.dwell_steps == 2
  scene = _targets_on_home_tips(cfg, [1, 1])
  # Only one target to solve: the second goes far away.
  scene = scene._replace(target_pos=torch.cat(
    [scene.target_pos[:, :1], torch.full((1, 1, 3), 9.0, device=DEVICE)], 1))
  env = _env(cfg, scene)
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  robot = env.step(zero).next_robot
  assert robot.target_done[0, 0].item()
  assert (robot.dwell.float() / cfg.dwell_steps)[0].cpu().tolist() == [0.0, 1.0, 0.0, 0.0]
  move = _vel(cfg, 1, {(r, j): 0.01 for r in range(4) for j in range(7)})
  for expected in (0.5, 0.0):
    before = robot.q.clone()
    robot = env.step(move).next_robot
    assert (robot.dwell.float() / cfg.dwell_steps)[0].cpu().tolist() == [0.0, expected, 0.0, 0.0]
    assert torch.equal(robot.q[:, 1], before[:, 1])
    assert (robot.q[:, [0, 2, 3]] - before[:, [0, 2, 3]]).abs().min() > 1e-3
  before = robot.q.clone()
  robot = env.step(move).next_robot
  assert (robot.q - before).abs().min() > 1e-3


@pytest.mark.parametrize('arm', ARMS)
def test_coasting_during_dwell_and_second_solve_at_t_plus_5(arm):
  # env_spec 9 and behaviour_probes P2. q = 0.64, 0.76, 0.80, 0.80 and v = 2.0, 1.2, 0.4, 0.
  cfg = _cfg(arm, num_targets=3)
  base = default_scene(cfg.world, 1, 3)
  # Where the arm will be: solving pose after 4 steps, final pose after the coast.
  at = lambda angle: _tip_at(cfg, base._replace(q0=_with_joint(angle)))
  solve_pos, solve_rot = at(0.64)
  end_pos, end_rot = at(0.80)
  scene = base._replace(
    target_pos=torch.stack([solve_pos[:, 0], end_pos[:, 0],
                            torch.full((1, 3), 9.0, device=DEVICE)], 1),
    target_rot=torch.stack([solve_rot[:, 0], end_rot[:, 0], end_rot[:, 0]], 1))
  env = _env(cfg, scene)
  action = _vel(cfg, 1, {(0, 0): 2.0})
  trace = [env.step(action) for _ in range(4)]
  robot = trace[-1].next_robot
  assert robot.q[0, 0, 0].item() == pytest.approx(0.64, abs=1e-5)
  assert robot.target_done[0].cpu().tolist() == [True, False, False]  # Solved at full speed.
  assert robot.dwell[0, 0].item() == 5
  q, v, dwell, second = [], [], [], []
  for _ in range(5):
    robot = env.step(action).next_robot
    q.append(robot.q[0, 0, 0].item())
    v.append(robot.qd[0, 0, 0].item())
    dwell.append(robot.dwell[0, 0].item())
    second.append(robot.target_done[0, 1].item())
  assert q == pytest.approx([0.76, 0.80, 0.80, 0.80, 0.80], abs=1e-5)
  assert v == pytest.approx([1.2, 0.4, 0.0, 0.0, 0.0], abs=1e-5)
  # The arm reaches the second pose at t+2 while dwelling and solves it only at t+5, when
  # scoring already sees dwell 0, which then starts a new dwell. env_spec 9.
  assert second == [False, False, False, False, True]
  assert dwell == [4, 3, 2, 1, 5]


def _with_joint(angle, robot=0, joint=0):
  q = HOME.repeat(1, 4, 1)
  q[0, robot, joint] = angle
  return q


def test_free_motion_resumes_at_t_plus_6():
  cfg = _cfg(num_targets=2)
  scene = _targets_on_home_tips(cfg, [0, 0])
  scene = scene._replace(target_pos=torch.cat(
    [scene.target_pos[:, :1], torch.full((1, 1, 3), 9.0, device=DEVICE)], 1))
  env = _env(cfg, scene)
  action = _vel(cfg, 1, {(0, 0): 0.2})
  speeds = [env.step(action).next_robot.qd[0, 0, 0].item() for _ in range(7)]
  # Solved at step 1 with v = 0.2, then 0 through t+5, then moving at t+6.
  assert speeds == pytest.approx([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], abs=1e-6)


def test_terminal_after_last_dwell_and_terminal_state_is_returned():
  cfg = _cfg(num_targets=2)
  scene = _targets_on_home_tips(cfg, [0, 1])
  env = _env(cfg, scene)
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  out = env.step(zero)
  assert out.next_robot.target_done.all() and not out.terminal.item()
  assert out.reward.item() == pytest.approx(1.0)
  flags = [env.step(zero) for _ in range(5)]
  assert [f.terminal.item() for f in flags] == [False] * 4 + [True]
  last = flags[-1]
  # StepOut is the terminal state, observe() the first of the new episode.
  assert last.next_robot.step.item() == 6 and last.next_robot.score.item() == pytest.approx(1.0)
  assert last.next_robot.target_done.all()
  obs = env.observe().robot
  assert obs.step.item() == 0 and obs.score.item() == 0 and not obs.target_done.any()
  assert (obs.dwell == 0).all() and (obs.qd == 0).all()


@pytest.mark.parametrize('arm', ARMS)
def test_reward_sum_is_final_score_with_no_penalty(arm):
  cfg = _cfg(arm, num_targets=4, collision_penalty_scaling=0.0)
  scene = _targets_on_home_tips(cfg, [0, 1, 2, 3])
  env = _env(cfg, scene)
  rewards = []
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  while True:
    out = env.step(zero)
    rewards.append(out.reward.item())
    if out.terminal.item():
      break
  assert rewards[0] == pytest.approx(out.next_robot.score.item())  # First reward is S_1.
  assert sum(rewards) == pytest.approx(out.next_robot.score.item())


def test_acceleration_metric():
  # planning_env_test.py:250-279 total_acceleration. The cost itself has no EnvCfg field
  # because it is zero in the original and not overridden (env_spec 7.2).
  cfg = _cfg(num_targets=2)
  env = _env(cfg, default_scene(cfg.world, 1, 2))
  first = env.step(_vel(cfg, 1, {(0, 1): 1.0, (0, 2): 1.0})).acceleration.item()
  second = env.step(_vel(cfg, 1, {(0, 1): 1.2, (0, 2): 1.2})).acceleration.item()
  assert first == pytest.approx(0.05714286, abs=1e-6)
  assert second == pytest.approx(0.02857143, abs=1e-6)


# Return to start, planning_env_test.py:598-675.


@pytest.mark.parametrize('arm', ARMS)
def test_return_to_start_score(arm):
  cfg = _cfg(arm, num_targets=2, timestep_s=0.5, dwell_time_s=0.0, return_to_start_reward_weight=0.4,
             return_to_start_threshold=0.1, return_to_start_required=True)
  # Arms 3 and 4 away from home by 0.15: the shaped branch.
  q0 = HOME.repeat(1, 4, 1)
  q0[0, 2] += 0.15
  q0[0, 3] -= 0.15
  q0 = torch.minimum(q0, torch.as_tensor(JOINT_UPPER, device=DEVICE))
  scene = _targets_on_home_tips(cfg, [0, 1])
  env = _env(cfg, scene._replace(q0=q0))
  zero = torch.zeros(1, 4, 7, device=DEVICE)
  out = env.step(zero)
  assert out.next_robot.target_done.all() and not out.terminal.item()
  shaped = 1.0 - 3.0 * 0.15**2
  rts = (1.0 + 1.0 + shaped + shaped) / 4
  assert out.next_robot.score.item() == pytest.approx(0.6 + 0.4 * rts, abs=1e-6)
  # Still outside of the takeover threshold: the policy's zero action keeps the pose.
  out = env.step(zero)
  assert torch.equal(out.next_robot.q, q0) and not out.terminal.item()


@pytest.mark.parametrize('arm', ARMS)
def test_return_to_start_partial_score_before_all_done(arm):
  cfg = _cfg(arm, num_targets=2, return_to_start_reward_weight=0.4, return_to_start_threshold=0.1,
             return_to_start_required=True, dwell_time_s=0.0)
  scene = _targets_on_home_tips(cfg, [0, 0])
  scene = scene._replace(target_pos=torch.cat(
    [scene.target_pos[:, :1], torch.full((1, 1, 3), 9.0, device=DEVICE)], 1))
  env = _env(cfg, scene)
  out = env.step(torch.zeros(1, 4, 7, device=DEVICE))
  assert out.next_robot.score.item() == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize('arm', ARMS)
def test_takeover_parks_in_one_step_and_overrides_the_action(arm):
  cfg = _cfg(arm, num_targets=2, timestep_s=0.5, dwell_time_s=0.0, return_to_start_reward_weight=0.4,
             return_to_start_threshold=0.1, return_to_start_required=True)
  q0 = HOME.repeat(1, 4, 1)
  q0[0, 2] += 0.07
  q0[0, 3] -= 0.07
  q0 = torch.minimum(q0, torch.as_tensor(JOINT_UPPER, device=DEVICE))
  scene = _targets_on_home_tips(cfg, [0, 1])
  env = _env(cfg, scene._replace(q0=q0))
  out = env.step(torch.zeros(1, 4, 7, device=DEVICE))
  # Score 1.0 within the threshold, but not terminal: the arms are not parked yet.
  assert out.next_robot.score.item() == pytest.approx(1.0, abs=1e-6)
  assert not out.terminal.item()
  out = env.step(torch.ones(1, 4, 7, device=DEVICE))  # Overridden.
  assert out.terminal.item()
  assert (out.next_robot.q[0] - HOME).abs().max() < 1e-3


@pytest.mark.parametrize('arm', ARMS)
def test_takeover_is_active_without_return_to_start_required(arm):
  # Reference run settings: the takeover in planning_env.py:800-816 has no such condition.
  cfg = _cfg(arm, num_targets=1, dwell_time_s=0.3)
  scene = _targets_on_home_tips(cfg, [0])
  q0 = HOME.repeat(1, 4, 1)
  q0[0, 2, 0] += 0.02
  env = _env(cfg, scene._replace(q0=q0))
  env.step(torch.zeros(1, 4, 7, device=DEVICE))
  out = env.step(torch.ones(1, 4, 7, device=DEVICE))  # All robots within 0.25 of home.
  assert (out.next_robot.q[0] - HOME).abs().max() < 1e-5


# Observation, augmentation, reset.


def test_box_augmentation_same_vertices():
  spans = torch.tensor([2.0, 4.0, 6.0], device=DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(3)
  a = torch.randn(4, generator=gen, device=DEVICE)
  w, x, y, z = (a / a.norm()).tolist()
  rot = torch.tensor([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], device=DEVICE)
  codes = torch.arange(4, device=DEVICE)
  new_rot, new_spans = augment_boxes(rot.expand(4, 3, 3), spans.expand(4, 3), codes)
  assert new_spans.cpu().tolist() == [[2, 4, 6], [2, 6, 4], [6, 4, 2], [4, 2, 6]]
  signs = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
                       device=DEVICE)
  def vertices(i):
    points = (new_rot[i] @ (signs * new_spans[i] / 2)[..., None]).squeeze(-1).cpu().numpy()
    return points[np.lexsort(points.T)]
  for i in range(1, 4):
    assert np.allclose(vertices(0), vertices(i), atol=1e-5)
  assert torch.equal(new_rot[0], rot)


def test_augmentation_is_per_observation_and_shared_by_s_prime():
  cfg = env_cfg(num_targets=2)
  scene = default_scene(cfg.world, 1, 2)
  env = _env(cfg, scene, n=16)
  first = env.observe().robot.aug_code
  assert first.shape == (16, 30) and first.dtype == torch.long
  assert {int(x) for x in first.flatten().tolist()} == {0, 1, 2, 3}
  out = env.step(torch.zeros(16, 4, 7, device=DEVICE))
  assert torch.equal(out.next_robot.aug_code, env.observe().robot.aug_code)
  assert not torch.equal(out.next_robot.aug_code, first)
  off = env_cfg(num_targets=2, augment_obstacles=False)
  env = _env(off, scene, n=4)
  assert (env.observe().robot.aug_code == 0).all()


def test_reset_draws_uniformly_with_replacement():
  pool = stored_scenes(4)
  size = 8
  pool = Scene(*[x[:size] for x in pool])
  tag = pool.target_pos.clone()  # Tags each scene so that observe() reveals the draw.
  tag[:, 0, 0] = 100.0 + torch.arange(size, device=DEVICE)
  pool = pool._replace(target_pos=tag)
  n = 4096
  env = BalletEnv(EnvCfg(), n, pool, DEVICE, seed=5)
  drawn = (env.observe().scene.target_pos[:, 0, 0] - 100.0).long()
  counts = torch.bincount(drawn, minlength=size).float()
  assert counts.sum() == n and (counts > 0).all()
  expected = n / size
  chi2 = ((counts - expected) ** 2 / expected).sum().item()
  assert chi2 < 24.3  # 0.999 quantile for 7 degrees of freedom.
  # Partial reset leaves other worlds alone and draws fresh targets for the chosen ones.
  before = env.observe().scene
  ids = torch.tensor([0, 7, 11], device=DEVICE)
  mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
  mask[ids] = False
  env.reset(ids)
  after = env.observe().scene
  assert torch.equal(after.target_pos[mask], before.target_pos[mask])
  assert torch.equal(after.box_pos[mask], before.box_pos[mask])


def test_scene_held_by_caller_is_not_modified_by_auto_reset():
  cfg = EnvCfg(timestep_s=0.5, num_targets=10)
  pool = stored_scenes(4)
  env = BalletEnv(cfg, 8, pool, DEVICE, seed=2)
  held = env.observe()
  copies = [x.clone() for x in held.scene] + [x.clone() for x in held.robot]
  zero = torch.zeros(8, 4, 7, device=DEVICE)
  while not env.step(zero).terminal.all():
    pass
  for x, y in zip(list(held.scene) + list(held.robot), copies):
    assert torch.equal(x, y)


def test_reset_redraws_scenes_colliding_at_home():
  good, bad = stored_scenes(4, valid=True), stored_scenes(4, valid=False)
  assert len(bad.q0) > 16
  pool = Scene(*[torch.cat([a[:16], b[:16]]) for a, b in zip(good, bad)])
  n = 256
  env = BalletEnv(EnvCfg(), n, pool, DEVICE, seed=3)
  assert env.invalid_scene_draws > 0
  env.world.forward()
  assert not env.world.robots_in_collision().any()
  # Every remaining world holds a valid scene, its world arrays included.
  scene_q = env.observe().robot.q
  assert torch.equal(env.world.sim.data.qpos[:, :28], scene_q.flatten(1))
  boxes = env.world.sim.data.mocap_pos[:, env.world._box_mocap]
  assert torch.equal(boxes, env.observe().scene.box_pos)


def test_reset_gives_up_on_a_pool_of_bad_scenes():
  bad = stored_scenes(4, valid=False)
  with pytest.raises(RuntimeError, match='bad scene pool'):
    BalletEnv(EnvCfg(), 4, Scene(*[x[:2] for x in bad]), DEVICE, seed=1)


def test_reset_with_chosen_scene_raises_on_collision():
  good, bad = stored_scenes(4, valid=True), stored_scenes(4, valid=False)
  pool = Scene(*[torch.cat([a[:1], b[:1]]) for a, b in zip(good, bad)])
  env = BalletEnv(EnvCfg(), 2, Scene(*[x[:1] for x in good]), DEVICE, seed=4)
  env.pool = pool  # The redraw at construction would have hidden the bad scene.
  with pytest.raises(ValueError, match='collide at their start pose'):
    env.reset(scene_ids=torch.tensor([0, 1], device=DEVICE))


def test_pool_shape_is_checked():
  with pytest.raises(ValueError, match='scene pool'):
    BalletEnv(env_cfg(num_targets=40), 2, stored_scenes(4), DEVICE, seed=1)


def test_same_seed_same_rollout():
  pool = stored_scenes(4)
  runs = []
  for _ in range(2):
    env = BalletEnv(EnvCfg(), 32, pool, DEVICE, seed=11)
    gen = torch.Generator(device=DEVICE).manual_seed(12)
    last = None
    for _ in range(40):
      action = torch.rand(32, 4, 7, generator=gen, device=DEVICE) * 2 - 1
      last = env.step(action)
    runs.append((last.next_robot.q.clone(), last.reward.clone(), last.next_robot.aug_code.clone(),
                 env.observe().scene.box_pos.clone()))
  for a, b in zip(*runs):
    assert torch.equal(a, b)
