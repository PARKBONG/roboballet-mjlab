"""Gate G2: E4, E5 and a full rollout against an independent C MuJoCo reference loop."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, EnvCfg, WorldCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.structs import Scene
from tests.test_sim_common import DEVICE, CWorld, stored_scenes

E4_ENV_STEPS = 10_000_000


def _saturated(n, num_robots, gen):
  return (torch.randint(2, (n, num_robots, 7), generator=gen, device=DEVICE) * 2 - 1).float()


# E4: no residual collision, no NaN, no start-state collision.


@pytest.mark.parametrize('num_robots', [4, 8])
def test_e4_no_residual_collision(num_robots):
  W = 4096
  cfg = EnvCfg(world=WorldCfg(num_robots=num_robots))
  env = BalletEnv(cfg, W, stored_scenes(num_robots), DEVICE, seed=21)
  gen = torch.Generator(device=DEVICE).manual_seed(22)
  steps = math.ceil(E4_ENV_STEPS / W)
  terminals = rollbacks = 0
  for i in range(steps):
    out = env.step(_saturated(W, num_robots, gen))
    # After any reset, so this also covers every start state of the new episodes.
    assert not env.world.robots_in_collision().any(), f'step {i}'
    terminals += out.terminal.sum()
    rollbacks += out.collision_robot_steps.sum()
    for x in (out.next_robot.q, out.next_robot.qd, out.next_robot.tip_pos, out.reward):
      assert torch.isfinite(x).all(), f'step {i}'
  total = W * steps
  print(f'E4 {num_robots}r: {total} env steps, residual {env.residual_collisions.item()}, '
        f'start-state collisions {env.invalid_scene_draws.item()}, episodes {int(terminals)}, '
        f'rollbacks per env step {int(rollbacks) / total:.3f}, '
        f'launches per step {env.world.launches / steps:.2f}')
  assert total >= E4_ENV_STEPS
  assert env.residual_collisions == 0
  assert env.invalid_scene_draws == 0  # The test pool is all valid.
  assert rollbacks > 0


# E5: reward identity and exact penalty per rolled-back robot.


def _trace_targets(env, cfg, pool, gen_seed, action_fn, steps, targets_per_world):
  """Target poses taken from where the arms really go, so that many targets get solved."""
  gen = torch.Generator(device=DEVICE).manual_seed(gen_seed)
  n = env.num_worlds
  tips_pos, tips_rot = [], []
  env.reset(scene_ids=torch.arange(n, device=DEVICE) % len(pool.q0))
  for _ in range(steps):
    out = env.step(action_fn(gen))
    tips_pos.append(out.next_robot.tip_pos)
    tips_rot.append(out.next_robot.tip_rot)
  tips_pos, tips_rot = torch.stack(tips_pos, 1), torch.stack(tips_rot, 1)
  w = torch.arange(n, device=DEVICE)[:, None]
  t = torch.randint(steps, (n, targets_per_world), generator=gen, device=DEVICE)
  r = torch.randint(cfg.world.num_robots, (n, targets_per_world), generator=gen, device=DEVICE)
  return tips_pos[w, t, r], tips_rot[w, t, r]


@pytest.mark.parametrize('num_robots,num_targets,penalty', [
  (4, 10, 0.03125), (4, 40, 0.00893), (8, 40, 0.00446),
], ids=['4r10t', '4r40t', '8r40t'])
def test_e5_reward_identity(num_robots, num_targets, penalty):
  W = 256
  cfg = EnvCfg(world=WorldCfg(num_robots=num_robots), num_targets=num_targets)
  stored = stored_scenes(num_robots)
  repeat = num_targets // 10
  pool = Scene(*[x[:W] if len(x) >= W else x for x in stored])
  pool = pool._replace(target_pos=pool.target_pos.repeat(1, repeat, 1),
                       target_rot=pool.target_rot.repeat(1, repeat, 1, 1))
  size = len(pool.q0)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=31)
  action_fn = lambda gen: torch.rand(W, num_robots, 7, generator=gen, device=DEVICE) * 2 - 1
  tp, tr = _trace_targets(env, cfg, pool, 32, action_fn, min(cfg.max_steps, 200), num_targets)
  # One scene per world from here on, targets from the trace of that world.
  index = torch.arange(W, device=DEVICE) % size
  pool = Scene(*[x[index] for x in pool])._replace(target_pos=tp, target_rot=tr)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=31)
  env.reset(scene_ids=torch.arange(W, device=DEVICE))

  per_robot = 15.0 / (cfg.max_steps * num_robots)
  assert per_robot == pytest.approx(penalty, abs=5e-6)
  gen = torch.Generator(device=DEVICE).manual_seed(32)
  ret = torch.zeros(W, dtype=torch.float64, device=DEVICE)
  pen = torch.zeros(W, dtype=torch.float64, device=DEVICE)
  finished = solved = 0
  worst = 0.0
  for _ in range(int(cfg.max_steps * 1.5)):
    out = env.step(action_fn(gen))
    expected = per_robot * out.collision_robot_steps.double()
    assert (out.collision_penalty.double() - expected).abs().max() < 1e-6
    ret += out.reward.double()
    pen += out.collision_penalty.double()
    ended = out.terminal
    if ended.any():
      final = out.next_robot.score[ended].double()
      worst = max(worst, (ret[ended] - (final - pen[ended])).abs().max().item())
      finished += ended.sum().item()
      solved += out.next_robot.target_done[ended].sum().item()
      ret[ended] = 0
      pen[ended] = 0
  print(f'E5 {num_robots}r{num_targets}t: {finished} episodes, {solved} targets solved, '
        f'worst identity error {worst:.2e}, penalty per robot-step {per_robot:.6f}')
  assert finished >= W
  assert solved > finished  # The score really moved.
  assert worst <= 1e-5


# Full rollout against C MuJoCo.


class ReferenceEnv:
  """The env of base_env.py and planning_env.py, one world, in NumPy over mj_collision.

  Nothing here comes from BalletEnv: only the numbers in EnvCfg are shared. q and qd
  are float32 as in the original, poses are C MuJoCo's doubles.
  """

  def __init__(self, cfg: EnvCfg, world: CWorld, scene: Scene, w: int) -> None:
    self.cfg, self.world = cfg, world
    get = lambda x: x[w].cpu().numpy()
    self.q = get(scene.q0).astype(np.float32)
    self.target_pos = get(scene.target_pos).astype(np.float64)
    self.target_rot = get(scene.target_rot).astype(np.float64)
    world.set_scene(get(scene.base_pos), get(scene.base_rot), get(scene.box_pos),
                    get(scene.box_rot), get(scene.box_spans))
    world.set_q(self.q)
    world.forward()
    assert not world.flags().any()
    self.R, self.T = self.q.shape[0], len(self.target_pos)
    self.qd = np.zeros_like(self.q)
    self.dwell = [0] * self.R
    self.done = [False] * self.T
    self.score = 0.0
    self.step_count = 0
    self.lower = np.array(JOINT_LOWER, dtype=np.float32)
    self.upper = np.array(JOINT_UPPER, dtype=np.float32)
    self.home = np.array(HOME_QPOS, dtype=np.float32)

  def _rollback(self, q_old, q_new, qd):
    w = self.world
    w.set_q(q_new)
    w.forward()
    colliding = w.flags()
    reset = set()
    while colliding.any():
      candidates = [r for r in range(self.R) if colliding[r] and r not in reset]
      r = min(candidates, key=lambda i: w.names[i])  # deterministic_mode, base_env.py:248.
      q_new[r] = q_old[r]
      qd[r] = 0.0
      reset.add(r)
      w.set_q(q_new)
      w.forward()
      colliding = w.flags()
      assert len(reset) <= self.R
    return q_new, qd, len(reset)

  def _solve(self):
    pos, rot = self.world.tcp()
    cos_limit = math.cos(self.cfg.angle_threshold_rad)
    for k in range(self.T):
      if self.done[k]:
        continue
      for r in range(self.R):
        if self.dwell[r] > 0:
          continue
        near = np.linalg.norm(pos[r] - self.target_pos[k]) < self.cfg.position_threshold
        cos_angle = (np.trace(rot[r].T @ self.target_rot[k]) - 1.0) / 2.0
        if near and cos_angle > cos_limit:
          self.done[k] = True
          self.dwell[r] = self.cfg.dwell_steps
          break

  def step(self, action: np.ndarray) -> dict:
    cfg = self.cfg
    requested = action.astype(np.float32) * np.float32(cfg.max_joint_velocity)
    for r in range(self.R):
      if self.dwell[r] > 0:
        requested[r] = 0.0
        self.dwell[r] -= 1
    if all(self.done):
      for r in range(self.R):
        diff = self.home - self.q[r]
        if (np.abs(diff) < cfg.return_to_start_threshold).all():
          requested[r] = diff / cfg.timestep_s
    self.step_count += 1
    dv = cfg.max_joint_acceleration * cfg.timestep_s
    qd = np.clip(requested, self.qd - dv, self.qd + dv).astype(np.float32)
    q_new = np.clip(self.q + qd * cfg.timestep_s, self.lower, self.upper).astype(np.float32)
    self.q, self.qd, rolled = self._rollback(self.q.copy(), q_new, qd)
    self._solve()
    score = sum(self.done) / self.T
    penalty = cfg.collision_penalty_scaling / cfg.max_steps / self.R * rolled
    reward = score - self.score - penalty
    self.score = score
    terminal = self.step_count >= cfg.max_steps or (all(self.done) and not any(self.dwell))
    return dict(q=self.q.copy(), qd=self.qd.copy(), reward=reward, done=list(self.done),
                dwell=list(self.dwell), rolled=rolled, terminal=terminal, tip=self.world.tcp()[0])


def _reference_scenes(cfg: EnvCfg, world: CWorld, num_scenes: int, actions: np.ndarray, seed: int):
  """Stored scenes. Half the targets per world stay stored, half come from a dry run of the
  reference so that they actually get solved. The last 10 worlds hold still with all
  targets on the home TCPs, which runs through ties and early termination."""
  pool = stored_scenes(cfg.world.num_robots)
  scene = Scene(*[x[:num_scenes] for x in pool])
  pos, rot = scene.target_pos.clone(), scene.target_rot.clone()
  rng = np.random.default_rng(seed)
  for w in range(num_scenes):
    ref = ReferenceEnv(cfg, world, scene, w)
    tips, mats = [], []
    for t in range(actions.shape[1]):
      ref.step(actions[w, t])
      p, r = world.tcp()
      tips.append(p)
      mats.append(r)
    for k in range(5, 10):
      t, robot = rng.integers(actions.shape[1]), rng.integers(ref.R)
      pos[w, k] = torch.as_tensor(tips[t][robot], dtype=torch.float32, device=DEVICE)
      rot[w, k] = torch.as_tensor(mats[t][robot], dtype=torch.float32, device=DEVICE)
    if w >= num_scenes - 10:
      ref = ReferenceEnv(cfg, world, scene, w)
      p, r = world.tcp()
      for k in range(10):
        pos[w, k] = torch.as_tensor(p[k % ref.R], dtype=torch.float32, device=DEVICE)
        rot[w, k] = torch.as_tensor(r[k % ref.R], dtype=torch.float32, device=DEVICE)
  return scene._replace(target_pos=pos, target_rot=rot)


@pytest.mark.parametrize('num_robots', [4, 8])
def test_rollout_matches_c_reference(num_robots):
  # 50 worlds run all 120 steps, 10 more are there to end early.
  num_scenes, num_steps = 60, 120
  cfg = EnvCfg(world=WorldCfg(num_robots=num_robots), rollback='deterministic')
  assert cfg.max_steps == num_steps
  rng = np.random.default_rng(41)
  shape = (num_scenes, num_steps, num_robots, 7)
  # Saturated in half of the worlds, uniform in the others, still in the last ten.
  actions = rng.uniform(-1, 1, shape).astype(np.float32)
  actions[::2] = (rng.integers(0, 2, shape)[::2] * 2 - 1).astype(np.float32)
  actions[-10:] = 0.0

  c_world = CWorld(num_robots, 30)
  scene = _reference_scenes(cfg, c_world, num_scenes, actions, 42)

  env = BalletEnv(cfg, num_scenes, scene, DEVICE, seed=43, debug=True)
  env.reset(scene_ids=torch.arange(num_scenes, device=DEVICE))
  records = []
  for t in range(num_steps):
    out = env.step(torch.as_tensor(actions[:, t], device=DEVICE))
    r = out.next_robot
    records.append(dict(
      q=r.q.cpu().numpy(), qd=r.qd.cpu().numpy(), reward=out.reward.cpu().numpy(),
      done=r.target_done.cpu().numpy(), dwell=r.dwell.cpu().numpy(), tip=r.tip_pos.cpu().numpy(),
      terminal=out.terminal.cpu().numpy(), rolled=out.collision_robot_steps.cpu().numpy()))

  worst = dict(q=0.0, qd=0.0, reward=0.0, tip=0.0)
  compared = rollbacks = solves = terminals = 0
  for w in range(num_scenes):
    ref = ReferenceEnv(cfg, c_world, scene, w)
    for t in range(num_steps):
      got = records[t]
      want = ref.step(actions[w, t])
      where = f'world {w} step {t}'
      for key in worst:
        value = np.abs(np.asarray(want[key]) - got[key][w]).max()
        worst[key] = max(worst[key], float(value))
        assert value <= (1e-4 if key == 'tip' else 1e-6 if key == 'reward' else 1e-5), f'{key} {where}'
      assert want['done'] == got['done'][w].tolist(), f'done {where}'
      assert want['dwell'] == got['dwell'][w].tolist(), f'dwell {where}'
      assert want['rolled'] == got['rolled'][w], f'rolled {where}'
      assert want['terminal'] == got['terminal'][w], f'terminal {where}'
      compared += 1
      rollbacks += want['rolled']
      if want['terminal']:
        terminals += 1
        solves += sum(want['done'])
        assert t == num_steps - 1 or w >= num_scenes - 10, f'world {w} ended at step {t}'
        break
    else:
      raise AssertionError(f'world {w} did not terminate')
  print(f'C reference {num_robots}r: {compared} world-steps compared, {rollbacks} rollbacks, '
        f'{solves} targets solved, {terminals} terminals, worst {worst}')
  assert rollbacks > 100
  assert solves > 100
  assert terminals == num_scenes
