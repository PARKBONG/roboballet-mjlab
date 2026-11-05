"""Independent skeptic's tests of BalletEnv: differential fuzz against the original env.

env_adversarial_dump.py runs the original in ../venv and writes traces under S/review, reused
while newer than that script. This file replays them. Targets sit on the arm paths, many right
at the tolerance boundary, and the worlds of a batch have unrelated histories.
"""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, EnvCfg
from roboballet_mjlab.env.ballet_env import BalletEnv, solve_targets, targets_within
from roboballet_mjlab.env.poses import augment_boxes
from roboballet_mjlab.structs import Scene
from tests import env_adversarial_dump as dump
from tests.test_env_semantics import _tip_at
from tests.test_sim_common import DEVICE, PROJECT, default_scene, env_cfg, stored_scenes

REVIEW = PROJECT / 'runs/mjlab_port_analysis/review'
DUMP = Path(dump.__file__)
ORIGINAL_PYTHON = PROJECT / 'venv/bin/python'
HOME = torch.as_tensor(HOME_QPOS, device=DEVICE)
LOWER = torch.as_tensor(JOINT_LOWER, device=DEVICE)
UPPER = torch.as_tensor(JOINT_UPPER, device=DEVICE)
# A solve may flip only this close to the tolerance: tips are f32 here and f64 there.
FLIP_MARGIN = 1e-5
LIMITS = dict(q=1e-6, qd=1e-6, tip=2e-5, rot=2e-5, reward=2e-7, score=2e-7, accel=1e-6, penalty=1e-6)


class Scripted(BalletEnv):
  """Auto-resets take the scene the original took, instead of drawing one."""

  forced = None

  def reset(self, world_ids=None, scene_ids=None):
    if world_ids is not None and scene_ids is None and self.forced is not None:
      scene_ids = self.forced[world_ids]
    super().reset(world_ids, scene_ids)


def _run_original(*args, out: Path) -> subprocess.Popen:
  environment = dict(os.environ, PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                     OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
  return subprocess.Popen([str(ORIGINAL_PYTHON), '-B', str(DUMP), *args, str(out)], cwd=PROJECT,
                          env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _fresh(path: Path) -> bool:
  return path.exists() and path.stat().st_mtime > DUMP.stat().st_mtime


@pytest.fixture(scope='module')
def traces() -> dict[str, Path]:
  """Runs the original for every stale case, in parallel."""
  if not ORIGINAL_PYTHON.exists():
    pytest.skip(f'{ORIGINAL_PYTHON} is missing: nothing was compared')
  REVIEW.mkdir(parents=True, exist_ok=True)
  paths = {case: REVIEW / f'env_adv_{case}.npz' for case in dump.CASES}
  jobs = {case: _run_original(case, '0', out=path.with_suffix('.tmp.npz'))
          for case, path in paths.items() if not _fresh(path)}
  for case, job in jobs.items():
    log, _ = job.communicate()
    assert job.returncode == 0, f'{case}:\n{log[-3000:]}'
    paths[case].with_suffix('.tmp.npz').replace(paths[case])
  return paths


def _tensor(x, dtype=torch.float32):
  return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def _rule_flags(q: torch.Tensor) -> torch.Tensor:
  """rule_flags of the dump script, batched: [W, R] from [W, R, 7]."""
  a, b = q[..., 0], q[..., 2]
  pair = (a[:, :, None] + a[:, None]) > dump.PAIR
  pair = pair & ~torch.eye(q.shape[1], dtype=torch.bool, device=q.device)
  hit = pair | ((b[:, :, None] - b[:, None]) > dump.SPREAD)
  return (q[..., 1] > dump.SOLO) | hit.any(1) | hit.any(2)


def _install_rule(env: BalletEnv) -> None:
  W, R = env.num_worlds, env.cfg.world.num_robots
  env.world.robots_in_collision = lambda: _rule_flags(env.world.sim.data.qpos.reshape(W, R, 7))


def _flip_margin(z, pool: Scene, cfg, w: int, t: int, port_done, previous_done) -> float:
  """How far the original's own numbers are from the tolerance, for the targets in dispute.

  With equal done flags and unequal dwell, a different robot took a target solved this step.
  With nothing solved this step, no rounding can explain the difference.
  """
  scene = int(z['scene'][w, t])
  tip, gold_done = z['tip'][w, t], z['done'][w, t]
  disputed = np.flatnonzero(port_done != gold_done)
  if not len(disputed):
    disputed = np.flatnonzero(gold_done & ~previous_done)
  if not len(disputed):
    return math.inf
  pos = pool.target_pos[scene][disputed].cpu().numpy().astype(np.float64)
  rot = pool.target_rot[scene][disputed].cpu().numpy().astype(np.float64)
  dist = np.linalg.norm(tip[None, :, :3, 3] - pos[:, None], axis=-1)
  trace = (tip[None, :, :3, :3] * rot[:, None]).sum((-1, -2))
  angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
  return float(np.abs(np.maximum(dist - cfg.position_threshold, angle - cfg.angle_threshold_rad)).min())


def _pool(z) -> Scene:
  return Scene(*[_tensor(z[f]) for f in Scene._fields])


def _config(case: str):
  p = dump.case_params(case)
  return env_cfg(
    num_robots=p['R'], num_obstacles=1, arm='mesh_tool', nconmax=p['nconmax'], num_targets=p['T'],
    timestep_s=p['dt'], dwell_time_s=p['dwell'], return_to_start_reward_weight=p['weight'],
    return_to_start_required=p['required'], return_to_start_threshold=p['thr'],
    collisions_enabled=p['rule'], rollback='deterministic')


def _make_env(case: str, z) -> Scripted:
  cfg = _config(case)
  pool = _pool(z)
  # Built without collisions, since real contact must not filter scenes behind the rule.
  env = Scripted(dataclasses.replace(cfg, collisions_enabled=False), z['q'].shape[0], pool, DEVICE, seed=0)
  env.cfg = cfg
  if cfg.collisions_enabled:
    _install_rule(env)
  env.reset(scene_ids=_tensor(z['scene'][:, 0], torch.long))
  return env


# Minimum coverage per case, or the case proved nothing. Keys are in _Coverage.
EXPECT = {
  'reference': dict(solved=100, all_done=5, timeouts=5, limit_hits=50, mixed=1),
  'odd_bases_far_q0': dict(solved=50, limit_hits=50),
  'few_targets_many_resets': dict(solved=100, all_done=30, mixed=1),
  'ties_on_stacked_bases': dict(solved=50, ties=10),
  'no_dwell': dict(solved=100, multi_solve=1),
  'coarse_step': dict(solved=100, all_done=5),
  'fine_step': dict(solved=50, all_done=5),
  'ceil_dwell': dict(solved=100, all_done=5),
  'rts_required': dict(solved=100, parked=5, takeover=20, takeover_dwelling=5),
  'rts_weight_only': dict(solved=100, all_done=5, takeover=1),
  'required_no_weight': dict(solved=100, parked=5, takeover=20),
  'takeover_no_rts': dict(solved=100, all_done=5, takeover=20, takeover_dwelling=20),
  'eight_robots': dict(solved=50, all_done=1),
  'rule': dict(solved=50, rolled=200, cascades=10),
  'rule_eight_robots': dict(solved=50, rolled=200, cascades=10),
  'rule_rts': dict(solved=50, rolled=50, parked=1, takeover=1),
}


@dataclasses.dataclass
class _Coverage:
  solved: int = 0
  all_done: int = 0
  timeouts: int = 0
  parked: int = 0
  limit_hits: int = 0
  mixed: int = 0
  ties: int = 0
  multi_solve: int = 0
  takeover: int = 0
  takeover_dwelling: int = 0
  rolled: int = 0
  cascades: int = 0
  flips: int = 0

  def update(self, cfg, st, out, within) -> None:
    robot, dwell = out.next_robot, out.next_robot.dwell
    new = robot.target_done & ~st.target_done
    self.solved += new.sum().item()
    finished = out.terminal & robot.target_done.all(1) & (dwell == 0).all(1)
    self.all_done += finished.sum().item()
    self.timeouts += (out.terminal & ~finished).sum().item()
    self.parked += (finished & ((robot.q - HOME).abs() < 1e-3).all(-1).all(-1)).sum().item()
    self.limit_hits += ((robot.q == LOWER) | (robot.q == UPPER)).sum().item()
    others = ~out.terminal & (dwell > 0).any(1)
    self.mixed += int(out.terminal.any() and others.any())
    free = within & ~st.target_done[:, :, None] & (st.dwell[:, None] <= 1)
    self.ties += ((free.sum(-1) >= 2) & new).sum().item()
    self.multi_solve += (new.sum(1) > cfg.world.num_robots).sum().item()
    near_home = ((HOME - st.q).abs() < cfg.return_to_start_threshold).all(-1)
    takeover = st.target_done.all(1, keepdim=True) & near_home
    self.takeover += takeover.sum().item()
    self.takeover_dwelling += (takeover & (st.dwell > 0)).sum().item()
    self.rolled += out.collision_robot_steps.sum().item()
    self.cascades += (out.collision_robot_steps >= 2).sum().item()


@pytest.mark.parametrize('case', list(dump.CASES))
def test_differential_fuzz(traces, case):
  z = dict(np.load(traces[case]))
  env = _make_env(case, z)
  cfg, pool = env.cfg, env.pool
  assert cfg.max_steps == int(z['max_steps']) and cfg.dwell_steps == int(z['dwell_steps'])
  W, steps = z['q'].shape[:2]
  alive = torch.ones(W, dtype=torch.bool, device=DEVICE)
  worst = dict(q=0.0, qd=0.0, tip=0.0, rot=0.0, reward=0.0, score=0.0, accel=0.0, penalty=0.0)
  inexact = 0
  coverage = _Coverage()
  kept = []  # (tensor as handed out, its value then): nothing handed out may change later.

  for t in range(steps):
    st = env.observe().robot
    env.forced = _tensor(z['scene'][:, t + 1], torch.long)
    out = env.step(_tensor(z['a'][:, t]))
    robot = out.next_robot
    if t % 10 == 0:
      kept += [(x, x.clone()) for x in (*robot, out.reward, out.terminal)]
    # On the terminal row a world is compared before the auto-reset, as StepOut promises.
    assert torch.equal(robot.step[alive], _tensor(z['step'][:, t], torch.long)[alive]), f'step at {t}'

    done, gold_done = robot.target_done, _tensor(z['done'][:, t], torch.bool)
    gold_dwell = _tensor(z['dwell'][:, t], torch.long)
    bad = alive & (((done != gold_done).any(1)) | ((robot.dwell != gold_dwell).any(1)))
    for w in bad.nonzero().flatten().tolist():
      margin = _flip_margin(z, pool, cfg, w, t, done[w].cpu().numpy(), st.target_done[w].cpu().numpy())
      detail = (f'{case} world {w} step {t} scene {z["scene"][w, t]}: done {done[w].cpu().tolist()} '
                f'vs {z["done"][w, t].tolist()}, dwell {robot.dwell[w].cpu().tolist()} vs '
                f'{z["dwell"][w, t].tolist()}, margin {margin}')
      assert margin < FLIP_MARGIN, detail
      print(f'precision flip: {detail}')
      alive[w] = False  # Rounding decided this solve. The rest of this world's trace is moot.
      coverage.flips += 1

    a = alive
    gold_terminal = _tensor(z['terminal'][:, t], torch.bool)
    assert torch.equal(out.terminal[a], gold_terminal[a]), f'{case} terminal at {t}'
    gold_rolled = _tensor(z['rolled'][:, t], torch.long)
    assert torch.equal(out.collision_robot_steps[a], gold_rolled[a]), f'{case} rolled at {t}'
    errors = dict(
      q=(robot.q - _tensor(z['q'][:, t]))[a], qd=(robot.qd - _tensor(z['qd'][:, t]))[a],
      tip=(robot.tip_pos - _tensor(z['tip'][:, t, :, :3, 3]))[a],
      rot=(robot.tip_rot - _tensor(z['tip'][:, t, :, :3, :3]))[a],
      reward=(out.reward.double() - _tensor(z['r'][:, t], torch.float64))[a],
      score=(robot.score.double() - _tensor(z['score'][:, t], torch.float64))[a],
      accel=(out.acceleration.double() - _tensor(z['accel'][:, t], torch.float64))[a],
      penalty=(out.collision_penalty.double() - _tensor(z['penalty'][:, t], torch.float64))[a])
    for key, x in errors.items():
      if len(x):
        worst[key] = max(worst[key], x.abs().max().item())
    inexact += (errors['q'] != 0).sum().item() + (errors['qd'] != 0).sum().item()
    for key, limit in LIMITS.items():
      assert worst[key] <= limit, f'{case} {key} error {worst[key]} at step {t}'
    coverage.update(cfg, st, out, _within_before_reset(z, pool, cfg, t))

  for x, value in kept:
    assert torch.equal(x, value), 'a handed-out tensor changed'
  assert env.residual_collisions == 0
  print(f'\n{case}: {W * steps} world-steps, {alive.sum().item()}/{W} never flipped, worst {worst}, '
        f'q or qd entries not bit-identical {inexact}, {coverage}')
  for key, least in EXPECT[case].items():
    assert getattr(coverage, key) >= least, f'{case} covered only {getattr(coverage, key)} of {key}'
  assert coverage.flips * 4 <= W, 'too many worlds lost to rounding to trust this case'


def _within_before_reset(z, pool: Scene, cfg, t: int) -> torch.Tensor:
  """targets_within on the original's tips and the scene each world had during step t."""
  scene = _tensor(z['scene'][:, t], torch.long)
  return targets_within(_tensor(z['tip'][:, t, :, :3, 3]), _tensor(z['tip'][:, t, :, :3, :3]),
                        pool.target_pos[scene], pool.target_rot[scene], cfg)


# Single functions against tables from the original.


@pytest.fixture(scope='module')
def tables() -> dict:
  if not ORIGINAL_PYTHON.exists():
    pytest.skip(f'{ORIGINAL_PYTHON} is missing: nothing was compared')
  REVIEW.mkdir(parents=True, exist_ok=True)
  path = REVIEW / 'env_adv_extras.npz'
  if not _fresh(path):
    job = _run_original('extras', '0', out=path.with_suffix('.tmp.npz'))
    log, _ = job.communicate()
    assert job.returncode == 0, log[-3000:]
    path.with_suffix('.tmp.npz').replace(path)
  return dict(np.load(path))


def test_augmentation_matches_box_info(tables):
  rot, spans = _tensor(tables['box_rot']), _tensor(tables['box_spans'])
  for code in range(4):
    codes = torch.full((len(rot), 1), code, device=DEVICE)
    new_rot, new_spans = augment_boxes(rot[:, None], spans[:, None], codes)
    assert (new_rot[:, 0] - _tensor(tables['aug_rot'][:, code])).abs().max() < 1e-6
    assert torch.equal(new_spans[:, 0], _tensor(tables['aug_spans'][:, code]))


def test_solve_check_differs_only_within_rounding_of_the_boundary(tables):
  cfg = env_cfg(num_obstacles=1)
  tip_pos, tip_rot = _tensor(tables['tip_pos']), _tensor(tables['tip_rot'])
  pos, rot = _tensor(tables['target_pos']), _tensor(tables['target_rot'])
  n = len(pos)
  # n worlds, one robot, one target.
  got = targets_within(tip_pos[:, None], tip_rot[:, None], pos[:, None], rot[:, None], cfg)[:, 0, 0]
  gold = _tensor(tables['solved'], torch.bool)
  dist = np.linalg.norm(tables['tip_pos'] - tables['target_pos'].astype(np.float64), axis=-1)
  trace = (tables['tip_rot'] * tables['target_rot'].astype(np.float64)).sum((-1, -2))
  angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
  margin = np.abs(np.maximum(dist - cfg.position_threshold, angle - cfg.angle_threshold_rad))
  wrong = (got != gold).cpu().numpy()
  print(f'solve check on {n} near-boundary poses: {gold.sum().item()} solved, {wrong.sum()} differ, '
        f'worst margin of a difference {margin[wrong].max() if wrong.any() else 0:.2e}')
  assert not wrong.any() or margin[wrong].max() < FLIP_MARGIN
  assert 0.1 * n < gold.sum() < 0.9 * n


def test_score_matches_the_original_with_return_to_start(tables):
  cfg = env_cfg(num_targets=5, num_obstacles=1, return_to_start_reward_weight=0.3)
  fake = SimpleNamespace(cfg=cfg, _home=HOME)
  got = BalletEnv._score(fake, _tensor(tables['score_done'], torch.bool), _tensor(tables['score_q']))
  gold = _tensor(tables['scores'], torch.float64)
  assert (got.double() - gold).abs().max() < 2e-7
  assert len(torch.unique(gold)) > 100  # The shaped branch was reached.


def test_random_rollback_distribution_matches_the_original(tables):
  """From states where several robots collide and rollback cascades, two-sample chi-square."""
  states, moves, outcomes = tables['states'], tables['moves'], tables['outcomes']
  W = outcomes.shape[1]
  cfg = env_cfg(num_obstacles=1, num_targets=2, rollback='random', nconmax=64)
  scene = default_scene(cfg.world, len(states), 2)._replace(q0=_tensor(states))
  env = BalletEnv(dataclasses.replace(cfg, collisions_enabled=False), W, scene, DEVICE, seed=9)
  env.cfg = cfg
  _install_rule(env)
  for k in range(len(states)):
    env.reset(scene_ids=torch.full((W, ), k, device=DEVICE))
    start = env.observe().robot.q
    robot = env.step(_tensor(moves[k]).expand(W, 4, 7)).next_robot
    rolled = (robot.qd == 0).all(-1)
    assert torch.equal(rolled, (robot.q == start).all(-1))
    masks = (rolled.long() << torch.arange(4, device=DEVICE)).sum(1).cpu().numpy()
    ours, gold = np.bincount(masks, minlength=16), np.bincount(outcomes[k], minlength=16)
    assert set(np.flatnonzero(ours)) == set(np.flatnonzero(gold)), f'state {k}: {ours} vs {gold}'
    seen = ours + gold > 0
    chi2 = (((ours - gold)[seen]) ** 2 / (ours + gold)[seen]).sum()
    df = seen.sum() - 1
    print(f'state {k}: outcomes {gold[seen].tolist()} original, {ours[seen].tolist()} here, '
          f'chi2 {chi2:.1f}, df {df}')
    assert chi2 < df + 4 * math.sqrt(2 * df) + 8, f'state {k}'  # About p = 1e-4.
    assert df >= 2


# Masking mistakes, without the original: only appear when worlds differ.


def _python_double_loop(within, done, dwell, dwell_steps):
  """planning_env.py:639-676 written out, for one world."""
  done, dwell = done.copy(), dwell.copy()
  for k in range(len(done)):
    if done[k]:
      continue
    for r in range(len(dwell)):
      if dwell[r] > 0:
        continue
      if within[k, r]:
        done[k] = True
        dwell[r] = dwell_steps
        break
  return done, dwell


@pytest.mark.parametrize('dwell_steps', [0, 2, 5])
def test_solve_targets_matches_the_double_loop(dwell_steps):
  gen = torch.Generator(device=DEVICE).manual_seed(dwell_steps)
  W, T, R = 3000, 12, 8
  rate = torch.rand(W, 1, 1, generator=gen, device=DEVICE) * 0.6
  within = torch.rand(W, T, R, generator=gen, device=DEVICE) < rate
  done = torch.rand(W, T, generator=gen, device=DEVICE) < rate[:, 0]
  dwelling = torch.rand(W, R, generator=gen, device=DEVICE) < rate[:, 0]
  dwell = torch.randint(3, (W, R), generator=gen, device=DEVICE) * dwelling
  done2, dwell2 = solve_targets(within, done, dwell, dwell_steps)
  for w in range(W):
    a, b = _python_double_loop(within[w].cpu().numpy(), done[w].cpu().numpy(), dwell[w].cpu().numpy(),
                               dwell_steps)
    assert done2[w].cpu().tolist() == a.tolist() and dwell2[w].cpu().tolist() == b.tolist(), w


def _real_collision_run(ids: torch.Tensor, pool: Scene, actions: torch.Tensor, steps: int) -> list:
  """The worlds `ids` of one fixed set of histories, stepped together with spheres and contacts on."""
  cfg = env_cfg(rollback='deterministic', num_targets=pool.target_pos.shape[1])
  env = Scripted(cfg, len(ids), pool, DEVICE, seed=0, debug=True)
  size = len(pool.q0)
  episode = torch.zeros(len(ids), dtype=torch.long, device=DEVICE)
  schedule = lambda: (ids * 7 + episode * 13) % size
  env.reset(scene_ids=schedule())
  out = []
  for t in range(steps):
    env.forced = schedule()
    # Worlds restart at different times, so histories are staggered: half at step 37.
    stepped = env.step(actions[t, ids])
    episode = episode + stepped.terminal.long()
    env.forced = schedule()
    if t == 37:
      late = (ids % 2 == 0).nonzero().flatten()
      episode = episode.index_add(0, late, torch.ones(len(late), dtype=torch.long, device=DEVICE))
      env.forced = schedule()
      env.reset(late)
    robot = stepped.next_robot
    out.append((robot.q, robot.qd, robot.dwell, robot.target_done, robot.tip_pos, stepped.reward,
                stepped.terminal, stepped.collision_robot_steps))
  return out


def test_a_world_does_not_notice_who_else_is_in_the_batch():
  pool = Scene(*[x[:64] for x in stored_scenes(4)])
  cfg = env_cfg(rollback='deterministic')
  # Half of the scenes have targets on the home tips: these end early and at varied times.
  pos, rot = _tip_at(cfg, pool)
  pick = torch.arange(10, device=DEVICE) % 4
  easy = (torch.arange(64, device=DEVICE) % 2 == 0)[:, None, None]
  pool = pool._replace(
    target_pos=torch.where(easy, pos[:, pick], pool.target_pos),
    target_rot=torch.where(easy[..., None], rot[:, pick], pool.target_rot))
  W, steps = 48, 300
  gen = torch.Generator(device=DEVICE).manual_seed(4)
  hold = torch.randint(5, 40, (W,), generator=gen, device=DEVICE)
  actions = torch.zeros(steps, W, 4, 7, device=DEVICE)
  current = torch.zeros(W, 4, 7, device=DEVICE)
  for t in range(steps):
    change = (t % hold == 0)[:, None, None] & (torch.rand(W, 4, 7, generator=gen, device=DEVICE) < 0.7)
    current = torch.where(change, torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1, current)
    actions[t] = current
  every = torch.arange(W, device=DEVICE)
  whole = _real_collision_run(every, pool, actions, steps)
  subset = torch.flip(every[::2], (0,))
  part = _real_collision_run(subset, pool, actions, steps)
  rolled = terminals = cascades = 0
  for t, (a, b) in enumerate(zip(whole, part)):
    for x, y in zip(a, b):
      assert torch.equal(x[subset], y), f'step {t}'
    rolled += a[7].sum().item()
    terminals += a[6].sum().item()
    cascades += (a[7] >= 2).sum().item()
  print(f'batch independence: {W * steps} world-steps, {rolled} rollbacks, {cascades} multi, '
        f'{terminals} terminals')
  assert rolled > 200 and cascades > 5 and terminals > 20


def test_every_observation_draws_fresh_independent_codes():
  cfg = EnvCfg(timestep_s=0.5)  # 24 steps per episode at most.
  W = 512
  env = BalletEnv(cfg, W, stored_scenes(4), DEVICE, seed=31)
  zero = torch.zeros(W, 4, 7, device=DEVICE)
  previous = env.observe().robot.aug_code
  same_step, same_world, redrawn = [], [], []
  counts = torch.zeros(4, device=DEVICE)
  for _ in range(60):
    out = env.step(zero)
    row, now = out.next_robot.aug_code, env.observe().robot.aug_code
    terminal = out.terminal
    assert torch.equal(row[~terminal], now[~terminal])  # s' of t is s of t+1.
    if terminal.any():
      redrawn.append((row[terminal] == now[terminal]).float().mean())
    same_step.append((row == previous).float().mean())
    same_world.append((row[1:] == row[:-1]).float().mean())
    counts += torch.bincount(row.flatten(), minlength=4)
    previous = now
  assert redrawn
  for name, x in (('over time', same_step), ('between worlds', same_world), ('at reset', redrawn)):
    mean = torch.stack(x).mean().item()
    print(f'aug codes equal {name}: {mean:.4f}')
    assert abs(mean - 0.25) < 0.01, name
  shares = counts / counts.sum()
  assert (shares - 0.25).abs().max() < 0.005
