"""Hindsight against the original code, gate G3b. Goldens come from golden/replay_make_goldens.py.

The goldens hold what agents/actor.py really stored for real and hindsight episodes, with its
random picks recorded. The tests replay those picks into `relabel` and compare per step.
"""

from __future__ import annotations

import collections
import functools
import itertools
import pickle
import re

import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, TrainCfg
from roboballet_mjlab.env import ballet_env
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.replay import hindsight
from roboballet_mjlab.replay.hindsight import HindsightQueue, Picks, draw_picks, eligible, mutated_action, relabel, score_of
from roboballet_mjlab.replay.replay import (
  STAT,
  STAT_NAMES,
  EpisodeBatch,
  EpisodeBuffer,
  ReplayBuffer,
  row_storage,
  select_episodes,
)
from roboballet_mjlab.structs import RobotState, Scene
from tests.test_sim_common import DEVICE, GOLDEN, ORIGINAL, PROJECT, env_cfg, stored_scenes

CASES = ('verify12', 'random', 'allrelabel', 'nearhome', 'mixed', 'rts', 'solved', '8r10t', '4r40t', '4r1t', '4r3t')
HOME = np.array(HOME_QPOS, dtype=np.float32)
PICKLE = PROJECT / 'runs/mjlab_port_analysis/scripts_baseline/out/golden_nocollision_her_4r10t.pkl'
SCORE_TOLERANCE = 1e-6
ACTION_TOLERANCE = 2e-6


def _tensor(x, dtype=None):
  return torch.as_tensor(x, dtype=dtype, device=DEVICE)


@functools.lru_cache(maxsize=None)
def load_case(name: str):
  z = dict(np.load(GOLDEN / f'replay_her_{name}.npz'))
  R, T, k, rts = int(z['num_robots']), int(z['num_targets']), int(z['k']), bool(z['rts'])
  cfg = env_cfg(num_robots=R, num_targets=T, return_to_start_reward_weight=0.25 if rts else 0.0,
                return_to_start_required=rts)
  return z, cfg, TrainCfg(hindsight_fraction=k / T), k


def batch_from_golden(z: dict, cfg) -> EpisodeBatch:
  n, height = z['q'].shape[:2]
  store = row_storage(cfg, (1,), 'cpu')
  dtype = {name: getattr(store, name).dtype for name in RobotState._fields}
  robot = RobotState(
    _tensor(z['q']), _tensor(z['qd']), _tensor(z['dwell'], dtype['dwell']), _tensor(z['tip_pos']),
    _tensor(z['tip_rot']), _tensor(z['done']), _tensor(z['score']), _tensor(z['step'], dtype['step']),
    torch.zeros(n, height, cfg.world.num_obstacles, dtype=dtype['aug_code'], device=DEVICE))
  stats = _tensor(z['stats'])
  return EpisodeBatch(
    robot, _tensor(z['action']), _tensor(z['reward']), _tensor(z['length'], torch.long),
    Scene(*[_tensor(z[f]) for f in Scene._fields]), stats[:, STAT['network_step']].long(),
    torch.zeros(n, dtype=torch.bool, device=DEVICE), stats)


def picks_from_golden(z: dict) -> Picks:
  return Picks(_tensor(z['chosen']), _tensor(z['arm'], torch.long), _tensor(z['time'], torch.long))


def generator(seed: int = 0) -> torch.Generator:
  return torch.Generator(device=DEVICE).manual_seed(seed)


def branch_counts(z: dict, cfg, k: int) -> collections.Counter:
  """How often the risky branches fired in the original, read from its own output alone."""
  counts = collections.Counter()
  max_steps = cfg.max_steps
  for i in range(len(z['length'])):
    A, H = int(z['length'][i]), int(z['h_length'][i])
    counts['real_episodes'] += 1
    counts['real_solved_some'] += int(z['done'][i, A].sum() > 0)
    if H < 0:
      counts['no_hindsight_solved_at_least_k'] += 1
      counts['no_hindsight_and_real_ended_early'] += int(A < max_steps)
      continue
    counts['hindsight_episodes'] += 1
    counts['fewer_than_k_solved'] += int(z['h_done'][i, H].sum() < k)
    counts['real_solved_target_missed_in_hindsight'] += int((z['done'][i, A] & ~z['h_done'][i, H]).any())
    all_done_end = bool(z['h_done'][i, H].all())
    counts['all_done_before_the_end'] += int(all_done_end and H < A)
    counts['all_done_and_timeout'] += int(all_done_end and H == A)
    counts['timeout_with_dwell_left'] += int(H == max_steps and z['h_dwell'][i, H].any())
    if cfg.return_to_start_required:
      parked = (np.abs(z['q'][i, H] - HOME) < cfg.return_to_start_epsilon).all()
      counts['parked_ending'] += int(H < A and all_done_end and parked)
    for t in range(H):
      counts['hindsight_steps'] += 1
      all_done = bool(z['h_done'][i, t].all())
      dwelling = z['h_dwell'][i, t] > 0
      to_home = HOME - z['q'][i, t]
      takeover = all_done & (np.abs(to_home) < cfg.return_to_start_threshold).all(-1)
      stored, raw = z['h_action'][i, t], z['action'][i, t]
      counts['modified_steps'] += int(not np.array_equal(stored, raw))
      counts['dwell_zeroed_robot_steps'] += int((dwelling & ~takeover).sum())
      counts['takeover_robot_steps'] += int(takeover.sum())
      counts['takeover_over_dwell_robot_steps'] += int((takeover & dwelling).sum())
      counts['final_dwell_steps'] += int(all_done and dwelling.any())
      counts['all_done_no_dwell_steps'] += int(all_done and not dwelling.any())
      # The golden's own arithmetic for these branches, from the formula in the source.
      for r in np.flatnonzero(dwelling & ~takeover):
        assert (stored[r] == 0).all(), (i, t, r)
      for r in np.flatnonzero(takeover):
        want = to_home[r] / cfg.timestep_s / cfg.max_joint_velocity
        assert np.abs(stored[r] - want).max() < ACTION_TOLERANCE, (i, t, r)
        counts['takeover_beyond_unit_bound_robot_steps'] += int(np.abs(stored[r]).max() > 1)
        counts['takeover_over_raw_robot_steps'] += int(not dwelling[r])
      for r in np.flatnonzero(~dwelling & ~takeover):
        assert np.array_equal(stored[r], raw[r]), (i, t, r)
  return counts


def compare_with_golden(name: str) -> None:
  z, cfg, train, k = load_case(name)
  batch = batch_from_golden(z, cfg)
  keep = z['h_length'] >= 0
  assert torch.equal(eligible(batch, k), _tensor(keep))
  out = relabel(batch, cfg, train, generator(), picks_from_golden(z))
  kept = np.flatnonzero(keep)
  assert out.length.tolist() == z['h_length'][kept].tolist(), 'episode lengths, so terminal rows'
  assert out.is_hindsight.all()
  got = {name_: x.cpu().numpy() for name_, x in out.robot._asdict().items()}
  action, reward = out.action.cpu().numpy(), out.reward.cpu().numpy()
  for j, i in enumerate(kept):
    H = int(z['h_length'][i])
    assert np.array_equal(got['dwell'][j, : H + 1], z['h_dwell'][i, : H + 1]), f'{name} episode {i} dwell'
    assert np.array_equal(got['target_done'][j, : H + 1], z['h_done'][i, : H + 1]), f'{name} episode {i} done'
    assert np.abs(got['score'][j, : H + 1] - z['h_score'][i, : H + 1]).max() <= SCORE_TOLERANCE, f'{name} {i} score'
    assert np.abs(reward[j, :H] - z['h_reward'][i, :H]).max() <= SCORE_TOLERANCE, f'{name} {i} reward'
    assert np.abs(action[j, :H] - z['h_action'][i, :H]).max() <= ACTION_TOLERANCE, f'{name} {i} action'
    assert np.array_equal(got['q'][j, : H + 1], z['q'][i, : H + 1])
    assert np.array_equal(got['qd'][j, : H + 1], z['qd'][i, : H + 1])
    assert np.array_equal(got['tip_pos'][j, : H + 1], z['tip_pos'][i, : H + 1])
    assert np.array_equal(got['tip_rot'][j, : H + 1], z['tip_rot'][i, : H + 1])
    assert np.array_equal(got['step'][j, : H + 1], z['h_step'][i, : H + 1])
    want_terminal = np.arange(H) == H - 1
    assert np.array_equal(z['h_terminal'][i, :H] == 1, want_terminal), 'the golden itself'
  assert torch.equal(out.scene.target_pos.cpu(), torch.as_tensor(z['h_target_pos'][kept]))
  assert torch.equal(out.scene.target_rot.cpu(), torch.as_tensor(z['h_target_rot'][kept]))
  for field in ('base_pos', 'base_rot', 'box_pos', 'box_rot', 'box_spans', 'q0'):
    assert np.array_equal(getattr(out.scene, field).cpu().numpy(), z[field][kept])
  # Statistics are the real episode's with the flag set, actor.py:397-400.
  assert np.array_equal(out.stats.cpu().numpy(), z['h_stats'][kept])
  assert (out.stats[:, STAT['is_hindsight_replay']] == 1).all()
  assert out.behaviour_step.tolist() == z['h_stats'][kept, STAT['network_step']].astype(np.int64).tolist()

  # Through the ring, the stored transition must be the original's writer item.
  replay = ReplayBuffer(cfg, int(z['h_length'][kept].sum()), DEVICE, max_episodes=len(kept))
  replay.insert(out)
  stored = replay.fetch(torch.arange(replay.size, device=DEVICE))
  position = 0
  for i in kept:
    H = int(z['h_length'][i])
    end = position + H
    span = slice(position, end)
    assert np.abs(stored.action[span].cpu().numpy() - z['h_action'][i, :H]).max() <= ACTION_TOLERANCE
    assert np.abs(stored.reward[span].cpu().numpy() - z['h_reward'][i, :H]).max() <= SCORE_TOLERANCE
    assert np.array_equal(stored.terminal[span].cpu().numpy(), z['h_terminal'][i, :H] == 1)
    assert np.array_equal(stored.obs.robot.dwell[span].cpu().numpy(), z['h_dwell'][i, :H])
    assert np.array_equal(stored.next_obs.robot.dwell[span].cpu().numpy(), z['h_dwell'][i, 1 : H + 1])
    assert np.array_equal(stored.next_obs.robot.target_done[span].cpu().numpy(), z['h_done'][i, 1 : H + 1])
    assert np.array_equal(stored.next_obs.robot.q[span].cpu().numpy(), z['q'][i, 1 : H + 1])
    assert np.array_equal(stored.obs.scene.target_pos[span].cpu().numpy()[0], z['h_target_pos'][i])
    position = end
  assert position == replay.size


@pytest.mark.parametrize('name', CASES)
def test_differential_against_original(name):
  compare_with_golden(name)


def test_risky_branches_fired():
  """A differential that never reached a branch proves nothing about it. Each one must fire."""
  total = collections.Counter()
  per_case = {}
  for name in CASES:
    z, cfg, _, k = load_case(name)
    per_case[name] = branch_counts(z, cfg, k)
    total += per_case[name]
  print('\nBranch counts in the original, by case:')
  for name, counts in per_case.items():
    print(f'  {name}: {dict(sorted(counts.items()))}')
  print(f'  total: {dict(sorted(total.items()))}')
  assert total['real_episodes'] >= 64 and total['hindsight_episodes'] >= 64
  for branch in ('all_done_before_the_end', 'all_done_and_timeout', 'final_dwell_steps', 'takeover_robot_steps', 'takeover_over_dwell_robot_steps',
                 'takeover_over_raw_robot_steps', 'takeover_beyond_unit_bound_robot_steps', 'dwell_zeroed_robot_steps',
                 'fewer_than_k_solved', 'real_solved_target_missed_in_hindsight', 'no_hindsight_solved_at_least_k',
                 'no_hindsight_and_real_ended_early', 'real_solved_some', 'parked_ending', 'all_done_no_dwell_steps',
                 'timeout_with_dwell_left'):
    assert total[branch] > 0, branch
  # Takeover has to come from cases without the return-to-start flags too, DESIGN 11.1.
  assert sum(per_case[n]['takeover_over_dwell_robot_steps'] for n in CASES if n != 'rts') > 0


def _rounded_values(rewards: np.ndarray) -> list[float]:
  return np.unique(np.round(rewards.astype(np.float64), 5)).tolist()


def test_e3_verify12_numbers():
  """The run behind algo_spec 5.3 and verify_her.log, reproduced by replay_make_goldens.py."""
  z, cfg, train, k = load_case('verify12')
  out = relabel(batch_from_golden(z, cfg), cfg, train, generator(), picks_from_golden(z))
  action, reward = out.action.cpu().numpy(), out.reward.cpu().numpy()
  raw = z['action']
  modified = (action != raw).any((2, 3)).sum()
  assert (action.shape[0], action.shape[1]) == (12, 120)
  assert modified == 216 and reward.size == 1440, modified
  # The original's own rewards, then ours. verify_her.log printed {0, 0.1} for the first episode
  # only. Over all 12 there are 46 positive steps for 47 solves: one step solves two targets.
  for values in (z['h_reward'], reward):
    assert _rounded_values(values[0]) == [0.0, 0.1]
    assert _rounded_values(values) == [0.0, 0.1, 0.2]
    assert (values > 0).sum() == 46 and (values > 0.15).sum() == 1
  assert (reward < 0).sum() == 0
  assert (z['reward'] < 0).sum() == 463, 'real episodes in the same run'
  solved = out.robot.target_done[:, -1].sum(1).tolist()
  assert solved == [4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4]
  returns = np.round(reward.astype(np.float64).sum(1), 5).tolist()
  assert returns == [0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]


def _nocollision_batch(Z, cfg) -> tuple[EpisodeBatch, Picks]:
  """env_nocollision_her_4r10t.npz has no row 0 and no picks. Row 0 is home, and the original
  never reads tip row 0. Picks come from finding each moved target among the stored tips."""
  n, A, R = *Z['a'].shape[:2], 4
  zeros = lambda x: np.zeros_like(x[:, :1])
  rows = lambda x, first=None: _tensor(np.concatenate([zeros(x) if first is None else first, x], 1))
  tips = Z['tip']
  robot = RobotState(
    rows(Z['q'], np.tile(HOME, (n, 1, R, 1))), rows(Z['qd']), _tensor(np.rint(np.concatenate([zeros(Z['dwell']), Z['dwell']], 1) * 5), torch.int8),
    rows(tips[..., :3, 3]), rows(tips[..., :3, :3]), rows(Z['done']), _tensor(np.concatenate([zeros(Z['score']), Z['score']], 1), torch.float32),
    _tensor(np.concatenate([zeros(Z['step']), Z['step']], 1), torch.int8),
    torch.zeros(n, A + 1, 30, dtype=torch.int8, device=DEVICE))
  place = Z['placements']
  scene = Scene(_tensor(place[:, :, :3, 3]), _tensor(place[:, :, :3, :3]), _tensor(Z['box_pos']), _tensor(Z['box_rot']),
                _tensor(Z['box_spans']), _tensor(Z['target_pos']), _tensor(Z['target_rot']), _tensor(HOME).repeat(n, R, 1))
  batch = EpisodeBatch(robot, _tensor(Z['a'] / cfg.max_joint_velocity), _tensor(Z['r'], torch.float32),
                       torch.full((n,), A, device=DEVICE), scene, torch.zeros(n, dtype=torch.long, device=DEVICE),
                       torch.zeros(n, dtype=torch.bool, device=DEVICE), torch.zeros(n, 12, device=DEVICE))
  chosen = (Z['h_target_pos'] != Z['target_pos']).any(-1)
  arm, time = np.zeros((n, 10), dtype=np.int64), np.zeros((n, 10), dtype=np.int64)
  for i, target in zip(*np.nonzero(chosen)):
    # tip row t of the npz is state t + 1.
    distance = np.abs(tips[i, :, :, :3, 3] - Z['h_target_pos'][i, target]).max(-1)
    row, robot_index = np.unravel_index(distance.argmin(), distance.shape)
    assert distance[row, robot_index] < 1e-5
    arm[i, target], time[i, target] = robot_index, row + 1
  return batch, Picks(_tensor(chosen), _tensor(arm), _tensor(time))


def test_e3_stored_actions_on_nocollision_golden():
  """The 20-episode golden of paper_baseline, a different run from verify12, so other totals."""
  Z = dict(np.load(GOLDEN / 'env_nocollision_her_4r10t.npz'))
  cfg = env_cfg()
  batch, picks = _nocollision_batch(Z, cfg)
  assert (picks.chosen.sum(1) == 4).all() and (picks.time >= 0).all()
  out = relabel(batch, cfg, TrainCfg(), generator(), picks)
  assert (out.length == 120).all()
  assert np.abs(out.scene.target_pos.cpu().numpy() - Z['h_target_pos']).max() < 1e-5
  assert torch.equal(out.robot.target_done, _tensor(Z['h_done']))
  gold_dwell = np.rint(Z['h_dwell'] * 5).astype(np.int8)
  assert np.array_equal(out.robot.dwell.cpu().numpy(), gold_dwell)
  assert np.abs(out.reward.cpu().numpy() - Z['h_r']).max() <= SCORE_TOLERANCE

  # Stored actions: zero exactly for arms dwelling before the step, raw otherwise. Takeover
  # never fires here, as no hindsight episode gets all ten done.
  assert not Z['h_done'][:, -1].all(1).any()
  stored = out.action.cpu().numpy()
  raw = Z['a'] / cfg.max_joint_velocity
  dwelling = gold_dwell[:, :-1] > 0
  assert (stored[dwelling] == 0).all()
  np.testing.assert_allclose(stored[~dwelling], raw[~dwelling].astype(np.float32), atol=1e-7)
  modified = (stored != raw.astype(np.float32)).any((2, 3)).sum()
  # Measured on this golden: not the 216 of 1440 from verify12, see test_e3_verify12_numbers.
  assert modified == 362 and stored.shape[0] * stored.shape[1] == 2400, modified
  # 0.2 when two targets are solved in the same step.
  assert _rounded_values(out.reward.cpu().numpy()) == [0.0, 0.1, 0.2]
  assert (out.reward >= 0).all()
  solved = out.robot.target_done[:, -1].sum(1).tolist()
  assert solved == Z['h_done'][:, -1].sum(1).tolist() and solved.count(3) == 2 and solved.count(4) == 18


def test_nocollision_npz_matches_its_pickle():
  gold = pickle.load(open(PICKLE, 'rb'))
  Z = np.load(GOLDEN / 'env_nocollision_her_4r10t.npz')
  assert len(gold) == 20
  for i, g in enumerate(gold):
    h = g['hindsight']
    assert h['nsteps'] == 120
    assert np.array_equal(h['r'], Z['h_r'][i])
    assert np.array_equal(h['done'], Z['h_done'][i])
    assert np.array_equal(np.asarray(h['dwell'], dtype=np.float32), Z['h_dwell'][i])
    assert np.array_equal(np.asarray(g['trace']['a'], dtype=np.float32), Z['a'][i])
    positions = np.array([t[0] for t in h['start_targets']], dtype=np.float32)
    assert np.array_equal(positions, Z['h_target_pos'][i])


# The shared expressions are duplicated from the env, so the env is the reference for them.


def test_stat_names_match_the_original():
  body = (ORIGINAL / 'train/stats.py').read_text().split('class TrajectoryStats')[1]
  assert tuple(re.findall(r'^  (\w+): (?:float|int) = ', body, re.M)) == STAT_NAMES
  assert len(STAT_NAMES) == 12


def test_scoring_functions_are_the_envs_own():
  assert hindsight.targets_within is ballet_env.targets_within
  assert hindsight.solve_targets is ballet_env.solve_targets


@pytest.mark.parametrize('weight,required', [(0.0, False), (0.25, True)])
def test_score_matches_the_env(weight, required):
  cfg = env_cfg(return_to_start_reward_weight=weight, return_to_start_required=required)
  env = BalletEnv(cfg, 4, stored_scenes(4), DEVICE, seed=0)
  gen = generator(1)
  home = _tensor(HOME)
  done = torch.rand(512, 10, generator=gen, device=DEVICE) < 0.8
  done[:128] = True
  q = home + (torch.rand(512, 4, 7, generator=gen, device=DEVICE) - 0.5) * torch.tensor([0.6, 2.0], device=DEVICE)[
    torch.randint(2, (512, 1, 1), generator=gen, device=DEVICE)]
  assert torch.equal(score_of(done, q, cfg, home), env._score(done, q))


def test_action_mutation_matches_what_the_env_does_to_its_request():
  """DESIGN 11.1: weight 0, required False, all done, dwell nonzero, joints in the box."""
  W = 256
  cfg = env_cfg(collisions_enabled=False)
  env = BalletEnv(cfg, W, stored_scenes(4), DEVICE, seed=2)
  gen = generator(2)
  home = _tensor(HOME)
  st = env.observe().robot
  q = home + (torch.rand(W, 4, 7, generator=gen, device=DEVICE) - 0.5) * 0.56  # Some in the box, some out.
  qd = (torch.rand(W, 4, 7, generator=gen, device=DEVICE) - 0.5) * 2.0
  dwell = torch.randint(6, (W, 4), generator=gen, device=DEVICE)
  done = torch.rand(W, 10, generator=gen, device=DEVICE) < 0.5
  done[: W // 2] = True
  env._robot = st._replace(q=q, qd=qd, dwell=dwell, target_done=done)
  action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
  out = env.step(action.clone())

  stored = mutated_action(action, dwell > 0, done, q, cfg, home)
  requested = stored * cfg.max_joint_velocity
  dv = cfg.max_joint_acceleration * cfg.timestep_s
  expected_qd = torch.minimum(torch.maximum(requested, qd - dv), qd + dv)
  assert (out.next_robot.qd - expected_qd).abs().max() < 1e-5
  # The branches this has to cover did occur.
  takeover = done.all(1, keepdim=True) & ((home - q).abs() < cfg.return_to_start_threshold).all(-1)
  assert (takeover & (dwell > 0)).sum() > 0 and (takeover & (dwell == 0)).sum() > 0
  assert ((dwell > 0) & ~takeover).sum() > 0 and ((dwell == 0) & ~takeover).sum() > 0
  # Takeover units: rad/s divided by max velocity, so beyond 1 is possible.
  assert stored[takeover].abs().max() > 1.0
  # Nothing written into the argument.
  fresh = action.clone()
  mutated_action(fresh, dwell > 0, done, q, cfg, home)
  assert torch.equal(fresh, action)


# Picks.


def _chi2(counts: np.ndarray) -> float:
  expected = counts.sum() / counts.size
  return float(((counts - expected) ** 2 / expected).sum())


def test_picks_uniform_without_replacement():
  n, T, k, R = 240_000, 5, 3, 4
  gen = generator(3)
  done = torch.zeros(n, T, dtype=torch.bool, device=DEVICE)
  # A third has nothing solved, a third has target 0 solved, a third has targets 1 and 3 solved.
  done[n // 3 : 2 * n // 3, 0] = True
  done[2 * n // 3 :, [1, 3]] = True
  length = torch.tensor([2, 3, 10, 120], device=DEVICE)[torch.randint(4, (n,), generator=gen, device=DEVICE)]
  picks = draw_picks(done, length, R, k, gen)
  chosen = picks.chosen.cpu().numpy()
  solved = done.sum(1).cpu().numpy()
  assert (chosen.sum(1) == k - solved).all(), 'exactly k - solved'
  assert not (picks.chosen & done).any(), 'never a solved target'
  # All subsets equally likely, in each group. The limits are 0.1% points: 10 subsets, df 9;
  # 6 subsets, df 5; 3 subsets, df 2.
  for group, unsolved, size, limit in [(0, range(5), 3, 27.9), (1, range(1, 5), 2, 20.5), (2, (0, 2, 4), 1, 13.8)]:
    rows = chosen[group * n // 3 : (group + 1) * n // 3]
    subsets = list(itertools.combinations(unsolved, size))
    counts = np.array([(rows[:, list(s)].sum(1) == size).sum() for s in subsets])
    assert counts.sum() == len(rows)
    assert _chi2(counts) < limit, (group, counts)

  arm, time = picks.arm.cpu().numpy(), picks.time.cpu().numpy()
  length_np = length.cpu().numpy()
  assert arm.min() == 0 and arm.max() == R - 1
  assert _chi2(np.bincount(arm[chosen], minlength=R)) < 16.3  # df 3.
  # Time is in [1, A - 1], inclusive at both ends, and uniform.
  for A, limit in [(2, None), (3, 10.8), (10, 26.1), (120, 170.0)]:
    rows = time[length_np == A]
    assert rows.min() == 1 and rows.max() == A - 1, A
    if limit is not None:
      assert _chi2(np.bincount(rows[chosen[length_np == A]], minlength=A)[1:]) < limit, A
  # Independent per target: the pair of arms of two chosen targets is uniform, repeats included.
  both = chosen[:, 0] & chosen[:, 1]
  pair = arm[both, 0] * R + arm[both, 1]
  assert _chi2(np.bincount(pair, minlength=R * R)) < 37.7  # df 15.


def _tip_tagged_batch(n: int, T: int, A: int, cfg) -> EpisodeBatch:
  """Each stored tip position is unique to (episode, row, arm), all outside the targets' reach."""
  R, O = cfg.world.num_robots, cfg.world.num_obstacles
  robot = row_storage(cfg, (n, A + 1), DEVICE)
  e = torch.arange(n, device=DEVICE)[:, None, None]
  t = torch.arange(A + 1, device=DEVICE)[None, :, None]
  r = torch.arange(R, device=DEVICE)[None, None, :]
  robot.tip_pos[...] = torch.stack([e.expand(n, A + 1, R).float(), t.expand(n, A + 1, R).float(), r.expand(n, A + 1, R).float()], -1)
  robot.tip_rot[...] = torch.eye(3, device=DEVICE)
  scene = Scene(
    torch.zeros(n, R, 3, device=DEVICE), torch.eye(3, device=DEVICE).repeat(n, R, 1, 1),
    torch.zeros(n, O, 3, device=DEVICE), torch.eye(3, device=DEVICE).repeat(n, O, 1, 1),
    torch.ones(n, O, 3, device=DEVICE), torch.full((n, T, 3), -5.0, device=DEVICE),
    torch.eye(3, device=DEVICE).repeat(n, T, 1, 1), torch.zeros(n, R, 7, device=DEVICE))
  return EpisodeBatch(
    robot, torch.rand(n, A, R, 7, device=DEVICE), torch.zeros(n, A, device=DEVICE), torch.full((n,), A, device=DEVICE),
    scene, torch.arange(n, device=DEVICE), torch.zeros(n, dtype=torch.bool, device=DEVICE), torch.zeros(n, 12, device=DEVICE))


def test_relabel_takes_unscaled_tips_from_valid_rows():
  n, A = 4096, 7
  cfg = env_cfg()
  batch = _tip_tagged_batch(n, 10, A, cfg)
  # Episode 0 to 3 have 1 to 4 targets solved at the end, with rows left in place.
  done = batch.robot.target_done
  for e in range(1000):
    done[e, A, : (e % 3) + 1] = True
  out = relabel(batch, cfg, TrainCfg(), generator(4))
  moved = out.scene.target_pos != batch.scene.target_pos
  moved = moved.any(-1)
  assert torch.equal(moved.sum(1), 4 - done[:, A].sum(1))
  assert not (moved & done[:, A]).any()
  pos = out.scene.target_pos[moved]
  e = moved.nonzero()[:, 0]
  assert torch.equal(pos[:, 0], e.float()), 'tip pose comes from the same episode, unscaled'
  assert pos[:, 1].min() == 1 and pos[:, 1].max() == A - 1, 'row range [1, A-1]'
  assert pos[:, 2].min() == 0 and pos[:, 2].max() == 3, 'arm range [0, R-1]'
  assert (out.scene.target_rot == torch.eye(3, device=DEVICE)).all()
  # Untouched targets keep their pose.
  assert (out.scene.target_pos[~moved] == -5.0).all()


# Fraction of a random policy, end to end through the real env, buffer, queue and replay.


def test_hindsight_fraction_on_real_pool():
  W = 256
  cfg = env_cfg()
  train = TrainCfg()
  pool = stored_scenes(4)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=5)
  buf = EpisodeBuffer(cfg, W, DEVICE)
  replay = ReplayBuffer(cfg, 400_000, DEVICE)
  gen = generator(5)
  queue = HindsightQueue(cfg, train, gen)
  real_transitions = real_episodes = hindsight_episodes = 0
  for step in range(3 * cfg.max_steps):
    obs = env.observe()
    action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    batch = buf.append(obs, action, env.step(action), behaviour_step=step)
    if batch is not None:
      replay.insert(batch)
      real_transitions += int(batch.length.sum())
      real_episodes += len(batch.length)
      relabelled = queue.add(batch)
      if relabelled is not None:
        replay.insert(relabelled)
        hindsight_episodes += len(relabelled.length)
  assert real_episodes == 3 * W
  assert replay.inserted > real_transitions, 'hindsight transitions count as inserts'
  sample = replay.sample(100_000, gen)
  fraction = sample.is_hindsight.float().mean().item()
  print(f'\nrandom policy on {len(pool.q0)} stored scenes: {real_episodes} real episodes, {hindsight_episodes} hindsight, '
        f'sampled hindsight fraction {fraction:.4f}, by episodes {hindsight_episodes / (real_episodes + hindsight_episodes):.4f}')
  assert 0.47 < fraction < 0.505  # All eligible and equally long here, so 0.5 up to sampling noise.
  assert torch.equal(sample.is_hindsight, sample.stats[:, STAT['is_hindsight_replay']] == 1)
  assert sample.behaviour_step.max() < cfg.max_steps * 3
  hind = sample.is_hindsight
  assert (sample.reward[hind] >= 0).all(), 'no collision penalty in hindsight'
  assert (sample.reward[~hind] < 0).any(), 'but real episodes do collide'
  # s' of a real terminal row is the terminal state, which has step 120 in this pool.
  assert (sample.next_obs.robot.step[sample.terminal] == cfg.max_steps).all()
  assert sample.terminal.any()
  # Hindsight episodes differ from the real ones in dwell, sometimes.
  assert (sample.obs.robot.dwell[hind] > 0).any()


# Flags, purity and small cases.


def test_flag_keeps_raw_stored_actions():
  z, cfg, train, k = load_case('nearhome')
  batch = batch_from_golden(z, cfg)
  flagged = relabel(batch, cfg, train, generator(), picks_from_golden(z))
  raw = relabel(batch, cfg, TrainCfg(hindsight_fraction=train.hindsight_fraction, hindsight_zero_dwelling_actions=False),
                generator(), picks_from_golden(z))
  keep = z['h_length'] >= 0
  assert torch.equal(raw.action, batch.action[_tensor(keep)])
  assert not torch.equal(flagged.action, raw.action)
  # Dwell, done, score and reward do not depend on it.
  for a, b in zip((flagged.robot.dwell, flagged.robot.target_done, flagged.reward, flagged.length),
                  (raw.robot.dwell, raw.robot.target_done, raw.reward, raw.length)):
    assert torch.equal(a, b)


def test_flag_keeps_collision_penalty():
  z, cfg, train, k = load_case('verify12')
  batch = batch_from_golden(z, cfg)
  zeroed = relabel(batch, cfg, train, generator(), picks_from_golden(z))
  kept = relabel(batch, cfg, TrainCfg(hindsight_zero_collision_penalty=False), generator(), picks_from_golden(z))
  assert (zeroed.reward >= 0).all() and (kept.reward < 0).any()
  real_delta = batch.robot.score[:, 1:] - batch.robot.score[:, :-1]
  penalty = real_delta - batch.reward
  assert (kept.reward - (zeroed.reward - penalty)).abs().max() < 1e-6
  # Against the real penalties in the file: count is 463 penalised steps.
  assert ((zeroed.reward - kept.reward) > 1e-3).sum() == 463


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@pytest.mark.parametrize('name', CASES)
def test_gpu_batch_gives_the_same_episodes_as_cpu(name):
  """The checks run where the batch is and the loop on the CPU, so the device must not show."""
  z, cfg, train, _ = load_case(name)
  batch, picks = batch_from_golden(z, cfg), picks_from_golden(z)

  def on(device):
    move = lambda x: x.to(device)
    moved = batch._replace(
      robot=type(batch.robot)(*map(move, batch.robot)),
      scene=type(batch.scene)(*map(move, batch.scene)),
      **{f: move(getattr(batch, f)) for f in ('action', 'reward', 'length', 'behaviour_step', 'is_hindsight', 'stats')},
    )
    return relabel(moved, cfg, train, torch.Generator(device).manual_seed(0), Picks(*map(move, picks)))

  here, there = on('cpu'), on('cuda')
  assert here.action.device.type == 'cpu'
  for tensor in (there.action, there.reward, there.length, there.robot.target_done, there.robot.dwell, there.robot.score):
    assert tensor.device.type == 'cuda'
  for field in ('dwell', 'target_done', 'score'):
    assert torch.equal(getattr(here.robot, field), getattr(there.robot, field).cpu()), field
  assert torch.equal(here.length, there.length.cpu())
  assert torch.equal(here.reward, there.reward.cpu())
  assert torch.equal(here.action, there.action.cpu())
  assert torch.equal(here.scene.target_pos, there.scene.target_pos.cpu())
  assert torch.equal(here.scene.target_rot, there.scene.target_rot.cpu())


def test_relabel_does_not_write_to_its_input():
  z, cfg, train, k = load_case('nearhome')
  batch = batch_from_golden(z, cfg)
  snapshot = [x.clone() for x in [*batch.robot, batch.action, batch.reward, batch.length, *batch.scene, batch.stats]]
  relabel(batch, cfg, train, generator(), picks_from_golden(z))
  for x, y in zip([*batch.robot, batch.action, batch.reward, batch.length, *batch.scene, batch.stats], snapshot):
    assert torch.equal(x, y)


def test_fresh_augmentation_on_every_row_including_terminal():
  z, cfg, train, k = load_case('random')
  batch = batch_from_golden(z, cfg)
  batch.robot.aug_code.fill_(3)
  out = relabel(batch, cfg, train, generator(7), picks_from_golden(z))
  codes = out.robot.aug_code
  assert codes.dtype == torch.int8 and codes.min() == 0 and codes.max() == 3
  final = codes[torch.arange(len(codes), device=DEVICE), out.length].cpu().numpy()
  assert _chi2(np.bincount(final.ravel(), minlength=4)) < 16.3, 'terminal row draws fresh too'
  assert _chi2(np.bincount(codes.cpu().numpy().ravel(), minlength=4)) < 16.3
  before_final = codes[torch.arange(len(codes), device=DEVICE), out.length - 1].cpu().numpy()
  assert (final != before_final).mean() > 0.65, 'not copied from the row before it'
  # Rows differ from their neighbours: one draw per observation, not per episode.
  assert (codes[:, 1:] != codes[:, :-1]).float().mean() > 0.7
  # s' of row i is row i + 1 in the ring, so a pair shares one draw.
  replay = ReplayBuffer(cfg, 10_000, DEVICE)
  replay.insert(out)
  got = replay.fetch(torch.arange(replay.size, device=DEVICE))
  same_episode = ~got.terminal[:-1]
  assert torch.equal(got.next_obs.robot.aug_code[:-1][same_episode], got.obs.robot.aug_code[1:][same_episode])
  off = relabel(batch, env_cfg(augment_obstacles=False), train, generator(7), picks_from_golden(z))
  assert (off.robot.aug_code == 0).all()


def test_solved_is_counted_from_the_terminal_observation():
  # actor.py:339 reads the final observation, which has no action of its own.
  cfg = env_cfg()
  batch = _tip_tagged_batch(2, 10, 5, cfg)
  done = batch.robot.target_done
  done[0, 4, :3] = True  # Three solved by the last action's start, row A - 1.
  done[0, 5, :4] = True  # The fourth only shows in the terminal row: so k = 4 is reached.
  done[1, 4, :3] = True
  done[1, 5, :3] = True
  assert eligible(batch, 4).tolist() == [False, True]
  out = relabel(batch, cfg, TrainCfg(), generator())
  assert len(out.length) == 1 and out.behaviour_step.tolist() == [1]
  # One target moves, and never a target solved in the terminal row.
  moved = (out.scene.target_pos != -5.0).any(-1)[0]
  assert moved.sum() == 1 and not moved[:3].any()


def test_no_episode_for_small_cases():
  cfg = env_cfg(num_targets=1)
  batch = _tip_tagged_batch(3, 1, 4, cfg)
  assert relabel(batch, cfg, TrainCfg(), generator()) is None, 'k = int(0.4) = 0'
  one = _tip_tagged_batch(3, 1, 1, cfg)
  assert relabel(one, cfg, TrainCfg(hindsight_fraction=1.0), generator()) is None, 'one action: no time to pick'
  two = _tip_tagged_batch(3, 1, 2, cfg)
  out = relabel(two, cfg, TrainCfg(hindsight_fraction=1.0), generator())
  assert out is not None and len(out.length) == 3
  assert (out.scene.target_pos[:, 0, 1] == 1).all(), 'the only row left is 1'


def test_queue_flushes_by_finished_count_and_on_request():
  z, cfg, train, k = load_case('solved')
  batch = batch_from_golden(z, cfg)
  n = len(batch.length)
  eligible_mask = eligible(batch, k)
  assert 0 < eligible_mask.sum() < n
  flush = 20
  queue = HindsightQueue(cfg, TrainCfg(hindsight_fraction=train.hindsight_fraction, hindsight_flush_episodes=flush), generator(8))
  order = np.arange(n)
  first = _tensor((order >= 8) & (order < 16))  # Four that qualify, four that do not.
  second = _tensor(order < 8) | _tensor((order >= 16) & (order < 25))
  assert int(eligible_mask[first].sum()) == 4
  assert queue.add(select_episodes(batch, first)) is None
  assert queue.finished == 8
  kept = sum(len(b.length) for b in queue.pending)
  assert kept == 4, 'only episodes that qualify take up memory'
  out = queue.add(select_episodes(batch, second))
  assert out is not None, 'flushed by count of finished episodes'
  assert len(out.length) == int(eligible_mask[first | second].sum())
  assert out.is_hindsight.all()
  assert queue.finished == 0 and not queue.pending
  # On request.
  assert queue.flush() is None
  rest = _tensor(order >= 25)
  assert queue.add(select_episodes(batch, rest)) is None
  out = queue.flush()
  assert out is not None and len(out.length) == int(eligible_mask[rest].sum())
  # Nothing eligible.
  none = select_episodes(batch, ~eligible_mask)
  assert queue.add(none) is None and queue.flush() is None
