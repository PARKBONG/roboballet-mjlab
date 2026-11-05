"""Regenerates the replay_her_*.npz goldens from the original code. Old venv only, no torch.

From $ROBOBALLET_PROJECT, which holds roboballet/ and venv/ (no GPU is touched):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  LD_LIBRARY_PATH="$(venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib" \
  venv/bin/python roboballet-mjlab/tests/golden/replay_make_goldens.py [case ...]

Real episodes run on the unmodified PlanningEnv with mesh collisions. Then the unmodified
generate_hindsight_trajectory and _write_episode run on them, with three stand-ins only:
`actor.random` records (and for some cases steers) the picks, a fake reverb client keeps
what the writer would store, and graph features are skipped because the writer only
forwards them. So h_action, h_reward and h_terminal are what the original really writes.

A random policy almost never reaches "all targets done", so the cases build the risky
branches on purpose: num_targets=T relabels everything so all get done before the end,
wander keeps arms inside the 0.25 rad takeover box, and `solved` first rolls out to find
tip poses, then puts real targets there so that the real episode solves them.
"""

import copy
import dataclasses
import os
import random
import sys
import types

import numpy as np

from roboballet.agents import actor, trajectory as trajectory_lib
from roboballet.environments import data_pb2, planning_env, target as target_lib
from roboballet.launch import hparams
from roboballet.train import stats as stats_lib

OUT = os.path.dirname(os.path.abspath(__file__))
HOME = np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0], dtype=np.float32)
MAX_VEL = 2.175
DWELL_STEPS = 5
STAT_NAMES = [f.name for f in dataclasses.fields(stats_lib.TrajectoryStats)]


class RecordingRandom:
  """Stands in for `random` inside agents/actor.py only: the env keeps the real module."""

  def begin(self, forced=None, time_range=None):
    self.sampled, self.ints = [], []
    self.forced, self.time_range = list(forced or []), time_range

  def sample(self, population, k):
    self.sampled = random.sample(population, k)
    return list(self.sampled)

  def randint(self, a, b):
    if self.forced:
      value = self.forced.pop(0)
    elif a == 1 and self.time_range:
      value = random.randint(self.time_range[0], min(self.time_range[1], b))
    else:
      value = random.randint(a, b)
    self.ints.append(value)
    return value


class FakeWriter:
  """The part of the reverb writer that _write_episode uses."""

  def __init__(self):
    self.steps, self.items = [], []

  def __enter__(self):
    return self

  def __exit__(self, *_):
    return False

  def append(self, data):
    self.steps.append(data)

  @property
  def history(self):
    obs = {k: [s['observation'][k] for s in self.steps] for k in ('nodes', 'edges', 'globals')}
    with_action = [s for s in self.steps if 'action' in s]
    out = {k: [s[k] for s in with_action] for k in ('action', 'reward', 'terminal')}
    out['observation'] = obs
    out['trajectory_stats'] = {k: [v] for k, v in self.steps[0]['trajectory_stats'].items()}
    return out

  def create_item(self, table, priority, trajectory):
    self.items.append(trajectory)

  def end_episode(self):
    pass


class FakeClient:
  def __init__(self):
    self.writer = FakeWriter()

  def trajectory_writer(self, num_keep_alive_refs):
    return self.writer


def written(trajectory, stats, env_config):
  """What the original stores for this episode: per-item action, reward, terminal, stats."""
  client = FakeClient()
  actor._write_episode(trajectory, stats, env_config, None, None, client)
  items = client.writer.items
  assert len(items) == len(trajectory.actions)
  return dict(
    action=np.stack([i['action'] for i in items]),
    reward=np.array([i['reward'] for i in items], dtype=np.float32),
    terminal=np.array([i['terminal'] for i in items], dtype=np.int8),
    stats=np.array([items[0]['trajectory_stats'][k] for k in STAT_NAMES], dtype=np.float32),
  )


def obs_rows(observations):
  """Row arrays from observation dicts. Tip pose as actor.py:325-327 forms it."""
  robots = [o['robots'] for o in observations]
  tip = np.stack([np.matmul(r['base_poses'], r['tip_relative_poses']) for r in robots])
  return dict(
    q=np.stack([r['joint_configurations'] for r in robots]),
    qd=np.stack([r['joint_velocities'] for r in robots]),
    tip_pos=tip[:, :, :3, 3],
    tip_rot=tip[:, :, :3, :3],
    dwell=np.rint(np.stack([r['dwelling'] for r in robots]) * DWELL_STEPS).astype(np.int8),
    done=np.stack([o['targets']['done'] for o in observations]),
    score=np.array([o['current_score'] for o in observations], dtype=np.float32),
    step=np.array([o['step'] for o in observations], dtype=np.int16),
  )


def start_states(num_robots, num_targets, count, first=0):
  path = f'roboballet/data/start_states/{num_robots}r_{num_targets}t_30o.pb'
  out = []
  with open(path, 'rb', buffering=1 << 20) as f:
    while len(out) < first + count:
      assert f.read(1) == b'\x0a'
      length, shift = 0, 0
      while True:
        c = f.read(1)[0]
        length |= (c & 0x7f) << shift
        shift += 7
        if not c & 0x80:
          break
      out.append(planning_env.StartState.from_proto(data_pb2.StartState.FromString(f.read(length))))
  return out[first:]


def make_env_config(num_robots, num_targets, rts=False, deterministic=False):
  cfg = hparams.make_optimised_configs(num_targets=num_targets, num_obstacles=30,
                                       num_robots=num_robots, pregen=True)
  ec = cfg.env_config
  if not rts:
    ec = ec.replace(return_to_start_reward_weight=0.0, return_to_start_required=False)
  else:
    ec = ec.replace(return_to_start_reward_weight=0.25, return_to_start_required=True)
  if deterministic:
    ec = ec.replace(base_config=ec.base_config.replace(deterministic_mode=True))
  return ec


# Policies. Each returns a velocity request in rad/s, like the original's actions.


def uniform_policy(num_robots):
  def policy(t, obs):
    return np.random.uniform(-MAX_VEL, MAX_VEL, size=(num_robots, 7)).astype(np.float32)
  return policy


def wander_policy(rng, num_robots, low, high, arms=None):
  """Tracks home + amp * sin, so with high below 0.25 the arm stays in the takeover box."""
  amp = rng.uniform(low, high, (num_robots, 7))
  period = rng.uniform(25.0, 80.0, (num_robots, 7))
  arms = range(num_robots) if arms is None else arms
  def policy(t, obs):
    action = np.random.uniform(-MAX_VEL, MAX_VEL, size=(num_robots, 7)).astype(np.float32)
    want = HOME + amp * np.sin(2 * np.pi * (t + 1) / period)
    v = np.clip((want - obs['robots']['joint_configurations']) / 0.1, -MAX_VEL, MAX_VEL)
    for r in arms:
      action[r] = v[r]
    return action.astype(np.float32)
  return policy


def returning_policy(rng, num_robots):
  """Goes out, comes back and parks exactly on home: the path to a return-to-start ending."""
  amp = rng.uniform(0.3, 0.8, (num_robots, 7))
  period = rng.uniform(40.0, 80.0, (num_robots, 7))
  back = rng.integers(40, 80)
  def policy(t, obs):
    fade = max(0.0, 1.0 - t / back)
    want = HOME + fade * amp * np.sin(2 * np.pi * (t + 1) / period)
    v = np.clip((want - obs['robots']['joint_configurations']) / 0.1, -MAX_VEL, MAX_VEL)
    return v.astype(np.float32)
  return policy


def replayed_policy(actions):
  def policy(t, obs):
    return actions[t].copy()
  return policy


def rollout(env_config, start_state, policy, tag=0):
  """The stepping of actor.py:run_episode with its stats, minus the network."""
  env = planning_env.PlanningEnv(env_config, start_state=start_state)
  assert env._dwell_steps == DWELL_STEPS
  obs = env.reset().observation
  tr = trajectory_lib.Trajectory(start_state=env.get_start_state(), observations=[], actions=[],
                                 rewards=[], target_states=[])
  total = 0.0
  while not env.terminal():
    action = policy(len(tr.actions), obs)
    tr.observations.append(obs)
    tr.actions.append(np.array(action))
    timestep = env.step(np.array(action))
    tr.rewards.append(timestep.reward)
    obs = timestep.observation
    total += timestep.reward
    tr.target_states.append(env.target_states())
  tr.observations.append(env.observation())
  stats = stats_lib.TrajectoryStats(
    final_score=env.current_score(), total_reward=total,
    total_collision_penalty=env.total_collision_penalty(),
    total_collision_robot_steps=env.total_collision_robot_steps(),
    total_mean_acceleration=env.total_acceleration(), num_targets_done=env.num_targets_done(),
    episode_length=len(tr.observations), env_sampling_time_s=0.25 + tag, episode_wall_time_s=1.5 + tag,
    episode_write_wall_time_s=0.5 + tag, network_step=1000 + tag)
  return tr, stats


def episode_record(tr, stats, env_config, num_targets_to_solve, recorder, forced=None, time_range=None):
  """One real episode, then the original's hindsight on it, as flat arrays."""
  real_written = written(tr, stats, env_config)
  rec = dict(length=len(tr.actions), **obs_rows(tr.observations))
  rec.update(action=real_written['action'], reward=real_written['reward'],
             terminal=real_written['terminal'], stats=real_written['stats'])
  start = tr.start_state
  place = np.asarray(start.initial_robot_placements, dtype=np.float32)
  rec.update(
    base_pos=place[:, :3, 3], base_rot=place[:, :3, :3],
    box_pos=np.array([b.centroid for b in start.generated_obstacles], dtype=np.float32),
    box_rot=np.array([b.rotation for b in start.generated_obstacles], dtype=np.float32),
    box_spans=np.array([b.spans for b in start.generated_obstacles], dtype=np.float32),
    target_pos=np.array([t.position for t in start.targets], dtype=np.float32),
    target_rot=np.array([t.rotation for t in start.targets], dtype=np.float32),
    q0=np.asarray(start.initial_joint_configurations, dtype=np.float32),
  )

  num_targets = len(start.targets)
  raw_actions = [a.copy() for a in tr.actions]
  recorder.begin(forced, time_range)
  out = actor.generate_hindsight_trajectory(tr, stats, env_config, num_targets=num_targets_to_solve)
  chosen = np.zeros(num_targets, dtype=bool)
  arm = np.zeros(num_targets, dtype=np.int64)
  time = np.zeros(num_targets, dtype=np.int64)
  for i, target in enumerate(recorder.sampled):
    chosen[target], arm[target], time[target] = True, recorder.ints[2 * i], recorder.ints[2 * i + 1]
  rec.update(chosen=chosen, arm=arm, time=time, h_length=-1)
  if out is None:
    return rec
  h, h_stats = out
  # The aliasing the port must reproduce: the same array objects, mutated in place.
  assert all(x is y for x, y in zip(h.actions, tr.actions))
  h_written = written(h, h_stats, env_config)
  rows = obs_rows(h.observations)
  # Replayed robot rows equal the real ones: q and qd exactly, tip through f32 rounding.
  a = len(h.actions)
  assert np.array_equal(rows['q'][:a + 1], rec['q'][:a + 1])
  assert np.array_equal(rows['qd'][:a + 1], rec['qd'][:a + 1])
  rec['h_tip_err'] = max(np.abs(rows['tip_pos'][:a + 1] - rec['tip_pos'][:a + 1]).max(),
                         np.abs(rows['tip_rot'][:a + 1] - rec['tip_rot'][:a + 1]).max())
  rec.update(
    h_length=a, h_action=h_written['action'], h_reward=h_written['reward'],
    h_terminal=h_written['terminal'], h_stats=h_written['stats'], h_dwell=rows['dwell'],
    h_done=rows['done'], h_score=rows['score'], h_step=rows['step'],
    h_target_pos=np.array([t.position for t in h.start_state.targets], dtype=np.float32),
    h_target_rot=np.array([t.rotation for t in h.start_state.targets], dtype=np.float32),
    raw_modified=sum(not np.array_equal(x, y) for x, y in zip(h.actions, raw_actions)),
  )
  return rec


def save_case(name, records, num_robots, num_targets, k, rts=False):
  """Stacks records into padded arrays, one file per case."""
  n = len(records)
  height = max(r['length'] for r in records)
  row_keys = ('q', 'qd', 'tip_pos', 'tip_rot', 'dwell', 'done', 'score', 'step')
  step_keys = ('action', 'reward', 'terminal')
  h_row_keys = ('h_dwell', 'h_done', 'h_score', 'h_step')
  h_step_keys = ('h_action', 'h_reward', 'h_terminal')
  flat_keys = ('base_pos', 'base_rot', 'box_pos', 'box_rot', 'box_spans', 'target_pos', 'target_rot',
               'q0', 'chosen', 'arm', 'time', 'stats', 'length', 'h_length')
  out = {k_: np.stack([np.asarray(r[k_]) for r in records]) for k_ in flat_keys}
  for key in row_keys:
    shape = records[0][key].shape[1:]
    out[key] = np.zeros((n, height + 1, *shape), dtype=records[0][key].dtype)
    for i, r in enumerate(records):
      out[key][i, : r['length'] + 1] = r[key]
  for key in step_keys:
    shape = records[0][key].shape[1:]
    out[key] = np.zeros((n, height, *shape), dtype=records[0][key].dtype)
    for i, r in enumerate(records):
      out[key][i, : r['length']] = r[key]
  with_h = [r for r in records if r['h_length'] >= 0]
  for key in h_row_keys:
    shape = with_h[0][key].shape[1:]
    out[key] = np.zeros((n, height + 1, *shape), dtype=with_h[0][key].dtype)
    for i, r in enumerate(records):
      if r['h_length'] >= 0:
        out[key][i, : r['h_length'] + 1] = r[key]
  for key in h_step_keys:
    shape = with_h[0][key].shape[1:]
    out[key] = np.zeros((n, height, *shape), dtype=with_h[0][key].dtype)
    for i, r in enumerate(records):
      if r['h_length'] >= 0:
        out[key][i, : r['h_length']] = r[key]
  out['h_stats'] = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
  out['h_target_pos'] = np.zeros_like(out['target_pos'])
  out['h_target_rot'] = np.zeros_like(out['target_rot'])
  out['raw_modified'] = np.zeros(n, dtype=np.int64)
  for i, r in enumerate(records):
    if r['h_length'] >= 0:
      for key in ('h_stats', 'h_target_pos', 'h_target_rot', 'raw_modified'):
        out[key][i] = r[key]
  tip_err = max(r.get('h_tip_err', 0.0) for r in records)
  out.update(k=k, num_robots=num_robots, num_targets=num_targets, rts=rts, h_tip_err=tip_err)
  path = os.path.join(OUT, f'replay_her_{name}.npz')
  np.savez_compressed(path, **out)
  print(f'{name}: {n} episodes, {len(with_h)} with hindsight, h_len '
        f'{sorted(set(r["h_length"] for r in with_h))[:6]}..., tip err {tip_err:.1e}, '
        f'{os.path.getsize(path) / 1e6:.2f} MB', flush=True)


# Cases.


def case_verify12(recorder):
  """The exact run behind the numbers in algo_spec 5.3: runs/.../algo/verify_her.py."""
  random.seed(1)
  np.random.seed(1)
  ec = make_env_config(4, 10)
  states = start_states(4, 10, 30)
  records = []
  for i in range(12):
    tr, stats = rollout(ec, states[i], uniform_policy(4), tag=i)
    records.append(episode_record(tr, stats, ec, 4, recorder))
  agg = dict(
    neg_actual=int(sum((r['reward'] < 0).sum() for r in records)),
    neg_her=int(sum((r['h_reward'] < 0).sum() for r in records)),
    mutated=int(sum(r['raw_modified'] for r in records)),
    solved=[int(r['h_done'][-1].sum()) for r in records],
    steps=int(sum(r['h_length'] for r in records)),
  )
  print('verify_her figures reproduced here:', agg, flush=True)
  save_case('verify12', records, 4, 10, 4)


def case_random(recorder):
  random.seed(11)
  np.random.seed(11)
  ec = make_env_config(4, 10)
  records = []
  for i, state in enumerate(start_states(4, 10, 36, first=12)):
    tr, stats = rollout(ec, state, uniform_policy(4), tag=i)
    records.append(episode_record(tr, stats, ec, 4, recorder))
  save_case('random', records, 4, 10, 4)


def case_allrelabel(recorder):
  """All ten targets relabelled, so most episodes get everything done before the end."""
  random.seed(12)
  np.random.seed(12)
  ec = make_env_config(4, 10)
  records = []
  for i, state in enumerate(start_states(4, 10, 24, first=60)):
    tr, stats = rollout(ec, state, uniform_policy(4), tag=i)
    span = None if i % 2 else (1, 60 + i)  # Half end much earlier than max_steps.
    records.append(episode_record(tr, stats, ec, 10, recorder, time_range=span))
  save_case('allrelabel', records, 4, 10, 10)


def case_nearhome(recorder):
  """Arms inside the takeover box when everything gets done: takeover, also over final dwell."""
  random.seed(13)
  np.random.seed(13)
  rng = np.random.default_rng(13)
  ec = make_env_config(4, 10)
  records = []
  for i, state in enumerate(start_states(4, 10, 36, first=90)):
    low, high = [(0.05, 0.24), (0.15, 0.245), (0.005, 0.03)][i % 3]
    tr, stats = rollout(ec, state, wander_policy(rng, 4, low, high), tag=i)
    span = None if i % 2 else (1, 50)
    records.append(episode_record(tr, stats, ec, 10, recorder, time_range=span))
  save_case('nearhome', records, 4, 10, 10)


def case_mixed(recorder):
  """Some arms in the box, some not, so takeover and dwell zeroing land on different arms."""
  random.seed(14)
  np.random.seed(14)
  rng = np.random.default_rng(14)
  ec = make_env_config(4, 10)
  records = []
  for i, state in enumerate(start_states(4, 10, 24, first=130)):
    arms = [(0, 1), (2,), (1, 3), (0, 2, 3)][i % 4]
    tr, stats = rollout(ec, state, wander_policy(rng, 4, 0.05, 0.2, arms), tag=i)
    span = None if i % 2 else (1, 70)
    records.append(episode_record(tr, stats, ec, 10, recorder, time_range=span))
  save_case('mixed', records, 4, 10, 10)


def case_rts(recorder):
  """Return-to-start on: all done without ending, takeover for long, parked endings."""
  random.seed(15)
  np.random.seed(15)
  rng = np.random.default_rng(15)
  ec = make_env_config(4, 10, rts=True)
  records = []
  for i, state in enumerate(start_states(4, 10, 24, first=160)):
    policy = returning_policy(rng, 4) if i % 2 else wander_policy(rng, 4, 0.05, 0.24)
    tr, stats = rollout(ec, state, policy, tag=i)
    records.append(episode_record(tr, stats, ec, 10, recorder, time_range=(1, 40)))
  save_case('rts', records, 4, 10, 10, rts=True)


def case_solved(recorder):
  """Real episodes that solve real targets: below k, at or above k, and all of them."""
  random.seed(16)
  np.random.seed(16)
  ec = make_env_config(4, 10, deterministic=True)
  records = []
  plan = [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3, 4, 5, 7, 10, 10, 10] * 2
  for i, state in enumerate(start_states(4, 10, len(plan), first=190)):
    # Fixed actions from one rollout. Each target goes on a tip pose of the latest rollout,
    # because a real solve freezes that arm for the dwell and so changes its later path.
    tr, _ = rollout(ec, state, uniform_policy(4), tag=i)
    actions = [a.copy() for a in tr.actions]
    solved = plan[i]
    arms = [(j + i) % 4 for j in range(solved)]
    times = [8 + 8 * j + (i % 5) for j in range(solved)]
    moved = copy.deepcopy(state)
    for j in range(solved):
      o = tr.observations[times[j]]['robots']
      pose = np.matmul(o['base_poses'], o['tip_relative_poses'])[arms[j]]
      moved.targets[j] = target_lib.TargetInfo(position=pose[:3, 3], rotation=pose[:3, :3])
      tr, stats = rollout(ec, moved, replayed_policy(actions), tag=i)
    assert int(tr.observations[-1]['targets']['done'].sum()) >= solved - 1
    forced = None
    if i % 4 == 3 and solved == 3:
      # A relabelled target 2 steps before a real solve of the same arm: the arm is dwelling
      # at the real solve, so the real target is missed and hindsight falls short of k.
      forced = [arms[0], times[0] - 2]
    records.append(episode_record(tr, stats, ec, 4, recorder, forced=forced))
  save_case('solved', records, 4, 10, 4)


def case_8r(recorder):
  random.seed(17)
  np.random.seed(17)
  rng = np.random.default_rng(17)
  ec = make_env_config(8, 10)
  records = []
  for i, state in enumerate(start_states(8, 10, 12)):
    policy = wander_policy(rng, 8, 0.05, 0.2, range(0, 8, 2)) if i % 2 else uniform_policy(8)
    tr, stats = rollout(ec, state, policy, tag=i)
    records.append(episode_record(tr, stats, ec, 10, recorder, time_range=(1, 60)))
  save_case('8r10t', records, 8, 10, 10)


def case_40t(recorder):
  random.seed(18)
  np.random.seed(18)
  ec = make_env_config(4, 40)
  records = []
  for i, state in enumerate(start_states(4, 40, 6)):
    tr, stats = rollout(ec, state, uniform_policy(4), tag=i)
    records.append(episode_record(tr, stats, ec, 16, recorder))
  save_case('4r40t', records, 4, 40, 16)


def case_small(recorder, num_targets):
  """Few targets from a 10-target start state: short episodes, frequent early endings."""
  random.seed(19 + num_targets)
  np.random.seed(19 + num_targets)
  ec = make_env_config(4, num_targets)
  records = []
  for i, state in enumerate(start_states(4, 10, 12, first=230)):
    state = dataclasses.replace(state, targets=list(state.targets[:num_targets]))
    tr, stats = rollout(ec, state, uniform_policy(4), tag=i)
    records.append(episode_record(tr, stats, ec, num_targets, recorder))
  save_case(f'4r{num_targets}t', records, 4, num_targets, num_targets)


CASES = {
  'verify12': case_verify12,
  'random': case_random,
  'allrelabel': case_allrelabel,
  'nearhome': case_nearhome,
  'mixed': case_mixed,
  'rts': case_rts,
  'solved': case_solved,
  '8r10t': case_8r,
  '4r40t': case_40t,
  '4r1t': lambda r: case_small(r, 1),
  '4r3t': lambda r: case_small(r, 3),
}


def main():
  # The writer only forwards features, and computing them would take most of the run.
  empty = types.SimpleNamespace(nodes=0, edges=0, globals=0)
  actor.compute_features.make_graph_features = lambda *a, **k: empty
  recorder = RecordingRandom()
  actor.random = recorder
  for name in sys.argv[1:] or CASES:
    CASES[name](recorder)


if __name__ == '__main__':
  main()
