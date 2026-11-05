"""Episode buffer and replay ring, against a slow pure-Python model and the real env.

Synthetic episodes carry a deterministic value in every field, computed from (episode, row),
so a stored or sampled row is checked by recomputing what it must be. Rows past an episode's
end are filled with junk, so that storing one shows up.

The throughput tests print and assert nothing else. Run them on an idle GPU, with -s:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m pytest -s tests/test_replay.py -k throughput
"""

from __future__ import annotations

import collections
import time

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.config import TrainCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.replay.hindsight import HindsightQueue
from roboballet_mjlab.replay.replay import (
  STAT,
  STAT_NAMES,
  EpisodeBatch,
  EpisodeBuffer,
  ReplayBuffer,
  cat_episodes,
  row_storage,
  select_episodes,
)
from roboballet_mjlab.structs import Obs, RobotState, Scene
from tests.test_sim_common import DEVICE, env_cfg, stored_scenes

R, O, T = 2, 3, 3
CFG = env_cfg(num_robots=R, num_obstacles=O, num_targets=T)
JUNK = -77.0


def _wave(base: torch.Tensor, *shape: int) -> torch.Tensor:
  """Distinct, deterministic f32 values for each base value and each element of `shape`."""
  count = int(np.prod(shape)) if shape else 1
  phase = torch.arange(count, device=DEVICE, dtype=torch.float64).view(shape) * 0.731
  base = base.double().view(*base.shape, *([1] * len(shape)))
  return torch.sin(base * 0.137 + phase).float()


def rows_of(e: torch.Tensor, t: torch.Tensor) -> RobotState:
  """The stored state of episode e at row t, in the wide dtypes the replay hands out."""
  tag = (e * 1024 + t).float()
  q = _wave(tag, R, 7)
  q[..., 0, 0] = tag  # Exact, so that a sample can be identified from q alone.
  bits = (e * 1024 + t)[..., None]
  return RobotState(
    q=q,
    qd=_wave(tag + 0.25, R, 7),
    dwell=(bits % 6).expand(*e.shape, R).long(),
    tip_pos=_wave(tag + 0.5, R, 3),
    tip_rot=_wave(tag + 0.75, R, 3, 3),
    target_done=(bits.expand(*e.shape, T) >> torch.arange(T, device=DEVICE)) % 2 == 1,
    score=_wave(tag + 0.125),
    step=t.long(),
    aug_code=(bits % 4).expand(*e.shape, O).long(),
  )


def action_of(e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  return _wave((e * 1024 + t).float() + 0.375, R, 7)


def reward_of(e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  return _wave((e * 1024 + t).float() + 0.625) + 2.0  # Never zero, the final reward included.


def scene_of(e: torch.Tensor) -> Scene:
  x = e.float()
  return Scene(_wave(x + 0.1, R, 3), _wave(x + 0.2, R, 3, 3), _wave(x + 0.3, O, 3), _wave(x + 0.4, O, 3, 3),
               _wave(x + 0.5, O, 3), _wave(x + 0.6, T, 3), _wave(x + 0.7, T, 3, 3), _wave(x + 0.8, R, 7))


def behaviour_of(e: torch.Tensor) -> torch.Tensor:
  return e * 3 + 1


def hindsight_of(e: torch.Tensor) -> torch.Tensor:
  return e % 3 == 0


def stats_of(e: torch.Tensor) -> torch.Tensor:
  return _wave(e.float() + 0.9, len(STAT_NAMES))


def make_batch(ids: list[int], lengths: list[int], height: int | None = None) -> EpisodeBatch:
  e = torch.tensor(ids, device=DEVICE)
  length = torch.tensor(lengths, device=DEVICE)
  H = max(lengths) + 1 if height is None else height
  t = torch.arange(H, device=DEVICE)[None].expand(len(ids), H)
  grid = e[:, None].expand(len(ids), H)
  out = row_storage(CFG, (len(ids), H), DEVICE)
  past = (t > length[:, None])
  for store, value in zip(out, rows_of(grid, t)):
    store.copy_(value)
    junk = JUNK if store.is_floating_point() else (True if store.dtype == torch.bool else 127)
    store.masked_fill_(past.view(*past.shape, *([1] * (store.dim() - 2))), junk)
  action = action_of(grid[:, :-1], t[:, :-1]).masked_fill(~(t[:, :-1] < length[:, None])[..., None, None], JUNK)
  reward = reward_of(grid[:, :-1], t[:, :-1]).masked_fill(~(t[:, :-1] < length[:, None]), JUNK)
  return EpisodeBatch(out, action, reward, length, scene_of(e), behaviour_of(e), hindsight_of(e), stats_of(e))


class SlowReplay:
  """The contract in plain Python lists, recomputed from scratch at every insert."""

  def __init__(self, capacity: int, max_episodes: int) -> None:
    self.capacity, self.max_episodes = capacity, max_episodes
    self.transitions: list[tuple[int, int, int]] = []  # (episode, row, actions), every one ever.
    self.episode_first: list[int] = []
    self.evicted = self.extra = 0

  def insert(self, ids: list[int], lengths: list[int]) -> None:
    for e, a in zip(ids, lengths):
      self.episode_first.append(len(self.transitions))
      self.transitions.extend((e, t, a) for t in range(a))
    by_capacity = max(self.evicted, len(self.transitions) - self.capacity)
    head = by_capacity
    if len(self.episode_first) > self.max_episodes:
      head = max(head, self.episode_first[-self.max_episodes])
    self.extra += head - by_capacity
    self.evicted = head

  @property
  def live(self) -> list[tuple[int, int, int]]:
    return self.transitions[self.evicted :]


def assert_robot_equal(got: RobotState, want: RobotState) -> None:
  for name, a, b in zip(RobotState._fields, got, want):
    assert a.dtype == b.dtype, name
    assert torch.equal(a, b), name


def assert_transitions(sample, live: list[tuple[int, int, int]]) -> None:
  """`sample` must be exactly these transitions, field by field, in this order."""
  e, t, a = torch.tensor(live, device=DEVICE).unbind(-1)
  assert_robot_equal(sample.obs.robot, rows_of(e, t))
  assert_robot_equal(sample.next_obs.robot, rows_of(e, t + 1))
  assert torch.equal(sample.action, action_of(e, t))
  assert torch.equal(sample.reward, reward_of(e, t))
  assert torch.equal(sample.terminal, t == a - 1)
  for got, want in zip(sample.obs.scene, scene_of(e)):
    assert torch.equal(got, want)
  for got, want in zip(sample.next_obs.scene, scene_of(e)):
    assert torch.equal(got, want)
  assert torch.equal(sample.behaviour_step, behaviour_of(e))
  assert torch.equal(sample.is_hindsight, hindsight_of(e))
  assert torch.equal(sample.stats, stats_of(e))


def identify(sample) -> tuple[list[int], list[int]]:
  tag = sample.obs.robot.q[:, 0, 0].long().tolist()
  return [x // 1024 for x in tag], [x % 1024 for x in tag]


def check_against_model(replay: ReplayBuffer, model: SlowReplay, gen: torch.Generator, coverage: bool) -> None:
  counters = replay.counters()
  assert counters['inserted'] == len(model.transitions)
  assert counters['evicted'] == model.evicted
  assert counters['extra_evicted'] == model.extra
  assert counters['size'] == len(model.live)
  live = model.live
  if not live:
    return
  # Every live transition, exactly, in FIFO order.
  assert_transitions(replay.fetch(torch.arange(len(live), device=DEVICE)), live)
  # Samples: each one must be a live transition with the right fields. Identify, then check.
  n = 30 * len(live) if coverage else max(256, len(live))
  sampled = replay.sample(n, gen)
  ids = identify(sampled)
  length_of = {(e, t): a for e, t, a in live}
  seen = list(zip(*ids))
  for key in set(seen):
    assert key in length_of, f'{key} is not a live transition: final state or evicted'
  assert_transitions(sampled, [(e, t, length_of[(e, t)]) for e, t in seen])
  if coverage:
    assert set(seen) == set(length_of), 'a live transition was never sampled'


LENGTH_MODES = {
  'mixed': lambda rng: int(rng.integers(1, 40)),
  'short': lambda rng: int(rng.integers(1, 4)),
  'ones': lambda rng: 1,
  'bimodal': lambda rng: 1 if rng.random() < 0.5 else 49,
}


@pytest.mark.parametrize('capacity', [1, 2, 3, 5, 7, 8, 16, 31, 64, 257, 1000])
@pytest.mark.parametrize('mode', LENGTH_MODES)
@pytest.mark.parametrize('tables', ['default', 'tight'])
def test_replay_matches_slow_model(capacity, mode, tables):
  rng = np.random.default_rng(capacity * 31 + len(mode))
  gen = torch.Generator(device=DEVICE).manual_seed(capacity)
  max_episodes = None if tables == 'default' else max(1, min(3, capacity))
  replay = ReplayBuffer(CFG, capacity, DEVICE, max_episodes=max_episodes)
  model = SlowReplay(capacity, replay.max_episodes)
  next_id = 0
  for i in range(14):
    count = int(rng.integers(1, 7))
    lengths = [LENGTH_MODES[mode](rng) for _ in range(count)]
    ids = list(range(next_id, next_id + count))
    next_id += count
    replay.insert(make_batch(ids, lengths))
    model.insert(ids, lengths)
    check_against_model(replay, model, gen, coverage=i % 7 == 6 and len(model.live) <= 512)


def test_one_action_episode_and_nonzero_final_reward():
  replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=4)
  replay.insert(make_batch([5], [1]))
  assert replay.size == 1
  e, t = torch.tensor([5], device=DEVICE), torch.tensor([0], device=DEVICE)
  sample = replay.sample(16, torch.Generator(device=DEVICE).manual_seed(0))
  assert sample.terminal.all()
  assert torch.equal(sample.obs.robot.q, rows_of(e, t).q.expand(16, R, 7))
  assert torch.equal(sample.next_obs.robot.q, rows_of(e, t + 1).q.expand(16, R, 7))
  assert (sample.reward == reward_of(e, t)).all() and (sample.reward != 0).all()


def test_wraparound_across_terminal_boundary():
  # 8 + 2 = 10 rows. Varying episode 0 makes the ring split episode 1 at every possible row.
  for first_rows in range(2, 10):
    replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=2)
    replay.insert(make_batch([0], [first_rows - 1]))
    assert replay._next_row == first_rows
    replay.insert(make_batch([1], [6]))
    live = [(1, t, 6) for t in range(6)]
    if first_rows - 1 + 6 <= 8:
      live = [(0, t, first_rows - 1) for t in range(first_rows - 1)] + live
    else:
      live = [(0, t, first_rows - 1) for t in range(first_rows - 1)][first_rows - 1 + 6 - 8 :] + live
    assert replay.size == len(live)
    assert_transitions(replay.fetch(torch.arange(len(live), device=DEVICE)), live)


def test_terminal_row_on_the_last_ring_row_has_its_successor_at_row_zero():
  replay = ReplayBuffer(CFG, 4, DEVICE, max_episodes=2)  # 6 rows.
  replay.insert(make_batch([0], [4]))  # Rows 0 to 4.
  replay.insert(make_batch([1], [4]))  # Rows 5, 0, 1, 2, 3: the terminal row is 2.
  assert_transitions(replay.fetch(torch.arange(4, device=DEVICE)), [(1, t, 4) for t in range(4)])
  replay = ReplayBuffer(CFG, 5, DEVICE, max_episodes=1)  # 6 rows.
  replay.insert(make_batch([0], [5]))  # Final row is the last row of the ring.
  replay.insert(make_batch([1], [1]))  # Rows 0 and 1: replaces the only table slot.
  assert_transitions(replay.fetch(torch.arange(1, device=DEVICE)), [(1, 0, 1)])
  assert replay.extra_evicted == 4


@pytest.mark.parametrize('n_step', [2, 3, 5, 9])
def test_n_step_matches_a_slow_sum_across_ring_splits_and_eviction(n_step):
  gamma = 0.94
  # 8 + 2 = 10 rows, as in the wraparound test, so the ring splits episode 1 at every possible row.
  for first_rows in range(2, 10):
    replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=2)
    replay.insert(make_batch([0], [first_rows - 1]))
    replay.insert(make_batch([1], [6]))
    live = [(0, t, first_rows - 1) for t in range(first_rows - 1)] + [(1, t, 6) for t in range(6)]
    live = live[-replay.size :]
    got = replay.fetch(torch.arange(len(live), device=DEVICE), n_step, gamma)
    for i, (e, t, length) in enumerate(live):
      k = min(n_step, length - t)
      episode, times = torch.tensor([e], device=DEVICE), torch.arange(t, t + k, device=DEVICE)
      rewards = reward_of(episode.expand(k), times)
      want = (rewards * gamma ** torch.arange(k, device=DEVICE, dtype=rewards.dtype)).sum()
      assert torch.isclose(got.reward[i], want, rtol=1e-6), (first_rows, e, t)
      assert int(got.steps[i]) == k
      assert bool(got.terminal[i]) == (t + k == length)
      state = rows_of(episode, torch.tensor([t + k], device=DEVICE))
      for name, field in state._asdict().items():
        assert torch.equal(getattr(got.next_obs.robot, name)[i], field[0].to(getattr(got.next_obs.robot, name).dtype)), name
      assert torch.equal(got.action[i], action_of(episode, torch.tensor([t], device=DEVICE))[0])


def test_n_step_of_one_leaves_the_sample_as_it_was():
  replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=2)
  replay.insert(make_batch([0], [5]))
  got = replay.fetch(torch.arange(5, device=DEVICE))
  assert got.steps is None
  assert_transitions(got, [(0, t, 5) for t in range(5)])


def test_partial_eviction_keeps_successor_and_scene():
  replay = ReplayBuffer(CFG, 6, DEVICE, max_episodes=4)
  replay.insert(make_batch([0], [10]))
  live = [(0, t, 10) for t in range(4, 10)]
  assert replay.size == 6 and replay.evicted == 4 and replay.extra_evicted == 0
  assert_transitions(replay.fetch(torch.arange(6, device=DEVICE)), live)
  replay.insert(make_batch([1], [2]))
  live = [(0, t, 10) for t in range(6, 10)] + [(1, 0, 2), (1, 1, 2)]
  assert replay.evicted == 6
  assert_transitions(replay.fetch(torch.arange(6, device=DEVICE)), live)
  gen = torch.Generator(device=DEVICE).manual_seed(0)
  rows = set(zip(*identify(replay.sample(600, gen))))
  assert rows == {(e, t) for e, t, _ in live}, 'evicted and final rows must not appear'


def test_row_ring_pressure_counts_extra_evictions():
  replay = ReplayBuffer(CFG, 16, DEVICE, max_episodes=2)  # 18 rows.
  for i in range(10):
    replay.insert(make_batch([i], [1]))
  assert replay.size == 2 and replay.extra_evicted == 8
  assert_transitions(replay.fetch(torch.arange(2, device=DEVICE)), [(8, 0, 1), (9, 0, 1)])


def test_default_slack_only_runs_out_below_mean_length_8():
  for length, expect_extra in [(8, False), (7, True), (3, True), (49, False)]:
    replay = ReplayBuffer(CFG, 1600, DEVICE)
    assert replay.num_rows == 1800
    for i in range(16):
      replay.insert(make_batch(list(range(i * 64, (i + 1) * 64)), [length] * 64))
    assert (replay.extra_evicted > 0) == expect_extra, length


def test_batch_larger_than_the_ring():
  replay = ReplayBuffer(CFG, 4, DEVICE, max_episodes=2)
  model = SlowReplay(4, 2)
  gen = torch.Generator(device=DEVICE).manual_seed(0)
  for ids, lengths in [([0, 1, 2], [9, 1, 2]), ([3, 4], [20, 3]), ([5, 6, 7, 8], [1, 1, 1, 1]), ([9], [30])]:
    replay.insert(make_batch(ids, lengths))
    model.insert(ids, lengths)
    check_against_model(replay, model, gen, coverage=True)


def test_fifo_counters_and_statistics():
  replay = ReplayBuffer(CFG, 20, DEVICE, max_episodes=4)
  gen = torch.Generator(device=DEVICE).manual_seed(0)
  assert not replay.ready(1) and replay.ready(0)
  with pytest.raises(ValueError):
    replay.sample(1, gen)
  replay.insert(make_batch([1, 2], [5, 5]))  # is_hindsight false for both.
  replay.insert(make_batch([3], [15]))  # is_hindsight true: e % 3 == 0.
  assert replay.counters() == dict(inserted=25, evicted=5, extra_evicted=0, sampled=0, size=20, episodes=3)
  assert replay.ready(20) and not replay.ready(21)
  sampled = replay.sample(4000, gen)
  replay.sample(96, gen)
  assert replay.sampled == 4096
  oldest = rows_of(torch.tensor([2], device=DEVICE), torch.tensor([0], device=DEVICE))
  assert torch.equal(replay.fetch(torch.tensor([0], device=DEVICE)).obs.robot.q, oldest.q)
  # Diagnostics average over sampled transitions, train/train.py:764-767, not over episodes.
  assert abs(sampled.is_hindsight.float().mean().item() - 15 / 20) < 0.03
  live = [(2, t, 5) for t in range(5)] + [(3, t, 15) for t in range(15)]
  want = torch.stack([stats_of(torch.tensor([e], device=DEVICE))[0] for e, _, _ in live]).mean(0)
  assert (sampled.stats.mean(0) - want).abs().max() < 0.05
  assert sampled.behaviour_step.min() == behaviour_of(torch.tensor(2))
  assert sampled.behaviour_step.max() == behaviour_of(torch.tensor(3))


def test_sampling_is_uniform_over_transitions():
  replay = ReplayBuffer(CFG, 90, DEVICE, max_episodes=30)
  replay.insert(make_batch(list(range(30)), [1, 2, 3, 4, 5] * 6))
  replay.insert(make_batch([30, 31], [1, 44]))
  size = replay.size
  assert size == 90
  n = 90_000
  e, t = identify(replay.sample(n, torch.Generator(device=DEVICE).manual_seed(3)))
  counts = np.array(list(collections.Counter(zip(e, t)).values()))
  assert len(counts) == size
  chi2 = ((counts - n / size) ** 2 / (n / size)).sum()
  assert chi2 < size * 1.5, chi2  # About 134 is the 0.1% point for 89 degrees of freedom.


def test_returned_tensors_do_not_alias_storage():
  replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=4)
  gen = torch.Generator(device=DEVICE).manual_seed(0)
  batch = make_batch([0, 1], [3, 5])
  replay.insert(batch)
  snapshots = []
  for n in (1, 2, 64):
    sample = replay.sample(n, gen)
    leaves = [*sample.obs.robot, *sample.obs.scene, *sample.next_obs.robot, *sample.next_obs.scene,
              sample.action, sample.reward, sample.terminal, sample.behaviour_step, sample.is_hindsight, sample.stats]
    snapshots.append((leaves, [x.clone() for x in leaves]))
  whole = replay.fetch(torch.arange(replay.size, device=DEVICE))
  whole_leaves = [*whole.obs.robot, whole.action, whole.reward]
  whole_before = [x.clone() for x in whole_leaves]
  for i in range(2, 30):
    replay.insert(make_batch([i], [3]))
  for leaves, before in snapshots:
    for now, then in zip(leaves, before):
      assert torch.equal(now, then)
  for now, then in zip(whole_leaves, whole_before):
    assert torch.equal(now, then)
  # And in the other direction: the replay copied what it was given.
  replay = ReplayBuffer(CFG, 8, DEVICE, max_episodes=4)
  batch = make_batch([0], [3])
  replay.insert(batch)
  for leaf in [*batch.robot, batch.action, batch.reward, *batch.scene, batch.stats]:
    leaf.fill_(0)
  assert_transitions(replay.fetch(torch.arange(3, device=DEVICE)), [(0, t, 3) for t in range(3)])


def test_compact_dtypes_keep_exact_values():
  long_cfg = env_cfg(num_robots=R, num_obstacles=O, num_targets=40)
  assert long_cfg.max_steps == 420
  store = row_storage(long_cfg, (1,), DEVICE)
  assert store.step.dtype == torch.int16
  assert row_storage(env_cfg(num_robots=R, num_obstacles=O), (1,), DEVICE).step.dtype == torch.int8
  assert store.dwell.dtype == torch.int8 and store.aug_code.dtype == torch.int8
  assert store.target_done.dtype == torch.bool and store.q.dtype == torch.float32

  replay = ReplayBuffer(long_cfg, 500, DEVICE)
  n, A = 2, 420
  batch = EpisodeBatch(
    row_storage(long_cfg, (n, A + 1), DEVICE), torch.zeros(n, A, R, 7, device=DEVICE), torch.zeros(n, A, device=DEVICE),
    torch.tensor([A, 80], device=DEVICE),
    Scene(*[torch.zeros(n, *shape, device=DEVICE) for shape in
            [(R, 3), (R, 3, 3), (O, 3), (O, 3, 3), (O, 3), (40, 3), (40, 3, 3), (R, 7)]]),
    torch.zeros(n, dtype=torch.long, device=DEVICE), torch.zeros(n, dtype=torch.bool, device=DEVICE),
    torch.zeros(n, len(STAT_NAMES), device=DEVICE))
  batch.robot.step.copy_(torch.arange(A + 1, device=DEVICE))
  replay.insert(batch)
  got = replay.fetch(torch.arange(replay.size, device=DEVICE))
  assert replay.size == 500 and got.obs.robot.step.max() == 419 and got.next_obs.robot.step.max() == 420
  for leaf in (got.obs.robot.dwell, got.obs.robot.step, got.obs.robot.aug_code):
    assert leaf.dtype == torch.long
  assert got.obs.robot.target_done.dtype == torch.bool


# The real env. T = 1 keeps episodes at 30 steps, and a target on a home tip pose ends one in 6.


def _short_pool(num_scenes: int) -> Scene:
  pool = stored_scenes(4)
  return Scene(*[x[:num_scenes] for x in pool._replace(target_pos=pool.target_pos[:, :1], target_rot=pool.target_rot[:, :1])])


def _env_with_early_endings(num_worlds: int) -> tuple[BalletEnv, Scene]:
  cfg = env_cfg(num_targets=1)
  pool = _short_pool(num_worlds)
  env = BalletEnv(cfg, num_worlds, pool, DEVICE, seed=3)
  env.reset(scene_ids=torch.arange(num_worlds, device=DEVICE))
  tip_pos, tip_rot = env.world.tcp()
  # Every third scene has its target on a home tip pose: solved at once, ends after the dwell.
  on_tip = torch.arange(num_worlds, device=DEVICE) % 3 == 0
  target_pos = torch.where(on_tip[:, None, None], tip_pos[:, 1:2], pool.target_pos)
  target_rot = torch.where(on_tip[:, None, None, None], tip_rot[:, 1:2], pool.target_rot)
  pool = pool._replace(target_pos=target_pos, target_rot=target_rot)
  env = BalletEnv(cfg, num_worlds, pool, DEVICE, seed=4)
  env.reset(scene_ids=torch.arange(num_worlds, device=DEVICE))
  return env, pool


def test_episode_buffer_matches_a_slow_trace_of_the_env():
  W = 12
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(5)
  traces = [[] for _ in range(W)]
  finished = 0
  lengths = set()
  for step in range(200):
    obs = env.observe()
    action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    out = env.step(action)
    batch = buf.append(obs, action, out, behaviour_step=1000 + step)
    for w in range(W):
      traces[w].append((RobotState(*[x[w].clone() for x in obs.robot]), action[w].clone(), out.reward[w].clone(),
                        out.collision_penalty[w].clone(), out.collision_robot_steps[w].clone(), out.acceleration[w].clone(),
                        1000 + step, Scene(*[x[w].clone() for x in obs.scene])))
    ended = out.terminal.nonzero().flatten().tolist()
    assert (batch is None) == (not ended)
    if not ended:
      continue
    assert batch.robot.q.shape[1] == env.cfg.max_steps + 1
    for i, w in enumerate(ended):
      trace = traces[w]
      A = len(trace)
      assert batch.length[i] == A
      lengths.add(A)
      for name in RobotState._fields:
        stored = getattr(batch.robot, name)[i]
        for t, row in enumerate(trace):
          assert torch.equal(stored[t].to(getattr(row[0], name).dtype), getattr(row[0], name)), (name, t)
        # The final row is the terminal state from before the auto-reset, not the new episode.
        final = getattr(out.next_robot, name)[w]
        assert torch.equal(stored[A].to(final.dtype), final), name
      assert batch.robot.step[i, A] == A and env.observe().robot.step[w] == 0
      for t, row in enumerate(trace):
        assert torch.equal(batch.action[i, t], row[1])
        assert torch.equal(batch.reward[i, t], row[2])
      for got, want in zip(batch.scene, trace[0][7]):
        assert torch.equal(got[i], want)
      assert batch.behaviour_step[i] == trace[0][6], 'latched at the first row'
      s = batch.stats[i]
      assert s[STAT['final_score']] == out.next_robot.score[w]
      assert (s[STAT['total_reward']] - sum(r[2] for r in trace)).abs() < 1e-5
      assert (s[STAT['total_collision_penalty']] - sum(r[3] for r in trace)).abs() < 1e-5
      assert s[STAT['total_collision_robot_steps']] == sum(r[4] for r in trace)
      assert (s[STAT['total_mean_acceleration']] - sum(r[5] for r in trace)).abs() < 1e-4
      assert s[STAT['num_targets_done']] == out.next_robot.target_done[w].sum()
      assert s[STAT['episode_length']] == A + 1
      assert s[STAT['episode_wall_time_s']] > 0
      assert s[STAT['env_sampling_time_s']] == 0 and s[STAT['episode_write_wall_time_s']] == 0
      assert s[STAT['is_hindsight_replay']] == 0 and s[STAT['network_step']] == trace[0][6]
      assert not batch.is_hindsight[i]
      traces[w] = []
      finished += 1
  assert finished >= 2 * W and {6, 30} <= lengths, (finished, lengths)


@pytest.mark.parametrize('num_worlds', [2, 6])
def test_returned_episodes_survive_later_appends(num_worlds):
  env, _ = _env_with_early_endings(num_worlds)
  buf = EpisodeBuffer(env.cfg, num_worlds, DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(6)
  kept = []
  last = torch.tensor([num_worlds - 1], device=DEVICE)
  for step in range(70):
    if step == 5:  # Staggered, so that batches of a single episode occur.
      env.reset(last)
      buf.discard(last)
    obs = env.observe()
    action = torch.rand(num_worlds, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    batch = buf.append(obs, action, env.step(action))
    if batch is not None:
      leaves = [*batch.robot, batch.action, batch.reward, batch.length, *batch.scene, batch.stats, batch.behaviour_step]
      kept.append((leaves, [x.clone() for x in leaves]))
  assert len(kept) >= 2 and any(len(leaves[-1]) == 1 for leaves, _ in kept)
  for leaves, before in kept:
    for leaf, then in zip(leaves, before):
      assert torch.equal(leaf, then)


def test_discard_restarts_an_episode():
  W = 4
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(7)
  world = torch.tensor([1], device=DEVICE)
  for step in range(4):
    obs = env.observe()
    action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    buf.append(obs, action, env.step(action), behaviour_step=step)
  env.reset(world)
  buf.discard(world)
  for step in range(40):
    obs = env.observe()
    action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    batch = buf.append(obs, action, env.step(action), behaviour_step=100 + step)
    if batch is not None and (env.observe().robot.step == 0).any():
      found = (batch.behaviour_step >= 100).nonzero().flatten()
      for i in found.tolist():
        row_steps = batch.robot.step[i, : batch.length[i] + 1]
        assert torch.equal(row_steps.long(), torch.arange(len(row_steps), device=DEVICE))
      if len(found):
        return
  raise AssertionError('the discarded world never finished')


def test_env_to_replay_end_to_end():
  """Rows from the env go in and come out as the same s, s', and features build from them."""
  W = 16
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  replay = ReplayBuffer(env.cfg, 2000, DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(8)
  # Every first row has q at home, so the per-observation aug codes make the key unique.
  key = lambda robot, i: (robot.q[i].cpu().numpy().tobytes(), robot.aug_code[i].cpu().numpy().tobytes())
  records = {}
  finished = 0
  for step in range(100):
    obs = env.observe()
    action = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    out = env.step(action)
    for w in range(W):
      records[key(obs.robot, w)] = (
        RobotState(*[x[w].cpu() for x in out.next_robot]), action[w].cpu(), out.reward[w].cpu(),
        out.terminal[w].cpu(), Obs(RobotState(*[x[w].cpu() for x in obs.robot]), Scene(*[x[w].cpu() for x in obs.scene])))
    batch = buf.append(obs, action, out)
    if batch is not None:
      replay.insert(batch)
      finished += 1
  assert finished > 0
  sample = replay.fetch(torch.arange(replay.size, device=DEVICE))
  topo = Topology(env.cfg, DEVICE)
  graph = build_graph(sample.obs, topo)
  next_graph = build_graph(sample.next_obs, topo)
  assert graph.nodes.isfinite().all() and next_graph.edges.isfinite().all()
  # What the env itself showed, found by each stored row's key, against what the ring gives back.
  live = [records[key(sample.obs.robot, i)] for i in range(replay.size)]
  stack = lambda rows: [torch.stack(x).to(DEVICE) for x in zip(*rows)]
  live_scene = Scene(*stack([record[4].scene for record in live]))
  live_graph = build_graph(Obs(RobotState(*stack([record[4].robot for record in live])), live_scene), topo)
  live_next_graph = build_graph(Obs(RobotState(*stack([record[0] for record in live])), live_scene), topo)
  for name in graph._fields:
    assert torch.equal(getattr(graph, name), getattr(live_graph, name)), f's: {name}'
    assert torch.equal(getattr(next_graph, name), getattr(live_next_graph, name)), f"s': {name}"
  terminals = 0
  for i in range(replay.size):
    next_state, action, reward, terminal, source = records[key(sample.obs.robot, i)]
    for name in RobotState._fields:
      assert torch.equal(getattr(sample.obs.robot, name)[i].cpu(), getattr(source.robot, name)), name
      assert torch.equal(getattr(sample.next_obs.robot, name)[i].cpu(), getattr(next_state, name)), name
    assert torch.equal(sample.action[i].cpu(), action)
    assert sample.reward[i].cpu() == reward and sample.terminal[i].cpu() == terminal
    for got, want in zip(sample.obs.scene, source.scene):
      assert torch.equal(got[i].cpu(), want)
    terminals += int(terminal)
  assert terminals > 0


# Throughput. Everything has the real shapes of 4 robots, 10 targets, 30 obstacles.


def _timed(fn, repeats: int) -> float:
  """Mean milliseconds per call, synchronized on both sides."""
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(repeats):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / repeats * 1e3


@pytest.mark.parametrize('num_worlds', [64, 1024, 4096])
def test_throughput(num_worlds):
  torch.cuda.empty_cache()
  cfg = env_cfg()
  train = TrainCfg()
  env = BalletEnv(cfg, num_worlds, stored_scenes(4), DEVICE, seed=9)
  buf = EpisodeBuffer(cfg, num_worlds, DEVICE)
  capacity = train.replay_capacity(cfg.num_targets)
  replay = ReplayBuffer(cfg, capacity, DEVICE)
  gen = torch.Generator(device=DEVICE).manual_seed(10)
  # One real episode from all worlds, with only the append timed.
  batches = []
  append_ms = 0.0
  torch.cuda.synchronize()
  for _ in range(cfg.max_steps):
    obs = env.observe()
    action = torch.rand(num_worlds, 4, 7, generator=gen, device=DEVICE) * 2 - 1
    out = env.step(action)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch = buf.append(obs, action, out)
    torch.cuda.synchronize()
    append_ms += time.perf_counter() - start
    if batch is not None:
      batches.append(batch)
  append_ms = append_ms / cfg.max_steps * 1e3
  assert len(batches) == 1 and len(batches[0].length) == num_worlds, 'all worlds run in step here'
  full = batches[0]

  # Inserting W episodes at once, the worst case of aligned resets, and W / A at a time, the
  # steady state of staggered resets.
  insert_all = _timed(lambda: replay.insert(full), 8 if num_worlds > 1024 else 32)
  each = max(1, num_worlds // cfg.max_steps)
  small = select_episodes(full, torch.arange(each, device=DEVICE))
  insert_small = _timed(lambda: replay.insert(small), 64)

  # Fill to capacity so that sampling and insertion see a full ring.
  while replay.size < capacity:
    replay.insert(full)
  evicting_small = _timed(lambda: replay.insert(small), 64)

  results = {}
  for flush in (256, 4096):
    # Exactly `flush` real episodes, repeating this run's if W is smaller.
    repeats = (flush + num_worlds - 1) // num_worlds
    episodes = select_episodes(cat_episodes([full] * repeats), torch.arange(flush, device=DEVICE))
    queue = HindsightQueue(cfg, TrainCfg(hindsight_flush_episodes=flush), gen)
    outputs = []
    def flush_once() -> None:
      queue.pending.append(episodes)
      outputs.append(queue.flush())
    results[flush] = _timed(flush_once, 4)
    relabelled = outputs[-1]
    assert len(relabelled.length) == flush
    results[f'insert{flush}'] = _timed(lambda: replay.insert(relabelled), 1)
    del outputs, episodes

  sample_small = _timed(lambda: replay.sample(2560, gen), 32)
  sample_large = _timed(lambda: replay.sample(40960, gen), 16)
  memory = torch.cuda.memory_allocated() / 2**30
  print(f'\nW={num_worlds}, capacity {capacity}, {replay.num_rows} rows, {replay.max_episodes} episode slots, '
        f'{memory:.2f} GiB allocated in this process\n'
        f'  buffer append per step    {append_ms:8.3f} ms\n'
        f'  insert {num_worlds:5d} episodes     {insert_all:8.3f} ms ({num_worlds * cfg.max_steps} transitions)\n'
        f'  insert {each:5d} episodes     {insert_small:8.3f} ms not full, {evicting_small:8.3f} ms when full and evicting\n'
        f'  hindsight flush   256 eps {results[256]:8.2f} ms, then insert {results["insert256"]:8.2f} ms\n'
        f'  hindsight flush  4096 eps {results[4096]:8.2f} ms, then insert {results["insert4096"]:8.2f} ms\n'
        f'  sample  2560              {sample_small:8.3f} ms\n'
        f'  sample 40960              {sample_large:8.3f} ms')
  assert replay.extra_evicted == 0
