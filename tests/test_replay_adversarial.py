"""Adversarial checks on replay/ and hindsight: what the goldens and the slow model cannot see.

The goldens stop at dwell, done, score, reward and action, with graph features stubbed out,
and they feed rows that came from the original env. The slow model in test_replay.py uses
synthetic rows. So here the question is end to end: does a stored transition, rebuilt by
build_graph after laps of the ring, equal to the bit what the live env showed the policy?

Run with -s to see the measured numbers:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m pytest -q -s tests/test_replay_adversarial.py
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import subprocess
import time
from typing import NamedTuple

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.td3 import TD3, Batch
from roboballet_mjlab.config import HOME_QPOS, ModelCfg, TrainCfg
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.replay.hindsight import HindsightQueue, Picks, draw_picks, eligible, relabel
from roboballet_mjlab.replay.replay import (
  STAT,
  EpisodeBatch,
  EpisodeBuffer,
  ReplayBuffer,
  cat_episodes,
  row_storage,
  select_episodes,
)
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.scenes.io import load_pool
from roboballet_mjlab.structs import Obs, RobotState, Scene
from tests.test_replay import _env_with_early_endings, action_of, make_batch, reward_of, scene_of, stats_of
from tests.test_sim_common import DATA, DEVICE, PROJECT, REPO, default_scene, env_cfg

HOME = torch.tensor(HOME_QPOS, dtype=torch.float32, device=DEVICE)


@functools.lru_cache(maxsize=None)
def pool_from_disk(name: str, count: int = 64) -> Scene:
  path = scene_pool(name)
  if not path.exists():
    pytest.fail(f'{path} is missing, so this check did not run')
  return Scene(*[x[:count].to(DEVICE) for x in load_pool(path, 'cpu').scene])


def generator(seed: int) -> torch.Generator:
  return torch.Generator(device=DEVICE).manual_seed(seed)


def digest(*arrays: np.ndarray) -> bytes:
  h = hashlib.blake2b(digest_size=16)
  for a in arrays:
    h.update(a.tobytes())
  return h.digest()


def graph_digests(obs: Obs, topo: Topology) -> list[bytes]:
  """One digest per batch element, over the exact bytes of nodes, edges and globals."""
  arrays = [x.cpu().numpy() for x in build_graph(obs, topo)]
  return [digest(*[a[i] for a in arrays]) for i in range(len(arrays[0]))]


def scene_digests(scene: Scene) -> list[bytes]:
  arrays = [x.cpu().numpy() for x in scene]
  return [digest(*[a[i] for a in arrays]) for i in range(len(arrays[0]))]


def widen(robot: RobotState) -> RobotState:
  """A stored row in the dtypes the env hands out."""
  return robot._replace(dwell=robot.dwell.long(), step=robot.step.long(), aug_code=robot.aug_code.long())


def uniform_action(n: int, R: int, gen: torch.Generator) -> torch.Tensor:
  return torch.rand(n, R, 7, generator=gen, device=DEVICE) * 2 - 1


# A: a shadow of every transition, keyed by its id, against a full read of the ring.


class Expected(NamedTuple):
  s: bytes
  next_s: bytes
  action: bytes
  reward: bytes
  terminal: bool
  behaviour: int
  hindsight: bool
  scene: bytes
  stats: bytes


class Shadow:
  """Everything that should come back out of the ring, in insertion order."""

  def __init__(self, replay: ReplayBuffer, topo: Topology) -> None:
    self.replay, self.topo = replay, topo
    self.items: list[Expected] = []
    self.episode_first: list[int] = []  # The id of each episode's first transition.

  def add_batch_from_rows(self, batch: EpisodeBatch) -> None:
    """For hindsight, whose only truth is the relabelled rows themselves."""
    length = batch.length.tolist()
    scene_keys = scene_digests(batch.scene)
    for i, A in enumerate(length):
      self.episode_first.append(len(self.items))
      robot = widen(RobotState(*[x[i, : A + 1] for x in batch.robot]))
      scene = Scene(*[x[i : i + 1].expand(A + 1, *x.shape[1:]) for x in batch.scene])
      keys = graph_digests(Obs(robot, scene), self.topo)
      action, reward = batch.action[i].cpu().numpy(), batch.reward[i].cpu().numpy()
      for t in range(A):
        self.items.append(Expected(
          keys[t], keys[t + 1], action[t].tobytes(), reward[t].tobytes(), t == A - 1,
          int(batch.behaviour_step[i]), bool(batch.is_hindsight[i]), scene_keys[i], batch.stats[i].cpu().numpy().tobytes()))

  def check(self) -> int:
    replay = self.replay
    if replay.size == 0:
      return 0
    first = replay.evicted
    assert replay.inserted == len(self.items)
    # Not one transition more or less than FIFO and the table allow, so nothing goes too early.
    oldest_in_table = self.episode_first[-replay.max_episodes] if len(self.episode_first) >= replay.max_episodes else 0
    assert first == max(replay.inserted - replay.capacity, oldest_in_table, 0), (first, replay.counters())
    checked = 0
    for start in range(0, replay.size, 4096):
      position = torch.arange(start, min(start + 4096, replay.size), device=DEVICE)
      sample = replay.fetch(position)
      s = graph_digests(sample.obs, self.topo)
      next_s = graph_digests(sample.next_obs, self.topo)
      scenes = scene_digests(sample.obs.scene)
      action, reward = sample.action.cpu().numpy(), sample.reward.cpu().numpy()
      for j, p in enumerate(position.tolist()):
        want = self.items[first + p]
        assert s[j] == want.s, f'features of s, transition {first + p}'
        assert next_s[j] == want.next_s, f"features of s', transition {first + p}"
        assert action[j].tobytes() == want.action, f'action, transition {first + p}'
        assert reward[j].tobytes() == want.reward, f'reward, transition {first + p}'
        assert bool(sample.terminal[j]) == want.terminal, f'terminal, transition {first + p}'
        assert int(sample.behaviour_step[j]) == want.behaviour
        assert bool(sample.is_hindsight[j]) == want.hindsight
        assert scenes[j] == want.scene, f'scene, transition {first + p}'
        assert sample.stats[j].cpu().numpy().tobytes() == want.stats
        checked += 1
    return checked


LIVE_CASES = {
  # name: (R, T, pool name or None, W, steps, capacity, max_episodes, flush)
  '4r10t_three_laps': (4, 10, '4r_10t_30o_train', 24, 300, 4000, None, 24),
  '4r10t_tight_table': (4, 10, '4r_10t_30o_train', 24, 300, 4000, 5, 24),
  '8r40t_two_laps': (8, 40, '8r_40t_30o_train', 8, 900, 5000, None, 8),
  '8r40t_tight_table': (8, 40, '8r_40t_30o_train', 8, 900, 5000, 3, 8),
}


def _shadow_rollout(env: BalletEnv, W: int, steps: int, capacity: int, max_episodes, flush: int, seed: int, stagger: bool = False):
  cfg = env.cfg
  topo = Topology(cfg, DEVICE)
  buf = EpisodeBuffer(cfg, W, DEVICE)
  replay = ReplayBuffer(cfg, capacity, DEVICE, max_episodes)
  gen = generator(seed)
  queue = HindsightQueue(cfg, TrainCfg(hindsight_flush_episodes=flush), gen)
  shadow = Shadow(replay, topo)
  pending: list[list[tuple[bytes, bytes, bytes]]] = [[] for _ in range(W)]
  scene_keys: list[bytes | None] = [None] * W
  behaviour = [0] * W
  checked = laps = hindsight = 0
  for step in range(steps):
    if stagger and step < W:  # What a trainer does to decorrelate: one world restarts each step.
      ids = torch.tensor([step], device=DEVICE)
      env.reset(ids)
      buf.discard(ids)
      pending[step] = []
    obs = env.observe()
    keys = graph_digests(obs, topo)
    now_scene = scene_digests(obs.scene)
    action = uniform_action(W, cfg.world.num_robots, gen)
    out = env.step(action)
    batch = buf.append(obs, action, out, behaviour_step=step)
    action_np, reward_np = action.cpu().numpy(), out.reward.cpu().numpy()
    for w in range(W):
      if not pending[w]:
        scene_keys[w], behaviour[w] = now_scene[w], step
      pending[w].append((keys[w], action_np[w].tobytes(), reward_np[w].tobytes()))
    if batch is None:
      continue
    final_keys = graph_digests(Obs(out.next_robot, obs.scene), topo)
    for i, w in enumerate(out.terminal.nonzero().flatten().tolist()):
      rows = pending[w]
      assert int(batch.length[i]) == len(rows)
      stats = batch.stats[i].cpu().numpy().tobytes()
      shadow.episode_first.append(len(shadow.items))
      for t, (key, a, r) in enumerate(rows):
        next_key = rows[t + 1][0] if t + 1 < len(rows) else final_keys[w]
        shadow.items.append(Expected(key, next_key, a, r, t == len(rows) - 1, behaviour[w], False, scene_keys[w], stats))
      pending[w] = []
    replay.insert(batch)
    relabelled = queue.add(batch)
    if relabelled is not None:
      shadow.add_batch_from_rows(relabelled)
      replay.insert(relabelled)
      hindsight += len(relabelled.length)
    checked += shadow.check()
    laps = replay.inserted / capacity
  return replay, shadow, checked, laps, hindsight


@pytest.mark.parametrize('name', LIVE_CASES)
def test_stored_features_equal_live_features_bit_exact(name):
  R, T, pool, W, steps, capacity, max_episodes, flush = LIVE_CASES[name]
  cfg = env_cfg(num_robots=R, num_targets=T)
  env = BalletEnv(cfg, W, pool_from_disk(pool), DEVICE, seed=11)
  replay, shadow, checked, laps, hindsight = _shadow_rollout(env, W, steps, capacity, max_episodes, flush, seed=12)
  print(f'\n{name}: {replay.counters()}, {checked} transitions compared bit-exact across reads, '
        f'{laps:.1f} laps, {hindsight} hindsight episodes')
  assert laps > 1.5 and hindsight > 0 and checked > 0
  live = replay.fetch(torch.arange(replay.size, device=DEVICE))
  print(f'  live: {int((live.reward < 0).sum())} of {replay.size} rewards negative (real rollbacks), '
        f'{int((live.obs.robot.dwell > 0).any(1).sum())} rows dwelling, {int(live.terminal.sum())} terminal, '
        f'{int(live.is_hindsight.sum())} hindsight')
  if max_episodes is None:  # A tight table ends up holding only the last flush, which never collides.
    assert (live.reward < 0).any(), 'rows that went through a rollback must be among those compared'
    assert not live.is_hindsight.all() and live.is_hindsight.any()
  assert (live.obs.robot.dwell > 0).any()
  if max_episodes is not None:
    assert replay.extra_evicted > 0, 'the tight table must force evictions'


def test_stored_features_with_early_endings_and_tiny_table():
  """Episodes of 6 and 30 actions, mid-batch endings, a ring that wraps many times."""
  W = 12
  env, _ = _env_with_early_endings(W)
  replay, shadow, checked, laps, _ = _shadow_rollout(env, W, 300, 97, 3, 12, seed=13)
  print(f'\nearly endings: {replay.counters()}, {checked} compared, {laps:.1f} laps')
  assert laps > 10 and replay.extra_evicted > 0


def test_stored_features_with_staggered_starts():
  """A read after every world's own terminal, so the ring is cut at every kind of boundary."""
  W = 12
  cfg = env_cfg()
  env = BalletEnv(cfg, W, pool_from_disk('4r_10t_30o_train'), DEVICE, seed=16)
  replay, shadow, checked, laps, hindsight = _shadow_rollout(env, W, 3 * cfg.max_steps + W, 1500, None, 5, seed=17, stagger=True)
  print(f'\nstaggered: {replay.counters()}, {checked} compared, {laps:.1f} laps, {hindsight} hindsight')
  assert laps > 3 and hindsight > 0 and checked > 10_000


def test_autocast_does_not_touch_stored_or_relabelled_data():
  """The bf16 path wraps networks in autocast. If a trainer also wraps the data path, nothing changes."""
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 8, seed=18, policy='uniform')
  plain = relabel(batch, cfg, TrainCfg(), generator(19))
  with torch.autocast('cuda', dtype=torch.bfloat16):
    cast = relabel(batch, cfg, TrainCfg(), generator(19))
    replay = ReplayBuffer(cfg, 2000, DEVICE)
    replay.insert(cast)
    sample = replay.fetch(torch.arange(replay.size, device=DEVICE))
    graph = build_graph(sample.obs, Topology(cfg, DEVICE))
  for x, y in zip(plain.robot, cast.robot):
    assert torch.equal(x, y) and x.dtype == y.dtype
  assert torch.equal(plain.action, cast.action) and torch.equal(plain.reward, cast.reward)
  for field, x in zip(graph._fields, graph):
    assert x.dtype == torch.float32, field
  exact = build_graph(sample.obs, Topology(cfg, DEVICE))
  for x, y in zip(graph, exact):
    assert torch.equal(x, y)


@pytest.mark.parametrize('R,T,O', [(1, 1, 0), (1, 3, 1), (2, 3, 0), (4, 5, 1)])
def test_stored_features_in_small_worlds(R, T, O):
  """G11's small task is 1 robot, 1 target, no obstacles: empty columns must survive the ring."""
  W = 6
  cfg = env_cfg(num_robots=R, num_obstacles=O, num_targets=T)
  env = BalletEnv(cfg, W, default_scene(cfg.world, 4, num_targets=T), DEVICE, seed=14)
  replay, shadow, checked, laps, hindsight = _shadow_rollout(env, W, 3 * cfg.max_steps, 2 * W * cfg.max_steps, None, W, seed=15)
  print(f'\n{R}r{T}t{O}o: {replay.counters()}, {checked} compared, {laps:.1f} laps, {hindsight} hindsight')
  assert laps > 1 and checked > 0
  assert (hindsight > 0) == (int(0.4 * T) > 0)


def test_sampling_stays_uniform_after_wrap_with_partial_eviction():
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=3)
  replay = ReplayBuffer(cfg, 211, DEVICE, 20)
  rng = np.random.default_rng(5)
  next_id = 0
  for _ in range(40):
    lengths = rng.integers(1, 40, size=rng.integers(1, 4)).tolist()
    replay.insert(make_batch(list(range(next_id, next_id + len(lengths))), lengths))
    next_id += len(lengths)
  assert replay.evicted > 0 and replay.inserted > 2 * replay.capacity
  size = replay.size
  n = 4_000_000
  sample = replay.sample(n, generator(6))
  tags = sample.obs.robot.q[:, 0, 0].long()
  counts = torch.unique(tags, return_counts=True)[1].float().cpu().numpy()
  assert len(counts) == size
  chi2 = ((counts - n / size) ** 2 / (n / size)).sum()
  print(f'\nchi-square {chi2:.1f} for {size - 1} degrees of freedom over {size} live transitions')
  assert chi2 < size + 4 * np.sqrt(2 * size), chi2


def test_episode_row_semantics_on_the_real_env():
  """Row t holds the score before step t, reward[t] is its own step, terminal only on the last."""
  W = 12
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  gen = generator(21)
  solved_steps = checks = 0
  for _ in range(150):
    obs = env.observe()
    action = uniform_action(W, 4, gen)
    out = env.step(action)
    batch = buf.append(obs, action, out)
    if batch is None:
      continue
    for i, A in enumerate(batch.length.tolist()):
      score = batch.robot.score[i]
      step = batch.robot.step[i].long()
      assert torch.equal(step[: A + 1], torch.arange(A + 1, device=DEVICE))
      assert score[0] == 0 and not batch.robot.target_done[i, 0].any() and (batch.robot.dwell[i, 0] == 0).all()
      delta = score[1 : A + 1] - score[: A]
      penalty = delta - batch.reward[i, : A]
      # Reward is the score delta of row t to row t + 1 less a whole number of per-robot penalties.
      units = penalty / env.cfg.collision_penalty_per_robot
      assert (units - units.round()).abs().max() < 1e-3
      assert (units.round() >= 0).all() and (units.round() <= 4).all()
      solved_steps += int((delta > 0).sum())
      checks += A
  assert solved_steps > 0 and checks > 0


# B: hindsight against a literal, per-episode reading of planning_env.step, on rows from the
# port's own env, in every mode of motion. The goldens only ever saw the original env's rows.


@dataclasses.dataclass
class Literal:
  length: int
  dwell: list[list[int]]
  done: list[list[bool]]
  score: list[float]
  action: list[np.ndarray]
  reward: list[float]
  takeovers: int = 0
  takeover_over_dwell: int = 0
  zeroed: int = 0


def literal_hindsight(q, tip_pos, tip_rot, raw_action, target_pos, target_rot, cfg) -> Literal:
  """actor.py:378-395 and planning_env.py:792-816, :639-691, :864-883, in plain f64 Python."""
  q = q.astype(np.float64)
  home = np.array(HOME_QPOS, dtype=np.float64)
  R, T = q.shape[1], len(target_pos)
  dwell, done, score, step = [0] * R, [False] * T, 0.0, 0
  out = Literal(0, [list(dwell)], [list(done)], [0.0], [], [])
  weight, threshold = cfg.return_to_start_reward_weight, cfg.return_to_start_threshold
  cos_limit = np.cos(cfg.angle_threshold_rad)

  def terminal() -> bool:
    if step >= cfg.max_steps:
      return True
    if not all(done) or any(dwell):
      return False
    if not cfg.return_to_start_required:
      return True
    return bool((np.abs(q[step] - home) < cfg.return_to_start_epsilon).all())

  while not terminal():
    action = raw_action[step].astype(np.float64) * cfg.max_joint_velocity
    was_dwelling = [d > 0 for d in dwell]
    for r in range(R):
      if dwell[r] > 0:
        action[r] = 0.0
        dwell[r] -= 1
        out.zeroed += 1
    if all(done):
      for r in range(R):
        diff = home - q[step, r]
        if (np.abs(diff) < threshold).all():
          action[r] = diff / cfg.timestep_s
          out.takeovers += 1
          out.takeover_over_dwell += int(was_dwelling[r])
    step += 1
    total = 0.0
    all_done = True
    for k in range(T):
      if done[k]:
        total += 1.0
        continue
      for r in range(R):
        if dwell[r] > 0:
          continue
        near = np.linalg.norm(tip_pos[step, r].astype(np.float64) - target_pos[k]) < cfg.position_threshold
        cos_angle = (np.trace(tip_rot[step, r].astype(np.float64).T @ target_rot[k].astype(np.float64)) - 1) / 2
        if near and cos_angle > cos_limit:
          done[k] = True
          dwell[r] = cfg.dwell_steps
          break
      if done[k]:
        total += 1.0
      else:
        all_done = False
    new_score = (1 - weight) * total / T
    if all_done:
      rts = 0.0
      for r in range(R):
        diff = np.abs(q[step, r] - home)
        rts += 1.0 if (diff < threshold).all() else 1.0 - np.clip(cfg.return_to_start_coeff * np.mean(diff**2), 0.0, 1.0)
      new_score += weight * rts / R
    out.action.append(action / cfg.max_joint_velocity)
    out.reward.append(new_score - score)
    score = new_score
    out.dwell.append(list(dwell))
    out.done.append(list(done))
    out.score.append(score)
  out.length = step
  return out


def wander_action(obs: Obs, step: int, amp: torch.Tensor, period: torch.Tensor, cfg) -> torch.Tensor:
  """Tracks home + amp * sin, so arms stay in the takeover box when amp < 0.25."""
  want = HOME + amp * torch.sin(2 * torch.pi * (step + 1) / period)
  return ((want - obs.robot.q) / cfg.timestep_s / cfg.max_joint_velocity).clamp(-1, 1)


def real_episodes(cfg, pool: Scene, W: int, seed: int, policy: str, rounds: int = 1) -> EpisodeBatch:
  env = BalletEnv(cfg, W, pool, DEVICE, seed=seed)
  env.reset(scene_ids=torch.arange(W, device=DEVICE) % len(pool.q0))
  buf = EpisodeBuffer(cfg, W, DEVICE)
  gen = generator(seed + 1)
  R = cfg.world.num_robots
  amp = (torch.rand(W, R, 7, generator=gen, device=DEVICE) * 0.2).clamp(max=0.2)
  if policy == 'wide':  # Still inside the 0.25 box, but beyond 0.2175, so a takeover request exceeds 1.
    amp = 0.225 + amp * 0.1
  period = torch.rand(W, R, 7, generator=gen, device=DEVICE) * 55 + 25
  batches = []
  step = 0
  while sum(len(b.length) for b in batches) < W * rounds:
    obs = env.observe()
    if policy in ('wander', 'wide'):
      action = wander_action(obs, step, amp, period, cfg)
    else:
      action = uniform_action(W, R, gen)
    out = env.step(action)
    batch = buf.append(obs, action, out, behaviour_step=step)
    if batch is not None:
      batches.append(batch)
    step += 1
  return cat_episodes(batches)


LITERAL_CASES = {
  # name: (R, T, fraction, return to start, policy, W)
  '4r10t_default_uniform': (4, 10, 0.4, False, 'uniform', 32),
  '4r10t_default_wander': (4, 10, 0.4, False, 'wander', 32),
  '4r10t_all_wander': (4, 10, 1.0, False, 'wander', 32),
  '4r3t_all_wander': (4, 3, 1.0, False, 'wander', 32),
  '4r1t_all_wander': (4, 1, 1.0, False, 'wander', 32),
  '4r10t_all_rts_wander': (4, 10, 1.0, True, 'wander', 32),
  '4r10t_all_rts_wide': (4, 10, 1.0, True, 'wide', 32),
  '4r10t_all_uniform': (4, 10, 1.0, False, 'uniform', 32),
  '8r40t_default_uniform': (8, 40, 0.4, False, 'uniform', 8),
  '8r40t_all_wander': (8, 40, 1.0, False, 'wander', 8),
}


def _pool_for(R: int, T: int) -> Scene:
  pool = pool_from_disk('8r_40t_30o_train') if R == 8 else pool_from_disk('4r_10t_30o_train')
  return pool._replace(target_pos=pool.target_pos[:, :T], target_rot=pool.target_rot[:, :T])


@pytest.mark.parametrize('name', LITERAL_CASES)
def test_hindsight_matches_a_literal_reading_on_rows_from_this_env(name):
  R, T, fraction, rts, policy, W = LITERAL_CASES[name]
  cfg = env_cfg(num_robots=R, num_targets=T, return_to_start_reward_weight=0.25 if rts else 0.0,
                return_to_start_required=rts)
  train = TrainCfg(hindsight_fraction=fraction)
  batch = real_episodes(cfg, _pool_for(R, T), W, seed=31, policy=policy)
  k = int(fraction * T)
  gen = generator(32)
  out = relabel(batch, cfg, train, gen)
  assert out is not None
  kept = eligible(batch, k)
  source = select_episodes(batch, kept)
  totals = dict(takeovers=0, over_dwell=0, zeroed=0, shorter=0, below_k=0, steps=0, beyond_unit=0)
  for i in range(len(out.length)):
    A_real = int(source.length[i])
    want = literal_hindsight(
      source.robot.q[i, : A_real + 1].cpu().numpy(), source.robot.tip_pos[i, : A_real + 1].cpu().numpy(),
      source.robot.tip_rot[i, : A_real + 1].cpu().numpy(), source.action[i, : A_real].cpu().numpy(),
      out.scene.target_pos[i].cpu().numpy(), out.scene.target_rot[i].cpu().numpy(), cfg)
    H = int(out.length[i])
    assert H == want.length, f'episode {i}: length'
    assert out.robot.dwell[i, : H + 1].tolist() == want.dwell, f'episode {i}: dwell'
    assert out.robot.target_done[i, : H + 1].tolist() == want.done, f'episode {i}: done'
    assert np.abs(out.robot.score[i, : H + 1].cpu().numpy() - np.array(want.score)).max() < 1e-6, f'episode {i}: score'
    assert np.abs(out.reward[i, : H].cpu().numpy() - np.array(want.reward)).max() < 1e-6, f'episode {i}: reward'
    assert np.abs(out.action[i, : H].cpu().numpy() - np.stack(want.action)).max() < 2e-6, f'episode {i}: action'
    # The motion is the real episode's, untouched, with the real robot rows.
    for field in ('q', 'qd', 'tip_pos', 'tip_rot', 'step'):
      assert torch.equal(getattr(out.robot, field)[i, : H + 1], getattr(source.robot, field)[i, : H + 1]), field
    totals['takeovers'] += want.takeovers
    totals['over_dwell'] += want.takeover_over_dwell
    totals['zeroed'] += want.zeroed
    totals['shorter'] += int(H < A_real)
    totals['below_k'] += int(sum(want.done[-1]) < k)
    totals['steps'] += H
    totals['beyond_unit'] += int((np.abs(np.stack(want.action)) > 1.0).any(-1).sum())
  print(f'\n{name}: {len(out.length)} hindsight of {len(batch.length)} real, {totals}')
  assert totals['steps'] > 0 and totals['zeroed'] > 0
  if fraction == 1.0 and policy in ('wander', 'wide'):  # These cases exist to reach the takeover branches.
    assert totals['takeovers'] > 0 and totals['over_dwell'] > 0
    assert (totals['shorter'] == 0) == rts, 'return-to-start keeps running until parked, otherwise it ends early'
  if policy == 'wide':
    assert totals['beyond_unit'] > 0, 'this case exists to reach takeover requests beyond the action bounds'


def test_unmoved_targets_reproduce_the_envs_own_timeline_bit_exact():
  """With nothing relabelled, the replay's scoring must equal the env's on the same rows, early end included."""
  W = 12
  cfg = env_cfg(num_targets=3)
  pool = _pool_for(4, 3)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=33)
  ids = torch.arange(W, device=DEVICE)
  env.reset(scene_ids=ids)
  tip_pos, tip_rot = env.world.tcp()
  # Half the scenes get all targets on home tips of arms 0, 1, 2, so they end after the final
  # dwell. The rest get two, so they time out with 2 of 3 solved and arms still in the box.
  all_three = ids % 2 == 0
  target_pos, target_rot = pool.target_pos[ids].clone(), pool.target_rot[ids].clone()
  count = torch.where(all_three, 3, 2)
  for k in range(3):
    on = k < count
    target_pos[:, k] = torch.where(on[:, None], tip_pos[:, k], target_pos[:, k])
    target_rot[:, k] = torch.where(on[:, None, None], tip_rot[:, k], target_rot[:, k])
  pool = Scene(*[x[ids] for x in pool])._replace(target_pos=target_pos, target_rot=target_rot)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=34)
  env.reset(scene_ids=ids)
  buf = EpisodeBuffer(cfg, W, DEVICE)
  gen = generator(35)
  amp = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 0.05
  period = torch.rand(W, 4, 7, generator=gen, device=DEVICE) * 55 + 25
  found: list[EpisodeBatch] = []
  for step in range(cfg.max_steps * 3):
    obs = env.observe()
    # The same target tips stay in reach only for the first episode's opening steps, which is all this needs.
    action = wander_action(obs, step, amp, period, cfg)
    out = env.step(action)
    batch = buf.append(obs, action, out)
    if batch is not None:
      found.append(batch)
    if sum(len(b.length) for b in found) >= W:
      break
  batch = cat_episodes(found)
  n = len(batch.length)
  lengths = batch.length.tolist()
  assert min(lengths) < cfg.max_steps and max(lengths) == cfg.max_steps, lengths
  train = TrainCfg(hindsight_fraction=2.0)  # k = 6 of 3, so every episode qualifies, then nothing moves.
  chosen = torch.zeros(n, 3, dtype=torch.bool, device=DEVICE)
  zeros = torch.ones(n, 3, dtype=torch.long, device=DEVICE)
  out = relabel(batch, cfg, train, generator(36), Picks(chosen, zeros, zeros))
  assert torch.equal(out.scene.target_pos, batch.scene.target_pos)
  assert torch.equal(out.length, batch.length), 'the same early endings'
  penalties = 0
  for i, A in enumerate(lengths):
    assert torch.equal(out.robot.dwell[i, : A + 1], batch.robot.dwell[i, : A + 1]), f'episode {i}: dwell'
    assert torch.equal(out.robot.target_done[i, : A + 1], batch.robot.target_done[i, : A + 1]), f'episode {i}: done'
    assert torch.equal(out.robot.score[i, : A + 1], batch.robot.score[i, : A + 1]), f'episode {i}: score'
    delta = batch.robot.score[i, 1 : A + 1] - batch.robot.score[i, : A]
    assert torch.equal(out.reward[i, : A], delta), f'episode {i}: reward is the score delta, no penalty'
    penalties += int((batch.reward[i, : A] < delta).sum())
  print(f'\nidentity relabel: lengths {sorted(set(lengths))}, final solved {batch.robot.target_done[torch.arange(n), batch.length].sum(1).tolist()}, '
        f'{int((batch.robot.dwell > 0).sum())} dwelling robot rows, {penalties} real steps with a penalty that hindsight drops')


def test_hindsight_copies_statistics_and_behaviour_step_but_not_its_flag():
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 8, seed=37, policy='uniform')
  stats = batch.stats.clone()
  stats[:, STAT['network_step']] = batch.behaviour_step.float()
  batch = batch._replace(behaviour_step=batch.behaviour_step + 7, stats=stats)
  out = relabel(batch, cfg, TrainCfg(), generator(38))
  keep = eligible(batch, 4)
  assert torch.equal(out.behaviour_step, batch.behaviour_step[keep])
  flag = STAT['is_hindsight_replay']
  mask = torch.ones(12, dtype=torch.bool, device=DEVICE)
  mask[flag] = False
  assert torch.equal(out.stats[:, mask], batch.stats[keep][:, mask])
  assert (out.stats[:, flag] == 1).all() and (batch.stats[:, flag] == 0).all()
  # The non-parity flag's penalty is the real one, read back from score and reward.
  real_penalty = (batch.robot.score[:, 1:] - batch.robot.score[:, :-1]) - batch.reward
  assert (real_penalty > 0.01).any()


def test_relabel_works_with_the_fasttd3_config():
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 4, seed=39, policy='uniform')
  a = relabel(batch, cfg, TrainCfg(), generator(40))
  b = relabel(batch, cfg, FastTD3Cfg(), generator(40))
  assert torch.equal(a.action, b.action) and torch.equal(a.reward, b.reward) and torch.equal(a.robot.dwell, b.robot.dwell)


def _pinned_pick(batch: EpisodeBatch, target: int, arm: int, time_: int, count: int) -> Picks:
  """Moves `count` targets from index `target`, all to the same arm and time."""
  n, T = batch.robot.target_done.shape[0], batch.robot.target_done.shape[2]
  chosen = torch.zeros(n, T, dtype=torch.bool, device=DEVICE)
  chosen[:, target : target + count] = True
  return Picks(chosen, torch.full((n, T), arm, device=DEVICE), torch.full((n, T), time_, device=DEVICE))


def test_first_steps_of_a_hindsight_episode_start_fresh():
  """Row 0 knows nothing, the first solve lands in row t + 1, and the dwell counts 5..0 there."""
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 16, seed=41, policy='uniform')
  train = TrainCfg(hindsight_fraction=0.4)
  time_ = 1
  arm = 2
  gen = generator(42)
  out = relabel(batch, cfg, train, gen, _pinned_pick(batch, 3, arm, time_, 4))
  n = len(out.length)
  assert n == len(batch.length)
  assert (out.robot.score[:, 0] == 0).all() and not out.robot.target_done[:, 0].any() and (out.robot.dwell[:, 0] == 0).all()
  # Before the step that reaches the tip, nothing moved: the first action stays the raw one.
  assert torch.equal(out.action[:, 0], batch.action[:, 0])
  # One target lands on row 1 for that arm, the lowest index first, and it scores at once.
  assert out.robot.target_done[:, 1, 3].all()
  assert torch.equal(out.robot.dwell[:, 1, arm], torch.full((n,), 5, dtype=torch.int8, device=DEVICE))
  assert (out.reward[:, 0] >= 0.1 - 1e-6).all()
  assert torch.equal(out.robot.score[:, 1], out.reward[:, 0])
  # The arm is frozen for the next 5 requests, and not for the sixth. A dwelling arm can solve
  # nothing, so the count is exact, and the free request below needs row 6 at zero.
  for t, left in zip(range(1, 7), (5, 4, 3, 2, 1, 0)):
    assert (out.robot.dwell[:, t, arm] == left).all(), (t, left)
  frozen = out.action[:, 1:6, arm]
  assert (frozen == 0).all()
  assert torch.equal(out.action[:, 6, arm], batch.action[:, 6, arm]), 'the sixth request is free again'
  # Other arms keep their raw requests unless they solved something themselves.
  other = [r for r in range(4) if r != arm]
  free = out.robot.dwell[:, 1:6, other] == 0
  got = out.action[:, 1:6, other]
  raw = batch.action[:, 1:6, other]
  assert (got[free] == raw[free]).all()


def test_target_left_on_a_dwelling_arm_is_missed():
  """Four targets at one arm and time: the first takes it. The others only count if revisited."""
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 16, seed=43, policy='uniform')
  out = relabel(batch, cfg, TrainCfg(hindsight_fraction=0.4), generator(44), _pinned_pick(batch, 0, 1, 50, 4))
  row = out.robot.target_done[:, 50]
  assert row[:, 0].all(), 'lowest target index goes first'
  # A uniform arm has moved by far more than 2.5 cm before its dwell ends, so most are missed.
  print('\nmoved targets solved in the end, per episode:', out.robot.target_done[torch.arange(16), out.length][:, :4].sum(1).tolist())
  assert (out.robot.target_done[torch.arange(16), out.length][:, :4].sum(1) < 4).any()


def test_stale_rows_past_the_length_change_nothing():
  """EpisodeBuffer leaves another episode's rows past `length`. Scoring must not see them."""
  cfg = env_cfg(num_robots=4, num_targets=3)
  batch = real_episodes(cfg, _pool_for(4, 3), 16, seed=45, policy='wander')
  train = TrainCfg(hindsight_fraction=1.0)
  k = 3
  final = batch.robot.target_done[torch.arange(16), batch.length]
  # A forced shorter episode: pretend these ended at 40 actions, then poison everything after.
  short = batch._replace(length=torch.full_like(batch.length, 40))
  final = short.robot.target_done[torch.arange(16), short.length]
  picks = draw_picks(final, short.length, 4, k, generator(46))
  # Row 41 onward get junk that lands exactly on the moved targets.
  a = relabel(short, cfg, train, generator(47), picks)
  poisoned_robot = short.robot._replace(
    tip_pos=short.robot.tip_pos.clone(), tip_rot=short.robot.tip_rot.clone(), q=short.robot.q.clone())
  poisoned_robot.tip_pos[:, 41:] = a.scene.target_pos[:, :1].unsqueeze(1).expand(-1, poisoned_robot.tip_pos.shape[1] - 41, 4, -1)
  poisoned_robot.q[:, 41:] = HOME
  b = relabel(short._replace(robot=poisoned_robot), cfg, train, generator(47), picks)
  assert torch.equal(a.length, b.length)
  for i in range(16):
    H = int(a.length[i])
    for name in ('dwell', 'target_done', 'score', 'q', 'tip_pos'):
      assert torch.equal(getattr(a.robot, name)[i, : H + 1], getattr(b.robot, name)[i, : H + 1]), name
    assert torch.equal(a.action[i, : H], b.action[i, : H])
    assert torch.equal(a.reward[i, : H], b.reward[i, : H])


def test_hindsight_of_terminal_row_keeps_real_final_state_and_fresh_augmentation():
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 16, seed=48, policy='uniform')
  out = relabel(batch, cfg, TrainCfg(), generator(49))
  assert out.robot.aug_code.dtype == torch.int8
  stored = out.robot.aug_code[:, : cfg.max_steps + 1]
  assert not torch.equal(stored, batch.robot.aug_code[:, : cfg.max_steps + 1]), 'fresh draws, not the real ones'
  counts = torch.bincount(stored.long().flatten(), minlength=4).float()
  assert ((counts / counts.sum() - 0.25).abs() < 0.01).all(), counts.tolist()
  # No two adjacent rows share their draw, which also tests independence along time.
  assert (stored[:, 1:] == stored[:, :-1]).float().mean().item() == pytest.approx(0.25, abs=0.02)


# C: counters, dtypes, devices, memory.


@pytest.mark.parametrize('offset_bits', [31, 33, 52, 54, 60])
def test_counters_far_along_a_long_run(offset_bits):
  """The same inserts give the same contents whatever the counters have reached."""
  C, E = 64, 4
  a = ReplayBuffer(env_cfg(num_robots=2, num_obstacles=3, num_targets=3), C, DEVICE, E)
  b = ReplayBuffer(a.cfg, C, DEVICE, E)
  shift_t = (2**offset_bits // C) * C
  shift_e = (2**offset_bits // E) * E
  b.inserted, b.evicted, b.episodes = shift_t, shift_t, shift_e
  rng = np.random.default_rng(offset_bits)
  next_id = 0
  for _ in range(40):
    lengths = rng.integers(1, 30, size=rng.integers(1, 5)).tolist()
    batch = make_batch(list(range(next_id, next_id + len(lengths))), lengths)
    next_id += len(lengths)
    a.insert(batch)
    b.insert(batch)
    # Table eviction reads first_transition, which starts at 0 for a counter that was shifted,
    # so the two only agree once the shifted table has been written through.
    if a.episodes > E:
      assert a.size == b.size, (a.counters(), b.counters())
      assert a.evicted + shift_t == b.evicted
      position = torch.arange(a.size, device=DEVICE)
      sa, sb = a.fetch(position), b.fetch(position)
      for x, y in zip(sa.obs.robot, sb.obs.robot):
        assert torch.equal(x, y)
      for x, y in zip(sa.next_obs.robot, sb.next_obs.robot):
        assert torch.equal(x, y)
      for x, y in zip(sa.obs.scene, sb.obs.scene):
        assert torch.equal(x, y)
      assert torch.equal(sa.terminal, sb.terminal) and torch.equal(sa.action, sb.action)
  assert b.inserted > 2**offset_bits


def test_behaviour_step_beyond_float_precision_keeps_its_value():
  """long in the table. The f32 copy in stats loses it, as the original's stats do at 2**24."""
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=3)
  batch = make_batch([0], [3])
  big = 2**40 + 1
  batch = batch._replace(behaviour_step=torch.full_like(batch.behaviour_step, big))
  replay = ReplayBuffer(cfg, 8, DEVICE, 2)
  replay.insert(batch)
  assert int(replay.fetch(torch.tensor([0], device=DEVICE)).behaviour_step[0]) == big
  buf = EpisodeBuffer(env_cfg(), 2, DEVICE)
  assert buf.behaviour_step.dtype == torch.long
  # Through append as well: a Python int must not pass through a float on its way in.
  W = 3
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  still = torch.zeros(W, 4, 7, device=DEVICE)
  finished = None
  for step in range(env.cfg.max_steps):
    obs = env.observe()
    finished = buf.append(obs, still, env.step(still), behaviour_step=big + step)
    if finished is not None:
      break
  assert finished is not None and int(finished.length[0]) == 6, 'the scene with its target on a home tip'
  assert finished.behaviour_step.tolist() == [big], 'latched at the first row, not the last'
  replay = ReplayBuffer(env.cfg, 16, DEVICE, 2)
  replay.insert(finished)
  assert int(replay.fetch(torch.tensor([5], device=DEVICE)).behaviour_step[0]) == big


@pytest.mark.parametrize('dt,T,expected', [(12.0 / 127, 10, torch.int8), (12.0 / 128, 10, torch.int16),
                                           (0.1, 10, torch.int8), (0.1, 40, torch.int16)])
def test_integer_fields_survive_their_extreme_values(dt, T, expected):
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=T, timestep_s=dt)
  L = cfg.max_steps
  n = cfg.dwell_steps
  robot = row_storage(cfg, (1, L + 1), DEVICE)
  assert robot.step.dtype == expected
  robot.step[0] = torch.arange(L + 1, device=DEVICE).to(robot.step.dtype)
  robot.dwell[0] = (torch.arange(L + 1, device=DEVICE) % (n + 1))[:, None].to(robot.dwell.dtype)
  robot.aug_code[0] = (torch.arange(L + 1, device=DEVICE) % 4)[:, None].to(robot.aug_code.dtype)
  scene = Scene(*[torch.zeros(1, *s, device=DEVICE) for s in (
    (2, 3), (2, 3, 3), (3, 3), (3, 3, 3), (3, 3), (T, 3), (T, 3, 3), (2, 7))])
  batch = EpisodeBatch(robot, torch.zeros(1, L, 2, 7, device=DEVICE), torch.zeros(1, L, device=DEVICE),
                       torch.tensor([L], device=DEVICE), scene, torch.zeros(1, dtype=torch.long, device=DEVICE),
                       torch.zeros(1, dtype=torch.bool, device=DEVICE), torch.zeros(1, 12, device=DEVICE))
  replay = ReplayBuffer(cfg, L + 3, DEVICE, 2)
  replay.insert(batch)
  sample = replay.fetch(torch.arange(L, device=DEVICE))
  index = torch.arange(L, device=DEVICE)
  assert torch.equal(sample.obs.robot.step, index) and sample.obs.robot.step.dtype == torch.long
  assert torch.equal(sample.next_obs.robot.step, index + 1)
  assert int(sample.next_obs.robot.step.max()) == L, 'the largest stored value, at the terminal row'
  assert (sample.obs.robot.step >= 0).all()
  assert sample.obs.robot.dwell.dtype == torch.long and (sample.obs.robot.dwell >= 0).all()
  assert int(sample.obs.robot.dwell.max()) == n
  assert sample.obs.robot.aug_code.dtype == torch.long and set(sample.obs.robot.aug_code.unique().tolist()) == {0, 1, 2, 3}
  assert sample.action.dtype == sample.reward.dtype == torch.float32


def test_sampled_tensors_for_build_graph_and_autograd():
  W = 8
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), W, seed=51, policy='uniform')
  replay = ReplayBuffer(cfg, 5000, DEVICE)
  replay.insert(batch)
  gen = generator(52)
  sample = replay.sample(512, gen)
  floats = ('q', 'qd', 'tip_pos', 'tip_rot', 'score')
  for name in RobotState._fields:
    for obs in (sample.obs, sample.next_obs):
      x = getattr(obs.robot, name)
      assert not x.requires_grad and x.grad_fn is None, name
      if name in floats:
        assert x.dtype == torch.float32, name
      elif name == 'target_done':
        assert x.dtype == torch.bool, name
      else:
        assert x.dtype == torch.long, name
  for x in sample.obs.scene:
    assert not x.requires_grad and x.dtype == torch.float32
  assert not sample.action.requires_grad and sample.action.dtype == torch.float32
  assert not sample.reward.requires_grad and sample.reward.dtype == torch.float32
  assert sample.terminal.dtype == torch.bool and not sample.terminal.requires_grad
  graph = build_graph(sample.obs, Topology(cfg, DEVICE))
  for x in graph:
    assert x.dtype == torch.float32 and not x.requires_grad and x.isfinite().all()
  # Ranges of the input features the network relies on, from rows that came through int8.
  assert graph.globals[:, 0].min() >= -1 and graph.globals[:, 0].max() <= 1
  assert graph.nodes[:, :4, 14].min() >= -1 and graph.nodes[:, :4, 14].max() <= 1


def test_real_path_from_act_to_one_update_keeps_the_ring_free_of_graphs():
  """TD3.act plus noise into the buffer, real and hindsight into the ring, a sample into train_step."""
  W = 8
  cfg = env_cfg(num_targets=3)
  pool = _pool_for(4, 3)
  topo = Topology(cfg, DEVICE)
  small = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2,
                   head_sizes=(8,))
  train = TrainCfg(batch_size=16, fused_steps=1, compile=False, matmul_precision='highest', hindsight_flush_episodes=1)
  td3 = TD3(small, train, topo, DEVICE)
  env = BalletEnv(cfg, W, pool, DEVICE, seed=52)
  buf = EpisodeBuffer(cfg, W, DEVICE)
  replay = ReplayBuffer(cfg, 2000, DEVICE)
  gen = generator(53)
  queue = HindsightQueue(cfg, train, gen)
  with torch.enable_grad():
    for step in range(cfg.max_steps + 2):
      obs = env.observe()
      noise = torch.randn(W, 4, 7, generator=gen, device=DEVICE)
      action = (td3.act(build_graph(obs, topo)) + 0.7 * noise).clamp(-1, 1)
      assert not action.requires_grad, 'TD3.act is under no_grad, so nothing can follow the action into the buffer'
      out = env.step(action)
      batch = buf.append(obs, action, out, behaviour_step=step)
      assert not buf.action.requires_grad
      if batch is not None:
        replay.insert(batch)
        relabelled = queue.add(batch)
        if relabelled is not None:
          replay.insert(relabelled)
    assert replay.is_hindsight.any()
    sample = replay.sample(16, gen)
    before = sample.action.clone()
    losses = td3.train_step(Batch(build_graph(sample.obs, topo), build_graph(sample.next_obs, topo), sample.action, sample.reward, sample.terminal))
  assert torch.isfinite(losses.critic) and torch.isfinite(losses.policy)
  for x in (replay.action, replay.reward, replay.robot.q, replay.scene.target_pos, sample.action, sample.obs.robot.q):
    assert not x.requires_grad and x.grad is None
  assert torch.equal(sample.action, before)


def test_a_graph_carrying_action_leaves_no_graph_in_the_buffer_or_the_ring():
  """TD3.act is under no_grad, but another acting path might not be, and a ring keeps what it holds for a lap.

  env.step keeps such a graph in its own state, because it does not guard itself. That is the
  env's to fix. Here only the replay side is under test.
  """
  W = 4
  env, _ = _env_with_early_endings(W)
  buf = EpisodeBuffer(env.cfg, W, DEVICE)
  replay = ReplayBuffer(env.cfg, 256, DEVICE)
  parameter = torch.nn.Parameter(torch.zeros(W, 4, 7, device=DEVICE))
  finished = 0
  for _ in range(40):
    obs = env.observe()
    action = parameter * 1.0
    out = env.step(action)
    batch = buf.append(obs, action, out)
    assert all(not x.requires_grad for x in (*buf.robot, buf.action, buf.reward, buf.totals))
    if batch is not None:
      assert all(not x.requires_grad for x in (*batch.robot, *batch.scene, batch.action, batch.reward, batch.stats))
      replay.insert(batch)
      finished += 1
  assert finished > 0 and action.requires_grad
  assert all(not x.requires_grad for x in (*replay.robot, *replay.scene, replay.action, replay.reward, replay.stats))


def test_generators_must_be_on_the_replay_device_and_it_is_loud():
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=3)
  replay = ReplayBuffer(cfg, 32, DEVICE, 4)
  replay.insert(make_batch([0, 1], [5, 6]))
  with pytest.raises(RuntimeError):
    replay.sample(4, torch.Generator(device='cpu'))
  batch = real_episodes(env_cfg(), _pool_for(4, 10), 4, seed=54, policy='uniform')
  with pytest.raises(RuntimeError):
    relabel(batch, env_cfg(), TrainCfg(), torch.Generator(device='cpu'))
  # 'cuda' and 'cuda:0' both work on a replay declared with either spelling.
  for spelling in ('cuda', 'cuda:0'):
    other = ReplayBuffer(cfg, 32, spelling, 4)
    other.insert(make_batch([0], [5]))
    assert other.sample(4, torch.Generator(device='cuda:0').manual_seed(1)).reward.shape == (4,)
    assert other.sample(4, torch.Generator(device='cuda').manual_seed(1)).reward.shape == (4,)


def test_same_seed_same_sample_same_relabel():
  cfg = env_cfg()
  batch = real_episodes(cfg, _pool_for(4, 10), 8, seed=55, policy='uniform')
  a = relabel(batch, cfg, TrainCfg(), generator(56))
  b = relabel(batch, cfg, TrainCfg(), generator(56))
  for x, y in zip(a.robot, b.robot):
    assert torch.equal(x, y)
  assert torch.equal(a.scene.target_pos, b.scene.target_pos) and torch.equal(a.action, b.action)
  replay = ReplayBuffer(cfg, 5000, DEVICE)
  replay.insert(a)
  assert torch.equal(replay.sample(64, generator(1)).action, replay.sample(64, generator(1)).action)


def test_batch_order_is_the_fifo_order():
  """Oldest first means episode 0 of a batch goes before episode 1, and real before hindsight."""
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=3)
  replay = ReplayBuffer(cfg, 10, DEVICE, 8)
  replay.insert(make_batch([0, 1, 2], [4, 4, 4]))  # 12 transitions in a ring of 10.
  sample = replay.fetch(torch.arange(replay.size, device=DEVICE))
  tags = sample.obs.robot.q[:, 0, 0].long().tolist()
  # Rows carry e * 1024 + t. Two of episode 0 are gone, then 1 and 2 are whole.
  assert tags == [2, 3, 1024, 1025, 1026, 1027, 2048, 2049, 2050, 2051]
  replay.insert(make_batch([3], [3]))
  sample = replay.fetch(torch.arange(replay.size, device=DEVICE))
  assert sample.obs.robot.q[:, 0, 0].long().tolist() == [1025, 1026, 1027, 2048, 2049, 2050, 2051, 3072, 3073, 3074]
  assert replay.evicted == 5 and replay.inserted == 15


def _memory_row(cfg, capacity: int) -> dict[str, float]:
  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  before = torch.cuda.memory_allocated()
  torch.cuda.reset_peak_memory_stats()
  replay = ReplayBuffer(cfg, capacity, DEVICE)
  allocated = torch.cuda.memory_allocated() - before
  return dict(replay=replay, allocated=allocated, before=before)


def _nvidia_used_mib() -> int:
  out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=False).stdout
  import os
  pid = str(os.getpid())
  return sum(int(line.split(',')[1]) for line in out.splitlines() if line.split(',')[0].strip() == pid)


@pytest.mark.parametrize('R,T,limit_gib', [(4, 10, 3.0), (8, 40, 1.5)])
def test_memory_at_full_capacity(R, T, limit_gib):
  """Fills the ring past one lap with episodes of the real shape, then samples a megabatch."""
  cfg = env_cfg(num_robots=R, num_targets=T)
  train = TrainCfg()
  capacity = train.replay_capacity(T)
  row = _memory_row(cfg, capacity)
  replay: ReplayBuffer = row['replay']
  bytes_per_row = sum(x.element_size() * int(np.prod(x.shape[1:])) for x in (
    *replay.robot, replay.action, replay.reward, replay.terminal, replay.row_episode))
  n_table = sum(x.element_size() * int(np.prod(x.shape[1:])) for x in (
    *replay.scene, replay.behaviour_step, replay.is_hindsight, replay.stats, replay.first_transition))
  computed = bytes_per_row * replay.num_rows + n_table * replay.max_episodes + 8 * capacity
  assert row['allocated'] == pytest.approx(computed, rel=0.01)

  # Episodes of the real shape, without an env: only sizes matter here, and every value is live.
  L = cfg.max_steps
  n = 512
  gen = generator(61)
  robot = row_storage(cfg, (n, L + 1), DEVICE)
  robot = robot._replace(q=torch.rand(*robot.q.shape, generator=gen, device=DEVICE))
  scene = Scene(*[torch.rand(n, *s, generator=gen, device=DEVICE) for s in (
    (R, 3), (R, 3, 3), (30, 3), (30, 3, 3), (30, 3), (T, 3), (T, 3, 3), (R, 7))])
  batch = EpisodeBatch(robot, torch.rand(n, L, R, 7, generator=gen, device=DEVICE), torch.rand(n, L, generator=gen, device=DEVICE),
                       torch.full((n,), L, device=DEVICE), scene, torch.zeros(n, dtype=torch.long, device=DEVICE),
                       torch.zeros(n, dtype=torch.bool, device=DEVICE), torch.zeros(n, 12, device=DEVICE))
  start = time.perf_counter()
  inserts = 0
  while replay.inserted < 1.2 * capacity:
    replay.insert(batch)
    inserts += 1
  torch.cuda.synchronize()
  fill_s = time.perf_counter() - start
  assert replay.size == capacity and replay.extra_evicted == 0
  sample = replay.sample(2560, gen)
  torch.cuda.synchronize()
  # At the real size the ids say what each live position must hold.
  position = torch.randint(capacity, (200_000,), generator=gen, device=DEVICE)
  ids = replay.evicted + position
  t, ep = ids % L, (ids // L) % n
  got = replay.fetch(position)
  assert torch.equal(got.obs.robot.q, robot.q[ep, t]) and torch.equal(got.next_obs.robot.q, robot.q[ep, t + 1])
  assert torch.equal(got.action, batch.action[ep, t]) and torch.equal(got.reward, batch.reward[ep, t])
  assert torch.equal(got.terminal, t == L - 1)
  assert torch.equal(got.obs.scene.target_rot, scene.target_rot[ep])
  assert int(replay.transition_row.max()) < replay.num_rows and int(replay.row_episode.max()) < replay.max_episodes
  del got, ids, t, ep, position
  allocated_gib = (torch.cuda.memory_allocated() - row['before']) / 2**30
  peak_gib = (torch.cuda.max_memory_allocated() - row['before']) / 2**30
  reserved_gib = torch.cuda.memory_reserved() / 2**30
  used_mib = _nvidia_used_mib()
  print(f'\n{R}r{T}t capacity {capacity}, table {replay.max_episodes}, rows {replay.num_rows}: '
        f'{bytes_per_row} B per row, {n_table} B per table slot, ring alone {row["allocated"] / 2**30:.3f} GiB. '
        f'After {inserts} inserts of {n} episodes in {fill_s:.1f} s, allocated {allocated_gib:.3f} GiB, '
        f'peak {peak_gib:.3f} GiB (template and a 2560 sample included), reserved {reserved_gib:.3f} GiB, '
        f'nvidia-smi for this process {used_mib} MiB. Live episodes about {capacity // L}.')
  assert row['allocated'] / 2**30 < limit_gib
  assert sample.obs.robot.q.shape == (2560, R, 7)
  del replay, batch, robot, scene, sample, row
  torch.cuda.synchronize()
  torch.cuda.empty_cache()


def _full_size_batch(cfg, n: int, seed: int) -> EpisodeBatch:
  """Timed-out episodes of the real shape, tips scattered so that nothing is solved by accident."""
  R, T, L = cfg.world.num_robots, cfg.num_targets, cfg.max_steps
  gen = generator(seed)
  robot = row_storage(cfg, (n, L + 1), DEVICE)
  robot = robot._replace(q=torch.rand(*robot.q.shape, generator=gen, device=DEVICE),
                         tip_pos=torch.rand(*robot.tip_pos.shape, generator=gen, device=DEVICE),
                         tip_rot=torch.eye(3, device=DEVICE).expand(*robot.tip_rot.shape).clone())
  scene = Scene(*[torch.rand(n, *s, generator=gen, device=DEVICE) for s in (
    (R, 3), (R, 3, 3), (30, 3), (30, 3, 3), (30, 3), (T, 3), (T, 3, 3), (R, 7))])
  return EpisodeBatch(robot, torch.rand(n, L, R, 7, generator=gen, device=DEVICE), torch.rand(n, L, generator=gen, device=DEVICE),
                      torch.full((n,), L, device=DEVICE), scene, torch.zeros(n, dtype=torch.long, device=DEVICE),
                      torch.zeros(n, dtype=torch.bool, device=DEVICE), torch.zeros(n, 12, device=DEVICE))


@pytest.mark.parametrize('R,T,W', [(4, 10, 64), (4, 10, 4096), (8, 40, 64), (8, 40, 1024)])
def test_memory_and_time_of_the_episode_buffer_and_a_flush(R, T, W):
  cfg = env_cfg(num_robots=R, num_targets=T)
  train = TrainCfg()
  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  before = torch.cuda.memory_allocated()
  torch.cuda.reset_peak_memory_stats()
  buf = EpisodeBuffer(cfg, W, DEVICE)
  allocated = torch.cuda.memory_allocated() - before
  del buf
  batch = _full_size_batch(cfg, W, 62)
  held = torch.cuda.memory_allocated() - before
  gen = generator(63)
  times = []
  for _ in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = relabel(batch, cfg, train, gen)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
  peak = torch.cuda.max_memory_allocated() - before
  print(f'\n{R}r{T}t W={W}: EpisodeBuffer {allocated / 2**30:.3f} GiB, a finished batch of W episodes {held / 2**30:.3f} GiB, '
        f'relabel peak {peak / 2**30:.3f} GiB, relabel {min(times) * 1e3:.0f} ms (best of 3), max_steps {cfg.max_steps}, '
        f'nvidia-smi this process {_nvidia_used_mib()} MiB')
  assert allocated / 2**30 < 5.0 and out is not None
  del batch, out
  torch.cuda.empty_cache()


def test_a_table_that_wraps_with_long_episodes_evicts_nothing_extra():
  """At real rates the table wraps after 250,000 episodes, with at most 16,667 live."""
  cfg = env_cfg()
  L = cfg.max_steps
  C = 20 * L
  replay = ReplayBuffer(cfg, C, DEVICE, max_episodes=64)
  batch = _full_size_batch(cfg, 7, 65)
  rounds = 40
  for _ in range(rounds):
    replay.insert(batch)
  assert replay.episodes == 7 * rounds > 2 * replay.max_episodes
  assert replay.extra_evicted == 0 and replay.size == C and replay.evicted == replay.inserted - C
  position = torch.arange(C, device=DEVICE)
  ids = replay.evicted + position
  t, ep = ids % L, (ids // L) % 7
  got = replay.fetch(position)
  assert torch.equal(got.obs.robot.q, batch.robot.q[ep, t]) and torch.equal(got.next_obs.robot.q, batch.robot.q[ep, t + 1])
  assert torch.equal(got.obs.scene.target_pos, batch.scene.target_pos[ep])


def test_no_growth_over_many_insert_and_sample_cycles():
  cfg = env_cfg()
  replay = ReplayBuffer(cfg, 5000, DEVICE, max_episodes=32)
  buf_batch = _full_size_batch(cfg, 8, 66)
  queue = HindsightQueue(cfg, TrainCfg(hindsight_flush_episodes=8), generator(67))
  gen = generator(68)
  for _ in range(5):
    replay.insert(buf_batch)
    replay.insert(queue.add(buf_batch))
  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  before = torch.cuda.memory_allocated()
  for _ in range(60):
    replay.insert(buf_batch)
    replay.insert(queue.add(buf_batch))
    sample = replay.sample(256, gen)
    build_graph(sample.obs, Topology(cfg, DEVICE))
  del sample
  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  assert torch.cuda.memory_allocated() == before
  assert queue.pending == []


@pytest.mark.parametrize('R,T', [(4, 10), (8, 40)])
def test_insert_time_once_the_table_has_wrapped(R, T):
  """The oversize branch, not the steady state: 64 episodes into 32 slots writes 32 of them and
  evicts everything older. test_insert_time_after_the_default_table_has_wrapped has the steady state."""
  cfg = env_cfg(num_robots=R, num_targets=T)
  capacity = TrainCfg().replay_capacity(T)
  replay = ReplayBuffer(cfg, capacity, DEVICE, max_episodes=32)
  batch = _full_size_batch(cfg, 64, 64)
  for _ in range(3):
    replay.insert(batch)
  assert replay.episodes > replay.max_episodes
  times = []
  for _ in range(10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    replay.insert(batch)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
  print(f'\n{R}r{T}t insert of 64 episodes, table wrapped: {min(times) * 1e3:.2f} ms best, {max(times) * 1e3:.2f} ms worst, '
        f'extra_evicted {replay.extra_evicted}')
  assert replay.extra_evicted > 0
  del replay, batch
  torch.cuda.empty_cache()


@pytest.mark.parametrize('R,T', [(4, 10), (8, 40)])
def test_insert_time_after_the_default_table_has_wrapped(R, T):
  """The steady state after the table's first lap: real capacity, default table, nothing evicted early."""
  cfg = env_cfg(num_robots=R, num_targets=T)
  capacity = TrainCfg().replay_capacity(T)
  replay = ReplayBuffer(cfg, capacity, DEVICE)
  n, L = 512, cfg.max_steps
  batch = _full_size_batch(cfg, n, 64)
  small = select_episodes(batch, torch.arange(64, device=DEVICE))

  def best_ms(repeats: int = 16) -> float:
    times = []
    for _ in range(repeats):
      torch.cuda.synchronize()
      start = time.perf_counter()
      replay.insert(small)
      torch.cuda.synchronize()
      times.append(time.perf_counter() - start)
    return min(times) * 1e3

  before_wrap = best_ms()
  assert replay.episodes <= replay.max_episodes
  while replay.episodes <= replay.max_episodes:
    replay.insert(batch)
  assert replay.extra_evicted == 0 and replay.size == capacity and replay.evicted == replay.inserted - capacity
  # The last inserts were whole copies of `batch`, so ids give what each of the newest rows holds.
  gen = generator(69)
  newest = 8 * n * L if 8 * n * L < capacity else capacity
  position = capacity - 1 - torch.randint(newest, (200_000,), generator=gen, device=DEVICE)
  back = capacity - 1 - position  # 0 is the last transition inserted.
  t, ep = L - 1 - back % L, n - 1 - (back // L) % n
  got = replay.fetch(position)
  assert torch.equal(got.obs.robot.q, batch.robot.q[ep, t]) and torch.equal(got.next_obs.robot.q, batch.robot.q[ep, t + 1])
  assert torch.equal(got.obs.scene.target_rot, batch.scene.target_rot[ep]) and torch.equal(got.terminal, t == L - 1)
  del got, position, back, t, ep
  after_wrap = best_ms()
  assert replay.extra_evicted == 0 and replay.episodes > replay.max_episodes
  print(f'\n{R}r{T}t insert of 64 episodes, table of {replay.max_episodes}: {before_wrap:.2f} ms before its first lap, '
        f'{after_wrap:.2f} ms after, best of 16, extra_evicted {replay.extra_evicted}, {replay.episodes} episodes so far')
  del replay, batch, small
  torch.cuda.empty_cache()


# D: the whole path against the original code, features included. The goldens stub
# make_graph_features out of _write_episode, so hindsight rows never met build_graph.
# Rows come from this env with the mesh_tool arm on the original's own start states, so the
# unmodified generate_hindsight_trajectory recomputes the same TCP from the stored q.

_ORIGINAL_DRIVER = r'''
import copy, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import replay_make_goldens as G
from roboballet.agents import actor, trajectory as trajectory_lib
from roboballet.environments import planning_env
from roboballet.launch import hparams
from roboballet.train import stats as stats_lib

z = dict(np.load(sys.argv[2]))
R, T, k, rts = int(z['R']), int(z['T']), int(z['k']), bool(z['rts'])
states = G.start_states(R, T, len(z['length']))
cfg = hparams.make_optimised_configs(num_targets=T, num_obstacles=30, num_robots=R, pregen=True)
env_config = G.make_env_config(R, T, rts=rts)
feature_config = cfg.model_config.feature_config


class Forced:
  def begin(self, chosen, arm, time):
    self.chosen = [int(i) for i in np.flatnonzero(chosen)]
    self.values = []
    for i in self.chosen:
      self.values += [int(arm[i]), int(time[i])]

  def sample(self, population, count):
    assert len(self.chosen) == count, (self.chosen, count)
    assert set(self.chosen) <= set(population), (self.chosen, population)
    return list(self.chosen)

  def randint(self, a, b):
    value = self.values.pop(0)
    assert a <= value <= b, (a, value, b)
    return value


forced = Forced()
actor.random = forced

codes = []
_original_aug = planning_env.BoxInfo.get_augmentation


def _recording(box, selection):
  codes.append(selection)
  return _original_aug(box, selection)


planning_env.BoxInfo.get_augmentation = _recording

results = {}
for i, start in enumerate(states):
  A = int(z['length'][i])
  env = planning_env.PlanningEnv(env_config, start_state=copy.deepcopy(start))
  env.reset()
  spec = env.observation_spec()
  place = np.asarray(start.initial_robot_placements, dtype=np.float64)
  observations = []
  for t in range(A + 1):
    tip = np.tile(np.eye(4), (R, 1, 1))
    tip[:, :3, 3] = z['tip_pos'][i, t]
    tip[:, :3, :3] = z['tip_rot'][i, t]
    relative = np.matmul(np.linalg.inv(place), tip).astype(np.float32)
    observations.append(dict(
      robots=dict(base_poses=place.astype(np.float32), tip_relative_poses=relative,
                  joint_configurations=z['q'][i, t], joint_velocities=z['qd'][i, t]),
      targets=dict(done=z['done'][i, t])))
  actions = [np.array(z['action'][i, t] * 2.175, dtype=np.float32) for t in range(A)]
  tr = trajectory_lib.Trajectory(start_state=copy.deepcopy(start), observations=observations, actions=actions,
                                 rewards=[float(r) for r in z['reward'][i, :A]], target_states=[])
  stats = stats_lib.TrajectoryStats(final_score=1.0, total_reward=2.0, total_collision_penalty=3.0,
                                    total_collision_robot_steps=4, total_mean_acceleration=5.0, num_targets_done=6,
                                    episode_length=A + 1, env_sampling_time_s=7.0, episode_wall_time_s=8.0,
                                    episode_write_wall_time_s=9.0, network_step=10)
  forced.begin(z['chosen'][i], z['arm'][i], z['time'][i])
  del codes[:]
  out = actor.generate_hindsight_trajectory(tr, stats, env_config, num_targets=k)
  if out is None:
    results[f'{i}_none'] = np.array(True)
    continue
  h, h_stats = out
  # Aug codes of the stored observations. The final one was drawn after the terminal step's own.
  O = len(start.generated_obstacles)
  H = len(h.actions)
  assert len(codes) == O * (H + 2), (len(codes), O, H)
  rows = [codes[O * t: O * (t + 1)] for t in range(H)] + [codes[O * (H + 1):]]
  client = G.FakeClient()
  actor._write_episode(h, h_stats, env_config, spec, feature_config, client)
  steps = client.writer.steps
  assert len(steps) == H + 1
  items = client.writer.items
  assert len(items) == H
  results[f'{i}_length'] = np.array(H)
  results[f'{i}_aug'] = np.array(rows, dtype=np.int8)
  results[f'{i}_nodes'] = np.stack([np.asarray(s['observation']['nodes']) for s in steps])
  results[f'{i}_edges'] = np.stack([np.asarray(s['observation']['edges']) for s in steps])
  results[f'{i}_globals'] = np.stack([np.asarray(s['observation']['globals']).reshape(-1) for s in steps])
  results[f'{i}_action'] = np.stack([it['action'] for it in items])
  results[f'{i}_reward'] = np.array([it['reward'] for it in items], dtype=np.float32)
  results[f'{i}_terminal'] = np.array([it['terminal'] for it in items], dtype=np.int8)
  results[f'{i}_stats_is_hindsight'] = np.array(items[0]['trajectory_stats']['is_hindsight_replay'])
  results[f'{i}_stats_network_step'] = np.array(items[0]['trajectory_stats']['network_step'])
  rows_h = G.obs_rows(h.observations)
  results[f'{i}_dwell'] = rows_h['dwell']
  results[f'{i}_done'] = rows_h['done']
  results[f'{i}_score'] = rows_h['score']
  results[f'{i}_step'] = rows_h['step']
  results[f'{i}_q'] = rows_h['q']
  results[f'{i}_qd'] = rows_h['qd']
  results[f'{i}_tip_err'] = np.array(max(np.abs(rows_h['tip_pos'] - z['tip_pos'][i, :H + 1]).max(),
                                         np.abs(rows_h['tip_rot'] - z['tip_rot'][i, :H + 1]).max()))
np.savez(sys.argv[3], **results)
print('original driver done', len(states))
'''

ORIGINAL_CASES = {
  # name: (R, T, fraction, return to start, policy, W)
  '4r10t_default_uniform': (4, 10, 0.4, False, 'uniform', 6),
  '4r10t_all_wander': (4, 10, 1.0, False, 'wander', 6),
  '4r10t_all_rts_wander': (4, 10, 1.0, True, 'wander', 4),
  '4r10t_all_rts_wide': (4, 10, 1.0, True, 'wide', 4),
  '4r40t_default_uniform': (4, 40, 0.4, False, 'uniform', 2),
  '8r10t_all_wander': (8, 10, 1.0, False, 'wander', 2),
  '8r40t_default_uniform': (8, 40, 0.4, False, 'uniform', 2),
}


def _run_original_driver(rows_path, out_path, script_path, *extra) -> None:
  project = PROJECT
  venv = project / 'venv/bin/python'
  if not venv.exists():
    pytest.fail(f'{venv} is missing, so this check did not run')
  base_prefix = subprocess.run([str(venv), '-c', 'import sys; print(sys.base_prefix)'], capture_output=True,
                               text=True, check=True).stdout.strip()
  import os
  env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=str(project), LD_LIBRARY_PATH=f'{base_prefix}/lib')
  golden = REPO / 'tests/golden'
  done = subprocess.run([str(venv), str(script_path), str(golden), str(rows_path), str(out_path), *extra], cwd=project, env=env,
                        capture_output=True, text=True, timeout=540)
  # Exit 0 can follow a teardown message from the Reverb server, and a real failure writes no output.
  assert done.returncode == 0 and out_path.exists(), done.stderr[-3000:]


@pytest.mark.parametrize('name', ORIGINAL_CASES)
def test_hindsight_through_the_ring_against_original_code(name, tmp_path):
  from roboballet_mjlab.scenes.io import load_start_states

  R, T, fraction, rts, policy, W = ORIGINAL_CASES[name]
  cfg = env_cfg(num_robots=R, num_targets=T, arm='mesh_tool', return_to_start_reward_weight=0.25 if rts else 0.0,
                return_to_start_required=rts)
  train = TrainCfg(hindsight_fraction=fraction)
  k = int(fraction * T)
  pool = load_start_states(DATA / f'start_states/{R}r_{T}t_30o.pb', DEVICE, limit=W)
  batch = real_episodes(cfg, pool, W, seed=71, policy=policy)
  final = batch.robot.target_done[torch.arange(W, device=DEVICE), batch.length]
  picks = draw_picks(final, batch.length, R, k, generator(72))

  rows_path, out_path, script_path = tmp_path / 'rows.npz', tmp_path / 'out.npz', tmp_path / 'driver.py'
  script_path.write_text(_ORIGINAL_DRIVER)
  np.savez(rows_path, R=R, T=T, k=k, rts=rts, length=batch.length.cpu().numpy(), q=batch.robot.q.cpu().numpy(),
           qd=batch.robot.qd.cpu().numpy(), tip_pos=batch.robot.tip_pos.cpu().numpy(), tip_rot=batch.robot.tip_rot.cpu().numpy(),
           done=batch.robot.target_done.cpu().numpy(), action=batch.action.cpu().numpy(), reward=batch.reward.cpu().numpy(),
           chosen=picks.chosen.cpu().numpy(), arm=picks.arm.cpu().numpy(), time=picks.time.cpu().numpy())
  _run_original_driver(rows_path, out_path, script_path)
  z = dict(np.load(out_path))

  out = relabel(batch, cfg, train, generator(73), picks)
  kept = eligible(batch, k).nonzero().flatten().tolist()
  assert sorted(kept) == sorted(i for i in range(W) if f'{i}_length' in z)
  # The original's own aug draws replace ours, then everything goes through the ring.
  aug = out.robot.aug_code.clone()
  for j, i in enumerate(kept):
    H = int(out.length[j])
    assert int(z[f'{i}_length']) == H, f'episode {i}: length'
    aug[j, : H + 1] = torch.as_tensor(z[f'{i}_aug'], device=DEVICE, dtype=aug.dtype)
  out = out._replace(robot=out.robot._replace(aug_code=aug))
  replay = ReplayBuffer(cfg, int(out.length.sum()), DEVICE)
  replay.insert(out)
  assert replay.size == int(out.length.sum())
  sample = replay.fetch(torch.arange(replay.size, device=DEVICE))
  topo = Topology(cfg, DEVICE)
  graph, next_graph = [], []
  for start in range(0, replay.size, 512):
    slice_ = lambda obs: Obs(RobotState(*[x[start : start + 512] for x in obs.robot]),
                             Scene(*[x[start : start + 512] for x in obs.scene]))
    graph.append(build_graph(slice_(sample.obs), topo))
    next_graph.append(build_graph(slice_(sample.next_obs), topo))
  stack = lambda graphs, field: torch.cat([getattr(g, field) for g in graphs]).cpu().numpy()

  errors = dict(nodes=0.0, edges=0.0, globals=0.0, action=0.0, reward=0.0, score=0.0, tip=0.0)
  position = 0
  tally = dict(transitions=0, dwelling=0, solved=0, takeover=0, takeover_over_dwell=0, beyond_unit=0, shorter=0)
  for j, i in enumerate(kept):
    H = int(out.length[j])
    index = np.arange(position, position + H)
    position += H
    for field in ('nodes', 'edges', 'globals'):
      want = z[f'{i}_{field}']
      assert want.shape[0] == H + 1
      s = stack(graph, field)[index]
      n = stack(next_graph, field)[index]
      if field == 'globals':
        s, n = s.reshape(H, -1), n.reshape(H, -1)
      errors[field] = max(errors[field], float(np.abs(s - want[:H]).max()), float(np.abs(n - want[1:]).max()))
    h_action = sample.action[index].cpu().numpy()
    errors['action'] = max(errors['action'], float(np.abs(h_action - z[f'{i}_action']).max()))
    errors['reward'] = max(errors['reward'], float(np.abs(sample.reward[index].cpu().numpy() - z[f'{i}_reward']).max()))
    assert sample.terminal[index].cpu().numpy().tolist() == z[f'{i}_terminal'].astype(bool).tolist(), f'episode {i}: terminal'
    assert sample.obs.robot.dwell[index].cpu().numpy().tolist() == z[f'{i}_dwell'][:H].tolist(), f'episode {i}: dwell'
    assert sample.next_obs.robot.dwell[index].cpu().numpy().tolist() == z[f'{i}_dwell'][1:].tolist()
    assert sample.obs.robot.target_done[index].cpu().numpy().tolist() == z[f'{i}_done'][:H].tolist(), f'episode {i}: done'
    assert sample.next_obs.robot.target_done[index].cpu().numpy().tolist() == z[f'{i}_done'][1:].tolist()
    assert sample.obs.robot.step[index].cpu().numpy().tolist() == z[f'{i}_step'][:H].tolist()
    assert np.array_equal(sample.obs.robot.q[index].cpu().numpy(), z[f'{i}_q'][:H])
    assert np.array_equal(sample.next_obs.robot.qd[index].cpu().numpy(), z[f'{i}_qd'][1:])
    errors['score'] = max(errors['score'], float(np.abs(sample.next_obs.robot.score[index].cpu().numpy() - z[f'{i}_score'][1:]).max()))
    errors['tip'] = max(errors['tip'], float(z[f'{i}_tip_err']))
    assert float(z[f'{i}_stats_is_hindsight']) == 1
    tally['transitions'] += H
    tally['dwelling'] += int((z[f'{i}_dwell'] > 0).sum())
    tally['solved'] += int(z[f'{i}_done'][H].sum())
    near_home = (np.abs(HOME.cpu().numpy() - z[f'{i}_q'][:H]) < cfg.return_to_start_threshold).all(-1)
    takeover = z[f'{i}_done'][:H].all(1, keepdims=True) & near_home
    tally['takeover'] += int(takeover.sum())
    tally['takeover_over_dwell'] += int((takeover & (z[f'{i}_dwell'][:H] > 0)).sum())
    tally['beyond_unit'] += int((np.abs(h_action) > 1.0).any(-1).sum())
    tally['shorter'] += int(H < int(batch.length[i]))
  print(f'\n{name}: {len(kept)} hindsight episodes, {tally}, max abs errors {errors}')
  assert errors['nodes'] < 1e-5 and errors['edges'] < 1e-5 and errors['globals'] < 1e-6
  assert errors['action'] < 2e-6 and errors['reward'] < 1e-6 and errors['score'] < 1e-6
  if fraction == 1.0 and policy in ('wander', 'wide'):
    assert tally['takeover'] > 0 and tally['takeover_over_dwell'] > 0, 'this case exists to reach those branches'
  if policy == 'wide':
    assert tally['beyond_unit'] > 0, 'this case exists to reach takeover requests beyond the action bounds'


# E: the ring against a live Reverb table, written by the unmodified _write_episode. Only the
# features are stubbed, to a row tag, and the rate limiter, which is the trainer's job here.

_REVERB_DRIVER = r'''
import dataclasses, sys, tempfile, types
import numpy as np

tempfile.tempdir = sys.argv[4]  # The server opens a checkpoint directory here.
import reverb
import tensorflow as tf
from roboballet.agents import actor, trajectory as trajectory_lib
from roboballet.train import stats as stats_lib


class Features:
  def __init__(self, tag):
    self.nodes = self.edges = self.globals = np.array([tag], dtype=np.float32)


actor.compute_features = types.SimpleNamespace(make_graph_features=lambda observation, spec, config: Features(observation))
TABLE = actor.REPLAY_TABLE_NAME
feature = lambda: {k: tf.TensorSpec((1,), tf.float32) for k in ('nodes', 'edges', 'globals')}
signature = dict(observation=feature(), observation_next=feature(), action=tf.TensorSpec((1,), tf.float32),
                 reward=tf.TensorSpec((), tf.float32), terminal=tf.TensorSpec((), tf.int8),
                 trajectory_stats={f.name: tf.TensorSpec((), tf.float32) for f in dataclasses.fields(stats_lib.TrajectoryStats)})
env_config = types.SimpleNamespace(base_config=types.SimpleNamespace(max_joint_velocity=2.0))
z = dict(np.load(sys.argv[2]))
results = {}
for case in range(int(z['cases'])):
  capacity = int(z[f'{case}_capacity'])
  table = reverb.Table(name=TABLE, sampler=reverb.selectors.Uniform(), remover=reverb.selectors.Fifo(), max_size=capacity,
                       rate_limiter=reverb.rate_limiters.MinSize(1), signature=signature)
  server = reverb.Server([table])
  client = reverb.Client(f'localhost:{server.port}')
  episode = 0
  for read, count in enumerate(z[f'{case}_batches']):
    for A in z[f'{case}_lengths'][episode : episode + count]:
      tags = [float(episode * 1024 + t) for t in range(int(A) + 1)]
      trajectory = trajectory_lib.Trajectory(start_state=None, observations=tags, target_states=[],
                                             actions=[np.array([tag * 2.0], dtype=np.float32) for tag in tags[:-1]],
                                             rewards=[tag / 2.0 for tag in tags[:-1]])
      actor._write_episode(trajectory, stats_lib.TrajectoryStats(final_score=float(episode)), env_config, None, None, client)
      episode += 1
    size = client.server_info()[TABLE].current_size
    items = {}
    # Uniform draws, 40 per item: a live item is missed with odds of e^-40, and an evicted one must never show up.
    for sample in client.sample(TABLE, num_samples=40 * size, emit_timesteps=False, unpack_as_table_signature=True):
      d = sample.data
      items[int(sample.info.key)] = [float(np.ravel(x)[0]) for x in (
        d['observation']['nodes'], d['observation_next']['nodes'], d['action'], d['reward'], d['terminal'], d['trajectory_stats']['final_score'])]
    results[f'{case}_{read}_size'] = np.array(size)
    results[f'{case}_{read}_items'] = np.array(sorted(items.values()), dtype=np.float64).reshape(-1, 6)
  server.stop()
np.savez(sys.argv[3], **results)
'''


def _reverb_cases() -> list[tuple[str, int, int, list[int], list[int]]]:
  """name, capacity, table slots, episode lengths, episodes per batch. Tables large enough that only FIFO evicts."""
  rng = np.random.default_rng(81)
  cases = []
  lengths = rng.integers(1, 40, size=60).tolist()
  cases.append(('mixed', 64, 64, lengths, [1] * len(lengths)))
  lengths = rng.integers(1, 40, size=60).tolist()
  batches = []
  while sum(batches) < len(lengths):
    batches.append(min(int(rng.integers(1, 5)), len(lengths) - sum(batches)))
  cases.append(('batched', 97, 97, lengths, batches))
  cases.append(('one_action_and_oversize', 16, 16, [1, 1, 16, 1, 17, 40, 2, 1, 15, 33, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 1, 2, 3]))
  lengths = rng.integers(9, 40, size=40).tolist()  # The default table only evicts by itself below a mean of 8.
  cases.append(('default_table', 97, max(2, 97 // 8), lengths, [1] * len(lengths)))
  return cases


def test_ring_equals_a_live_reverb_table(tmp_path):
  cases = _reverb_cases()
  rows_path, out_path, script_path = tmp_path / 'rows.npz', tmp_path / 'out.npz', tmp_path / 'driver.py'
  script_path.write_text(_REVERB_DRIVER)
  arrays = dict(cases=len(cases))
  for i, (_, capacity, _, lengths, batches) in enumerate(cases):
    assert sum(batches) == len(lengths)
    arrays.update({f'{i}_capacity': capacity, f'{i}_lengths': lengths, f'{i}_batches': batches})
  np.savez(rows_path, **arrays)
  _run_original_driver(rows_path, out_path, script_path, str(tmp_path))
  z = dict(np.load(out_path))
  cfg = env_cfg(num_robots=2, num_obstacles=3, num_targets=3)
  for i, (name, capacity, slots, lengths, batches) in enumerate(cases):
    replay = ReplayBuffer(cfg, capacity, DEVICE, slots)
    episode = compared = evicted_reads = partial = 0
    for read, count in enumerate(batches):
      ids = list(range(episode, episode + count))
      replay.insert(make_batch(ids, lengths[episode : episode + count]))
      episode += count
      assert replay.size == int(z[f'{i}_{read}_size']), (name, read, replay.counters())
      want = z[f'{i}_{read}_items']
      assert len(want) == replay.size, (name, read, 'an item was not drawn, or one that is gone was')
      got = replay.fetch(torch.arange(replay.size, device=DEVICE))
      s, next_s = got.obs.robot.q[:, 0, 0].cpu().numpy(), got.next_obs.robot.q[:, 0, 0].cpu().numpy()
      assert np.array_equal(s, want[:, 0]), (name, read, 'live transitions, oldest first')
      assert np.array_equal(next_s, want[:, 1]), (name, read, 'successor states')
      assert np.array_equal(got.terminal.cpu().numpy().astype(np.float64), want[:, 4]), (name, read, 'terminal')
      # In Reverb, action, reward and statistics carry the tag of their own item, so the writer keeps them
      # on the right transition. Here they must be the ones the same transition was inserted with.
      assert np.array_equal(want[:, 2], want[:, 0]) and np.array_equal(want[:, 3], want[:, 0] / 2.0)
      assert np.array_equal(want[:, 5], np.floor(want[:, 0] / 1024)), (name, read, 'statistics of the right episode')
      tag = torch.as_tensor(want[:, 0], device=DEVICE).long()
      e, t = tag // 1024, tag % 1024
      assert torch.equal(got.action, action_of(e, t)) and torch.equal(got.reward, reward_of(e, t))
      assert torch.equal(got.obs.scene.q0, scene_of(e).q0) and torch.equal(got.stats, stats_of(e))
      compared += replay.size
      evicted_reads += int(replay.evicted > 0)
      partial += int(s[0] % 1024 != 0)
    print(f'\n{name}: capacity {capacity}, {len(lengths)} episodes, {compared} live items compared across {len(batches)} reads, '
          f'{evicted_reads} reads after an eviction, {partial} with the oldest episode partly gone, {replay.counters()}')
    assert replay.extra_evicted == 0 and evicted_reads > 0 and partial > 0
