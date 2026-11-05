"""The lockstep trainer against agents/actor.py, train/train.py and DESIGN 14.2.

The recorded run below is a real env, replay and learner on a small model, so most checks read
one long run. tests/golden/trainer_mutation_check.py injects bugs and confirms these tests fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import CriticNet, PolicyNet, load_checkpoint
from roboballet_mjlab.agent.td3 import TD3, Batch, matmul_precision
from roboballet_mjlab.config import EnvCfg, ModelCfg, TrainCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.replay.replay import STAT
from roboballet_mjlab.structs import Graph
from tests.golden.agent_load import CHECKPOINT, GOLDEN as AGENT_GOLDEN, load_e6
from tests.test_sim_common import DEVICE, PROJECT, stored_scenes
from roboballet_mjlab.train.evaluator import Evaluator
from roboballet_mjlab.train.logger import MetricLogger, write_provenance
from roboballet_mjlab.train.profiler import Profiler
from roboballet_mjlab.train.trainer import (
  CHECKPOINT_NAME, FAILED_CHECKPOINT_NAME, EpisodeLatches, Trainer, atomic_save, behaviour_action,
)

REPO = Path(__file__).resolve().parents[1]
TB_JSON = PROJECT / 'runs/mjlab_port_analysis/algo/tb.json'
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
T = 3
RANDOM_EPISODES = 12
RECORDED_ROUNDS = 713  # Past 12 episodes of 50 actions in every staggered world, and between refreshes.
E7_TOLERANCE = 2e-3  # As tests/test_agent_networks.py.
E7_OBSERVED_MEAN = 2e-6
E7_OBSERVED_P999 = 5e-5


def small_pool(targets: int = T):
  scenes = stored_scenes(4)
  return scenes._replace(target_pos=scenes.target_pos[:, :targets], target_rot=scenes.target_rot[:, :targets])


def small_train_cfg(**changes) -> TrainCfg:
  # One fused call of 80 samples per round, 8 real and 8 hindsight transitions per round, min_size 400.
  base = dict(num_worlds=8, batch_size=16, fused_steps=5, replay_max_size=4000, hindsight_flush_episodes=4,
              compile=False, log_every_steps=100, checkpoint_every_s=1e9, profile=False)
  return TrainCfg(**{**base, **changes})


def make_trainer(run_dir, targets: int = T, model: ModelCfg = SMALL, **changes) -> Trainer:
  return Trainer(EnvCfg(num_targets=targets), small_train_cfg(**changes), model, small_pool(targets), DEVICE, run_dir)


class LiteralActor:
  """One actor, statement by statement. actor.py lines in the comments."""

  def __init__(self, random_episodes: int = 32, sigma0: float = 0.7, decay: float = 0.8) -> None:
    self.random_episodes, self.sigma0, self.decay = random_episodes, sigma0, decay
    self.network_step = 0  # :551
    self.episode_count = 0  # :584

  def start_episode(self, published_step: int) -> None:
    self.in_exploration = self.episode_count < self.random_episodes or self.network_step == 0  # :597-600
    passed_network_step = self.network_step  # :612
    self.network_step = published_step  # :224, then :566, inside run_episode.
    self.sigma = self.sigma0 * self.decay ** (passed_network_step / 100000.0)  # :242-246, from the argument.
    self.recorded_network_step = self.network_step  # :616

  def finish_episode(self) -> None:
    self.episode_count += 1  # :654


def assert_latches_match(latches: EpisodeLatches, actors: list[LiteralActor], where: str) -> None:
  use_random = latches.use_random.tolist()
  held = latches.held_step.tolist()
  finished = latches.finished.tolist()
  sigma = latches.sigma.tolist()
  for w, actor in enumerate(actors):
    assert use_random[w] == actor.in_exploration, f'{where}: world {w} random'
    assert held[w] == actor.network_step, f'{where}: world {w} held step'
    assert finished[w] == actor.episode_count, f'{where}: world {w} episodes'
    # The f32 of the same number, as jnp.float32(sigma) at actor.py:254.
    assert sigma[w] == pytest.approx(actor.sigma, rel=2e-7), f'{where}: world {w} sigma'


class Recorded:
  """Wraps a Trainer and keeps what crosses each seam, and drives the literal actors beside it."""

  def __init__(self, trainer: Trainer) -> None:
    self.t = t = trainer
    W, max_steps = t.num_worlds, t.env_cfg.max_steps
    self.actors = [LiteralActor(t.cfg.random_episodes_per_world) for _ in range(W)]
    self.actor_published = 0
    for actor in self.actors:
      actor.start_episode(0)
    self.inserts: list[SimpleNamespace] = []
    self.samples: list[SimpleNamespace] = []
    self.refreshes: list[tuple[int, int]] = []
    self.resets: list[tuple[int, list[int]]] = []
    self.actions: list[torch.Tensor] = []
    self.terminals: list[torch.Tensor] = []
    self.formula_errors = 0.0
    self.formula_checked = 0
    self.saturated = 0
    self.random_rows = 0
    self.policy_rows = 0
    self.episodes: list[SimpleNamespace] = []
    self.acting_snapshot = [p.clone() for p in t.td3.policy_acting.parameters()]
    self.world_start_round = [0] * W
    self.real_rows = 0
    self.hindsight_rows = 0
    self.max_steps = max_steps

    self._in_step = False
    self._uniform = self._normals = self._mean = None
    draw_uniform, draw_normals, act = t._draw_uniform, t._draw_normals, t.td3.act_behaviour

    def uniform():
      self._uniform = draw_uniform()
      return self._uniform

    def normals():
      self._normals = draw_normals()
      return self._normals

    def behaviour(graph):
      self._mean = act(graph)
      return self._mean

    t._draw_uniform, t._draw_normals, t.td3.act_behaviour = uniform, normals, behaviour

    env_reset, env_step = t.env.reset, t.env.step

    def reset(world_ids=None, scene_ids=None):
      if world_ids is not None and not self._in_step:  # Not the env's own reset of a finished world.
        self.resets.append((t.counters.rounds, world_ids.tolist()))
        for w in world_ids.tolist():
          self.world_start_round[w] = t.counters.rounds
      return env_reset(world_ids, scene_ids)

    def step(action):
      self._check_action(action)
      self.actions.append(action.clone())
      self._in_step = True
      out = env_step(action)
      self._in_step = False
      self.terminals.append(out.terminal.clone())
      return out

    t.env.reset, t.env.step = reset, step

    insert, sample = t.replay.insert, t.replay.sample

    def insert_(batch):
      self.inserts.append(SimpleNamespace(
        round=t.counters.rounds, hindsight=batch.is_hindsight.tolist(), length=batch.length.tolist(), base=batch.scene.base_pos.clone(),
        target=batch.scene.target_pos.clone(), behaviour=batch.behaviour_step.tolist(), network_stat=batch.stats[:, STAT['network_step']].tolist(),
        action=batch.action.clone(), first_step=batch.robot.step[:, 0].tolist(), inserted_before=t.replay.inserted))
      if batch.is_hindsight.all():
        self.hindsight_rows += int(batch.length.sum())
      else:
        assert not batch.is_hindsight.any()
        self.real_rows += int(batch.length.sum())
      return insert(batch)

    def sample_(n, generator):
      self.samples.append(SimpleNamespace(n=n, inserted=t.replay.inserted, sampled=t.replay.sampled, real_rows=self.real_rows, round=t.counters.rounds))
      return sample(n, generator)

    t.replay.insert, t.replay.sample = insert_, sample_
    refresh = t.td3.refresh_acting

    def refresh_():
      refresh()
      self.refreshes.append((t.counters.rounds, t.td3.grad_steps))
      self.acting_snapshot = [p.clone() for p in t.td3.policy_acting.parameters()]
      for mine, online in zip(t.td3.policy_acting.parameters(), t.td3.policy.parameters()):
        assert torch.equal(mine, online)

    t.td3.refresh_acting = refresh_

  def _check_action(self, action: torch.Tensor) -> None:
    """Rebuilds the action from the recorded draws, per actor.py:232-239 and :159-163."""
    t = self.t
    latches = t.latches
    uniform = self._uniform
    if latches.use_random.all():  # The network must not have run at all, actor.py:232-239.
      assert self._mean is None and self._normals is None
      expected = uniform
    else:
      noisy = self._mean + latches.sigma[:, None, None] * self._normals
      self.saturated += int((noisy.abs() > 1).sum())
      expected = torch.where(latches.use_random[:, None, None], uniform, noisy.clamp(-1, 1))
    assert torch.equal(action, expected)
    assert (uniform >= -1).all() and (uniform <= 1).all()
    assert uniform.min() < -0.5 and uniform.max() > 0.5
    self.random_rows += int(latches.use_random.sum())
    self.policy_rows += int((~latches.use_random).sum())
    self._uniform = self._normals = self._mean = None

  def round(self) -> None:
    t = self.t
    before_refreshes = len(self.refreshes)
    t.round()
    # Published after learning, once as many episodes as worlds have ended, by what the td3 says here.
    if len(self.refreshes) > before_refreshes:
      self.actor_published = self.refreshes[-1][1]
    ended = self.terminals[-1].nonzero().flatten().tolist()
    for w in ended:
      self.actors[w].finish_episode()
      self.episodes.append(SimpleNamespace(
        world=w, end_round=t.counters.rounds - 1, start_round=self.world_start_round[w], fetched=self.actors[w].recorded_network_step))
      self.world_start_round[w] = t.counters.rounds
      self.actors[w].start_episode(self.actor_published)
    assert_latches_match(t.latches, self.actors, f'round {t.counters.rounds}')
    for mine, snapshot in zip(t.td3.policy_acting.parameters(), self.acting_snapshot):
      assert torch.equal(mine, snapshot), 'the acting copy changed outside refresh_acting'


@pytest.fixture(scope='module')
def recorded(tmp_path_factory) -> Recorded:
  t = make_trainer(tmp_path_factory.mktemp('recorded'), random_episodes_per_world=RANDOM_EPISODES)
  run = Recorded(t)
  for _ in range(RECORDED_ROUNDS):
    run.round()
  t.logger.close()
  return run


# 1. Random phase and sigma latching, against the literal actor.


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_latches_against_the_literal_actor_under_arbitrary_events(seed):
  """Random end patterns and publication times, far more varied than one real run reaches."""
  rng = random.Random(seed)
  W = 32
  td3 = TD3(SMALL, small_train_cfg(), Topology(EnvCfg(num_targets=T), DEVICE), DEVICE)
  latches = EpisodeLatches(td3.cfg, td3, W, DEVICE)
  actors = [LiteralActor() for _ in range(W)]
  published = 0
  latches.begin(torch.arange(W, device=DEVICE), published)
  for actor in actors:
    actor.start_episode(published)
  assert_latches_match(latches, actors, 'start')
  assert latches.use_random.all()
  transitions = 0
  for tick in range(400):
    # Nothing is published for a while, then the step moves in jumps of varied size.
    if tick > 60 and rng.random() < 0.3:
      published += rng.choice([20, 20, 40, 300, 100000])
    ended = [w for w in range(W) if rng.random() < rng.choice([0.05, 0.5])]
    if ended:
      ids = torch.as_tensor(ended, device=DEVICE)
      was_random = latches.use_random[ids].tolist()
      latches.finish(ids)
      latches.begin(ids, published)
      for w in ended:
        actors[w].finish_episode()
        actors[w].start_episode(published)
      transitions += sum(a and not b for a, b in zip(was_random, latches.use_random[ids].tolist()))
    assert_latches_match(latches, actors, f'tick {tick}')
  assert transitions == W, 'every world must leave the random phase once, so all branches ran'
  assert latches.sigma.max() < 0.7


def test_literal_reference_uses_the_previous_fetch():
  """Pins the reference itself to the reading of actor.py:224 and :612."""
  actor = LiteralActor(random_episodes=1)
  actor.start_episode(published_step=0)
  assert actor.in_exploration and actor.recorded_network_step == 0
  actor.finish_episode()
  actor.start_episode(published_step=100000)  # Count reached 1, but the actor still holds step 0.
  assert actor.in_exploration and actor.sigma == 0.7 and actor.recorded_network_step == 100000
  actor.finish_episode()
  actor.start_episode(published_step=200000)  # Network on, sigma from the fetch of the last episode.
  assert not actor.in_exploration and actor.sigma == pytest.approx(0.7 * 0.8) and actor.recorded_network_step == 200000


def test_recorded_run_reached_every_phase(recorded):
  """Otherwise the comparison made at every round would have been empty."""
  t = recorded.t
  assert t.latches.finished.min() >= RANDOM_EPISODES and t.td3.grad_steps > 0
  assert recorded.random_rows > 0 and recorded.policy_rows > 0, 'random and network actions both ran'
  assert recorded.saturated > 0, 'clipping must have fired'
  assert not t.latches.use_random.all() and (t.latches.sigma < 0.7).all()
  assert len(set(t.latches.held_step.tolist())) >= 1 and t.latches.held_step.min() > 0


def test_a_world_stays_random_while_nothing_is_published(tmp_path):
  """Past 32 episodes the held step is still 0, so the second clause alone keeps it random."""
  t = make_trainer(tmp_path, targets=1, num_worlds=2, replay_min_divisor=1, random_episodes_per_world=4)  # min_size = capacity.
  run = Recorded(t)
  while t.latches.finished.min() < 6:
    run.round()
  assert t.td3.grad_steps == 0 and t.published_step == 0
  assert t.latches.use_random.all() and run.policy_rows == 0
  t.logger.close()


# 2. Rate.


def test_samples_per_insert_is_five_within_one_fused_call(recorded):
  t = recorded.t
  megabatch = t.cfg.batch_size * t.cfg.fused_steps
  assert t.replay.sampled == t.td3.grad_steps * t.cfg.batch_size
  assert t.td3.grad_steps % t.cfg.fused_steps == 0, 'full fused calls only'
  allowed = 5.0 * (t.replay.inserted - t.min_size)
  assert t.replay.sampled <= allowed < t.replay.sampled + megabatch
  checked = 0
  previous_sampled = None
  for s in recorded.samples:
    assert s.n == megabatch
    assert s.inserted >= t.min_size, 'no sample before min_size'
    assert s.sampled + megabatch <= 5.0 * (s.inserted - t.min_size)
    # Each call is the first the budget permits: without it, the previous state was still allowed.
    if previous_sampled is not None:
      assert s.sampled == previous_sampled + megabatch
    previous_sampled = s.sampled
    checked += 1
  assert checked > 100


def test_one_megabatch_and_one_feature_build_per_fused_call(tmp_path, monkeypatch):
  import roboballet_mjlab.train.trainer as module

  t = make_trainer(tmp_path, num_worlds=4)
  calls = []
  build = module.build_graph
  monkeypatch.setattr(module, 'build_graph', lambda obs, topo: (calls.append('build'), build(obs, topo))[1])
  sample, fused = t.replay.sample, t.td3.fused_step
  t.replay.sample = lambda n, generator: (calls.append(n), sample(n, generator))[1]
  t.td3.fused_step = lambda batch, **kw: (calls.append('fused'), fused(batch, **kw))[1]
  while t.counters.fused_calls < 3:
    t.round()
  megabatch = t.cfg.batch_size * t.cfg.fused_steps
  start = calls.index(megabatch)
  assert calls[start : start + 4] == [megabatch, 'build', 'build', 'fused'], 's and s\' once, then the whole fused call'
  t.logger.close()


def test_the_learner_gets_the_sample_as_drawn(tmp_path):
  """s from each row, s' from its successor, and the stored action, reward and terminal."""
  t = make_trainer(tmp_path, num_worlds=4)
  seen = []
  sample, fused = t.replay.sample, t.td3.fused_step

  def sample_(n, generator):
    seen.append(sample(n, generator))
    return seen[-1]

  def fused_(batch, **kw):
    drawn = seen[-1]
    assert all(torch.equal(a, b) for a, b in zip(batch.graph, build_graph(drawn.obs, t.topo)))
    assert all(torch.equal(a, b) for a, b in zip(batch.next_graph, build_graph(drawn.next_obs, t.topo)))
    assert not torch.equal(batch.graph.globals, batch.next_graph.globals)
    assert torch.equal(drawn.next_obs.robot.step, drawn.obs.robot.step + 1)
    assert batch.action is drawn.action and batch.reward is drawn.reward and batch.terminal is drawn.terminal
    return fused(batch, **kw)

  t.replay.sample, t.td3.fused_step = sample_, fused_
  while t.counters.fused_calls < 3:
    t.round()
  t.logger.close()


def test_reported_numbers_are_the_original_definitions(tmp_path):
  """train.py:398-404, :764-767 and :873-917. Each from what crossed the seam, per fused call."""
  t = make_trainer(tmp_path, num_worlds=4, log_every_steps=50)
  per_call, stats_calls = [], []
  update, sample = t.td3._update, t.replay.sample
  current: list[tuple[float, float]] = []

  def update_(batch):
    out = update(batch)
    current.append((out.policy.item(), out.critic.item()))
    return out

  def sample_(n, generator):
    out = sample(n, generator)
    if current:
      per_call.append(list(current))
      current.clear()
    stats_calls.append(out.stats.double().mean(0).cpu().numpy())
    return out

  t.td3._update, t.replay.sample = update_, sample_
  while t.td3.grad_steps < 155:
    t.round()
  t.logger.flush()
  per_call.append(list(current))
  rows = [line.split(',') for line in (tmp_path / 'metrics.csv').read_text().splitlines()[1:]]
  by_tag = {}
  for step, tag, value, _ in rows:
    by_tag.setdefault(tag, []).append((int(step), float(value)))
  report_steps = [step for step, _ in by_tag['counters/grad_steps']]
  assert report_steps == [50, 100, 150], 'on the multiples, also through a hindsight flush'
  # The first report goes unwritten, as train.py:876-914, but the counters are this port's own.
  assert [step for step, _ in by_tag['train/policy_loss']] == report_steps[1:]
  losses = np.array([[sum(x) for x in zip(*call)] for call in per_call[:30]])
  assert all(len(call) == t.cfg.fused_steps for call in per_call[:30]), 'sums over the fused steps'
  fused_per_report = t.cfg.log_every_steps // t.cfg.fused_steps
  window = losses[fused_per_report : 2 * fused_per_report]
  assert by_tag['train/policy_loss'][0][1] == pytest.approx(window[:, 0].mean(), rel=1e-5)
  assert by_tag['train/twin_critic_loss'][0][1] == pytest.approx(window[:, 1].mean(), rel=1e-5)
  stats = np.array(stats_calls)[fused_per_report : 2 * fused_per_report].mean(0)
  from roboballet_mjlab.replay.replay import STAT_NAMES

  for i, name in enumerate(STAT_NAMES):
    assert by_tag[f'act/{name}'][0][1] == pytest.approx(stats[i], rel=1e-5, abs=1e-5), name
  assert by_tag['train/policy_lr'][0][1] == t.td3.policy_lr(report_steps[1])
  assert by_tag['train/twin_critic_lr'][0][1] == t.td3.critic_lr(report_steps[1])
  t.logger.close()


def test_budget_never_lags_by_a_full_call_at_any_round(tmp_path):
  """Checks the bound after each round, not only at the end, including the flush lumps."""
  t = make_trainer(tmp_path, num_worlds=4)
  megabatch = t.cfg.batch_size * t.cfg.fused_steps
  largest_jump = 0
  for _ in range(400):
    inserted = t.replay.inserted
    t.round()
    largest_jump = max(largest_jump, t.replay.inserted - inserted)
    if t.replay.inserted >= t.min_size:
      allowed = 5.0 * (t.replay.inserted - t.min_size)
      assert t.replay.sampled <= allowed < t.replay.sampled + megabatch
    else:
      assert t.replay.sampled == 0
  assert largest_jump >= 50, 'a hindsight flush arrives in one lump, and the loop must catch up in one round'
  assert t.replay.sampled > 0
  t.logger.close()


def test_no_sample_before_min_size_and_hindsight_counts_toward_it(recorded):
  t = recorded.t
  first = recorded.samples[0]
  assert first.inserted >= t.min_size + t.cfg.batch_size * t.cfg.fused_steps / 5.0
  assert first.real_rows < t.min_size, 'hindsight rows must count, or sampling would start later'
  # Exactly the first insert after which the rule allows it.
  rows = 0
  for insert in recorded.inserts:
    rows += sum(insert.length)
    if 5.0 * (rows - t.min_size) >= t.cfg.batch_size * t.cfg.fused_steps:
      break
  assert first.round == insert.round
  assert t.replay.inserted == recorded.real_rows + recorded.hindsight_rows


def test_min_size_is_capacity_over_ten():
  cfg = TrainCfg()
  assert cfg.replay_capacity(10) // cfg.replay_min_divisor == 200_000
  assert cfg.batch_size * cfg.fused_steps == 2560
  assert cfg.num_worlds == 64 and cfg.random_episodes_per_world == 32 and cfg.bf16 and cfg.matmul_precision == 'high'
  assert cfg.hindsight_flush_episodes == 8 and cfg.samples_per_insert == 5.0
  # A flush must not push samples-per-insert past the band of launch.py:145-149, as one of 64 episodes did.
  assert cfg.hindsight_flush_episodes * 120 * cfg.samples_per_insert <= 10_240


# 3. Stored action, insert order, behaviour step, refresh.


def test_stored_real_actions_are_the_requests_and_are_not_changed(recorded):
  """Row for row what went into env.step, which already equals the clipped noisy request."""
  actions = torch.stack(recorded.actions)  # [rounds, W, R, 7]
  real = [i for i in recorded.inserts if not any(i.hindsight)]
  episodes = iter(recorded.episodes)
  checked = 0
  for insert in real:
    for n, length in enumerate(insert.length):
      episode = next(episodes)
      rows = actions[episode.end_round - length + 1 : episode.end_round + 1, episode.world]
      assert torch.equal(insert.action[n, :length], rows)
      assert (rows.abs() <= 1).all()
      checked += length
  assert checked == recorded.real_rows
  assert checked > 3000


def test_env_changes_never_reach_a_stored_real_action(tmp_path):
  """Dwell zeroes the request inside the env. The stored action must keep the request.

  A target on the tip at home, and an action small enough to stay within tolerance, so robots dwell.
  """
  pool = small_pool()
  probe = BalletEnv(EnvCfg(num_targets=T), len(pool.q0), pool, DEVICE, seed=1)
  probe.reset(scene_ids=torch.arange(len(pool.q0), device=DEVICE))
  tips, rots = probe.observe().robot.tip_pos, probe.observe().robot.tip_rot
  pool = pool._replace(target_pos=tips[:, :T].clone(), target_rot=rots[:, :T].clone())
  t = Trainer(EnvCfg(num_targets=T), small_train_cfg(num_worlds=2), SMALL, pool, DEVICE, tmp_path)
  request = 1e-3
  t._act = lambda obs: torch.full(t._action_shape, request, device=DEVICE)
  dwelled = 0
  rows = []
  original = t.replay.insert

  def insert(batch):
    rows.append(batch.action[:, : int(batch.length.min())].clone())
    original(batch)

  t.replay.insert = insert
  for _ in range(30):
    dwelled += int((t.env.observe().robot.dwell > 0).sum())
    t.round()
  assert dwelled > 0 and rows
  assert all(torch.equal(r, torch.full_like(r, request)) for r in rows)
  t.logger.close()


def test_behaviour_action_formula():
  mean = torch.tensor([0.9, -0.9, 0.0], device=DEVICE).expand(2, 1, 3)
  normals = torch.tensor([1.0, -1.0, 1.0], device=DEVICE).expand(2, 1, 3)
  uniform = torch.full((2, 1, 3), 0.25, device=DEVICE)
  out = behaviour_action(mean, torch.tensor([0.5, 0.5], device=DEVICE), normals, uniform, torch.tensor([False, True], device=DEVICE))
  assert out[0].tolist() == [[1.0, -1.0, 0.5]]
  assert out[1].tolist() == [[0.25, 0.25, 0.25]]


def test_real_episodes_enter_before_their_hindsight(recorded):
  inserts = recorded.inserts
  assert not any(inserts[0].hindsight), 'the very first insert is real'
  seen_real: list[torch.Tensor] = []
  flushes = 0
  for insert in inserts:
    if any(insert.hindsight):
      flushes += 1
      assert all(insert.hindsight)
      real = torch.cat(seen_real)
      assert insert.round == inserts[inserts.index(insert) - 1].round or insert.inserted_before > 0
      # Every hindsight episode's bases are those of a real episode already in.
      for base in insert.base:
        assert (real == base).all(-1).all(-1).any(), 'hindsight of an episode not yet inserted'
    else:
      seen_real.append(insert.base)
  assert flushes > 10


def test_behaviour_step_is_the_fetched_step(recorded):
  """What actor.py:616 records: the step fetched at this episode's start, not the one before."""
  real = [i for i in recorded.inserts if not any(i.hindsight)]
  episodes = iter(recorded.episodes)
  nonzero = 0
  for insert in real:
    for behaviour, stat in zip(insert.behaviour, insert.network_stat):
      episode = next(episodes)
      assert behaviour == episode.fetched
      assert stat == behaviour
      nonzero += behaviour > 0
  assert nonzero > 10
  for insert in recorded.inserts:
    assert insert.behaviour == [int(x) for x in insert.network_stat]


def test_refresh_follows_finished_episodes_and_staleness(recorded):
  t = recorded.t
  rounds = [r for r, _ in recorded.refreshes]
  ended_at = {}
  for episode in recorded.episodes:
    ended_at[episode.end_round + 1] = ended_at.get(episode.end_round + 1, 0) + 1
  expected, count = [], 0
  for r in range(1, t.counters.rounds + 1):
    count += ended_at.get(r, 0)
    if count >= t.num_worlds:
      expected.append(r)
      count = 0
  assert rounds == expected
  assert t.published_step == recorded.refreshes[-1][1]
  assert recorded.refreshes[-1][1] > recorded.refreshes[0][1]
  assert t.published_step < t.td3.grad_steps, 'this run must end with the acting copy behind'
  # Parameters are stale between refreshes, and the forward really uses the stale ones.
  online = list(t.td3.policy.parameters())
  acting = list(t.td3.policy_acting.parameters())
  assert any(not torch.equal(a, b) for a, b in zip(online, acting))
  graph = build_graph(t.env.observe(), t.topo)
  stale = t.td3.act_behaviour(graph)
  assert (stale - t.td3.act(graph)).abs().max() > 0
  plain = PolicyNet(SMALL, t.topo).to(DEVICE)
  plain.load_state_dict(t.td3.policy_acting.state_dict())
  with torch.no_grad():
    assert torch.equal(stale, plain(graph))


def test_trajectory_network_age_is_from_sampled_transitions(tmp_path):
  """train.py:862-866: the learner's step minus the mean of the sampled network steps."""
  t = make_trainer(tmp_path, num_worlds=4, log_every_steps=10**9)
  while t.td3.grad_steps < 100:
    t.round()
  sample_age = []
  original = t.replay.sample

  def sample(n, generator):
    out = original(n, generator)
    sample_age.append(t.td3.grad_steps - out.behaviour_step.double().mean().item())
    assert (out.stats[:, STAT['network_step']] == out.behaviour_step.float()).all()
    return out

  t.replay.sample = sample
  t._zero_accumulators()
  while len(sample_age) < 20:
    t.round()
  assert t._age_sum.item() / t._fused_in_report == pytest.approx(np.mean(sample_age), rel=1e-9)
  t.logger.close()


# 4. Stagger.


def test_each_world_resets_once_at_its_offset(recorded):
  t = recorded.t
  W, max_steps = t.num_worlds, t.env_cfg.max_steps
  stagger = [(r, ids) for r, ids in recorded.resets if r < max_steps]
  worlds = [w for _, ids in stagger for w in ids]
  assert sorted(worlds) == list(range(1, W)), 'once each, and world 0 runs from round 0'
  offsets = {w: r for r, ids in stagger for w in ids}
  assert len(set(offsets.values())) == W - 1, 'different offsets'
  assert max(offsets.values()) < max_steps and min(offsets.values()) > 0
  assert t.counters.discarded_rows == sum(offsets.values())


def test_no_partial_episode_enters_the_replay_and_ends_differ(recorded):
  t = recorded.t
  W, max_steps = t.num_worlds, t.env_cfg.max_steps
  offsets = {w: r for r, ids in recorded.resets if r < max_steps for w in ids}
  first_end = {}
  for episode in recorded.episodes:
    first_end.setdefault(episode.world, episode.end_round)
  for w in range(W):
    assert first_end[w] == offsets.get(w, 0) + max_steps - 1, f'world {w} ended off its offset'
  assert len(set(first_end.values())) == W, 'episodes must end at different rounds'
  for insert in recorded.inserts:
    assert all(s == 0 for s in insert.first_step), 'an episode must begin at step 0'
    assert all(a <= max_steps for a in insert.length)
  # Whole-ring view: no splice from a discarded prefix can survive in the rows.
  sample = t.replay.fetch(torch.arange(t.replay.size, device=DEVICE))
  assert torch.equal(sample.next_obs.robot.step, sample.obs.robot.step + 1)
  assert (sample.terminal == (sample.next_obs.robot.step == max_steps)).all()


def test_stagger_adds_nothing_to_the_episode_count(recorded):
  t = recorded.t
  assert t.counters.real_episodes == len(recorded.episodes) == int(t.latches.finished.sum())
  assert t.counters.real_episodes == sum(a.episode_count for a in recorded.actors)
  total = t.num_worlds * t.counters.rounds
  in_progress = int(t.buffer.count.sum())
  assert total == recorded.real_rows + t.counters.discarded_rows + in_progress, 'every env step is accounted for'


def test_world_ending_before_its_offset_counts_and_is_not_reset(tmp_path):
  """A full episode from step 0 is a real one, so its later offset must not fire.

  One target on a tip at home and an action too small to leave it: solved at once, then 5
  steps of dwell, so every world ends after 6 actions, before any offset (7, 15, 22 of 30).
  """
  pool = small_pool(targets=1)
  probe = BalletEnv(EnvCfg(num_targets=1), len(pool.q0), pool, DEVICE, seed=1)
  probe.reset(scene_ids=torch.arange(len(pool.q0), device=DEVICE))
  pool = pool._replace(target_pos=probe.observe().robot.tip_pos[:, :1].clone(), target_rot=probe.observe().robot.tip_rot[:, :1].clone())
  t = Trainer(EnvCfg(num_targets=1), small_train_cfg(num_worlds=4), SMALL, pool, DEVICE, tmp_path)
  assert t.env_cfg.max_steps == 30 and t._stagger_at.tolist() == [-1, 7, 15, 22]
  t._act = lambda obs: torch.full(t._action_shape, 1e-3, device=DEVICE)
  stagger_resets = []
  in_step = [False]
  env_reset, env_step = t.env.reset, t.env.step

  def reset(world_ids=None, scene_ids=None):
    if not in_step[0]:
      stagger_resets.append(world_ids)
    return env_reset(world_ids, scene_ids)

  def step(action):
    in_step[0] = True
    out = env_step(action)
    in_step[0] = False
    return out

  t.env.reset, t.env.step = reset, step
  for _ in range(6):
    t.round()
  assert t.latches.finished.tolist() == [1, 1, 1, 1] and t.counters.real_episodes == 4
  assert t.replay.inserted == 4 * 6
  for _ in range(t.env_cfg.max_steps):
    t.round()
  assert not stagger_resets and t.counters.discarded_rows == 0
  assert t.latches.finished.tolist() == [6, 6, 6, 6], 'one episode per 6 rounds, never cut'
  t.logger.close()


# 5. Precision scoping.


def test_td3_init_leaves_the_global_alone():
  torch.set_float32_matmul_precision('highest')
  TD3(SMALL, TrainCfg(compile=False, matmul_precision='high'), Topology(EnvCfg(), DEVICE), DEVICE)
  assert torch.get_float32_matmul_precision() == 'highest'


def _small_batch(td3: TD3, env_cfg: EnvCfg, n: int) -> Batch:
  env = BalletEnv(env_cfg, n, small_pool(env_cfg.num_targets), DEVICE, seed=3)
  graph = build_graph(env.observe(), Topology(env_cfg, DEVICE))
  action = torch.zeros(n, 4, 7, device=DEVICE)
  zero = torch.zeros(n, device=DEVICE)
  return Batch(graph, graph, action, zero, zero)


def test_learner_precision_is_scoped_also_through_an_exception():
  env_cfg = EnvCfg(num_targets=T)
  cfg = TrainCfg(batch_size=4, fused_steps=2, compile=False, matmul_precision='high')
  td3 = TD3(SMALL, cfg, Topology(env_cfg, DEVICE), DEVICE)
  batch = _small_batch(td3, env_cfg, 8)
  seen = []
  update = td3._update

  def spy(b):
    seen.append(torch.get_float32_matmul_precision())
    return update(b)

  td3._update = spy
  torch.set_float32_matmul_precision('highest')
  td3.fused_step(batch)
  assert seen == ['high', 'high'] and torch.get_float32_matmul_precision() == 'highest'
  td3.train_step(batch.slice(0, 4))
  assert seen[-1] == 'high' and torch.get_float32_matmul_precision() == 'highest'
  torch.set_float32_matmul_precision('medium')  # It restores what was there, whatever that was.
  td3.fused_step(batch)
  assert torch.get_float32_matmul_precision() == 'medium'
  torch.set_float32_matmul_precision('highest')

  def boom(b):
    raise RuntimeError('in the learner')

  td3._update = boom
  with pytest.raises(RuntimeError, match='in the learner'):
    td3.fused_step(batch)
  assert torch.get_float32_matmul_precision() == 'highest'
  with pytest.raises(RuntimeError, match='in the learner'):
    td3.train_step(batch.slice(0, 4))
  assert torch.get_float32_matmul_precision() == 'highest'
  with pytest.raises(RuntimeError, match='in here'):
    with matmul_precision('high'):
      raise RuntimeError('in here')
  assert torch.get_float32_matmul_precision() == 'highest'


def _probe_bytes(env: BalletEnv, topo: Topology, generator_state: torch.Tensor) -> list[bytes]:
  """The env's and the features' bytes for the same input. The matmul canary shows a leak directly."""
  canary = torch.linspace(-1.0, 1.0, 512 * 512, device=DEVICE).view(512, 512).sin()
  env.generator.set_state(generator_state)
  env.reset(scene_ids=torch.arange(env.num_worlds, device=DEVICE))
  out = []
  action = torch.linspace(-1, 1, env.num_worlds * 4 * 7, device=DEVICE).view(env.num_worlds, 4, 7)
  for _ in range(4):
    obs = env.observe()
    graph = build_graph(obs, topo)
    step = env.step(action)
    for x in (*graph, *step.next_robot, step.reward):
      out.append(x.cpu().numpy().tobytes())
  out.append(torch.matmul(canary, canary).cpu().numpy().tobytes())
  return out


def test_env_and_features_give_the_same_bytes_after_a_training_round(tmp_path):
  t = make_trainer(tmp_path, matmul_precision='high')
  topo = t.topo
  probe = BalletEnv(t.env_cfg, 8, small_pool(), DEVICE, seed=9)
  state = probe.generator.get_state()
  before = _probe_bytes(probe, topo, state)
  # A leak of `high` must be visible to this probe, or the comparison below proves nothing.
  torch.set_float32_matmul_precision('high')
  leaked = _probe_bytes(probe, topo, state)
  torch.set_float32_matmul_precision('highest')
  assert leaked != before, 'the probe cannot see a precision leak'
  while t.td3.grad_steps == 0:
    assert torch.get_float32_matmul_precision() == 'highest'
    t.round()
  assert torch.get_float32_matmul_precision() == 'highest'
  assert _probe_bytes(probe, topo, state) == before
  t.logger.close()


@pytest.fixture(scope='module')
def e7():
  e6 = load_e6('4r10t', shards=2)
  with np.load(AGENT_GOLDEN / 'agent_e7.npz') as z:
    return e6, z['action']


def _act_rows(td3: TD3, graph: Graph, online: bool = False) -> torch.Tensor:
  act = td3.act if online else td3.act_behaviour
  return torch.cat([act(type(graph)(*[x[i:i + 100] for x in graph])) for i in range(0, 1000, 100)])


def _assert_e7(acted: torch.Tensor, expected: np.ndarray) -> None:
  error = np.abs(acted.cpu().numpy() - expected)
  assert error.max() <= E7_TOLERANCE
  assert error.mean() <= E7_OBSERVED_MEAN and np.quantile(error, 0.999) <= E7_OBSERVED_P999


@pytest.mark.parametrize('compile', [False, True])
def test_acting_matches_jax_after_a_bf16_learner_call(e7, compile):
  """E7 with the learner's autocast and TF32 having run in between, in the parity settings."""
  e6, expected = e7
  env_cfg = EnvCfg()
  cfg = TrainCfg(batch_size=8, fused_steps=2, compile=compile)
  assert cfg.bf16 and cfg.matmul_precision == 'high'
  td3 = TD3(ModelCfg(), cfg, Topology(env_cfg, DEVICE), DEVICE)
  load_checkpoint(str(CHECKPOINT), td3.policy, td3.q1, td3.q2)
  td3.sync_targets()
  td3.refresh_acting()
  torch.set_float32_matmul_precision('highest')
  graph = e6.graph
  _assert_e7(_act_rows(td3, graph), expected)
  rows = torch.arange(16, device=DEVICE)
  batch = Batch(Graph(*[x[rows] for x in graph]), Graph(*[x[rows + 1] for x in graph]),
                torch.as_tensor(e6.raw['action'][:16], device=DEVICE), torch.as_tensor(e6.raw['reward'][:16], device=DEVICE),
                torch.as_tensor(e6.raw['terminal'][:16], device=DEVICE))
  before = [p.clone() for p in td3.policy_acting.parameters()]
  td3.fused_step(batch)
  assert torch.get_float32_matmul_precision() == 'highest'
  # The acting copy still holds the checkpoint and reproduces JAX exactly as before.
  assert all(torch.equal(a, b) for a, b in zip(before, td3.policy_acting.parameters()))
  _assert_e7(_act_rows(td3, graph), expected)
  # In-place refresh reaches the compiled copy, and the online net has moved.
  assert (_act_rows(td3, graph, online=True).cpu().numpy() != expected).any()
  load_checkpoint(str(CHECKPOINT), td3.policy, td3.q1, td3.q2)
  td3.fused_step(batch)
  moved = _act_rows(td3, graph, online=True)
  # The online and acting forwards are two compiled graphs, so rounding may differ.
  stale = _act_rows(td3, graph)
  assert (stale - moved).abs().max() > 1e-4, 'the learner must have moved the online net past the rounding'
  td3.refresh_acting()
  assert (_act_rows(td3, graph) - moved).abs().max() <= 1e-4
  load_checkpoint(str(CHECKPOINT), td3.policy, td3.q1, td3.q2)
  td3.refresh_acting()
  _assert_e7(_act_rows(td3, graph), expected)


# 7. Checkpoints. Isolation of the evaluator, item 6, is in test_evaluator.py.


def _assert_tensors_equal(a, b, path: str = '') -> None:
  if isinstance(a, dict):
    assert a.keys() == b.keys(), path
    for k in a:
      _assert_tensors_equal(a[k], b[k], f'{path}/{k}')
  elif isinstance(a, (list, tuple)):
    assert len(a) == len(b), path
    for i, (x, y) in enumerate(zip(a, b)):
      _assert_tensors_equal(x, y, f'{path}/{i}')
  elif isinstance(a, torch.Tensor):
    assert torch.equal(a.cpu(), b.cpu()), path
  else:
    assert a == b, path


def test_save_and_resume_restores_everything(tmp_path):
  a = make_trainer(tmp_path / 'a', num_worlds=4)
  while a.td3.grad_steps < 100:
    a.round()
  a.publish()
  a.round()
  random.random(), np.random.random(), torch.rand(1), torch.rand(1, device=DEVICE)
  path = a.save_checkpoint('test')
  saved = torch.load(path, map_location='cpu', weights_only=True)
  assert saved['reason'] == 'test' and not saved['failed'] and saved['step'] == a.td3.grad_steps

  b = make_trainer(tmp_path / 'b', num_worlds=4)
  # Moved state in b, so that a mere default could not pass.
  for _ in range(3):
    b.round()

  def global_draws():
    return random.random(), np.random.random(), torch.rand(4).tolist(), torch.rand(4, device=DEVICE).tolist()

  random.seed(123), np.random.seed(5), torch.manual_seed(9)
  b.load_checkpoint(path)
  after_first = global_draws()
  random.seed(321), np.random.seed(6), torch.manual_seed(8)
  # Twice, because the global streams belong to the process: the second must rewind them all.
  b.load_checkpoint(path)
  assert global_draws() == after_first

  _assert_tensors_equal(a.td3.state_dict(), b.td3.state_dict())
  for mine, theirs in zip(a.td3.policy_acting.parameters(), b.td3.policy_acting.parameters()):
    assert torch.equal(mine, theirs)
  for frozen, live in zip(b.td3.q1_frozen.parameters(), b.td3.q1.parameters()):
    assert frozen.data_ptr() == live.data_ptr()
  assert b.td3.grad_steps == a.td3.grad_steps > 0
  assert b.td3.policy_lr(b.td3.grad_steps) == a.td3.policy_lr(a.td3.grad_steps)
  assert b.published_step == a.published_step > 0
  assert b.counters.rounds == a.counters.rounds
  assert b.totals()['inserted'] == a.totals()['inserted'] and b.totals()['sampled'] == a.totals()['sampled']
  assert torch.equal(b.latches.finished, a.latches.finished)
  # Draws after the restore are the same draws.
  for gen_a, gen_b in ((a.act_rng, b.act_rng), (a.sample_rng, b.sample_rng), (a.hindsight_rng, b.hindsight_rng),
                       (a.env.generator, b.env.generator), (a.td3.rng, b.td3.rng)):
    assert torch.equal(torch.rand(8, generator=gen_a, device=DEVICE), torch.rand(8, generator=gen_b, device=DEVICE))
  # Same learner update afterwards, the optimiser moments included.
  batch = _small_batch(a.td3, a.env_cfg, 32)
  a.td3.fused_step(batch, num_steps=1)
  b.td3.fused_step(batch, num_steps=1)
  for x, y in zip(a.td3.policy.parameters(), b.td3.policy.parameters()):
    assert torch.equal(x, y)
  for x, y in zip(a.td3.q1_target.parameters(), b.td3.q1_target.parameters()):
    assert torch.equal(x, y)
  for logger in (a.logger, b.logger):
    logger.close()


def test_resumed_run_refills_the_replay_before_sampling(tmp_path):
  a = make_trainer(tmp_path / 'a', num_worlds=8)
  while a.td3.grad_steps < 50:
    a.round()
  path = a.save_checkpoint('test')
  steps = a.td3.grad_steps
  b = make_trainer(tmp_path / 'b', num_worlds=8)
  b.load_checkpoint(path)
  assert b.replay.inserted == 0 and b.replay.size == 0
  sampled_at = []
  original = b.replay.sample

  def sample(n, generator):
    sampled_at.append(b.replay.inserted)
    return original(n, generator)

  b.replay.sample = sample
  while not sampled_at:
    b.round()
  assert sampled_at[0] >= b.min_size
  assert b.td3.grad_steps > steps
  assert b.replay.sampled == (b.td3.grad_steps - steps) * b.cfg.batch_size
  assert b.totals()['sampled'] == a.replay.sampled + b.replay.sampled
  assert not (b.latches.use_random & (b.latches.finished >= 32) & (b.latches.held_step > 0)).any()
  # Rows must start at step 0 after the resume too, stagger included.
  rows = b.replay.fetch(torch.arange(b.replay.size, device=DEVICE))
  assert torch.equal(rows.next_obs.robot.step, rows.obs.robot.step + 1)
  a.logger.close()
  b.logger.close()


def test_resume_refuses_a_failed_or_foreign_checkpoint(tmp_path):
  a = make_trainer(tmp_path / 'a', num_worlds=4)
  failed = a.save_checkpoint('bad', failed=True)
  assert failed.name == FAILED_CHECKPOINT_NAME
  fine = a.save_checkpoint('good')
  b = make_trainer(tmp_path / 'b', num_worlds=4)
  with pytest.raises(ValueError, match='failed'):
    b.load_checkpoint(failed)
  c = make_trainer(tmp_path / 'c', num_worlds=4, batch_size=32)
  with pytest.raises(ValueError, match='batch_size'):
    c.load_checkpoint(fine)
  d = make_trainer(tmp_path / 'd', num_worlds=4, seed=2)
  with pytest.raises(ValueError, match='seed'):
    d.load_checkpoint(fine)
  e = make_trainer(tmp_path / 'e', num_worlds=4, total_grad_steps=5)  # This one may change.
  e.load_checkpoint(fine)
  for t in (a, b, c, d, e):
    t.logger.close()


def test_trainer_checkpoint_fits_the_network_loader(tmp_path):
  t = make_trainer(tmp_path, num_worlds=4)
  path = t.save_checkpoint('test')
  topo = Topology(t.env_cfg, 'cpu')
  policy, q1, q2 = PolicyNet(SMALL, topo), CriticNet(SMALL, topo), CriticNet(SMALL, topo)
  assert load_checkpoint(str(path), policy, q1, q2) == 0
  for x, y in zip(policy.parameters(), t.td3.policy.parameters()):
    assert torch.equal(x, y.cpu())
  t.logger.close()


def test_interrupted_write_by_exception_leaves_the_old_file(tmp_path, monkeypatch):
  path = tmp_path / 'c.pt'
  atomic_save({'value': torch.tensor(1)}, path)
  good = path.read_bytes()

  def partial(payload, f):
    f.write(b'partial')
    raise OSError('disk full')

  monkeypatch.setattr(torch, 'save', partial)
  with pytest.raises(OSError, match='disk full'):
    atomic_save({'value': torch.tensor(2)}, path)
  assert path.read_bytes() == good
  assert list(tmp_path.iterdir()) == [path], 'no temporary file left behind'


def _child_env() -> dict[str, str]:
  """The package under test: the mutation check's overlay when there is one, else this repo."""
  return {**os.environ, 'PYTHONPATH': str(Path(os.environ.get('PYTHONPATH') or REPO / 'src').resolve())}


def test_killed_mid_write_leaves_the_old_file(tmp_path):
  """A real SIGKILL halfway through the bytes, so no cleanup code can run."""
  path = tmp_path / 'c.pt'
  atomic_save({'value': torch.tensor(1)}, path)
  good = path.read_bytes()
  script = f"""
import os, signal, torch
from pathlib import Path
from roboballet_mjlab.train.trainer import atomic_save
real_save = torch.save
def killed(payload, f):
  f.write(b'x' * 1000)
  f.flush()
  os.kill(os.getpid(), signal.SIGKILL)
torch.save = killed
atomic_save({{'value': torch.tensor(2)}}, Path({str(path)!r}))
"""
  done = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=_child_env(), capture_output=True)
  assert done.returncode == -signal.SIGKILL, done.stderr.decode()
  assert path.read_bytes() == good
  assert torch.load(path, weights_only=True)['value'] == 1


def test_nonfinite_loss_saves_a_failed_checkpoint_and_raises(tmp_path):
  t = make_trainer(tmp_path, num_worlds=4)
  while t.td3.grad_steps == 0:
    t.round()
  good = t.save_checkpoint('good')
  good_bytes = good.read_bytes()
  fused = t.td3.fused_step

  def poisoned(batch, **kwargs):
    out = fused(batch, **kwargs)
    return out._replace(critic=out.critic * float('nan'))

  t.td3.fused_step = poisoned
  with pytest.raises(FloatingPointError, match='nonfinite'):
    for _ in range(100):
      t.round()
  failed = torch.load(tmp_path / FAILED_CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert failed['failed'] and 'nonfinite' in failed['reason'] and failed['step'] == t.td3.grad_steps
  assert good.read_bytes() == good_bytes, 'the last good checkpoint must survive'
  t.logger.close()


def test_periodic_checkpoint_fires_by_wall_clock(tmp_path):
  t = make_trainer(tmp_path, num_worlds=4, checkpoint_every_s=0.0)
  t.round()
  assert (tmp_path / CHECKPOINT_NAME).exists()
  saved = torch.load(tmp_path / CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert saved['reason'] == 'periodic'
  t.logger.close()


def test_overflow_raises_through_the_trainer(tmp_path):
  from roboballet_mjlab.sim.world import ContactOverflow

  t = make_trainer(tmp_path, num_worlds=4)
  t.env.world._naconmax = 0  # Any contact candidate is now over the pool, which is what the guard reads.
  with pytest.raises(ContactOverflow):
    for _ in range(50):
      t.round()
  t.logger.close()


def test_collision_counters_reach_the_report(tmp_path):
  t = make_trainer(tmp_path, num_worlds=4, log_every_steps=1)
  t.env.residual_collisions += 3
  t.env.invalid_scene_draws += 5
  assert t.totals()['residual_collisions'] == 3 and t.totals()['invalid_scene_draws'] == 5
  while t.td3.grad_steps == 0:
    t.round()
  t.logger.flush()
  rows = (tmp_path / 'metrics.csv').read_text()
  assert 'counters/residual_collisions,3.0' in rows and 'counters/invalid_scene_draws,5.0' in rows
  saved = t.save_checkpoint('test')
  b = make_trainer(tmp_path / 'b', num_worlds=4, log_every_steps=1)
  b.load_checkpoint(saved)
  assert b.totals()['residual_collisions'] == 3 and b.totals()['invalid_scene_draws'] == 5
  t.logger.close()
  b.logger.close()


# Separate streams, because one seed everywhere would make action noise and target noise equal.


def test_random_streams_are_distinct(tmp_path):
  t = make_trainer(tmp_path, num_worlds=4)
  draws = [torch.rand(16, generator=g, device=DEVICE).cpu() for g in (t.act_rng, t.sample_rng, t.hindsight_rng, t.env.generator, t.td3.rng)]
  for i in range(len(draws)):
    for j in range(i):
      assert not torch.equal(draws[i], draws[j])
  t.logger.close()


# 8. End to end, the full model on a small real config, compiled, bf16.


def _read_tags(path: Path) -> dict[str, list[float]]:
  rows: dict[str, list[float]] = {}
  for line in path.read_text().splitlines()[1:]:
    _, tag, value, _ = line.split(',')
    rows.setdefault(tag, []).append(float(value))
  return rows


def test_end_to_end_small_real_config(tmp_path):
  env_cfg = EnvCfg()
  scenes = stored_scenes(4)
  cfg = TrainCfg(num_worlds=32, replay_max_size=30_000, hindsight_flush_episodes=32, log_every_steps=100, checkpoint_every_s=1e9,
                 eval_every_steps=400, eval_episodes=16, total_grad_steps=800, random_episodes_per_world=3)
  evaluator = Evaluator(env_cfg, ModelCfg(), {'training_dist': scenes, 'static_paper': scenes._replace(
    **{k: getattr(scenes, k)[:4] for k in scenes._fields})}, DEVICE, seed=3, num_worlds=16, static_worlds=8, compile=True)
  t = Trainer(env_cfg, cfg, ModelCfg(), scenes, DEVICE, tmp_path, evaluator)
  assert t.cfg.bf16 and t.cfg.compile and t.cfg.fused_steps == 20 and t.cfg.batch_size == 128 and t.min_size == 3000
  allocated = []
  while t.td3.grad_steps < 800:
    t.round()
    if t.counters.rounds % 25 == 0 and t.td3.grad_steps >= 100:  # After Adam's moments exist.
      torch.cuda.synchronize()
      allocated.append(torch.cuda.memory_allocated())
  t.save_checkpoint('end')
  t.logger.flush()
  totals = t.totals()

  # Counters agree.
  assert totals['grad_steps'] == 800 and totals['sampled'] == 800 * 128 and totals['fused_calls'] == 40
  assert totals['env_steps'] == totals['rounds'] * 32
  assert totals['real_episodes'] == int(t.latches.finished.sum())
  assert totals['hindsight_episodes'] <= totals['real_episodes']
  rows = totals['inserted'] - totals['hindsight_episodes'] * env_cfg.max_steps
  assert rows + totals['discarded_rows'] + int(t.buffer.count.sum()) == totals['env_steps']
  assert totals['residual_collisions'] == 0
  assert totals['replay_size'] == totals['inserted'] - totals['evicted']
  assert totals['extra_evicted'] == 0

  tags = _read_tags(tmp_path / 'metrics.csv')
  # Finite, and in the reference's range at the start: loss sums over the 20 steps.
  assert all(np.isfinite(v).all() for v in tags.values())
  assert tags['train/twin_critic_loss'] and tags['train/policy_loss']
  assert 0 < np.mean(tags['train/twin_critic_loss']) < 1.0
  assert abs(np.mean(tags['train/policy_loss'])) < 20.0
  # Hindsight fraction near 0.5 in the random phase: the reference's 0.49.
  # This small ring begins before the first flush, so its first reports read lower than parity's.
  fraction = np.array(tags['act/is_hindsight_replay'])
  assert fraction[-3:].mean() == pytest.approx(0.5, abs=0.05) and fraction.max() < 0.55
  assert totals['hindsight_episodes'] * env_cfg.max_steps / totals['inserted'] == pytest.approx(0.5, abs=0.05)
  # Rate as a measured number.
  assert tags['rate/samples_per_insert'][-1] == pytest.approx(5.0, rel=0.05)
  # The reference's tags, all of them present.
  reference = json.loads(TB_JSON.read_text())
  wanted = {k for k in reference if k.startswith(('train/', 'act/'))}
  wanted |= {k.replace('static_dist', 'static_paper') for k in reference if k.startswith('eval/')}
  assert wanted <= set(tags), sorted(wanted - set(tags))
  assert tags['train/policy_lr'][0] == pytest.approx(5e-5 * 0.98 ** (t.td3.grad_steps * 0 + 200 / 1e5), rel=1e-3)
  # Memory stable once the ring exists: no growth from accumulators or the queue.
  assert len(allocated) >= 4
  assert max(allocated) - min(allocated) < 32 * 2**20, allocated
  # TensorBoard has them too.
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  events = EventAccumulator(str(tmp_path / 'tb'))
  events.Reload()
  assert wanted <= set(events.Tags()['scalars'])
  t.logger.close()


# Profiler and logger.


def test_profiler_synchronises_only_inside_windows(tmp_path, monkeypatch):
  t = make_trainer(tmp_path, num_worlds=4, profile=True, log_every_steps=10**9)
  t.profiler.first_window, t.profiler.window_rounds, t.profiler.every_rounds = 3, 4, 1000
  calls = []
  real = torch.cuda.synchronize
  monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: (calls.append(t.counters.rounds), real(*a))[1])
  for _ in range(3):
    t.round()
  assert not calls, 'no GPU wait outside a window'
  while t.counters.rounds < 8:  # Rounds 3 to 6, closed at the top of round 7.
    t.round()
  assert calls and all(3 <= r <= 7 for r in calls)
  assert t.profiler.in_window is False
  count = len(calls)
  t.round()
  t.round()
  assert len(calls) == count, 'none after the window closes'
  report = t.profiler.report(t.counters.rounds)
  for key in ('profile/env_step_ms', 'profile/world_forward_ms', 'profile/rollback_ms', 'profile/episode_buffer_ms',
              'profile/window_launches_per_round', 'profile/window_main_thread_cpu_fraction',
              'profile/main_thread_cpu_fraction', 'profile/launches_per_round', 'profile/mem_allocated_gib', 'profile/rss_gib'):
    assert key in report, key
  assert report['profile/world_forward_ms'] > 0 and report['profile/window_launches_per_round'] >= 1
  assert 0 < report['profile/window_main_thread_cpu_fraction'] <= 1.5
  assert report.get('profile/gpu_util_percent', 0.0) >= 0.0
  assert 'profile/gpu_mem_used_gib' in report, 'NVML did not answer'
  assert report['profile/gpu_mem_used_gib'] > 0
  t.logger.close()


def test_item_counter_is_restored_after_a_window(tmp_path):
  original = torch.Tensor.item
  t = make_trainer(tmp_path, num_worlds=2, profile=True)
  t.profiler.first_window, t.profiler.window_rounds = 0, 2
  t.round()
  assert torch.Tensor.item is not original
  torch.zeros(1).item()
  t.round()
  t.round()
  assert t.profiler.in_window is False
  assert t.profiler.report(3)['profile/window_item_calls_per_round'] >= 0.5
  assert torch.Tensor.item is original, 'the patch must not outlive the window'
  assert torch.tensor(2.0).item() == 2.0
  t.logger.close()


def test_logger_writes_tensorboard_and_csv_and_never_needs_wandb(tmp_path, monkeypatch):
  class Fake:
    calls = []

    @staticmethod
    def init(**kwargs):
      Fake.calls.append(kwargs)
      return Fake

    @staticmethod
    def log(values):
      Fake.calls.append(values)

    @staticmethod
    def finish():
      Fake.calls.append('finish')

  monkeypatch.setitem(sys.modules, 'wandb', Fake)
  logger = MetricLogger(tmp_path / 'plain')
  logger.scalars({'a/b': 1.5}, 7)
  logger.close()
  assert Fake.calls == []
  assert '7,a/b,1.5,' in (tmp_path / 'plain/metrics.csv').read_text()
  logger = MetricLogger(tmp_path / 'wandb', wandb_project='p', config={'seed': 1})
  logger.scalars({'a/b': 2.5}, 9)
  logger.close()
  assert Fake.calls[0]['project'] == 'p' and Fake.calls[1]['a/b'] == 2.5 and Fake.calls[-1] == 'finish'
  source = (REPO / 'src/roboballet_mjlab/train/logger.py').read_text() + (REPO / 'src/roboballet_mjlab/scripts/train.py').read_text()
  for word in ('os.environ', 'getenv', 'api_key', 'wandb.login', 'netrc'):
    assert word not in source, word


def test_provenance_is_complete_read_only_and_free_of_secrets(tmp_path, monkeypatch):
  monkeypatch.setenv('WANDB_API_KEY', 'sekret-value-1234')
  index = REPO / '.git/index'
  before = index.stat().st_mtime_ns
  out = write_provenance(tmp_path, {'train': TrainCfg(), 'env': EnvCfg(), 'source': 'x'}, DEVICE)
  assert index.stat().st_mtime_ns == before, 'provenance wrote to the index'
  names = {p.name for p in out.iterdir()}
  assert names == {'config.json', 'git_revision.txt', 'git_diff.patch', 'git_untracked.txt', 'system.json'}
  from roboballet_mjlab.scenes.io import git_revision

  assert (out / 'git_revision.txt').read_text().strip() == git_revision(REPO)
  for path in out.iterdir():
    assert 'sekret-value' not in path.read_text()
  config = json.loads((out / 'config.json').read_text())
  assert config['train']['batch_size'] == 128 and config['env']['world']['arm_model'] == 'spheres_stick'
  system = json.loads((out / 'system.json').read_text())
  assert system['hostname'] and 'torch' in system['packages'] and 'B200' in system['gpu_model']
  # The patch is limited to src/roboballet_mjlab/. git diff names paths from the top, ls-files from src/.
  diff = (out / 'git_diff.patch').read_text()
  assert 'git failed' not in diff
  for line in diff.splitlines():
    if line.startswith('diff --git'):
      assert line.startswith('diff --git a/src/roboballet_mjlab/'), line
  for line in (out / 'git_untracked.txt').read_text().splitlines():
    assert line.startswith('roboballet_mjlab/'), line


# The script, as a detached run uses it: SIGTERM, a clean exit, and a resume.


def _train_command(run_dir: Path, *extra: str) -> list[str]:
  return [sys.executable, '-m', 'roboballet_mjlab.scripts.train', '--run-dir', str(run_dir),
          '--train-pool', str(scene_pool('4r_10t_30o_heldout')), '--no-eval',
          '--set', 'num_worlds=4', '--set', 'batch_size=4', '--set', 'fused_steps=2', '--set', 'replay_max_size=4000',
          '--set', 'hindsight_flush_episodes=4', '--set', 'compile=false', '--set', 'log_every_steps=10', *extra]


def test_script_sigterm_saves_exits_clean_and_resumes(tmp_path):
  run_dir = tmp_path / 'run'
  env = _child_env()
  process = subprocess.Popen(_train_command(run_dir), cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  try:
    deadline = time.time() + 240
    metrics = run_dir / 'metrics.csv'
    while time.time() < deadline and process.poll() is None:
      if metrics.exists() and 'counters/grad_steps' in metrics.read_text():
        break
      time.sleep(1)
    assert process.poll() is None, process.communicate()[0][-3000:]
    process.send_signal(signal.SIGTERM)
    output, _ = process.communicate(timeout=120)
  finally:
    process.kill()
  assert process.returncode == 0, output[-3000:]
  assert 'signal 15' in output
  saved = torch.load(run_dir / CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert saved['reason'] == 'signal 15' and saved['step'] > 0
  summary = json.loads((run_dir / 'summary.json').read_text())
  assert summary['grad_steps'] == saved['step']
  assert len(list((run_dir / 'provenance').iterdir())) == 1

  done = subprocess.run(_train_command(run_dir, '--resume', '--stop-after-seconds', '5'), cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)
  assert done.returncode == 0, done.stdout[-3000:]
  assert 'resumed from' in done.stdout and 'stop_after_seconds' in done.stdout
  assert json.loads((run_dir / 'summary.json').read_text())['rounds'] > summary['rounds']
  assert len(list((run_dir / 'provenance').iterdir())) == 2
  assert not any(path.name.startswith('checkpoint.pt.tmp') for path in run_dir.iterdir())


def test_no_runtime_import_of_the_original_code():
  sources = [*REPO.glob('src/roboballet_mjlab/train/*.py'), REPO / 'src/roboballet_mjlab/scripts/train.py', REPO / 'src/roboballet_mjlab/scripts/eval.py', REPO / 'src/roboballet_mjlab/agent/td3.py']
  for path in sources:
    for line in path.read_text().splitlines():
      stripped = line.strip()
      if stripped.startswith(('import ', 'from ')):
        # This package's own name starts with 'roboballet' since the rename, so it is exempt.
        assert not any(stripped.startswith(f'{kind} {name}') for kind in ('import', 'from')
                       for name in ('agents', 'environments', 'train', 'launch', 'roboballet', 'jax', 'reverb')
                       if not stripped.startswith(f'{kind} roboballet_mjlab')), f'{path}: {line}'
  for path in sources:
    assert 'fasttd3' not in path.read_text().lower().replace('no fasttd3', ''), path


def test_initial_weights_reach_every_network_the_learner_and_actors_use(tmp_path):
  trainer = make_trainer(tmp_path)
  td3 = trainer.td3
  path = tmp_path / 'weights.pt'
  torch.save({
    'step': 7,
    'policy': {k: v + 0.25 for k, v in td3.policy.state_dict().items()},
    'q1': {k: v + 0.5 for k, v in td3.q1.state_dict().items()},
    'q2': {k: v + 0.75 for k, v in td3.q2.state_dict().items()},
  }, path)
  assert load_checkpoint(str(path), td3.policy, td3.q1, td3.q2) == 7
  td3.sync_targets()
  td3.refresh_acting()
  for online, others in ((td3.policy, (td3.policy_target, td3.policy_acting)), (td3.q1, (td3.q1_target, td3.q1_frozen)), (td3.q2, (td3.q2_target,))):
    for other in others:
      for a, b in zip(online.parameters(), other.parameters()):
        assert torch.equal(a, b)
  assert td3.grad_steps == 0


def test_a_run_directory_keeps_one_checkpoint(tmp_path):
  (tmp_path / f'{CHECKPOINT_NAME}.tmp123').write_bytes(b'left by a killed save')
  trainer = make_trainer(tmp_path, num_worlds=4)
  assert not list(tmp_path.glob(f'{CHECKPOINT_NAME}.tmp*'))
  for reason in ('first', 'second', 'third'):
    trainer.save_checkpoint(reason)
  names = [p.name for p in tmp_path.iterdir() if 'checkpoint' in p.name]
  assert names == [CHECKPOINT_NAME]


def test_resume_keeps_the_fetch_the_first_latch_reads(tmp_path):
  """Late in a run, a resumed world acts on the policy with a small sigma, not uniformly at sigma 0.7."""
  a = make_trainer(tmp_path / 'a', num_worlds=4)
  a.latches.finished[:] = 40
  a.latches.held_step[:] = 1_870_000
  a.published_step = 1_870_010
  path = a.save_checkpoint('late')
  b = make_trainer(tmp_path / 'b', num_worlds=4)
  b.load_checkpoint(path)
  assert not b.latches.use_random.any()
  cfg = a.cfg
  expected = cfg.policy_sigma * cfg.policy_sigma_decay ** (1_870_000 / cfg.decay_every_steps)
  assert b.latches.sigma.cpu().numpy() == pytest.approx(expected, rel=1e-5)
  assert (b.latches.held_step == 1_870_010).all(), 'and the episode it starts records the published step'
  for t in (a, b):
    t.logger.close()


def test_refresh_period_follows_episode_length(tmp_path):
  """Late on, episodes end well before the timeout, and a refresh must come as often as they do."""
  t = make_trainer(tmp_path, num_worlds=4)
  period = 10
  assert period < t.env_cfg.max_steps
  append, published = t.buffer.append, []

  def short_episodes(*args, **kwargs):
    append(*args, **kwargs)
    return SimpleNamespace(length=torch.zeros(t.num_worlds)) if (t.counters.rounds + 1) % period == 0 else None

  refresh = t.publish

  def publish():
    published.append(t.counters.rounds)
    refresh()

  t.buffer.append, t._store, t.publish = short_episodes, lambda finished: None, publish
  for _ in range(6 * period):
    t.round()
  assert published == list(range(period, 6 * period + 1, period))
  t.logger.close()


def test_policy_snapshots_land_on_the_multiples_and_stay(tmp_path):
  from roboballet_mjlab.agent.networks import load_policy
  from roboballet_mjlab.train.trainer import POLICY_DIR

  every = 100
  run = tmp_path / 'run'
  t = make_trainer(run, num_worlds=4, policy_snapshot_every_steps=every, total_grad_steps=2 * every)
  taken = {}
  save = t._snapshot_if_due

  def spy():
    before = t._next_snapshot
    save()
    if t._next_snapshot != before:
      taken[t.td3.grad_steps] = {k: v.clone() for k, v in t.td3.policy.state_dict().items()}

  t._snapshot_if_due = spy
  t.run()
  t.logger.close()
  files = sorted(p.name for p in (run / POLICY_DIR).iterdir())
  assert files == ['policy_00000000.pt', f'policy_{every:08d}.pt', f'policy_{2 * every:08d}.pt']
  assert sorted(taken) == [0, every, 2 * every]
  assert not (run / POLICY_DIR / files[0]).read_bytes() == (run / POLICY_DIR / files[-1]).read_bytes()
  for step, state in taken.items():
    policy = PolicyNet(SMALL, t.topo)
    assert load_policy(str(run / POLICY_DIR / f'policy_{step:08d}.pt'), policy) == step
    for name, value in policy.state_dict().items():
      assert torch.equal(value, state[name].cpu()), f'the snapshot at {step} holds that step\'s weights'

  # A resumed run goes on from the next multiple and rewrites nothing.
  stamps = {p.name: p.stat().st_mtime_ns for p in (run / POLICY_DIR).iterdir()}
  resumed = make_trainer(run, num_worlds=4, policy_snapshot_every_steps=every, total_grad_steps=2 * every)
  resumed.load_checkpoint(run / CHECKPOINT_NAME)
  assert resumed._next_snapshot == 3 * every
  resumed._snapshot_if_due()
  assert {p.name: p.stat().st_mtime_ns for p in (run / POLICY_DIR).iterdir()} == stamps
  resumed.logger.close()


def test_snapshots_can_be_turned_off(tmp_path):
  from roboballet_mjlab.train.trainer import POLICY_DIR

  t = make_trainer(tmp_path, num_worlds=4, policy_snapshot_every_steps=0)
  t._snapshot_if_due()
  assert not (tmp_path / POLICY_DIR).exists()
  t.logger.close()

