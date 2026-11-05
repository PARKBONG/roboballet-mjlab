"""The FastTD3 loop on a real env, replay and learner, on a small model. DESIGN 12.1.

The schedule and the exploration come from the reference's numbers in tests/golden/fasttd3_*.npz.
tests/golden/fasttd3_mutation_check.py injects bugs and confirms that these tests fail.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import build_graph
from roboballet_mjlab.agent.networks import PolicyNet, load_policy
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.fasttd3.trainer import (
  CHECKPOINT_NAME, FAILED_CHECKPOINT_NAME, POLICY_DIR, Trainer, atomic_save, behaviour_action, draw_scales, resample_scales,
)
from roboballet_mjlab.replay.replay import STAT
from tests.test_sim_common import DEVICE, stored_scenes
from roboballet_mjlab.train.evaluator import Evaluator

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / 'golden'
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
T = 3
MAX_STEPS = 50  # Of T = 3.


def small_pool(targets: int = T):
  scenes = stored_scenes(4)
  return scenes._replace(target_pos=scenes.target_pos[:, :targets], target_rot=scenes.target_rot[:, :targets])


def small_cfg(**changes) -> FastTD3Cfg:
  base = dict(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight_flush_episodes=4, compile=False,
              log_every_rounds=25, eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, matmul_precision='high', resume_min_rows=16)
  return FastTD3Cfg(**{**base, **changes})


def make_trainer(run_dir, evaluator=None, **changes) -> Trainer:
  return Trainer(EnvCfg(num_targets=T), small_cfg(**changes), SMALL, small_pool(), DEVICE, run_dir, evaluator=evaluator)


def flat(module) -> torch.Tensor:
  return torch.cat([p.detach().flatten() for p in module.parameters()])


def run_rounds(trainer: Trainer, rounds: int) -> None:
  for _ in range(rounds):
    trainer.round()


def spy_step(trainer: Trainer) -> list:
  """Keeps every action request and StepOut of env.step."""
  log: list = []
  original = trainer.env.step

  def step(action):
    out = original(action)
    log.append((action.clone(), out))
    return out

  trainer.env.step = step
  return log


# Exploration.


def test_explore_matches_the_reference():
  z = np.load(GOLDEN / 'fasttd3_noise.npz')
  cfg = FastTD3Cfg()
  initial = torch.as_tensor(z['initial_scales'])
  assert initial.min() >= cfg.std_min and initial.max() < cfg.std_max
  for t in range(len(z['explore_dones'])):
    ended = torch.as_tensor(z['explore_dones'][t])
    before = torch.as_tensor(z['explore_before'][t])[..., None]
    fresh = torch.as_tensor(z['explore_fresh'][t])[..., None]
    scales = resample_scales(before, ended, fresh, cfg)
    error = (scales - torch.as_tensor(z['explore_after'][t])[..., None]).abs().max().item()
    assert error < 1e-6, f'step {t}: {error}'
    action = behaviour_action(torch.as_tensor(z['explore_mean'][t])[:, None], scales, torch.as_tensor(z['explore_normal'][t])[:, None])
    # The reference leaves the clamp to its env, DESIGN 12.1.
    assert (action[:, 0] - torch.as_tensor(z['explore_action'][t]).clamp(-1, 1)).abs().max().item() < 1e-5


def test_the_scale_range_and_the_draw():
  cfg = FastTD3Cfg()
  assert draw_scales(torch.zeros(1), cfg).item() == pytest.approx(0.001)
  assert draw_scales(torch.ones(1), cfg).item() == pytest.approx(0.4)
  scales = draw_scales(torch.rand(100_000, generator=torch.Generator().manual_seed(0)), cfg)
  assert scales.mean().item() == pytest.approx(0.2005, abs=2e-3), 'uniform'
  assert scales.std().item() == pytest.approx(0.399 / math.sqrt(12), abs=2e-3)


def test_noise_scale_changes_only_for_a_world_that_ended(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9, num_worlds=16)
  steps = spy_step(t)
  ended_total = 0
  assert t.noise_scale.shape == (16, 1, 1)
  first = t.noise_scale.clone()
  assert (first >= 0.001).all() and (first <= 0.4).all() and first.unique().numel() == 16
  for _ in range(MAX_STEPS * 3):
    before = t.noise_scale.clone()
    t.round()
    ended = steps[-1][1].terminal
    ended_total += int(ended.sum())
    changed = (t.noise_scale != before)[:, 0, 0]
    assert torch.equal(changed, ended), f'round {t.counters.rounds}'
    assert (t.noise_scale >= 0.001).all() and (t.noise_scale <= 0.4).all()
  assert ended_total >= 16 * 2, 'each world ended at least twice'


def test_the_stagger_reset_keeps_the_noise_scale(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9)
  first = t.noise_scale.clone()
  run_rounds(t, MAX_STEPS - 1)  # All offsets have fired, nothing has ended by itself.
  assert t.counters.discarded_rows > 0 and t.counters.real_episodes == 0
  assert torch.equal(t.noise_scale, first)


def test_action_is_the_online_mean_plus_scale_times_normal_clamped_and_stored(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 5)  # Learning has begun.
  means, normals = [], []
  act, draw = t.agent.act, t._draw_normals

  def spy_act(graph):
    means.append(act(graph))
    return means[-1]

  def spy_draw(like):
    normals.append(draw(like))
    return normals[-1]

  t.agent.act, t._draw_normals = spy_act, spy_draw
  steps = spy_step(t)
  before = t.buffer.count.clone()
  scales = t.noise_scale.clone()
  t.round()
  action = steps[0][0]
  assert torch.equal(action, (means[0] + scales * normals[0]).clamp(-1.0, 1.0))
  assert means[0].abs().max() < 1 and normals[0].std() > 0.5
  # Per world and per joint.
  assert normals[0].shape == (8, 4, 7) and (normals[0][0, 0] != normals[0][0, 1]).all() and (normals[0][0] != normals[0][1]).any()
  # The buffer keeps what went into env.step. Worlds that just ended have moved it to the ring.
  ended = steps[0][1].terminal
  worlds = (~ended).nonzero().flatten()
  assert torch.equal(t.buffer.action[worlds, before[worlds]], action[worlds])
  assert action.abs().max() <= 1.0


def test_stored_actions_stay_in_range_at_large_noise(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9, std_min=3.0, std_max=3.5)
  steps = spy_step(t)
  run_rounds(t, 5)
  assert all(action.abs().max() <= 1.0 for action, _ in steps)
  assert any((action.abs() == 1.0).any() for action, _ in steps)


def test_acting_uses_the_online_weights_with_no_copy(tmp_path):
  t = make_trainer(tmp_path)
  assert not hasattr(t.agent, 'policy_acting')
  obs = t.env.observe()
  graph = build_graph(obs, t.topo)
  before = t.agent.act(graph)
  run_rounds(t, MAX_STEPS + 3)
  assert t.agent.policy_steps > 0
  assert not torch.equal(before, t.agent.act(graph))
  # Weights used for acting are the ones the optimiser steps.
  assert t.agent._policy is t.agent.policy or t.agent._policy._orig_mod is t.agent.policy


# Schedule, counted on the real loop.


def filled_trainer(tmp_path, name='a', **changes) -> Trainer:
  """A trainer whose ring already holds episodes, so the reference's schedule shows from its first round."""
  filler = make_trainer(tmp_path / f'{name}_filler', learning_starts=10**9)
  run_rounds(filler, MAX_STEPS + 12)
  assert filler.replay.size >= filler.cfg.batch_size and filler.agent.grad_steps == 0
  trainer = make_trainer(tmp_path / f'{name}_trainer', **changes)
  trainer.replay = filler.replay
  return trainer


def recorded_schedule(t: Trainer, rounds: int) -> list[tuple[int, int]]:
  calls: list[tuple[int, int]] = []
  sample, critic, actor, target = t.replay.sample, t.agent._update_critic, t.agent._update_actor, t.agent._update_target

  def spy(kind, fn):
    def wrapped(*args, **kwargs):
      calls.append((t.counters.rounds - 1, kind))
      return fn(*args, **kwargs)
    return wrapped

  t.replay.sample = spy(0, sample)
  t.agent._update_critic, t.agent._update_actor, t.agent._update_target = spy(1, critic), spy(2, actor), spy(3, target)
  run_rounds(t, rounds)
  return calls


def test_update_schedule_equals_the_reference_call_for_call(tmp_path):
  t = filled_trainer(tmp_path)
  calls = recorded_schedule(t, 40)
  z = np.load(GOLDEN / 'fasttd3_schedule.npz')['default']
  assert calls == [tuple(row) for row in z.tolist()]
  assert t.agent.grad_steps == 2 * 29 and t.agent.policy_steps == 29


def test_one_update_per_round_takes_the_other_branch_of_the_reference(tmp_path):
  t = filled_trainer(tmp_path, num_updates=1)
  calls = recorded_schedule(t, 40)
  z = np.load(GOLDEN / 'fasttd3_schedule.npz')['one_update']
  assert calls == [tuple(row) for row in z.tolist()]


def test_each_update_has_a_fresh_batch_and_the_actor_shares_the_critics(tmp_path):
  t = filled_trainer(tmp_path, learning_starts=-1)
  critic_batches, actor_batches, positions = [], [], []
  critic, actor = t.agent._update_critic, t.agent._update_actor
  fetch = t.replay.fetch

  def spy_fetch(position, *rest):
    positions.append(position.clone())
    return fetch(position, *rest)

  t.replay.fetch = spy_fetch
  t.agent._update_critic = lambda batch: (critic_batches.append(batch), critic(batch))[1]
  t.agent._update_actor = lambda batch: (actor_batches.append(batch), actor(batch))[1]
  run_rounds(t, 3)
  assert len(critic_batches) == 6 and len(actor_batches) == 3
  assert all(a is c for a, c in zip(actor_batches, critic_batches[1::2]))
  assert len(positions) == 6
  for p, q in zip(positions, positions[1:]):
    assert p.shape == (16,) and not torch.equal(p, q)


def test_the_learner_gets_the_sampled_rows_with_features_of_s_and_of_s_prime(tmp_path):
  t = filled_trainer(tmp_path, learning_starts=-1)
  samples, batches = [], []
  sample, critic = t.replay.sample, t.agent._update_critic
  t.replay.sample = lambda *args: (samples.append(sample(*args)), samples[-1])[1]
  t.agent._update_critic = lambda batch: (batches.append(batch), critic(batch))[1]
  run_rounds(t, 2)
  assert len(samples) == len(batches) == 4
  for s, b in zip(samples, batches):
    assert s.next_obs.robot.step.ne(s.obs.robot.step).all(), 's and s\' are different rows'
    assert all(torch.equal(x, y) for x, y in zip(b.graph, build_graph(s.obs, t.topo)))
    assert all(torch.equal(x, y) for x, y in zip(b.next_graph, build_graph(s.next_obs, t.topo)))
    assert torch.equal(b.action, s.action) and torch.equal(b.reward, s.reward) and torch.equal(b.terminal, s.terminal)


def test_updates_wait_for_a_batch_in_the_ring(tmp_path):
  """Whole episodes enter this ring, so no row exists before max_steps: DESIGN 12.1 against train.py:643."""
  t = make_trainer(tmp_path, batch_size=120, num_worlds=8)
  for _ in range(MAX_STEPS * 2):
    size_before = t.replay.size
    grads_before = t.agent.grad_steps
    t.round()
    if t.agent.grad_steps > grads_before:
      assert t.replay.size >= 120
      break
    assert t.replay.size < 120 or t.counters.rounds <= t.cfg.learning_starts + 1
  assert t.agent.grad_steps == 2
  assert t.counters.rounds > MAX_STEPS
  assert size_before < 120


def test_learning_starts_counts_rounds_as_the_reference(tmp_path):
  t = filled_trainer(tmp_path, learning_starts=6)
  calls = recorded_schedule(t, 9)
  assert {step for step, _ in calls} == {7, 8}, 'global_step > learning_starts'


# Replay and hindsight.


def test_real_episodes_enter_before_their_hindsight(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9)
  order = []
  insert = t.replay.insert

  def spy(batch):
    order.append(bool(batch.is_hindsight.any()))
    assert batch.is_hindsight.all() == batch.is_hindsight.any()
    return insert(batch)

  t.replay.insert = spy
  run_rounds(t, MAX_STEPS * 4)
  assert any(order) and not all(order)
  assert order[0] is False
  assert t.counters.hindsight_episodes > 0
  # Half the transitions are hindsight under a policy that solves nothing.
  fraction = t.replay.is_hindsight[:t.replay.episodes].float().mean().item()
  assert fraction == pytest.approx(0.5, abs=0.1)


def test_within_one_store_the_real_episodes_go_first(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9)
  stores: list[list[bool]] = []
  insert, store = t.replay.insert, t._store

  def spy_insert(batch):
    stores[-1].append(bool(batch.is_hindsight.any()))
    return insert(batch)

  def spy_store(finished):
    stores.append([])
    return store(finished)

  t.replay.insert, t._store = spy_insert, spy_store
  run_rounds(t, MAX_STEPS * 4)
  assert stores and all(entry in ([False], [False, True]) for entry in stores), stores
  assert [False, True] in stores


def test_hindsight_uses_the_configured_flush_and_can_be_turned_off(tmp_path):
  t = make_trainer(tmp_path / 'on', learning_starts=10**9, hindsight_flush_episodes=7)
  assert t.queue.train_cfg is t.cfg and t.queue.train_cfg.hindsight_flush_episodes == 7
  assert t.queue.k == int(t.cfg.hindsight_fraction * T)
  off = make_trainer(tmp_path / 'off', learning_starts=10**9, hindsight=False)
  run_rounds(off, MAX_STEPS * 2)
  assert off.queue is None and off.counters.hindsight_episodes == 0 and off.counters.real_episodes > 0
  assert not off.replay.is_hindsight.any()


def test_ring_capacity_and_compact_rows(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9, replay_max_size=300)
  run_rounds(t, MAX_STEPS * 3)
  assert t.replay.capacity == 300 and t.replay.size <= 300 and t.replay.evicted > 0


def test_no_terminal_bootstraps_in_the_batch_the_learner_gets(tmp_path):
  """The ring's terminal flag is True on timeouts: every last action of an episode reaches the learner so."""
  t = make_trainer(tmp_path, learning_starts=-1, batch_size=128)
  seen = []
  critic = t.agent._update_critic
  t.agent._update_critic = lambda batch: (seen.append(batch.terminal.clone()), critic(batch))[1]
  run_rounds(t, MAX_STEPS * 2)
  assert seen and sum(int(x.sum()) for x in seen) > 0
  assert seen[-1].dtype == torch.bool


def test_every_stream_has_its_own_seed(tmp_path):
  """One seed everywhere would make action noise and target noise the same draws, and sampling follow them."""
  t = make_trainer(tmp_path)
  draws = [generator.manual_seed(generator.initial_seed()) and torch.randint(2**31 - 1, (4,), generator=generator, device=DEVICE).tolist()
           for generator in (t.act_rng, t.sample_rng, t.hindsight_rng, t.env.generator, t.agent.rng)]
  assert all(a != b for i, a in enumerate(draws) for b in draws[i + 1:])
  other = make_trainer(tmp_path / 'b', seed=2)
  assert t.act_rng.initial_seed() != other.act_rng.initial_seed()
  assert t.agent.rng.initial_seed() != other.agent.rng.initial_seed()


# Stagger.


def test_each_world_resets_once_at_its_offset(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9)
  resets = []
  reset = t.env.reset

  def spy(world_ids=None, scene_ids=None):
    resets.append((t._rounds_this_run, world_ids.tolist()))
    return reset(world_ids, scene_ids)

  t.env.reset = spy
  run_rounds(t, MAX_STEPS - 1)  # World 0's own end, and the env's reset after it, comes next.
  assert resets == [(i * MAX_STEPS // 8, [i]) for i in range(1, 8)]
  assert t.counters.discarded_rows == sum(i * MAX_STEPS // 8 for i in range(1, 8))


def test_no_partial_episode_enters_the_ring(tmp_path):
  t = make_trainer(tmp_path, learning_starts=10**9, hindsight=False)
  run_rounds(t, MAX_STEPS * 2)
  stats = t.replay.stats[:t.replay.episodes]
  assert (stats[:, STAT['episode_length']] == MAX_STEPS + 1).all(), 'random motion solves nothing, so all last 50 actions'


# The hot path.


def test_the_hot_path_reads_nothing_back_from_this_package(tmp_path):
  """Python-level reads, from fasttd3/ files. Shared modules are as the original's."""
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  reads = []
  saved = {}

  def counted(name):
    original = saved[name] = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
      caller = sys._getframe(1).f_code.co_filename
      if '/fasttd3/' in caller:  # Not the overlay's temporary directory of the mutation check.
        reads.append((name, caller, sys._getframe(1).f_lineno))
      return original(self, *args, **kwargs)

    return wrapper

  names = ('item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', '__float__', '__index__')
  for name in names:
    setattr(torch.Tensor, name, counted(name))
  try:
    start = t.counters.rounds
    while t.counters.rounds < start + 3 * t.cfg.log_every_rounds:
      if (t.counters.rounds + 1) % t.cfg.log_every_rounds:  # Report rounds may read.
        t.round()
      else:
        for name in names:
          setattr(torch.Tensor, name, saved[name])
        t.round()
        for name in names:
          setattr(torch.Tensor, name, counted(name))
  finally:
    for name in names:
      setattr(torch.Tensor, name, saved[name])
  assert reads == []


def test_a_report_writes_names_of_the_original_and_the_reference(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=30)
  run_rounds(t, 2 * MAX_STEPS)
  t.logger.flush()
  tags = {row['tag'] for row in csv.DictReader(open(tmp_path / 'metrics.csv'))}
  for name in ('train/qf_loss', 'train/actor_loss', 'train/critic_grad_norm', 'train/actor_grad_norm', 'train/target_value_mean',
               'train/target_value_min', 'train/target_value_max', 'train/target_mass_at_vmin', 'train/target_mass_at_vmax',
               'train/trajectory_network_age', 'train/policy_lr', 'train/twin_critic_lr',
               'rate/env_steps_per_s', 'rate/grad_steps_per_s', 'counters/env_steps', 'counters/grad_steps',
               'act/num_targets_done', 'act/is_hindsight_replay', 'act/total_collision_robot_steps', 'act/final_score',
               'act/policy_action_abs', 'act/policy_action_saturated'):
    assert name in tags, name


def test_totals_add_up(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=30, batch_size=16)
  run_rounds(t, 2 * MAX_STEPS)
  totals = t.totals()
  assert totals['env_steps'] == 2 * MAX_STEPS * 8
  assert totals['rounds'] == 2 * MAX_STEPS
  assert totals['grad_steps'] == 2 * totals['policy_steps']
  assert totals['sampled'] == totals['grad_steps'] * 16
  real = totals['real_episodes']
  assert real > 0 and totals['inserted'] == real * MAX_STEPS + totals['hindsight_episodes'] * MAX_STEPS
  assert totals['residual_collisions'] == 0


# Float modes in the running loop, DESIGN 14.1.


def test_precision_stays_highest_for_everything_but_the_learner(tmp_path):
  t = make_trainer(tmp_path)
  seen = {}
  act, step, update = t.agent.act, t.env.step, t.agent._update_critic

  def note(name, fn):
    def wrapped(*args, **kwargs):
      seen.setdefault(name, set()).add(torch.get_float32_matmul_precision())
      return fn(*args, **kwargs)
    return wrapped

  t.agent.act, t.env.step, t.agent._update_critic = note('act', act), note('env', step), note('learner', update)
  run_rounds(t, MAX_STEPS + 3)
  assert seen == {'act': {'highest'}, 'env': {'highest'}, 'learner': {'high'}}
  assert torch.get_float32_matmul_precision() == 'highest'


def test_features_give_the_same_bytes_after_a_learner_round(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 3)
  obs = t.env.observe()
  before = [x.clone() for x in build_graph(obs, t.topo)]
  t.round()
  assert all(torch.equal(a, b) for a, b in zip(before, build_graph(obs, t.topo)))


# Evaluator.


def test_the_evaluator_takes_the_actor_unchanged(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 3)
  evaluator = Evaluator(t.env_cfg, SMALL, {'training_dist': small_pool()}, DEVICE, 1, num_worlds=8, compile=False, notice=lambda _: None)
  assert type(t.agent.policy) is PolicyNet
  results = evaluator.evaluate(t.agent.policy, t.agent.grad_steps)['training_dist']
  assert torch.equal(flat(evaluator.policy), flat(t.agent.policy))
  assert 0.0 <= results['task_rate'] <= 1.0 and results['worlds'] == 8.0 and results['network_step'] == t.agent.grad_steps


def test_evaluation_mid_training_changes_nothing_in_training(tmp_path):
  """The fairness of DESIGN 12.2 needs this: a twin that evaluates stays bit-exact with one that does not."""
  evaluator = Evaluator(EnvCfg(num_targets=T), SMALL, {'training_dist': small_pool()}, DEVICE, 1, num_worlds=8, compile=False, notice=lambda _: None)
  plain = make_trainer(tmp_path / 'plain')
  evaluated = make_trainer(tmp_path / 'evaluated', evaluator=evaluator, eval_every_rounds=20)
  for _ in range(MAX_STEPS + 30):
    plain.round()
    evaluated.round()
  assert evaluated._eval_seconds > 0
  assert torch.equal(flat(plain.agent.policy), flat(evaluated.agent.policy))
  assert torch.equal(flat(plain.agent.q1), flat(evaluated.agent.q1))
  assert torch.equal(plain.noise_scale, evaluated.noise_scale)
  assert torch.equal(plain.env.generator.get_state(), evaluated.env.generator.get_state())
  assert torch.equal(plain.act_rng.get_state(), evaluated.act_rng.get_state())


# Failure handling and checkpoints.


def test_nonfinite_loss_saves_a_failed_checkpoint_and_raises(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  good = t.save_checkpoint('good')
  before = good.read_bytes()
  for p in t.agent.q1_target.parameters():
    p.data.fill_(float('nan'))
  for p in t.agent.q2_target.parameters():
    p.data.fill_(float('nan'))
  with pytest.raises(FloatingPointError, match='nonfinite loss'):
    run_rounds(t, t.cfg.log_every_rounds + 1)
  assert (tmp_path / FAILED_CHECKPOINT_NAME).exists()
  assert good.read_bytes() == before, 'the last good checkpoint stays'
  fresh = make_trainer(tmp_path / 'fresh')
  with pytest.raises(ValueError, match='failed checkpoint'):
    fresh.load_checkpoint(tmp_path / FAILED_CHECKPOINT_NAME)
  fresh.load_checkpoint(good)


def test_a_nan_policy_reaches_the_report_through_the_env(tmp_path):
  """Acting on NaN for up to a report's rounds must end in the same clean failure, not in the env."""
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  for p in t.agent.policy.parameters():
    p.data.fill_(float('nan'))
  with pytest.raises(FloatingPointError, match='nonfinite loss'):
    run_rounds(t, t.cfg.log_every_rounds + 1)
  assert (tmp_path / FAILED_CHECKPOINT_NAME).exists()


def test_atomic_save_keeps_the_old_file_when_writing_fails(tmp_path):
  path = tmp_path / 'x.pt'
  atomic_save({'a': 1}, path)
  with pytest.raises(Exception):
    atomic_save({'a': lambda: 0}, path)
  assert torch.load(path, weights_only=True) == {'a': 1}
  assert [p.name for p in tmp_path.iterdir()] == ['x.pt']


def test_killed_mid_write_leaves_the_old_file(tmp_path):
  path = tmp_path / 'x.pt'
  atomic_save({'a': 1}, path)
  code = f"""
import os, signal, torch
from pathlib import Path
from roboballet_mjlab.fasttd3.trainer import atomic_save
original = torch.save
def killed(payload, f):
  f.write(b'garbage')
  f.flush()
  os.kill(os.getpid(), signal.SIGKILL)
torch.save = killed
atomic_save({{'a': 2}}, Path({str(path)!r}))
"""
  done = subprocess.run([sys.executable, '-c', code], env=script_env(), capture_output=True)
  assert done.returncode == -signal.SIGKILL
  assert torch.load(path, weights_only=True) == {'a': 1}


def test_save_and_resume_restores_everything(tmp_path):
  t = make_trainer(tmp_path / 'a', log_every_rounds=30)
  run_rounds(t, MAX_STEPS + 20)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b', seed=1, log_every_rounds=30)
  r.load_checkpoint(path)
  a, b = t.agent, r.agent
  assert (a.grad_steps, a.policy_steps) == (b.grad_steps, b.policy_steps) and a.grad_steps > 0
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(flat(getattr(a, name)), flat(getattr(b, name))), name
  assert torch.equal(flat(b.q1), flat(b.q1_frozen))
  for p, q in zip(a._critic_params, b._critic_params):
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
      assert torch.equal(a.critic_optimizer.state[p][key], b.critic_optimizer.state[q][key]), key
  for p, q in zip(a._policy_params, b._policy_params):
    if p.grad is not None:
      assert torch.equal(a.policy_optimizer.state[p]['exp_avg'], b.policy_optimizer.state[q]['exp_avg'])
  assert torch.equal(t.noise_scale, r.noise_scale)
  assert torch.equal(t.act_rng.get_state(), r.act_rng.get_state())
  assert torch.equal(t.sample_rng.get_state(), r.sample_rng.get_state())
  assert torch.equal(t.hindsight_rng.get_state(), r.hindsight_rng.get_state())
  assert torch.equal(t.env.generator.get_state(), r.env.generator.get_state())
  assert torch.equal(a.rng.get_state(), b.rng.get_state())
  assert r.counters.rounds == t.counters.rounds and r.totals()['inserted'] == t.totals()['inserted']
  assert r.totals()['sampled'] == t.totals()['sampled']


def test_resumed_run_refills_the_ring_before_updating(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, MAX_STEPS + 5)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b')
  r.load_checkpoint(path)
  assert r.replay.size == 0
  grad_steps = r.agent.grad_steps
  rounds = r.counters.rounds
  run_rounds(r, MAX_STEPS - 1)
  assert r.agent.grad_steps == grad_steps, 'nothing to sample yet'
  run_rounds(r, 2)
  assert r.agent.grad_steps > grad_steps
  assert r.counters.rounds == rounds + MAX_STEPS + 1
  assert r.totals()['inserted'] > t.totals()['inserted']


def test_resumed_run_waits_for_resume_min_rows_and_a_fresh_run_does_not(tmp_path):
  assert FastTD3Cfg().resume_min_rows == 200_000
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, MAX_STEPS + 5)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b', resume_min_rows=300)
  r.load_checkpoint(path)
  grad_steps, size = r.agent.grad_steps, 0
  while r.agent.grad_steps == grad_steps:
    assert size < 300, f'{size} rows and no update'
    size = r.replay.size
    r.round()
  assert size < 300 <= r.replay.size, 'starts on the round the ring reaches it'
  assert r.agent.grad_steps == grad_steps + 2
  # The same setting on a fresh run changes nothing, down to the weights.
  fresh = [make_trainer(tmp_path / name, resume_min_rows=rows) for name, rows in (('c', 16), ('d', 10**6))]
  for x in fresh:
    run_rounds(x, MAX_STEPS + 5)
  assert fresh[0].agent.grad_steps == fresh[1].agent.grad_steps == t.agent.grad_steps > 0
  assert torch.equal(flat(fresh[0].agent.policy), flat(fresh[1].agent.policy))


def test_a_ring_smaller_than_resume_min_rows_does_not_wait_forever(tmp_path):
  r = make_trainer(tmp_path / 'b', replay_max_size=200, resume_min_rows=300)
  r.load_checkpoint(make_trainer(tmp_path / 'c', replay_max_size=200).save_checkpoint('test'))
  run_rounds(r, MAX_STEPS * 3)
  assert r.agent.grad_steps > 0 and r.replay.size <= 200


def test_resume_refuses_a_changed_config_but_not_a_changed_cadence(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, 2)
  path = t.save_checkpoint('test')
  with pytest.raises(ValueError, match='gamma'):
    make_trainer(tmp_path / 'b', gamma=0.9).load_checkpoint(path)
  with pytest.raises(ValueError, match='not a FastTD3 checkpoint'):
    bad = tmp_path / 'other.pt'
    torch.save({'format': 1}, bad)
    make_trainer(tmp_path / 'c').load_checkpoint(bad)
  make_trainer(tmp_path / 'd', total_env_rounds=7, eval_every_rounds=5, log_every_rounds=3, profile_every_rounds=9, checkpoint_every_s=5.0,
               policy_snapshot_every_steps=3, resume_min_rows=7).load_checkpoint(path)


def test_a_checkpoint_older_than_a_field_resumes_at_its_default(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, 2)
  state = torch.load(t.save_checkpoint('test'), map_location='cpu', weights_only=True)
  for key in ('n_step', 'policy_snapshot_every_steps', 'resume_min_rows'):
    del state['config']['train'][key]
  old = tmp_path / 'old.pt'
  torch.save(state, old)
  make_trainer(tmp_path / 'b').load_checkpoint(old)
  with pytest.raises(ValueError, match='n_step'):
    make_trainer(tmp_path / 'c', n_step=3).load_checkpoint(old)


def test_policy_snapshots_are_kept_load_as_a_policy_and_continue_after_a_resume(tmp_path):
  assert FastTD3Cfg().checkpoint_every_s == 1800.0
  run_dir = tmp_path / 'a'
  t = make_trainer(run_dir, total_env_rounds=1, policy_snapshot_every_steps=20)
  t.run()
  t.cfg = dataclasses.replace(t.cfg, total_env_rounds=10**6)
  directory = run_dir / POLICY_DIR
  assert [p.name for p in directory.iterdir()] == ['policy_00000000.pt'], 'the untrained policy, at the start of a new run'
  while t.agent.grad_steps < 20:
    t.round()
  policy = PolicyNet(SMALL, t.topo).to(DEVICE)
  assert load_policy(str(directory / 'policy_00000020.pt'), policy) == 20
  assert torch.equal(flat(policy), flat(t.agent.policy)), 'the weights of that step'
  state = torch.load(directory / 'policy_00000020.pt', weights_only=True)
  assert set(state) == {'step', 'policy', 'model'} and state['model'] == dataclasses.asdict(SMALL)
  run_rounds(t, 12)
  steps = [int(p.name[len('policy_'):-len('.pt')]) for p in sorted(directory.iterdir())]
  assert steps == [0, 20, 40], f'at step 0 and on each multiple, so {t.agent.grad_steps} grad steps give these'
  path = t.save_checkpoint('test')

  r = make_trainer(run_dir, policy_snapshot_every_steps=20)
  r.load_checkpoint(path)
  r._snapshot_if_due()
  assert len(list(directory.iterdir())) == 3, 'a resume does not write the step it starts from'
  assert r._next_snapshot == (r.agent.grad_steps // 20 + 1) * 20
  for changes, why in (({'rank': 1}, 'rank 0 only'), ({'_next_snapshot': 0}, 'off')):
    other = make_trainer(tmp_path / why.replace(' ', '_'), policy_snapshot_every_steps=20 if 'rank' in changes else 0)
    for key, value in changes.items():
      setattr(other, key, value)
    other._snapshot_if_due()
    assert not (other.run_dir / POLICY_DIR).exists(), why


def test_a_new_trainer_removes_what_a_killed_save_left(tmp_path):
  (tmp_path / POLICY_DIR).mkdir()
  stale = [tmp_path / f'{CHECKPOINT_NAME}.tmp99999', tmp_path / POLICY_DIR / 'policy_00000000.pt.tmp99999']
  kept = [tmp_path / CHECKPOINT_NAME, tmp_path / POLICY_DIR / 'policy_00000000.pt']
  for path in (*stale, *kept):
    path.write_bytes(b'x')
  make_trainer(tmp_path)
  assert not any(path.exists() for path in stale) and all(path.exists() for path in kept)


def test_a_run_that_ends_inside_a_profiler_window_restores_tensor_item(tmp_path):
  item = torch.Tensor.item
  try:
    t = make_trainer(tmp_path / 'a', profile=True, total_env_rounds=8)
    t.profiler.first_window, t.profiler.window_rounds = 4, 100
    assert t.run() == 'complete'
    assert not t.profiler.in_window and torch.Tensor.item is item
    t = make_trainer(tmp_path / 'b', profile=True, total_env_rounds=8)
    t.profiler.first_window, t.profiler.window_rounds = 4, 100
    rounds = t.round

    def fails():
      rounds()
      if t.counters.rounds == 6:
        assert t.profiler.in_window
        raise FloatingPointError('in a window')

    t.round = fails
    with pytest.raises(FloatingPointError):
      t.run()
    assert torch.Tensor.item is item
  finally:
    torch.Tensor.item = item


def test_run_stops_on_a_signal_and_on_the_time_limit_and_saves(tmp_path):
  t = make_trainer(tmp_path / 'signal', total_env_rounds=10**6)
  t.install_signal_handlers()
  try:
    original = t.round

    def signalled():
      original()
      if t.counters.rounds == 5:
        os.kill(os.getpid(), signal.SIGTERM)

    t.round = signalled
    assert t.run() == f'signal {int(signal.SIGTERM)}'
    assert t.counters.rounds == 5 and (tmp_path / 'signal' / CHECKPOINT_NAME).exists()
  finally:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
  t = Trainer(EnvCfg(num_targets=T), small_cfg(total_env_rounds=10**6), SMALL, small_pool(), DEVICE, tmp_path / 'timed', stop_after_s=0.2)
  assert t.run() == 'stop_after_seconds'
  t = make_trainer(tmp_path / 'done', total_env_rounds=3)
  assert t.run() == 'complete' and t.counters.rounds == 3


def test_periodic_checkpoint(tmp_path):
  t = make_trainer(tmp_path, total_env_rounds=3, checkpoint_every_s=0.0)
  assert t.run() == 'complete'
  assert t._checkpoint_seconds > 0 and (tmp_path / CHECKPOINT_NAME).exists()
  assert not list(tmp_path.glob('*.tmp*'))


def script_env() -> dict:
  """PYTHONPATH as it came, so an overlay of the mutation check reaches the subprocess too."""
  return dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH', str(REPO / 'src')), PYTHONUNBUFFERED='1')


def small_pool_file(path: Path) -> Path:
  from roboballet_mjlab.scenes.io import Pool, save_pool
  from roboballet_mjlab.scenes.sampler import Witnesses

  save_pool(path, Pool(small_pool(), Witnesses(torch.zeros(1), torch.zeros(1)), {'test': True}))
  return path


def script_args(run_dir: Path, pool: Path, *extra: str) -> list[str]:
  sets = ['num_worlds=8', 'batch_size=16', 'replay_max_size=4000', 'hindsight_flush_episodes=4', 'compile=False', 'log_every_rounds=25',
          'eval_every_rounds=40', 'eval_episodes=8', 'profile=False', 'resume_min_rows=16']
  models = ['node_layers=2', 'node_size=16', 'edge_embed_layers=2', 'edge_update_layers=2', 'edge_size=8', 'global_layers=2', 'head_sizes=8']
  args = [sys.executable, '-m', 'roboballet_mjlab.fasttd3.train', '--run-dir', str(run_dir), '--train-pool', str(pool)]
  for item in sets:
    args += ['--set', item]
  for item in models:
    args += ['--model-set', item]
  return args + list(extra)


def wait_for(path: Path, process: subprocess.Popen, timeout: float = 300.0) -> None:
  deadline = time.monotonic() + timeout
  while not path.exists():
    assert process.poll() is None, f'exited early with {process.returncode}'
    assert time.monotonic() < deadline, f'no {path}'
    time.sleep(0.5)


def test_script_records_provenance_stops_on_sigterm_and_resumes(tmp_path):
  pool = small_pool_file(tmp_path / 'pool.pt')
  run_dir = tmp_path / 'run'
  log = open(tmp_path / 'log.txt', 'w')
  process = subprocess.Popen(script_args(run_dir, pool), env=script_env(), stdout=log, stderr=subprocess.STDOUT)
  try:
    wait_for(run_dir / 'tb', process)
    deadline = time.monotonic() + 300
    # Past the first report.
    while b'counters/rounds' not in (run_dir / 'metrics.csv').read_bytes():
      assert process.poll() is None and time.monotonic() < deadline
      time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=120) == 0
  finally:
    process.kill()
  state = torch.load(run_dir / CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert state['format'] == 'fasttd3-1' and state['reason'] == f'signal {int(signal.SIGTERM)}'
  provenance = next((run_dir / 'provenance').iterdir())
  assert {p.name for p in provenance.iterdir()} >= {'config.json', 'git_revision.txt', 'git_diff.patch', 'git_untracked.txt', 'system.json'}
  config = json.loads((provenance / 'config.json').read_text())
  assert config['train']['gamma'] == 0.94 and config['model']['node_size'] == 16
  rounds = json.loads((run_dir / 'summary.json').read_text())['rounds']
  assert rounds == state['counters']['rounds'] and rounds >= 25

  done = subprocess.run(script_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 3)), env=script_env(), capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  assert 'resumed from' in done.stdout
  assert json.loads((run_dir / 'summary.json').read_text())['rounds'] == rounds + 3
  assert len(list((run_dir / 'provenance').iterdir())) == 2

  done = subprocess.run(script_args(run_dir, pool, '--eval-only', str(run_dir / CHECKPOINT_NAME)), env=script_env(), capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  results = json.loads(done.stdout[done.stdout.index('{\n'):])
  assert results['results']['training_dist']['worlds'] == 8.0


def test_end_to_end_compiled_bf16_small_model(tmp_path):
  """As training runs: compiled, bf16, profiler on. Finite, counters agree, memory flat."""
  t = make_trainer(tmp_path, compile=True, bf16=True, profile=True, profile_every_rounds=40, num_worlds=16, batch_size=32, log_every_rounds=20)
  t.profiler.first_window = 60
  t.profiler.window_rounds = 10
  run_rounds(t, MAX_STEPS + 40)
  memory = torch.cuda.memory_allocated()
  run_rounds(t, 60)
  assert torch.cuda.memory_allocated() - memory < 32 * 2**20
  assert t.agent.grad_steps == 2 * (t.counters.rounds - MAX_STEPS + 1), 'from the round the first episode ends'
  assert all(p.isfinite().all() for p in t.agent.policy.parameters())
  values = {}
  t.logger.flush()
  for row in csv.DictReader(open(tmp_path / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  assert math.isfinite(values['train/qf_loss']) and math.isfinite(values['train/actor_loss'])
  assert 0.0 < values['train/qf_loss'] < 2 * 2 * math.log(101) + 1.0
  assert 'profile/critic_update_ms' in values and 'profile/actor_update_ms' in values and 'profile/target_update_ms' in values
  assert values['profile/window_item_calls_per_round'] < 2
