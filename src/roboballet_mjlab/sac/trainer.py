# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The loop of github.com/leggedrobotics/rsl_rl_sac at e0d243a, rsl_rl/runners/off_policy_runner.py,
# rewritten for the shared env, replay and hindsight modules of this package.
"""The SAC training loop: one process per GPU. DESIGN section 16, off_policy_runner.py:84-111.

Every `num_steps_per_env` rounds, a round being one env step of every world, the actor's samples are
followed by `num_learning_epochs * num_mini_batches` updates, each on a fresh batch, sac.py:233-237.
There is no random phase. With several processes each has its own env and replay and the gradients are
averaged, sac.py:299-300. All of it is on the GPU, and nothing reads a tensor back between reports.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Callable
import zlib

import numpy as np
import torch
import torch.distributed as dist

from roboballet_mjlab.agent.features import ACTION_DIM, Topology, build_graph
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.replay.hindsight import HindsightQueue
from roboballet_mjlab.replay.replay import STAT_NAMES, EpisodeBatch, EpisodeBuffer, ReplayBuffer
from roboballet_mjlab.sac.agent import SAC, ActorUpdate, Batch, Update
from roboballet_mjlab.sac.config import SACCfg
from roboballet_mjlab.structs import Obs, Scene
from roboballet_mjlab.train.evaluator import Evaluator
from roboballet_mjlab.train.logger import MetricLogger
from roboballet_mjlab.train.profiler import Profiler

CHECKPOINT_FORMAT = 'sac-1'
CHECKPOINT_NAME = 'checkpoint.pt'
FAILED_CHECKPOINT_NAME = 'checkpoint_failed.pt'
POLICY_DIR = 'policies'
# SACCfg fields a resumed run may change: they do not affect what has been computed so far.
# resume_min_rows came after the first runs, so older checkpoints do not hold it.
RESUMABLE_CHANGES = {'total_env_rounds', 'eval_every_rounds', 'eval_episodes', 'checkpoint_every_s', 'policy_snapshot_every_steps',
                     'log_every_rounds', 'profile', 'profile_every_rounds', 'resume_min_rows'}
UPDATE_TAGS = ('qf_loss', 'critic1_loss', 'critic2_loss', 'critic_grad_norm', 'target_value_mean', 'target_value_min',
               'target_value_max', 'q_mean', 'q_std', 'alpha_loss', 'alpha', 'entropy', 'log_std')


def atomic_save(payload: dict[str, Any], path: Path) -> None:
  """A reader sees the old file or the whole new one."""
  path = Path(path)
  tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
  try:
    with open(tmp, 'wb') as f:
      torch.save(payload, f)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  except BaseException:
    tmp.unlink(missing_ok=True)
    raise


def average_gradients(world_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
  """A flat all-reduce, sac.py:658-681."""

  def hook(flat: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    return flat / world_size

  return hook


class _NoLogger:
  """What a rank other than 0 logs to."""

  def scalars(self, values: dict[str, float], step: int) -> None:
    pass

  def notice(self, message: str) -> None:
    pass

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


@dataclasses.dataclass
class Counters:
  rounds: int = 0
  real_episodes: int = 0
  hindsight_episodes: int = 0
  discarded_rows: int = 0  # Partial rows of the stagger. Never inserted, never counted as episodes.
  # From earlier parts of a resumed run. Replay counters restart with the empty ring.
  inserted_before: int = 0
  sampled_before: int = 0
  residual_collisions_before: int = 0
  invalid_scene_draws_before: int = 0


class Trainer:
  def __init__(
    self, env_cfg: EnvCfg, cfg: SACCfg, model_cfg: ModelCfg, pool: Scene, device, run_dir: str | Path,
    evaluator: Evaluator | None = None, logger: MetricLogger | None = None, stop_after_s: float | None = None,
    rank: int = 0, world_size: int = 1,
  ) -> None:
    # Everything but the learner call runs at highest, and SAC.update restores it. DESIGN 14.1.
    torch.set_float32_matmul_precision('highest')
    self.env_cfg, self.cfg, self.model_cfg = env_cfg, cfg, model_cfg
    self.device = device
    self.rank, self.world_size = rank, world_size
    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.logger = (logger if logger is not None else MetricLogger(self.run_dir)) if rank == 0 else _NoLogger()
    self.evaluator = evaluator if rank == 0 else None
    self.stop_after_s = stop_after_s
    W = self.num_worlds = cfg.num_worlds

    self.env = BalletEnv(env_cfg, W, pool, device, seed=self._seed('env'))
    self.topo = Topology(env_cfg, device)
    hook = average_gradients(world_size) if world_size > 1 else None
    self.agent = SAC(model_cfg, cfg, self.topo, device, grad_hook=hook, noise_seed=self._seed('learner_noise'))
    if world_size > 1:
      self._broadcast_parameters()
    self.buffer = EpisodeBuffer(env_cfg, W, device)
    self.replay = ReplayBuffer(env_cfg, cfg.replay_max_size, device)
    self.act_rng = self._generator('act')
    self.sample_rng = self._generator('sample')
    self.hindsight_rng = self._generator('hindsight')
    self.queue = HindsightQueue(env_cfg, cfg, self.hindsight_rng) if cfg.hindsight else None
    self._action_shape = (W, self.topo.num_robots, ACTION_DIM)

    self.counters = Counters()
    self._all = torch.arange(W, device=device)
    self._rounds_this_run = 0
    self._learning = False
    self._min_rows = cfg.batch_size  # load_checkpoint raises it.
    # Each world resets once at its own offset, so whole episodes reach the ring evenly. World 0 runs as it is.
    offsets = (self._all * env_cfg.max_steps) // W
    self._stagger_at = offsets.masked_fill(offsets == 0, -1)
    self._stagger_end = int(offsets.max()) + 1

    self.profiler = Profiler(cfg.profile, device, launches=lambda: self.env.world.launches,
                             every_rounds=cfg.profile_every_rounds)
    self.profiler.wrap(self.env.world, 'forward', 'world_forward')
    self.profiler.wrap(self.env, '_move', 'rollback')
    self.profiler.wrap(self.env, 'reset', 'reset')
    self.profiler.wrap(self.agent, '_update_critic', 'critic_update')
    self.profiler.wrap(self.agent, '_sample', 'actor_sample')  # Inside the critic update too, as exclusive time.
    self.profiler.wrap(self.agent, '_update_alpha', 'alpha_update')
    self.profiler.wrap(self.agent, '_update_actor', 'actor_update')
    self.profiler.wrap(self.agent, '_update_target', 'target_update')

    self._stop_signal: int | None = None
    self._start_wall = time.monotonic()
    self._last_checkpoint = self._start_wall
    self._next_eval = 0
    self._next_snapshot = 0
    self._eval_seconds = 0.0
    self._checkpoint_seconds = 0.0
    self._rate_mark = (time.perf_counter(), 0, 0)
    self._zero_accumulators()

  def _seed(self, name: str) -> int:
    """Separate streams per process. The networks start equal, _broadcast_parameters."""
    return zlib.crc32(f'{self.cfg.seed}/{self.rank}/{name}'.encode())

  def _generator(self, name: str) -> torch.Generator:
    return torch.Generator(device=self.device).manual_seed(self._seed(name))

  def _broadcast_parameters(self) -> None:
    """sac.py:647-656, by tensor. The temperature starts from the config on every rank."""
    agent = self.agent
    for module in (agent.actor, agent.q1, agent.q2, agent.q1_target, agent.q2_target):
      for parameter in module.parameters():
        dist.broadcast(parameter.data, 0)

  def _zero_accumulators(self) -> None:
    device = self.device
    self._update_sum = torch.zeros(len(Update._fields), device=device)
    self._actor_sum = torch.zeros(len(ActorUpdate._fields), device=device)
    self._updates = 0
    self._actor_updates = 0
    self._age_sum = torch.zeros((), dtype=torch.float64, device=device)
    self._stats_sum = torch.zeros(len(STAT_NAMES), dtype=torch.float64, device=device)
    self._act_sum = torch.zeros(3, device=device)  # |tanh(mean)|, its share past 0.99, std.
    self._acted_rounds = 0

  # Acting.

  def _act(self, obs: Obs) -> torch.Tensor:
    with self.profiler.phase('feature_build'):
      graph = build_graph(obs, self.topo)
    normals = torch.randn(self._action_shape, generator=self.act_rng, device=self.device)
    with self.profiler.phase('policy_forward'):
      acted = self.agent.act(graph, normals)
    self._act_sum += torch.stack([acted.mean.abs().mean(), (acted.mean.abs() > 0.99).float().mean(), acted.std.mean()])
    self._acted_rounds += 1
    return acted.action

  def _apply_stagger(self) -> None:
    due = (self._stagger_at == self._rounds_this_run).nonzero().flatten()
    if len(due):
      self.env.reset(due)
      self.buffer.discard(due)
      self.counters.discarded_rows += self._rounds_this_run * len(due)

  # Replay and learning.

  def _store(self, finished: EpisodeBatch) -> None:
    """Real first, then hindsight, the same module and rule as the original, DESIGN 16."""
    profiler = self.profiler
    self.counters.real_episodes += len(finished.length)
    with profiler.phase('replay_insert'):
      self.replay.insert(finished)
    if self.queue is None:
      return
    with profiler.phase('hindsight'):
      relabelled = self.queue.add(finished)
    if relabelled is not None:
      self.counters.hindsight_episodes += len(relabelled.length)
      with profiler.phase('replay_insert'):
        self.replay.insert(relabelled)

  def _can_learn(self) -> bool:
    """Enough rows, inserted rather than held, so that a ring smaller than resume_min_rows cannot wait forever.
    The fork's start_training defaults to 0, and its buffer holds a row per
    step, where whole episodes enter this ring when they end. All ranks start together."""
    if self._learning:
      return True
    ready = self.replay.inserted >= self._min_rows
    if self.world_size > 1:
      flag = torch.tensor(ready, dtype=torch.long, device=self.device)
      dist.all_reduce(flag, op=dist.ReduceOp.MIN)
      ready = bool(flag)
    self._learning = ready
    return ready

  def _learn(self) -> None:
    """off_policy_runner.py:104-105, sac.py:233-237."""
    profiler, cfg, agent = self.profiler, self.cfg, self.agent
    for _ in range(cfg.num_learning_epochs * cfg.num_mini_batches):
      with profiler.phase('sampling'):
        sample = self.replay.sample(cfg.batch_size, self.sample_rng, cfg.n_step, cfg.gamma)
      with profiler.phase('feature_rebuild'):
        discount = None if sample.steps is None else cfg.gamma ** sample.steps.float()
        batch = Batch(build_graph(sample.obs, self.topo), build_graph(sample.next_obs, self.topo),
                      sample.action, sample.reward, sample.terminal, discount)
      self._stats_sum += sample.stats.mean(0).double()
      self._age_sum += agent.grad_steps - sample.behaviour_step.double().mean()
      update, actor = agent.update(batch)
      self._update_sum += torch.stack(update)
      self._updates += 1
      if actor is not None:
        self._actor_sum += torch.stack(actor)
        self._actor_updates += 1

  # One round.

  def round(self) -> None:
    profiler = self.profiler
    profiler.begin_round(self.counters.rounds)
    if self._rounds_this_run < self._stagger_end:
      self._apply_stagger()
    obs = self.env.observe()
    action = self._act(obs)
    with profiler.phase('env_step'):
      out = self.env.step(action)
    with profiler.phase('episode_buffer'):
      finished = self.buffer.append(obs, action, out, behaviour_step=self.agent.grad_steps)
    self.counters.rounds += 1
    self._rounds_this_run += 1
    if finished is not None:
      self._store(finished)
      self._stagger_at = self._stagger_at.masked_fill(out.terminal, -1)  # Ended on its own: already out of step.
    if self.counters.rounds % self.cfg.num_steps_per_env == 0 and self._can_learn():
      self._learn()
      self._snapshot_if_due()
    if self.evaluator is not None and self.counters.rounds >= self._next_eval:
      self.evaluate()
    if self.counters.rounds % self.cfg.log_every_rounds == 0:
      self.report()

  def evaluate(self) -> dict[str, dict[str, float]]:
    """On the actor's mean, sac_mlp_model.py:130-131, by the evaluator's protocol."""
    start = time.perf_counter()
    step, rounds = self.agent.grad_steps, self.counters.rounds
    with self.profiler.phase('evaluation'):
      results = self.evaluator.evaluate(self.agent.mean_policy, step)
    env_steps = float(rounds * self.num_worlds * self.world_size)
    for name, metrics in results.items():
      self.logger.scalars({f'eval/{name}/{key}': value for key, value in {**metrics, 'env_steps': env_steps}.items()}, step)
      self.logger.notice(f'eval {name} round {rounds} step {step}: task_rate {metrics["task_rate"]:.4f}, '
                         f'length {metrics["episode_length"]:.1f}, '
                         f'collision steps {metrics["total_collision_robot_steps"]:.2f}')
    self._eval_seconds += time.perf_counter() - start
    self._next_eval = (rounds // self.cfg.eval_every_rounds + 1) * self.cfg.eval_every_rounds
    self.logger.flush()
    return results

  # Logging. Names of the original and of FastTD3 where the meaning is the same.

  def totals(self) -> dict[str, int]:
    c, agent, replay = self.counters, self.agent, self.replay
    return {
      'rounds': c.rounds,
      'env_steps': c.rounds * self.num_worlds * self.world_size,  # Effective, over all processes.
      'grad_steps': agent.grad_steps,
      'policy_steps': agent.policy_steps,
      'real_episodes': c.real_episodes,
      'hindsight_episodes': c.hindsight_episodes,
      'discarded_rows': c.discarded_rows,
      'inserted': c.inserted_before + replay.inserted,
      'sampled': c.sampled_before + replay.sampled,
      'replay_size': replay.size,
      'evicted': replay.evicted,
      'extra_evicted': replay.extra_evicted,
      'residual_collisions': c.residual_collisions_before + int(self.env.residual_collisions),
      'invalid_scene_draws': c.invalid_scene_draws_before + int(self.env.invalid_scene_draws),
    }

  def _check_finite(self, update: list[float], actor: list[float]) -> None:
    """At report time, since reading a loss back is a host sync. NaN in the weights stays, and an
    averaged gradient spreads it, but a rank must not stop alone, so all of them agree first."""
    nonfinite = not all(math.isfinite(x) for x in (*update, *actor))
    if self.world_size > 1:
      flag = torch.tensor(nonfinite, dtype=torch.long, device=self.device)
      dist.all_reduce(flag, op=dist.ReduceOp.MAX)
      nonfinite = bool(flag)
    if nonfinite:
      reason = f'nonfinite loss by round {self.counters.rounds}, grad step {self.agent.grad_steps}: update {update}, actor {actor}'
      self.save_checkpoint(reason, failed=True)
      self.logger.flush()
      raise FloatingPointError(reason)

  def report(self) -> None:
    with self.profiler.phase('logging'):
      step = self.agent.grad_steps
      update = (self._update_sum / max(self._updates, 1)).tolist()
      actor = (self._actor_sum / max(self._actor_updates, 1)).tolist()
      self._check_finite(update, actor)
      totals = self.totals()
      values = {f'counters/{key}': float(value) for key, value in totals.items()}
      now = time.perf_counter()
      mark_time, mark_rounds, mark_steps = self._rate_mark
      elapsed = max(now - mark_time, 1e-9)
      values['rate/env_steps_per_s'] = (totals['rounds'] - mark_rounds) * self.num_worlds * self.world_size / elapsed
      values['rate/grad_steps_per_s'] = (step - mark_steps) / elapsed
      if self.replay.inserted:
        values['rate/samples_per_insert'] = self.replay.sampled / self.replay.inserted
      self._rate_mark = (now, totals['rounds'], step)
      values['profile/evaluation_s'] = self._eval_seconds
      values['profile/checkpoint_s'] = self._checkpoint_seconds
      values.update(self.profiler.report(totals['rounds']))
      if self._acted_rounds:
        abs_mean, saturated, std = (self._act_sum / self._acted_rounds).tolist()
        values.update({'act/policy_action_abs': abs_mean, 'act/policy_action_saturated': saturated, 'act/policy_std': std})
      if self._updates:
        values.update({f'train/{name}': value for name, value in zip(UPDATE_TAGS, update)})
        values['train/trajectory_network_age'] = self._age_sum.item() / self._updates
        values['train/policy_lr'] = self.agent.actor_optimizer.param_groups[0]['lr']
        values['train/twin_critic_lr'] = self.agent.critic_optimizer.param_groups[0]['lr']
        values['train/alpha_lr'] = self.cfg.lr_alpha
        values['train/target_entropy'] = self.agent.target_entropy
        means = (self._stats_sum / self._updates).tolist()
        values.update({f'act/{name}': value for name, value in zip(STAT_NAMES, means)})
      if self._actor_updates:
        values['train/actor_loss'], values['train/actor_grad_norm'] = actor
      self._zero_accumulators()
      self.logger.scalars(values, step)
      self.logger.flush()

  # Checkpoints.

  def _rng_state(self) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
      'env': self.env.generator.get_state(),
      'act': self.act_rng.get_state(),
      'sample': self.sample_rng.get_state(),
      'hindsight': self.hindsight_rng.get_state(),
      'evaluator': self.evaluator.state_dict() if self.evaluator is not None else {},
      'torch_cpu': torch.get_rng_state(),
      'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
      'python': random.getstate(),
      'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
    }

  def _set_rng_state(self, state: dict[str, Any]) -> None:
    self.env.generator.set_state(state['env'].cpu())
    self.act_rng.set_state(state['act'].cpu())
    self.sample_rng.set_state(state['sample'].cpu())
    self.hindsight_rng.set_state(state['hindsight'].cpu())
    if self.evaluator is not None:
      self.evaluator.load_state_dict(state['evaluator'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state['torch_cuda']:
      torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])
    random.setstate(state['python'])
    kind, key, *rest = state['numpy']
    np.random.set_state((kind, key.numpy(), *rest))

  def checkpoint_path(self, failed: bool = False) -> Path:
    """Rank 0 writes the usual name, and the other ranks their own RNG beside it."""
    name = FAILED_CHECKPOINT_NAME if failed else CHECKPOINT_NAME
    return self.run_dir / (name if self.rank == 0 else f'rank{self.rank}_{name}')

  def _mean_policy_state(self) -> dict[str, torch.Tensor]:
    """Clones, or a file would hold the whole of the storage that the last layer's rows are a slice of."""
    return {name: value.detach().cpu().clone() for name, value in self.agent.mean_policy.state_dict().items()}

  def checkpoint_state(self, reason: str, failed: bool = False) -> dict[str, Any]:
    """Everything but the replay. `policy` and `step` fit agent.networks.load_policy, so scripts/eval.py takes it."""
    totals = self.totals()
    return {
      'format': CHECKPOINT_FORMAT,
      'failed': failed,
      'reason': reason,
      'step': self.agent.grad_steps,
      'policy': self._mean_policy_state(),
      'learner': self.agent.state_dict(),
      'counters': {
        'rounds': self.counters.rounds,
        'real_episodes': self.counters.real_episodes,
        'hindsight_episodes': self.counters.hindsight_episodes,
        'discarded_rows': self.counters.discarded_rows,
        'inserted': totals['inserted'],
        'sampled': totals['sampled'],
        'residual_collisions': totals['residual_collisions'],
        'invalid_scene_draws': totals['invalid_scene_draws'],
      },
      'next_eval': self._next_eval,
      'rng': self._rng_state(),
      'config': self._config(),
    }

  def _config(self) -> dict[str, Any]:
    return {'env': dataclasses.asdict(self.env_cfg), 'train': dataclasses.asdict(self.cfg),
            'model': dataclasses.asdict(self.model_cfg), 'world_size': self.world_size}

  def save_checkpoint(self, reason: str, failed: bool = False) -> Path:
    start = time.perf_counter()
    path = self.checkpoint_path(failed)
    with self.profiler.phase('checkpoint'):
      atomic_save(self.checkpoint_state(reason, failed), path)
    self._checkpoint_seconds += time.perf_counter() - start
    self._last_checkpoint = time.monotonic()
    self.logger.notice(f'checkpoint {path.name} at round {self.counters.rounds}, grad step {self.agent.grad_steps}: {reason}')
    return path

  def _snapshot_if_due(self) -> None:
    """The mean policy, kept for later evaluation and video. Unlike checkpoint.pt these stay, and resume nothing."""
    every, step = self.cfg.policy_snapshot_every_steps, self.agent.grad_steps
    if self.rank != 0 or every <= 0 or step < self._next_snapshot:
      return
    start = time.perf_counter()
    directory = self.run_dir / POLICY_DIR
    directory.mkdir(exist_ok=True)
    with self.profiler.phase('checkpoint'):
      atomic_save({'step': step, 'policy': self._mean_policy_state()}, directory / f'policy_{step:08d}.pt')
    self._checkpoint_seconds += time.perf_counter() - start
    self._next_snapshot = (step // every + 1) * every

  def _check_resumable(self, saved: dict[str, Any]) -> None:
    for section, values in self._config().items():
      if section == 'world_size':
        if saved[section] != values:
          raise ValueError(f'checkpoint has world_size = {saved[section]}, this run {values}')
        continue
      for key, value in values.items():
        if section == 'train' and key in RESUMABLE_CHANGES:
          continue
        # A field newer than the checkpoint: that run computed with its default.
        then = saved[section].get(key, getattr(SACCfg, key, None) if section == 'train' else None)
        if then != value:
          raise ValueError(f'checkpoint has {section}.{key} = {then}, this run {value}')

  def load_checkpoint(self, path: str | Path) -> None:
    """Restores the learner, every counter and RNG. The ring starts empty, so updates wait for resume_min_rows,
    as the original's for its min_size, train/trainer.py:238-245: FastTD3 lost its policy on a batch's worth of
    fresh rows. In-flight episodes are gone, so every world starts one and the stagger runs again."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != CHECKPOINT_FORMAT:
      raise ValueError(f'{path}: not a SAC checkpoint of format {CHECKPOINT_FORMAT}')
    if state['failed']:
      raise ValueError(f'{path} is a failed checkpoint: {state["reason"]}')
    self._check_resumable(state['config'])
    self.agent.load_state_dict(state['learner'])
    saved = state['counters']
    self.counters = Counters(
      rounds=saved['rounds'], real_episodes=saved['real_episodes'], hindsight_episodes=saved['hindsight_episodes'],
      discarded_rows=saved['discarded_rows'], inserted_before=saved['inserted'], sampled_before=saved['sampled'],
      residual_collisions_before=saved['residual_collisions'], invalid_scene_draws_before=saved['invalid_scene_draws'])
    self._next_eval = state['next_eval']
    every = self.cfg.policy_snapshot_every_steps
    self._next_snapshot = (self.agent.grad_steps // every + 1) * every if every > 0 else 0
    self._min_rows = max(self.cfg.batch_size, self.cfg.resume_min_rows)
    self._rate_mark = (time.perf_counter(), self.counters.rounds, self.agent.grad_steps)
    self._set_rng_state(state['rng'])
    self.logger.notice(f'resumed from {path} at round {self.counters.rounds}, grad step {self.agent.grad_steps}, '
                       f'updates wait for {self._min_rows} rows')

  # Running.

  def install_signal_handlers(self) -> None:
    """Runs go detached with setsid nohup, so SIGTERM is how they stop. Acted on between rounds."""
    def stop(signum, frame):
      self._stop_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

  def _poll(self) -> str | None:
    """Why to stop, if so, and the periodic checkpoint. Several ranks decide together, at report
    rounds only, because one rank stopping or saving alone would hang or stall the others."""
    status = None
    if self._stop_signal is not None:
      status = f'signal {self._stop_signal}'
    elif self.stop_after_s is not None and time.monotonic() - self._start_wall >= self.stop_after_s:
      status = 'stop_after_seconds'
    due = time.monotonic() - self._last_checkpoint >= self.cfg.checkpoint_every_s
    if self.world_size > 1:
      if self.counters.rounds % self.cfg.log_every_rounds:
        return None
      flags = torch.tensor([status is not None, due], dtype=torch.long, device=self.device)
      dist.all_reduce(flags, op=dist.ReduceOp.MAX)
      any_stop, due = flags.tolist()
      status = status or ('another rank stopped' if any_stop else None)
    if due and status is None:
      self.save_checkpoint('periodic')
    return status

  def run(self) -> str:
    """Trains until the round budget, the time limit or a signal. Returns why it stopped."""
    self._start_wall = time.monotonic()
    self._rate_mark = (time.perf_counter(), self.counters.rounds, self.agent.grad_steps)
    if self.evaluator is not None and self.counters.rounds >= self._next_eval:
      self.evaluate()
    self._snapshot_if_due()  # The untrained policy, on a new run.
    status = 'complete'
    while self.counters.rounds < self.cfg.total_env_rounds:
      self.round()
      stopped = self._poll()
      if stopped is not None:
        status = stopped
        break
    self.save_checkpoint(status)
    self.logger.flush()
    return status
