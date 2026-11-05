# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""The lockstep trainer of the original agent: one process, one GPU. DESIGN 14.2 and 14.3.

One round is one env step of every world. The 64 asynchronous actors of the original become
per-world tensors, updated at the events the original has per actor. All of it is on the GPU.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any
import zlib

import numpy as np
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.td3 import TD3, Batch
from roboballet_mjlab.config import EnvCfg, ModelCfg, TrainCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.replay.hindsight import HindsightQueue
from roboballet_mjlab.replay.replay import STAT_NAMES, EpisodeBatch, EpisodeBuffer, ReplayBuffer
from roboballet_mjlab.structs import Obs, Scene
from roboballet_mjlab.train.evaluator import Evaluator
from roboballet_mjlab.train.logger import MetricLogger
from roboballet_mjlab.train.profiler import Profiler

CHECKPOINT_FORMAT = 1
CHECKPOINT_NAME = 'checkpoint.pt'
FAILED_CHECKPOINT_NAME = 'checkpoint_failed.pt'
POLICY_DIR = 'policies'
# TrainCfg fields a resumed run may change: they do not affect what has been computed so far.
RESUMABLE_CHANGES = {'total_grad_steps', 'eval_every_steps', 'eval_episodes', 'checkpoint_every_s', 'policy_snapshot_every_steps',
                     'log_every_steps', 'profile'}


def atomic_save(payload: dict[str, Any], path: Path) -> None:
  """A reader sees the old file or the whole new one, also if this process is killed midway."""
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


class EpisodeLatches:
  """What each actor of agents/actor.py:584-654 holds, one element per world.

  `held_step` is the actor's self._network_step. Random check and sigma read it before this
  episode's fetch (actor.py:597-600, :612, :241-246), and the fetch comes afterwards, inside
  run_episode (actor.py:224, :566). So both use the step fetched at the previous episode. The
  value fetched now is the one recorded for this episode (actor.py:616).
  """

  def __init__(self, cfg: TrainCfg, td3: TD3, num_worlds: int, device) -> None:
    self.cfg, self.td3 = cfg, td3
    self.finished = torch.zeros(num_worlds, dtype=torch.long, device=device)  # episode_count.
    self.held_step = torch.zeros(num_worlds, dtype=torch.long, device=device)
    self.use_random = torch.ones(num_worlds, dtype=torch.bool, device=device)
    self.sigma = torch.zeros(num_worlds, device=device)

  def begin(self, ids: torch.Tensor, published_step: int) -> None:
    held = self.held_step[ids]
    self.use_random[ids] = (self.finished[ids] < self.cfg.random_episodes_per_world) | (held == 0)
    self.sigma[ids] = self.td3.exploration_sigma(held.double()).float()
    self.held_step[ids] = published_step

  def finish(self, ids: torch.Tensor) -> None:
    self.finished[ids] += 1


def behaviour_action(
  mean: torch.Tensor, sigma: torch.Tensor, normals: torch.Tensor, uniform: torch.Tensor, use_random: torch.Tensor
) -> torch.Tensor:
  """actor.py:232-239 and :159-163: noise per joint, clipped. This is also what gets stored."""
  noisy = (mean + sigma[:, None, None] * normals).clamp(-1.0, 1.0)
  return torch.where(use_random[:, None, None], uniform, noisy)


@dataclasses.dataclass
class Counters:
  rounds: int = 0
  real_episodes: int = 0
  hindsight_episodes: int = 0
  discarded_rows: int = 0  # Partial rows of the stagger. Never inserted, never counted as episodes.
  fused_calls: int = 0
  # From earlier parts of a resumed run. Replay counters restart with the empty ring.
  inserted_before: int = 0
  sampled_before: int = 0
  residual_collisions_before: int = 0
  invalid_scene_draws_before: int = 0


class Trainer:
  def __init__(
    self, env_cfg: EnvCfg, cfg: TrainCfg, model_cfg: ModelCfg, pool: Scene, device, run_dir: str | Path,
    evaluator: Evaluator | None = None, logger: MetricLogger | None = None, stop_after_s: float | None = None,
  ) -> None:
    # Everything but the learner call runs at highest, and fused_step restores it. DESIGN 14.1.
    torch.set_float32_matmul_precision('highest')
    self.env_cfg, self.cfg, self.model_cfg = env_cfg, cfg, model_cfg
    self.device = device
    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(parents=True, exist_ok=True)
    # What a killed save left behind. Otherwise a run directory holds one checkpoint, not a growing set.
    for stale in [*self.run_dir.glob(f'{CHECKPOINT_NAME}.tmp*'), *self.run_dir.glob(f'{POLICY_DIR}/*.tmp*')]:
      stale.unlink()
    self.logger = logger if logger is not None else MetricLogger(self.run_dir)
    self.evaluator = evaluator
    self.stop_after_s = stop_after_s
    W, self.num_worlds = cfg.num_worlds, cfg.num_worlds

    self.env = BalletEnv(env_cfg, W, pool, device, seed=self._seed('env'))
    self.topo = Topology(env_cfg, device)
    self.td3 = TD3(model_cfg, cfg, self.topo, device)
    self.td3.sync_targets()  # train.py:519-522.
    self.buffer = EpisodeBuffer(env_cfg, W, device)
    capacity = cfg.replay_capacity(env_cfg.num_targets)
    self.min_size = capacity // cfg.replay_min_divisor  # launch.py:145-149.
    self.replay = ReplayBuffer(env_cfg, capacity, device)
    self.megabatch = cfg.batch_size * cfg.fused_steps
    self.act_rng = self._generator('act')
    self.sample_rng = self._generator('sample')
    self.hindsight_rng = self._generator('hindsight')
    self.queue = HindsightQueue(env_cfg, cfg, self.hindsight_rng)

    self.counters = Counters()
    self.latches = EpisodeLatches(cfg, self.td3, W, device)
    self.published_step = 0  # The step of the parameters in the acting copy.
    self._finished_since_publish = 0
    self._all = torch.arange(W, device=device)
    self._action_shape = (W, env_cfg.world.num_robots, 7)
    self._rounds_this_run = 0
    # Each world resets once at its own offset, so episodes do not end together. World 0 runs as it is.
    offsets = (self._all * env_cfg.max_steps) // W
    self._stagger_at = offsets.masked_fill(offsets == 0, -1)
    self._stagger_end = int(offsets.max()) + 1

    self.profiler = Profiler(cfg.profile, device, launches=lambda: self.env.world.launches)
    self.profiler.wrap(self.env.world, 'forward', 'world_forward')
    self.profiler.wrap(self.env, '_move', 'rollback')
    self.profiler.wrap(self.env, 'reset', 'reset')

    self._stop_signal: int | None = None
    self._start_wall = time.monotonic()
    self._last_checkpoint = self._start_wall
    self._next_eval = 0
    self._next_snapshot = 0
    self._next_log = cfg.log_every_steps
    self._first_report_discarded = False
    self._last_fused_end = time.perf_counter()
    self._eval_seconds = 0.0
    self._checkpoint_seconds = 0.0
    self._rate_mark = (time.perf_counter(), 0, 0)
    self._zero_accumulators()
    self.latches.begin(self._all, self.published_step)

  def _seed(self, name: str) -> int:
    """Separate streams: a shared seed would make action noise and target noise the same draws."""
    return zlib.crc32(f'{self.cfg.seed}/{name}'.encode())

  def _generator(self, name: str) -> torch.Generator:
    return torch.Generator(device=self.device).manual_seed(self._seed(name))

  def _zero_accumulators(self) -> None:
    device = self.device
    self._fused_in_report = 0
    self._loss_sum = [0.0, 0.0]  # Policy, critic. Each one a sum over the fused steps of a call.
    self._age_sum = torch.zeros((), dtype=torch.float64, device=device)
    self._stats_sum = torch.zeros(len(STAT_NAMES), dtype=torch.float64, device=device)
    self._sampling_s = 0.0
    self._total_s = 0.0

  # Acting.

  def _draw_uniform(self) -> torch.Tensor:
    return torch.rand(self._action_shape, generator=self.act_rng, device=self.device) * 2.0 - 1.0

  def _draw_normals(self) -> torch.Tensor:
    return torch.randn(self._action_shape, generator=self.act_rng, device=self.device)

  def _act(self, obs: Obs) -> torch.Tensor:
    uniform = self._draw_uniform()
    latches = self.latches
    if latches.use_random.all():
      return uniform  # The original's random actors run no network, actor.py:232-239.
    with self.profiler.phase('feature_build'):
      graph = build_graph(obs, self.topo)
    with self.profiler.phase('policy_forward'):
      mean = self.td3.act_behaviour(graph)
    return behaviour_action(mean, latches.sigma, self._draw_normals(), uniform, latches.use_random)

  def _apply_stagger(self) -> None:
    due = (self._stagger_at == self._rounds_this_run).nonzero().flatten()
    if len(due):
      self.env.reset(due)
      self.buffer.discard(due)
      self.latches.begin(due, self.published_step)
      self.counters.discarded_rows += self._rounds_this_run * len(due)

  def publish(self) -> None:
    """DESIGN 14.3 item 4. A world that starts an episode now fetches this step and these weights."""
    self.td3.refresh_acting()
    self.published_step = self.td3.grad_steps
    self._finished_since_publish = 0

  # Replay and learning.

  def _store(self, finished: EpisodeBatch) -> None:
    """Real first, then hindsight, actor.py:619-646."""
    profiler = self.profiler
    self.counters.real_episodes += len(finished.length)
    with profiler.phase('replay_insert'):
      self.replay.insert(finished)
    with profiler.phase('hindsight'):
      relabelled = self.queue.add(finished)
    if relabelled is not None:
      self.counters.hindsight_episodes += len(relabelled.length)
      with profiler.phase('replay_insert'):
        self.replay.insert(relabelled)

  def budget_allows(self) -> bool:
    """The centre of the limiter, launch.py:145-149, full fused calls only.

    replay.sampled is grad_steps * batch_size within this part of a run. A resumed run starts
    with an empty ring and counts from zero, so it refills to min_size before sampling again.
    """
    allowed = self.cfg.samples_per_insert * (self.replay.inserted - self.min_size)
    return self.replay.sampled + self.megabatch <= allowed

  def _learn(self) -> None:
    profiler = self.profiler
    start = self._last_fused_end
    grad_steps = self.td3.grad_steps
    with profiler.phase('sampling'):
      sample = self.replay.sample(self.megabatch, self.sample_rng)
    with profiler.phase('feature_rebuild'):
      batch = Batch(build_graph(sample.obs, self.topo), build_graph(sample.next_obs, self.topo),
                    sample.action, sample.reward, sample.terminal)
    self._stats_sum += sample.stats.mean(0).double()  # train.py:764-767.
    self._age_sum += grad_steps - sample.behaviour_step.double().mean()  # train.py:862-866.
    sampled = time.perf_counter()
    with profiler.phase('td3_step'):
      losses = self.td3.fused_step(batch, summed=True)  # Summed as logged, train.py:398-404.
    policy, critic = torch.stack([losses.policy, losses.critic]).tolist()
    if not (math.isfinite(policy) and math.isfinite(critic)):
      reason = f'nonfinite loss at grad step {self.td3.grad_steps}: policy {policy}, critic {critic}'
      self.save_checkpoint(reason, failed=True)
      self.logger.flush()
      raise FloatingPointError(reason)
    self.counters.fused_calls += 1
    self._fused_in_report += 1
    self._loss_sum[0] += policy
    self._loss_sum[1] += critic
    now = time.perf_counter()
    self._sampling_s += sampled - start
    self._total_s += now - start
    self._last_fused_end = now

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
      finished = self.buffer.append(obs, action, out, behaviour_step=self.latches.held_step)
    self.counters.rounds += 1
    self._rounds_this_run += 1
    if finished is not None:
      self._store(finished)
      self._finished_since_publish += len(finished.length)
    while self.budget_allows():
      self._learn()
      self._after_learning()
    if self._finished_since_publish >= self.num_worlds:
      self.publish()
    if finished is not None:
      # After learning and any refresh, the last moment before this episode's first action.
      ended = out.terminal.nonzero().flatten()
      self.latches.finish(ended)
      self.latches.begin(ended, self.published_step)
      self._stagger_at[ended] = -1  # Ended on its own before its offset: already out of step.
    if time.monotonic() - self._last_checkpoint >= self.cfg.checkpoint_every_s:
      self.save_checkpoint('periodic')

  def _after_learning(self) -> None:
    """Inside the loop, because a hindsight flush permits many fused calls in one round."""
    if self.evaluator is not None and self.td3.grad_steps >= self._next_eval:
      self.evaluate()
    if self.td3.grad_steps >= self._next_log:
      self.report()
    self._snapshot_if_due()

  def evaluate(self) -> dict[str, dict[str, float]]:
    start = time.perf_counter()
    step = self.td3.grad_steps
    with self.profiler.phase('evaluation'):
      results = self.evaluator.evaluate(self.td3.policy, step)
    for name, metrics in results.items():
      self.logger.scalars({f'eval/{name}/{key}': value for key, value in metrics.items()}, step)
      self.logger.notice(f'eval {name} step {step}: task_rate {metrics["task_rate"]:.4f}, '
                         f'length {metrics["episode_length"]:.1f}, '
                         f'collision steps {metrics["total_collision_robot_steps"]:.2f}')
    self._eval_seconds += time.perf_counter() - start
    self._next_eval = (step // self.cfg.eval_every_steps + 1) * self.cfg.eval_every_steps
    self.logger.flush()
    return results

  # Logging, under the original's names.

  def totals(self) -> dict[str, int]:
    c = self.counters
    return {
      'rounds': c.rounds,
      'env_steps': c.rounds * self.num_worlds,
      'grad_steps': self.td3.grad_steps,
      'fused_calls': c.fused_calls,
      'real_episodes': c.real_episodes,
      'hindsight_episodes': c.hindsight_episodes,
      'discarded_rows': c.discarded_rows,
      'inserted': c.inserted_before + self.replay.inserted,
      'sampled': c.sampled_before + self.replay.sampled,
      'replay_size': self.replay.size,
      'evicted': self.replay.evicted,
      'extra_evicted': self.replay.extra_evicted,
      'residual_collisions': c.residual_collisions_before + int(self.env.residual_collisions),
      'invalid_scene_draws': c.invalid_scene_draws_before + int(self.env.invalid_scene_draws),
    }

  def report(self) -> None:
    with self.profiler.phase('logging'):
      step = self.td3.grad_steps
      self._next_log = (step // self.cfg.log_every_steps + 1) * self.cfg.log_every_steps
      totals = self.totals()
      values = {f'counters/{key}': float(value) for key, value in totals.items()}
      now = time.perf_counter()
      mark_time, mark_rounds, mark_steps = self._rate_mark
      elapsed = max(now - mark_time, 1e-9)
      values['rate/env_steps_per_s'] = (totals['rounds'] - mark_rounds) * self.num_worlds / elapsed
      values['rate/grad_steps_per_s'] = (step - mark_steps) / elapsed
      if self.replay.inserted > self.min_size:
        values['rate/samples_per_insert'] = self.replay.sampled / (self.replay.inserted - self.min_size)
      self._rate_mark = (now, totals['rounds'], step)
      values['profile/evaluation_s'] = self._eval_seconds
      values['profile/checkpoint_s'] = self._checkpoint_seconds
      values.update(self.profiler.report(totals['rounds']))
      n = self._fused_in_report
      # train.py:873-917: the first report goes unwritten, its numbers hold the compile.
      if n and self._first_report_discarded:
        per_step = 1e3 / n / self.cfg.fused_steps
        values.update({
          'train/policy_loss': self._loss_sum[0] / n,
          'train/twin_critic_loss': self._loss_sum[1] / n,
          'train/sampling_wall_time_ms': self._sampling_s * per_step,
          'train/total_step_time_ms': self._total_s * per_step,
          'train/trajectory_network_age': self._age_sum.item() / n,
          'train/policy_lr': self.td3.policy_lr(step),
          'train/twin_critic_lr': self.td3.critic_lr(step),
        })
        means = (self._stats_sum / n).tolist()
        values.update({f'act/{name}': value for name, value in zip(STAT_NAMES, means)})
      self._first_report_discarded = self._first_report_discarded or n > 0
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

  def checkpoint_state(self, reason: str, failed: bool = False) -> dict[str, Any]:
    """Everything but the replay, also env generator and counters. Fits networks.load_checkpoint."""
    learner = self.td3.state_dict()
    totals = self.totals()
    return {
      'format': CHECKPOINT_FORMAT,
      'failed': failed,
      'reason': reason,
      'step': self.td3.grad_steps,
      'policy': learner['policy'],
      'q1': learner['q1'],
      'q2': learner['q2'],
      'learner': learner,
      'counters': {
        'rounds': self.counters.rounds,
        'real_episodes': self.counters.real_episodes,
        'hindsight_episodes': self.counters.hindsight_episodes,
        'discarded_rows': self.counters.discarded_rows,
        'fused_calls': self.counters.fused_calls,
        'inserted': totals['inserted'],
        'sampled': totals['sampled'],
        'residual_collisions': totals['residual_collisions'],
        'invalid_scene_draws': totals['invalid_scene_draws'],
      },
      'published_step': self.published_step,
      'finished': self.latches.finished.cpu(),
      'held_step': self.latches.held_step.cpu(),
      'next_eval': self._next_eval,
      'rng': self._rng_state(),
      'config': {
        'env': dataclasses.asdict(self.env_cfg), 'train': dataclasses.asdict(self.cfg),
        'model': dataclasses.asdict(self.model_cfg),
      },
    }

  def save_checkpoint(self, reason: str, failed: bool = False) -> Path:
    start = time.perf_counter()
    path = self.run_dir / (FAILED_CHECKPOINT_NAME if failed else CHECKPOINT_NAME)
    with self.profiler.phase('checkpoint'):
      atomic_save(self.checkpoint_state(reason, failed), path)
    self._checkpoint_seconds += time.perf_counter() - start
    self._last_checkpoint = time.monotonic()
    self.logger.notice(f'checkpoint {path.name} at grad step {self.td3.grad_steps}: {reason}')
    return path

  def _snapshot_if_due(self) -> None:
    """Policies kept for later evaluation and video. Unlike checkpoint.pt these stay, and resume nothing."""
    every, step = self.cfg.policy_snapshot_every_steps, self.td3.grad_steps
    if every <= 0 or step < self._next_snapshot:
      return
    start = time.perf_counter()
    directory = self.run_dir / POLICY_DIR
    directory.mkdir(exist_ok=True)
    state = {name: value.cpu() for name, value in self.td3.policy.state_dict().items()}
    with self.profiler.phase('checkpoint'):
      atomic_save({'step': step, 'policy': state}, directory / f'policy_{step:08d}.pt')
    self._checkpoint_seconds += time.perf_counter() - start
    self._next_snapshot = (step // every + 1) * every

  def _check_resumable(self, saved: dict[str, Any]) -> None:
    current = {'env': dataclasses.asdict(self.env_cfg), 'train': dataclasses.asdict(self.cfg),
               'model': dataclasses.asdict(self.model_cfg)}
    for section, values in current.items():
      for key, value in values.items():
        if section == 'train' and key in RESUMABLE_CHANGES:
          continue
        if saved[section].get(key) != value:
          raise ValueError(f'checkpoint has {section}.{key} = {saved[section].get(key)}, this run {value}')

  def load_checkpoint(self, path: str | Path) -> None:
    """Restores the learner, the acting copy, every counter and RNG. The ring starts empty.

    In-flight episodes are gone, so every world starts one and the stagger runs again.
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != CHECKPOINT_FORMAT:
      raise ValueError(f'{path}: not a trainer checkpoint of format {CHECKPOINT_FORMAT}')
    if state['failed']:
      raise ValueError(f'{path} is a failed checkpoint: {state["reason"]}')
    self._check_resumable(state['config'])
    self.td3.load_state_dict(state['learner'])
    saved = state['counters']
    self.counters = Counters(
      rounds=saved['rounds'], real_episodes=saved['real_episodes'], hindsight_episodes=saved['hindsight_episodes'],
      discarded_rows=saved['discarded_rows'], fused_calls=saved['fused_calls'], inserted_before=saved['inserted'],
      sampled_before=saved['sampled'], residual_collisions_before=saved['residual_collisions'],
      invalid_scene_draws_before=saved['invalid_scene_draws'])
    self.published_step = state['published_step']
    self.latches.finished.copy_(state['finished'])
    self.latches.held_step.copy_(state['held_step'])
    self.latches.begin(self._all, self.published_step)
    self._next_eval = state['next_eval']
    every = self.cfg.policy_snapshot_every_steps
    self._next_snapshot = (self.td3.grad_steps // every + 1) * every if every > 0 else 0
    self._next_log = (self.td3.grad_steps // self.cfg.log_every_steps + 1) * self.cfg.log_every_steps
    self._rate_mark = (time.perf_counter(), self.counters.rounds, self.td3.grad_steps)
    self._set_rng_state(state['rng'])
    self.logger.notice(f'resumed from {path} at grad step {self.td3.grad_steps}, replay refills to {self.min_size}')

  # Running.

  def install_signal_handlers(self) -> None:
    """Runs go detached with setsid nohup, so SIGTERM is how they stop. Acted on between rounds."""
    def stop(signum, frame):
      self._stop_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

  def run(self) -> str:
    """Trains until the step budget, the time limit or a signal. Returns why it stopped."""
    self._start_wall = time.monotonic()
    self._last_fused_end = time.perf_counter()
    self._rate_mark = (time.perf_counter(), self.counters.rounds, self.td3.grad_steps)
    if self.evaluator is not None and self.td3.grad_steps >= self._next_eval:
      self.evaluate()
    self._snapshot_if_due()  # The untrained policy, on a new run.
    status = 'complete'
    while self.td3.grad_steps < self.cfg.total_grad_steps:
      self.round()
      if self._stop_signal is not None:
        status = f'signal {self._stop_signal}'
        break
      if self.stop_after_s is not None and time.monotonic() - self._start_wall >= self.stop_after_s:
        status = 'stop_after_seconds'
        break
    self.save_checkpoint(status)
    self.logger.flush()
    return status
