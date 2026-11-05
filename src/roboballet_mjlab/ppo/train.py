"""Trains rsl-rl's PPO on the benchmark's task, through its own OnPolicyRunner. DESIGN section 17, gate G14.

Detached, one GPU, from the clone:
  CUDA_VISIBLE_DEVICES=4 setsid nohup uv run python -m roboballet_mjlab.ppo.train \
    --run-dir runs/ppo_dev/smoke --train-pool data/scenes/4r_10t_30o_train.pt \
    --eval-heldout data/scenes/4r_10t_30o_heldout.pt \
    --eval-static-dist src/roboballet_mjlab/data/scenes/4r_10t_30o_static.pt \
    --eval-static-paper src/roboballet_mjlab/data/scenes/4r_10t_30o_static_paper.pt > runs/ppo_dev/smoke.log 2>&1 &

SIGTERM saves a checkpoint and exits with status 0. --resume continues from it. --eval-only CHECKPOINT
evaluates the policy's mean and prints a table. A checkpoint or a policies/ file also goes to scripts/eval.py.

The runner's learn(1) is called once per iteration with no log directory, so its loop and PPO run as
released, and what the other agents' trainers do between rounds happens here between iterations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any
import zlib

import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import PolicyNet
from roboballet_mjlab.config import EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.ppo.config import PPOCfg
from roboballet_mjlab.ppo.vecenv import BalletVecEnv
from roboballet_mjlab.structs import Scene
from roboballet_mjlab.train.evaluator import Evaluator, load_scenes
from roboballet_mjlab.train.logger import MetricLogger, write_provenance
from roboballet_mjlab.train.profiler import Profiler

CHECKPOINT_FORMAT = 'ppo-1'
CHECKPOINT_NAME = 'checkpoint.pt'
FAILED_CHECKPOINT_NAME = 'checkpoint_failed.pt'
POLICY_DIR = 'policies'
MESH_TOOL_MIN_NCONMAX = 256
# PPOCfg fields a resumed run may change: they do not affect what has been computed so far.
RESUMABLE_CHANGES = {'total_env_rounds', 'eval_every_rounds', 'eval_episodes', 'checkpoint_every_s', 'log_every_rounds',
                     'policy_snapshot_every_rounds', 'profile', 'profile_every_rounds', 'profile_window_rounds'}


def atomic_save(payload: dict[str, Any], path: Path) -> None:
  """As train/trainer.py, which this package does not import: a reader sees the old file or the whole new one."""
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


@dataclasses.dataclass
class Counters:
  """From earlier parts of a resumed run, since the adapter's own start from zero with its new env."""

  iterations: int = 0
  rounds_before: int = 0
  episodes_before: int = 0
  discarded_rows_before: int = 0
  residual_collisions_before: int = 0
  invalid_scene_draws_before: int = 0


class Trainer:
  def __init__(
    self, env_cfg: EnvCfg, cfg: PPOCfg, model_cfg: ModelCfg, pool: Scene, device, run_dir: str | Path,
    evaluator: Evaluator | None = None, logger: MetricLogger | None = None, stop_after_s: float | None = None,
  ) -> None:
    # Everything but the learner call runs at highest, and BalletPPO.update restores it. DESIGN 14.1.
    torch.set_float32_matmul_precision('highest')
    self.env_cfg, self.cfg, self.model_cfg = env_cfg, cfg, model_cfg
    self.device = device
    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.logger = logger if logger is not None else MetricLogger(self.run_dir)
    self.evaluator = evaluator
    self.stop_after_s = stop_after_s

    self.vec = BalletVecEnv(env_cfg, cfg.num_worlds, pool, device, seed=self._seed('env'))
    self.runner = OnPolicyRunner(self.vec, cfg.runner_dict(env_cfg, model_cfg), log_dir=None, device=device)
    # rsl-rl draws its actions and mini-batches from the global generators. After the models, because
    # torch.manual_seed in their fork_rng(devices=[]) reaches the GPU's generator and stays there.
    torch.manual_seed(self._seed('torch'))
    self.alg = self.runner.alg
    self.actor, self.critic = self.alg.actor, self.alg.critic

    self.counters = Counters()
    self.profiler = Profiler(cfg.profile, device, launches=lambda: self.vec.env.world.launches, every_rounds=cfg.profile_every_rounds,
                             window_rounds=cfg.profile_window_rounds)
    self._acts_this_run = 0
    self._instrument()

    self._stop_signal: int | None = None
    self._start_wall = time.monotonic()
    self._last_checkpoint = self._start_wall
    self._next_eval = 0
    self._next_report = 0
    self._next_snapshot = 0
    self._eval_seconds = 0.0
    self._checkpoint_seconds = 0.0
    self._rate_mark = (time.perf_counter(), self.vec.rounds, 0)
    self._zero_accumulators()

  def _seed(self, name: str) -> int:
    return zlib.crc32(f'{self.cfg.seed}/{name}'.encode())

  def _instrument(self) -> None:
    """Phases for the runner's loop, wrapped from outside, as Profiler.wrap does to the env. The models
    run under both acting and learning, so those phases get a name by which one it is."""
    profiler = self.profiler
    env = self.vec.env
    profiler.wrap(env.world, 'forward', 'world_forward')
    profiler.wrap(env, '_move', 'rollback')
    profiler.wrap(env, 'reset', 'reset')
    profiler.wrap(self.vec, 'step', 'env_step')
    profiler.wrap(self.alg, 'process_env_step', 'storage')
    profiler.wrap(self.alg, 'compute_returns', 'returns')
    profiler.wrap(self.alg, 'update', 'learner')

    def by_mode(owner: Any, method: str, name: str) -> None:
      original = getattr(owner, method)

      def timed(*args, **kwargs):
        with profiler.phase(f'learner_{name}' if torch.is_grad_enabled() else name):
          return original(*args, **kwargs)

      setattr(owner, method, timed)

    by_mode(self.actor, 'graph', 'feature_build')
    by_mode(self.critic, 'graph', 'feature_build')
    by_mode(self.actor, 'run_net', 'policy_forward')
    by_mode(self.critic, 'run_net', 'value_forward')

    act = self.alg.act
    update = self.alg.update

    def timed_act(obs):
      profiler.begin_round(self._acts_this_run)
      self._acts_this_run += 1
      with profiler.phase('act_other'):
        actions = act(obs)
      mean = self.actor.output_mean
      self._action_sum += torch.stack([mean.abs().mean(), (mean.abs() > 0.99).float().mean(), (actions.abs() > 1.0).float().mean()])
      self._acted_rounds += 1
      return actions

    def timed_update():
      start = time.perf_counter()
      losses = update()
      self._learn_seconds += time.perf_counter() - start
      return losses

    self.alg.act = timed_act
    self.alg.update = timed_update

  def _zero_accumulators(self) -> None:
    self._loss_sum = {'value': 0.0, 'surrogate': 0.0, 'entropy': 0.0}
    self._returns_sum = torch.zeros(4, device=self.device)
    self._health_sum = torch.zeros(3, device=self.device)
    self._action_sum = torch.zeros(3, device=self.device)
    self._acted_rounds = 0
    self._iterations_since_report = 0
    self._iteration_seconds = 0.0
    self._learn_seconds = 0.0
    self.actor.kl_sum.zero_()
    self.actor.kl_count = 0

  # One iteration.

  def iteration(self) -> None:
    """24 steps of every world, the returns, 20 updates: OnPolicyRunner.learn, on_policy_runner.py:83-108."""
    start = time.perf_counter()
    self.runner.learn(1)
    self._iteration_seconds += time.perf_counter() - start
    self.runner.current_learning_iteration += 1  # learn() leaves it at the iteration it ran.
    self.counters.iterations += 1
    self._iterations_since_report += 1
    self._check_finite()
    storage = self.alg.storage
    self._returns_sum += torch.stack([storage.returns.mean(), storage.returns.min(), storage.returns.max(), storage.values.mean()])
    self._health_sum += self.rollout_health()
    for name, value in self.alg.last_losses.items():
      self._loss_sum[name] += value
    rounds = self.rounds()
    self._snapshot_if_due()
    if self.evaluator is not None and rounds >= self._next_eval:
      self.evaluate()
    if rounds >= self._next_report:
      self.report()

  def rounds(self) -> int:
    return self.counters.rounds_before + self.vec.rounds

  def grad_steps(self) -> int:
    return self.counters.iterations * self.cfg.num_learning_epochs * self.cfg.num_mini_batches

  def mini_batch_size(self) -> int:
    """storage.mini_batch_generator's, rollout_storage.py:229-230."""
    return self.cfg.num_worlds * self.cfg.num_steps_per_env // self.cfg.num_mini_batches

  def _check_finite(self) -> None:
    """Every iteration, before the next action: the losses, already host numbers, and one read for every
    weight and everything the rollout stored, which is what rsl-rl's check_nan looks at per round."""
    losses, storage = self.alg.last_losses, self.alg.storage
    tensors = [*self.actor.parameters(), *self.critic.parameters(), storage.rewards.float(), storage.returns, storage.actions,
               *[field.float() for field in storage.observations.values()]]
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors))).item()
    if not (math.isfinite(norm) and all(math.isfinite(x) for x in losses.values())):
      reason = f'nonfinite at iteration {self.counters.iterations}, grad step {self.grad_steps()}: losses {losses}, norm of the weights and the rollout {norm}'
      self.save_checkpoint(reason, failed=True)
      self.logger.flush()
      raise FloatingPointError(reason)

  def rollout_health(self) -> torch.Tensor:
    """Whether the networks still tell states apart, from the rollout just stored: the critic's explained variance,
    then the std over its rows of the mean, averaged over the joints, and of the value. The losses and the task rate
    do not show a network that collapsed to a constant, docs/reviews/ppo_status.md. A rollout with one return explains nothing."""
    storage = self.alg.storage
    variance = storage.returns.var()
    explained = torch.where(variance > 0, 1.0 - (storage.returns - storage.values).var() / variance, torch.zeros_like(variance))
    return torch.stack([explained, storage.distribution_params[0].flatten(0, 1).std(0).mean(), storage.values.std()])

  def evaluate(self) -> dict[str, dict[str, float]]:
    start = time.perf_counter()
    step, rounds = self.grad_steps(), self.rounds()
    with self.profiler.phase('evaluation'):
      results = self.evaluator.evaluate(self.actor.net, step)
    env_steps = float(rounds * self.cfg.num_worlds)
    for name, metrics in results.items():
      self.logger.scalars({f'eval/{name}/{key}': value for key, value in {**metrics, 'env_steps': env_steps}.items()}, step)
      self.logger.notice(f'eval {name} round {rounds} step {step}: task_rate {metrics["task_rate"]:.4f}, '
                         f'length {metrics["episode_length"]:.1f}, '
                         f'collision steps {metrics["total_collision_robot_steps"]:.2f}')
    self._eval_seconds += time.perf_counter() - start
    self._next_eval = (rounds // self.cfg.eval_every_rounds + 1) * self.cfg.eval_every_rounds
    self.logger.flush()
    return results

  # Logging. Names of the other agents where the meaning is the same, rsl-rl's otherwise.

  def totals(self) -> dict[str, int]:
    c, env = self.counters, self.vec.env
    return {
      'iterations': c.iterations,
      'rounds': self.rounds(),
      'env_steps': self.rounds() * self.cfg.num_worlds,
      'grad_steps': self.grad_steps(),
      'real_episodes': c.episodes_before + self.vec.episodes,
      'discarded_rows': c.discarded_rows_before + self.vec.discarded_rows,
      'residual_collisions': c.residual_collisions_before + int(env.residual_collisions),
      'invalid_scene_draws': c.invalid_scene_draws_before + int(env.invalid_scene_draws),
    }

  def report(self) -> None:
    with self.profiler.phase('logging'):
      step = self.grad_steps()
      totals = self.totals()
      values = {f'counters/{key}': float(value) for key, value in totals.items()}
      now = time.perf_counter()
      mark_time, mark_rounds, mark_steps = self._rate_mark
      elapsed = max(now - mark_time, 1e-9)
      values['rate/env_steps_per_s'] = (totals['rounds'] - mark_rounds) * self.cfg.num_worlds / elapsed
      values['rate/grad_steps_per_s'] = (step - mark_steps) / elapsed
      values['rate/samples_per_s'] = (step - mark_steps) * self.mini_batch_size() / elapsed
      self._rate_mark = (now, totals['rounds'], step)
      values['profile/evaluation_s'] = self._eval_seconds
      values['profile/checkpoint_s'] = self._checkpoint_seconds
      values.update(self.profiler.report(totals['rounds']))
      if self._acted_rounds:
        means = (self._action_sum / self._acted_rounds).tolist()
        values['act/policy_action_abs'], values['act/policy_action_saturated'], values['act/action_clipped'] = means
      values.update({f'act/{name}': value for name, value in self.vec.take_episode_stats().items()})
      n = self._iterations_since_report
      if n:
        values['profile/iteration_s'] = self._iteration_seconds / n
        values['profile/learn_s'] = self._learn_seconds / n
        values['profile/collect_s'] = (self._iteration_seconds - self._learn_seconds) / n
        values['train/value_loss'] = self._loss_sum['value'] / n
        values['train/surrogate_loss'] = self._loss_sum['surrogate'] / n
        values['train/entropy'] = self._loss_sum['entropy'] / n
        values['train/returns_mean'], values['train/returns_min'], values['train/returns_max'], values['train/value_mean'] = (self._returns_sum / n).tolist()
        values['train/explained_variance'], values['train/mean_std_over_states'], values['train/value_std_over_states'] = (self._health_sum / n).tolist()
        # One optimizer holds both networks, ppo.py:92-94, so the two tags are one rate.
        values['train/policy_lr'] = values['train/critic_lr'] = self.alg.learning_rate
        values['train/action_std'] = self.actor.output_std.mean().item()
        if self.actor.kl_count:
          values['train/kl'] = self.actor.kl_sum.item() / self.actor.kl_count
      self._zero_accumulators()
      self.logger.scalars(values, step)
      self.logger.flush()
      self._next_report = (totals['rounds'] // self.cfg.log_every_rounds + 1) * self.cfg.log_every_rounds

  # Checkpoints.

  def _rng_state(self) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
      'env': self.vec.env.generator.get_state(),
      'evaluator': self.evaluator.state_dict() if self.evaluator is not None else {},
      'torch_cpu': torch.get_rng_state(),
      'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
      'python': random.getstate(),
      'numpy': (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
    }

  def _set_rng_state(self, state: dict[str, Any]) -> None:
    self.vec.env.generator.set_state(state['env'].cpu())
    if self.evaluator is not None:
      self.evaluator.load_state_dict(state['evaluator'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state['torch_cuda']:
      torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])
    random.setstate(state['python'])
    kind, key, *rest = state['numpy']
    np.random.set_state((kind, key.numpy(), *rest))

  def checkpoint_path(self, failed: bool = False) -> Path:
    return self.run_dir / (FAILED_CHECKPOINT_NAME if failed else CHECKPOINT_NAME)

  def _policy_state(self) -> dict[str, torch.Tensor]:
    return {name: value.cpu() for name, value in self.actor.net.state_dict().items()}

  def checkpoint_state(self, reason: str, failed: bool = False) -> dict[str, Any]:
    """rsl-rl's own save(): both models with the std, and the optimizer, whose lr is the adapted one.
    `policy` is the mean alone, under the key agent.networks.load_policy reads. In-flight rows and worlds are not kept."""
    totals = self.totals()
    return {
      'format': CHECKPOINT_FORMAT,
      'failed': failed,
      'reason': reason,
      'step': self.grad_steps(),
      'policy': self._policy_state(),
      'learner': self.alg.save(),
      'counters': {key: totals[key] for key in ('iterations', 'rounds', 'real_episodes', 'discarded_rows', 'residual_collisions', 'invalid_scene_draws')},
      'next_eval': self._next_eval,
      'next_report': self._next_report,
      'next_snapshot': self._next_snapshot,
      'rng': self._rng_state(),
      'config': self._config(),
    }

  def _config(self) -> dict[str, Any]:
    return {'env': dataclasses.asdict(self.env_cfg), 'train': dataclasses.asdict(self.cfg), 'model': dataclasses.asdict(self.model_cfg)}

  def save_checkpoint(self, reason: str, failed: bool = False) -> Path:
    start = time.perf_counter()
    path = self.checkpoint_path(failed)
    with self.profiler.phase('checkpoint'):
      atomic_save(self.checkpoint_state(reason, failed), path)
    self._checkpoint_seconds += time.perf_counter() - start
    self._last_checkpoint = time.monotonic()
    self.logger.notice(f'checkpoint {path.name} at round {self.rounds()}, grad step {self.grad_steps()}: {reason}')
    return path

  def _snapshot_if_due(self) -> None:
    """Policies kept for later evaluation and video, as train/trainer.py, with the period in rounds,
    since PPO's grad steps are not comparable. Unlike checkpoint.pt these stay, and resume nothing."""
    every, rounds = self.cfg.policy_snapshot_every_rounds, self.rounds()
    if every <= 0 or rounds < self._next_snapshot:
      return
    start = time.perf_counter()
    directory = self.run_dir / POLICY_DIR
    directory.mkdir(exist_ok=True)
    step = self.grad_steps()
    with self.profiler.phase('checkpoint'):
      atomic_save({'step': step, 'policy': self._policy_state()}, directory / f'policy_{step:08d}.pt')
    self._checkpoint_seconds += time.perf_counter() - start
    self._next_snapshot = (rounds // every + 1) * every

  def _check_resumable(self, saved: dict[str, Any]) -> None:
    for section, values in self._config().items():
      for key, value in values.items():
        if section == 'train' and key in RESUMABLE_CHANGES:
          continue
        if saved[section].get(key) != value:
          raise ValueError(f'checkpoint has {section}.{key} = {saved[section].get(key)}, this run {value}')

  def load_checkpoint(self, path: str | Path) -> None:
    """Restores the learner, every counter and RNG. Rows and worlds in flight are gone, so every world
    starts an episode and the stagger has run again, in this object's constructor. A rollout holds no
    state that outlives its iteration, so nothing else is lost."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != CHECKPOINT_FORMAT:
      raise ValueError(f'{path}: not a PPO checkpoint of format {CHECKPOINT_FORMAT}')
    if state['failed']:
      raise ValueError(f'{path} is a failed checkpoint: {state["reason"]}')
    self._check_resumable(state['config'])
    self.alg.load(state['learner'], None, strict=True)
    saved = state['counters']
    self.counters = Counters(
      iterations=saved['iterations'], rounds_before=saved['rounds'], episodes_before=saved['real_episodes'],
      discarded_rows_before=saved['discarded_rows'], residual_collisions_before=saved['residual_collisions'],
      invalid_scene_draws_before=saved['invalid_scene_draws'])
    self.runner.current_learning_iteration = saved['iterations']
    self._next_eval, self._next_report, self._next_snapshot = state['next_eval'], state['next_report'], state['next_snapshot']
    self._rate_mark = (time.perf_counter(), self.rounds(), self.grad_steps())
    self._set_rng_state(state['rng'])
    self.logger.notice(f'resumed from {path} at iteration {saved["iterations"]}, round {self.rounds()}, '
                       f'grad step {self.grad_steps()}, learning rate {self.alg.learning_rate}')

  # Running.

  def install_signal_handlers(self) -> None:
    """Runs go detached with setsid nohup, so SIGTERM is how they stop. Acted on between iterations."""
    def stop(signum, frame):
      self._stop_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

  def _poll(self) -> str | None:
    """Why to stop, if so, and the periodic checkpoint."""
    if self._stop_signal is not None:
      return f'signal {self._stop_signal}'
    if self.stop_after_s is not None and time.monotonic() - self._start_wall >= self.stop_after_s:
      return 'stop_after_seconds'
    if time.monotonic() - self._last_checkpoint >= self.cfg.checkpoint_every_s:
      self.save_checkpoint('periodic')
    return None

  def run(self) -> str:
    """Trains until the round budget, the time limit or a signal. Returns why it stopped."""
    self._start_wall = time.monotonic()
    self._rate_mark = (time.perf_counter(), self.rounds(), self.grad_steps())
    self._snapshot_if_due()  # The untrained policy, on a new run.
    if self.evaluator is not None and self.rounds() >= self._next_eval:
      self.evaluate()
    status = 'complete'
    while self.rounds() < self.cfg.total_env_rounds:
      self.iteration()
      stopped = self._poll()
      if stopped is not None:
        status = stopped
        break
    self.save_checkpoint(status)
    self.logger.flush()
    return status


def overrides(pairs: list[str], config):
  """KEY=VALUE for any field of a config dataclass. A tuple takes comma-separated integers."""
  kinds = {field.name: type(getattr(config, field.name)) for field in dataclasses.fields(config)}
  changes = {}
  for pair in pairs:
    key, value = pair.split('=', 1)
    if key not in kinds:
      raise ValueError(f'{type(config).__name__} has no field {key}')
    kind = kinds[key]
    if kind is bool:
      changes[key] = value.lower() in ('1', 'true', 'yes')
    elif kind is tuple:
      changes[key] = tuple(int(x) for x in value.split(','))
    else:
      changes[key] = kind(value)
  return dataclasses.replace(config, **changes)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=PPOCfg.seed)
  parser.add_argument('--total-env-rounds', type=int, default=PPOCfg.total_env_rounds)
  parser.add_argument('--arm-model', default=WorldCfg.arm_model, choices=('spheres_stick', 'mesh_tool'))
  pool = parser.add_mutually_exclusive_group(required=True)
  pool.add_argument('--train-pool', type=Path, help='A .pt pool from gen_scenes.py.')
  pool.add_argument('--start-states', type=Path, help='A .pb of original start states.')
  parser.add_argument('--eval-training-dist', type=Path, help='Default: the training pool, as the original.')
  parser.add_argument('--eval-heldout', type=Path)
  parser.add_argument('--eval-static-dist', type=Path)
  parser.add_argument('--eval-static-paper', type=Path)
  parser.add_argument('--no-eval', action='store_true', help='No evaluator at all, for measuring the training rate.')
  parser.add_argument('--eval-only', type=Path, metavar='CHECKPOINT', help='Evaluate this checkpoint, print a table and exit.')
  parser.add_argument('--resume', nargs='?', const='', default=None, metavar='CHECKPOINT',
                      help=f'Continue from a checkpoint, by default {CHECKPOINT_NAME} in the run directory.')
  parser.add_argument('--wandb-project', help='Also log to this W&B project. Credentials are wandb\'s own business.')
  parser.add_argument('--stop-after-seconds', type=float)
  parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override a PPOCfg field.')
  parser.add_argument('--model-set', action='append', default=[], metavar='KEY=VALUE', help='Override a ModelCfg field.')
  return parser.parse_args()


def evaluate_only(args, env_cfg: EnvCfg, model_cfg: ModelCfg, cfg: PPOCfg, sets: dict, device: str) -> None:
  torch.set_float32_matmul_precision('highest')
  policy = PolicyNet(model_cfg, Topology(env_cfg, 'cpu'))
  state = torch.load(args.eval_only, map_location='cpu', weights_only=True)
  policy.load_state_dict(state['policy'], strict=True)
  evaluator = Evaluator(env_cfg, model_cfg, sets, device, cfg.seed, num_worlds=cfg.eval_episodes, compile=cfg.compile)
  results = evaluator.evaluate(policy.to(device), state['step'])
  print(json.dumps({'checkpoint': str(args.eval_only), 'step': state['step'], 'results': results}, indent=2))


def main() -> int:
  args = parse_args()
  device = 'cuda:0'
  source = args.train_pool or args.start_states
  scenes = load_scenes(source, device)
  world = WorldCfg(num_robots=scenes.base_pos.shape[1], num_obstacles=scenes.box_pos.shape[1], arm_model=args.arm_model)
  if args.arm_model == 'mesh_tool' and world.nconmax < MESH_TOOL_MIN_NCONMAX:
    raise ValueError(f'mesh_tool needs nconmax of at least {MESH_TOOL_MIN_NCONMAX}, have {world.nconmax}')
  env_cfg = EnvCfg(world=world, num_targets=scenes.target_pos.shape[1])
  model_cfg = overrides(args.model_set, ModelCfg())
  cfg = overrides(args.set, PPOCfg(seed=args.seed, total_env_rounds=args.total_env_rounds))
  sets = {'training_dist': args.eval_training_dist or scenes, 'heldout': args.eval_heldout,
          'static_dist': args.eval_static_dist, 'static_paper': args.eval_static_paper}
  if args.eval_only is not None:
    evaluate_only(args, env_cfg, model_cfg, cfg, sets, device)
    return 0

  config = {'env': env_cfg, 'train': cfg, 'model': model_cfg, 'source': str(source), 'args': vars(args)}
  args.run_dir.mkdir(parents=True, exist_ok=True)
  write_provenance(args.run_dir, config, device)
  logger = MetricLogger(args.run_dir, args.wandb_project, config={k: dataclasses.asdict(config[k]) for k in ('env', 'train', 'model')})
  mini_batch = cfg.num_worlds * cfg.num_steps_per_env // cfg.num_mini_batches
  logger.notice(f'{len(scenes.q0)} scenes from {source}, {cfg.num_worlds} worlds, {cfg.num_steps_per_env} steps per iteration, '
                f'mini-batch {mini_batch}, learner bf16 {cfg.bf16} matmul {cfg.matmul_precision}, arm {args.arm_model}')
  evaluator = None
  if not args.no_eval:
    evaluator = Evaluator(env_cfg, model_cfg, sets, device, cfg.seed, num_worlds=cfg.eval_episodes, compile=cfg.compile, notice=logger.notice)
  trainer = Trainer(env_cfg, cfg, model_cfg, scenes, device, args.run_dir, evaluator, logger, args.stop_after_seconds)
  trainer.install_signal_handlers()
  if args.resume is not None:
    trainer.load_checkpoint(args.resume or trainer.checkpoint_path())
  try:
    status = trainer.run()
  finally:
    (args.run_dir / 'summary.json').write_text(json.dumps(trainer.totals(), indent=2))
    trainer.logger.close()
  print(f'stopped: {status}. {trainer.totals()}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
