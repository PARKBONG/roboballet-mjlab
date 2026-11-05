# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Hindsight relabelling as a batched pure function, agents/actor.py:298-402, algo_spec section 5.

The original builds a new env with the moved targets and replays the stored motion through
env.step: q and qd come from the stored next observation, so nothing moves or collides
(base_env.py:293-298) and the collision set stays empty, while dwell, done, score and reward
run the normal code. The action array it stores is the one it passed to env.step, and that
step writes into it (actor.py:382-386, planning_env.py:793-796, :814).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from roboballet_mjlab.config import HOME_QPOS, EnvCfg, TrainCfg
from roboballet_mjlab.env.ballet_env import solve_targets, targets_within
from roboballet_mjlab.replay.replay import STAT, EpisodeBatch, cat_episodes, select_episodes
from roboballet_mjlab.structs import RobotState


class Picks(NamedTuple):
  """The random choices of actor.py:345-356, per target. arm and time count only where chosen."""

  chosen: torch.Tensor  # [n, T], bool
  arm: torch.Tensor  # [n, T], long
  time: torch.Tensor  # [n, T], long, a row index.


def num_to_solve(env_cfg: EnvCfg, train_cfg: TrainCfg) -> int:
  return int(train_cfg.hindsight_fraction * env_cfg.num_targets)


def eligible(batch: EpisodeBatch, k: int) -> torch.Tensor:
  """[n], bool. solved >= k gets no episode, counted from the terminal row, actor.py:339-342.

  A single-action episode below k has no valid time to pick, and the original raises in
  random.randint(1, 0). It needs max_steps == 1, so here it just gets no episode.
  """
  final = batch.robot.target_done[torch.arange(len(batch.length), device=batch.length.device), batch.length]
  return (final.sum(1) < k) & (batch.length >= 2)


def draw_picks(
  final_done: torch.Tensor, length: torch.Tensor, num_robots: int, k: int, generator: torch.Generator
) -> Picks:
  """k - solved unsolved targets without replacement, then per target an arm and a time in [1, A-1]."""
  shape = final_done.shape
  device = final_done.device
  keys = torch.rand(shape, generator=generator, device=device).masked_fill(final_done, 2.0)
  chosen = keys.argsort(1).argsort(1) < (k - final_done.sum(1, keepdim=True))
  arm = torch.randint(num_robots, shape, generator=generator, device=device)
  # A large integer modulo keeps the range exact, which scaling a float draw does not.
  time = torch.randint(2**31 - 1, shape, generator=generator, device=device) % (length[:, None] - 1) + 1
  return Picks(chosen, arm, time)


def score_of(done: torch.Tensor, q: torch.Tensor, cfg: EnvCfg, home: torch.Tensor) -> torch.Tensor:
  """BalletEnv._score, tests/test_hindsight.py checks that they agree."""
  weight = cfg.return_to_start_reward_weight
  score = (1.0 - weight) * done.float().mean(1)
  if weight == 0.0:
    return score
  diff = (q - home).abs()
  within = (diff < cfg.return_to_start_threshold).all(-1)
  shaped = 1.0 - (cfg.return_to_start_coeff * diff.square().mean(-1)).clamp(0.0, 1.0)
  return score + weight * torch.where(within, 1.0, shaped).mean(1) * done.all(1)


def mutated_action(
  action: torch.Tensor, dwelling: torch.Tensor, done: torch.Tensor, q: torch.Tensor, cfg: EnvCfg, home: torch.Tensor
) -> torch.Tensor:
  """What env.step leaves in the stored action, planning_env.py:793-796 then :801-816.

  `done` is from before this step. Takeover overrides dwell zeroing. It runs whatever the
  return-to-start flags say (DESIGN 11.1), and BalletEnv.step does the same to its request,
  which tests/test_hindsight.py checks. The units are the stored ones: the original writes
  rad/s and the writer divides by max_joint_velocity (actor.py:450, :476), so a takeover
  action can reach 0.25 / 0.1 / 2.175 = 1.15, beyond the action bounds.
  """
  action = action.masked_fill(dwelling[..., None], 0.0)
  all_done = done.all(1, keepdim=True)
  if not all_done.any():
    return action
  to_home = home - q
  takeover = all_done & (to_home.abs() < cfg.return_to_start_threshold).all(-1)
  return torch.where(takeover[..., None], to_home / cfg.timestep_s / cfg.max_joint_velocity, action)


def relabel(
  batch: EpisodeBatch, env_cfg: EnvCfg, train_cfg: TrainCfg, generator: torch.Generator, picks: Picks | None = None
) -> EpisodeBatch | None:
  """One hindsight episode per eligible episode, or None if there are none. Does not write to `batch`.

  `picks`, covering all of `batch`, replaces the random choices of targets, arms and times.
  Hindsight episodes can be shorter than the real ones: if everything gets done the episode
  ends after the final dwell, as `while not env.terminal()` does at actor.py:378.
  """
  k = num_to_solve(env_cfg, train_cfg)
  keep = eligible(batch, k)
  if not keep.any():
    return None
  batch = select_episodes(batch, keep)
  robot, device = batch.robot, batch.length.device
  n, height = robot.q.shape[:2]
  R, T = env_cfg.world.num_robots, env_cfg.num_targets
  if picks is None:
    final_done = robot.target_done[torch.arange(n, device=device), batch.length]
    picks = draw_picks(final_done, batch.length, R, k, generator)
  else:
    picks = Picks(*[x[keep] for x in picks])

  # Tip poses are stored unscaled, in the world frame. No IK or collision check, actor.py:357-362.
  episodes = torch.arange(n, device=device)[:, None]
  chosen = picks.chosen
  target_pos = torch.where(chosen[..., None], robot.tip_pos[episodes, picks.time, picks.arm], batch.scene.target_pos)
  target_rot = torch.where(chosen[..., None, None], robot.tip_rot[episodes, picks.time, picks.arm], batch.scene.target_rot)

  # All the checks in one GPU call, then a loop on the CPU. The loop reads a result at every step,
  # and on the GPU those reads took 44% of a FastTD3 round with 256 episodes per flush.
  rows = height - 1
  within_rows = targets_within(
    robot.tip_pos[:, 1:].flatten(0, 1),
    robot.tip_rot[:, 1:].flatten(0, 1),
    target_pos.repeat_interleave(rows, 0),
    target_rot.repeat_interleave(rows, 0),
    env_cfg,
  ).unflatten(0, (n, rows)).cpu()

  home = torch.as_tensor(HOME_QPOS, dtype=torch.float32)
  q = robot.q.cpu()
  action = batch.action.to('cpu', copy=True)
  reward = torch.zeros_like(batch.reward, device='cpu')
  all_dwell = torch.zeros_like(robot.dwell, device='cpu')
  all_done = torch.zeros_like(robot.target_done, device='cpu')
  all_score = torch.zeros_like(robot.score, device='cpu')
  dwell, done, score = all_dwell[:, 0], all_done[:, 0], all_score[:, 0]
  # Real collision penalties, for the non-parity option. The env's reward is score delta minus penalty.
  penalty = None
  if not train_cfg.hindsight_zero_collision_penalty:
    penalty = ((robot.score[:, 1:] - robot.score[:, :-1]) - batch.reward).cpu()

  length = batch.length.cpu()
  for t in range(int(length.max())):
    dwelling = dwell > 0
    dwell = (dwell - 1).clamp(min=0)  # Before scoring, planning_env.py:796.
    if train_cfg.hindsight_zero_dwelling_actions:
      action[:, t] = mutated_action(action[:, t], dwelling, done, q[:, t], env_cfg, home)
    done, dwell = solve_targets(within_rows[:, t], done, dwell, env_cfg.dwell_steps)
    new_score = score_of(done, q[:, t + 1], env_cfg, home)
    reward[:, t] = new_score - score if penalty is None else new_score - score - penalty[:, t]
    score = new_score
    all_dwell[:, t + 1], all_done[:, t + 1], all_score[:, t + 1] = dwell, done, score

    # planning_env.py:864-883. Episodes still running have t + 1 < length, and a real
    # episode's own length is its timeout, because one that ended early solved everything.
    everything = done.all(1)
    if everything.any():
      finished = everything & (dwell == 0).all(1)
      if env_cfg.return_to_start_required:
        finished &= ((q[:, t + 1] - home).abs() < env_cfg.return_to_start_epsilon).all(-1).all(-1)
      length = torch.where(finished & (t + 1 < length), t + 1, length)
      if (length <= t + 1).all():
        break

  shape = (n, height, env_cfg.world.num_obstacles)
  if env_cfg.augment_obstacles:  # Drawn fresh for every row, the terminal one too (DESIGN 11.8).
    aug_code = torch.randint(4, shape, generator=generator, device=device, dtype=torch.int8)
  else:
    aug_code = torch.zeros(shape, dtype=torch.int8, device=device)
  stats = batch.stats.clone()
  stats[:, STAT['is_hindsight_replay']] = 1.0
  return EpisodeBatch(
    RobotState(
      robot.q, robot.qd, all_dwell.to(device), robot.tip_pos, robot.tip_rot,
      all_done.to(device), all_score.to(device), robot.step, aug_code,
    ),
    action.to(device), reward.to(device), length.to(device),
    batch.scene._replace(target_pos=target_pos, target_rot=target_rot),
    batch.behaviour_step, torch.ones_like(batch.is_hindsight), stats)


class HindsightQueue:
  """Holds finished real episodes until `hindsight_flush_episodes` have finished.

  The count includes episodes that get no hindsight, so the delay stays bounded when few
  qualify. Only the ones that do are kept.
  """

  def __init__(self, env_cfg: EnvCfg, train_cfg: TrainCfg, generator: torch.Generator) -> None:
    self.env_cfg, self.train_cfg, self.generator = env_cfg, train_cfg, generator
    self.k = num_to_solve(env_cfg, train_cfg)
    self.finished = 0
    self.pending: list[EpisodeBatch] = []

  def add(self, batch: EpisodeBatch) -> EpisodeBatch | None:
    """Queues real episodes. Returns hindsight episodes when this add reaches the flush count."""
    self.finished += len(batch.length)
    keep = eligible(batch, self.k)
    if keep.any():
      self.pending.append(select_episodes(batch, keep))
    return self.flush() if self.finished >= self.train_cfg.hindsight_flush_episodes else None

  def flush(self) -> EpisodeBatch | None:
    """Relabels everything queued, now. Returns None if nothing qualifies."""
    pending, self.pending, self.finished = self.pending, [], 0
    if not pending:
      return None
    return relabel(cat_episodes(pending), self.env_cfg, self.train_cfg, self.generator)
