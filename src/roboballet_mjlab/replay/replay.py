# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Episode buffer and compact replay ring, on the GPU. DESIGN sections 7, 11.2, 11.3, 11.5.

An episode of A actions has A + 1 states, each one a row, and A transitions. The final
state has no transition, so it is never sampled and does not count toward capacity.
Removal is FIFO per transition as in launch/launch.py:154, so an episode can be partly gone.

Three rings: contiguous rows (state, action, reward, terminal, episode slot), so that s' of
row i is row i + 1; a map from transition id to row; and an episode table. The ring lengths
are `capacity` transitions, `max_episodes` slots and capacity + max_episodes rows.

Why rows cannot run out: live rows are exactly live transitions plus live episodes, because
each episode adds one final row and rows are contiguous in FIFO order. So keeping at most
`capacity` transitions and `max_episodes` episodes alive bounds rows too, and the one extra
rule is that overwriting a table slot evicts what is left of that episode. Those evictions
are counted in `extra_evicted`, and they need a mean episode length below
capacity / max_episodes, which is 8 steps by default.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import torch

from roboballet_mjlab.config import EnvCfg
from roboballet_mjlab.structs import Obs, RobotState, Scene, StepOut

# train/stats.py:43-73 in field order, filled as in agents/actor.py:283-293 and :610-611.
STAT_NAMES = (
  'final_score',
  'total_reward',
  'total_collision_penalty',
  'total_collision_robot_steps',
  'total_mean_acceleration',
  'num_targets_done',
  'episode_length',  # Observations, so A + 1, actor.py:290.
  'env_sampling_time_s',  # Always 0 here: scenes are drawn in batches, not per episode.
  'episode_wall_time_s',
  'episode_write_wall_time_s',  # Always 0 here: rows are written during the episode.
  'is_hindsight_replay',
  'network_step',
)
STAT = {name: i for i, name in enumerate(STAT_NAMES)}


class EpisodeBatch(NamedTuple):
  """n episodes, padded in time. Rows past `length` hold stale values and mean nothing."""

  robot: RobotState  # Leaves [n, H, ...], H >= length.max() + 1, in the compact dtypes of row_storage.
  action: torch.Tensor  # [n, H - 1, R, 7], in [-1, 1], but hindsight takeover can exceed that.
  reward: torch.Tensor  # [n, H - 1]
  length: torch.Tensor  # [n], long, actions A. Row A is the final state.
  scene: Scene  # Leaves [n, ...]
  behaviour_step: torch.Tensor  # [n], long. The grad step when the episode began.
  is_hindsight: torch.Tensor  # [n], bool
  stats: torch.Tensor  # [n, 12], f32, ordered as STAT_NAMES.


class Sample(NamedTuple):
  """Uniform transitions. Nothing here shares memory with the replay."""

  obs: Obs  # Integer fields widened to long, as the env hands them out.
  next_obs: Obs  # Shares its Scene tensors with `obs`.
  action: torch.Tensor  # [n, R, 7]
  reward: torch.Tensor  # [n]
  terminal: torch.Tensor  # [n], bool
  behaviour_step: torch.Tensor  # [n], long
  is_hindsight: torch.Tensor  # [n], bool
  stats: torch.Tensor  # [n, 12], ordered as STAT_NAMES.
  steps: torch.Tensor | None = None  # [n], long. Steps between obs and next_obs, when n_step is above 1.


def _small_int(largest: int) -> torch.dtype:
  return torch.int8 if largest <= torch.iinfo(torch.int8).max else torch.int16


def row_storage(cfg: EnvCfg, shape: tuple[int, ...], device) -> RobotState:
  """Zeroed RobotState leaves in the smallest dtypes that keep exact values."""
  R, T, O = cfg.world.num_robots, cfg.num_targets, cfg.world.num_obstacles
  zeros = lambda *s, dtype=torch.float32: torch.zeros((*shape, *s), dtype=dtype, device=device)
  return RobotState(
    q=zeros(R, 7),
    qd=zeros(R, 7),
    dwell=zeros(R, dtype=_small_int(cfg.dwell_steps)),
    tip_pos=zeros(R, 3),
    tip_rot=zeros(R, 3, 3),
    target_done=zeros(T, dtype=torch.bool),
    score=zeros(),
    step=zeros(dtype=_small_int(cfg.max_steps)),
    aug_code=zeros(O, dtype=torch.int8),
  )


def select_episodes(batch: EpisodeBatch, index: torch.Tensor) -> EpisodeBatch:
  take = lambda x: x[index]
  return EpisodeBatch(
    RobotState(*map(take, batch.robot)), take(batch.action), take(batch.reward), take(batch.length),
    Scene(*map(take, batch.scene)), take(batch.behaviour_step), take(batch.is_hindsight), take(batch.stats))


def cat_episodes(batches: list[EpisodeBatch]) -> EpisodeBatch:
  """Concatenates batches of the same padded height."""
  join = lambda *x: torch.cat(x)
  return EpisodeBatch(
    RobotState(*map(join, *[b.robot for b in batches])),
    join(*[b.action for b in batches]),
    join(*[b.reward for b in batches]),
    join(*[b.length for b in batches]),
    Scene(*map(join, *[b.scene for b in batches])),
    join(*[b.behaviour_step for b in batches]),
    join(*[b.is_hindsight for b in batches]),
    join(*[b.stats for b in batches]),
  )


class EpisodeBuffer:
  """The running episode of each of W lockstep worlds, up to max_steps + 1 rows."""

  def __init__(self, cfg: EnvCfg, num_worlds: int, device) -> None:
    R, L = cfg.world.num_robots, cfg.max_steps
    self.robot = row_storage(cfg, (num_worlds, L + 1), device)
    self.action = torch.zeros(num_worlds, L, R, 7, device=device)
    self.reward = torch.zeros(num_worlds, L, device=device)
    self.count = torch.zeros(num_worlds, dtype=torch.long, device=device)  # Actions so far.
    self.behaviour_step = torch.zeros(num_worlds, dtype=torch.long, device=device)
    # Running reward, collision penalty, collision robot steps and acceleration.
    self.totals = torch.zeros(num_worlds, 4, device=device)
    self.started = torch.full((num_worlds,), time.perf_counter(), dtype=torch.float64, device=device)
    self._worlds = torch.arange(num_worlds, device=device)

  @torch.no_grad()
  def append(
    self, obs: Obs, action: torch.Tensor, out: StepOut, behaviour_step: int | torch.Tensor = 0
  ) -> EpisodeBatch | None:
    """Records one env step. Returns the episodes it finished, as copies, or None.

    `obs` is env.observe() from before the step and `action` is what went into env.step:
    the raw request, because the env does not write into it (actor.py:263-264).
    `behaviour_step` is the current grad step: the first row of an episode latches it.
    Stored rows never carry a graph, so a ring cannot keep one alive for its whole lap.
    """
    worlds, count = self._worlds, self.count
    for store, value in zip(self.robot, obs.robot):
      store[worlds, count] = value.to(store.dtype)
    self.action[worlds, count] = action
    self.reward[worlds, count] = out.reward
    self.behaviour_step = torch.where(count == 0, behaviour_step, self.behaviour_step)
    self.totals += torch.stack(
      [out.reward, out.collision_penalty, out.collision_robot_steps.to(out.reward.dtype), out.acceleration], 1)
    self.count = count + 1
    ended = out.terminal.nonzero().flatten()
    return self._finish(ended, obs.scene, out.next_robot) if len(ended) else None

  def _finish(self, ended: torch.Tensor, scene: Scene, final: RobotState) -> EpisodeBatch:
    length = self.count[ended]
    for store, value in zip(self.robot, final):
      store[ended, length] = value[ended].to(store.dtype)
    total = self.totals[ended]
    stats = torch.zeros(len(ended), len(STAT_NAMES), device=length.device)
    stats[:, STAT['final_score']] = final.score[ended]
    stats[:, STAT['total_reward']] = total[:, 0]
    stats[:, STAT['total_collision_penalty']] = total[:, 1]
    stats[:, STAT['total_collision_robot_steps']] = total[:, 2]
    stats[:, STAT['total_mean_acceleration']] = total[:, 3]
    stats[:, STAT['num_targets_done']] = final.target_done[ended].sum(1).to(stats.dtype)
    stats[:, STAT['episode_length']] = (length + 1).to(stats.dtype)
    stats[:, STAT['episode_wall_time_s']] = (time.perf_counter() - self.started[ended]).to(stats.dtype)
    behaviour = self.behaviour_step[ended]
    stats[:, STAT['network_step']] = behaviour.to(stats.dtype)
    batch = EpisodeBatch(
      RobotState(*[x[ended] for x in self.robot]), self.action[ended], self.reward[ended], length,
      Scene(*[x[ended] for x in scene]), behaviour, torch.zeros_like(length, dtype=torch.bool), stats)
    self.discard(ended)
    return batch

  def discard(self, world_ids: torch.Tensor) -> None:
    """Forgets running episodes, for worlds the trainer resets without a terminal."""
    self.count[world_ids] = 0
    self.totals[world_ids] = 0.0
    self.started[world_ids] = time.perf_counter()


class ReplayBuffer:
  """FIFO over transitions, uniform sampling with replacement. Rate limiting is the trainer's."""

  def __init__(self, cfg: EnvCfg, capacity: int, device, max_episodes: int | None = None) -> None:
    self.cfg = cfg
    self.device = device
    self.capacity = capacity
    self.max_episodes = max(2, capacity // 8) if max_episodes is None else max_episodes
    self.num_rows = capacity + self.max_episodes
    M, E, R = self.num_rows, self.max_episodes, cfg.world.num_robots
    T, O = cfg.num_targets, cfg.world.num_obstacles

    self.robot = row_storage(cfg, (M,), device)
    self.action = torch.zeros(M, R, 7, device=device)
    self.reward = torch.zeros(M, device=device)
    self.terminal = torch.zeros(M, dtype=torch.bool, device=device)
    self.row_episode = torch.zeros(M, dtype=torch.long, device=device)
    self.transition_row = torch.zeros(capacity, dtype=torch.long, device=device)

    zeros = lambda *s: torch.zeros((E, *s), device=device)
    self.scene = Scene(
      base_pos=zeros(R, 3), base_rot=zeros(R, 3, 3), box_pos=zeros(O, 3), box_rot=zeros(O, 3, 3),
      box_spans=zeros(O, 3), target_pos=zeros(T, 3), target_rot=zeros(T, 3, 3), q0=zeros(R, 7))
    self.behaviour_step = torch.zeros(E, dtype=torch.long, device=device)
    self.is_hindsight = torch.zeros(E, dtype=torch.bool, device=device)
    self.stats = zeros(len(STAT_NAMES))
    self.first_transition = torch.zeros(E, dtype=torch.long, device=device)

    # Every count is a Python int. A transition's id is the count of inserts before it, so
    # the live ones are [evicted, inserted) and the oldest live id is `evicted`.
    self.inserted = 0
    self.evicted = 0
    self.extra_evicted = 0
    self.sampled = 0
    self.episodes = 0
    self._next_row = 0

  @property
  def size(self) -> int:
    return self.inserted - self.evicted

  def ready(self, min_size: int) -> bool:
    return self.size >= min_size

  def counters(self) -> dict[str, int]:
    return dict(inserted=self.inserted, evicted=self.evicted, extra_evicted=self.extra_evicted,
                sampled=self.sampled, size=self.size, episodes=self.episodes)

  def insert(self, batch: EpisodeBatch) -> None:
    """Appends a batch of episodes of varying lengths as one operation, in batch order.

    Host syncs: the nonzero, which a variable-length batch cannot avoid, and one read of the
    table once it has wrapped. A batch too large for the ring adds a few more.
    """
    C, E, M = self.capacity, self.max_episodes, self.num_rows
    device, length = self.device, batch.length
    n = len(length)
    if n == 0:
      return
    stored = torch.arange(batch.robot.q.shape[1], device=device)[None] <= length[:, None]
    ep, t = stored.nonzero(as_tuple=True)
    total = len(ep) - n
    self.inserted += total

    # Episode lengths from the first one kept. Rows are written only if they will stay.
    first, kept_length = 0, length
    if total > C or n > E:  # Only for a batch that does not fit: its own FIFO drops its head.
      later = length.flip(0).cumsum(0).flip(0) - length  # Transitions after each episode.
      first = max(int((later >= C).sum()), n - E)
      skip = max(int(later[first] + length[first]) - C, 0)
      stored[:first] = False
      stored[first, :skip] = False
      ep, t = stored.nonzero(as_tuple=True)
      kept_length = length[first:].clone()
      kept_length[0] -= skip
    kept = n - first
    rows = len(ep)
    kept_total = rows - kept

    dst = (self._next_row + torch.arange(rows, device=device)) % M
    self._next_row = (self._next_row + rows) % M
    for store, source in zip(self.robot, batch.robot):
      store[dst] = source[ep, t]
    is_transition = t < length[ep]
    dst_t, ep_t, t_t = dst[is_transition], ep[is_transition], t[is_transition]
    self.action[dst_t] = batch.action[ep_t, t_t]
    self.reward[dst_t] = batch.reward[ep_t, t_t]
    self.terminal[dst_t] = t_t == length[ep_t] - 1
    first_id = self.inserted - kept_total
    self.transition_row[(first_id + torch.arange(kept_total, device=device)) % C] = dst_t

    slots = (self.episodes + torch.arange(kept, device=device)) % E
    self.episodes += kept
    self.row_episode[dst] = slots[ep - first]
    for store, source in zip(self.scene, batch.scene):
      store[slots] = source[first:]
    self.behaviour_step[slots] = batch.behaviour_step[first:]
    self.is_hindsight[slots] = batch.is_hindsight[first:]
    self.stats[slots] = batch.stats[first:]
    self.first_transition[slots] = first_id + kept_length.cumsum(0) - kept_length

    by_capacity = max(self.evicted, self.inserted - C)
    oldest = max(by_capacity, first_id) if first else by_capacity
    if self.episodes > E:  # Slot episodes % E holds the oldest episode still in the table.
      oldest = max(oldest, int(self.first_transition[self.episodes % E]))
    self.extra_evicted += oldest - by_capacity
    self.evicted = oldest

  def sample(self, n: int, generator: torch.Generator, n_step: int = 1, gamma: float = 1.0) -> Sample:
    if self.size == 0:
      raise ValueError('sampling from an empty replay')
    self.sampled += n
    return self.fetch(torch.randint(self.size, (n,), generator=generator, device=self.device), n_step, gamma)

  def fetch(self, position: torch.Tensor, n_step: int = 1, gamma: float = 1.0) -> Sample:
    """Live transitions by age, 0 being the oldest. `sample` draws positions uniformly.
    Not counted in `sampled`, so tests and diagnostics can read without moving the budget.

    With n_step above 1, reward is the discounted sum up to n_step rewards, next_obs is that many
    steps on, and a terminal among them ends the sum, as in FastTD3's fast_td3_utils.py:250-306.
    Later rows of an episode are younger than the row sampled, so eviction cannot have reached them.
    """
    row = self.transition_row[(self.evicted + position) % self.capacity]
    episode = self.row_episode[row]
    scene = Scene(*[x[episode] for x in self.scene])
    steps = None
    if n_step == 1:
      next_row, reward, terminal = (row + 1) % self.num_rows, self.reward[row], self.terminal[row]
    else:
      offsets = torch.arange(n_step, device=self.device)
      rows = (row[:, None] + offsets) % self.num_rows
      ended = self.terminal[rows]
      # Rows after a terminal belong to whatever comes next in the ring, so they must not count.
      alive = torch.cat([torch.ones_like(ended[:, :1]), ~ended[:, :-1]], 1).long().cumprod(1).bool()
      reward = (self.reward[rows] * alive * gamma ** offsets.float()).sum(1)
      terminal = (ended & alive).any(1)
      steps = alive.sum(1)
      next_row = (row + steps) % self.num_rows
    return Sample(
      Obs(self._state(row), scene), Obs(self._state(next_row), scene), self.action[row], reward, terminal,
      self.behaviour_step[episode], self.is_hindsight[episode], self.stats[episode], steps)

  def _state(self, row: torch.Tensor) -> RobotState:
    r = self.robot
    return RobotState(r.q[row], r.qd[row], r.dwell[row].long(), r.tip_pos[row], r.tip_rot[row],
                      r.target_done[row], r.score[row], r.step[row].long(), r.aug_code[row].long())
