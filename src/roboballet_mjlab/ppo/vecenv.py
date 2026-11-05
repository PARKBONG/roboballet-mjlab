"""rsl-rl's VecEnv on BalletEnv. DESIGN section 17, rsl_rl/env/vec_env.py, mjlab/rl/vecenv_wrapper.py.

Observations are the compact fields of Obs, so the rollout stores those and the models build the
features. Dones hold every terminal, timeouts too, and `time_outs` is never in extras: ppo.py:151-155
would add gamma * V to the reward of those rows otherwise. So no terminal bootstraps, as in the other
agents, since the step count is an input feature (DESIGN 12.1).
"""

from __future__ import annotations

import zlib

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from roboballet_mjlab.agent.features import ACTION_DIM
from roboballet_mjlab.config import EnvCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.structs import Obs, RobotState, Scene

# Episode totals, named as replay.STAT_NAMES names them.
_SUMMED = ('total_reward', 'total_collision_penalty', 'total_collision_robot_steps', 'total_mean_acceleration')
EPISODE_STATS = ('final_score', *_SUMMED, 'num_targets_done', 'episode_length')


def obs_to_tensordict(obs: Obs) -> TensorDict:
  """The same tensors under their field names, without a copy. The env never writes to them afterwards."""
  fields = {**obs.robot._asdict(), **obs.scene._asdict()}
  return TensorDict(fields, batch_size=[len(obs.robot.q)])


def tensordict_to_obs(fields: TensorDict) -> Obs:
  return Obs(RobotState(*[fields[name] for name in RobotState._fields]), Scene(*[fields[name] for name in Scene._fields]))


def stagger_offsets(num_worlds: int, max_steps: int, device) -> torch.Tensor:
  """The round each world resets at, -1 for never: fasttd3/trainer.py and train/trainer.py have the same rule."""
  offsets = (torch.arange(num_worlds, device=device) * max_steps) // num_worlds
  return offsets.masked_fill(offsets == 0, -1)


class BalletVecEnv(VecEnv):
  """`stagger` runs the worlds before any rollout exists, on uniform actions, resetting each once at its
  own offset. The other agents reset at the same offsets and discard the partial rows. Here there are no
  rows to discard, so no reset ever falls inside a rollout and no terminal is made up for one.
  """

  def __init__(self, env_cfg: EnvCfg, num_worlds: int, pool: Scene, device, seed: int, stagger: bool = True) -> None:
    self.cfg = env_cfg
    self.env = BalletEnv(env_cfg, num_worlds, pool, device, seed=seed)
    self.num_envs = num_worlds
    self.num_robots = env_cfg.world.num_robots
    self.num_actions = self.num_robots * ACTION_DIM
    self.max_episode_length = env_cfg.max_steps
    self.device = device
    self.rounds = 0  # Env steps of every world since this object was made, the stagger's too.
    self.stagger_rounds = 0
    self.discarded_rows = 0
    if stagger:
      self._stagger(zlib.crc32(f'{seed}/stagger'.encode()))
    self._totals = torch.zeros(num_worlds, len(_SUMMED), device=device)
    self.episodes = 0
    self._stat_sum = torch.zeros(len(EPISODE_STATS), dtype=torch.float64, device=device)
    self._stat_count = 0

  @property
  def episode_length_buf(self) -> torch.Tensor:
    """Read-only, so runner.learn(init_at_random_ep_len=True) fails: the count is an input, and `stagger` does this job."""
    return self.env.observe().robot.step

  def _stagger(self, seed: int) -> None:
    W, device = self.num_envs, self.device
    generator = torch.Generator(device=device).manual_seed(seed)
    reset_at = stagger_offsets(W, self.cfg.max_steps, device)
    last = int(reset_at.max())
    all_worlds = torch.arange(W, device=device)
    for round_index in range(last + 1):
      due = all_worlds[reset_at == round_index]
      if len(due):
        self.env.reset(due)
      if round_index == last:
        break
      action = torch.rand(W, self.num_robots, ACTION_DIM, generator=generator, device=device) * 2.0 - 1.0
      out = self.env.step(action)
      self.rounds += 1
      reset_at = reset_at.masked_fill(out.terminal, -1)  # Ended on its own: already out of step.
    self.stagger_rounds = self.rounds
    self.discarded_rows = W * self.rounds  # No rollout holds any of them.

  def get_observations(self) -> TensorDict:
    return obs_to_tensordict(self.env.observe())

  def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """The rows rsl-rl stores keep its sampled action. The env gets it clamped, as the other agents' actions are."""
    out = self.env.step(actions.view(self.num_envs, self.num_robots, ACTION_DIM).clamp(-1.0, 1.0))
    self.rounds += 1
    self._record(out)
    return self.get_observations(), out.reward, out.terminal.long(), {}

  def _record(self, out) -> None:
    """Episode totals, actor.py:283-293. Bookkeeping only, and all of it stays on the GPU."""
    self._totals += torch.stack([out.reward, out.collision_penalty, out.collision_robot_steps.to(out.reward.dtype),
                                 out.acceleration], 1)
    if not out.terminal.any():
      return
    final = out.next_robot
    ended = out.terminal.nonzero().flatten()
    totals = self._totals[ended]
    stats = torch.stack([final.score[ended], *totals.T, final.target_done[ended].sum(1).to(totals.dtype),
                         final.step[ended].to(totals.dtype) + 1.0], 1)  # Observations, actor.py:290.
    self._stat_sum += stats.sum(0).double()
    self._stat_count += len(ended)
    self.episodes += len(ended)
    self._totals[ended] = 0.0

  def take_episode_stats(self) -> dict[str, float]:
    """Mean of each statistic over the episodes that ended since the last call. Empty if none did."""
    if not self._stat_count:
      return {}
    means = (self._stat_sum / self._stat_count).tolist()
    self._stat_sum.zero_()
    self._stat_count = 0
    return dict(zip(EPISODE_STATS, means))
