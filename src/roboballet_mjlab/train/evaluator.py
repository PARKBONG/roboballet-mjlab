# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Batched evaluation on separate envs. DESIGN 7 and 14.3 item 6, agents/evaluator.py, algo_spec 8.

Every set has its own BalletEnv with its own generator, and the policy is a private copy, so an
evaluation cannot change the training stream or world state. Sigma 0, augmentation on, float32
at highest. A world counts its first terminal only: after that it resets and keeps stepping,
because the batch does, and nothing it does then is counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
import zlib

import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import PolicyNet
from roboballet_mjlab.agent.td3 import matmul_precision
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.scenes.io import load_pool, load_start_states
from roboballet_mjlab.structs import Scene

EVAL_SETS = ('training_dist', 'heldout', 'static_dist', 'static_paper')
STATIC_SETS = ('static_dist', 'static_paper')
_COUNTED = ('total_collision_robot_steps', 'total_collision_penalty', 'total_mean_acceleration')


def load_scenes(path: str | Path, device) -> Scene:
  """A scene file: original start states from a .pb, anything else a pool of scenes/io.py."""
  path = Path(path)
  return load_start_states(path, device) if path.suffix == '.pb' else load_pool(path, device).scene


def task_rate_se(done: torch.Tensor, num_targets: int) -> float:
  """Standard error of the task rate over worlds. One world has no spread to measure, so 0."""
  if len(done) < 2:
    return 0.0
  return (done.std() / num_targets / len(done) ** 0.5).item()


class Evaluator:
  """`sets` maps a name in EVAL_SETS to a pool path or Scene. An absent set is skipped, with a notice."""

  def __init__(
    self, env_cfg: EnvCfg, model_cfg: ModelCfg, sets: Mapping[str, str | Path | Scene | None], device, seed: int,
    num_worlds: int = 1024, static_worlds: int = 80, compile: bool = True,
    notice: Callable[[str], None] = print,
  ) -> None:
    self.env_cfg = env_cfg
    self.device = device
    self.topo = Topology(env_cfg, device)
    self.envs: dict[str, BalletEnv] = {}
    self.scene_ids: dict[str, torch.Tensor | None] = {}
    self.last_per_world: dict[str, dict[str, torch.Tensor]] = {}
    unknown = set(sets) - set(EVAL_SETS)
    if unknown:
      raise ValueError(f'unknown evaluation sets {sorted(unknown)}, expected some of {EVAL_SETS}')
    for name in EVAL_SETS:
      source = sets.get(name)
      if source is None or (not isinstance(source, Scene) and not Path(source).exists()):
        notice(f'evaluation set {name}: skipped, no pool{"" if source is None else f" at {source}"}')
        continue
      pool = source if isinstance(source, Scene) else load_scenes(source, device)
      size = len(pool.q0)
      ids = None
      worlds = num_worlds
      if name in STATIC_SETS:
        # Every world runs a forced scene. The original runs one episode per set.
        repeats = max(1, round(static_worlds / size))
        ids = torch.arange(size, device=device).repeat(repeats)
        worlds = len(ids)
      self.scene_ids[name] = ids
      self.envs[name] = BalletEnv(env_cfg, worlds, pool, device, seed=zlib.crc32(f'{seed}/eval/{name}'.encode()))
      notice(f'evaluation set {name}: {size} scenes, {worlds} worlds')
    self.policy = PolicyNet(model_cfg, self.topo).to(device).requires_grad_(False)
    self._forward = torch.compile(self.policy) if compile else self.policy

  @torch.no_grad()
  def load_policy(self, policy: PolicyNet) -> None:
    for mine, theirs in zip(self.policy.parameters(), policy.parameters(), strict=True):
      mine.copy_(theirs)

  @torch.no_grad()
  def _rollout(self, env: BalletEnv, scene_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Runs every world to its first terminal. Per-world metrics, as the original names them."""
    W, cfg, device = env.num_worlds, self.env_cfg, self.device
    env.reset(scene_ids=scene_ids)
    running = torch.ones(W, dtype=torch.bool, device=device)
    counted = {name: torch.zeros(W, device=device) for name in _COUNTED}
    done = torch.zeros(W, device=device)
    score = torch.zeros(W, device=device)
    length = torch.zeros(W, device=device)
    for _ in range(cfg.max_steps):
      action = self._forward(build_graph(env.observe(), self.topo))
      out = env.step(action)
      counted['total_collision_robot_steps'] += running * out.collision_robot_steps
      counted['total_collision_penalty'] += running * out.collision_penalty
      counted['total_mean_acceleration'] += running * out.acceleration
      first = running & out.terminal
      final = out.next_robot
      done = torch.where(first, final.target_done.sum(1).float(), done)
      score = torch.where(first, final.score, score)
      length = torch.where(first, final.step + 1.0, length)  # Observations, actor.py:290.
      running = running & ~out.terminal
      if not running.any():
        break
    if running.any():
      raise RuntimeError(f'{int(running.sum())} worlds did not end within max_steps')
    return {'num_targets_done': done, 'final_score': score, 'episode_length': length, **counted}

  def evaluate(self, policy: PolicyNet, network_step: int) -> dict[str, dict[str, float]]:
    """Mean metrics per set. Tag them eval/{set}/{metric}, at global step `network_step`."""
    self.load_policy(policy)
    results = {}
    with matmul_precision('highest'):
      for name, env in self.envs.items():
        per_world = self._rollout(env, self.scene_ids[name])
        self.last_per_world[name] = per_world
        metrics = {key: value.mean().item() for key, value in per_world.items()}
        done = per_world['num_targets_done']
        metrics['task_rate'] = metrics['num_targets_done'] / self.env_cfg.num_targets
        metrics['task_rate_se'] = task_rate_se(done, self.env_cfg.num_targets)
        metrics['all_done_fraction'] = (done == self.env_cfg.num_targets).float().mean().item()
        metrics['network_step'] = float(network_step)
        metrics['worlds'] = float(env.num_worlds)
        metrics['residual_collisions'] = float(env.residual_collisions.item())
        results[name] = metrics
    return results

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {name: env.generator.get_state() for name, env in self.envs.items()}

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    for name, env in self.envs.items():
      if name in state:
        env.generator.set_state(state[name].cpu())
