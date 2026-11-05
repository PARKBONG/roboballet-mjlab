# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""TD3 without policy delay, docs/reports/algo_spec.md section 4, train/train.py:194-410."""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import CriticNet, PolicyNet
from roboballet_mjlab.config import ModelCfg, TrainCfg
from roboballet_mjlab.structs import Graph

# Both schedules count gradient steps in units of this many sampled transitions.
_REFERENCE_BATCH = 128


class Batch(NamedTuple):
  """Transitions, one batch or a megabatch of several batches in sampling order."""

  graph: Graph
  next_graph: Graph
  action: torch.Tensor  # [B, R, 7] in [-1, 1].
  reward: torch.Tensor  # [B]
  terminal: torch.Tensor  # [B], 1 on the last action of every episode, timeouts too.

  def slice(self, start: int, stop: int) -> Batch:
    return Batch(
      Graph(*[x[start:stop] for x in self.graph]), Graph(*[x[start:stop] for x in self.next_graph]),
      self.action[start:stop], self.reward[start:stop], self.terminal[start:stop],
    )


class Losses(NamedTuple):
  critic: torch.Tensor
  policy: torch.Tensor


@contextlib.contextmanager
def matmul_precision(value: str) -> Iterator[None]:
  """Sets the process-wide matmul precision for a block, then restores it, also on an exception.

  The global stays at `highest`, because env and feature code share the process, DESIGN 14.1.
  """
  previous = torch.get_float32_matmul_precision()
  torch.set_float32_matmul_precision(value)
  try:
    yield
  finally:
    torch.set_float32_matmul_precision(previous)


def _polyak(targets: list[torch.Tensor], online: list[torch.Tensor], tau: float) -> None:
  torch._foreach_mul_(targets, 1.0 - tau)
  torch._foreach_add_(targets, online, alpha=tau)


class TD3:
  """Learner state and the update. No .item() and no host copies in the update path."""

  def __init__(
    self,
    model_cfg: ModelCfg,
    cfg: TrainCfg,
    topo: Topology,
    device: torch.device | str,
    world_size: int = 1,
    grad_hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
  ):
    """`grad_hook` maps a flat gradient vector to a new one, the place for an all-reduce."""
    self.cfg = cfg
    self.grad_hook = grad_hook
    self.grad_steps = 0
    # DESIGN 7 and 11.6: sampled-transition equivalents per update. Adam still advances
    # once per update, so this is a changed optimiser rather than sequential updates.
    self.clock_scale = cfg.batch_size * world_size / _REFERENCE_BATCH if cfg.schedule_by_samples else 1.0
    self.tau_policy = 1.0 - (1.0 - cfg.tau_policy) ** self.clock_scale
    self.tau_critic = 1.0 - (1.0 - cfg.tau_critic) ** self.clock_scale

    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(cfg.seed)
      self.policy = PolicyNet(model_cfg, topo).to(device)
      self.q1 = CriticNet(model_cfg, topo).to(device)
      self.q2 = CriticNet(model_cfg, topo).to(device)
    # What the actors hold, DESIGN 14.3 item 4. Refreshed in place by refresh_acting, so its
    # compiled graph stays valid. It only ever runs in float32 at highest.
    self.policy_acting = copy.deepcopy(self.policy).requires_grad_(False)
    self.policy_target = copy.deepcopy(self.policy).requires_grad_(False)
    self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
    self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
    # Critic 1 on the same storage, without gradients for its parameters, for the policy
    # loss. A compiled backward computes every gradient of its module whatever was asked
    # for, so this saves 13 to 20% per step over torch.autograd.grad on q1 itself, and
    # unlike toggling requires_grad it does not recompile.
    self.q1_frozen = copy.deepcopy(self.q1).requires_grad_(False)
    for frozen, live in zip(self.q1_frozen.parameters(), self.q1.parameters()):
      frozen.data = live.data
    self.rng = torch.Generator(device=device).manual_seed(cfg.seed)

    self._policy_params = list(self.policy.parameters())
    self._policy_acting_params = list(self.policy_acting.parameters())
    self._critic_params = [*self.q1.parameters(), *self.q2.parameters()]
    self._policy_target_params = list(self.policy_target.parameters())
    self._critic_target_params = [*self.q1_target.parameters(), *self.q2_target.parameters()]
    adam = dict(betas=(0.9, 0.999), eps=1e-8, fused=torch.device(device).type == 'cuda')
    self.policy_optimizer = torch.optim.Adam(self._policy_params, lr=cfg.lr_policy, **adam)
    self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=cfg.lr_critic, **adam)

    self._autocast = torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.bf16)
    wrap = torch.compile if cfg.compile else lambda module: module
    self._policy = wrap(self.policy)
    self._policy_acting = wrap(self.policy_acting)
    self._q1, self._q2, self._q1_frozen = wrap(self.q1), wrap(self.q2), wrap(self.q1_frozen)
    self._policy_target = wrap(self.policy_target)
    self._q1_target, self._q2_target = wrap(self.q1_target), wrap(self.q2_target)

  def sync_targets(self) -> None:
    """Sets targets equal to the online nets, train.py:519-522."""
    self.policy_target.load_state_dict(self.policy.state_dict())
    self.q1_target.load_state_dict(self.q1.state_dict())
    self.q2_target.load_state_dict(self.q2.state_dict())

  def _clock(self, grad_steps: int) -> float:
    return self.clock_scale * grad_steps / self.cfg.decay_every_steps

  def policy_lr(self, grad_steps: int) -> float:
    return self.cfg.lr_mult * self.cfg.lr_policy * self.cfg.lr_policy_decay ** self._clock(grad_steps)

  def critic_lr(self, grad_steps: int) -> float:
    return self.cfg.lr_mult * self.cfg.lr_critic * self.cfg.lr_critic_decay ** self._clock(grad_steps)

  def exploration_sigma(self, grad_steps: int) -> float:
    return self.cfg.policy_sigma * self.cfg.policy_sigma_decay ** self._clock(grad_steps)

  @torch.no_grad()
  def act(self, graph: Graph) -> torch.Tensor:
    """Deterministic policy forward. Float32 even with bf16, as the original's actors."""
    return self._policy(graph)

  @torch.no_grad()
  def act_behaviour(self, graph: Graph) -> torch.Tensor:
    """Forward of the acting copy, the parameters of the last refresh_acting. Float32."""
    return self._policy_acting(graph)

  @torch.no_grad()
  def refresh_acting(self) -> None:
    """Copies the online policy into the acting copy, in place."""
    torch._foreach_copy_(self._policy_acting_params, self._policy_params)

  def _draw_normals(self, like: torch.Tensor) -> torch.Tensor:
    return torch.randn(like.shape, generator=self.rng, device=like.device, dtype=like.dtype)

  def _target_action(self, next_graph: Graph) -> torch.Tensor:
    """The clip acts on the unit normal, before scaling, train.py:222-227."""
    action = self._policy_target(next_graph)
    clip = self.cfg.target_noise_clip
    noise = self.cfg.target_noise_sigma * self._draw_normals(action).clamp(-clip, clip)
    return (action + noise).clamp(-1.0, 1.0)

  def _apply(
    self,
    optimizer: torch.optim.Optimizer,
    params: list[torch.Tensor],
    grads: tuple[torch.Tensor | None, ...],
    lr: float,
  ) -> None:
    """A parameter without a gradient stays put, with or without the zeros of the hook."""
    if self.grad_hook is not None:
      filled = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
      flat = self.grad_hook(torch.cat([g.flatten() for g in filled]))
      grads = tuple(g.view_as(p) for g, p in zip(flat.split([p.numel() for p in params]), params))
    for p, g in zip(params, grads):
      p.grad = g
    optimizer.param_groups[0]['lr'] = lr
    optimizer.step()

  def train_step(self, batch: Batch) -> Losses:
    """One update, under TrainCfg.matmul_precision only."""
    with matmul_precision(self.cfg.matmul_precision):
      return self._update(batch)

  def _update(self, batch: Batch) -> Losses:
    """One update, in the order of algo_spec 4.1."""
    with torch.no_grad(), self._autocast:
      next_action = self._target_action(batch.next_graph)
      next_q = torch.minimum(self._q1_target(batch.next_graph, next_action), self._q2_target(batch.next_graph, next_action))
      target = batch.reward + (1 - batch.terminal.to(next_q.dtype)) * self.cfg.gamma * next_q
    with self._autocast:
      critic_loss = (target - self._q1(batch.graph, batch.action)).square().mean() + (
        target - self._q2(batch.graph, batch.action)
      ).square().mean()
    grads = torch.autograd.grad(critic_loss, self._critic_params, allow_unused=True)
    self._apply(self.critic_optimizer, self._critic_params, grads, self.critic_lr(self.grad_steps))
    with torch.no_grad():
      _polyak(self._critic_target_params, self._critic_params, self.tau_critic)

    # Critic 1 after its update, through the view of its parameters.
    with self._autocast:
      policy_loss = -self._q1_frozen(batch.graph, self._policy(batch.graph)).mean()
    grads = torch.autograd.grad(policy_loss, self._policy_params, allow_unused=True)
    self._apply(self.policy_optimizer, self._policy_params, grads, self.policy_lr(self.grad_steps))
    with torch.no_grad():
      _polyak(self._policy_target_params, self._policy_params, self.tau_policy)

    self.grad_steps += 1
    return Losses(critic_loss.detach(), policy_loss.detach())

  def fused_step(self, megabatch: Batch, num_steps: int | None = None, summed: bool = False) -> Losses:
    """Consecutive updates on consecutive slices, train.py:363-410.

    Returns losses averaged over the updates, or summed as the original logs them with
    `summed`. `num_steps` defaults to TrainCfg.fused_steps and may be less when the
    samples-per-insert budget permits only that many (DESIGN 11.5). With none, nothing
    changes and both losses are zero, as the original's fori_loop from its zero init.
    """
    num_steps = self.cfg.fused_steps if num_steps is None else num_steps
    size = self.cfg.batch_size
    critic = policy = megabatch.action.new_zeros(())
    with matmul_precision(self.cfg.matmul_precision):
      for i in range(num_steps):
        losses = self._update(megabatch.slice(i * size, (i + 1) * size))
        critic, policy = critic + losses.critic, policy + losses.policy
    divisor = 1 if summed else max(num_steps, 1)
    return Losses(critic / divisor, policy / divisor)

  def state_dict(self) -> dict[str, Any]:
    return {
      'grad_steps': self.grad_steps,
      'policy': self.policy.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
      'policy_acting': self.policy_acting.state_dict(),
      'policy_target': self.policy_target.state_dict(),
      'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
      'policy_optimizer': self.policy_optimizer.state_dict(),
      'critic_optimizer': self.critic_optimizer.state_dict(),
      'rng': self.rng.get_state(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.grad_steps = state['grad_steps']
    for name in ('policy', 'q1', 'q2', 'policy_acting', 'policy_target', 'q1_target', 'q2_target'):
      getattr(self, name).load_state_dict(state[name], strict=True)
    self.policy_optimizer.load_state_dict(state['policy_optimizer'])
    self.critic_optimizer.load_state_dict(state['critic_optimizer'])
    self.rng.set_state(state['rng'].cpu())
