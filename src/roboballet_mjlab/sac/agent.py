# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The update rule of github.com/leggedrobotics/rsl_rl_sac at e0d243a, rsl_rl/algorithms/sac.py and
# rsl_rl/models/sac_mlp_model.py, rewritten for a graph network and fused GPU code.
"""The SAC update of rsl_rl_sac on the RoboBallet GNN. DESIGN section 16, arXiv 2605.24975.

tests/golden/sac_make_goldens.py holds a verbatim excerpt of the fork, and tests/test_sac.py compares
everything here with the numbers that excerpt produces.

Policy initialization, the paper's first change, section 3.3 and sac_mlp_model.py:82-96: on the last
layer the mean rows have weights N(0, 1e-3) and zero bias, the log-std rows zero weights and a constant
bias log(init_noise_std + 1e-7). Every other layer keeps the GNN's own initialization.

The critic has no tanh, as the fork, unlike agent.networks.CriticNet. A soft Q holds the entropy bonus,
which leaves [-1, 1] at once: a fresh actor at 4 robots has an entropy near +19 and the default
temperature is 0.05, so 0.95 per step, about 16 discounted at gamma 0.94.

Learner in bf16, compiled. Acting in f32 at the process's `highest`. SACCfg.matmul_precision holds for
the learner call only, DESIGN 14.1. Log-probabilities, targets and losses are f32 in both, because
agent.networks.MLP returns f32.
"""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from roboballet_mjlab.agent.features import ACTION_DIM, NODE_DIM, Topology
from roboballet_mjlab.agent.networks import MLP, GraphNet, PolicyNet
from roboballet_mjlab.config import ModelCfg
from roboballet_mjlab.sac.config import SACCfg
from roboballet_mjlab.structs import Graph


class Batch(NamedTuple):
  graph: Graph
  next_graph: Graph
  action: torch.Tensor  # [B, R, 7]
  reward: torch.Tensor  # [B]
  terminal: torch.Tensor  # [B], bool. True on timeouts too: no bootstrap on any terminal, DESIGN 16.
  discount: torch.Tensor | None = None  # [B], gamma ** steps, when n_step is above 1.


class Acted(NamedTuple):
  action: torch.Tensor  # The sample, stored and sent to the env.
  mean: torch.Tensor  # tanh of the mean, what evaluation runs.
  std: torch.Tensor  # Before the squash.


class Update(NamedTuple):
  """Scalars of one update, still on the GPU. Every field is float32."""

  critic_loss: torch.Tensor
  critic1_loss: torch.Tensor
  critic2_loss: torch.Tensor
  critic_grad_norm: torch.Tensor  # Before the clip.
  target_mean: torch.Tensor
  target_min: torch.Tensor
  target_max: torch.Tensor
  q_mean: torch.Tensor  # Of the first critic, on the stored action.
  q_std: torch.Tensor  # Its spread over the batch. Near zero is a critic that has collapsed to a constant.
  alpha_loss: torch.Tensor
  alpha: torch.Tensor  # After this update's change.
  entropy: torch.Tensor  # -log_prob of the actions used for the temperature and actor updates.
  log_std: torch.Tensor  # Mean of the clamped log-std.


class ActorUpdate(NamedTuple):
  loss: torch.Tensor
  grad_norm: torch.Tensor  # Before the clip.


@contextlib.contextmanager
def matmul_precision(value: str) -> Iterator[None]:
  """Sets the process-wide matmul precision for a block, then restores it, also on an exception."""
  previous = torch.get_float32_matmul_precision()
  torch.set_float32_matmul_precision(value)
  try:
    yield
  finally:
    torch.set_float32_matmul_precision(previous)


class GaussianActor(nn.Module):
  """agent.networks.PolicyNet's graph, with a last layer of mean then log-std per robot node.

  One Dense with 14 outputs per node, as the fork's has 2 x d, sac_mlp_model.py:159-160.
  """

  def __init__(self, cfg: ModelCfg, topo: Topology, init_noise_std: float):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM)
    self.policy_head = MLP(cfg.node_size, [*cfg.head_sizes, 2 * ACTION_DIM], cfg.layernorm_eps)
    last = self.policy_head[-1]
    with torch.no_grad():  # sac_mlp_model.py:92-96.
      nn.init.normal_(last.weight[:ACTION_DIM], mean=0.0, std=1e-3)
      nn.init.zeros_(last.bias[:ACTION_DIM])
      nn.init.zeros_(last.weight[ACTION_DIM:])
      nn.init.constant_(last.bias[ACTION_DIM:], torch.log(torch.tensor(init_noise_std + 1e-7)))

  def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and raw log-std, each [B, R, 7] and float32."""
    gnn, R = self.gnn, self.gnn.R
    n = gnn.embed_nodes(graph.nodes[:, :R], graph.nodes)
    g = gnn.embed_global(graph.globals)
    e = gnn.update_edges(n, graph.edges, g)
    from_robots, from_targets, from_obstacles = gnn.edge_blocks(e)
    received = from_robots.sum(2) + from_targets.sum(2) + from_obstacles.sum(2)
    sent = e[:, gnn.robot_edges_by_sender].sum(2)
    robots = gnn.update_node(torch.cat([n[:, :R], sent, received, g[:, None].expand(-1, R, -1)], -1))
    return self.policy_head(robots).split(ACTION_DIM, -1)


class QNet(nn.Module):
  """agent.networks.CriticNet with no tanh and no reward head, sac_mlp_model.py:177-235. [B]."""

  def __init__(self, cfg: ModelCfg, topo: Topology):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM + ACTION_DIM)
    self.qvalue = MLP(cfg.node_size, [*cfg.head_sizes, 1], cfg.layernorm_eps, cfg.head_last_scale)

  def forward(self, graph: Graph, action: torch.Tensor) -> torch.Tensor:
    gnn, R = self.gnn, self.gnn.R
    n = gnn.embed_nodes(torch.cat([graph.nodes[:, :R], action], -1), graph.nodes)
    g = gnn.embed_global(graph.globals)
    e = gnn.update_edges(n, graph.edges, g)
    from_robots, from_targets, from_obstacles = gnn.edge_blocks(e)
    received = from_robots.sum(2) + from_targets.sum(2) + from_obstacles.sum(2)  # Only robots receive.
    received = torch.cat([received, e.new_zeros(e.shape[0], gnn.T + gnn.O, e.shape[-1])], 1)
    sent = torch.cat([e[:, gnn.robot_edges_by_sender].sum(2), from_targets.sum(1), from_obstacles.sum(1)], 1)
    n = gnn.update_node(torch.cat([n, sent, received, g[:, None].expand(-1, gnn.N, -1)], -1))
    g = gnn.update_global(torch.cat([n.sum(1), e.sum(1), g], -1))
    return self.qvalue(g)[:, 0]


def mean_policy_of(actor: GaussianActor, cfg: ModelCfg, topo: Topology) -> PolicyNet:
  """A PolicyNet on the actor's storage, without the log-std rows. Its tanh(head) is the actor's
  tanh(mean), sac_mlp_model.py:130-131, and it is what the Evaluator and scripts/eval.py take."""
  view = PolicyNet(cfg, topo).to(next(actor.parameters()).device).requires_grad_(False)
  source = dict(actor.named_parameters())
  for name, parameter in view.named_parameters():
    storage = source[name].data
    parameter.data = storage if parameter.shape == storage.shape else storage[:ACTION_DIM]
  return view


def squash(
  mean: torch.Tensor, log_std: torch.Tensor, normals: torch.Tensor, log_std_min: float, log_std_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Action and its log-probability per batch row, sac_mlp_model.py:144-162.

  The action range is 1 and its bias 0 here, because the env takes actions in [-1, 1], so the fork's
  terms for them vanish. torch's Normal.rsample and .log_prob written out, without their draw and
  without the argument checks, which the fork turns off too, mlp_model.py:102.
  """
  std = log_std.clamp(log_std_min, log_std_max).exp()
  x = mean + normals * std
  tanh_x = torch.tanh(x)
  gaussian = -((x - mean) ** 2) / (2 * std**2) - std.log() - math.log(math.sqrt(2 * math.pi))
  log_prob = gaussian.flatten(1).sum(1)
  log_prob = log_prob - torch.log(1 - tanh_x.pow(2) + 1e-6).flatten(1).sum(1)
  return tanh_x, log_prob


def decayed_lr(start: float, step: int, decay_steps: int, end_fraction: float) -> float:
  """CosineAnnealingLR's value at this grad step, held at the end once reached, as in fasttd3/agent.py."""
  if decay_steps <= 0:
    return start
  end = start * end_fraction
  return end + (start - end) * (1.0 + math.cos(math.pi * min(step, decay_steps) / decay_steps)) / 2.0


def critic_target(
  reward: torch.Tensor, terminal: torch.Tensor, q1_next: torch.Tensor, q2_next: torch.Tensor, next_log_prob: torch.Tensor,
  log_alpha: torch.Tensor, gamma: float | torch.Tensor,
) -> torch.Tensor:
  """sac.py:277-288 with no timeout mask: any terminal stops the return, DESIGN 16."""
  next_state_entropy = -log_alpha.exp() * next_log_prob
  q_target_next = torch.min(q1_next, q2_next) + next_state_entropy
  return reward + gamma * (~terminal).to(q_target_next.dtype) * q_target_next


def critic_loss(q1: torch.Tensor, q2: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """sac.py:292-295. The total and each critic's."""
  loss1, loss2 = F.mse_loss(q1, target), F.mse_loss(q2, target)
  return 0.5 * (loss1 + loss2), loss1, loss2


def temperature_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, target_entropy: float) -> torch.Tensor:
  """sac.py:312. Falls with log_alpha while the entropy is below its target, so the temperature rises."""
  return -(log_alpha * (log_prob + target_entropy).detach()).mean()


def actor_loss(log_alpha: torch.Tensor, log_prob: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """sac.py:326, :336-337."""
  return (log_alpha.exp().detach() * log_prob - torch.min(q1, q2)).mean()


def soft_update(targets: list[torch.Tensor], online: list[torch.Tensor], tau: float) -> None:
  """sac_mlp_model.py:313-316."""
  torch._foreach_mul_(targets, 1.0 - tau)
  torch._foreach_add_(targets, online, alpha=tau)


class SAC:
  """Learner state and the update, in the fork's order. No .item() and no host copies in the update path.

  `grad_hook` maps a flat gradient vector to a new one, the place for an all-reduce. `nets` replaces
  the actor and the two critics, for the goldens, where they are the fork's MLPs, and `action_dim`
  then gives the target entropy.
  """

  def __init__(
    self, model_cfg: ModelCfg | None, cfg: SACCfg, topo: Topology | None, device: torch.device | str,
    grad_hook: Callable[[torch.Tensor], torch.Tensor] | None = None, noise_seed: int | None = None,
    nets: tuple[nn.Module, nn.Module, nn.Module] | None = None, action_dim: int | None = None,
  ):
    self.cfg = cfg
    self.grad_hook = grad_hook
    self.grad_steps = 0  # update_step of sac.py:147.
    self.policy_steps = 0
    self.mean_policy = None
    if nets is None:
      with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        nets = (GaussianActor(model_cfg, topo, cfg.init_noise_std), QNet(model_cfg, topo), QNet(model_cfg, topo))
        self.actor, self.q1, self.q2 = (net.to(device) for net in nets)
        self.mean_policy = mean_policy_of(self.actor, model_cfg, topo)
      action_dim = topo.num_robots * ACTION_DIM
    else:
      self.actor, self.q1, self.q2 = (net.to(device) for net in nets)
    self.target_entropy = -cfg.target_entropy_scale * action_dim  # sac.py:151.
    self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
    self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
    # The critics on the same storage, without gradients for their parameters, for the actor loss,
    # in place of sac.py:332-333: a compiled backward computes every gradient of its module.
    self.q1_frozen = copy.deepcopy(self.q1).requires_grad_(False)
    self.q2_frozen = copy.deepcopy(self.q2).requires_grad_(False)
    for frozen, live in zip([*self.q1_frozen.parameters(), *self.q2_frozen.parameters()],
                            [*self.q1.parameters(), *self.q2.parameters()]):
      frozen.data = live.data
    self.rng = torch.Generator(device=device).manual_seed(cfg.seed if noise_seed is None else noise_seed)

    self._actor_params = list(self.actor.parameters())
    self._critic_params = [*self.q1.parameters(), *self.q2.parameters()]
    self._critic_target_params = [*self.q1_target.parameters(), *self.q2_target.parameters()]
    self.log_alpha = torch.log(torch.tensor(cfg.alpha, device=device)).detach().clone().requires_grad_(True)  # sac.py:155.
    fused = torch.device(device).type == 'cuda'
    self.actor_optimizer = torch.optim.Adam(self._actor_params, lr=cfg.lr_actor, fused=fused)
    self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=cfg.lr_critic, fused=fused)
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha, fused=fused)

    self._autocast = torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.bf16)
    wrap = torch.compile if cfg.compile else lambda module: module
    self._actor = wrap(self.actor)
    self._q1, self._q2 = wrap(self.q1), wrap(self.q2)
    self._q1_target, self._q2_target = wrap(self.q1_target), wrap(self.q2_target)
    self._q1_frozen, self._q2_frozen = wrap(self.q1_frozen), wrap(self.q2_frozen)

  @torch.no_grad()
  def act(self, graph: Graph, normals: torch.Tensor) -> Acted:
    """A sample of the online actor, f32 and without autocast, sac.py:178-184."""
    mean, log_std = self._actor(graph)
    std = log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max).exp()
    return Acted(torch.tanh(mean + normals * std), torch.tanh(mean), std)

  def _draw_normals(self, like: torch.Tensor) -> torch.Tensor:
    return torch.randn(like.shape, generator=self.rng, device=like.device, dtype=like.dtype)

  def _sample(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Action, log-probability and clamped log-std."""
    cfg = self.cfg
    with self._autocast:
      mean, log_std = self._actor(graph)
    action, log_prob = squash(mean, log_std, self._draw_normals(mean), cfg.log_std_min, cfg.log_std_max)
    return action, log_prob, log_std.clamp(cfg.log_std_min, cfg.log_std_max)

  def _apply(
    self, optimizer: torch.optim.Optimizer, params: list[torch.Tensor], grads: tuple[torch.Tensor | None, ...]
  ) -> torch.Tensor:
    """All-reduce, clip, step, sac.py:299-303. The actor's update_global never runs, so it keeps None."""
    used = [p for p, g in zip(params, grads) if g is not None]
    grads = [g for g in grads if g is not None]
    if self.grad_hook is not None:
      flat = self.grad_hook(torch.cat([g.flatten() for g in grads]))
      grads = [g.view_as(p) for g, p in zip(flat.split([p.numel() for p in used]), used)]
    for p, g in zip(used, grads):
      p.grad = g
    norm = torch.nn.utils.clip_grad_norm_(used, self.cfg.max_grad_norm)
    optimizer.step()
    return norm

  def _update_critic(self, batch: Batch) -> tuple[torch.Tensor, ...]:
    """sac.py:276-303. The target's entropy uses the temperature before this update changes it."""
    with torch.no_grad():
      next_action, next_log_prob, _ = self._sample(batch.next_graph)
      with self._autocast:
        q1_next, q2_next = self._q1_target(batch.next_graph, next_action), self._q2_target(batch.next_graph, next_action)
      gamma = self.cfg.gamma if batch.discount is None else batch.discount
      target = critic_target(batch.reward, batch.terminal, q1_next, q2_next, next_log_prob, self.log_alpha, gamma)
    with self._autocast:
      q1, q2 = self._q1(batch.graph, batch.action), self._q2(batch.graph, batch.action)
    total, loss1, loss2 = critic_loss(q1, q2, target)
    grads = torch.autograd.grad(total, self._critic_params, allow_unused=True)
    norm = self._apply(self.critic_optimizer, self._critic_params, grads)
    q1 = q1.detach()
    return total.detach(), loss1.detach(), loss2.detach(), norm, target.mean(), target.min(), target.max(), q1.mean(), q1.std(unbiased=False)

  def _update_alpha(self, log_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sac.py:312-322. On every update, also those without an actor step. Not clipped, as the fork."""
    loss = temperature_loss(self.log_alpha, log_prob, self.target_entropy)
    grad, = torch.autograd.grad(loss, [self.log_alpha])
    self.log_alpha.grad = grad if self.grad_hook is None else self.grad_hook(grad.view(1)).view_as(grad)
    self.alpha_optimizer.step()
    return loss.detach(), self.log_alpha.detach().exp()

  def _update_actor(self, batch: Batch, action: torch.Tensor, log_prob: torch.Tensor) -> ActorUpdate:
    """sac.py:335-381: the temperature just changed, and the online critics just updated."""
    with self._autocast:
      q1, q2 = self._q1_frozen(batch.graph, action), self._q2_frozen(batch.graph, action)
    loss = actor_loss(self.log_alpha, log_prob, q1, q2)
    grads = torch.autograd.grad(loss, self._actor_params, allow_unused=True)
    return ActorUpdate(loss.detach(), self._apply(self.actor_optimizer, self._actor_params, grads))

  @torch.no_grad()
  def _update_target(self) -> None:
    soft_update(self._critic_target_params, self._critic_params, self.cfg.tau)  # sac.py:392-393.

  def update(self, batch: Batch) -> tuple[Update, ActorUpdate | None]:
    """One update, sac.py:274-393 and :420, under SACCfg.matmul_precision only."""
    update_policy = self.grad_steps % self.cfg.policy_frequency == 0
    cfg = self.cfg
    if cfg.lr_decay_steps > 0:
      for optimizer, start in ((self.actor_optimizer, cfg.lr_actor), (self.critic_optimizer, cfg.lr_critic)):
        optimizer.param_groups[0]['lr'] = decayed_lr(start, self.grad_steps, cfg.lr_decay_steps, cfg.lr_end_fraction)
    with matmul_precision(self.cfg.matmul_precision):
      critic = self._update_critic(batch)
      # sac.py:307 keeps this graph on every update, and uses it only on an actor step.
      with torch.set_grad_enabled(update_policy):
        action, log_prob, log_std = self._sample(batch.graph)
      alpha_loss, alpha = self._update_alpha(log_prob)
      actor = self._update_actor(batch, action, log_prob) if update_policy else None
      self._update_target()
    self.grad_steps += 1
    self.policy_steps += update_policy
    return Update(*critic, alpha_loss, alpha, -log_prob.detach().mean(), log_std.detach().mean()), actor

  def state_dict(self) -> dict[str, Any]:
    return {
      'grad_steps': self.grad_steps, 'policy_steps': self.policy_steps,
      'actor': self.actor.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
      'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
      'log_alpha': self.log_alpha.detach().clone(),
      'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict(),
      'alpha_optimizer': self.alpha_optimizer.state_dict(), 'rng': self.rng.get_state(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.grad_steps, self.policy_steps = state['grad_steps'], state['policy_steps']
    for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target'):
      getattr(self, name).load_state_dict(state[name], strict=True)
    self.log_alpha.data.copy_(state['log_alpha'].to(self.log_alpha.device))
    self.actor_optimizer.load_state_dict(state['actor_optimizer'])
    self.critic_optimizer.load_state_dict(state['critic_optimizer'])
    self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
    self.rng.set_state(state['rng'].cpu())
