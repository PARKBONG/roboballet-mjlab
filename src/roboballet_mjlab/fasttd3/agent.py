"""FastTD3's update on the RoboBallet GNN. DESIGN 12.1.

Reference: github.com/younggyoseo/FastTD3 at commit 229ed59, MIT licence, files fast_td3/fast_td3.py
and fast_td3/train.py. tests/golden/fasttd3_make_goldens.py holds a verbatim excerpt of it, and
tests/test_fasttd3_agent.py compares everything here with the numbers that excerpt produces.

Learner in bf16, compiled. Acting in f32 at the process's `highest`. FastTD3Cfg.matmul_precision
holds only for the learner call, DESIGN 14.1.
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
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.structs import Graph


class Batch(NamedTuple):
  graph: Graph
  next_graph: Graph
  action: torch.Tensor  # [B, R, 7]
  reward: torch.Tensor  # [B]
  terminal: torch.Tensor  # [B], bool. True on timeouts too: no bootstrap on any terminal, DESIGN 12.1.
  discount: torch.Tensor | None = None  # [B], gamma ** steps, train.py:435, when n_step is above 1.


class CriticStats(NamedTuple):
  """Scalars of one critic update, still on the GPU."""

  loss: torch.Tensor
  grad_norm: torch.Tensor
  target_mean: torch.Tensor  # Expected value of the chosen target distribution.
  target_min: torch.Tensor
  target_max: torch.Tensor
  mass_at_vmin: torch.Tensor  # Mean target mass on the first atom, the sign of a saturated support.
  mass_at_vmax: torch.Tensor


class ActorStats(NamedTuple):
  loss: torch.Tensor
  grad_norm: torch.Tensor


@contextlib.contextmanager
def matmul_precision(value: str) -> Iterator[None]:
  """Sets the process-wide matmul precision for a block, then restores it, also on an exception."""
  previous = torch.get_float32_matmul_precision()
  torch.set_float32_matmul_precision(value)
  try:
    yield
  finally:
    torch.set_float32_matmul_precision(previous)


class DistributionalCritic(nn.Module):
  """agent.networks.CriticNet with logits in place of its tanh and no reward head, fast_td3.py:58-107."""

  def __init__(self, cfg: ModelCfg, topo: Topology, num_atoms: int):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM + ACTION_DIM)
    self.logits = MLP(cfg.node_size, [*cfg.head_sizes, num_atoms], cfg.layernorm_eps, cfg.head_last_scale)

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
    return self.logits(g)


def target_noise(normals: torch.Tensor, sigma: float, clip: float) -> torch.Tensor:
  """The clip acts after scaling, train.py:427-430, the opposite of agent/td3.py."""
  return normals.mul(sigma).clamp(-clip, clip)


def decayed_lr(start: float, step: int, decay_steps: int, end_fraction: float) -> float:
  """CosineAnnealingLR's value at this grad step, held at the end once reached, train.py:302-313."""
  if decay_steps <= 0:
    return start
  end = start * end_fraction
  return end + (start - end) * (1.0 + math.cos(math.pi * min(step, decay_steps) / decay_steps)) / 2.0


def project(
  next_dist: torch.Tensor, rewards: torch.Tensor, bootstrap: torch.Tensor, discount: float | torch.Tensor, support: torch.Tensor,
  v_min: float, v_max: float,
) -> torch.Tensor:
  """Categorical projection onto the support, fast_td3.py:119-154. [B, A] to [B, A].

  Two things differ, and neither changes a value the reference gets right. b is clamped to the
  last atom: at v_max it lands just past it in f32 on 6% of random supports, though on none of
  [-1, 1] and [-2.1, 1], and there the reference writes into the next row, or past the array on the
  last row, where index_add_ raises on the CPU and asserts on the GPU. The sum is over a dense
  [B, A, A] rather than index_add_, whose result on a GPU varies from run to run in the last bits
  (0 of 50 repeats equal at B=2048, as agent/networks.py found for scatter_add). 0.13 ms at B=2048.
  """
  num_atoms = support.shape[0]
  delta_z = (v_max - v_min) / (num_atoms - 1)
  target_z = (rewards[:, None] + bootstrap[:, None] * discount * support).clamp(v_min, v_max)
  b = ((target_z - v_min) / delta_z).clamp(0.0, num_atoms - 1)
  l = b.floor().long()
  u = b.ceil().long()
  is_int = l == u
  l_mask = is_int & (l > 0)
  u_mask = is_int & (l == 0)
  l = torch.where(l_mask, l - 1, l)
  u = torch.where(u_mask, u + 1, u)
  # Row j of the dense form holds atom j's two shares, so nothing is written twice before the sum.
  empty = next_dist.new_zeros(next_dist.shape[0], num_atoms, num_atoms)
  lower = empty.scatter(2, l[..., None], (next_dist * (u.float() - b))[..., None])
  upper = empty.scatter(2, u[..., None], (next_dist * (b - l.float()))[..., None])
  return (lower + upper).sum(1)


def expected_value(probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
  """fast_td3.py:234-236."""
  return torch.sum(probs * support, dim=1)


def clipped_target(proj1: torch.Tensor, proj2: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
  """The whole projected distribution with the lower expected value, train.py:447-455."""
  lower = expected_value(proj1, support)[:, None] < expected_value(proj2, support)[:, None]
  return torch.where(lower, proj1, proj2)


def critic_loss(logits1: torch.Tensor, logits2: torch.Tensor, target1: torch.Tensor, target2: torch.Tensor) -> torch.Tensor:
  """Cross-entropy, summed over both critics, train.py:464-470."""
  loss1 = -torch.sum(target1 * F.log_softmax(logits1, dim=1), dim=1).mean()
  loss2 = -torch.sum(target2 * F.log_softmax(logits2, dim=1), dim=1).mean()
  return loss1 + loss2


def actor_loss(logits1: torch.Tensor, logits2: torch.Tensor, support: torch.Tensor, use_cdq: bool) -> torch.Tensor:
  """On expected values, train.py:503-509."""
  value1 = expected_value(F.softmax(logits1, dim=1), support)
  value2 = expected_value(F.softmax(logits2, dim=1), support)
  value = torch.minimum(value1, value2) if use_cdq else (value1 + value2) / 2.0
  return -value.mean()


def soft_update(targets: list[torch.Tensor], online: list[torch.Tensor], tau: float) -> None:
  """train.py:527-533."""
  torch._foreach_mul_(targets, 1.0 - tau)
  torch._foreach_add_(targets, online, alpha=tau)


class FastTD3:
  """Learner state and the update. No .item() and no host copies in the update path.

  `grad_hook` maps a flat gradient vector to a new one, the place for an all-reduce. `nets`
  replaces the policy and the two critics, for the goldens, where they are the reference's MLPs.
  """

  def __init__(
    self, model_cfg: ModelCfg, cfg: FastTD3Cfg, topo: Topology | None, device: torch.device | str,
    grad_hook: Callable[[torch.Tensor], torch.Tensor] | None = None, noise_seed: int | None = None,
    nets: tuple[nn.Module, nn.Module, nn.Module] | None = None,
  ):
    self.cfg = cfg
    self.grad_hook = grad_hook
    self.grad_steps = 0
    self.policy_steps = 0
    if nets is None:
      with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        nets = (PolicyNet(model_cfg, topo), DistributionalCritic(model_cfg, topo, cfg.num_atoms),
                DistributionalCritic(model_cfg, topo, cfg.num_atoms))
    self.policy, self.q1, self.q2 = (net.to(device) for net in nets)
    # Only the critic has a target, train.py:288-289. The next action comes from the online actor.
    self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
    self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
    # The critics on the same storage, without gradients for their parameters, for the actor
    # loss, as agent/td3.py: a compiled backward computes every gradient of its module.
    self.q1_frozen = copy.deepcopy(self.q1).requires_grad_(False)
    self.q2_frozen = copy.deepcopy(self.q2).requires_grad_(False)
    for frozen, live in zip([*self.q1_frozen.parameters(), *self.q2_frozen.parameters()],
                            [*self.q1.parameters(), *self.q2.parameters()]):
      frozen.data = live.data
    self.rng = torch.Generator(device=device).manual_seed(cfg.seed if noise_seed is None else noise_seed)
    self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms, device=device)

    self._policy_params = list(self.policy.parameters())
    self._critic_params = [*self.q1.parameters(), *self.q2.parameters()]
    self._critic_target_params = [*self.q1_target.parameters(), *self.q2_target.parameters()]
    # train.py:291-300, but fused and with a float LR, where the reference has neither: rounding only, status item 11.
    adamw = dict(weight_decay=cfg.weight_decay, fused=torch.device(device).type == 'cuda')
    self.policy_optimizer = torch.optim.AdamW(self._policy_params, lr=cfg.lr_policy, **adamw)
    self.critic_optimizer = torch.optim.AdamW(self._critic_params, lr=cfg.lr_critic, **adamw)

    self._autocast = torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=cfg.bf16)
    wrap = torch.compile if cfg.compile else lambda module: module
    self._policy = wrap(self.policy)
    self._q1, self._q2 = wrap(self.q1), wrap(self.q2)
    self._q1_target, self._q2_target = wrap(self.q1_target), wrap(self.q2_target)
    self._q1_frozen, self._q2_frozen = wrap(self.q1_frozen), wrap(self.q2_frozen)

  @torch.no_grad()
  def act(self, graph: Graph) -> torch.Tensor:
    """The online policy, f32 and without autocast. It shares its weights with the learner, train.py:282-285."""
    return self._policy(graph)

  def _draw_normals(self, like: torch.Tensor) -> torch.Tensor:
    return torch.randn(like.shape, generator=self.rng, device=like.device, dtype=like.dtype)

  def _apply(
    self, optimizer: torch.optim.Optimizer, params: list[torch.Tensor], grads: tuple[torch.Tensor | None, ...]
  ) -> torch.Tensor:
    """No clipping, train.py:476-483. A parameter without a gradient keeps None, because AdamW
    would decay it under a zero. That is the policy's update_global, which PolicyNet never runs."""
    used = [p for p, g in zip(params, grads) if g is not None]
    grads = [g for g in grads if g is not None]
    if self.grad_hook is not None:
      flat = self.grad_hook(torch.cat([g.flatten() for g in grads]))
      grads = [g.view_as(p) for g, p in zip(flat.split([p.numel() for p in used]), used)]
    for p, g in zip(used, grads):
      p.grad = g
    optimizer.step()
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))

  def _update_critic(self, batch: Batch) -> CriticStats:
    """train.py:427-483."""
    cfg, support = self.cfg, self.support
    with torch.no_grad(), self._autocast:
      noise = target_noise(self._draw_normals(batch.action), cfg.target_noise_sigma, cfg.target_noise_clip)
      next_action = (self._policy(batch.next_graph) + noise).clamp(-1.0, 1.0)
      bootstrap = (~batch.terminal).to(next_action.dtype)
      discount = cfg.gamma if batch.discount is None else batch.discount[:, None]
      proj1 = project(F.softmax(self._q1_target(batch.next_graph, next_action), dim=1), batch.reward, bootstrap,
                      discount, support, cfg.v_min, cfg.v_max)
      proj2 = project(F.softmax(self._q2_target(batch.next_graph, next_action), dim=1), batch.reward, bootstrap,
                      discount, support, cfg.v_min, cfg.v_max)
      if cfg.use_cdq:
        target1 = target2 = clipped_target(proj1, proj2, support)
      else:
        target1, target2 = proj1, proj2
      value = expected_value(target1, support)
    with self._autocast:
      loss = critic_loss(self._q1(batch.graph, batch.action), self._q2(batch.graph, batch.action), target1, target2)
    grads = torch.autograd.grad(loss, self._critic_params, allow_unused=True)
    norm = self._apply(self.critic_optimizer, self._critic_params, grads)
    return CriticStats(loss.detach(), norm, value.mean(), value.min(), value.max(),
                       target1[:, 0].mean(), target1[:, -1].mean())

  def _update_actor(self, batch: Batch) -> ActorStats:
    """train.py:501-522. The critics hold the weights of the critic update just made."""
    with self._autocast:
      action = self._policy(batch.graph)
      loss = actor_loss(self._q1_frozen(batch.graph, action), self._q2_frozen(batch.graph, action),
                        self.support, self.cfg.use_cdq)
    grads = torch.autograd.grad(loss, self._policy_params, allow_unused=True)
    return ActorStats(loss.detach(), self._apply(self.policy_optimizer, self._policy_params, grads))

  @torch.no_grad()
  def _update_target(self) -> None:
    soft_update(self._critic_target_params, self._critic_params, self.cfg.tau)

  def update(self, batch: Batch, update_policy: bool) -> tuple[CriticStats, ActorStats | None]:
    """One update, train.py:668-676, under FastTD3Cfg.matmul_precision only."""
    cfg = self.cfg
    if cfg.lr_decay_steps > 0:
      for optimizer, start in ((self.policy_optimizer, cfg.lr_policy), (self.critic_optimizer, cfg.lr_critic)):
        optimizer.param_groups[0]['lr'] = decayed_lr(start, self.grad_steps, cfg.lr_decay_steps, cfg.lr_end_fraction)
    with matmul_precision(self.cfg.matmul_precision):
      critic = self._update_critic(batch)
      actor = self._update_actor(batch) if update_policy else None
      self._update_target()
    self.grad_steps += 1
    self.policy_steps += update_policy
    return critic, actor

  def state_dict(self) -> dict[str, Any]:
    return {
      'grad_steps': self.grad_steps, 'policy_steps': self.policy_steps,
      'policy': self.policy.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
      'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
      'policy_optimizer': self.policy_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict(),
      'rng': self.rng.get_state(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.grad_steps, self.policy_steps = state['grad_steps'], state['policy_steps']
    for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
      getattr(self, name).load_state_dict(state[name], strict=True)
    self.policy_optimizer.load_state_dict(state['policy_optimizer'])
    self.critic_optimizer.load_state_dict(state['critic_optimizer'])
    self.rng.set_state(state['rng'].cpu())
