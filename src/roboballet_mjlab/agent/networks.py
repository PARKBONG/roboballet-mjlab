# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Policy and critic graph networks, docs/reports/algo_spec.md section 3.

Every parameter of the original is registered, so counts and checkpoints are exact
(DESIGN 11.9). Forward skips what has no effect on the output: the reward heads, the
policy's update_global, the policy's update_node on non-robot nodes, and repeated
embeddings of target and obstacle nodes, whose inputs take only three values.

Convert an original checkpoint with tests/golden/agent_orbax_dump.py, then:
  uv run python -m roboballet_mjlab.agent.networks FLAT.npz OUT.pt
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch
from torch import nn

from roboballet_mjlab.agent.features import (
  ACTION_DIM, EDGE_DIM, GLOBAL_DIM, NODE_DIM, OBSTACLE_COLUMN, TARGET_COLUMN, Topology,
)
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.structs import Graph

_TRUNCATED_NORMAL_STD = 0.87962566103423978  # jax/_src/nn/initializers.py:322-335
# jax.nn.gelu's constants 0.044715 and sqrt(2 / pi) once they are bf16, as the original's are.
_GELU_CUBIC_BF16 = 0.044677734375
_GELU_SCALE_BF16 = 0.796875

# Without this, a compiled graph drops every rounding between bf16 operations, and the learner
# no longer matches the original or its own eager path. tests/test_agent_bf16.py.
torch._inductor.config.emulate_precision_casts = True


class Dense(nn.Linear):
  """In bf16, rounds the product and then the sum, as flax. F.linear would round them as one."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
      return super().forward(x)
    return x @ self.weight.bfloat16().T + self.bias.bfloat16()


class LayerNorm(nn.LayerNorm):
  """In bf16, the statements of flax's _compute_stats and _normalize: f32, rounded at the end."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
      return super().forward(x)
    x = x.float()
    mean = x.mean(-1, keepdim=True)
    variance = ((x * x).mean(-1, keepdim=True) - mean * mean).clamp_min(0.0)
    scale = torch.sqrt(variance + self.eps).reciprocal() * self.weight
    return ((x - mean) * scale + self.bias).bfloat16()


class GELU(nn.GELU):
  """In bf16, jax.nn.gelu operation by operation, each rounded. Bit-exact to the original on GPU."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
      return super().forward(x)
    return x * (0.5 * (1 + torch.tanh(_GELU_SCALE_BF16 * (x + _GELU_CUBIC_BF16 * (x * x * x)))))


class MLP(nn.Sequential):
  """Dense, LayerNorm and GELU per layer, Dense only on the last, networks.py:117-152."""

  def __init__(self, in_size: int, sizes: list[int], eps: float, last_scale: float = 1.0):
    layers: list[nn.Module] = []
    for i, size in enumerate(sizes):
      is_last = i == len(sizes) - 1
      dense = Dense(in_size, size)
      nn.init.trunc_normal_(dense.weight, 0.0, 1.0, -2.0, 2.0)
      scale = last_scale if is_last else 1.0
      with torch.no_grad():
        dense.weight.mul_(math.sqrt(scale / in_size) / _TRUNCATED_NORMAL_STD)
      nn.init.zeros_(dense.bias)
      layers.append(dense)
      if not is_last:
        layers += [LayerNorm(size, eps=eps), GELU(approximate='tanh')]
      in_size = size
    super().__init__(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # The input cast and the output cast of networks.py:127 and :151-152. Outputs get summed.
    if torch.is_autocast_enabled(x.device.type):
      x = x.to(torch.get_autocast_dtype(x.device.type))
    return super().forward(x).float()


class GraphNet(nn.Module):
  """One round of message passing, edges then nodes then globals, networks.py:166-295."""

  def __init__(self, cfg: ModelCfg, topo: Topology, node_in: int):
    super().__init__()
    h, he, eps = cfg.node_size, cfg.edge_size, cfg.layernorm_eps
    self.embed_node = MLP(node_in, [h] * cfg.node_layers, eps)
    self.embed_edge = MLP(EDGE_DIM, [he] * cfg.edge_embed_layers, eps)
    self.embed_global = MLP(GLOBAL_DIM, [h] * cfg.global_layers, eps)
    self.update_edge = MLP(he + 3 * h, [he] * cfg.edge_update_layers, eps)
    self.update_node = MLP(2 * h + 2 * he, [h] * cfg.node_layers, eps)
    self.update_global = MLP(2 * h + he, [h] * cfg.global_layers, eps)

    self.R, self.T, self.O = topo.num_robots, topo.num_targets, topo.num_obstacles
    self.N, self.E = topo.num_nodes, topo.num_edges
    self.num_robot_edges = topo.num_robot_edges
    self.register_buffer('senders', topo.senders, persistent=False)
    self.register_buffer('receivers', topo.receivers, persistent=False)
    self.register_buffer('robot_edges_by_sender', topo.robot_edges_by_sender, persistent=False)
    # The inputs of a target not done, a target done and an obstacle.
    fixed = torch.zeros(3, node_in)
    fixed[0, TARGET_COLUMN], fixed[1, TARGET_COLUMN], fixed[2, OBSTACLE_COLUMN] = -1.0, 1.0, 1.0
    self.register_buffer('fixed_inputs', fixed, persistent=False)

  def embed_nodes(self, robot_inputs: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    B = nodes.shape[0]
    embedded = self.embed_node(torch.cat([robot_inputs.flatten(0, 1), self.fixed_inputs]))
    robots, fixed = embedded[:-3].view(B, self.R, -1), embedded[-3:]
    done = nodes[:, self.R : self.R + self.T, TARGET_COLUMN : TARGET_COLUMN + 1] > 0
    return torch.cat([robots, torch.where(done, fixed[1], fixed[0]), fixed[2].expand(B, self.O, -1)], 1)

  def update_edges(self, n: torch.Tensor, edges: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    B = n.shape[0]
    return self.update_edge(torch.cat([
      self.embed_edge(edges), n[:, self.senders], n[:, self.receivers], g[:, None].expand(B, self.E, -1),
    ], -1))

  def edge_blocks(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The robot, target and obstacle edges as [B, R, S, He], by receiver then sender.

    Summing over these blocks replaces scatter_add, whose result on a GPU varies from
    run to run in the last bits, which makes updates unrepeatable.
    """
    B, R, He = e.shape[0], self.R, e.shape[-1]
    a, b = self.num_robot_edges, self.num_robot_edges + R * self.T
    return e[:, :a].view(B, R, R - 1, He), e[:, a:b].view(B, R, self.T, He), e[:, b:].view(B, R, self.O, He)


class PolicyNet(nn.Module):
  """Shared tanh head on the robot nodes: [B, R, 7] in [-1, 1], networks.py:361-387."""

  def __init__(self, cfg: ModelCfg, topo: Topology):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM)
    self.policy_head = MLP(cfg.node_size, [*cfg.head_sizes, ACTION_DIM], cfg.layernorm_eps, cfg.head_last_scale)

  def forward(self, graph: Graph) -> torch.Tensor:
    gnn, R = self.gnn, self.gnn.R
    n = gnn.embed_nodes(graph.nodes[:, :R], graph.nodes)
    g = gnn.embed_global(graph.globals)
    e = gnn.update_edges(n, graph.edges, g)
    from_robots, from_targets, from_obstacles = gnn.edge_blocks(e)
    received = from_robots.sum(2) + from_targets.sum(2) + from_obstacles.sum(2)
    sent = e[:, gnn.robot_edges_by_sender].sum(2)
    robots = gnn.update_node(torch.cat([n[:, :R], sent, received, g[:, None].expand(-1, R, -1)], -1))
    return torch.tanh(self.policy_head(robots))


class CriticNet(nn.Module):
  """Q in [-1, 1] from the updated global, actions on robot nodes, networks.py:403-433."""

  def __init__(self, cfg: ModelCfg, topo: Topology):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM + ACTION_DIM)
    head = [*cfg.head_sizes, 1]
    self.qvalue = MLP(cfg.node_size, head, cfg.layernorm_eps, cfg.head_last_scale)
    self.rewards = MLP(cfg.node_size, head, cfg.layernorm_eps, cfg.head_last_scale)  # Loss weight 0.

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
    return torch.tanh(self.qvalue(g)[:, 0])


_CORE_MLPS = ('update_edge', 'update_node', 'update_global')


def flax_names(net: nn.Module, prefix: str) -> dict[str, str]:
  """Maps each torch parameter name to its flax key."""
  names = {}
  for name, mlp in net.named_modules():
    if not isinstance(mlp, MLP):
      continue
    leaf = name.rsplit('.', 1)[-1]
    if name.startswith('gnn.'):
      path = f'GraphNet_0/GraphNetCore_0/{leaf}' if leaf in _CORE_MLPS else f'GraphNet_0/{leaf}'
    else:
      path = leaf
    dense = norm = 0
    for i, layer in enumerate(mlp):
      if isinstance(layer, nn.Linear):
        names[f'{name}.{i}.weight'] = f'{prefix}/{path}/Dense_{dense}/kernel'
        names[f'{name}.{i}.bias'] = f'{prefix}/{path}/Dense_{dense}/bias'
        dense += 1
      elif isinstance(layer, nn.LayerNorm):
        names[f'{name}.{i}.weight'] = f'{prefix}/{path}/LayerNorm_{norm}/scale'
        names[f'{name}.{i}.bias'] = f'{prefix}/{path}/LayerNorm_{norm}/bias'
        norm += 1
  return names


def state_from_flax(names: dict[str, str], flat: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
  """A state dict from flax leaves. Dense kernels are (in, out), torch weights (out, in)."""
  return {
    name: torch.from_numpy(flat[key].T.copy() if key.endswith('kernel') else flat[key].copy())
    for name, key in names.items()
  }


def _reference_nets() -> tuple[PolicyNet, CriticNet]:
  """Nets of any topology, since parameters do not depend on it. Parity model sizes."""
  topo = Topology(EnvCfg(), 'cpu')
  return PolicyNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo)


def convert_flat_checkpoint(flat_path: str, out_path: str) -> None:
  """Converts the output of agent_orbax_dump.py to a torch file. Every leaf must be used."""
  with np.load(flat_path) as z:
    flat = {k: z[k] for k in z.files}
  policy, critic = _reference_nets()
  policy_names = flax_names(policy, 'policy/params')
  q1_names = flax_names(critic, 'critic/params/critic_1')
  q2_names = flax_names(critic, 'critic/params/critic_2')
  used = {*policy_names.values(), *q1_names.values(), *q2_names.values(), 'step'}
  if used != set(flat):
    raise ValueError(f'leaves differ: {sorted(used ^ set(flat))[:8]}')
  torch.save({
    'step': int(flat['step']),
    'policy': state_from_flax(policy_names, flat),
    'q1': state_from_flax(q1_names, flat),
    'q2': state_from_flax(q2_names, flat),
  }, out_path)


def load_checkpoint(path: str, policy: PolicyNet, q1: CriticNet, q2: CriticNet) -> int:
  """Strictly loads a converted checkpoint and returns its gradient step."""
  checkpoint = torch.load(path, map_location='cpu', weights_only=True)
  policy.load_state_dict(checkpoint['policy'], strict=True)
  q1.load_state_dict(checkpoint['q1'], strict=True)
  q2.load_state_dict(checkpoint['q2'], strict=True)
  return checkpoint['step']


def saved_model_cfg(path: str) -> ModelCfg:
  """What built this checkpoint or snapshot. Files from before `model` was saved hold the default."""
  checkpoint = torch.load(path, map_location='cpu', weights_only=True)
  saved = checkpoint.get('model') or checkpoint.get('config', {}).get('model')
  if saved is None:
    return ModelCfg()
  return ModelCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})


def load_policy(path: str, policy: PolicyNet) -> int:
  """Loads the policy of any checkpoint, also a policy snapshot, and returns its gradient step."""
  checkpoint = torch.load(path, map_location='cpu', weights_only=True)
  policy.load_state_dict(checkpoint['policy'], strict=True)
  return checkpoint['step']


if __name__ == '__main__':
  convert_flat_checkpoint(sys.argv[1], sys.argv[2])
