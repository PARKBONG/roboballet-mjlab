"""The RoboBallet GNN as rsl-rl's actor and critic, given to it by class_name. DESIGN section 17.

Both subclass rsl_rl.models.MLPModel for its surface towards PPO (distribution, log probability, KL,
no recurrence, no normalisation, mlp_model.py:132-163) and replace its MLP with a graph network on the
compact fields, so a rollout stores those fields and features are built here.

The value comes from the updated global with no tanh. CriticNet's tanh limits Q to [-1, 1], but the
returns rsl-rl regresses to reach about -2.08, 15 / 120 per step of collisions at gamma 0.94, and its
value loss (ppo.py:270-276) is a plain squared error with a clip around the stored value, in the
value's own units. So the limit could only stop the critic from fitting.

The learner defaults to f32 at matmul `high` and not to bf16, unlike the other two agents. PPO compares
the log probability of the learner's forward with the one stored by f32 acting, and bf16's rounding of
the mean would count as policy change in the ratio and in the KL that sets the learning rate. Numbers
in docs/reviews/ppo_status.md. `bf16=True` is there for measuring that.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
import zlib

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from roboballet_mjlab.agent.features import NODE_DIM, Topology, build_graph
from roboballet_mjlab.agent.networks import MLP, GraphNet, PolicyNet
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.ppo.vecenv import tensordict_to_obs
from roboballet_mjlab.structs import Graph


@contextlib.contextmanager
def matmul_precision(value: str) -> Iterator[None]:
  """Sets the process-wide matmul precision for a block, then restores it, also on an exception."""
  previous = torch.get_float32_matmul_precision()
  torch.set_float32_matmul_precision(value)
  try:
    yield
  finally:
    torch.set_float32_matmul_precision(previous)


class ValueNet(nn.Module):
  """agent.networks.CriticNet with no action on the robot nodes, no reward head and no tanh. [B, 1]."""

  def __init__(self, cfg: ModelCfg, topo: Topology):
    super().__init__()
    self.gnn = GraphNet(cfg, topo, NODE_DIM)
    self.value = MLP(cfg.node_size, [*cfg.head_sizes, 1], cfg.layernorm_eps, cfg.head_last_scale)

  def forward(self, graph: Graph) -> torch.Tensor:
    gnn, R = self.gnn, self.gnn.R
    n = gnn.embed_nodes(graph.nodes[:, :R], graph.nodes)
    g = gnn.embed_global(graph.globals)
    e = gnn.update_edges(n, graph.edges, g)
    from_robots, from_targets, from_obstacles = gnn.edge_blocks(e)
    received = from_robots.sum(2) + from_targets.sum(2) + from_obstacles.sum(2)  # Only robots receive.
    received = torch.cat([received, e.new_zeros(e.shape[0], gnn.T + gnn.O, e.shape[-1])], 1)
    sent = torch.cat([e[:, gnn.robot_edges_by_sender].sum(2), from_targets.sum(1), from_obstacles.sum(1)], 1)
    n = gnn.update_node(torch.cat([n, sent, received, g[:, None].expand(-1, gnn.N, -1)], -1))
    g = gnn.update_global(torch.cat([n.sum(1), e.sum(1), g], -1))
    return self.value(g)


class _GraphModel(MLPModel):
  """What PPO calls on a model, with `net` in the place of MLPModel's MLP. The constructor is the one
  ppo.py:433 and :437 call, and it does not run MLPModel's, which would refuse these observations as not 1D."""

  def __init__(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, net: nn.Module, topo: Topology,
    compile: bool, bf16: bool, distribution_cfg: dict | None = None,
  ) -> None:
    nn.Module.__init__(self)
    self.obs_groups, self.obs_dim = obs_groups[obs_set], 0
    self.obs_normalization = False
    self.obs_normalizer = nn.Identity()
    self.distribution = None
    if distribution_cfg is not None:
      self.distribution = resolve_callable(distribution_cfg.pop('class_name'))(output_dim, **distribution_cfg)
    self.topo = topo
    self.bf16 = bf16
    self.net = net
    # In a tuple so that nn.Module does not register the compiled wrapper a second time in parameters().
    self._forward = (torch.compile(net) if compile else net,)

  def graph(self, obs: TensorDict) -> Graph:
    return build_graph(tensordict_to_obs(obs), self.topo)

  def run_net(self, graph: Graph) -> torch.Tensor:
    """bf16 only for the learner's forward: acting and compute_returns run with gradients off, in f32."""
    enabled = self.bf16 and torch.is_grad_enabled()
    with torch.autocast(graph.nodes.device.type, dtype=torch.bfloat16, enabled=enabled):
      return self._forward[0](graph)

  def __repr__(self) -> str:
    return f'{type(self).__name__}: {type(self.net).__name__}, {sum(p.numel() for p in self.parameters()):,} parameters'  # ppo.py:434 prints it.

  def as_jit(self) -> nn.Module:
    raise NotImplementedError('the graph networks have no JIT export')

  def as_onnx(self, verbose: bool) -> nn.Module:
    raise NotImplementedError('the graph networks have no ONNX export')


class BalletActor(_GraphModel):
  """Gaussian on PolicyNet's tanh mean with rsl-rl's learned std, one per joint of every robot.

  `self.net` is a PolicyNet, the module to hand to Evaluator.evaluate and the one the snapshots hold,
  so the evaluated policy is the mean. A sample can leave [-1, 1]: BalletVecEnv clamps it for the env,
  and the rollout keeps the sample, so its log probability is the Gaussian's.
  """

  def __init__(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, env_cfg: EnvCfg,
    model_cfg: ModelCfg, seed: int, compile: bool, bf16: bool, distribution_cfg: dict,
  ) -> None:
    topo = Topology(env_cfg, obs['q'].device)
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(seed)  # The first draws of the seed: the same initial policy as TD3 and FastTD3, not SAC, whose actor is another net.
      net = PolicyNet(model_cfg, topo)
    super().__init__(obs, obs_groups, obs_set, output_dim, net, topo, compile, bf16, distribution_cfg)
    self.kl_sum = torch.zeros((), device=obs['q'].device)
    self.kl_count = 0

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    mean = self.run_net(self.graph(obs)).flatten(1)
    if stochastic_output:
      self.distribution.update(mean)
      return self.distribution.sample()
    return mean

  def get_kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """As MLPModel, and kept for the log, since the adaptive learning rate is what this drives."""
    kl = super().get_kl_divergence(old_params, new_params)
    self.kl_sum += kl.mean()
    self.kl_count += 1
    return kl


class BalletCritic(_GraphModel):
  def __init__(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, env_cfg: EnvCfg,
    model_cfg: ModelCfg, seed: int, compile: bool, bf16: bool,
  ) -> None:
    topo = Topology(env_cfg, obs['q'].device)
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(zlib.crc32(f'{seed}/critic'.encode()))
      net = ValueNet(model_cfg, topo)
    super().__init__(obs, obs_groups, obs_set, output_dim, net, topo, compile, bf16)

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    return self.run_net(self.graph(obs))


class BalletPPO(PPO):
  """rsl-rl's PPO, unchanged but for two things: the learner's matmul precision holds only inside update,
  DESIGN 14.1, and the last losses are kept, since ppo/train.py gives the runner no logger of its own.

  Under the adaptive schedule the first of those shows. The first mini-batch of an iteration reads a KL of rounding
  rather than the exact 0 of equal precisions, so ppo.py:248's guard passes and the rate rises by 1.5 where stock rsl-rl
  holds it. `matmul_precision='highest'` removes that, and under `schedule='fixed'` no KL is computed."""

  def __init__(self, *args, matmul_precision: str = 'high', **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.matmul_precision = matmul_precision
    self.last_losses: dict[str, float] = {}

  def update(self) -> dict[str, float]:
    with matmul_precision(self.matmul_precision):
      self.last_losses = super().update()
    return self.last_losses
