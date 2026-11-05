"""The FastTD3 update against numbers from the reference at commit 229ed59.

tests/golden/fasttd3_make_goldens.py holds the verbatim excerpt and writes fasttd3_*.npz. Nothing
here imports from a clone. The tiny networks below are the reference's architecture, so that its
whole update can be followed step by step, with weights and normals from the stored trace.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import MLP, CriticNet, PolicyNet
from roboballet_mjlab.config import EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.fasttd3.agent import (
  Batch, decayed_lr, DistributionalCritic, FastTD3, actor_loss, clipped_target, critic_loss, expected_value, matmul_precision, project,
  soft_update, target_noise,
)
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.fasttd3.trainer import policy_update_due, updates_due
from roboballet_mjlab.structs import Graph

GOLDEN = Path(__file__).resolve().parent / 'golden'
DEVICES = ['cpu', 'cuda:0']
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
ENV = EnvCfg(world=WorldCfg(num_robots=2, num_obstacles=3), num_targets=2)
# Float32 against float32 on another device or sum order. On the GPU the division that gives b rounds
# differently by up to 1 ulp of 100, which is 7.6e-6 of mass on an atom. Far below what any mutant moves.
PROJECTION_TOLERANCE = {'cpu': 3e-6, 'cuda:0': 2e-5}
LOSS_TOLERANCE = 1e-5
# Observed 4.5e-8 on the CPU and 5.3e-6 on the GPU, where one update moves weights by about 3e-4.
PARAMETER_TOLERANCE = {'cpu': 5e-7, 'cuda:0': 1.5e-5}


def load(name: str):
  return np.load(GOLDEN / f'fasttd3_{name}.npz')


class RefQ(nn.Module):
  """fast_td3.py:75-97, 103-107."""

  def __init__(self, n_obs, n_act, atoms, hidden):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_obs + n_act, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
    self.fc_head = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.ReLU(), nn.Linear(hidden // 4, atoms))

  def forward(self, obs, action):
    return self.fc_head(self.net(torch.cat([obs, action], 1)))


class RefActor(nn.Module):
  """fast_td3.py:258-303."""

  def __init__(self, n_obs, n_act, hidden):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
    self.fc_head = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.ReLU())
    self.fc_mu = nn.Sequential(nn.Linear(hidden // 4, n_act), nn.Tanh())

  def forward(self, obs):
    return self.fc_mu(self.fc_head(self.net(obs)))


def set_flat(module: nn.Module, values: np.ndarray) -> None:
  values = torch.as_tensor(values)
  at = 0
  for p in module.parameters():
    p.data.copy_(values[at : at + p.numel()].view_as(p))
    at += p.numel()
  assert at == len(values)


def flat(module: nn.Module) -> torch.Tensor:
  return torch.cat([p.detach().flatten() for p in module.parameters()])


def trace_agent(z, name: str, device: str, bf16: bool = False, matmul: str = 'highest') -> FastTD3:
  args = lambda key: z[f'{name}_arg_{key}'].item()
  n_obs, n_act = z[f'{name}_batch_observations'].shape[-1], z[f'{name}_batch_actions'].shape[-1]
  hidden, atoms = int(z[f'{name}_hidden']), int(args('num_atoms'))
  actor, q1, q2 = RefActor(n_obs, n_act, hidden), RefQ(n_obs, n_act, atoms, hidden), RefQ(n_obs, n_act, atoms, hidden)
  set_flat(actor, z[f'{name}_init_actor'])
  set_flat(q1, z[f'{name}_init_q1'])
  set_flat(q2, z[f'{name}_init_q2'])
  cfg = FastTD3Cfg(
    gamma=args('gamma'), tau=args('tau'), lr_policy=args('actor_learning_rate'), lr_critic=args('critic_learning_rate'),
    weight_decay=args('weight_decay'), target_noise_sigma=args('policy_noise'), target_noise_clip=args('noise_clip'),
    num_atoms=atoms, v_min=args('v_min'), v_max=args('v_max'), use_cdq=args('use_cdq'), num_updates=int(args('num_updates')),
    policy_frequency=int(args('policy_frequency')), compile=False, bf16=bf16, matmul_precision=matmul)
  return FastTD3(None, cfg, None, device, nets=(actor, q1, q2))


def trace_batch(z, name: str, k: int, device: str) -> Batch:
  at = lambda key: torch.as_tensor(z[f'{name}_batch_{key}'][k], device=device)
  return Batch(at('observations'), at('next_observations'), at('actions'), at('rewards'), at('dones').bool())


# Projection.


@pytest.mark.parametrize('device', DEVICES)
def test_projection_matches_the_reference_on_every_case(device):
  z = load('projection')
  for case in range(int(z['num_cases'])):
    at = lambda key: torch.as_tensor(z[f'case{case}_{key}'], device=device)
    v_min, v_max, atoms, gamma = z[f'case{case}_params'].tolist()
    next_dist = F.softmax(at('logits'), dim=1)
    got = project(next_dist, at('rewards'), at('bootstrap'), gamma, at('support'), v_min, v_max)
    error = (got - at('proj')).abs().max().item()
    assert error < PROJECTION_TOLERANCE[device], f'case {case}, support [{v_min}, {v_max}], {int(atoms)} atoms, discount {gamma}: {error}'


def test_projection_edge_cases_land_where_they_must():
  """Independent of the golden: the same edge cases by hand."""
  support = torch.linspace(-1.0, 1.0, 101)
  uniform = torch.full((1, 101), 1 / 101)
  one = torch.ones(1)
  zero = torch.zeros(1)
  go = lambda reward, bootstrap, discount=0.94: project(uniform, torch.tensor([reward]), bootstrap, discount, support, -1.0, 1.0)[0]
  # Terminal on an interior atom, on each end and past each end: all mass on one atom.
  for reward, atom in ((0.0, 50), (0.2, 60), (-1.0, 0), (1.0, 100), (5.0, 100), (-5.0, 0)):
    p = go(reward, zero)
    assert p[atom].item() == pytest.approx(1.0, abs=1e-5), reward
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
  # Terminal between two atoms: shared in proportion.
  p = go(0.005, zero)
  assert p[50].item() == pytest.approx(0.75, abs=1e-4) and p[51].item() == pytest.approx(0.25, abs=1e-4)
  # Everything past the support with a bootstrap.
  assert go(5.0, one)[100].item() == pytest.approx(1.0, abs=1e-5)
  assert go(-5.0, one)[0].item() == pytest.approx(1.0, abs=1e-5)
  # Discount 1, reward 0: the identity, so every b is an integer, the branch of fast_td3.py:131-136.
  assert (go(0.0, one, 1.0) - uniform[0]).abs().max().item() < 1e-5
  # Atoms past the top end pile on the last atom.
  p = go(0.5, one, 1.0)
  assert p[100].item() == pytest.approx(26 / 101, abs=1e-5)


def test_projection_preserves_the_mean_inside_the_support():
  torch.manual_seed(0)
  support = torch.linspace(-1.0, 1.0, 101)
  dist = torch.softmax(torch.randn(512, 101), 1)
  rewards = torch.rand(512) * 0.02
  proj = project(dist, rewards, torch.ones(512), 0.94, support, -1.0, 1.0)
  assert (proj.sum(1) - 1).abs().max().item() < 1e-5
  raw = expected_value(dist, support) * 0.94 + rewards
  assert (expected_value(proj, support) - raw).abs().max().item() < 1e-5


def test_projection_on_a_support_where_float32_passes_the_last_atom():
  """The reference would write into the next row here. Each row must keep its own mass."""
  found = None
  generator = torch.Generator().manual_seed(0)
  for _ in range(2000):
    v_min, v_max = sorted((torch.rand(2, generator=generator) * 20 - 10).tolist())
    atoms = int(torch.randint(11, 302, (1,), generator=generator))
    z = torch.tensor([v_max], dtype=torch.float32).clamp(v_min, v_max)
    if ((z - v_min) / ((v_max - v_min) / (atoms - 1))).item() > atoms - 1:
      found = (v_min, v_max, atoms)
      break
  assert found is not None
  v_min, v_max, atoms = found
  support = torch.linspace(v_min, v_max, atoms)
  n = 4
  dist = torch.softmax(torch.randn(n, atoms), 1)
  proj = project(dist, torch.tensor([v_max + 1, 0.0, v_max + 1, 0.0]), torch.tensor([0.0, 1.0, 0.0, 1.0]), 0.94, support, v_min, v_max)
  assert (proj.sum(1) - 1).abs().max().item() < 1e-5
  assert proj[0, -1].item() == pytest.approx(1.0, abs=1e-5)
  assert proj[1, 0].item() < 1.0


def test_projection_is_repeatable_on_the_gpu():
  """What index_add_ is not, and why the dense sum."""
  torch.manual_seed(0)
  device = 'cuda:0'
  support = torch.linspace(-1.0, 1.0, 101, device=device)
  dist = torch.softmax(torch.randn(2048, 101, device=device), 1)
  rewards = torch.rand(2048, device=device) * 0.2
  bootstrap = torch.ones(2048, device=device)
  first = project(dist, rewards, bootstrap, 0.94, support, -1.0, 1.0)
  for _ in range(20):
    assert torch.equal(first, project(dist, rewards, bootstrap, 0.94, support, -1.0, 1.0))


# Clipped double Q and the losses.


@pytest.mark.parametrize('device', DEVICES)
def test_clipped_double_q_choice_matches_the_reference(device):
  z = load('losses')
  at = lambda key: torch.as_tensor(z[key], device=device)
  chosen = clipped_target(at('proj1'), at('proj2'), at('support'))
  assert torch.equal(chosen, at('cdq_dist1'))
  value1, value2 = at('value1'), at('value2')
  assert (value1 < value2).any() and (value1 > value2).any() and (value1 == value2).any()
  # On a tie of values the reference takes the second, which shows only when the distributions differ.
  tied = (value1 == value2) & (at('proj1') != at('proj2')).any(1)
  assert tied.any() and torch.equal(chosen[tied], at('proj2')[tied])
  # Per the whole distribution, by the value and not by atoms.
  lower = value1 < value2
  assert torch.equal(chosen[lower], at('proj1')[lower])
  assert torch.equal(chosen[~lower], at('proj2')[~lower])


@pytest.mark.parametrize('device', DEVICES)
def test_both_losses_match_the_reference(device):
  z = load('losses')
  at = lambda key: torch.as_tensor(z[key], device=device)
  for name, use_cdq in (('cdq', True), ('no_cdq', False)):
    got = critic_loss(at('logits1'), at('logits2'), at(f'{name}_dist1'), at(f'{name}_dist2'))
    assert got.item() == pytest.approx(z[f'{name}_qf_loss'].item(), abs=LOSS_TOLERANCE)
    got = actor_loss(at('logits1'), at('logits2'), at('support'), use_cdq)
    assert got.item() == pytest.approx(z[f'{name}_actor_loss'].item(), abs=LOSS_TOLERANCE)


# Noise.


@pytest.mark.parametrize('device', DEVICES)
def test_target_noise_clips_after_scaling_as_the_reference(device):
  z = load('noise')
  normals = torch.as_tensor(z['normals'], device=device)
  for k in range(int(z['num_sizes'])):
    sigma, clip = z[f'clipped_{k}_args'].tolist()
    assert torch.equal(target_noise(normals, sigma, clip), torch.as_tensor(z[f'clipped_{k}'], device=device)), (sigma, clip)
  # At sigma 2 and clip 0.5 a normal of 0.3 becomes 0.5, where the original's order gives 0.6.
  assert target_noise(torch.tensor([0.3]), 2.0, 0.5).item() == 0.5


# The whole update.


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', ['default', 'no_cdq', 'loud', 'edge'])
def test_every_update_follows_the_reference_trace(device, name):
  """Weights of the actor, both critics and both targets after each update, and both losses."""
  z = load('update')
  follow_trace(z, name, trace_agent(z, name, device), device)


def test_the_bf16_update_follows_the_reference_on_the_gpu():
  """The learner's real path, eager. A golden of the GPU alone: under CPU autocast the reference's projection raises.
  Its largest weight error is 4.2e-7, where running this agent in f32 instead gives 1.6e-3."""
  z = load('update_bf16')
  follow_trace(z, 'bf16', trace_agent(z, 'bf16', 'cuda:0', bf16=True, matmul='high'), 'cuda:0')
  with pytest.raises(AssertionError):
    follow_trace(z, 'bf16', trace_agent(z, 'bf16', 'cuda:0', bf16=False, matmul='high'), 'cuda:0')


def follow_trace(z, name: str, agent: FastTD3, device: str) -> None:
  normals = iter(torch.as_tensor(z[f'{name}_normals'], device=device))
  agent._draw_normals = lambda like: next(normals)
  actor_losses = iter(z[f'{name}_actor_loss'].tolist())
  rounds = len(z[f'{name}_qf_loss']) // agent.cfg.num_updates
  k = 0
  for global_step in range(agent.cfg.learning_starts + 1, agent.cfg.learning_starts + 1 + rounds):
    for i in range(agent.cfg.num_updates):
      critic, actor = agent.update(trace_batch(z, name, k, device), policy_update_due(i, global_step, agent.cfg))
      assert critic.loss.item() == pytest.approx(z[f'{name}_qf_loss'][k].item(), abs=LOSS_TOLERANCE), f'update {k}'
      assert (actor is not None) == (i == 1), f'update {k}'
      if actor is not None:
        assert actor.loss.item() == pytest.approx(next(actor_losses), abs=LOSS_TOLERANCE), f'update {k}'
      for key, module in (('actor', agent.policy), ('q1', agent.q1), ('q2', agent.q2),
                          ('q1_target', agent.q1_target), ('q2_target', agent.q2_target)):
        expected = torch.as_tensor(z[f'{name}_{key}'][k], device=device)
        error = (flat(module) - expected).abs().max().item()
        assert error < PARAMETER_TOLERANCE[device], f'update {k}, {key}: {error}'
      k += 1
  assert agent.grad_steps == 10 and agent.policy_steps == 5


def test_the_trace_can_tell_a_changed_weight():
  """The tolerance is far below what one real update moves, so the comparison means something."""
  z = load('update')
  assert np.abs(z['default_q1'][0] - z['default_init_q1']).max() > 10 * PARAMETER_TOLERANCE['cuda:0']
  assert np.abs(z['default_actor'][1] - z['default_init_actor']).max() > 10 * PARAMETER_TOLERANCE['cuda:0']


def test_no_target_actor_and_the_next_action_uses_the_online_weights():
  z = load('update')
  agent = trace_agent(z, 'default', 'cpu')
  assert not hasattr(agent, 'policy_target')
  seen = []
  policy = agent._policy
  agent._policy = lambda obs: (seen.append(obs), policy(obs))[1]
  batch = trace_batch(z, 'default', 0, 'cpu')
  agent.update(batch, False)
  assert len(seen) == 1 and seen[0] is batch.next_graph
  # Next actions must follow the policy's current weights: a jump in them moves the target.
  before = flat(agent.q1).clone()
  losses = [agent.update(batch, False)[0].loss.item()]
  agent2 = trace_agent(z, 'default', 'cpu')
  agent2.update(batch, False)
  for p in agent2.policy.parameters():
    p.data.add_(1.0)
  losses.append(agent2.update(batch, False)[0].loss.item())
  assert losses[0] != losses[1]
  assert not torch.equal(before, flat(agent.q1))


def test_target_update_is_critic_only_with_tau():
  z = load('update')
  agent = trace_agent(z, 'default', 'cpu')
  online = torch.cat([flat(agent.q1), flat(agent.q2)])
  target = torch.cat([flat(agent.q1_target), flat(agent.q2_target)])
  assert torch.equal(online, target), 'target starts as the online critic, train.py:288-289'
  for p in agent.q1.parameters():
    p.data.add_(1.0)
  policy_before = flat(agent.policy).clone()
  before = flat(agent.q1_target).clone()
  agent._update_target()
  assert (flat(agent.q1_target) - (before + 0.1)).abs().max().item() < 1e-6
  assert torch.equal(flat(agent.policy), policy_before)
  # Pure function on plain tensors.
  a, b = [torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 5.0])]
  soft_update(a, b, 0.1)
  assert a[0].tolist() == pytest.approx([1.2, 2.3])


@pytest.mark.parametrize('device', DEVICES)
def test_adamw_matches_the_reference(device):
  z = load('adamw')

  class Holder(nn.Module):
    def __init__(self):
      super().__init__()
      self.a = nn.Parameter(torch.as_tensor(z['before_0']).clone())
      self.b = nn.Parameter(torch.as_tensor(z['before_1']).clone())

  cfg = FastTD3Cfg(compile=False, bf16=False, matmul_precision='highest')
  agent = FastTD3(None, cfg, None, device, nets=(Holder(), Holder(), Holder()))
  group = agent.critic_optimizer.param_groups[0]
  assert isinstance(agent.critic_optimizer, torch.optim.AdamW) and isinstance(agent.policy_optimizer, torch.optim.AdamW)
  assert (group['lr'], group['weight_decay'], group['betas'], group['eps']) == (3e-4, 0.1, (0.9, 0.999), 1e-8)
  assert agent.policy_optimizer.param_groups[0]['lr'] == 3e-4 and agent.policy_optimizer.param_groups[0]['weight_decay'] == 0.1
  params = list(agent.q1.parameters())
  for step in range(3):
    grads = [torch.as_tensor(z[f'grad_{step}_{k}'], device=device) for k in range(2)]
    agent._apply(agent.critic_optimizer, agent._critic_params, (*grads, None, None))
    for k, p in enumerate(params):
      error = (p - torch.as_tensor(z[f'after_{step}_{k}'], device=device)).abs().max().item()
      assert error < 1e-6, f'step {step}, parameter {k}: {error}'
  assert all(p.grad is None for p in agent.q2.parameters())


def test_no_gradient_clipping():
  cfg = FastTD3Cfg(compile=False, bf16=False, matmul_precision='highest', weight_decay=0.0, lr_critic=1.0)

  class Holder(nn.Module):
    def __init__(self):
      super().__init__()
      self.a = nn.Parameter(torch.zeros(1))

  agent = FastTD3(None, cfg, None, 'cpu', nets=(Holder(), Holder(), Holder()))
  norm = agent._apply(agent.critic_optimizer, agent._critic_params, (torch.tensor([1e6]), torch.tensor([0.0])))
  assert norm.item() == pytest.approx(1e6)
  assert agent.q1.a.grad.item() == 1e6


def test_defaults_are_the_reference_and_the_design_table():
  z = load('adamw')
  value = lambda key: z[f'default_{key}'].item()
  cfg = FastTD3Cfg()
  assert cfg.lr_critic == value('critic_learning_rate') and cfg.lr_policy == value('actor_learning_rate')
  assert cfg.tau == value('tau') and cfg.weight_decay == value('weight_decay')
  assert cfg.target_noise_sigma == value('policy_noise') and cfg.target_noise_clip == value('noise_clip')
  assert cfg.std_min == value('std_min') and cfg.std_max == value('std_max')
  assert cfg.learning_starts == value('learning_starts') and cfg.num_updates == value('num_updates')
  assert cfg.policy_frequency == value('policy_frequency') and cfg.num_atoms == value('num_atoms')
  assert cfg.use_cdq == value('use_cdq') and cfg.bf16 == value('amp')
  # The deviations of DESIGN 12.1.
  assert (cfg.gamma, cfg.v_min, cfg.v_max, cfg.batch_size, cfg.num_worlds) == (0.94, -1.0, 1.0, 2048, 1024)
  assert cfg.replay_max_size == 2_000_000 and cfg.hindsight


# Schedule, on the pure functions. tests/test_fasttd3_trainer.py counts the same on the real loop.


@pytest.mark.parametrize('name', ['default', 'one_update', 'three_updates', 'frequency_three'])
def test_schedule_matches_the_reference(name):
  z = load('schedule')
  learning_starts, num_updates, policy_frequency = z[f'{name}_args'].tolist()
  cfg = FastTD3Cfg(learning_starts=learning_starts, num_updates=num_updates, policy_frequency=policy_frequency)
  calls = []
  for global_step in range(40):
    if updates_due(global_step, cfg):
      for i in range(cfg.num_updates):
        calls += [(global_step, 0), (global_step, 1)]
        if policy_update_due(i, global_step, cfg):
          calls.append((global_step, 2))
        calls.append((global_step, 3))
  assert calls == [tuple(row) for row in z[name].tolist()]


def test_the_default_schedule_in_numbers():
  z = load('schedule')['default']
  assert z[0].tolist() == [11, 0], 'the first update follows the 12th env step, train.py:643'
  per_round = (z[:, 1] == 1).sum() / (40 - 11)
  assert per_round == 2.0 and (z[:, 1] == 2).sum() / (40 - 11) == 1.0


# The network.


def small_graph(batch: int = 4, device: str = 'cuda:0') -> tuple[Graph, torch.Tensor, Topology]:
  topo = Topology(ENV, device)
  torch.manual_seed(1)
  graph = Graph(torch.rand(batch, topo.num_nodes, 17, device=device), torch.rand(batch, topo.num_edges, 30, device=device),
                torch.rand(batch, 2, device=device))
  return graph, torch.rand(batch, ENV.world.num_robots, 7, device=device) * 2 - 1, topo


def real_batch(batch: int = 4, device: str = 'cuda:0') -> tuple[Batch, Topology]:
  graph, action, topo = small_graph(batch, device)
  next_graph = Graph(*[x.flip(0) for x in graph])
  reward = torch.rand(batch, device=device) * 0.2
  terminal = torch.zeros(batch, dtype=torch.bool, device=device)
  return Batch(graph, next_graph, action, reward, terminal), topo


def test_the_critic_is_the_gnn_with_logits_and_no_tanh_or_reward_head():
  graph, action, topo = small_graph(device='cpu')
  critic = DistributionalCritic(SMALL, topo, 101)
  logits = critic(graph, action)
  assert logits.shape == (4, 101)
  assert isinstance(critic.logits, MLP) and not hasattr(critic, 'rewards') and not hasattr(critic, 'qvalue')
  with torch.no_grad():
    critic.logits[-1].weight.mul_(1e3)
  assert critic(graph, action).abs().max().item() > 1.0, 'a tanh would bound these'
  # Everything else has the sizes of the original's critic.
  original = CriticNet(SMALL, topo)
  mine = {k: v.shape for k, v in critic.named_parameters() if k.startswith('gnn.')}
  theirs = {k: v.shape for k, v in original.named_parameters() if k.startswith('gnn.')}
  assert mine == theirs


def test_the_critic_starts_near_zero_value():
  graph, action, topo = small_graph(device='cpu')
  critic = DistributionalCritic(ModelCfg(), topo, 101)
  support = torch.linspace(-1.0, 1.0, 101)
  value = expected_value(F.softmax(critic(graph, action), 1), support)
  assert value.abs().max().item() < 0.1


def test_the_actor_is_policynet_unchanged():
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False), small_graph(device='cpu')[2], 'cpu')
  assert type(agent.policy) is PolicyNet


def test_parameters_without_a_gradient_stay_put_under_weight_decay():
  """PolicyNet never runs update_global, and AdamW must not decay it, with or without the hook."""
  batch, topo = real_batch()
  for hook in (None, lambda flat: flat):
    agent = FastTD3(SMALL, FastTD3Cfg(compile=False, weight_decay=0.5, lr_policy=0.1), topo, 'cuda:0', grad_hook=hook)
    before = flat(agent.policy.gnn.update_global).clone()
    others = flat(agent.policy.policy_head).clone()
    agent.update(batch, True)
    assert torch.equal(flat(agent.policy.gnn.update_global), before)
    assert not torch.equal(flat(agent.policy.policy_head), others)


def test_the_hook_sees_one_flat_vector_and_its_result_is_applied():
  batch, topo = real_batch()
  seen = []

  def hook(vector):
    seen.append(vector.clone())
    return torch.zeros_like(vector)

  cfg = FastTD3Cfg(compile=False, weight_decay=0.0, matmul_precision='highest')
  agent = FastTD3(SMALL, cfg, topo, 'cuda:0', grad_hook=hook)
  before = [flat(agent.q1).clone(), flat(agent.policy).clone()]
  agent.update(batch, True)
  assert len(seen) == 2 and seen[0].dim() == 1 and seen[0].abs().max().item() > 0
  assert len(seen[0]) == sum(p.numel() for p in agent._critic_params)
  # A zero gradient and no decay: Adam leaves everything as it was.
  assert torch.equal(flat(agent.q1), before[0]) and torch.equal(flat(agent.policy), before[1])


# Float modes, DESIGN 14.1.


def spy_on(module: nn.Module, log: list) -> None:
  def record(_module, _inputs):
    log.append((torch.get_float32_matmul_precision(), torch.is_autocast_enabled('cuda'),
                torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None))

  module.register_forward_pre_hook(record)


def test_learner_runs_in_bf16_at_the_configured_precision_and_acting_in_f32_at_highest():
  batch, topo = real_batch()
  torch.set_float32_matmul_precision('highest')
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False, bf16=True, matmul_precision='high'), topo, 'cuda:0')
  log: list = []
  for net in (agent.policy, agent.q1, agent.q2, agent.q1_target, agent.q2_target, agent.q1_frozen, agent.q2_frozen):
    spy_on(net, log)
  agent.update(batch, True)
  assert len(log) == 1 + 2 + 2 + 1 + 2, 'next action, targets, online critics, actor, frozen critics'
  assert all(entry == ('high', True, torch.bfloat16) for entry in log), log
  assert torch.get_float32_matmul_precision() == 'highest'
  log.clear()
  action = agent.act(batch.graph)
  assert action.dtype == torch.float32 and not action.requires_grad
  assert log == [('highest', False, None)]


def test_learner_outputs_are_float32_and_gradients_reach_every_used_parameter():
  batch, topo = real_batch()
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False, bf16=True), topo, 'cuda:0')
  critic, actor = agent.update(batch, True)
  assert critic.loss.dtype == torch.float32 and actor.loss.dtype == torch.float32
  assert all(p.dtype == torch.float32 for p in agent._critic_params)
  assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in agent._critic_params)
  grads_missing = {name for name, p in agent.policy.named_parameters() if p.grad is None}
  assert {name.split('.')[1] for name in grads_missing} == {'update_global'}


def test_precision_is_restored_through_an_exception():
  batch, topo = real_batch()
  torch.set_float32_matmul_precision('highest')
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False, matmul_precision='high'), topo, 'cuda:0')

  def boom(_batch):
    assert torch.get_float32_matmul_precision() == 'high'
    raise RuntimeError('boom')

  agent._update_critic = boom
  with pytest.raises(RuntimeError, match='boom'):
    agent.update(batch, False)
  assert torch.get_float32_matmul_precision() == 'highest'
  with pytest.raises(RuntimeError, match='boom'), matmul_precision('high'):
    raise RuntimeError('boom')
  assert torch.get_float32_matmul_precision() == 'highest'


def test_init_leaves_the_global_precision_alone():
  torch.set_float32_matmul_precision('highest')
  FastTD3(SMALL, FastTD3Cfg(compile=False, matmul_precision='high'), small_graph(device='cpu')[2], 'cpu')
  assert torch.get_float32_matmul_precision() == 'highest'


def test_learning_rate_follows_a_cosine_and_then_holds():
  assert decayed_lr(3e-4, 500, 0, 0.1) == 3e-4
  assert decayed_lr(3e-4, 0, 1000, 0.1) == pytest.approx(3e-4)
  assert decayed_lr(3e-4, 500, 1000, 0.1) == pytest.approx(1.65e-4)
  assert decayed_lr(3e-4, 1000, 1000, 0.1) == pytest.approx(3e-5)
  assert decayed_lr(3e-4, 5000, 1000, 0.1) == pytest.approx(3e-5)
  optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=3e-4)
  schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=3e-5)
  for step in range(1000):
    assert decayed_lr(3e-4, step, 1000, 0.1) == pytest.approx(schedule.get_last_lr()[0], rel=1e-9)
    optimizer.step()
    schedule.step()


def test_update_sets_the_rate_of_this_grad_step_on_both_optimizers():
  graph, action, topo = small_graph(device='cpu')
  reward, terminal = torch.zeros(action.shape[0]), torch.zeros(action.shape[0], dtype=torch.bool)
  batch = Batch(graph, graph, action, reward, terminal)
  cfg = FastTD3Cfg(compile=False, bf16=False, lr_policy=2e-4, lr_critic=4e-4, lr_decay_steps=10, lr_end_fraction=0.5)
  agent = FastTD3(SMALL, cfg, topo, 'cpu')
  agent.grad_steps = 10
  agent.update(batch, True)
  assert agent.policy_optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
  assert agent.critic_optimizer.param_groups[0]['lr'] == pytest.approx(2e-4)
  fixed = FastTD3(SMALL, FastTD3Cfg(compile=False, bf16=False, lr_policy=2e-4), topo, 'cpu')
  fixed.grad_steps = 10
  fixed.update(batch, True)
  assert fixed.policy_optimizer.param_groups[0]['lr'] == 2e-4


def test_per_sample_discount_matches_a_scalar_discount():
  torch.manual_seed(0)
  support = torch.linspace(-1.0, 1.0, 101, device='cuda:0')
  dist = torch.softmax(torch.randn(64, 101, device='cuda:0'), dim=1)
  rewards, bootstrap = torch.rand(64, device='cuda:0') - 0.5, (torch.rand(64, device='cuda:0') > 0.2).float()
  for discount in (0.94, 0.94**3):
    scalar = project(dist, rewards, bootstrap, discount, support, -1.0, 1.0)
    tensor = project(dist, rewards, bootstrap, torch.full((64, 1), discount, device='cuda:0'), support, -1.0, 1.0)
    assert torch.allclose(scalar, tensor, atol=1e-5)
  mixed = torch.tensor([0.94, 0.94**3] * 32, device='cuda:0')
  both = project(dist, rewards, bootstrap, mixed[:, None], support, -1.0, 1.0)
  assert torch.allclose(both[0], project(dist, rewards, bootstrap, 0.94, support, -1.0, 1.0)[0], atol=1e-5)
  assert torch.allclose(both[1], project(dist, rewards, bootstrap, 0.94**3, support, -1.0, 1.0)[1], atol=1e-5)


def test_compiled_bf16_update_runs_and_agrees_with_eager():
  """The learner compiled, as in training, against the same update without torch.compile, at the sweep's widths.

  By rms, because under Adam a weight with a near-zero gradient moves by the LR with the sign of its noise: 4 steps at
  3e-4 move a weight by at most 1.2e-3, whatever the arithmetic. Compiled and eager bf16 are not one arithmetic, here or
  in the reference. Each bound sits between this path and the same path with emulate_precision_casts off, cold caches:
  q1 1.9e-5 against 4.1e-5, policy 1.3e-5 against 3.1e-5, and the bf16 action on one set of weights 8.4e-5 against 5.1e-4.
  """
  batch, topo = real_batch(16)
  torch.set_float32_matmul_precision('highest')
  model = ModelCfg(node_layers=3, node_size=256, edge_embed_layers=3, edge_update_layers=3, edge_size=128, global_layers=3)
  agents, losses = {}, {}
  for compiled in (False, True):
    agent = agents[compiled] = FastTD3(model, FastTD3Cfg(compile=compiled, bf16=True, seed=3), topo, 'cuda:0', noise_seed=5)
    losses[compiled] = []
    for i in range(4):
      critic, _ = agent.update(batch, i % 2 == 1)
      losses[compiled].append(critic.loss.item())
      assert torch.isfinite(critic.grad_norm).item()
    assert torch.get_float32_matmul_precision() == 'highest'
  rms = lambda a, b: (a - b).pow(2).mean().sqrt().item()
  eager, compiled = agents[False], agents[True]
  assert losses[True] == pytest.approx(losses[False], abs=1e-4)
  assert rms(flat(compiled.q1), flat(eager.q1)) < 3e-5
  assert rms(flat(compiled.policy), flat(eager.policy)) < 2.5e-5
  # One set of weights, so that only the path differs.
  eager.load_state_dict(compiled.state_dict())
  with torch.no_grad():
    assert (compiled.act(batch.graph) - eager.act(batch.graph)).abs().max().item() < 1e-6, 'acting is f32'
    with compiled._autocast:
      assert rms(compiled._policy(batch.graph).float(), eager.policy(batch.graph).float()) < 2e-4


def test_the_excerpt_says_what_its_verbatim_blocks_change_and_whose_notices_it_carries():
  text = (GOLDEN / 'fasttd3_make_goldens.py').read_text()
  header = text[:text.index('from __future__')]
  assert 'Copyright (c) 2025 Younggyo Seo' in header and 'Copyright (c)\n2024 LeanRL developers' in header and 'BAIR Commons' in header
  assert 'docs/LICENSES.md' in header
  assert '# BEGIN VERBATIM fast_td3/train.py:427-430, with normals, drawn above, in place of torch.randn_like(actions).' in text
  assert "# BEGIN VERBATIM fast_td3/train.py:291-295, for a list of parameters in place of the critic's, and 3e-4 and 0.1 in place of the args." in text


def test_saturation_shows_in_the_stats():
  """What the smoke run reads: target mass on the end atoms."""
  batch, topo = real_batch(8)
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False), topo, 'cuda:0')
  terminal = torch.ones(8, dtype=torch.bool, device='cuda:0')
  critic, _ = agent.update(batch._replace(reward=torch.full((8,), 5.0, device='cuda:0'), terminal=terminal), False)
  assert critic.mass_at_vmax.item() == pytest.approx(1.0, abs=1e-5) and critic.mass_at_vmin.item() < 1e-5
  assert critic.target_max.item() == pytest.approx(1.0, abs=1e-5) and critic.target_mean.item() == pytest.approx(1.0, abs=1e-5)
  critic, _ = agent.update(batch._replace(reward=torch.full((8,), -5.0, device='cuda:0'), terminal=terminal), False)
  assert critic.mass_at_vmin.item() == pytest.approx(1.0, abs=1e-5) and critic.mass_at_vmax.item() < 1e-5
  assert critic.target_min.item() == pytest.approx(-1.0, abs=1e-5)
  critic, _ = agent.update(batch._replace(reward=torch.zeros(8, device='cuda:0'), terminal=terminal), False)
  assert critic.mass_at_vmin.item() < 1e-5 and critic.mass_at_vmax.item() < 1e-5


def test_state_dict_round_trip_is_exact_and_frozen_views_stay_shared():
  batch, topo = real_batch()
  cfg = FastTD3Cfg(compile=False)
  a = FastTD3(SMALL, cfg, topo, 'cuda:0', noise_seed=1)
  a.update(batch, True)
  # Through a file, as a resume does: an optimizer keeps the very tensors of an in-memory state.
  buffer = io.BytesIO()
  torch.save(a.state_dict(), buffer)
  buffer.seek(0)
  b = FastTD3(SMALL, cfg, topo, 'cuda:0', noise_seed=2)
  b.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True))
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(flat(getattr(a, name)), flat(getattr(b, name))), name
  assert torch.equal(flat(b.q1), flat(b.q1_frozen)) and torch.equal(flat(b.q2), flat(b.q2_frozen))
  ca, _ = a.update(batch, True)
  cb, _ = b.update(batch, True)
  assert ca.loss.item() == cb.loss.item()
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(flat(getattr(a, name)), flat(getattr(b, name))), f'{name} after one more update'
  assert all(fp is qp or fp.data_ptr() == qp.data_ptr() for fp, qp in zip(b.q1_frozen.parameters(), b.q1.parameters()))


def test_nonfinite_values_reach_the_loss_rather_than_a_crash():
  """A NaN critic gives a NaN loss, which the trainer reads at report time. No index error on the way."""
  batch, topo = real_batch()
  agent = FastTD3(SMALL, FastTD3Cfg(compile=False), topo, 'cuda:0')
  for p in agent.q1_target.parameters():
    p.data.fill_(float('nan'))
  for p in agent.q2_target.parameters():
    p.data.fill_(float('nan'))
  critic, _ = agent.update(batch, False)
  assert not torch.isfinite(critic.loss).item()
