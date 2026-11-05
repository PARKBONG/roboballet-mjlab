# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The SAC agent against numbers from rsl_rl_sac at commit e0d243a, and its loop on a real env. DESIGN 16.

tests/golden/sac_make_goldens.py holds the verbatim excerpt and writes sac_*.npz. Nothing here imports
from a clone. The tiny networks below are the fork's architecture, so that its whole update can be
followed step by step, with weights and normals from the stored trace.

The second half needs one GPU and CUDA_VISIBLE_DEVICES set. The multi-process tests run two real ranks
on that GPU through gloo, at small sizes: averaging logic only, no throughput claim. This file is also
their worker: python -m tests.test_sac.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from roboballet_mjlab.agent.features import ACTION_DIM, Topology, build_graph
from roboballet_mjlab.agent.networks import PolicyNet, load_policy
from roboballet_mjlab.config import EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.replay.replay import STAT
from roboballet_mjlab.sac import agent as agent_module
from roboballet_mjlab.sac.agent import (
  SAC, Batch, GaussianActor, QNet, actor_loss, critic_loss, critic_target, matmul_precision, soft_update, squash, temperature_loss,
)
from roboballet_mjlab.sac.config import SACCfg
from roboballet_mjlab.sac.trainer import CHECKPOINT_NAME, FAILED_CHECKPOINT_NAME, POLICY_DIR, Trainer, atomic_save, average_gradients
from roboballet_mjlab.structs import Graph
from tests.test_sim_common import DEVICE, stored_scenes
from roboballet_mjlab.train.evaluator import Evaluator

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / 'golden'
DEVICES = ['cpu', 'cuda:0']
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
ENV = EnvCfg(world=WorldCfg(num_robots=2, num_obstacles=3), num_targets=2)
TRACES = ['default', 'fork', 'every', 'third', 'tight', 'paper']
T = 3
MAX_STEPS = 50  # Of T = 3.
NO_LEARNING = dict(batch_size=10**7)
# Float32 against float32 on another device, over 12 updates. Observed at most 6e-8 on the CPU and 2.3e-7 on the GPU, where
# one update moves weights by up to 1e-3. test_update_trace prints the worst with -s.
PARAMETER_TOLERANCE = {'cpu': 1e-7, 'cuda:0': 5e-7}
LOSS_TOLERANCE = {'cpu': 2e-6, 'cuda:0': 2e-6}
LOG_PROB_TOLERANCE = 1e-5  # Per row, a sum over the dimensions of values near 3 in magnitude. The temperature's loss is near 20.


def load(name: str):
  return np.load(GOLDEN / f'sac_{name}.npz')


# The fork's networks, sac_mlp_model.py with rsl_rl/modules/mlp.py and ELU.


def elu_mlp(in_size: int, out_size: int, hidden: list[int]) -> nn.Sequential:
  layers = []
  for size in hidden:
    layers += [nn.Linear(in_size, size), nn.ELU()]
    in_size = size
  return nn.Sequential(*layers, nn.Linear(in_size, out_size))


class RefActor(nn.Module):
  def __init__(self, n_obs, n_act, hidden):
    super().__init__()
    self.mlp = elu_mlp(n_obs, 2 * n_act, hidden)

  def forward(self, obs):
    return self.mlp(obs).chunk(2, -1)


class RefQ(nn.Module):
  def __init__(self, n_obs, n_act, hidden):
    super().__init__()
    self.mlp = elu_mlp(n_obs + n_act, 1, hidden)

  def forward(self, obs, action):
    return self.mlp(torch.cat([obs, action], -1))[:, 0]


def set_flat(module: nn.Module, values: np.ndarray) -> None:
  values = torch.as_tensor(values)
  at = 0
  for p in module.parameters():
    p.data.copy_(values[at : at + p.numel()].view_as(p))
    at += p.numel()
  assert at == len(values)


def flat(module) -> torch.Tensor:
  return torch.cat([p.detach().flatten() for p in module.parameters()])


def trace_cfg(z, name: str) -> SACCfg:
  args = lambda key: z[f'{name}_arg_{key}'].item()
  return SACCfg(
    gamma=args('gamma'), tau=args('tau'), lr_actor=args('actor_learning_rate'), lr_critic=args('critic_learning_rate'),
    lr_alpha=args('alpha_learning_rate'), alpha=args('alpha'), target_entropy_scale=args('target_entropy_scale'),
    max_grad_norm=args('max_grad_norm'), policy_frequency=int(args('policy_frequency')), init_noise_std=args('init_noise_std'),
    compile=False, bf16=False, matmul_precision='highest')


def trace_agent(z, name: str, device: str) -> SAC:
  n_obs, n_act, hidden = z[f'{name}_obs'].shape[-1], z[f'{name}_action'].shape[-1], [32, 32]
  actor, q1, q2 = RefActor(n_obs, n_act, hidden), RefQ(n_obs, n_act, hidden), RefQ(n_obs, n_act, hidden)
  set_flat(actor, z[f'{name}_init_actor'])
  set_flat(q1, z[f'{name}_init_q1'])
  set_flat(q2, z[f'{name}_init_q2'])
  agent = SAC(None, trace_cfg(z, name), None, device, nets=(actor, q1, q2), action_dim=n_act)
  normals = iter(torch.as_tensor(z[f'{name}_normals'], device=device).flatten(0, 1))
  agent._draw_normals = lambda like: next(normals)
  return agent


def trace_batch(z, name: str, k: int, device: str) -> Batch:
  at = lambda key: torch.as_tensor(z[f'{name}_{key}'][k], device=device)
  return Batch(at('obs'), at('next_obs'), at('action'), at('reward'), at('done').bool())


# The log-probability and the losses alone.


def test_critic_target_takes_a_per_sample_discount():
  reward, terminal = torch.tensor([0.1, 0.2, 0.3]), torch.tensor([False, False, True])
  q1, q2, log_prob = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 1.0, 4.0]), torch.tensor([0.5, 0.5, 0.5])
  log_alpha = torch.tensor(-2.0)
  gamma = torch.tensor([0.94, 0.94**3, 0.94**5])
  got = critic_target(reward, terminal, q1, q2, log_prob, log_alpha, gamma)
  next_value = torch.min(q1, q2) - log_alpha.exp() * log_prob
  assert torch.allclose(got, reward + gamma * (~terminal).float() * next_value)
  assert got[2] == reward[2]


@pytest.mark.parametrize('device', DEVICES)
def test_log_probability_matches_the_fork_including_saturation_and_both_clamps(device):
  z = load('log_prob')
  at = lambda key: torch.as_tensor(z[key], device=device)
  action, log_prob = squash(at('mean'), at('log_std'), at('normals'), z['log_std_min'].item(), z['log_std_max'].item())
  assert (action - at('action')).abs().max().item() < 1e-6
  assert (log_prob - at('log_prob')).abs().max().item() < 2e-4, 'over 7 dimensions, at values up to 158'
  assert log_prob.dtype == torch.float32 and log_prob.shape == (256,)
  # The cases that the tanh correction and the clamps decide.
  assert (at('action').abs() == 1.0).any()
  assert (at('log_std') < -20.0).any() and (at('log_std') > 2.0).any()


def test_log_probability_sums_over_robots_and_joints():
  z = load('log_prob')
  mean, log_std, normals = (torch.as_tensor(z[key]) for key in ('mean', 'log_std', 'normals'))
  doubled, joint_log_prob = squash(mean[:, None].repeat(1, 3, 1), log_std[:, None].repeat(1, 3, 1), normals[:, None].repeat(1, 3, 1), -20.0, 2.0)
  assert doubled.shape == (256, 3, 7)
  assert (joint_log_prob - 3 * torch.as_tensor(z['log_prob'])).abs().max().item() < 1e-3


def test_the_log_probability_is_the_density_of_the_squashed_action():
  """A histogram of the actions against exp(log_prob). Without the tanh term this is out by large factors."""
  normals = torch.randn(400_000, 1, generator=torch.Generator().manual_seed(0))
  action, log_prob = squash(torch.zeros(400_000, 1), torch.full((400_000, 1), 0.5), normals, -20.0, 2.0)
  edges = torch.linspace(-1.0, 1.0, 21)
  index = torch.bucketize(action[:, 0], edges) - 1
  for b in range(1, 19):
    inside = index == b
    assert log_prob[inside].exp().mean().item() == pytest.approx(inside.float().mean().item() / 0.1, rel=0.03), f'bin {b}'


def test_losses_match_the_fork():
  z = load('losses')
  at = lambda key: torch.as_tensor(z[key])
  log_alpha = at('log_alpha')
  target = critic_target(at('reward'), at('done').bool(), at('q1_target'), at('q2_target'), at('next_log_prob'), log_alpha, z['gamma'].item())
  assert (target - at('target_no_timeouts')).abs().max().item() < 1e-6
  total, loss1, loss2 = critic_loss(at('q1_pred'), at('q2_pred'), at('target_no_timeouts'))
  assert (total.item(), loss1.item(), loss2.item()) == pytest.approx((z['critic_loss'].item(), z['critic1_loss'].item(), z['critic2_loss'].item()), abs=1e-6)
  assert temperature_loss(log_alpha, at('log_prob'), z['target_entropy'].item()).item() == pytest.approx(z['alpha_loss'].item(), abs=1e-5)
  assert actor_loss(log_alpha, at('log_prob'), at('q1'), at('q2')).item() == pytest.approx(z['actor_loss'].item(), abs=1e-6)


def test_timeouts_are_the_only_rows_where_the_fork_differs():
  """DESIGN 16: no bootstrap on any terminal. The fork's rule, stored too, bootstraps where timeout is 1."""
  z = load('losses')
  at = lambda key: torch.as_tensor(z[key])
  target = critic_target(at('reward'), at('done').bool(), at('q1_target'), at('q2_target'), at('next_log_prob'), at('log_alpha'), z['gamma'].item())
  timeout, done = at('timeout').bool(), at('done').bool()
  assert timeout.any() and (done & ~timeout).any()
  assert torch.equal(target[timeout], at('reward')[timeout]), 'the reward alone'
  assert torch.equal(target[done], at('reward')[done])
  assert (target[~timeout] - at('target_fork_timeouts')[~timeout]).abs().max().item() < 1e-6
  assert (target[timeout] - at('target_fork_timeouts')[timeout]).abs().min().item() > 0.0


def test_terminal_rows_take_the_reward_only_whatever_the_value_and_the_entropy():
  reward = torch.tensor([0.1, 0.1, -0.2, -0.2])
  terminal = torch.tensor([True, False, True, False])
  big = torch.full((4,), 100.0)
  target = critic_target(reward, terminal, big, big + 1, -big, torch.tensor(0.0), 0.9)
  assert target[0].item() == pytest.approx(0.1) and target[2].item() == pytest.approx(-0.2)
  assert target[1].item() == pytest.approx(0.1 + 0.9 * (100.0 + 100.0))
  assert target[3].item() == pytest.approx(-0.2 + 0.9 * (100.0 + 100.0))


def test_target_takes_the_lower_critic_and_adds_the_entropy():
  reward, terminal = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
  log_alpha, gamma = torch.tensor(math.log(0.5)), 0.5
  target = critic_target(reward, terminal, torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([-2.0]), log_alpha, gamma)
  assert target.item() == pytest.approx(0.5 * (1.0 + 0.5 * 2.0))


def test_temperature_falls_above_the_target_entropy_and_rises_below():
  high_entropy, low_entropy = torch.full((8,), -10.0), torch.full((8,), 10.0)  # Log-probabilities.
  for log_prob, sign in ((high_entropy, -1.0), (low_entropy, 1.0)):
    log_alpha = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([log_alpha], lr=0.1)
    temperature_loss(log_alpha, log_prob, -3.0).backward()
    optimizer.step()
    assert log_alpha.item() * sign > 0.0
  # The gradient stops at the temperature.
  log_prob = torch.zeros(4, requires_grad=True)
  temperature_loss(torch.tensor(0.5, requires_grad=True), log_prob, -3.0).backward()
  assert log_prob.grad is None


def test_actor_loss_wants_entropy_and_value():
  log_prob = torch.tensor([-1.0], requires_grad=True)
  q = torch.tensor([2.0], requires_grad=True)
  actor_loss(torch.tensor(math.log(0.1)), log_prob, q, q + 1.0).backward()
  assert log_prob.grad.item() == pytest.approx(0.1), 'lowering the log-probability lowers the loss'
  assert q.grad.item() == pytest.approx(-1.0), 'raising the lower critic lowers the loss'
  # The temperature gets nothing from the actor's loss.
  log_alpha = torch.tensor(0.0, requires_grad=True)
  actor_loss(log_alpha, torch.tensor([-1.0], requires_grad=True), torch.zeros(1), torch.zeros(1)).backward()
  assert log_alpha.grad is None


def test_soft_update_is_the_forks_direction():
  target, online = [torch.tensor([1.0])], [torch.tensor([0.0])]
  soft_update(target, online, 0.25)
  assert target[0].item() == 0.75


@pytest.mark.parametrize('name', ['default', 'paper'])
def test_initialization_of_the_last_layer_is_the_forks(name):
  z = load('init')
  sigma = z[f'{name}_sigma'].item()
  topo = Topology(ENV, 'cpu')
  last = GaussianActor(ModelCfg(), topo, sigma).policy_head[-1]
  fork_weight, fork_bias = torch.as_tensor(z[f'{name}_weight']), torch.as_tensor(z[f'{name}_bias'])
  assert last.weight.shape == fork_weight.shape == (14, 64) and last.bias.shape == fork_bias.shape == (14,)
  assert torch.equal(last.bias.detach(), fork_bias), 'zero for the mean, log(sigma + 1e-7) for the log-std'
  assert not last.weight[7:].any() and not fork_weight[7:].any()
  assert last.weight[:7].std().item() == pytest.approx(fork_weight[:7].std().item(), rel=0.2)
  assert last.weight[:7].std().item() == pytest.approx(1e-3, rel=0.2) and last.weight[:7].any()


def test_a_fresh_actor_is_centred_with_the_configured_spread():
  topo = Topology(ENV, 'cpu')
  graph = random_graph(topo, 32, 'cpu')
  mean, log_std = GaussianActor(SMALL, topo, 0.15)(graph)
  assert mean.shape == log_std.shape == (32, 2, 7)
  assert mean.abs().max().item() < 0.05
  assert (log_std - math.log(0.15)).abs().max().item() < 1e-5, 'the same at every state, sac paper equation 8'


# The whole update, on the fork's networks, against its trace.


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', TRACES)
def test_update_trace(name, device, monkeypatch, capsys):
  z = load('update')
  key = lambda field: z[f'{name}_{field}']
  agent = trace_agent(z, name, device)
  assert agent.target_entropy == key('target_entropy').item()
  assert agent.log_alpha.item() == pytest.approx(key('init_log_alpha').item())
  targets, log_probs = [], []
  real_target, real_squash = critic_target, squash

  def spy_target(*args):
    targets.append(real_target(*args))
    return targets[-1]

  def spy_squash(*args):
    result = real_squash(*args)
    log_probs.append(result[1].detach())
    return result

  monkeypatch.setattr(agent_module, 'critic_target', spy_target)
  monkeypatch.setattr(agent_module, 'squash', spy_squash)
  worst = 0.0
  for k in range(len(key('target'))):
    update, actor = agent.update(trace_batch(z, name, k, device))
    assert (actor is not None) == bool(key('actor_step')[k]), f'update {k}'
    gold = lambda field: torch.as_tensor(key(field)[k], device=device)
    assert (targets[-1] - gold('target')).abs().max().item() < LOSS_TOLERANCE[device], f'target {k}'
    assert (log_probs[-2] - gold('next_log_prob')).abs().max().item() < LOG_PROB_TOLERANCE, f'next log-prob {k}'
    assert (log_probs[-1] - gold('log_prob')).abs().max().item() < LOG_PROB_TOLERANCE, f'log-prob {k}'
    for mine, field in ((update.critic_loss, 'critic_loss'), (update.critic1_loss, 'critic1_loss'), (update.critic2_loss, 'critic2_loss')):
      assert abs(mine.item() - key(field)[k]) < LOSS_TOLERANCE[device], f'{field} {k}'
    assert abs(update.alpha_loss.item() - key('alpha_loss')[k]) < LOG_PROB_TOLERANCE, f'alpha loss {k}'
    assert abs(update.critic_grad_norm.item() - key('critic_norm')[k]) < LOSS_TOLERANCE[device], f'critic norm {k}'
    assert abs(agent.log_alpha.item() - key('log_alpha')[k]) < 1e-5, f'log_alpha {k}'
    assert abs(update.alpha.item() - math.exp(key('log_alpha')[k])) < 1e-5
    assert abs(update.entropy.item() + key('log_prob')[k].mean()) < LOSS_TOLERANCE[device]
    if actor is not None:
      assert abs(actor.loss.item() - key('actor_loss')[k]) < LOSS_TOLERANCE[device], f'actor loss {k}'
      assert abs(actor.grad_norm.item() - key('actor_norm')[k]) < LOSS_TOLERANCE[device], f'actor norm {k}'
    for module, field in ((agent.actor, 'actor'), (agent.q1, 'q1'), (agent.q2, 'q2'), (agent.q1_target, 'q1_target'), (agent.q2_target, 'q2_target')):
      error = (flat(module) - gold(field)).abs().max().item()
      worst = max(worst, error)
      assert error < PARAMETER_TOLERANCE[device], f'{field} after update {k}: {error}'
  with capsys.disabled():
    print(f'\n{name} {device}: worst parameter error {worst:.2e}')
  assert agent.grad_steps == len(key('target'))
  assert agent.policy_steps == int(key('actor_step').sum())


def test_the_traces_exercise_the_clips_and_the_frequencies():
  z = load('update')
  assert (z['default_critic_norm'] > 1.0).any() and (z['default_critic_norm'] < 1.0).any()
  assert (z['tight_actor_norm'] > 0.05).any() and (z['tight_critic_norm'] > 0.05).all()
  assert z['every_actor_step'].all() and z['third_actor_step'].tolist()[:6] == [True, False, False, True, False, False]
  assert z['default_actor_step'].tolist()[:4] == [True, False, True, False], 'on even steps, from the first'


def test_every_difference_from_the_forks_defaults_is_declared():
  """config.py comments each. Anything new here has to be argued in the status note."""
  z = load('update')
  fork = lambda key: z[f'fork_arg_{key}'].item()
  cfg = SACCfg()
  # The same as the fork's constructor.
  assert cfg.tau == fork('tau') == 0.005
  assert cfg.lr_actor == fork('actor_learning_rate') == 1e-3
  assert cfg.lr_critic == fork('critic_learning_rate') == 1e-3
  assert cfg.lr_alpha == fork('alpha_learning_rate') == 1e-3
  assert cfg.alpha == fork('alpha') == 0.05
  assert cfg.target_entropy_scale == fork('target_entropy_scale') == 1.0
  assert cfg.max_grad_norm == fork('max_grad_norm') == 1.0
  assert cfg.policy_frequency == fork('policy_frequency') == 2
  assert cfg.num_learning_epochs == fork('num_learning_epochs') == 1
  assert cfg.init_noise_std == fork('init_noise_std') == 1.0
  assert (cfg.log_std_min, cfg.log_std_max) == (fork('log_std_min'), fork('log_std_max')) == (-20.0, 2.0)
  assert fork('n_steps') == 1 and fork('auto_alpha') is True
  # DESIGN 16's deviation in the update rule.
  assert fork('gamma') == 0.998 and cfg.gamma == 0.94
  # FastTD3Cfg's, so that the two compare directly.
  assert fork('mini_batch_size') == 256 and cfg.batch_size == 2048
  assert fork('replay_buffer_size') == 1_000_000 and cfg.replay_max_size == 2_000_000
  assert fork('num_mini_batches') == 1 and cfg.num_mini_batches == 2 and cfg.num_steps_per_env == 1
  assert cfg.num_worlds == 1024
  assert (cfg.hindsight, cfg.hindsight_fraction, cfg.hindsight_flush_episodes) == (True, 0.4, 256)
  assert cfg.hindsight_zero_dwelling_actions and cfg.hindsight_zero_collision_penalty
  assert (cfg.bf16, cfg.compile, cfg.matmul_precision) == (True, True, 'high')
  assert cfg.eval_episodes == 1024


# The update on the GNN.


def random_graph(topo: Topology, B: int, device, seed: int = 0) -> Graph:
  generator = torch.Generator(device=device).manual_seed(seed)
  rand = lambda *shape: torch.rand(shape, generator=generator, device=device)
  return Graph(rand(B, topo.num_nodes, 17), rand(B, topo.num_edges, 30), rand(B, 2))


def random_batch(topo: Topology, B: int = 8, device: str = DEVICE, seed: int = 0) -> Batch:
  generator = torch.Generator(device=device).manual_seed(seed + 1)
  rand = lambda *shape: torch.rand(shape, generator=generator, device=device)
  return Batch(random_graph(topo, B, device, seed + 2), random_graph(topo, B, device, seed + 3), rand(B, topo.num_robots, 7) * 2 - 1,
               rand(B) * 0.2, rand(B) < 0.25)


def small_agent(device: str = DEVICE, **changes) -> tuple[SAC, Topology]:
  topo = Topology(ENV, device)
  base = dict(compile=False, bf16=False, matmul_precision='highest')
  return SAC(SMALL, SACCfg(**{**base, **changes}), topo, device), topo


def test_target_entropy_is_minus_the_action_dimension():
  for robots in (1, 2, 4, 8):
    topo = Topology(EnvCfg(world=WorldCfg(num_robots=robots, num_obstacles=1), num_targets=1), 'cpu')
    assert SAC(SMALL, SACCfg(compile=False), topo, 'cpu').target_entropy == -7.0 * robots
  assert SAC(SMALL, SACCfg(compile=False, target_entropy_scale=0.5), topo, 'cpu').target_entropy == -28.0


def test_order_of_one_update(monkeypatch):
  agent, topo = small_agent()
  calls = []

  def spy(name, owner=agent):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
      calls.append(name)
      return original(*args, **kwargs)

    setattr(owner, name, wrapped)

  for name in ('_update_critic', '_sample', '_update_alpha', '_update_actor', '_update_target'):
    spy(name)
  batch = random_batch(topo)
  agent.update(batch)
  assert calls == ['_update_critic', '_sample', '_sample', '_update_alpha', '_update_actor', '_update_target']
  calls.clear()
  agent.update(batch)
  assert calls == ['_update_critic', '_sample', '_sample', '_update_alpha', '_update_target'], 'no actor step on odd updates'


def test_critics_update_before_the_actor_reads_them_and_the_temperature_before_the_actor(monkeypatch):
  agent, topo = small_agent(lr_alpha=0.1)
  seen = {}
  frozen, real_target, real_actor = agent._q1_frozen, critic_target, actor_loss

  def spy_frozen(*args):
    seen['q1_when_actor_reads'] = flat(agent.q1_frozen).clone()
    return frozen(*args)

  def spy_target(*args):
    seen['alpha_in_target'] = args[5].item()
    return real_target(*args)

  def spy_actor(log_alpha, *args):
    seen['alpha_in_actor'] = log_alpha.item()
    return real_actor(log_alpha, *args)

  agent._q1_frozen = spy_frozen
  monkeypatch.setattr(agent_module, 'critic_target', spy_target)
  monkeypatch.setattr(agent_module, 'actor_loss', spy_actor)
  start_q1, start_alpha = flat(agent.q1).clone(), agent.log_alpha.item()
  agent.update(random_batch(topo))
  assert seen['alpha_in_target'] == start_alpha, 'the temperature before this update changes it, sac.py:282'
  assert seen['alpha_in_actor'] == agent.log_alpha.item() != start_alpha, 'after, sac.py:326'
  assert not torch.equal(seen['q1_when_actor_reads'], start_q1), 'the critic update came first'
  assert torch.equal(seen['q1_when_actor_reads'], flat(agent.q1))


@pytest.mark.parametrize('frequency', [1, 2, 3])
def test_policy_frequency_counts_from_the_first_update(frequency):
  agent, topo = small_agent(policy_frequency=frequency)
  batch = random_batch(topo)
  for k in range(7):
    actor_before, critic_before, alpha_before = flat(agent.actor).clone(), flat(agent.q1).clone(), agent.log_alpha.item()
    update, actor = agent.update(batch)
    moved = not torch.equal(actor_before, flat(agent.actor))
    assert moved == (k % frequency == 0) == (actor is not None), f'update {k}'
    assert not torch.equal(critic_before, flat(agent.q1)) and agent.log_alpha.item() != alpha_before, 'on every update'
  assert agent.grad_steps == 7 and agent.policy_steps == sum(k % frequency == 0 for k in range(7))


def test_target_update_is_polyak_on_the_new_critic_every_update_and_only_there():
  agent, topo = small_agent()
  batch = random_batch(topo)
  assert torch.equal(flat(agent.q1_target), flat(agent.q1)) and torch.equal(flat(agent.q2_target), flat(agent.q2)), 'start equal'
  assert not any(p.requires_grad for p in [*agent.q1_target.parameters(), *agent.q2_target.parameters()])
  for _ in range(4):
    before1, before2 = flat(agent.q1_target).clone(), flat(agent.q2_target).clone()
    agent.update(batch)
    expected1 = 0.995 * before1 + 0.005 * flat(agent.q1)
    expected2 = 0.995 * before2 + 0.005 * flat(agent.q2)
    assert (flat(agent.q1_target) - expected1).abs().max().item() < 1e-6
    assert (flat(agent.q2_target) - expected2).abs().max().item() < 1e-6
    assert not torch.equal(flat(agent.q1_target), before1)
  assert not hasattr(agent, 'actor_target'), 'SAC has no target actor'


def test_the_actor_step_does_not_move_the_critics_and_reads_the_online_ones():
  agent, topo = small_agent()
  batch = random_batch(topo)
  action, log_prob, _ = agent._sample(batch.graph)
  q1, q2 = flat(agent.q1).clone(), flat(agent.q2).clone()
  targets = {'q1': 0, 'q2': 0}
  t1, t2 = agent._q1_target, agent._q2_target
  agent._q1_target = lambda *a: (targets.__setitem__('q1', 1), t1(*a))[1]
  agent._q2_target = lambda *a: (targets.__setitem__('q2', 1), t2(*a))[1]
  agent._update_actor(batch, action, log_prob)
  assert torch.equal(q1, flat(agent.q1)) and torch.equal(q2, flat(agent.q2))
  assert targets == {'q1': 0, 'q2': 0}
  assert all(p.grad is None for p in agent._critic_params), 'nothing written to the critics'
  assert agent._q1_frozen is agent.q1_frozen
  assert all(a.data_ptr() == b.data_ptr() for a, b in zip(agent.q1_frozen.parameters(), agent.q1.parameters()))
  assert all(a.data_ptr() == b.data_ptr() for a, b in zip(agent.q2_frozen.parameters(), agent.q2.parameters()))


@pytest.mark.parametrize('compile_', [False, True])
def test_the_value_gradient_reaches_the_actor_on_the_gnn(compile_):
  """With the temperature at zero only Q moves the actor. In bf16, compiled and not, as training runs."""
  agent, topo = small_agent(compile=compile_, bf16=True, matmul_precision='high')
  batch = random_batch(topo)
  with torch.no_grad():
    agent.log_alpha.fill_(-200.0)
  with matmul_precision('high'):
    action, log_prob, _ = agent._sample(batch.graph)
    before = flat(agent.actor).clone()
    actor = agent._update_actor(batch, action, log_prob)
  assert log_prob.requires_grad and action.requires_grad
  assert actor.grad_norm.item() > 1e-5
  assert not torch.equal(before, flat(agent.actor))
  last = agent.actor.policy_head[-1]
  assert last.weight.grad[:7].abs().max().item() > 0.0, 'the mean rows'
  assert last.weight.grad[7:].abs().max().item() > 0.0, 'the log-std rows, through the sample'


def test_q_spread_is_of_the_first_critic_on_the_stored_action():
  agent, topo = small_agent()
  batch = random_batch(topo)
  expected = agent.q1(batch.graph, batch.action).std(unbiased=False).item()
  update, _ = agent.update(batch)
  assert update.q_std.item() == pytest.approx(expected, rel=1e-5) and expected > 0.0


def test_critic_loss_sees_the_stored_action_and_the_gradient_is_clipped():
  agent, topo = small_agent(max_grad_norm=1e-3)
  batch = random_batch(topo)
  before = flat(agent.q1).clone()
  update, _ = agent.update(batch)
  assert update.critic_grad_norm.item() > 1e-3
  grads = torch.cat([p.grad.flatten() for p in agent._critic_params])
  assert grads.norm().item() == pytest.approx(1e-3, rel=1e-2), 'clipped to the limit before the step'
  assert not torch.equal(before, flat(agent.q1))
  actor_grads = torch.cat([p.grad.flatten() for p in agent._actor_params if p.grad is not None])
  assert actor_grads.norm().item() == pytest.approx(1e-3, rel=1e-2)


def test_the_clip_acts_after_the_all_reduce():
  """sac.py:299-302. A hook that scales the gradient must meet it unclipped, and the step sees it clipped."""
  seen = []

  def hook(vector):
    seen.append(vector.norm().item())
    return vector * 1000.0

  topo = Topology(ENV, DEVICE)
  agent = SAC(SMALL, SACCfg(compile=False, bf16=False, matmul_precision='highest', max_grad_norm=1e-3), topo, DEVICE, grad_hook=hook)
  update, actor = agent.update(random_batch(topo))
  assert seen[0] > 1e-3, 'unclipped on the way in'
  assert update.critic_grad_norm.item() == pytest.approx(1000.0 * seen[0], rel=1e-3), 'the norm before the clip is of the reduced one'
  assert torch.cat([p.grad.flatten() for p in agent._critic_params]).norm().item() == pytest.approx(1e-3, rel=1e-2)
  assert actor.grad_norm.item() == pytest.approx(1000.0 * seen[2], rel=1e-3)
  assert len(seen) == 3, 'critic, temperature, actor'


def test_the_temperature_gradient_is_not_clipped():
  agent, topo = small_agent(max_grad_norm=1e-6)
  agent.update(random_batch(topo))
  assert abs(agent.log_alpha.grad.item()) > 1.0, 'near entropy minus target entropy, sac.py:312-321'


def test_the_critic_has_no_tanh_and_needs_none():
  """DESIGN 16 asks which. A soft Q holds alpha * entropy, which the original's tanh could not represent."""
  topo = Topology(ENV, DEVICE)
  q = QNet(SMALL, topo).to(DEVICE)
  with torch.no_grad():
    q.qvalue[-1].bias.fill_(5.0)
  batch = random_batch(topo)
  assert (q(batch.graph, batch.action) > 4.0).all()
  assert not hasattr(q, 'rewards')
  # A fresh actor at 4 robots and the configured temperature and discount.
  four = Topology(EnvCfg(), 'cpu')
  agent = SAC(SMALL, SACCfg(compile=False, bf16=False), four, 'cpu')
  graph = random_graph(four, 512, 'cpu')
  _, log_prob, _ = agent._sample(graph)
  entropy = -log_prob.mean().item()
  soft_value = 0.05 * entropy / (1 - 0.94)
  assert entropy == pytest.approx(19.0, abs=1.0)
  assert soft_value > 10.0, f'far outside [-1, 1]: {soft_value}'


def test_mean_policy_is_a_plain_policy_net_on_the_actors_storage():
  agent, topo = small_agent()
  policy = agent.mean_policy
  assert type(policy) is PolicyNet
  assert policy.policy_head[-1].weight.shape == (7, 8)
  batch = random_batch(topo)
  for _ in range(3):
    agent.update(batch)
  mean, _ = agent.actor(batch.graph)
  assert torch.equal(policy(batch.graph), torch.tanh(mean)), 'it moved with the actor'
  acted = agent.act(batch.graph, torch.zeros_like(mean))
  assert torch.equal(acted.mean, torch.tanh(mean)) and torch.equal(acted.action, torch.tanh(mean))


# Float modes, DESIGN 14.1.


def test_learner_is_bf16_with_f32_log_probabilities_and_acting_is_f32(monkeypatch):
  agent, topo = small_agent(bf16=True, matmul_precision='high')
  seen = {'dtype': [], 'autocast': [], 'matmul': []}

  def hook(module, inputs):
    seen['dtype'].append(inputs[0].dtype)
    seen['autocast'].append(torch.is_autocast_enabled('cuda'))
    seen['matmul'].append(torch.get_float32_matmul_precision())

  agent.actor.policy_head[0].register_forward_pre_hook(hook)
  dtypes = {}
  real_squash, real_target = squash, critic_target

  def spy_squash(mean, log_std, *args):
    dtypes['mean'], dtypes['log_std'] = mean.dtype, log_std.dtype
    action, log_prob = real_squash(mean, log_std, *args)
    dtypes['log_prob'] = log_prob.dtype
    return action, log_prob

  def spy_target(*args):
    target = real_target(*args)
    dtypes['target'] = target.dtype
    return target

  monkeypatch.setattr(agent_module, 'squash', spy_squash)
  monkeypatch.setattr(agent_module, 'critic_target', spy_target)
  batch = random_batch(topo)
  update, actor = agent.update(batch)
  assert set(seen['dtype']) == {torch.bfloat16} and set(seen['autocast']) == {True}
  assert set(seen['matmul']) == {'high'} and torch.get_float32_matmul_precision() == 'highest'
  assert set(dtypes.values()) == {torch.float32}
  assert all(x.dtype == torch.float32 for x in update) and all(x.dtype == torch.float32 for x in actor)
  for values in seen.values():
    values.clear()
  acted = agent.act(batch.graph, torch.zeros(8, 2, 7, device=DEVICE))
  assert set(seen['dtype']) == {torch.float32} and set(seen['autocast']) == {False} and set(seen['matmul']) == {'highest'}
  assert all(x.dtype == torch.float32 for x in acted)


def test_nothing_sets_the_global_precision_outside_the_learner_call():
  torch.set_float32_matmul_precision('highest')
  agent, topo = small_agent(matmul_precision='high')
  assert torch.get_float32_matmul_precision() == 'highest'
  batch = random_batch(topo)
  agent.update(batch)
  assert torch.get_float32_matmul_precision() == 'highest'
  original = agent._update_alpha

  def boom(*args):
    assert torch.get_float32_matmul_precision() == 'high'
    raise RuntimeError('boom')

  agent._update_alpha = boom
  with pytest.raises(RuntimeError, match='boom'):
    agent.update(batch)
  assert torch.get_float32_matmul_precision() == 'highest', 'restored through the exception'
  agent._update_alpha = original
  with matmul_precision('medium'):
    assert torch.get_float32_matmul_precision() == 'medium'
  assert torch.get_float32_matmul_precision() == 'highest'


def test_compiled_acting_is_f32_after_a_compiled_bf16_learner_call():
  """Acting in f32 at highest, also with graphs compiled for bf16 by the learner in between."""
  torch.set_float32_matmul_precision('highest')
  agent, topo = small_agent(compile=True, bf16=True, matmul_precision='high')
  batch = random_batch(topo)
  for _ in range(2):
    agent.update(batch)
  normals = torch.zeros(8, 2, 7, device=DEVICE)
  compiled = agent.act(batch.graph, normals)
  with torch.no_grad():
    mean, log_std = agent.actor(batch.graph)
    with torch.autocast('cuda', dtype=torch.bfloat16):
      rounded, _ = agent.actor(batch.graph)
  assert (compiled.mean - torch.tanh(mean)).abs().max().item() < 2e-5
  assert compiled.std.dtype == torch.float32
  assert (compiled.std - log_std.exp()).abs().max().item() < 2e-5
  assert (torch.tanh(rounded) - torch.tanh(mean)).abs().max().item() > 1e-5, 'bf16 would show'


# The loop, on a real env, replay and learner, on a small model.


def small_pool(targets: int = T):
  scenes = stored_scenes(4)
  return scenes._replace(target_pos=scenes.target_pos[:, :targets], target_rot=scenes.target_rot[:, :targets])


def small_cfg(**changes) -> SACCfg:
  base = dict(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight_flush_episodes=4, compile=False,
              log_every_rounds=25, eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, matmul_precision='high', resume_min_rows=16)
  return SACCfg(**{**base, **changes})


def make_trainer(run_dir, evaluator=None, **changes) -> Trainer:
  return Trainer(EnvCfg(num_targets=T), small_cfg(**changes), SMALL, small_pool(), DEVICE, run_dir, evaluator=evaluator)


def run_rounds(trainer: Trainer, rounds: int) -> None:
  for _ in range(rounds):
    trainer.round()


def spy_step(trainer: Trainer) -> list:
  """Keeps every action request and StepOut of env.step."""
  log: list = []
  original = trainer.env.step

  def step(action):
    out = original(action)
    log.append((action.clone(), out))
    return out

  trainer.env.step = step
  return log


def filled_trainer(tmp_path, name='a', **changes) -> Trainer:
  """A trainer whose ring already holds episodes, so that the schedule shows from its first round."""
  filler = make_trainer(tmp_path / f'{name}_filler', **NO_LEARNING)
  run_rounds(filler, MAX_STEPS + 12)
  assert filler.replay.size >= 16 and filler.agent.grad_steps == 0
  trainer = make_trainer(tmp_path / f'{name}_trainer', **changes)
  trainer.replay = filler.replay
  return trainer


def test_action_is_tanh_of_mean_plus_std_times_normal_and_stored(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 5)  # Learning has begun.
  seen = []
  act = t.agent.act

  def spy_act(graph, normals):
    seen.append((graph, normals.clone(), act(graph, normals)))
    return seen[-1][2]

  t.agent.act = spy_act
  steps = spy_step(t)
  before = t.buffer.count.clone()
  t.round()
  action = steps[0][0]
  _, normals, acted = seen[0]
  assert torch.equal(acted.action, action), 'the sample goes to the env'
  assert normals.shape == (8, 4, 7) and normals.std() > 0.5
  assert (normals[0, 0] != normals[0, 1]).all() and (normals[0] != normals[1]).any(), 'per world, robot and joint'
  assert action.abs().max() <= 1.0 and acted.mean.abs().max() < 1.0
  assert not torch.equal(action, acted.mean), 'exploration is in the sample'
  # The buffer keeps what went into env.step. Worlds that just ended have moved it to the ring.
  worlds = (~steps[0][1].terminal).nonzero().flatten()
  assert torch.equal(t.buffer.action[worlds, before[worlds]], action[worlds])


def test_acting_rebuilds_the_action_from_the_actor_on_fresh_weights(tmp_path):
  t = make_trainer(tmp_path, **NO_LEARNING)
  steps = spy_step(t)
  graph = build_graph(t.env.observe(), t.topo)
  replay = torch.Generator(device=DEVICE)
  replay.set_state(t.act_rng.get_state())
  t.round()
  normals = torch.randn(t._action_shape, generator=replay, device=DEVICE)
  mean, log_std = t.agent.actor(graph)
  rebuilt = torch.tanh(mean + normals * log_std.clamp(-20.0, 2.0).exp())
  assert torch.equal(rebuilt, steps[0][0]), 'no random phase: a network sample from the first round, sac.py:178-184'
  assert steps[0][0].std() > 0.3, 'init_noise_std 1 is near uniform'


def test_acting_clamps_the_log_std_as_the_learner_does():
  agent, topo = small_agent()
  graph = random_batch(topo).graph
  last = agent.actor.policy_head[-1]
  normals = torch.ones(8, 2, 7, device=DEVICE)
  for bias, limit in ((30.0, 2.0), (-30.0, -20.0)):
    with torch.no_grad():
      last.bias[7:] = bias
    assert (agent.act(graph, normals).std / math.exp(limit) - 1.0).abs().max().item() < 1e-5
    _, log_std = agent.actor(graph)
    _, _, clamped = agent._sample(graph)
    assert torch.equal(clamped, log_std.clamp(-20.0, 2.0))


def test_acting_uses_the_online_weights_with_no_copy(tmp_path):
  t = make_trainer(tmp_path)
  graph = build_graph(t.env.observe(), t.topo)
  normals = torch.zeros(t._action_shape, device=DEVICE)
  before = t.agent.act(graph, normals).mean
  run_rounds(t, MAX_STEPS + 3)
  assert t.agent.policy_steps > 0
  assert not torch.equal(before, t.agent.act(graph, normals).mean)
  assert t.agent._actor is t.agent.actor or t.agent._actor._orig_mod is t.agent.actor


def recorded_schedule(t: Trainer, rounds: int) -> list[tuple[int, str]]:
  calls: list[tuple[int, str]] = []
  sample, critic, actor, target = t.replay.sample, t.agent._update_critic, t.agent._update_actor, t.agent._update_target

  def spy(kind, fn):
    def wrapped(*args, **kwargs):
      calls.append((t.counters.rounds - 1, kind))
      return fn(*args, **kwargs)
    return wrapped

  t.replay.sample = spy('sample', sample)
  t.agent._update_critic, t.agent._update_actor, t.agent._update_target = spy('critic', critic), spy('actor', actor), spy('target', target)
  run_rounds(t, rounds)
  return calls


def test_schedule_is_two_batches_a_round_and_the_actor_on_even_updates(tmp_path):
  t = filled_trainer(tmp_path)
  calls = recorded_schedule(t, 3)
  one_round = [('sample'), ('critic'), ('actor'), ('target'), ('sample'), ('critic'), ('target')]
  assert calls == [(r, kind) for r in range(3) for kind in one_round]
  assert t.agent.grad_steps == 6 and t.agent.policy_steps == 3


def test_schedule_of_the_forks_runner_with_more_steps_per_env(tmp_path):
  """off_policy_runner.py:88-105: num_steps_per_env rounds, then the updates."""
  t = filled_trainer(tmp_path, num_steps_per_env=3, num_mini_batches=4, num_learning_epochs=2)
  calls = recorded_schedule(t, 7)
  rounds_with_updates = sorted({r for r, _ in calls})
  assert rounds_with_updates == [2, 5]
  assert sum(kind == 'sample' for _, kind in calls) == 2 * 8
  assert t.agent.grad_steps == 16


def test_each_update_has_a_fresh_batch_with_features_of_s_and_of_s_prime(tmp_path):
  t = filled_trainer(tmp_path)
  samples, batches = [], []
  sample, critic = t.replay.sample, t.agent._update_critic
  t.replay.sample = lambda *args: (samples.append(sample(*args)), samples[-1])[1]
  t.agent._update_critic = lambda batch: (batches.append(batch), critic(batch))[1]
  run_rounds(t, 2)
  assert len(samples) == len(batches) == 4
  assert not torch.equal(samples[0].action, samples[1].action)
  for s, b in zip(samples, batches):
    assert s.next_obs.robot.step.ne(s.obs.robot.step).all(), 's and s\' are different rows'
    assert all(torch.equal(x, y) for x, y in zip(b.graph, build_graph(s.obs, t.topo)))
    assert all(torch.equal(x, y) for x, y in zip(b.next_graph, build_graph(s.next_obs, t.topo)))
    assert torch.equal(b.action, s.action) and torch.equal(b.reward, s.reward) and torch.equal(b.terminal, s.terminal)


def test_updates_wait_for_a_batch_in_the_ring(tmp_path):
  """Whole episodes enter this ring, so no row exists before max_steps. The fork has no such wait."""
  t = make_trainer(tmp_path, batch_size=120, num_worlds=8)
  for _ in range(MAX_STEPS * 2):
    size_before, grads_before = t.replay.size, t.agent.grad_steps
    t.round()
    if t.agent.grad_steps > grads_before:
      assert t.replay.size >= 120
      break
    assert t.replay.size < 120
  assert t.agent.grad_steps == 2 and t.counters.rounds > MAX_STEPS and size_before < 120


def test_real_episodes_enter_before_their_hindsight(tmp_path):
  t = make_trainer(tmp_path, **NO_LEARNING)
  stores: list[list[bool]] = []
  insert, store = t.replay.insert, t._store

  def spy_insert(batch):
    assert batch.is_hindsight.all() == batch.is_hindsight.any()
    stores[-1].append(bool(batch.is_hindsight.any()))
    return insert(batch)

  def spy_store(finished):
    stores.append([])
    return store(finished)

  t.replay.insert, t._store = spy_insert, spy_store
  run_rounds(t, MAX_STEPS * 4)
  assert stores and all(entry in ([False], [False, True]) for entry in stores), stores
  assert [False, True] in stores and t.counters.hindsight_episodes > 0
  # Half the transitions are hindsight under a policy that solves nothing.
  assert t.replay.is_hindsight[:t.replay.episodes].float().mean().item() == pytest.approx(0.5, abs=0.1)


def test_hindsight_uses_the_configured_flush_and_can_be_turned_off(tmp_path):
  t = make_trainer(tmp_path / 'on', hindsight_flush_episodes=7, **NO_LEARNING)
  assert t.queue.train_cfg is t.cfg and t.queue.train_cfg.hindsight_flush_episodes == 7
  assert t.queue.k == int(t.cfg.hindsight_fraction * T)
  off = make_trainer(tmp_path / 'off', hindsight=False, **NO_LEARNING)
  run_rounds(off, MAX_STEPS * 2)
  assert off.queue is None and off.counters.hindsight_episodes == 0 and off.counters.real_episodes > 0


def test_ring_capacity(tmp_path):
  t = make_trainer(tmp_path, replay_max_size=300, **NO_LEARNING)
  run_rounds(t, MAX_STEPS * 3)
  assert t.replay.capacity == 300 and t.replay.size <= 300 and t.replay.evicted > 0


def test_a_timeout_reaches_the_learner_as_a_terminal_and_takes_the_reward_only(tmp_path, monkeypatch):
  """On the loop, the ring's terminal is True on timeouts, and the last action of an episode is one."""
  t = make_trainer(tmp_path, batch_size=128)
  seen = []
  real_target = critic_target

  def spy(reward, terminal, *args):
    target = real_target(reward, terminal, *args)
    seen.append((reward.clone(), terminal.clone(), target.clone()))
    return target

  monkeypatch.setattr(agent_module, 'critic_target', spy)
  run_rounds(t, MAX_STEPS * 2)
  terminals = sum(int(terminal.sum()) for _, terminal, _ in seen)
  assert terminals > 0, 'random motion solves nothing, so every terminal here is a timeout'
  for reward, terminal, target in seen:
    assert torch.equal(target[terminal], reward[terminal])
    assert terminal.dtype == torch.bool
  assert (t.replay.stats[:t.replay.episodes, STAT['episode_length']] == MAX_STEPS + 1).all()


def test_every_stream_has_its_own_seed(tmp_path):
  t = make_trainer(tmp_path)
  draws = [torch.randint(2**31 - 1, (4,), generator=generator, device=DEVICE).tolist()
           for generator in (t.act_rng, t.sample_rng, t.hindsight_rng, t.env.generator, t.agent.rng)]
  assert all(a != b for i, a in enumerate(draws) for b in draws[i + 1:])
  other = make_trainer(tmp_path / 'b', seed=2)
  assert t.act_rng.initial_seed() != other.act_rng.initial_seed()
  assert t.agent.rng.initial_seed() != other.agent.rng.initial_seed()


def test_each_world_resets_once_at_its_offset(tmp_path):
  t = make_trainer(tmp_path, **NO_LEARNING)
  resets = []
  reset = t.env.reset

  def spy(world_ids=None, scene_ids=None):
    resets.append((t._rounds_this_run, world_ids.tolist()))
    return reset(world_ids, scene_ids)

  t.env.reset = spy
  run_rounds(t, MAX_STEPS - 1)
  assert resets == [(i * MAX_STEPS // 8, [i]) for i in range(1, 8)]
  assert t.counters.discarded_rows == sum(i * MAX_STEPS // 8 for i in range(1, 8))


def test_the_hot_path_reads_nothing_back_from_this_package(tmp_path):
  """Python-level reads, from sac/ files. Shared modules are as the original's."""
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  reads = []
  saved = {}

  def counted(name):
    original = saved[name] = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
      caller = sys._getframe(1).f_code.co_filename
      if '/sac/' in caller:
        reads.append((name, caller, sys._getframe(1).f_lineno))
      return original(self, *args, **kwargs)

    return wrapper

  names = ('item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', '__float__', '__index__')
  for name in names:
    setattr(torch.Tensor, name, counted(name))
  try:
    start = t.counters.rounds
    while t.counters.rounds < start + 3 * t.cfg.log_every_rounds:
      if (t.counters.rounds + 1) % t.cfg.log_every_rounds:  # Report rounds may read.
        t.round()
      else:
        for name in names:
          setattr(torch.Tensor, name, saved[name])
        t.round()
        for name in names:
          setattr(torch.Tensor, name, counted(name))
  finally:
    for name in names:
      setattr(torch.Tensor, name, saved[name])
  assert reads == []
  assert t.agent.grad_steps > 0


def test_a_report_writes_the_shared_names_and_the_temperature_and_entropy(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=30)
  run_rounds(t, 2 * MAX_STEPS)
  t.logger.flush()
  values = {}
  for row in csv.DictReader(open(tmp_path / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  for name in ('train/qf_loss', 'train/critic1_loss', 'train/critic2_loss', 'train/actor_loss', 'train/alpha_loss', 'train/alpha', 'train/entropy',
               'train/target_entropy', 'train/log_std', 'train/q_mean', 'train/q_std', 'train/critic_grad_norm', 'train/actor_grad_norm', 'train/target_value_mean',
               'train/target_value_min', 'train/target_value_max', 'train/trajectory_network_age', 'train/policy_lr', 'train/twin_critic_lr',
               'train/alpha_lr', 'rate/env_steps_per_s', 'rate/grad_steps_per_s', 'counters/env_steps', 'counters/grad_steps',
               'act/num_targets_done', 'act/is_hindsight_replay', 'act/total_collision_robot_steps', 'act/final_score',
               'act/policy_action_abs', 'act/policy_action_saturated', 'act/policy_std'):
    assert name in values, name
  assert values['train/target_entropy'] == -28.0
  assert values['train/policy_lr'] == values['train/twin_critic_lr'] == values['train/alpha_lr'] == pytest.approx(1e-3)
  assert 0.0 < values['train/alpha'] < 0.05


def test_totals_add_up(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=30)
  run_rounds(t, 2 * MAX_STEPS)
  totals = t.totals()
  assert totals['env_steps'] == 2 * MAX_STEPS * 8 and totals['rounds'] == 2 * MAX_STEPS
  assert totals['grad_steps'] == 2 * totals['policy_steps']
  assert totals['sampled'] == totals['grad_steps'] * 16
  real = totals['real_episodes']
  assert real > 0 and totals['inserted'] == real * MAX_STEPS + totals['hindsight_episodes'] * MAX_STEPS
  assert totals['residual_collisions'] == 0


def test_precision_stays_highest_for_everything_but_the_learner(tmp_path):
  t = make_trainer(tmp_path)
  seen = {}
  act, step, update = t.agent.act, t.env.step, t.agent._update_critic

  def note(name, fn):
    def wrapped(*args, **kwargs):
      seen.setdefault(name, set()).add(torch.get_float32_matmul_precision())
      return fn(*args, **kwargs)
    return wrapped

  t.agent.act, t.env.step, t.agent._update_critic = note('act', act), note('env', step), note('learner', update)
  run_rounds(t, MAX_STEPS + 3)
  assert seen == {'act': {'highest'}, 'env': {'highest'}, 'learner': {'high'}}
  assert torch.get_float32_matmul_precision() == 'highest'


def test_features_give_the_same_bytes_after_a_learner_round(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 3)
  obs = t.env.observe()
  before = [x.clone() for x in build_graph(obs, t.topo)]
  t.round()
  assert all(torch.equal(a, b) for a, b in zip(before, build_graph(obs, t.topo)))


# Evaluator.


def small_evaluator(cfg: EnvCfg | None = None, worlds: int = 8) -> Evaluator:
  return Evaluator(cfg or EnvCfg(num_targets=T), SMALL, {'training_dist': small_pool()}, DEVICE, 1, num_worlds=worlds, compile=False, notice=lambda _: None)


def test_the_evaluator_accepts_the_actor_and_runs_its_mean(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 3)
  evaluator = small_evaluator()
  results = evaluator.evaluate(t.agent.mean_policy, t.agent.grad_steps)['training_dist']
  assert 0.0 <= results['task_rate'] <= 1.0 and results['worlds'] == 8.0 and results['network_step'] == t.agent.grad_steps
  graph = build_graph(t.env.observe(), t.topo)
  with torch.no_grad():
    mean, _ = t.agent.actor(graph)
    assert torch.equal(evaluator.policy(graph), torch.tanh(mean)), 'the private copy is the tanh of the mean, no noise'
  assert flat(evaluator.policy).numel() == flat(t.agent.actor).numel() - 7 * 8 - 7, 'all but the log-std rows'


def test_evaluation_mid_training_changes_nothing_in_training(tmp_path):
  """The fairness of DESIGN 12.2: a twin that evaluates stays bit-exact with one that does not."""
  plain = make_trainer(tmp_path / 'plain')
  evaluated = make_trainer(tmp_path / 'evaluated', evaluator=small_evaluator(), eval_every_rounds=20)
  for _ in range(MAX_STEPS + 30):
    plain.round()
    evaluated.round()
  assert evaluated._eval_seconds > 0
  assert torch.equal(flat(plain.agent.actor), flat(evaluated.agent.actor))
  assert torch.equal(flat(plain.agent.q1), flat(evaluated.agent.q1))
  assert plain.agent.log_alpha.item() == evaluated.agent.log_alpha.item()
  assert torch.equal(plain.env.generator.get_state(), evaluated.env.generator.get_state())
  assert torch.equal(plain.act_rng.get_state(), evaluated.act_rng.get_state())


def test_checkpoint_and_snapshots_load_as_a_plain_policy(tmp_path):
  t = make_trainer(tmp_path, policy_snapshot_every_steps=20)
  t._snapshot_if_due()
  run_rounds(t, MAX_STEPS + 15)
  path = t.save_checkpoint('test')
  graph = build_graph(t.env.observe(), t.topo)
  policy = PolicyNet(SMALL, t.topo).to(DEVICE)
  assert load_policy(str(path), policy) == t.agent.grad_steps
  with torch.no_grad():
    mean, _ = t.agent.actor(graph)
    assert torch.equal(policy(graph), torch.tanh(mean))
  names = sorted(p.name for p in (tmp_path / POLICY_DIR).iterdir())
  steps = [int(name[len('policy_') : -len('.pt')]) for name in names]
  assert steps[0] == 0 and all(s % 20 in (0, 1) for s in steps) and len(steps) >= 2, 'at step 0 and on each multiple'
  assert all(b - a in (20, 21, 22) or a == 0 for a, b in zip(steps, steps[1:]))
  assert load_policy(str(tmp_path / POLICY_DIR / names[-1]), policy) == steps[-1]
  # A snapshot holds only the mean rows' storage, not the whole of the actor's last layer.
  state = torch.load(tmp_path / POLICY_DIR / names[-1], weights_only=True)
  assert state['policy']['policy_head.3.weight'].untyped_storage().nbytes() == 7 * 8 * 4


# Failure handling and checkpoints.


def test_nonfinite_loss_saves_a_failed_checkpoint_and_raises(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  good = t.save_checkpoint('good')
  before = good.read_bytes()
  for p in t.agent.q1_target.parameters():
    p.data.fill_(float('nan'))
  for p in t.agent.q2_target.parameters():
    p.data.fill_(float('nan'))
  with pytest.raises(FloatingPointError, match='nonfinite loss'):
    run_rounds(t, t.cfg.log_every_rounds + 1)
  assert (tmp_path / FAILED_CHECKPOINT_NAME).exists()
  assert good.read_bytes() == before, 'the last good checkpoint stays'
  fresh = make_trainer(tmp_path / 'fresh')
  with pytest.raises(ValueError, match='failed checkpoint'):
    fresh.load_checkpoint(tmp_path / FAILED_CHECKPOINT_NAME)
  fresh.load_checkpoint(good)


def test_a_nan_actor_reaches_the_report_through_the_env(tmp_path):
  """Acting on NaN for up to a report's rounds must end in the same clean failure, not in the env."""
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  for p in t.agent.actor.parameters():
    p.data.fill_(float('nan'))
  with pytest.raises(FloatingPointError, match='nonfinite loss'):
    run_rounds(t, t.cfg.log_every_rounds + 1)
  assert (tmp_path / FAILED_CHECKPOINT_NAME).exists()


def test_a_nan_temperature_is_caught_too(tmp_path):
  t = make_trainer(tmp_path)
  run_rounds(t, MAX_STEPS + 2)
  t.agent.log_alpha.data.fill_(float('nan'))
  with pytest.raises(FloatingPointError, match='nonfinite loss'):
    run_rounds(t, t.cfg.log_every_rounds + 1)


def test_atomic_save_keeps_the_old_file_when_writing_fails(tmp_path):
  path = tmp_path / 'x.pt'
  atomic_save({'a': 1}, path)
  with pytest.raises(Exception):
    atomic_save({'a': lambda: 0}, path)
  assert torch.load(path, weights_only=True) == {'a': 1}
  assert [p.name for p in tmp_path.iterdir()] == ['x.pt']


def test_save_and_resume_restores_everything(tmp_path):
  t = make_trainer(tmp_path / 'a', log_every_rounds=30)
  run_rounds(t, MAX_STEPS + 20)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b', log_every_rounds=30)
  r.load_checkpoint(path)
  a, b = t.agent, r.agent
  assert (a.grad_steps, a.policy_steps) == (b.grad_steps, b.policy_steps) and a.grad_steps > 0
  for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(flat(getattr(a, name)), flat(getattr(b, name))), name
  assert torch.equal(flat(b.q1), flat(b.q1_frozen))
  assert torch.equal(flat(b.mean_policy.policy_head), flat(a.mean_policy.policy_head))
  assert a.log_alpha.item() == b.log_alpha.item() != math.log(0.05)
  for p, q in zip(a._critic_params, b._critic_params):
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
      assert torch.equal(a.critic_optimizer.state[p][key], b.critic_optimizer.state[q][key]), key
  for p, q in zip(a._actor_params, b._actor_params):
    if p.grad is not None:
      assert torch.equal(a.actor_optimizer.state[p]['exp_avg'], b.actor_optimizer.state[q]['exp_avg'])
  assert torch.equal(a.alpha_optimizer.state[a.log_alpha]['exp_avg'], b.alpha_optimizer.state[b.log_alpha]['exp_avg'])
  assert torch.equal(t.act_rng.get_state(), r.act_rng.get_state())
  assert torch.equal(t.sample_rng.get_state(), r.sample_rng.get_state())
  assert torch.equal(t.hindsight_rng.get_state(), r.hindsight_rng.get_state())
  assert torch.equal(t.env.generator.get_state(), r.env.generator.get_state())
  assert torch.equal(a.rng.get_state(), b.rng.get_state())
  assert r.counters.rounds == t.counters.rounds and r.totals()['inserted'] == t.totals()['inserted']
  assert r.totals()['sampled'] == t.totals()['sampled']
  # The same batch gives the same update afterwards: every part of the learner came back, the phase of the frequency too.
  topo = t.topo
  sample = t.replay.sample(16, torch.Generator(device=DEVICE).manual_seed(0))
  batch = Batch(build_graph(sample.obs, topo), build_graph(sample.next_obs, topo), sample.action, sample.reward, sample.terminal)
  for _ in range(3):
    a.update(batch)
    b.update(batch)
  for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(flat(getattr(a, name)), flat(getattr(b, name))), name
  assert a.log_alpha.item() == b.log_alpha.item()


def test_resumed_run_refills_the_ring_before_updating(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, MAX_STEPS + 5)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b')
  r.load_checkpoint(path)
  assert r.replay.size == 0
  grad_steps, rounds = r.agent.grad_steps, r.counters.rounds
  run_rounds(r, MAX_STEPS - 1)
  assert r.agent.grad_steps == grad_steps, 'nothing to sample yet'
  run_rounds(r, 2)
  assert r.agent.grad_steps > grad_steps
  assert r.counters.rounds == rounds + MAX_STEPS + 1
  assert r.totals()['inserted'] > t.totals()['inserted']


def test_resumed_run_waits_for_resume_min_rows_and_a_fresh_run_does_not(tmp_path):
  assert SACCfg().resume_min_rows == 200_000
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, MAX_STEPS + 5)
  path = t.save_checkpoint('test')
  r = make_trainer(tmp_path / 'b', resume_min_rows=300)
  r.load_checkpoint(path)
  grad_steps, size = r.agent.grad_steps, 0
  while r.agent.grad_steps == grad_steps:
    assert size < 300, f'{size} rows and no update'
    size = r.replay.size
    r.round()
  assert size < 300 <= r.replay.size, 'starts on the round the ring reaches it'
  # The same setting on a fresh run changes nothing, down to the weights.
  fresh = [make_trainer(tmp_path / name, resume_min_rows=rows) for name, rows in (('c', 16), ('d', 10**6))]
  for x in fresh:
    run_rounds(x, MAX_STEPS + 5)
  assert fresh[0].agent.grad_steps == fresh[1].agent.grad_steps == t.agent.grad_steps > 0
  assert torch.equal(flat(fresh[0].agent.actor), flat(fresh[1].agent.actor))


def test_a_ring_smaller_than_resume_min_rows_does_not_wait_forever(tmp_path):
  r = make_trainer(tmp_path / 'b', replay_max_size=200, resume_min_rows=300)
  r.load_checkpoint(make_trainer(tmp_path / 'c', replay_max_size=200).save_checkpoint('test'))
  run_rounds(r, MAX_STEPS * 3)
  assert r.agent.grad_steps > 0 and r.replay.size <= 200


def test_a_checkpoint_older_than_a_field_resumes_at_its_default(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, 2)
  state = torch.load(t.save_checkpoint('test'), map_location='cpu', weights_only=True)
  for key in ('n_step', 'resume_min_rows'):
    del state['config']['train'][key]
  old = tmp_path / 'old.pt'
  torch.save(state, old)
  make_trainer(tmp_path / 'b').load_checkpoint(old)
  with pytest.raises(ValueError, match='n_step'):
    make_trainer(tmp_path / 'c', n_step=3).load_checkpoint(old)


def test_resume_refuses_a_changed_config_but_not_a_changed_cadence(tmp_path):
  t = make_trainer(tmp_path / 'a')
  run_rounds(t, 2)
  path = t.save_checkpoint('test')
  with pytest.raises(ValueError, match='gamma'):
    make_trainer(tmp_path / 'b', gamma=0.9).load_checkpoint(path)
  with pytest.raises(ValueError, match='not a SAC checkpoint'):
    bad = tmp_path / 'other.pt'
    torch.save({'format': 1}, bad)
    make_trainer(tmp_path / 'c').load_checkpoint(bad)
  make_trainer(tmp_path / 'd', total_env_rounds=7, eval_every_rounds=5, log_every_rounds=3, profile_every_rounds=9,
               resume_min_rows=7).load_checkpoint(path)


def test_run_stops_on_a_signal_and_on_the_time_limit_and_saves(tmp_path):
  t = make_trainer(tmp_path / 'signal', total_env_rounds=10**6)
  t.install_signal_handlers()
  try:
    original = t.round

    def signalled():
      original()
      if t.counters.rounds == 5:
        os.kill(os.getpid(), signal.SIGTERM)

    t.round = signalled
    assert t.run() == f'signal {int(signal.SIGTERM)}'
    assert t.counters.rounds == 5 and (tmp_path / 'signal' / CHECKPOINT_NAME).exists()
  finally:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
  t = Trainer(EnvCfg(num_targets=T), small_cfg(total_env_rounds=10**6), SMALL, small_pool(), DEVICE, tmp_path / 'timed', stop_after_s=0.2)
  assert t.run() == 'stop_after_seconds'
  t = make_trainer(tmp_path / 'done', total_env_rounds=3)
  assert t.run() == 'complete' and t.counters.rounds == 3


def test_periodic_checkpoint(tmp_path):
  t = make_trainer(tmp_path, total_env_rounds=3, checkpoint_every_s=0.0)
  assert t.run() == 'complete'
  assert t._checkpoint_seconds > 0 and (tmp_path / CHECKPOINT_NAME).exists()
  assert not list(tmp_path.glob('*.tmp*'))
  assert [p.name for p in (tmp_path / POLICY_DIR).iterdir()] == ['policy_00000000.pt']


# The real command.


def script_env() -> dict:
  """PYTHONPATH as it came, so an overlay of the mutation check reaches the subprocess too."""
  return dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH', str(REPO / 'src')), PYTHONUNBUFFERED='1')


def small_pool_file(path: Path) -> Path:
  from roboballet_mjlab.scenes.io import Pool, save_pool
  from roboballet_mjlab.scenes.sampler import Witnesses

  save_pool(path, Pool(small_pool(), Witnesses(torch.zeros(1), torch.zeros(1)), {'test': True}))
  return path


def script_args(run_dir: Path, pool: Path, *extra: str) -> list[str]:
  sets = ['num_worlds=8', 'batch_size=16', 'replay_max_size=4000', 'hindsight_flush_episodes=4', 'compile=False', 'log_every_rounds=25',
          'eval_every_rounds=40', 'eval_episodes=8', 'profile=False', 'resume_min_rows=16']
  models = ['node_layers=2', 'node_size=16', 'edge_embed_layers=2', 'edge_update_layers=2', 'edge_size=8', 'global_layers=2', 'head_sizes=8']
  args = [sys.executable, '-m', 'roboballet_mjlab.sac.train', '--run-dir', str(run_dir), '--train-pool', str(pool)]
  for item in sets:
    args += ['--set', item]
  for item in models:
    args += ['--model-set', item]
  return args + list(extra)


def wait_for(path: Path, process: subprocess.Popen, timeout: float = 300.0) -> None:
  deadline = time.monotonic() + timeout
  while not path.exists():
    assert process.poll() is None, f'exited early with {process.returncode}'
    assert time.monotonic() < deadline, f'no {path}'
    time.sleep(0.5)


def test_script_records_provenance_stops_on_sigterm_resumes_and_evaluates(tmp_path):
  pool = small_pool_file(tmp_path / 'pool.pt')
  run_dir = tmp_path / 'run'
  log = open(tmp_path / 'log.txt', 'w')
  process = subprocess.Popen(script_args(run_dir, pool), env=script_env(), stdout=log, stderr=subprocess.STDOUT)
  try:
    wait_for(run_dir / 'tb', process)
    deadline = time.monotonic() + 300
    # Past the first report.
    while b'counters/rounds' not in (run_dir / 'metrics.csv').read_bytes():
      assert process.poll() is None and time.monotonic() < deadline
      time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=120) == 0
  finally:
    process.kill()
  state = torch.load(run_dir / CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert state['format'] == 'sac-1' and state['reason'] == f'signal {int(signal.SIGTERM)}'
  provenance = next((run_dir / 'provenance').iterdir())
  assert {p.name for p in provenance.iterdir()} >= {'config.json', 'git_revision.txt', 'git_diff.patch', 'git_untracked.txt', 'system.json'}
  config = json.loads((provenance / 'config.json').read_text())
  assert config['train']['gamma'] == 0.94 and config['model']['node_size'] == 16
  rounds = json.loads((run_dir / 'summary.json').read_text())['rounds']
  assert rounds == state['counters']['rounds'] and rounds >= 25
  assert (run_dir / POLICY_DIR / 'policy_00000000.pt').exists()

  done = subprocess.run(script_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 3)), env=script_env(), capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  assert 'resumed from' in done.stdout
  assert json.loads((run_dir / 'summary.json').read_text())['rounds'] == rounds + 3
  assert len(list((run_dir / 'provenance').iterdir())) == 2

  done = subprocess.run(script_args(run_dir, pool, '--eval-only', str(run_dir / CHECKPOINT_NAME)), env=script_env(), capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  results = json.loads(done.stdout[done.stdout.index('{\n'):])
  assert results['results']['training_dist']['worlds'] == 8.0


def test_end_to_end_compiled_bf16_small_model(tmp_path):
  """As training runs: compiled, bf16, profiler on. Finite, counters agree, memory flat."""
  t = make_trainer(tmp_path, compile=True, bf16=True, profile=True, profile_every_rounds=40, num_worlds=16, batch_size=32, log_every_rounds=20)
  t.profiler.first_window = 60
  t.profiler.window_rounds = 10
  run_rounds(t, MAX_STEPS + 40)
  memory = torch.cuda.memory_allocated()
  run_rounds(t, 60)
  assert torch.cuda.memory_allocated() - memory < 32 * 2**20
  assert t.agent.grad_steps == 2 * (t.counters.rounds - MAX_STEPS + 1), 'from the round the first episode ends'
  assert all(p.isfinite().all() for p in t.agent.actor.parameters())
  values = {}
  t.logger.flush()
  for row in csv.DictReader(open(tmp_path / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  assert math.isfinite(values['train/qf_loss']) and math.isfinite(values['train/actor_loss']) and math.isfinite(values['train/alpha'])
  for phase in ('critic_update', 'actor_sample', 'alpha_update', 'actor_update', 'target_update', 'world_forward', 'sampling'):
    assert f'profile/{phase}_ms' in values, phase
  assert values['profile/window_item_calls_per_round'] < 2


# Two processes, gloo on one GPU. Averaging logic only.


UPDATES = 4
RANK_CFG = SACCfg(compile=False, bf16=False, matmul_precision='highest', seed=11)


def worker_average(out: Path) -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  topo = Topology(ENV, 'cuda:0')
  average = average_gradients(world_size)
  record: dict[str, list[torch.Tensor]] = {'before': [], 'after': []}

  def hook(vector):
    record['before'].append(vector.cpu().clone())
    result = average(vector)
    record['after'].append(result.cpu().clone())
    return result

  agent = SAC(SMALL, RANK_CFG, topo, 'cuda:0', grad_hook=hook, noise_seed=500 + rank)
  start = (flat(agent.actor).cpu(), flat(agent.q1).cpu(), flat(agent.q2).cpu())
  for update in range(UPDATES):
    agent.update(random_batch(topo, 8, 'cuda:0', 1000 * rank + update))
  torch.save({
    'start': start, 'before': record['before'], 'after': record['after'], 'log_alpha': agent.log_alpha.detach().cpu(),
    'actor': flat(agent.actor).cpu(), 'q1': flat(agent.q1).cpu(), 'q2': flat(agent.q2).cpu(),
    'q1_target': flat(agent.q1_target).cpu(), 'q2_target': flat(agent.q2_target).cpu(),
  }, out)


def worker_trainer(out: Path, poison_rank: int | None) -> None:
  """Two trainers on a real env. Each has its own env, replay, seeds and worlds."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  run_dir = out.parent / f'trainer_{rank}'
  cfg = SACCfg(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight_flush_episodes=4, compile=False, log_every_rounds=25,
               eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5 + rank)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', run_dir, rank=rank, world_size=world_size)
  # The seeds differ here, so equal starts can only come from the broadcast.
  start = flat(t.agent.actor).cpu()
  learned_at = None
  if poison_rank is None:
    for _ in range(MAX_STEPS + 10):
      t.round()
      if learned_at is None and t.agent.grad_steps:
        learned_at = t.counters.rounds
    torch.save({
      'start': start, 'actor': flat(t.agent.actor).cpu(), 'q1': flat(t.agent.q1).cpu(), 'q2': flat(t.agent.q2).cpu(),
      'q1_target': flat(t.agent.q1_target).cpu(), 'q2_target': flat(t.agent.q2_target).cpu(), 'log_alpha': t.agent.log_alpha.detach().cpu(),
      'grad_steps': t.agent.grad_steps, 'learned_at': learned_at, 'env_steps': t.totals()['env_steps'],
      'act_state': t.act_rng.get_state(), 'q': t.env.observe().robot.q.cpu(),
    }, out)
    return
  run_ok = False
  try:
    run = 0
    while True:
      t.round()
      run += 1
      if run == MAX_STEPS and rank == poison_rank:
        t._update_sum[0] = float('nan')
      if run > MAX_STEPS + 2 * cfg.log_every_rounds:
        run_ok = True
        break
  except FloatingPointError as error:
    torch.save({'raised': str(error), 'failed_exists': t.checkpoint_path(failed=True).exists(), 'rounds': t.counters.rounds}, out)
  if run_ok:
    torch.save({'raised': None}, out)


def worker_uneven(out: Path) -> None:
  """Rank 1 has fewer worlds, so its ring holds a batch later. Neither may start before the other."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  cfg = SACCfg(num_worlds=8 if rank == 0 else 2, batch_size=60, replay_max_size=4000, hindsight=False, compile=False, log_every_rounds=25,
               eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', out.parent / f'uneven_{rank}', rank=rank, world_size=world_size)
  began = size = None
  for _ in range(MAX_STEPS + 35):
    t.round()
    if began is None and t.agent.grad_steps:
      began, size = t.counters.rounds, t.replay.size
  torch.save({'rounds': began, 'size': size, 'grad_steps': t.agent.grad_steps, 'actor': flat(t.agent.actor).cpu()}, out)


def worker_stop(out: Path) -> None:
  """One signal, on rank 1 only. Both must stop, at the same round, and rank 0 must not hang."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  cfg = SACCfg(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight=False, compile=False, log_every_rounds=25,
               eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5, total_env_rounds=10**6)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', out.parent / f'stop_{rank}', rank=rank, world_size=world_size)
  t.install_signal_handlers()
  original = t.round

  def signalled():
    original()
    if rank == 1 and t.counters.rounds == 7:
      os.kill(os.getpid(), signal.SIGTERM)

  t.round = signalled
  status = t.run()
  torch.save({'status': status, 'rounds': t.counters.rounds, 'saved': t.checkpoint_path().exists()}, out)


def worker_main(argv: list[str]) -> None:
  mode, out = argv[0], Path(argv[1])
  rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
  dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)
  torch.set_float32_matmul_precision('highest')
  try:
    if mode == 'average':
      worker_average(out / f'average_{rank}.pt')
    elif mode == 'trainer':
      worker_trainer(out / f'trainer_{rank}.pt', None)
    elif mode == 'uneven':
      worker_uneven(out / f'uneven_{rank}.pt')
    elif mode == 'stop':
      worker_stop(out / f'stop_{rank}.pt')
    else:
      worker_trainer(out / f'poison_{rank}.pt', 1)
  finally:
    dist.destroy_process_group()


def spawn_two(*args: str) -> list[subprocess.Popen]:
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
  processes = []
  for rank in range(2):
    env = dict(os.environ, RANK=str(rank), WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORKER='1',
               PYTHONPATH=os.environ.get('PYTHONPATH', str(REPO / 'src')))
    processes.append(subprocess.Popen([sys.executable, '-m', 'tests.test_sac', *args], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
  return processes


def join(processes: list[subprocess.Popen], timeout: float = 240.0) -> None:
  for process in processes:
    try:
      output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
      for other in processes:
        other.kill()
      raise AssertionError('a rank hung, the failure these tests exist to catch')
    assert process.returncode == 0, output[-3000:]


def test_gradients_are_averaged_also_the_temperatures_and_the_ranks_stay_identical(tmp_path):
  join(spawn_two('average', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'average_{rank}.pt', weights_only=True) for rank in range(2))
  assert all(torch.equal(x, y) for x, y in zip(a['start'], b['start'])), 'same weights to start, from the config seed'
  # Critic and temperature on every update, the actor on every second one.
  assert len(a['before']) == 2 * UPDATES + UPDATES // 2
  assert sum(len(g) == 1 for g in a['before']) == UPDATES, 'the temperature, alone'
  for k, (ga, gb, avg_a, avg_b) in enumerate(zip(a['before'], b['before'], a['after'], b['after'])):
    assert not torch.equal(ga, gb), f'gradient {k}: the batches differ, so the raw gradients must'
    assert torch.equal(avg_a, avg_b), f'gradient {k}'
    assert (avg_a - (ga + gb) / 2).abs().max().item() < 1e-5 * max(ga.abs().max().item(), 1.0), f'gradient {k}'
  for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target', 'log_alpha'):
    assert torch.equal(a[name], b[name]), name

  # One process given those averages lands on the same weights, so all the average changes is the gradient.
  torch.set_float32_matmul_precision('highest')
  topo = Topology(ENV, 'cuda:0')
  feed = iter(a['after'])
  agent = SAC(SMALL, RANK_CFG, topo, 'cuda:0', grad_hook=lambda vector: next(feed).to(vector.device), noise_seed=500)
  for update in range(UPDATES):
    agent.update(random_batch(topo, 8, 'cuda:0', update))
  for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(a[name], flat(getattr(agent, name)).cpu()), name
  plain = SAC(SMALL, RANK_CFG, topo, 'cuda:0', noise_seed=500)
  for update in range(UPDATES):
    plain.update(random_batch(topo, 8, 'cuda:0', update))
  assert not torch.equal(a['q1'], flat(plain.q1).cpu()), 'against no averaging at all, the weights move elsewhere'


def test_two_trainers_on_a_real_env_stay_in_step(tmp_path):
  join(spawn_two('trainer', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'trainer_{rank}.pt', weights_only=True) for rank in range(2))
  assert torch.equal(a['start'], b['start'])
  assert a['grad_steps'] == b['grad_steps'] > 0
  assert a['learned_at'] == b['learned_at'], 'all ranks start updating together'
  for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target', 'log_alpha'):
    assert torch.equal(a[name], b[name]), name
  # Each has its own env, noise and data, and the counter is effective: two processes count double.
  assert not torch.equal(a['act_state'], b['act_state'])
  assert not torch.equal(a['q'], b['q'])
  assert a['env_steps'] == (MAX_STEPS + 10) * 8 * 2


def test_learning_starts_when_the_last_rank_is_ready(tmp_path):
  join(spawn_two('uneven', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'uneven_{rank}.pt', weights_only=True) for rank in range(2))
  # Rank 1's two worlds end at rounds 50 and 75, and 60 rows need both. Rank 0 alone would start at 56.
  assert a['rounds'] == b['rounds'] == 75
  assert a['size'] > b['size'] >= 60
  assert a['grad_steps'] == b['grad_steps'] and torch.equal(a['actor'], b['actor'])


def test_a_signal_to_one_rank_stops_both_at_the_same_round(tmp_path):
  join(spawn_two('stop', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'stop_{rank}.pt', weights_only=True) for rank in range(2))
  assert b['status'] == f'signal {int(signal.SIGTERM)}' and a['status'] == 'another rank stopped'
  assert a['rounds'] == b['rounds'] == 25, 'the first report round after the signal'
  assert a['saved'] and b['saved']


def test_a_nonfinite_loss_on_one_rank_stops_both_without_a_hang(tmp_path):
  join(spawn_two('poison', str(tmp_path)))
  for rank in range(2):
    state = torch.load(tmp_path / f'poison_{rank}.pt', weights_only=True)
    assert state['raised'] is not None and 'nonfinite' in state['raised'], rank
    assert state['failed_exists'], rank


def physical_gpu() -> str:
  visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
  assert visible and ',' not in visible, 'run these tests with CUDA_VISIBLE_DEVICES set to one GPU'
  return visible


def test_the_launcher_runs_two_ranks_stops_on_sigterm_and_resumes(tmp_path):
  pool = small_pool_file(tmp_path / 'pool.pt')
  run_dir = tmp_path / 'run'
  launch_args = lambda *extra: script_args(run_dir, pool, '--gpus', f'{physical_gpu()},{physical_gpu()}', *extra)
  log = open(tmp_path / 'log.txt', 'w')
  env = script_env()
  process = subprocess.Popen(launch_args(), env=env, stdout=log, stderr=subprocess.STDOUT)
  try:
    deadline = time.monotonic() + 300
    while not (run_dir / 'metrics.csv').exists() or b'train/qf_loss' not in (run_dir / 'metrics.csv').read_bytes():
      assert process.poll() is None, (tmp_path / 'log.txt').read_text()[-3000:]
      assert time.monotonic() < deadline, 'no report'
      time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=180) == 0, (tmp_path / 'log.txt').read_text()[-3000:]
  finally:
    process.kill()
  zero = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
  one = torch.load(run_dir / 'rank1_checkpoint.pt', map_location='cpu', weights_only=True)
  assert zero['step'] == one['step'] and zero['counters']['rounds'] == one['counters']['rounds']
  assert zero['config']['world_size'] == 2
  for key in zero['policy']:
    assert torch.equal(zero['policy'][key], one['policy'][key]), key
  assert zero['learner']['log_alpha'] == one['learner']['log_alpha']
  assert not torch.equal(zero['rng']['act'], one['rng']['act'])
  values = {}
  for row in csv.DictReader(open(run_dir / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  assert values['counters/env_steps'] == values['counters/rounds'] * 8 * 2, 'rank 0 alone writes, and the count is effective'

  rounds = zero['counters']['rounds']
  done = subprocess.run(launch_args('--resume', '--total-env-rounds', str(rounds + 30)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
  assert json.loads((run_dir / 'summary.json').read_text())['rounds'] == rounds + 30
  # A resume refuses a changed process count.
  done = subprocess.run(script_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 60)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode != 0 and 'world_size' in done.stderr


if __name__ == '__main__':
  if os.environ.get('WORKER') == '1':
    worker_main(sys.argv[1:])
