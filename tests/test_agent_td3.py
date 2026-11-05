"""E8, the 20-step TD3 golden against JAX, schedules, target noise, K=1 and the gradient hook."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.td3 import TD3, Batch
from roboballet_mjlab.config import EnvCfg, ModelCfg, TrainCfg
from roboballet_mjlab.structs import Graph
from tests.golden import agent_common
from tests.golden.agent_load import DEVICE, GOLDEN, deterministic_init, flat_nets, load_e6, transitions

GOLDEN_BATCH = 16
GOLDEN_STEPS = 20
LOSS_TOLERANCE = 1e-4  # algo_spec 12.2
PARAM_RMS_TOLERANCE = 5e-8  # algo_spec 12.2
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))


class Injected:
  """Replaces the target noise normals of a TD3 with the ones the original drew."""

  def __init__(self, td3: TD3, normals: np.ndarray):
    self.normals = torch.as_tensor(normals, device=DEVICE)
    self.i = 0
    td3._draw_normals = self

  def __call__(self, like: torch.Tensor) -> torch.Tensor:
    out = self.normals[self.i]
    self.i += 1
    return out


@pytest.fixture(scope='module')
def golden():
  z = np.load(GOLDEN / 'agent_td3.npz')
  return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def e6():
  return load_e6('4r10t', shards=2)


@pytest.fixture(scope='module')
def megabatch(golden, e6) -> Batch:
  batch = transitions(e6, golden['index'])
  assert batch.terminal.sum() > 0 and (batch.reward > 0).sum() > 0 and (batch.reward < 0).sum() > 0
  return batch


def golden_td3(compile: bool, **train_kwargs) -> TD3:
  cfg = TrainCfg(batch_size=GOLDEN_BATCH, fused_steps=GOLDEN_STEPS, matmul_precision='highest', compile=compile, bf16=False, **train_kwargs)
  td3 = TD3(ModelCfg(), cfg, Topology(EnvCfg(), DEVICE), DEVICE)
  deterministic_init(td3)
  return td3


def param_errors(td3: TD3, golden, label: str) -> dict[str, float]:
  """Largest RMS bound of each net: from the sampled entries, and from each full leaf.

  |norm(a) - norm(b)| / sqrt(n) is a lower bound on the RMS of a - b over the whole leaf,
  so the second number also sees the entries the sample leaves out.
  """
  errors = {}
  for net, flat in flat_nets(td3).items():
    sample, norm = agent_common.sample_leaves(flat)
    sizes = np.array([flat[k].size for k in sorted(flat)])
    errors[net] = float(np.sqrt(np.mean((sample - golden[f'{label}/{net}/sample']) ** 2)))
    errors[net + '_norm'] = float(np.max(np.abs(norm - golden[f'{label}/{net}/norm']) / np.sqrt(sizes)))
  return errors


@pytest.mark.parametrize('compile', [False, True])
def test_20_steps_against_jax(golden, megabatch, compile):
  td3 = golden_td3(compile)
  Injected(td3, golden['normals'])
  losses = []
  for i in range(GOLDEN_STEPS):
    out = td3.train_step(megabatch.slice(i * GOLDEN_BATCH, (i + 1) * GOLDEN_BATCH))
    losses.append((out.critic.item(), out.policy.item()))
    if i == 0:
      first = param_errors(td3, golden, 'step1')
  losses = np.array(losses)
  relative = np.abs(losses / golden['losses'] - 1).max(0)
  last = param_errors(td3, golden, f'step{GOLDEN_STEPS}')
  print(f'TD3 compile={compile} rel loss critic {relative[0]:.3e} policy {relative[1]:.3e}\n  step1 {first}\n  step20 {last}')
  assert relative.max() <= LOSS_TOLERANCE
  for errors in (first, last):
    assert max(errors[n] for n in ('pol', 'cri', 'tpol', 'tcri')) <= PARAM_RMS_TOLERANCE
    assert max(errors[n + '_norm'] for n in ('pol', 'cri', 'tpol', 'tcri')) <= PARAM_RMS_TOLERANCE


def test_e8_one_update(golden, megabatch):
  """Losses within 1% and parameter deltas within 1%, paper_baseline 10."""
  td3 = golden_td3(False)
  before = {k: agent_common.sample_leaves(v)[0] for k, v in flat_nets(td3).items()}
  Injected(td3, golden['normals'])
  out = td3.train_step(megabatch.slice(0, GOLDEN_BATCH))
  critic = abs(out.critic.item() / golden['losses'][0, 0] - 1)
  policy = abs(out.policy.item() / golden['losses'][0, 1] - 1)
  relative = {}
  for net, flat in flat_nets(td3).items():
    delta = agent_common.sample_leaves(flat)[0] - before[net]
    want = golden[f'step1/{net}/sample'] - before[net]
    relative[net] = float(np.linalg.norm(delta - want) / np.linalg.norm(want))
  print(f'E8 loss critic {critic:.3e} policy {policy:.3e} delta {relative}')
  assert critic <= 0.01 and policy <= 0.01
  assert max(relative.values()) <= 0.01


@pytest.mark.parametrize('compile', [False, True])
def test_fused_matches_original_logging(golden, megabatch, compile):
  td3 = golden_td3(compile)
  Injected(td3, golden['normals'])
  out = td3.fused_step(megabatch, summed=True)
  got = np.array([out.critic.item(), out.policy.item()])
  relative = np.abs(got / golden['fused_sums'] - 1)
  fused = param_errors(td3, golden, 'fused')
  print(f'fused compile={compile} rel sums {relative} params {fused}')
  assert relative.max() <= LOSS_TOLERANCE
  assert max(fused[n] for n in ('pol', 'cri', 'tpol', 'tcri')) <= PARAM_RMS_TOLERANCE
  # The original's per-step chain and its fused entry point agree.
  assert np.abs(golden['losses'].sum(0) / golden['fused_sums'] - 1).max() <= 1e-5
  assert td3.grad_steps == GOLDEN_STEPS


def test_fused_returns_means_and_honours_the_budget(golden, megabatch):
  td3 = golden_td3(False)
  Injected(td3, golden['normals'])
  out = td3.fused_step(megabatch, num_steps=3)
  got = np.array([out.critic.item(), out.policy.item()])
  assert td3.grad_steps == 3
  assert np.abs(got / golden['losses'][:3].mean(0) - 1).max() <= LOSS_TOLERANCE


@pytest.mark.parametrize('label', ['near_a', 'near_b'])
def test_target_noise_near_the_bounds_against_jax(golden, megabatch, label):
  """DESIGN 11.7: normals beyond the clip, targets next to the bounds, against JAX."""
  sigma, bias = float(golden[f'{label}/sigma']), float(golden[f'{label}/bias'])
  td3 = golden_td3(False, target_noise_sigma=sigma)
  head = td3.policy_target.policy_head[-1]
  head.bias.copy_(torch.tensor([bias, -bias, bias, -bias, bias, -bias, bias], device=DEVICE))
  for q in (td3.q1_target, td3.q2_target):
    q.qvalue[-1].weight.mul_(50.0)
  normals = golden[f'{label}/normals']
  assert (np.abs(normals) > td3.cfg.target_noise_clip).mean() > 0.3

  batch = megabatch.slice(0, GOLDEN_BATCH)
  Injected(td3, normals[None])
  with torch.no_grad():
    noisy = td3._target_action(batch.next_graph).cpu().numpy()
  error = np.abs(noisy - golden[f'{label}/noisy_action']).max()
  bound_hits = (np.abs(noisy) == 1.0).mean()
  print(f'{label} action error {error:.3e}, fraction exactly on a bound {bound_hits:.3f}')
  assert error <= 1e-6
  assert 0.05 < bound_hits < 1.0

  Injected(td3, normals[None])
  out = td3.train_step(batch)
  got = np.array([out.critic.item(), out.policy.item()])
  relative = np.abs(got / golden[f'{label}/losses'] - 1)
  errors = param_errors(td3, golden, label)
  print(f'{label} rel losses {relative} params {errors}')
  assert relative.max() <= LOSS_TOLERANCE
  assert max(errors[n] for n in ('pol', 'cri', 'tpol', 'tcri')) <= PARAM_RMS_TOLERANCE


def test_noise_shows_in_the_critic_loss(golden, megabatch):
  """The near-bound golden can only catch noise bugs if noise moves the loss."""
  batch = megabatch.slice(0, GOLDEN_BATCH)
  sigma, bias = float(golden['near_a/sigma']), float(golden['near_a/bias'])
  td3 = golden_td3(False, target_noise_sigma=0.0)
  td3.policy_target.policy_head[-1].bias.copy_(torch.tensor([bias, -bias, bias, -bias, bias, -bias, bias], device=DEVICE))
  for q in (td3.q1_target, td3.q2_target):
    q.qvalue[-1].weight.mul_(50.0)
  Injected(td3, golden['near_a/normals'][None])
  no_noise = td3.train_step(batch).critic.item()
  assert sigma > 0
  assert abs(no_noise / golden['near_a/losses'][0] - 1) > 10 * LOSS_TOLERANCE


def small_td3(**kwargs) -> TD3:
  defaults = dict(batch_size=4, fused_steps=2, compile=False, matmul_precision='highest', bf16=False)
  return TD3(SMALL, TrainCfg(**{**defaults, **kwargs}), Topology(EnvCfg(), DEVICE), DEVICE)


def test_noise_is_clipped_before_scaling_and_then_to_the_bounds(megabatch):
  td3 = small_td3(target_noise_sigma=0.1, target_noise_clip=0.4)
  batch = Graph(*[x[:1] for x in megabatch.graph])
  values = torch.tensor([-3.0, -0.5, -0.4, -0.1, 0.0, 0.3, 2.5], device=DEVICE).expand(1, 4, 7)
  td3._draw_normals = lambda like: values
  # Force the raw target action next to +1 and -1 through the head's bias.
  td3.policy_target.policy_head[-1].weight.zero_()
  head = td3.policy_target.policy_head[-1].bias
  for bias, raw in ((10.0, 1.0), (-10.0, -1.0), (0.0, 0.0)):
    head.fill_(bias)
    with torch.no_grad():
      got = td3._target_action(batch)
    want = (raw + 0.1 * values.clamp(-0.4, 0.4)).clamp(-1.0, 1.0)
    assert (got - want).abs().max() <= 1e-6
    assert (got - raw).abs().max() <= 0.04 + 1e-6


def test_lr_schedule_values():
  td3 = small_td3()
  assert td3.policy_lr(0) == 5e-5 and td3.critic_lr(0) == 5e-5
  for step in (1, 1000, 100_000, 1_878_000):
    assert td3.policy_lr(step) == pytest.approx(5e-5 * 0.98 ** (step / 1e5), rel=1e-12)
    assert td3.critic_lr(step) == pytest.approx(5e-5 * 0.95 ** (step / 1e5), rel=1e-12)
  # Continuous, not a staircase, and the values the reference run logged at this step.
  assert td3.policy_lr(50_000) == pytest.approx(5e-5 * 0.98 ** 0.5, rel=1e-12)
  assert td3.policy_lr(1_878_000) == pytest.approx(3.4213e-5, abs=1e-9)
  assert td3.critic_lr(1_878_000) == pytest.approx(1.9082e-5, abs=1e-9)
  assert td3.exploration_sigma(0) == 0.7
  assert td3.exploration_sigma(100_000) == pytest.approx(0.56, rel=1e-12)
  assert td3.exploration_sigma(1_000_000) == pytest.approx(0.0752, abs=1e-4)


def test_optimizer_uses_schedule_of_the_update_count(megabatch):
  td3 = small_td3(fused_steps=2)
  td3.fused_step(megabatch, num_steps=2)
  assert td3.critic_optimizer.param_groups[0]['lr'] == td3.critic_lr(1)
  assert td3.policy_optimizer.param_groups[0]['lr'] == td3.policy_lr(1)


def test_adam_settings():
  group = small_td3().critic_optimizer.param_groups[0]
  assert group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8
  assert group['weight_decay'] == 0 and not group['amsgrad']


def test_schedule_by_samples_scales_tau_lr_and_sigma():
  batch_size, world_size = 512, 4
  k = batch_size * world_size / 128
  cfg = TrainCfg(batch_size=batch_size, schedule_by_samples=True, compile=False, lr_mult=2.0)
  td3 = TD3(SMALL, cfg, Topology(EnvCfg(), DEVICE), DEVICE, world_size=world_size)
  assert td3.clock_scale == k == 16.0
  assert td3.tau_critic == pytest.approx(1 - (1 - 8e-5) ** k, rel=1e-12)
  assert td3.tau_policy == pytest.approx(1 - (1 - 4e-5) ** k, rel=1e-12)
  assert td3.policy_lr(1000) == pytest.approx(2.0 * 5e-5 * 0.98 ** (k * 1000 / 1e5), rel=1e-12)
  assert td3.critic_lr(1000) == pytest.approx(2.0 * 5e-5 * 0.95 ** (k * 1000 / 1e5), rel=1e-12)
  assert td3.exploration_sigma(1000) == pytest.approx(0.7 * 0.8 ** (k * 1000 / 1e5), rel=1e-12)


def _megabatch_small(megabatch: Batch, batch_size: int, steps: int) -> Batch:
  repeat = lambda x: x.repeat(1 + batch_size * steps // len(x), *[1] * (x.ndim - 1))[: batch_size * steps]
  return Batch(Graph(*map(repeat, megabatch.graph)), Graph(*map(repeat, megabatch.next_graph)),
               repeat(megabatch.action), repeat(megabatch.reward), repeat(megabatch.terminal))


def test_k_equal_one_is_exactly_parity(megabatch):
  """B=128 on one rank is bit for bit parity. B=64 on two ranks has the same clock."""
  data = _megabatch_small(megabatch, 128, 2)
  torch.manual_seed(1)
  parity = small_td3(batch_size=128)
  torch.manual_seed(1)
  scaled = small_td3(batch_size=128, schedule_by_samples=True)
  assert scaled.clock_scale == 1.0
  assert parity.tau_critic == scaled.tau_critic == pytest.approx(8e-5, rel=1e-10)
  a = parity.fused_step(data, 2)
  b = scaled.fused_step(data, 2)
  assert a.critic.item() == b.critic.item() and a.policy.item() == b.policy.item()
  assert parity.policy_lr(10**6) == scaled.policy_lr(10**6)
  for net in ('policy', 'q1', 'q2', 'policy_target', 'q1_target', 'q2_target'):
    for x, y in zip(getattr(parity, net).parameters(), getattr(scaled, net).parameters()):
      assert torch.equal(x, y), net
  two_ranks = TrainCfg(batch_size=64, schedule_by_samples=True, compile=False)
  assert TD3(SMALL, two_ranks, Topology(EnvCfg(), DEVICE), DEVICE, world_size=2).clock_scale == 1.0


def test_flat_gradient_hook(megabatch):
  data = _megabatch_small(megabatch, 4, 2)
  plain = small_td3()
  sizes = []

  def identity(flat: torch.Tensor) -> torch.Tensor:
    sizes.append(flat.numel())
    assert flat.ndim == 1
    return flat

  hooked = small_td3()
  hooked.grad_hook = identity
  hooked.load_state_dict(plain.state_dict())
  plain.fused_step(data, 2)
  hooked.fused_step(data, 2)
  critic_size = sum(p.numel() for p in hooked._critic_params)
  policy_size = sum(p.numel() for p in hooked._policy_params)
  assert sizes == [critic_size, policy_size] * 2
  for x, y in zip(plain.policy.parameters(), hooked.policy.parameters()):
    assert torch.equal(x, y)
  for x, y in zip(plain.q1.parameters(), hooked.q1.parameters()):
    assert torch.equal(x, y)


def test_hook_output_is_what_gets_applied(megabatch):
  """A real all-reduce replaces the vector, and absent gradients arrive as zeros."""
  data = _megabatch_small(megabatch, 4, 1)
  seen = []

  def zero(flat: torch.Tensor) -> torch.Tensor:
    seen.append(flat.clone())
    return torch.zeros_like(flat)

  td3 = small_td3(fused_steps=1)
  td3.grad_hook = zero
  before = [p.clone() for p in td3._critic_params]
  td3.train_step(data)
  for x, y in zip(before, td3._critic_params):
    assert torch.equal(x, y)
  assert seen[0].abs().sum() > 0
  # The reward heads are registered, unused, and so exactly zero in the critic vector.
  offset = 0
  for name, p in [*td3.q1.named_parameters(), *td3.q2.named_parameters()]:
    if name.startswith('rewards'):
      assert seen[0][offset : offset + p.numel()].abs().max() == 0
    offset += p.numel()


def test_parameters_without_effect_stay_put_and_critics_ignore_policy_step(megabatch):
  data = _megabatch_small(megabatch, 4, 1)
  td3 = small_td3(fused_steps=1)
  q1_before = {k: v.clone() for k, v in td3.q1.state_dict().items()}
  policy_before = {k: v.clone() for k, v in td3.policy.state_dict().items()}
  td3.train_step(data)
  for k, v in td3.policy.state_dict().items():
    if k.startswith('gnn.update_global'):
      assert torch.equal(v, policy_before[k]), k
  assert not torch.equal(td3.policy.state_dict()['policy_head.0.weight'], policy_before['policy_head.0.weight'])
  for k, v in td3.q1.state_dict().items():
    if k.startswith('rewards'):
      assert torch.equal(v, q1_before[k]), k
  assert not torch.equal(td3.q1.state_dict()['qvalue.0.weight'], q1_before['qvalue.0.weight'])
  assert all(p.requires_grad for p in td3.q1.parameters())
  for frozen, live in zip(td3.q1_frozen.parameters(), td3.q1.parameters()):
    assert not frozen.requires_grad and frozen.data_ptr() == live.data_ptr()


def test_terminal_cuts_the_bootstrap(megabatch):
  """Terminal on every transition: target = reward, whatever the target nets say."""
  data = _megabatch_small(megabatch, 4, 1)
  data = data._replace(terminal=torch.ones_like(data.terminal))
  td3 = small_td3(fused_steps=1)
  with torch.no_grad():
    q1 = td3.q1(data.graph, data.action)
    q2 = td3.q2(data.graph, data.action)
  want = (data.reward - q1).square().mean() + (data.reward - q2).square().mean()
  got = td3.train_step(data).critic
  assert got.item() == pytest.approx(want.item(), rel=1e-6)


def test_checkpoint_round_trip_continues_identically(megabatch, tmp_path):
  data = _megabatch_small(megabatch, 4, 2)
  first = small_td3()
  first.fused_step(data, 2)
  torch.save(first.state_dict(), tmp_path / 'learner.pt')
  second = small_td3(seed=99)
  second.load_state_dict(torch.load(tmp_path / 'learner.pt', map_location=DEVICE, weights_only=True))
  assert second.grad_steps == 2
  a = first.fused_step(data, 2)
  b = second.fused_step(data, 2)
  assert a.critic.item() == b.critic.item() and a.policy.item() == b.policy.item()
  for x, y in zip(first.policy.parameters(), second.policy.parameters()):
    assert torch.equal(x, y)


def test_bf16_learns_with_float32_parameters_and_acts_in_float32(megabatch):
  data = _megabatch_small(megabatch, 4, 2)
  plain = small_td3().fused_step(data, 2)
  td3 = small_td3(bf16=True)
  out = td3.fused_step(data, 2)
  assert out.critic.isfinite() and out.policy.isfinite()
  assert out.critic.dtype == torch.float32
  assert out.critic.item() != plain.critic.item()  # The arithmetic did change.
  assert all(p.dtype == torch.float32 for p in td3.policy.parameters())
  assert td3.act(Graph(*[x[:4] for x in data.graph])).dtype == torch.float32


def test_policy_and_critics_have_distinct_init():
  td3 = small_td3()
  assert not torch.equal(td3.q1.qvalue[0].weight, td3.q2.qvalue[0].weight)
  assert all(torch.equal(a, b) for a, b in zip(td3.q1.parameters(), td3.q1_target.parameters()))


def test_goldens_are_small():
  for path in GOLDEN.glob('agent_*.npz'):
    assert path.stat().st_size < 5_000_000, path


def test_compiled_step_does_not_recompile(megabatch):
  from torch._dynamo.utils import counters

  data = _megabatch_small(megabatch, 4, 6)
  td3 = small_td3(compile=True, fused_steps=6)
  td3.fused_step(data, 2)
  graphs = counters['stats']['unique_graphs']
  td3.fused_step(data.slice(8, 24), 4)
  assert counters['stats']['unique_graphs'] == graphs
