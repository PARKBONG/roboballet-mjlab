"""The learner's bf16 arithmetic against the original's on GPU, value by value.

agents/networks.py:126-152 runs Dense, LayerNorm and GELU in bf16 with rounding at every operation.
The golden is the original's flax MLP under JAX on GPU, from golden/agent_bf16_ops_make.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from roboballet_mjlab.agent import networks
from roboballet_mjlab.agent.networks import GELU, MLP, Dense, LayerNorm

GOLDEN = np.load(Path(__file__).parent / 'golden' / 'agent_bf16_ops.npz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# About 1 element in 7000 lands on the other side of a rounding boundary, from LayerNorm's reductions.
MLP_MAX_ERROR = 2e-5
MLP_MEAN_ERROR = 1e-8
GRAD_MAX_ERROR = 5e-4
GRAD_RELATIVE_ERROR = 5e-3


def tensor(key: str) -> torch.Tensor:
  return torch.as_tensor(GOLDEN[key].copy(), device=DEVICE)


def golden_mlp() -> MLP:
  mlp = MLP(128, [64, 64, 7], 1e-6, last_scale=0.005)
  dense = norm = 0
  with torch.no_grad():
    for layer in mlp:
      if isinstance(layer, nn.Linear):
        layer.weight.copy_(tensor(f'param/Dense_{dense}/kernel').T)
        layer.bias.copy_(tensor(f'param/Dense_{dense}/bias'))
        dense += 1
      elif isinstance(layer, nn.LayerNorm):
        layer.weight.copy_(tensor(f'param/LayerNorm_{norm}/scale'))
        layer.bias.copy_(tensor(f'param/LayerNorm_{norm}/bias'))
        norm += 1
  return mlp.to(DEVICE)


def autocast():
  return torch.autocast(DEVICE, dtype=torch.bfloat16)


@pytest.mark.parametrize('compiled', [False, True])
def test_gelu_matches_on_every_bf16_value(compiled):
  x = tensor('gelu_in').bfloat16()
  assert len(x) > 30_000
  gelu = GELU(approximate='tanh').to(DEVICE)
  y = (torch.compile(gelu) if compiled else gelu)(x)
  assert y.dtype == torch.bfloat16
  assert torch.equal(y.float(), tensor('gelu_out'))


@pytest.mark.parametrize('compiled', [False, True])
def test_gelu_in_float32_is_the_plain_one(compiled):
  x = torch.linspace(-6, 6, 4097, device=DEVICE)
  gelu = GELU(approximate='tanh').to(DEVICE)
  y = (torch.compile(gelu) if compiled else gelu)(x)
  assert y.dtype == torch.float32
  plain = torch.nn.functional.gelu(x, approximate='tanh')
  # A compiled kernel may contract the last bit, which it also did before this class existed.
  assert torch.equal(y, plain) if not compiled else (y - plain).abs().max().item() <= 1e-5


def test_layernorm_rounds_only_in_bf16():
  norm = LayerNorm(32, eps=1e-6).to(DEVICE)
  x = torch.randn(64, 32, device=DEVICE)
  assert norm(x).dtype == torch.float32
  rounded = norm(x.bfloat16())
  assert rounded.dtype == torch.bfloat16
  plain = nn.LayerNorm(32, eps=1e-6).to(DEVICE)(x.bfloat16().float())
  assert (rounded.float() - plain).abs().max().item() <= 0.02


def test_dense_rounds_the_product_before_adding_the_bias():
  dense = Dense(16, 8).to(DEVICE)
  with torch.no_grad():
    dense.bias.fill_(3.14159)
  x = torch.randn(32, 16, device=DEVICE).bfloat16()
  product = (x @ dense.weight.bfloat16().T)
  assert torch.equal(dense(x), product + dense.bias.bfloat16())
  assert dense(x).dtype == torch.bfloat16
  assert dense(x.float()).dtype == torch.float32


@pytest.mark.parametrize('compiled', [False, True])
def test_mlp_output_matches_the_original(compiled):
  mlp = golden_mlp()
  net = torch.compile(mlp) if compiled else mlp
  with torch.no_grad(), autocast():
    y = net(tensor('mlp_in'))
  assert y.dtype == torch.float32  # Outputs get summed, networks.py:151-152.
  error = (y - tensor('mlp_out')).abs()
  assert error.max().item() <= MLP_MAX_ERROR
  assert error.mean().item() <= MLP_MEAN_ERROR


@pytest.mark.parametrize('compiled', [False, True])
def test_mlp_gradient_matches_the_original(compiled):
  mlp = golden_mlp()
  net = torch.compile(mlp) if compiled else mlp
  with autocast():
    loss = (net(tensor('mlp_in')) * tensor('loss_weights')).sum(-1).mean()
  loss.backward()
  dense = norm = 0
  pairs = []
  for layer in mlp:
    if isinstance(layer, nn.Linear):
      pairs += [(layer.weight.grad, tensor(f'grad/Dense_{dense}/kernel').T), (layer.bias.grad, tensor(f'grad/Dense_{dense}/bias'))]
      dense += 1
    elif isinstance(layer, nn.LayerNorm):
      pairs += [(layer.weight.grad, tensor(f'grad/LayerNorm_{norm}/scale')), (layer.bias.grad, tensor(f'grad/LayerNorm_{norm}/bias'))]
      norm += 1
  ours, theirs = torch.cat([a.flatten() for a, _ in pairs]), torch.cat([b.flatten() for _, b in pairs])
  error = (ours - theirs).abs()
  relative = (error.norm() / theirs.norm()).item()
  print(f'gradient error: max {error.max().item():.2e}, mean {error.mean().item():.2e}, relative {relative:.2e}, '
        f'differ {(error > 0).float().mean().item():.4f}')
  assert error.max().item() <= GRAD_MAX_ERROR
  assert relative <= GRAD_RELATIVE_ERROR


def test_compile_keeps_the_rounding():
  assert torch._inductor.config.emulate_precision_casts is True
  mlp = golden_mlp()
  with torch.no_grad(), autocast():
    assert torch.equal(torch.compile(mlp)(tensor('mlp_in')), mlp(tensor('mlp_in')))


def test_float32_forward_has_no_bf16_on_the_way():
  mlp = golden_mlp()
  seen = []
  handles = [layer.register_forward_hook(lambda m, i, o: seen.append(o.dtype)) for layer in mlp]
  mlp(tensor('mlp_in'))
  for handle in handles:
    handle.remove()
  assert seen and all(kind == torch.float32 for kind in seen)
  assert networks.GELU is GELU


def test_heads_of_the_real_nets_have_the_flax_init():
  """DESIGN 11.9. These are the constructor arguments a wiring mistake would change."""
  from roboballet_mjlab.agent.features import Topology
  from roboballet_mjlab.agent.networks import CriticNet, PolicyNet
  from roboballet_mjlab.config import EnvCfg, ModelCfg

  cfg = ModelCfg()
  topo = Topology(EnvCfg(), 'cpu')
  torch.manual_seed(3)
  policy, critic = PolicyNet(cfg, topo), CriticNet(cfg, topo)
  for head in (policy.policy_head, critic.qvalue, critic.rewards):
    dense = [layer for layer in head if isinstance(layer, nn.Linear)]
    last = dense[-1].weight.detach()
    fan_in = last.shape[1]
    expected = np.sqrt(cfg.head_last_scale / fan_in)
    assert last.std().item() == pytest.approx(expected, rel=0.25)
    assert last.abs().max().item() <= 2.0 * expected / 0.87962566103423978 * (1 + 1e-6)
    for layer in dense[:-1]:
      weight = layer.weight.detach()
      assert weight.std().item() == pytest.approx(np.sqrt(1.0 / weight.shape[1]), rel=0.05)
