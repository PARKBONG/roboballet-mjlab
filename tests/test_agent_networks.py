"""Parameter counts, init and E7 against the reference checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import MLP, CriticNet, PolicyNet, convert_flat_checkpoint, flax_names, load_checkpoint
from roboballet_mjlab.config import EnvCfg, ModelCfg
from tests.golden.agent_load import CHECKPOINT, DEVICE, GOLDEN, load_e6

E7_TOLERANCE = 2e-3  # paper_baseline 10, E7
# Tighter than E7, from what was measured. The action maximum is left at the E7 value:
# against a float64 forward both float32 results are off at the same rare steep inputs,
# by up to 3e-4 here and 1e-4 in JAX, while the mean stays near 1e-6.
OBSERVED_ACTION_MEAN = 2e-6
OBSERVED_ACTION_P999 = 5e-5
OBSERVED_Q = 2e-6


def count(net: nn.Module) -> int:
  return sum(p.numel() for p in net.parameters())


@pytest.fixture(scope='module')
def nets():
  topo = Topology(EnvCfg(), 'cpu')
  return PolicyNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo)


def test_parameter_counts(nets):
  policy, q1, q2 = nets
  assert count(policy) == 8_949_575
  assert count(q1) + count(q2) == 17_980_164


def test_module_counts_of_algo_spec(nets):
  policy, q1, _ = nets
  assert count(policy.gnn.embed_node) == 1_591_296
  assert count(q1.gnn.embed_node) == 1_594_880
  assert count(policy.gnn.embed_edge) == 339_456
  assert count(policy.gnn.embed_global) == 1_583_616
  assert count(policy.gnn.update_edge) + count(policy.gnn.update_node) + count(policy.gnn.update_global) == 5_397_504
  assert count(policy.policy_head) == 37_703
  assert count(q1.qvalue) == count(q1.rewards) == 37_313


def test_layer_settings(nets):
  policy, _, _ = nets
  layers = list(policy.gnn.embed_node)
  assert isinstance(layers[-1], nn.Linear)
  assert sum(isinstance(m, nn.Linear) for m in layers) == 7
  assert sum(isinstance(m, nn.LayerNorm) for m in layers) == 6
  for module in policy.modules():
    if isinstance(module, nn.LayerNorm):
      assert module.eps == 1e-6
    if isinstance(module, nn.GELU):
      assert module.approximate == 'tanh'


def test_init_matches_flax(nets):
  policy, _, _ = nets
  torch.manual_seed(0)
  weight = MLP(512, [512, 7], 1e-6, last_scale=0.005)
  dense = weight[0].weight.detach()
  last = weight[-1].weight.detach()
  # Variance scaling: sqrt(scale / fan_in) after correcting the truncation.
  assert dense.std().item() == pytest.approx(np.sqrt(1 / 512), rel=0.01)
  assert last.std().item() == pytest.approx(np.sqrt(0.005 / 512), rel=0.05)
  bound = 2.0 * np.sqrt(1 / 512) / 0.87962566103423978
  assert dense.abs().max().item() <= bound * (1 + 1e-6)
  assert dense.abs().max().item() >= bound * 0.98
  for m in policy.modules():
    if isinstance(m, nn.Linear):
      assert m.bias.abs().max() == 0
    if isinstance(m, nn.LayerNorm):
      assert (m.weight == 1).all() and (m.bias == 0).all()


def test_flax_names_cover_all_leaves(nets):
  policy, q1, _ = nets
  # Leaf counts seen in the reference checkpoint: 158 and 168 per critic.
  assert set(flax_names(policy, 'p')) == set(policy.state_dict())
  assert len(flax_names(policy, 'p')) == 158
  assert len(flax_names(q1, 'p')) == 168
  assert set(flax_names(q1, 'p')) == set(q1.state_dict())


def _require_checkpoint():
  if not CHECKPOINT.exists():
    pytest.fail(f'{CHECKPOINT} is missing: see agent/networks.py for the conversion.')


@pytest.fixture(scope='module')
def reference():
  _require_checkpoint()
  topo = Topology(EnvCfg(), DEVICE)
  policy, q1, q2 = PolicyNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo)
  assert load_checkpoint(str(CHECKPOINT), policy, q1, q2) == 1_870_000
  return policy.to(DEVICE), q1.to(DEVICE), q2.to(DEVICE)


@pytest.mark.parametrize('compiled', [False, True])
def test_e7_reference_checkpoint(reference, compiled):
  """Same inputs and weights as JAX CPU float32, at matmul highest, 1000 graphs."""
  torch.set_float32_matmul_precision('highest')
  policy, q1, q2 = [torch.compile(n) if compiled else n for n in reference]
  e6 = load_e6('4r10t', shards=2)
  ref = np.load(GOLDEN / 'agent_e7.npz')
  action = torch.as_tensor(e6.raw['action'], device=DEVICE)
  with torch.no_grad():
    a = torch.cat([policy(type(e6.graph)(*[x[i:i + 100] for x in e6.graph])) for i in range(0, 1000, 100)])
    v1 = torch.cat([q1(type(e6.graph)(*[x[i:i + 100] for x in e6.graph]), action[i:i + 100]) for i in range(0, 1000, 100)])
    v2 = torch.cat([q2(type(e6.graph)(*[x[i:i + 100] for x in e6.graph]), action[i:i + 100]) for i in range(0, 1000, 100)])
  da = np.abs(a.cpu().numpy() - ref['action'])
  d1 = np.abs(v1.cpu().numpy() - ref['q1'])
  d2 = np.abs(v2.cpu().numpy() - ref['q2'])
  p999 = np.quantile(da, 0.999)
  print(f'E7 compiled={compiled} action max {da.max():.3e} p99.9 {p999:.3e} mean {da.mean():.3e} q1 max {d1.max():.3e}'
        f' q2 max {d2.max():.3e} saturated {np.mean(np.abs(ref["action"]) > 0.99):.3f}')
  assert da.max() <= E7_TOLERANCE and max(d1.max(), d2.max()) <= E7_TOLERANCE
  assert da.mean() <= OBSERVED_ACTION_MEAN and p999 <= OBSERVED_ACTION_P999
  assert max(d1.max(), d2.max()) <= OBSERVED_Q


def test_strict_loader_rejects_mismatch(reference, tmp_path):
  _require_checkpoint()
  checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
  key = next(iter(checkpoint['policy']))
  policy, q1, q2 = reference
  missing = {**checkpoint, 'policy': {k: v for k, v in checkpoint['policy'].items() if k != key}}
  torch.save(missing, tmp_path / 'missing.pt')
  with pytest.raises(RuntimeError, match='Missing key'):
    load_checkpoint(str(tmp_path / 'missing.pt'), policy, q1, q2)
  extra = {**checkpoint, 'q1': {**checkpoint['q1'], 'unexpected': torch.zeros(1)}}
  torch.save(extra, tmp_path / 'extra.pt')
  with pytest.raises(RuntimeError, match='Unexpected key'):
    load_checkpoint(str(tmp_path / 'extra.pt'), policy, q1, q2)


def test_converter_rejects_unused_leaves(tmp_path):
  np.savez(tmp_path / 'flat.npz', step=1, **{'policy/params/unknown/kernel': np.zeros((2, 2))})
  with pytest.raises(ValueError, match='leaves differ'):
    convert_flat_checkpoint(str(tmp_path / 'flat.npz'), str(tmp_path / 'out.pt'))
