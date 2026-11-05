"""E10 of paper_baseline.md 10.1: the sampler on the original tool against the stored .pb.

Needs the pools from:
  gen_scenes --robots 4 --targets 10 --obstacles 30 --split train --count 50000 --seed 1 --arm-model mesh_tool
  gen_scenes --robots 8 --targets 40 --obstacles 30 --split train --count 50000 --seed 1 --arm-model mesh_tool
"""

from __future__ import annotations

import numpy as np
import pytest

from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.scenes.io import load_pool, load_start_states
from tests.test_scenes_common import START_STATES, e10_marginals, face_stats, ks
from tests.test_sim_common import DEVICE

KS_LIMIT = 0.02
CONFIGS = ('4r_10t_30o', '8r_40t_30o')
_tables: dict[str, dict] = {}


def _table(config: str) -> dict:
  """The generated pool and the marginals of pool and stored data. Built once per config."""
  if config not in _tables:
    path = scene_pool(f'{config}_train_mesh_tool')
    if not path.exists():
      pytest.fail(f'Missing {path}: this is a failure of E10, not a skip')
    pool = load_pool(path, DEVICE)
    stored = load_start_states(START_STATES / f'{config}.pb', DEVICE)
    mine, theirs = e10_marginals(pool.scene), e10_marginals(stored)
    _tables[config] = {'pool': pool, 'stored': stored, 'mine': mine, 'theirs': theirs,
                       'ks': {k: ks(mine[k], theirs[k]) for k in mine}}
  return _tables[config]


def _names(config: str) -> list[str]:
  robots = int(config.split('r')[0])
  rails = {4: ['dy_0_3', 'dy_1_2'], 8: ['dy_0_5', 'dy_1_4', 'dy_2_7', 'dy_3_6']}[robots]
  return ([f'{field}_{axis}' for field in ('spans', 'centroid', 'target') for axis in 'xyz']
          + ['target_radius'] + [f'base_y_{r}' for r in range(robots)] + rails + ['inward_angle_deg'])


CASES = [(config, name) for config in CONFIGS for name in _names(config)]


@pytest.mark.parametrize('config,name', CASES)
def test_marginal(config, name):
  table = _table(config)
  value = table['ks'][name]
  print(f'{config} {name}: KS {value:.4f}, n {len(table["mine"][name])} vs {len(table["theirs"][name])}')
  assert value <= KS_LIMIT


@pytest.mark.parametrize('config', CONFIGS)
def test_marginal_names_are_exhaustive(config):
  assert set(_table(config)['ks']) == set(_names(config))


@pytest.mark.parametrize('config', CONFIGS)
def test_offset_to_1e4(config):
  """E10: every target sits 3 cm outside a face, to 1e-4 m."""
  has_owner, residual, _ = face_stats(_table(config)['pool'].scene)
  print(f'{config}: owner fraction {has_owner.float().mean().item():.6f}, worst residual {residual.max().item():.2e}')
  assert has_owner.all()
  assert residual.max() < 1e-4


@pytest.mark.parametrize('config', CONFIGS)
def test_stored_data_meets_the_same_offset(config):
  """The test must accept the reference, or the check above would mean nothing."""
  has_owner, _, angle = face_stats(_table(config)['stored'])
  assert has_owner.float().mean() > 0.9999
  assert abs(float(np.median(angle.cpu().numpy())) - 15.85) < 0.1
