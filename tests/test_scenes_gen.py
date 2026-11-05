"""scripts/gen_scenes.py end to end, on small pools in a temporary directory."""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from roboballet_mjlab.scenes.ik import IkBudget
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.scenes.io import Pool, load_pool, save_pool
from roboballet_mjlab.scenes.sampler import Generated, SamplerCfg, Witnesses
from roboballet_mjlab.scripts import gen_scenes
from roboballet_mjlab.structs import Scene
from tests.test_scenes_common import (
  FROZEN_BOXES, check_static_boxes, cube_collision_vertices, recheck_chunk, vertices_outside_boxes)
from tests.test_sim_common import DEVICE

COUNT = 40
DEFAULT_ARGS = ['--robots', '4', '--targets', '10', '--obstacles', '30', '--split', 'train',
                '--count', str(COUNT), '--seed', '1']


def _run(monkeypatch, out_dir, split='train', seed=1, count=COUNT, robots=4, targets=10):
  monkeypatch.setattr(gen_scenes, 'OUT_DIR', out_dir)
  path = gen_scenes.main(['--robots', str(robots), '--targets', str(targets), '--obstacles', '30',
                          '--split', split, '--count', str(count), '--seed', str(seed)])
  return load_pool(path, DEVICE)


def _same(a, b) -> bool:
  return (all(torch.equal(x, y) for x, y in zip(a.scene, b.scene))
          and torch.equal(a.witnesses.q, b.witnesses.q)
          and torch.equal(a.witnesses.robot, b.witnesses.robot))


def test_same_seed_same_pool_and_seeds_differ(monkeypatch, tmp_path):
  one = _run(monkeypatch, tmp_path / 'one')
  two = _run(monkeypatch, tmp_path / 'two')
  other = _run(monkeypatch, tmp_path / 'other', seed=2)
  heldout = _run(monkeypatch, tmp_path / 'heldout', split='heldout')
  assert _same(one, two)
  assert one.meta['c_check'] == two.meta['c_check']
  for different in (other, heldout):
    assert not torch.equal(one.scene.box_pos, different.scene.box_pos)
    assert not torch.equal(one.scene.base_pos, different.scene.base_pos)
    assert not torch.equal(one.scene.target_pos, different.scene.target_pos)


def test_stored_heldout_pool_reproduces_from_its_metadata(monkeypatch, tmp_path):
  """A fresh process state and the stored seed give the pool on disk, bit for bit."""
  stored = load_pool(scene_pool('4r_10t_30o_heldout'), DEVICE)
  again = _run(monkeypatch, tmp_path, split='heldout', seed=stored.meta['seed'], count=stored.meta['count'])
  assert _same(stored, again)


def test_rerun_resumes_from_shards(monkeypatch, tmp_path, capsys):
  """A count past one shard: kill it after the first shard, rerun, get the same pool."""
  count = gen_scenes.WORLDS + 50
  whole = _run(monkeypatch, tmp_path / 'whole', count=count)
  capsys.readouterr()
  partial_dir = tmp_path / 'partial'
  shards = partial_dir / '4r_10t_30o_train.shards'
  args = gen_scenes.parse_args(['--robots', '4', '--targets', '10', '--obstacles', '30',
                                '--split', 'train', '--count', str(count), '--seed', '1'])
  monkeypatch.setattr(gen_scenes, 'OUT_DIR', partial_dir)
  gen_scenes.generate_shards(args, shards, gen_scenes.WORLDS - 200, DEVICE)
  assert [p.name for p in shards.iterdir()] == ['00000.pt']
  capsys.readouterr()
  resumed = _run(monkeypatch, partial_dir, count=count)
  printed = capsys.readouterr().out
  assert 'shard 0:' not in printed and 'shard 1:' in printed
  assert _same(whole, resumed)
  assert resumed.meta['generation_s'] > 0
  assert not shards.exists(), 'Shards go once the pool exists'


def test_pool_records_what_made_it(monkeypatch, tmp_path):
  pool = _run(monkeypatch, tmp_path, seed=5)
  meta = pool.meta
  assert meta['seed'] == 5 and meta['split'] == 'train' and meta['count'] == COUNT
  assert meta['arm_model'] == 'spheres_stick'
  assert meta['budget']['restarts'] == 2 and meta['budget']['iters'] == 40
  assert len(meta['git_revision']) == 40
  assert 0 < meta['rates']['target_accept'] < 1 and 0 < meta['rates']['box_accept'] < 1
  assert meta['counts']['target_failures'] + meta['counts']['box_failures'] + meta['counts']['rail_failures'] == 0
  assert meta['c_check']['witness_missing'] == 0
  assert len(pool.scene.q0) == COUNT


def _args(**changes):
  args = gen_scenes.parse_args(DEFAULT_ARGS)
  vars(args).update(changes)
  return args


@pytest.fixture(scope='module')
def real_shard(tmp_path_factory):
  """A directory holding one real shard of the default arguments. Tests must not change it."""
  path = tmp_path_factory.mktemp('shards')
  gen_scenes.generate_shards(_args(), path, 1, DEVICE)
  assert [p.name for p in path.iterdir()] == ['00000.pt']
  return path


def _assert_refused(shard_dir, args):
  """Resuming raises, names both fingerprints, and leaves the shard where it was."""
  shard = shard_dir / '00000.pt'
  before = shard.read_bytes()
  with pytest.raises(RuntimeError) as failure:
    gen_scenes.generate_shards(args, shard_dir, 1, DEVICE)
  message = str(failure.value)
  assert str(load_pool(shard, 'cpu').meta['fingerprint']) in message
  assert str(gen_scenes.fingerprint(args)) in message
  assert shard.read_bytes() == before


def test_shard_stores_its_fingerprint(real_shard):
  stored = load_pool(real_shard / '00000.pt', 'cpu').meta['fingerprint']
  assert stored == gen_scenes.fingerprint(_args())
  assert set(stored) == {'seed', 'split', 'robots', 'targets', 'obstacles', 'arm_model', 'sampler', 'budget',
                         'shard_size', 'code_sha256'}
  assert stored['seed'] == 1 and stored['split'] == 'train' and stored['arm_model'] == 'spheres_stick'
  assert stored['robots'] == 4 and stored['targets'] == 10 and stored['obstacles'] == 30
  assert stored['budget']['restarts'] == 2 and stored['sampler']['max_box_draws'] == 100
  assert stored['shard_size'] == gen_scenes.WORLDS and len(stored['code_sha256']) == 64


def test_same_configuration_resumes_whatever_the_count(real_shard, capsys):
  capsys.readouterr()
  pools = gen_scenes.generate_shards(_args(count=999), real_shard, 1, DEVICE)
  assert len(pools) == 1 and 'shard 0:' not in capsys.readouterr().out


@pytest.mark.parametrize('change', [
  {'seed': 2}, {'split': 'heldout'}, {'robots': 8}, {'targets': 40}, {'obstacles': 20}, {'arm_model': 'mesh_tool'},
], ids=lambda change: next(iter(change)))
def test_resume_refuses_a_shard_of_other_arguments(real_shard, change):
  _assert_refused(real_shard, _args(**change))


def test_resume_refuses_another_budget(real_shard, monkeypatch):
  monkeypatch.setattr(gen_scenes, 'IkBudget', lambda: IkBudget(restarts=3))
  _assert_refused(real_shard, _args())


def test_resume_refuses_another_sampler(real_shard, monkeypatch):
  monkeypatch.setattr(gen_scenes, 'SamplerCfg', lambda: SamplerCfg(draws_per_missing=3))
  _assert_refused(real_shard, _args())


def test_resume_refuses_another_shard_size(real_shard, monkeypatch):
  monkeypatch.setattr(gen_scenes, 'WORLDS', gen_scenes.WORLDS // 2)
  _assert_refused(real_shard, _args())


@pytest.mark.parametrize('arm,asset', [('spheres_stick', 'panda_spheres_stick.xml'), ('mesh_tool', 'panda_nohand.xml')])
def test_the_hash_covers_the_solver_the_sampler_and_the_arm(arm, asset):
  assert [p.name for p in gen_scenes.hashed_files(arm)] == ['ik.py', 'sampler.py', asset]


@pytest.mark.parametrize('which', [0, 1, 2], ids=['ik', 'sampler', 'arm'])
def test_resume_refuses_changed_code_or_asset(real_shard, monkeypatch, tmp_path, which):
  copies = []
  for path in gen_scenes.hashed_files('spheres_stick'):
    copies.append(tmp_path / path.name)
    shutil.copy(path, copies[-1])
  monkeypatch.setattr(gen_scenes, 'hashed_files', lambda arm: copies)
  assert gen_scenes.fingerprint(_args()) == load_pool(real_shard / '00000.pt', 'cpu').meta['fingerprint']
  copies[which].write_bytes(copies[which].read_bytes() + b' ')
  _assert_refused(real_shard, _args())


def test_resume_refuses_a_shard_with_no_fingerprint(real_shard, tmp_path):
  """What the code before the fingerprint left behind, and the review's stale-resume case."""
  old = load_pool(real_shard / '00000.pt', 'cpu')
  save_pool(tmp_path / '00000.pt', Pool(old.scene, old.witnesses, {'counts': old.meta['counts']}))
  with pytest.raises(RuntimeError) as failure:
    gen_scenes.generate_shards(_args(seed=2), tmp_path, 1, DEVICE)
  assert 'None' in str(failure.value) and str(gen_scenes.fingerprint(_args(seed=2))) in str(failure.value)
  assert (tmp_path / '00000.pt').exists()


def test_cli_refuses_a_stale_shard_directory(real_shard, monkeypatch, tmp_path):
  shutil.copytree(real_shard, tmp_path / '4r_10t_30o_train.shards')
  with pytest.raises(RuntimeError, match='fingerprint'):
    _run(monkeypatch, tmp_path, seed=2)
  assert not (tmp_path / '4r_10t_30o_train.pt').exists()


_SCENE_SHAPES = ((4, 3), (4, 3, 3), (30, 3), (30, 3, 3), (30, 3), (10, 3), (10, 3, 3), (4, 7))


def _fake_generate(monkeypatch, scenes):
  """Replaces the generator with one keeping `scenes` of its attempts. Stops an unbounded loop."""
  calls = []

  def generate(*_):
    calls.append(1)
    if len(calls) > 200:
      pytest.fail('The shard loop is not bounded')
    scene = Scene(*[torch.zeros(scenes, *shape) for shape in _SCENE_SHAPES])
    witnesses = Witnesses(torch.zeros(scenes, 10, 7), torch.zeros(scenes, 10, dtype=torch.long))
    counts = {'attempted': gen_scenes.WORLDS, 'rail_failures': 0, 'box_failures': 0,
              'target_failures': gen_scenes.WORLDS - scenes, 'target_draws': 7777, 'target_accepted': 10 * scenes}
    return Generated(scene, witnesses, counts)

  monkeypatch.setattr(gen_scenes, 'build', lambda *_: (None, None))
  monkeypatch.setattr(gen_scenes, 'generate', generate)
  return calls


def test_zero_progress_raises_with_the_counts(monkeypatch, tmp_path):
  calls = _fake_generate(monkeypatch, 0)
  with pytest.raises(RuntimeError, match=r"Shard 0 .*no scenes.*'target_failures': 4096.*'target_draws': 7777"):
    gen_scenes.generate_shards(_args(), tmp_path, 1, DEVICE)
  assert len(calls) == 1
  assert not list(tmp_path.iterdir()), 'An empty shard is not saved'


def test_slow_progress_stops_at_the_attempt_bound(monkeypatch, tmp_path):
  """One scene per shard never reaches 1,000 scenes within the bound, so the loop must end."""
  calls = _fake_generate(monkeypatch, 1)
  needed = 1000
  limit = gen_scenes.MAX_ATTEMPTS_PER_SCENE * max(needed, gen_scenes.WORLDS)
  shards = limit // gen_scenes.WORLDS
  with pytest.raises(RuntimeError, match=rf"{limit} attempts gave {shards} of {needed} scenes.*'attempted': {limit}"):
    gen_scenes.generate_shards(_args(), tmp_path, needed, DEVICE)
  assert len(calls) == shards


def test_real_sampler_that_fails_everything_raises(monkeypatch, tmp_path):
  """No target rounds at all: the real generator reports 4096 failures and no draws."""
  monkeypatch.setattr(gen_scenes, 'SamplerCfg', lambda: SamplerCfg(max_target_rounds=0))
  with pytest.raises(RuntimeError, match=r"no scenes.*'target_failures': 4096"):
    gen_scenes.generate_shards(_args(), tmp_path, 1, DEVICE)


def _static_case():
  """Ten good scenes of a stored pool, with their witnesses, as arrays for the recheck."""
  pool = load_pool(scene_pool('4r_10t_30o_static'), 'cpu')
  return pool.scene, pool.witnesses


def _rows(scene, witnesses):
  arrays = {name: field.numpy() for name, field in scene._asdict().items()}
  return recheck_chunk((arrays, witnesses.q.numpy(), witnesses.robot.numpy(), 'spheres_stick'))


def test_final_rejection_drops_every_kind_of_bad_scene():
  """Real rows from the C MuJoCo recheck, with one defect planted in each of seven scenes."""
  scene, witnesses = _static_case()
  q, robot = witnesses.q.clone(), witnesses.robot.clone()
  box_pos, target_pos = scene.box_pos.clone(), scene.target_pos.clone()
  robot[0, 1] = -1  # Missing.
  q[1, 0, 0] = 9.0  # Past a joint limit.
  q[2, 0, 0] = float('nan')
  robot[3, 0] = 4  # Names no robot.
  robot[4, 0] = -2
  box_pos[5, 0, 0] = float('nan')
  target_pos[6, 0, 0] = float('inf')
  bad_scene, bad = scene._replace(box_pos=box_pos, target_pos=target_pos), Witnesses(q, robot)
  rows = _rows(bad_scene, bad)
  kept, kept_witnesses, dropped = gen_scenes._keep_passing(bad_scene, bad, rows, 3)
  assert dropped == 7
  assert torch.equal(kept.box_pos, scene.box_pos[7:]) and torch.equal(kept_witnesses.q, witnesses.q[7:])
  assert torch.equal(kept_witnesses.robot, witnesses.robot[7:])
  with pytest.raises(RuntimeError, match='3 scenes passed'):
    gen_scenes._keep_passing(bad_scene, bad, rows, 4)


@pytest.mark.parametrize('field', ['witness_missing', 'witness_limits', 'witness_collision', 'witness_far'])
def test_each_failure_field_alone_drops_the_scene(field):
  scene, witnesses = _static_case()
  rows = _rows(scene, witnesses)
  rows[0] = {**rows[0], field: np.array([True] + [False] * 9)}
  _, kept, dropped = gen_scenes._keep_passing(scene, witnesses, rows, 9)
  assert dropped == 1 and torch.equal(kept.q, witnesses.q[1:])


def test_final_rejection_reads_the_tensors_it_keeps():
  """Clean rows do not excuse a nonfinite value or an index naming no robot."""
  scene, witnesses = _static_case()
  rows = _rows(scene, witnesses)
  q, robot = witnesses.q.clone(), witnesses.robot.clone()
  q[0, 0, 0] = float('nan')
  robot[1, 0] = 4
  robot[2, 0] = -1
  spans = scene.box_spans.clone()
  spans[3, 0, 0] = float('inf')
  _, kept, dropped = gen_scenes._keep_passing(scene._replace(box_spans=spans), Witnesses(q, robot), rows, 6)
  assert dropped == 4 and torch.equal(kept.q, witnesses.q[4:])


def test_only_static_paper_may_lack_a_witness():
  scene, witnesses = _static_case()
  robot = witnesses.robot.clone()
  robot[0, 0] = -1
  missing = Witnesses(witnesses.q, robot)
  assert gen_scenes._malformed(scene, missing)[0] and not gen_scenes._malformed(scene, missing)[1:].any()
  assert not gen_scenes._malformed(scene, missing, allow_missing=True).any()
  for value in (-2, 4):
    robot = witnesses.robot.clone()
    robot[0, 0] = value
    assert gen_scenes._malformed(scene, Witnesses(witnesses.q, robot), allow_missing=True)[0]


def test_static_paper_drops_a_witness_past_a_limit_and_refuses_a_home_collision(monkeypatch, tmp_path):
  """static_paper keeps its sets and records a failing witness as missing, whatever the failure."""
  monkeypatch.setattr(gen_scenes, 'CEILING_BUDGET', IkBudget(restarts=3))
  monkeypatch.setattr(gen_scenes, 'OUT_DIR', tmp_path)
  real_recheck = gen_scenes.recheck
  seen = {}

  def planted(pool, arm):
    rows = real_recheck(pool, arm)
    target = int(pool.witnesses.robot[0].argmax())
    assert pool.witnesses.robot[0, target] >= 0
    seen['target'] = target
    rows[0] = {**rows[0], 'witness_limits': rows[0]['witness_limits'].copy()}
    rows[0]['witness_limits'][target] = True
    return rows

  monkeypatch.setattr(gen_scenes, 'recheck', planted)
  args = _args(split='static_paper')
  pool = gen_scenes.generate_static_paper(args, DEVICE)
  assert pool.witnesses.robot[0, seen['target']] == -1
  assert pool.meta['reachable_per_set'][0] == int((pool.witnesses.robot[0] >= 0).sum())

  def home_collides(pool, arm):
    rows = real_recheck(pool, arm)
    rows[1] = {**rows[1], 'home_collision': True}
    return rows

  monkeypatch.setattr(gen_scenes, 'recheck', home_collides)
  with pytest.raises(RuntimeError, match='home'):
    gen_scenes.generate_static_paper(args, DEVICE)


@pytest.mark.parametrize('robots', [4, 8])
def test_frozen_matrices_need_the_transpose(robots):
  """Independent of gen_scenes: only the transposed matrices hold every collision vertex."""
  z = np.load(str(FROZEN_BOXES).format(robots))
  vertices = cube_collision_vertices()
  assert len(vertices) == 280
  assert vertices_outside_boxes(vertices, z['centroid'], z['rotation'].transpose(0, 2, 1), z['spans']) == 0
  assert vertices_outside_boxes(vertices, z['centroid'], z['rotation'], z['spans']) == 8


@pytest.mark.parametrize('robots', [4, 8])
def test_static_boxes_hold_the_cube_collision_vertices(robots):
  pos, rot, spans = gen_scenes.static_boxes(robots, 2, DEVICE)
  assert pos.shape == (2, 30, 3) and rot.shape == (2, 30, 3, 3) and spans.shape == (2, 30, 3)
  check_static_boxes(pos, rot, spans, robots)


def _generator_without_the_transpose():
  """gen_scenes with the correction removed, as a throwaway module."""
  path = Path(gen_scenes.__file__)
  source = path.read_text()
  line = "rot = as_f32(z['rotation']).transpose(-1, -2)"
  assert line in source
  module = types.ModuleType('gen_scenes_no_transpose')
  module.__file__ = str(path)
  exec(compile(source.replace(line, "rot = as_f32(z['rotation'])"), str(path), 'exec'), module.__dict__)
  return module


def test_containment_rejects_a_pool_built_without_the_transpose(monkeypatch, tmp_path):
  """The whole generator with the transpose removed. Its pool passes the recheck and fails here."""
  mutant = _generator_without_the_transpose()
  monkeypatch.setattr(mutant, 'OUT_DIR', tmp_path)
  path = mutant.main(['--robots', '4', '--targets', '10', '--obstacles', '30', '--split', 'static', '--seed', '1'])
  scene = load_pool(path, DEVICE).scene
  with pytest.raises(AssertionError, match='8 of 280 collision mesh vertices'):
    check_static_boxes(scene.box_pos, scene.box_rot, scene.box_spans, 4)


def test_containment_rejects_8r_boxes_without_the_transpose():
  pos, rot, spans = _generator_without_the_transpose().static_boxes(8, 2, DEVICE)
  with pytest.raises(AssertionError, match='8 of 280 collision mesh vertices'):
    check_static_boxes(pos, rot, spans, 8)
