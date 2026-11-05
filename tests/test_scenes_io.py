"""scenes/io.py: pool round trip, and the importers against the official decoder."""

from __future__ import annotations

import subprocess
import textwrap
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch

from google.protobuf import text_format

from roboballet_mjlab.config import WorldCfg
from roboballet_mjlab.scenes import io
from roboballet_mjlab.scenes.sampler import Witnesses, any_robot_collides
from roboballet_mjlab.sim.world import build_world, default_bases
from roboballet_mjlab.structs import Scene
from tests.test_scenes_common import REPO, START_STATES
from tests.test_sim_common import DATA, DEVICE, PROJECT

OLD_PYTHON = PROJECT / 'venv/bin/python'
TASKSETS = DATA / 'paper_test_tasksets'
WIRE_VARIANTS = Path(__file__).resolve().parent / 'golden/scenes_wire_variants.binpb'
RECORDS = 40


def _official_dump(tmp_path: Path, script: str, records: int = RECORDS) -> dict:
  """Decodes with data_pb2 in the old venv, which is what produced the reference numbers."""
  out = tmp_path / 'official.npz'
  code = textwrap.dedent(script).replace('OUT', repr(str(out))).replace('RECORDS', str(records))
  subprocess.run([str(OLD_PYTHON), '-c', code], check=True, cwd=PROJECT,
                 env={'PYTHONPATH': str(PROJECT), 'PATH': '/usr/bin:/bin'})
  return dict(np.load(out))


def _official_start_states(tmp_path: Path, path: Path, records: int = RECORDS) -> dict:
  return _official_dump(tmp_path, f'''
    import numpy as np
    from roboballet.environments import data_pb2, util
    rows = []
    with open({str(path)!r}, 'rb') as f:
      for _ in range(RECORDS):
        assert f.read(1) == b'\\x0a'
        size, shift = 0, 0
        while True:
          byte = f.read(1)[0]
          size |= (byte & 0x7f) << shift
          shift += 7
          if not byte & 0x80:
            break
        rows.append(data_pb2.StartState.FromString(f.read(size)))
    get = util.from_proto
    np.savez(OUT,
      box_spans=np.array([[get(b.spans) for b in r.generated_obstacles] for r in rows]),
      box_pos=np.array([[get(b.centroid) for b in r.generated_obstacles] for r in rows]),
      box_rot=np.array([[get(b.rotation) for b in r.generated_obstacles] for r in rows]),
      target_pos=np.array([[get(t.position) for t in r.targets] for r in rows]),
      target_rot=np.array([[get(t.rotation) for t in r.targets] for r in rows]),
      q0=np.array([get(r.initial_joint_configurations) for r in rows]),
      base=np.array([get(r.initial_robot_placements) for r in rows]))
  ''', records)


def _assert_scene_is_official(scene: Scene, official: dict) -> None:
  for field in ('box_spans', 'box_pos', 'box_rot', 'target_pos', 'target_rot', 'q0'):
    assert np.array_equal(getattr(scene, field).numpy(), official[field].astype(np.float32)), field
  assert np.array_equal(scene.base_pos.numpy(), official['base'][:, :, :3, 3])
  assert np.array_equal(scene.base_rot.numpy(), official['base'][:, :, :3, :3])


@pytest.mark.parametrize('name', ['4r_10t_30o', '8r_40t_30o'])
def test_start_states_match_the_official_decoder(tmp_path, name):
  official = _official_start_states(tmp_path, START_STATES / f'{name}.pb')
  _assert_scene_is_official(io.load_start_states(START_STATES / f'{name}.pb', 'cpu', limit=RECORDS), official)


def test_other_legal_wire_forms_match_the_official_decoder(tmp_path):
  """Reordered fields, split packed segments and merged messages. See golden/scenes_make_wire_fixture.py."""
  official = _official_start_states(tmp_path, WIRE_VARIANTS, records=3)
  scene = io.load_start_states(WIRE_VARIANTS, 'cpu')
  assert len(scene.q0) == 3
  _assert_scene_is_official(scene, official)
  canonical = io.load_start_states(START_STATES / '4r_10t_30o.pb', 'cpu', limit=3)
  for mine, theirs in zip(scene, canonical):
    assert torch.equal(mine, theirs)
  assert (scene.box_spans > 0).all()


def test_fixture_really_holds_the_other_wire_forms():
  with open(WIRE_VARIANTS, 'rb') as f:
    records = [_read_record(f) for _ in range(3)]
  reordered, split, merged = records
  first_box = next(payload for field, payload in io._length_delimited(reordered) if field == 1)
  assert [field for field, _ in io._length_delimited(first_box)] == [3, 2, 1]
  q0 = next(payload for field, payload in io._length_delimited(split) if field == 3)
  assert sum(field == 1 for field, _ in io._length_delimited(q0)) == 2, 'dims in two segments'
  assert sum(field == 3 for field, _ in io._length_delimited(q0)) > 2, 'values in several segments'
  assert sum(field == 3 for field, _ in io._length_delimited(merged)) == 2, 'q0 written twice'


def _read_record(f) -> bytes:
  assert f.read(1) == b'\x0a'
  size, shift = 0, 0
  while True:
    byte = f.read(1)[0]
    size |= (byte & 0x7f) << shift
    shift += 7
    if not byte & 0x80:
      return f.read(size)


@pytest.mark.parametrize('message,error', [
  (b'\x08\x07', 'Unexpected wire type'),  # Unpacked dims: legal, not read, so it must not be misread.
  (b'\x0a\x01\x07', 'dtype'),  # No dtype.
  (b'\x0a\x01\x01\x12\x03<i4\x1a\x04\x00\x00\x00\x00', 'Unsupported'),
  (b'\x0a\x01\x02\x12\x03<f4\x1a\x04\x00\x00\x00\x00', 'reshape'),  # One value, two dims.
  (b'\x0a\x01\x01\x12\x03<f4\x22\x08\x00\x00\x00\x00\x00\x00\x00\x00', 'reshape'),  # Doubles under <f4.
])
def test_bad_ndarray_is_an_error(message, error):
  with pytest.raises(ValueError, match=error):
    io._ndarray(message)


def _box(fields) -> bytes:
  array = b'\x0a\x01\x01\x12\x03<f4\x1a\x04\x00\x00\x00\x00'
  return b''.join(bytes([number << 3 | 2, len(array)]) + array for number in fields)


@pytest.mark.parametrize('fields,error', [((1, 2), 'lacks'), ((1, 2, 3, 4), 'Unexpected')])
def test_nested_message_with_other_fields_is_an_error(fields, error):
  outer = _box(fields)
  with pytest.raises(ValueError, match=error):
    io._start_state(bytes([0x0a, len(outer)]) + outer)


def test_start_state_without_q0_is_an_error():
  with pytest.raises(ValueError, match='lacks'):
    io._start_state(b'')


@pytest.mark.parametrize('num_targets', [10, 20, 40])
def test_tasksets_match_the_official_parser(tmp_path, num_targets):
  path = TASKSETS / f'test_{num_targets}targets.pbtxt'
  official = _official_dump(tmp_path, f'''
    import numpy as np
    from google.protobuf import text_format
    from roboballet.environments import data_pb2, util
    tasks = data_pb2.PredefinedTasks()
    text_format.Parse(open({str(path)!r}).read(), tasks)
    np.savez(OUT,
      pos=np.array([[util.from_proto(t.position) for t in s.targets] for s in tasks.target_sets]),
      rot=np.array([[util.from_proto(t.rotation) for t in s.targets] for s in tasks.target_sets]))
  ''')
  pos, rot = io.load_taskset(path, 'cpu')
  assert pos.shape == (10, num_targets, 3) and rot.shape == (10, num_targets, 3, 3)
  assert np.array_equal(pos.numpy(), official['pos'])
  assert np.array_equal(rot.numpy(), official['rot'])


@pytest.mark.parametrize('form', ['as_one_line=True', 'use_short_repeated_primitives=True', 'pointy_brackets=True'])
def test_other_text_forms_match_the_official_parser(tmp_path, form):
  """The official printer writes the released taskset in another form, and its parser agrees."""
  written = tmp_path / 'tasks.pbtxt'
  official = _official_dump(tmp_path, f'''
    import numpy as np
    from google.protobuf import text_format
    from roboballet.environments import data_pb2, util
    tasks = data_pb2.PredefinedTasks()
    text_format.Parse(open({str(TASKSETS / 'test_10targets.pbtxt')!r}).read(), tasks)
    text = text_format.MessageToString(tasks, {form})
    open({str(written)!r}, 'w').write(text)
    again = data_pb2.PredefinedTasks()
    text_format.Parse(text, again)
    assert again == tasks
    np.savez(OUT,
      pos=np.array([[util.from_proto(t.position) for t in s.targets] for s in tasks.target_sets]),
      rot=np.array([[util.from_proto(t.rotation) for t in s.targets] for s in tasks.target_sets]))
  ''')
  pos, rot = io.load_taskset(written, 'cpu')
  assert pos.shape == (10, 10, 3) and rot.shape == (10, 10, 3, 3)
  assert np.array_equal(pos.numpy(), official['pos'])
  assert np.array_equal(rot.numpy(), official['rot'])


_POSITION = 'position { dims: 3 dtype: "<f4" float_values: 0.5 float_values: 0 float_values: 1 }'
_IDENTITY = ' '.join(f'float_values: {v}' for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
_ROTATION = f'rotation {{ dims: 3 dims: 3 dtype: "<f4" {_IDENTITY} }}'
_TARGET = f'targets {{ {_POSITION} {_ROTATION} }}'


def _taskset(tmp_path, text):
  path = tmp_path / 'tasks.pbtxt'
  path.write_text(text)
  return io.load_taskset(path, 'cpu')


def test_small_taskset_loads(tmp_path):
  pos, rot = _taskset(tmp_path, f'# A comment.\nworld_path: "w.xml" target_sets {{ {_TARGET} {_TARGET} }}\ntarget_sets {{ {_TARGET} {_TARGET} }}')
  assert pos.shape == (2, 2, 3) and rot.shape == (2, 2, 3, 3) and pos.dtype == torch.float32
  assert torch.equal(rot[1, 1], torch.eye(3))


@pytest.mark.parametrize('text,error', [
  ('', ValueError),
  ('# Only a comment.', ValueError),
  ('world_path: "w.xml"', ValueError),
  ('target_sets { }', ValueError),
  (f'target_sets {{ {_TARGET} }} target_sets {{ {_TARGET} {_TARGET} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_ROTATION} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION.replace("dims: 3", "dims: 4")} {_ROTATION} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION} {_ROTATION.replace("dims: 3 dims: 3", "dims: 9")} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION} {_ROTATION.replace("float_values: 1 ", "", 1)} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION.replace("0.5", "nan")} {_ROTATION} }} }}', ValueError),
  (f'target_sets {{ targets {{ {_POSITION.replace("<f4", "<i4")} {_ROTATION} }} }}', ValueError),
  (f'target_sets {{ goals {{ {_POSITION} {_ROTATION} }} }}', text_format.ParseError),
  (f'target_sets {{ {_TARGET}', text_format.ParseError),
  (f'{_TARGET}', text_format.ParseError),
  ('not protobuf at all', text_format.ParseError),
])
def test_bad_taskset_is_an_error(tmp_path, text, error):
  with pytest.raises(error):
    _taskset(tmp_path, text)


def test_streaming_does_not_load_the_file():
  """The 8r_40t file is 524.5 MB. Forty records must stay far below that."""
  stream = io.iter_start_states(START_STATES / '8r_40t_30o.pb')
  tracemalloc.start()
  first = [next(stream) for _ in range(RECORDS)]
  _, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  stream.close()
  assert first[0]['base_pos'].shape == (8, 3)
  assert first[0]['target_pos'].shape == (40, 3)
  assert peak < 20e6


@pytest.mark.parametrize('name,num_robots', [('4r_10t_30o', 4), ('8r_10t_30o', 8)])
def test_robot_order_matches_the_world(name, num_robots):
  """The arrays follow WorldCfg.robot_names: only y may differ from the default base."""
  scene = io.load_start_states(START_STATES / f'{name}.pb', 'cpu', limit=RECORDS)
  pos, rot = default_bases(WorldCfg(num_robots=num_robots), 'cpu')
  assert (scene.base_pos[..., [0, 2]] - pos[:, [0, 2]]).abs().max() < 1e-5
  assert (scene.base_rot - rot).abs().max() < 1e-4


def test_imported_scenes_are_free_at_home_on_the_mesh_world():
  """What G4 relies on: the stored scenes, imported, sit free in World with arm_model mesh_tool."""
  scene = io.load_start_states(START_STATES / '4r_10t_30o.pb', DEVICE, limit=256)
  world = build_world(WorldCfg(arm_model='mesh_tool'), 256, DEVICE)
  assert not any_robot_collides(world, scene, torch.arange(256, device=DEVICE)).any()


def test_bad_record_is_an_error(tmp_path):
  path = tmp_path / 'bad.pb'
  path.write_bytes(b'\x12\x01\x00')
  with pytest.raises(ValueError, match='Unexpected record tag'):
    next(io.iter_start_states(path))


def _random_pool(n=6, seed=0) -> io.Pool:
  gen = torch.Generator().manual_seed(seed)
  r = lambda *shape: torch.rand(*shape, generator=gen)
  scene = Scene(r(n, 4, 3), r(n, 4, 3, 3), r(n, 30, 3), r(n, 30, 3, 3), r(n, 30, 3),
                r(n, 10, 3), r(n, 10, 3, 3), r(n, 4, 7))
  robot = torch.randint(-1, 4, (n, 10), generator=gen)
  return io.Pool(scene, Witnesses(r(n, 10, 7), robot),
                 {'seed': 3, 'budget': {'restarts': 3, 'ang_tol': 0.26}, 'rates': {'box': 0.9}, 'path': 'a'})


def test_pool_round_trip(tmp_path):
  pool = _random_pool()
  path = tmp_path / 'pool.pt'
  io.save_pool(path, pool)
  loaded = io.load_pool(path, DEVICE)
  for a, b in zip(pool.scene, loaded.scene):
    assert torch.equal(a, b.cpu())
    assert b.device.type == 'cuda'
  assert torch.equal(pool.witnesses.q, loaded.witnesses.q.cpu())
  assert torch.equal(pool.witnesses.robot, loaded.witnesses.robot.cpu())
  assert loaded.witnesses.robot.dtype == torch.long
  assert loaded.meta == pool.meta
  assert [p.name for p in tmp_path.iterdir()] == ['pool.pt'], 'No temporary file may remain'


def test_git_revision_reads_head():
  revision = io.git_revision(REPO)
  assert len(revision) == 40 and int(revision, 16) >= 0
  assert io.git_revision(REPO / 'does_not_exist') == 'unknown'
