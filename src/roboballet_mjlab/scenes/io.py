# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Scene pools on disk, and importers for the original .pb and .pbtxt files.

DESIGN section 2 forbids a runtime import from environments/, and data_pb2.py is an untracked
generated file, so this code depends only on the schema in environments/data.proto:18-55. It
reads .pb records from the wire format, for speed, and parses .pbtxt with protobuf's own text
parser against that schema. tests/test_scenes_io.py compares both with the official decoder
in the old venv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, NamedTuple

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format
import numpy as np
import torch

from roboballet_mjlab.scenes.sampler import Witnesses
from roboballet_mjlab.structs import Scene


class Pool(NamedTuple):
  scene: Scene  # [N, ...]
  witnesses: Witnesses
  meta: dict  # Plain Python values: see scripts/gen_scenes.py for the keys.


def save_pool(path: Path, pool: Pool) -> None:
  """Atomic, so an interrupted run never leaves a truncated pool."""
  payload = {
    'scene': {name: field.cpu() for name, field in pool.scene._asdict().items()},
    'witness_q': pool.witnesses.q.cpu(),
    'witness_robot': pool.witnesses.robot.to(torch.int8).cpu(),
    'meta': pool.meta,
  }
  tmp = Path(f'{path}.tmp{os.getpid()}')
  torch.save(payload, tmp)
  os.replace(tmp, path)


def load_pool(path: Path, device) -> Pool:
  payload = torch.load(path, map_location=device, weights_only=True)
  return Pool(Scene(**payload['scene']),
              Witnesses(payload['witness_q'], payload['witness_robot'].long()),
              payload['meta'])


def git_revision(repo: Path) -> str:
  """HEAD from the files, without running git. 'unknown' if it cannot be read."""
  try:
    head = (repo / '.git/HEAD').read_text().strip()
    if not head.startswith('ref: '):
      return head
    ref = head[5:]
    loose = repo / '.git' / ref
    if loose.exists():
      return loose.read_text().strip()
    for line in (repo / '.git/packed-refs').read_text().splitlines():
      if line.endswith(' ' + ref):
        return line.split()[0]
  except OSError:
    pass
  return 'unknown'


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
  value, shift = 0, 0
  while True:
    byte = buf[pos]
    pos += 1
    value |= (byte & 0x7f) << shift
    shift += 7
    if not byte & 0x80:
      return value, pos


def _length_delimited(buf: bytes) -> Iterator[tuple[int, bytes]]:
  """(field number, payload) of a message whose fields are all length delimited."""
  pos, end = 0, len(buf)
  while pos < end:
    tag, pos = _varint(buf, pos)
    if tag & 7 != 2:
      raise ValueError(f'Unexpected wire type {tag & 7} in field {tag >> 3}')
    size, pos = _varint(buf, pos)
    yield tag >> 3, buf[pos:pos + size]
    pos += size


def _ndarray(buf: bytes) -> np.ndarray:
  """NDArray of data.proto:18-24, read as util.from_proto reads it.

  Fields go by number, in any order. A field seen twice accumulates, as protobuf does for
  packed segments and for an embedded message written twice. Unpacked values raise.
  """
  dims, dtype, values = b'', None, {3: b'', 4: b''}
  for field, payload in _length_delimited(buf):
    if field == 1:
      dims += payload
    elif field == 2:
      dtype = payload.decode()
    elif field in values:
      values[field] += payload
    else:
      raise ValueError(f'Unexpected NDArray field {field}')
  if not dtype:
    raise ValueError('NDArray without a dtype')
  shape, pos = [], 0
  while pos < len(dims):
    dim, pos = _varint(dims, pos)
    shape.append(dim)
  parsed = np.dtype(dtype)
  if parsed != np.float32 and parsed != np.float64:
    raise ValueError(f'Unsupported NDArray dtype {dtype}')
  return np.frombuffer(values[3 if parsed == np.float32 else 4], dtype=parsed).reshape(shape)


_BOX_FIELDS = {1: 'box_spans', 2: 'box_pos', 3: 'box_rot'}  # BoxInfo, data.proto:26-30.
_TARGET_FIELDS = {1: 'target_pos', 2: 'target_rot'}  # TargetInfo, data.proto:36-39.


def _arrays_by_field(buf: bytes, names: dict[int, str]) -> dict[str, np.ndarray]:
  """The NDArrays of a BoxInfo or TargetInfo. Fields go by number, and every one is required."""
  payloads: dict[int, bytes] = {}
  for field, payload in _length_delimited(buf):
    if field not in names:
      raise ValueError(f'Unexpected field {field}, expected {sorted(names)}')
    payloads[field] = payloads.get(field, b'') + payload
  missing = sorted(set(names) - set(payloads))
  if missing:
    raise ValueError(f'A nested message lacks fields {missing} of {sorted(names)}')
  return {names[field]: _ndarray(payload) for field, payload in payloads.items()}


def _start_state(buf: bytes) -> dict[str, np.ndarray]:
  """StartState of data.proto:42-52, as arrays in robot and box order."""
  boxes, targets, singular = [], [], {3: b'', 4: b''}
  for field, payload in _length_delimited(buf):
    if field == 1:
      boxes.append(_arrays_by_field(payload, _BOX_FIELDS))
    elif field == 2:
      targets.append(_arrays_by_field(payload, _TARGET_FIELDS))
    elif field in singular:
      singular[field] += payload
    else:
      raise ValueError(f'Unexpected StartState field {field}')
  if not (boxes and targets and singular[3] and singular[4]):
    raise ValueError('A StartState lacks boxes, targets, joint configurations or placements')
  base = _ndarray(singular[4])
  return {
    **{name: np.stack([box[name] for box in boxes]) for name in _BOX_FIELDS.values()},
    **{name: np.stack([target[name] for target in targets]) for name in _TARGET_FIELDS.values()},
    'q0': _ndarray(singular[3]),
    'base_pos': base[:, :3, 3],
    'base_rot': base[:, :3, :3],
  }


def iter_start_states(path: Path) -> Iterator[dict[str, np.ndarray]]:
  """Streams records of a StartStates .pb, each `0x0a, varint length, StartState`."""
  with open(path, 'rb', buffering=1 << 20) as f:
    while True:
      tag = f.read(1)
      if not tag:
        return
      if tag != b'\x0a':
        raise ValueError(f'Unexpected record tag {tag!r} at byte {f.tell() - 1}')
      size, shift = 0, 0
      while True:
        byte = f.read(1)[0]
        size |= (byte & 0x7f) << shift
        shift += 7
        if not byte & 0x80:
          break
      yield _start_state(f.read(size))


def load_start_states(path: Path, device, limit: int | None = None) -> Scene:
  """Stored start states as a float32 Scene, for gate G4 and for E10."""
  rows: dict[str, list[np.ndarray]] = {name: [] for name in Scene._fields}
  for i, record in enumerate(iter_start_states(path)):
    if limit is not None and i >= limit:
      break
    for name, value in record.items():
      rows[name].append(value.astype(np.float32))
  return Scene(**{name: torch.as_tensor(np.stack(values), device=device)
                  for name, values in rows.items()})


def _predefined_tasks_class():
  """PredefinedTasks of data.proto:18-45, for protobuf's text parser."""
  field_proto = descriptor_pb2.FieldDescriptorProto

  def field(name, number, kind, repeated=False, **message):
    return field_proto(name=name, number=number, type=kind, **message,
                       label=field_proto.LABEL_REPEATED if repeated else field_proto.LABEL_OPTIONAL)

  messages = {
    'NDArray': [field('dims', 1, field_proto.TYPE_INT32, repeated=True),
                field('dtype', 2, field_proto.TYPE_STRING),
                field('float_values', 3, field_proto.TYPE_FLOAT, repeated=True),
                field('double_values', 4, field_proto.TYPE_DOUBLE, repeated=True),
                field('int64_values', 5, field_proto.TYPE_INT64, repeated=True)],
    'TargetInfo': [field('position', 1, field_proto.TYPE_MESSAGE, type_name='.NDArray'),
                   field('rotation', 2, field_proto.TYPE_MESSAGE, type_name='.NDArray')],
    'TargetSet': [field('targets', 1, field_proto.TYPE_MESSAGE, repeated=True, type_name='.TargetInfo')],
    'PredefinedTasks': [field('world_path', 1, field_proto.TYPE_STRING),
                        field('target_sets', 2, field_proto.TYPE_MESSAGE, repeated=True, type_name='.TargetSet')],
  }
  schema = descriptor_pb2.FileDescriptorProto(
    name='roboballet_mjlab_tasks.proto', syntax='proto3',
    message_type=[descriptor_pb2.DescriptorProto(name=name, field=fields) for name, fields in messages.items()])
  pool = descriptor_pool.DescriptorPool()
  pool.AddSerializedFile(schema.SerializeToString())
  return message_factory.GetMessageClass(pool.FindMessageTypeByName('PredefinedTasks'))


def _taskset_array(proto, shape: tuple[int, ...], where: str) -> np.ndarray:
  """A taskset NDArray, as util.from_proto reads it. Raises unless it has this exact shape."""
  if tuple(proto.dims) != shape:
    raise ValueError(f'{where}: dims {list(proto.dims)}, expected {list(shape)}')
  if not proto.dtype or np.dtype(proto.dtype) not in (np.float32, np.float64):
    raise ValueError(f'{where}: unsupported dtype {proto.dtype!r}')
  dtype = np.dtype(proto.dtype)
  values = np.array(proto.float_values if dtype == np.float32 else proto.double_values, dtype=dtype)
  if values.size != np.prod(shape):
    raise ValueError(f'{where}: {values.size} values for shape {list(shape)}')
  if not np.isfinite(values).all():
    raise ValueError(f'{where}: nonfinite values')
  return values.reshape(shape)


def load_taskset(path: Path, device) -> tuple[torch.Tensor, torch.Tensor]:
  """PredefinedTasks text file: target pos [S, T, 3] and rot [S, T, 3, 3], float32.

  Raises text_format.ParseError on anything outside the schema, and ValueError unless there
  is at least one set, all sets hold the same positive number of targets, and every target
  has a position of shape [3] and a rotation of shape [3, 3].
  """
  tasks = _predefined_tasks_class()()
  text_format.Parse(Path(path).read_text(), tasks)
  if not tasks.target_sets:
    raise ValueError(f'{path}: no target sets')
  counts = sorted({len(target_set.targets) for target_set in tasks.target_sets})
  if len(counts) != 1 or counts[0] == 0:
    raise ValueError(f'{path}: target counts per set are {counts}, expected one positive count')
  pos, rot = [], []
  for s, target_set in enumerate(tasks.target_sets):
    for t, target in enumerate(target_set.targets):
      pos.append(_taskset_array(target.position, (3,), f'{path}: set {s}, target {t}, position'))
      rot.append(_taskset_array(target.rotation, (3, 3), f'{path}: set {s}, target {t}, rotation'))
  grid = len(tasks.target_sets), counts[0]
  return (torch.as_tensor(np.stack(pos).astype(np.float32).reshape(*grid, 3), device=device),
          torch.as_tensor(np.stack(rot).astype(np.float32).reshape(*grid, 3, 3), device=device))
