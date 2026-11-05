"""Writes scenes_wire_variants.binpb: stored records re-encoded in other legal wire forms.

Run from the project root, in the old venv, where the official decoder lives:
  cd .. && PYTHONPATH=. venv/bin/python roboballet-mjlab/tests/golden/scenes_make_wire_fixture.py

Records 0, 1 and 2 of 4r_10t_30o.pb, in that order:
  0: every message's fields in reverse order of field number.
  1: packed dims and values cut into several segments.
  2: BoxInfo.spans, TargetInfo.position and StartState.initial_joint_configurations each
     written twice with parts of the array, which protobuf merges.

The extension is not .pb because .gitignore excludes *.pb.
"""

import os
from pathlib import Path

from roboballet.environments import data_pb2

GOLDEN = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('ROBOBALLET_PROJECT', GOLDEN.parents[2]))  # As tests/test_sim_common.py.
SOURCE = PROJECT / 'roboballet/data/start_states/4r_10t_30o.pb'
OUT = GOLDEN / 'scenes_wire_variants.binpb'


def varint(value):
  out = bytearray()
  while value >= 128:
    out.append((value & 127) | 128)
    value >>= 7
  out.append(value)
  return bytes(out)


def read_varint(buf, pos):
  value, shift = 0, 0
  while True:
    byte = buf[pos]
    pos += 1
    value |= (byte & 127) << shift
    shift += 7
    if byte < 128:
      return value, pos


def fields(buf):
  out, pos = [], 0
  while pos < len(buf):
    tag, pos = read_varint(buf, pos)
    assert tag & 7 == 2
    size, pos = read_varint(buf, pos)
    out.append((tag >> 3, buf[pos:pos + size]))
    pos += size
  return out


def message(parts):
  return b''.join(varint(tag << 3 | 2) + varint(len(payload)) + payload for tag, payload in parts)


def reversed_fields(parts):
  """Highest field number first. Occurrences of a repeated field keep their order."""
  return sorted(parts, key=lambda part: -part[0])


def reorder_array(buf):
  return message(reversed_fields(fields(buf)))


def reorder_nested(buf):
  return message(reversed_fields([(tag, reorder_array(payload)) for tag, payload in fields(buf)]))


def reorder_state(buf):
  parts = [(tag, reorder_nested(payload) if tag in (1, 2) else reorder_array(payload))
           for tag, payload in fields(buf)]
  return message(reversed_fields(parts))


def split_array(buf):
  """Cuts dims after each element and values into segments of 1 and 2 elements."""
  parts = []
  for tag, payload in fields(buf):
    if tag == 1:
      pos = 0
      while pos < len(payload):
        _, end = read_varint(payload, pos)
        parts.append((tag, payload[pos:end]))
        pos = end
    elif tag in (3, 4):
      size = 4 if tag == 3 else 8
      for start in range(0, len(payload), 3 * size):
        parts.append((tag, payload[start:start + size]))
        parts.append((tag, payload[start + size:start + 3 * size]))
    else:
      parts.append((tag, payload))
  return message(parts)


def split_state(buf):
  parts = []
  for tag, payload in fields(buf):
    if tag in (1, 2):
      parts.append((tag, message([(t, split_array(p)) for t, p in fields(payload)])))
    else:
      parts.append((tag, split_array(payload)))
  return message(parts)


def in_two_parts(buf):
  """An NDArray as two messages: dims and dtype, then values."""
  parts = fields(buf)
  return message([p for p in parts if p[0] in (1, 2)]), message([p for p in parts if p[0] in (3, 4)])


def merged_state(buf):
  parts = []
  for tag, payload in fields(buf):
    if tag in (1, 2):
      nested = []
      for number, array in fields(payload):
        pieces = in_two_parts(array) if number == 1 else (array,)
        nested.extend((number, piece) for piece in pieces)
      parts.append((tag, message(nested)))
    elif tag == 3:
      parts.extend((tag, part) for part in in_two_parts(payload))
    else:
      parts.append((tag, payload))
  return message(parts)


def main():
  records, rest = [], SOURCE.open('rb')
  while len(records) < 3:
    assert rest.read(1) == b'\x0a'
    size, shift = 0, 0
    while True:
      byte = rest.read(1)[0]
      size |= (byte & 127) << shift
      shift += 7
      if byte < 128:
        break
    records.append(rest.read(size))
  variants = [reorder_state(records[0]), split_state(records[1]), merged_state(records[2])]
  for original, variant in zip(records, variants):
    assert variant != original
    assert data_pb2.StartState.FromString(variant) == data_pb2.StartState.FromString(original)
  OUT.write_bytes(b''.join(b'\x0a' + varint(len(v)) + v for v in variants))
  print(f'{OUT}: {OUT.stat().st_size} bytes, official decoder agrees on all {len(variants)}')


main()
