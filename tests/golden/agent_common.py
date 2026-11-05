"""Numpy-only helpers shared by the JAX golden generator and the torch tests.

Full-size network weights do not fit in git, so both sides derive them from each
leaf's flax name and compare a fixed sample of every leaf plus its full L2 norm.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

SAMPLES_PER_LEAF = 128
HEAD_LAST_LAYERS = ('policy_head/Dense_2/', 'qvalue/Dense_2/', 'rewards/Dense_2/')


def deterministic_leaf(key: str, shape: tuple[int, ...]) -> np.ndarray:
  """Same distribution as flax init, drawn from a stream numpy keeps frozen."""
  name = key.rsplit('/', 1)[-1]
  if name == 'scale':
    return np.ones(shape, dtype=np.float32)
  if name == 'bias':
    return np.zeros(shape, dtype=np.float32)
  scale = 0.005 if any(k in key for k in HEAD_LAST_LAYERS) else 1.0
  normal = np.random.RandomState(zlib.crc32(key.encode())).standard_normal(shape)
  return (np.clip(normal, -2.0, 2.0) * math.sqrt(scale / shape[0]) / 0.87962566103423978).astype(np.float32)


def sample_leaves(flat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """A fixed sample of every leaf in sorted key order, and each leaf's L2 norm."""
  samples, norms = [], []
  for key in sorted(flat):
    leaf = flat[key].reshape(-1)
    samples.append(leaf[:: max(1, leaf.size // SAMPLES_PER_LEAF)])
    norms.append(np.linalg.norm(leaf.astype(np.float64)))
  return np.concatenate(samples), np.array(norms)
