# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Rigid helpers and obstacle augmentation, all batched."""

from __future__ import annotations

import torch


def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
  """Rotation matrices [..., 3, 3] to unit quaternions wxyz [..., 4]."""
  m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
  m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
  m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]
  # Each row is proportional to the quaternion. Use the row with the largest
  # squared component, which is the numerically safe one.
  rows = torch.stack([
    torch.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
    torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
    torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
    torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
  ], -2)
  best = rows.diagonal(dim1=-2, dim2=-1).argmax(-1)
  quat = rows.gather(-2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
  return quat / quat.norm(dim=-1, keepdim=True)


# planning_env.py:264-304. Code 0 keeps the box, codes 1, 2, 3 rotate it by +90
# degrees about local x, y, z and swap the two matching spans.
_AUG_ROT = (
  ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
  ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
  ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
  ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
)
_AUG_SPANS = ((0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2))


def augment_boxes(
  box_rot: torch.Tensor, box_spans: torch.Tensor, aug_code: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """The same physical box, described four ways. Only observations see this."""
  extra = torch.as_tensor(_AUG_ROT, dtype=box_rot.dtype, device=box_rot.device)[aug_code]
  order = torch.as_tensor(_AUG_SPANS, device=box_spans.device)[aug_code]
  return box_rot @ extra, box_spans.gather(-1, order)
