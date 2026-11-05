# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""mjlab Simulation whose only graph is kinematics + collision. mjlab_api.md 3.5."""

from __future__ import annotations

import gc

import mujoco_warp as mjw
import warp as wp
from mjlab.sim.sim import Simulation
from mujoco_warp._src.collision_driver import (
  MJ_COLLISION_TABLE, _narrowphase, create_collision_context,
)
from mujoco_warp._src.math import upper_trid_index
from mujoco_warp._src.types import CollisionType, GeomType


@wp.kernel
def _clamp_candidates(naconmax: int, ncollision: wp.array[int], seen: wp.array[int]):
  seen[0] = ncollision[0]
  ncollision[0] = wp.min(ncollision[0], naconmax)


@wp.kernel
def _restore_candidates(seen: wp.array[int], ncollision: wp.array[int]):
  ncollision[0] = seen[0]


def _has_convex_pairs(m: mjw.Model) -> bool:
  """Whether convex_narrowphase launches, by its own test (collision_convex.py:1179-1183)."""
  n = len(GeomType)
  return any(
    kind == CollisionType.CONVEX and m.geom_pair_type_count[upper_trid_index(n, a.value, b.value)]
    for (a, b), kind in MJ_COLLISION_TABLE.items())


def _clamped_collision(m: mjw.Model, d: mjw.Data) -> None:
  """mjw.collision with the candidate count clamped during the narrowphase only.

  The convex kernels walk ncollision pairs through arrays of naconmax, and broadphase keeps
  counting past the end (collision_driver.py:356, collision_convex.py:1040). So an overflow dies
  with CUDA 700 before World.forward can report it. The count is restored for that guard.
  """
  ctx = create_collision_context(d.naconmax)
  seen = wp.zeros(1, dtype=int)
  d.ncollision.zero_()
  d.nacon.zero_()
  mjw.sap_broadphase(m, d, ctx)
  wp.launch(_clamp_candidates, dim=1, inputs=[d.naconmax, d.ncollision, seen])
  _narrowphase(m, d, ctx)
  wp.launch(_restore_candidates, dim=1, inputs=[seen, d.ncollision])


class KinematicSim(Simulation):
  """The original env never integrates physics (env_spec 2), so neither does this."""

  def create_graph(self) -> None:
    self.step_graph = self.forward_graph = self.sense_graph = self.reset_graph = None
    self.kc_graph = None
    if not self.use_cuda_graph:
      return
    # Stale graph finalizers can otherwise be recorded into the new graph (sim.py:37-50).
    enabled = gc.isenabled()
    gc.disable()
    try:
      with wp.ScopedDevice(self.wp_device), wp.ScopedCapture() as capture:
        self._kinematics_collision()
      self.kc_graph = capture.graph
    finally:
      if enabled:
        gc.enable()

  def _kinematics_collision(self) -> None:
    m, d = self.wp_model, self.wp_data
    mjw.kinematics(m, d)
    # Primitive kernels launch naconmax threads and cannot walk off, so spheres stay stock.
    # While disableflags is set (World clears it), stock collision is the no-op wanted.
    if _has_convex_pairs(m) and not m.opt.disableflags:
      _clamped_collision(m, d)
    else:
      mjw.collision(m, d)

  def kinematics_collision(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.kc_graph is not None:
        wp.capture_launch(self.kc_graph)
      else:
        self._kinematics_collision()
