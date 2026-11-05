"""Shared by the scene tests and scripts/gen_scenes.py: marginals, KS, C MuJoCo recheck."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from roboballet_mjlab.config import JOINT_LOWER, JOINT_UPPER
from roboballet_mjlab.structs import Scene
from roboballet_mjlab.paths import PACKAGE_DATA
from tests.test_sim_common import DATA, CWorld

REPO = Path(__file__).resolve().parents[1]
# Scene pools are found by roboballet_mjlab.paths.scene_pool: the small ones are in the package,
# the large ones in the data directory.
START_STATES = DATA / 'start_states'
CUBE_XML = DATA / 'mujoco_world/assets/table_cube/table_cube.xml'
FROZEN_BOXES = PACKAGE_DATA / 'static_boxes/static_boxes_{}r.npz'  # As scripts/gen_scenes.py.
STATIC_LIFT_8R = 0.6  # env_spec D3.
CONTAINMENT_TOL = 1e-6
OFFSET = 0.03
OFFSET_TOL = 1e-4  # E10.
POS_TOL = 0.025
ANG_TOL = math.radians(15.0)
# f32 joint values, compared as f32, as BalletEnv clips them.
LOWER = np.asarray(JOINT_LOWER, dtype=np.float32)
UPPER = np.asarray(JOINT_UPPER, dtype=np.float32)


def ks(a: np.ndarray, b: np.ndarray) -> float:
  """Two-sample Kolmogorov-Smirnov statistic."""
  a, b = np.sort(a), np.sort(b)
  x = np.concatenate([a, b])
  return float(np.abs(np.searchsorted(a, x, side='right') / len(a)
                      - np.searchsorted(b, x, side='right') / len(b)).max())


def face_stats(scene: Scene) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """For each target, finds the box whose face it sits 3 cm outside.

  Returns has_owner [K] bool, offset residual [K] in metres and the angle between target z
  and the inward normal [K] in degrees. S/scripts_baseline/decode_data.py at the E10 tolerance.
  """
  delta = scene.target_pos[:, :, None] - scene.box_pos[:, None]  # [N, T, O, 3]
  local = (delta[..., None, :] @ scene.box_rot[:, None])[..., 0, :]
  excess = local.abs() - scene.box_spans[:, None] / 2
  on_offset = (excess - OFFSET).abs() < OFFSET_TOL
  owned = (on_offset.sum(-1) == 1) & ((excess <= OFFSET_TOL).sum(-1) == 2)
  has_owner = owned.any(-1)
  n, t = has_owner.nonzero(as_tuple=True)
  # Another box can match by accident, so take the closest match.
  nearest = (excess - OFFSET).abs().amin(-1).masked_fill(~owned, math.inf)
  o = nearest[n, t].argmin(-1)
  axis = on_offset[n, t, o].float().argmax(-1)
  rows = torch.arange(len(n), device=n.device)
  residual = (excess[n, t, o, axis] - OFFSET).abs()
  sign = local[n, t, o, axis].sign()
  inward = -sign[:, None] * scene.box_rot[n, o][rows, :, axis]
  cos = (scene.target_rot[n, t][:, :, 2] * inward).sum(-1).clamp(-1, 1)
  return has_owner.flatten(), residual, torch.rad2deg(cos.acos())


def e10_marginals(scene: Scene) -> dict[str, np.ndarray]:
  """The marginals of E10, paper_baseline.md 10.1, by name."""
  out = {}
  numpy = lambda x: x.detach().cpu().numpy().ravel()
  for i, axis in enumerate('xyz'):
    out[f'spans_{axis}'] = numpy(scene.box_spans[..., i])
    out[f'centroid_{axis}'] = numpy(scene.box_pos[..., i])
    out[f'target_{axis}'] = numpy(scene.target_pos[..., i])
  out['target_radius'] = numpy(scene.target_pos[..., :2].norm(dim=-1))
  y = scene.base_pos[..., 1]
  for r in range(y.shape[1]):
    out[f'base_y_{r}'] = numpy(y[:, r])
  # Same-rail differences: the rail is the base x and the height, for every robot pair.
  rail = lambda r: (round(float(scene.base_pos[0, r, 0]), 3), round(float(scene.base_pos[0, r, 2]), 3))
  for a in range(y.shape[1]):
    for b in range(a + 1, y.shape[1]):
      if rail(a) == rail(b):
        out[f'dy_{a}_{b}'] = numpy(y[:, b] - y[:, a])
  out['inward_angle_deg'] = numpy(face_stats(scene)[2])
  return out


def cube_collision_vertices() -> np.ndarray:
  """World positions [V, 3] of the compiled collision mesh vertices of the static cube."""
  model = mujoco.MjSpec.from_file(str(CUBE_XML)).compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  parts = []
  for geom in range(model.ngeom):
    if model.geom_contype[geom] or model.geom_conaffinity[geom]:
      mesh = model.geom_dataid[geom]
      start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
      local = model.mesh_vert[start:start + count].astype(np.float64)
      parts.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
  return np.concatenate(parts)


def vertices_outside_boxes(vertices, box_pos, box_rot, box_spans) -> int:
  """Vertices [V, 3] further than CONTAINMENT_TOL outside every box. Box axes are rot's columns."""
  delta = vertices[:, None] - np.asarray(box_pos, dtype=np.float64)[None]
  local = np.einsum('vbi,bij->vbj', delta, np.asarray(box_rot, dtype=np.float64))
  excess = (np.abs(local) - np.asarray(box_spans, dtype=np.float64)[None] / 2).max(-1).min(-1)
  return int((excess > CONTAINMENT_TOL).sum())


def check_static_boxes(box_pos, box_rot, box_spans, robots: int) -> None:
  """Asserts that boxes [N, O, ...] hold the cube and are the frozen boxes, in every scene.

  The frozen file is read here, not through gen_scenes, so that a generator and a pool sharing
  a wrong matrix still fail. Containment comes first, as it needs no idea of the convention.
  """
  vertices = cube_collision_vertices()
  lift = np.array([0, 0, STATIC_LIFT_8R if robots == 8 else 0.0], dtype=np.float32)
  arrays = [x.detach().cpu().numpy() for x in (box_pos, box_rot, box_spans)]
  outside = vertices_outside_boxes(vertices + lift.astype(np.float64), *[a[0] for a in arrays])
  assert outside == 0, f'{outside} of {len(vertices)} collision mesh vertices lie outside the boxes'
  frozen = np.load(str(FROZEN_BOXES).format(robots))
  expected = (frozen['centroid'].astype(np.float32) + lift,
              frozen['rotation'].astype(np.float32).transpose(0, 2, 1),
              frozen['spans'].astype(np.float32))
  for name, array, want in zip(('box_pos', 'box_rot', 'box_spans'), arrays, expected):
    assert (array == want).all(), f'{name} differs from the frozen boxes'


def angular_distance(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
  """target.py:107-112 for 3x3 matrices, in f64."""
  relative = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, relative.ravel())
  return 2 * math.atan2(np.linalg.norm(quat[1:]), abs(quat[0]))


def recheck_scene(world: CWorld, scene: dict[str, np.ndarray], witness_q: np.ndarray,
                  witness_robot: np.ndarray) -> dict:
  """C MuJoCo check of one scene: home free, each stored witness in tolerance and free.

  Comparisons are strict, like in_solution_threshold, target.py:152-164. Uses neither
  sim/world.py nor scenes/ik.py. A target without a witness, robot -1, is reported missing.

  Values the C model cannot evaluate fail without reaching it: a nonfinite scene counts as a
  home collision with every witness far, and a nonfinite q or an index naming no robot as far.
  """
  num_targets = len(witness_robot)
  out = {'home_collision': False,
         'witness_missing': witness_robot < 0,
         'witness_collision': np.zeros(num_targets, dtype=bool),
         'witness_far': np.zeros(num_targets, dtype=bool),
         'witness_limits': np.zeros(num_targets, dtype=bool),
         'worst_pos': 0.0, 'worst_ang': 0.0}
  if not all(np.isfinite(value).all() for value in scene.values()):
    out['home_collision'] = True
    out['witness_far'][:] = True
    return out
  world.set_scene(scene['base_pos'], scene['base_rot'], scene['box_pos'], scene['box_rot'],
                  scene['box_spans'])
  world.set_q(scene['q0'])
  world.forward()
  out['home_collision'] = bool(world.flags().any())
  for t, robot in enumerate(witness_robot):
    if robot < 0:
      continue
    if robot >= len(scene['q0']) or not np.isfinite(witness_q[t]).all():
      out['witness_far'][t] = True
      continue
    q = scene['q0'].copy()
    q[robot] = witness_q[t]
    world.set_q(q)
    world.forward()
    out['witness_collision'][t] = world.flags().any()
    out['witness_limits'][t] = np.any(q[robot] < LOWER) or np.any(q[robot] > UPPER)
    pos, rot = world.tcp()
    pos_err = float(np.linalg.norm(pos[robot] - scene['target_pos'][t]))
    ang_err = angular_distance(rot[robot], scene['target_rot'][t])
    out['witness_far'][t] = not (pos_err < POS_TOL and ang_err < ANG_TOL)
    out['worst_pos'] = max(out['worst_pos'], pos_err)
    out['worst_ang'] = max(out['worst_ang'], ang_err)
  return out


def recheck_chunk(args) -> list[dict]:
  """Runs in a worker process. args: scene arrays [N, ...], witness q and robot, arm model."""
  scene, witness_q, witness_robot, arm = args
  world = CWorld(scene['base_pos'].shape[1], scene['box_pos'].shape[1], arm)
  rows = range(len(witness_robot))
  return [recheck_scene(world, {k: v[i] for k, v in scene.items()}, witness_q[i], witness_robot[i])
          for i in rows]


def summarise_recheck(results: list[dict]) -> dict:
  """Plain values for a pool's metadata."""
  return {
    'scenes': len(results),
    'home_collisions': sum(r['home_collision'] for r in results),
    'witnesses': sum(int((~r['witness_missing']).sum()) for r in results),
    'witness_missing': sum(int(r['witness_missing'].sum()) for r in results),
    'witness_collisions': sum(int(r['witness_collision'].sum()) for r in results),
    'witness_far': sum(int(r['witness_far'].sum()) for r in results),
    'witness_limits': sum(int(r['witness_limits'].sum()) for r in results),
    'worst_pos_m': max(r['worst_pos'] for r in results),
    'worst_ang_deg': math.degrees(max(r['worst_ang'] for r in results)),
  }
