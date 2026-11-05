"""E11 of paper_baseline.md 10.1: sphere against mesh collision labels, per robot.

Run-B policy rollouts do not exist, so the half of E11 that needs them is NOT RUN: 50,000
random states stand for the random half, and the policy false-positive limit is not assessed.
A reach proxy is printed for orientation only and passes nothing.

The primary reference is mesh+stick, the original arm meshes with the same stick spheres, as
in sphere_model.md 5.1: it isolates the sphere fit. Against mesh+tool the original tool's
mass is simply absent from the new arm, so that recall is reported and not gated.
"""

from __future__ import annotations

import multiprocessing
import os

import mujoco
import numpy as np
import pytest

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, WorldCfg
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.scenes.io import load_pool
from roboballet_mjlab.sim.world import default_bases
from tests.test_sim_common import WORLD_XML, CWorld, robot_names

RANDOM_STATES = 50_000
PROXY_STATES = 50_000
SCENES_USED = 250
PROCESSES = 16
STICK_Z = (0.0134, 0.0384, 0.0634, 0.0884, 0.1134, 0.1384)  # panda_spheres_stick.xml
STICK_RADIUS = 0.015
DEEP = -0.005  # Mesh penetration beyond 5 mm.


class MeshStickWorld(CWorld):
  """The original world with the tool's bodies replaced by the stick's spheres."""

  def __init__(self, num_robots: int, num_boxes: int) -> None:
    self.names = robot_names(num_robots)
    self.arm = 'mesh_stick'
    spec = mujoco.MjSpec.from_file(str(WORLD_XML[8 if num_robots > 4 else 4]))
    spec.delete(spec.body('workpiece'))
    for name in self.names:
      for part in ('tool_arm', 'tool_base', 'tool_tip'):
        spec.delete(spec.body(f'{name}/tool/{part}'))
      tool = spec.body(f'{name}/tool/tool')
      for z in STICK_Z:
        tool.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[STICK_RADIUS, 0, 0], pos=[0, 0, z],
                      contype=1, conaffinity=1)
    for i in range(num_boxes):
      body = spec.worldbody.add_body(name=f'box{i}', mocap=True, pos=[0, 0, 50 + i])
      body.add_geom(name=f'box{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05])
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    m = self.model
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    joint = lambda n, j: m.jnt_qposadr[m.joint(f'{n}/joint{j}').id]
    self.q_adr = np.array([[joint(n, j) for j in range(1, 8)] for n in self.names])
    self.base_mocap = np.array([m.body_mocapid[m.body(n).id] for n in self.names])
    self.box_mocap = np.array([m.body_mocapid[m.body(f'box{i}').id] for i in range(num_boxes)])
    self.box_geom = np.array([m.body_geomadr[m.body(f'box{i}').id] for i in range(num_boxes)])
    self.geom_robot = np.full(m.ngeom, -1)
    for g in range(m.ngeom):
      root = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_rootid[m.geom_bodyid[g]])
      if root in self.names:
        self.geom_robot[g] = self.names.index(root)


def make_world(kind: str, num_robots: int, num_boxes: int) -> CWorld:
  if kind == 'mesh_stick':
    return MeshStickWorld(num_robots, num_boxes)
  return CWorld(num_robots, num_boxes, kind)


def deepest_per_robot(world: CWorld) -> np.ndarray:
  """Most negative contact distance per robot, [R]. Positive infinity when none."""
  deepest = np.full(len(world.names), np.inf)
  contact = world.data.contact
  for i in range(world.data.ncon):
    if contact.dist[i] <= 0:
      for g in contact.geom[i]:
        r = world.geom_robot[g]
        if r >= 0:
          deepest[r] = min(deepest[r], contact.dist[i])
  return deepest


def label_chunk(args) -> dict[str, np.ndarray]:
  """One scene and its joint states, labelled by each model. Runs in a worker process."""
  scene, q = args
  out = {}
  for kind in ('spheres_stick', 'mesh_stick', 'mesh_tool'):
    world = make_world(kind, scene['base_pos'].shape[0], scene['box_pos'].shape[0])
    world.set_scene(scene['base_pos'], scene['base_rot'], scene['box_pos'], scene['box_rot'],
                    scene['box_spans'])
    rows = np.zeros((len(q), scene['base_pos'].shape[0]))
    for i, state in enumerate(q):
      world.set_q(state)
      world.forward()
      rows[i] = deepest_per_robot(world)
    out[kind] = rows
  return out


def _random_states(rng, n, num_robots):
  return rng.uniform(JOINT_LOWER, JOINT_UPPER, (n, num_robots, 7)).astype(np.float32)


def _reach_states(rng, n, witness_q, witness_robot, num_robots):
  """home + alpha (witness - home) per arm, the reach proxy of sphere_model.md 5.1."""
  home = np.asarray(HOME_QPOS, dtype=np.float32)
  q = np.tile(home, (n, num_robots, 1))
  for r in range(num_robots):
    solutions = witness_q[witness_robot == r]
    if len(solutions):
      pick = solutions[rng.integers(0, len(solutions), n)]
      q[:, r] = home + rng.uniform(0, 1, (n, 1)) * (pick - home)
  return q.astype(np.float32)


def _label(pool, make_states, total: int) -> dict[str, np.ndarray]:
  per_scene = total // SCENES_USED
  jobs = []
  for s in range(SCENES_USED):
    scene = {k: v[s].cpu().numpy() for k, v in pool.scene._asdict().items()}
    jobs.append((scene, make_states(np.random.default_rng(s), s, per_scene)))
  saved = os.environ.get('CUDA_VISIBLE_DEVICES', '')
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  try:
    with multiprocessing.get_context('spawn').Pool(PROCESSES) as workers:
      results = workers.map(label_chunk, jobs, chunksize=1)
  finally:
    os.environ['CUDA_VISIBLE_DEVICES'] = saved
  return {k: np.concatenate([r[k] for r in results]) for k in results[0]}


def _rates(sphere: np.ndarray, mesh: np.ndarray) -> dict[str, float]:
  sphere_hit, mesh_hit = np.isfinite(sphere), np.isfinite(mesh)
  deep = mesh < DEEP
  return {
    'recall': float(sphere_hit[mesh_hit].mean()),
    'recall_deep': float(sphere_hit[deep].mean()),
    'false_positive_robot': float(sphere_hit[~mesh_hit].mean()),
    'mesh_rate': float(mesh_hit.mean()),
    'deep_rate': float(deep.mean()),
    'robot_labels': int(mesh_hit.size),
  }


@pytest.fixture(scope='module')
def pool():
  path = scene_pool('4r_10t_30o_heldout')
  if not path.exists():
    pytest.fail(f'Missing {path}')
  return load_pool(path, 'cpu')


@pytest.fixture(scope='module')
def random_labels(pool):
  return _label(pool, lambda rng, s, n: _random_states(rng, n, 4), RANDOM_STATES)


def test_recall_on_random_states_against_mesh_stick(random_labels):
  rates = _rates(random_labels['spheres_stick'], random_labels['mesh_stick'])
  print(f'\nE11 random, spheres vs mesh+stick: {rates}')
  assert rates['recall'] >= 0.98
  assert rates['recall_deep'] >= 0.995


def test_report_against_mesh_tool(random_labels):
  rates = _rates(random_labels['spheres_stick'], random_labels['mesh_tool'])
  print(f'\nE11 random, spheres vs mesh+tool (reported, not gated): {rates}')


def test_reach_proxy_is_reported(pool):
  q, robot = pool.witnesses.q.numpy(), pool.witnesses.robot.numpy()
  labels = _label(pool, lambda rng, s, n: _reach_states(rng, n, q[s], robot[s], 4), PROXY_STATES)
  rates = _rates(labels['spheres_stick'], labels['mesh_stick'])
  print(f'\nReach proxy, not the E11 policy half, spheres vs mesh+stick: {rates}')


def _first_contact(world: CWorld, a: int, b: int) -> float:
  """Smallest free dy between robots a and b on a rail, at home, by bisection."""
  pos, rot = default_bases(WorldCfg(num_robots=len(world.names)), 'cpu')
  empty = np.zeros((0, 3))
  world.set_scene(pos.numpy(), rot.numpy(), empty, np.zeros((0, 3, 3)), empty)
  world.set_q(np.tile(np.asarray(HOME_QPOS, dtype=np.float32), (len(world.names), 1)))
  base = world.data.mocap_pos
  hit, free = 0.05, 0.6
  while free - hit > 1e-5:
    gap = 0.5 * (free + hit)
    base[world.base_mocap[a], 1] = -gap / 2
    base[world.base_mocap[b], 1] = gap / 2
    world.forward()
    if world.flags()[[a, b]].any():
      hit = gap
    else:
      free = gap
  return free


@pytest.mark.parametrize('num_robots,a,b', [(4, 0, 3), (4, 1, 2), (8, 2, 7), (8, 3, 6)])
def test_same_rail_first_contact(num_robots, a, b):
  """E11: within [0.24, 0.30] m, reference 0.2588. (a, b) share a rail, on the ceiling at 8r."""
  world = CWorld(num_robots, 0, 'spheres_stick')
  gap = _first_contact(world, a, b)
  print(f'\n{num_robots}r robots {a},{b}: first free same-rail gap {gap:.4f} m')
  assert 0.24 <= gap <= 0.30


def test_same_rail_reference_on_the_mesh_tool_world():
  """The test must recover the reference, 0.2588, on the original model."""
  world = CWorld(4, 0, 'mesh_tool')
  assert _first_contact(world, 0, 3) == pytest.approx(0.2588, abs=5e-4)
