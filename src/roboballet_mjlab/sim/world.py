# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""The batched robot cell: mjlab Scene plus KinematicSim. DESIGN section 5, mjlab_api.md 3.5."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene as MjlabScene
from mjlab.scene import SceneCfg
from mjlab.sim.sim import MujocoCfg, SimulationCfg

from roboballet_mjlab.config import WorldCfg
from roboballet_mjlab.env.poses import rot_to_quat
from roboballet_mjlab.sim.kinematic_sim import KinematicSim
from roboballet_mjlab.paths import PACKAGE_DIR, data_dir
from roboballet_mjlab.structs import Scene

SPHERE_ARM_XML = PACKAGE_DIR / 'assets/panda_spheres_stick.xml'
MESH_ARM_XML = data_dir() / 'mujoco_world/assets/franka_emika_panda/panda_nohand.xml'
TABLE_XML = data_dir() / 'mujoco_world/assets/quad_table/quad_table.xml'
# The sphere arm's visual meshes are the mesh arm's, in the data directory wherever that is.
SPHERE_ARM_MESHDIR = data_dir() / 'mujoco_world/assets/franka_emika_panda/assets'

# Default rails and orientations, 8_pandas_world.xml:28-51. Euler in degrees.
BASE_POSES = {
  'panda1': ((0.5515, -0.6285, 0.009), (0, 0, 180)),
  'panda2': ((-0.5515, -0.6285, 0.009), (0, 0, 0)),
  'panda3': ((-0.5515, 0.6285, 0.009), (0, 0, 0)),
  'panda4': ((0.5515, 0.6285, 0.009), (0, 0, 180)),
  'panda5': ((0.5515, -0.6285, 1.8), (180, 0, 180)),
  'panda6': ((-0.5515, -0.6285, 1.8), (180, 0, 0)),
  'panda7': ((-0.5515, 0.6285, 1.8), (180, 0, 0)),
  'panda8': ((0.5515, 0.6285, 1.8), (180, 0, 180)),
}
BOX_PARKED_Z = 100.0  # Above everything until set_scene places the box.


class ContactOverflow(RuntimeError):
  """Contacts were dropped. Colliding worlds would look free, so never continue."""


def _base_quat(euler_deg) -> tuple[float, float, float, float]:
  quat = np.zeros(4)
  mujoco.mju_euler2Quat(quat, np.deg2rad(euler_deg), 'xyz')
  return tuple(quat.tolist())


def default_bases(cfg: WorldCfg, device) -> tuple[torch.Tensor, torch.Tensor]:
  """Default base poses in robot order: pos [R, 3], rot [R, 3, 3]."""
  pos, rot = [], []
  for name in cfg.robot_names:
    p, euler = BASE_POSES[name]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.array(_base_quat(euler)))
    pos.append(p)
    rot.append(mat.reshape(3, 3))
  return (torch.as_tensor(np.array(pos), dtype=torch.float32, device=device),
          torch.as_tensor(np.array(rot), dtype=torch.float32, device=device))


def _strip_visuals(spec: mujoco.MjSpec) -> None:
  """Visual geoms still cost kinematics, so training builds without them."""
  for geom in list(spec.geoms):
    if geom.contype == 0 and geom.conaffinity == 0:
      spec.delete(geom)
  used = {geom.meshname for geom in spec.geoms}
  for mesh in list(spec.meshes):
    if mesh.name not in used:
      spec.delete(mesh)


def _arm_spec(cfg: WorldCfg) -> mujoco.MjSpec:
  path = SPHERE_ARM_XML if cfg.arm_model == 'spheres_stick' else MESH_ARM_XML
  spec = mujoco.MjSpec.from_file(str(path))
  # Attach keeps a relative meshdir pointing at the wrong place otherwise.
  spec.meshdir = str(SPHERE_ARM_MESHDIR if path == SPHERE_ARM_XML else (path.parent / spec.meshdir).resolve())
  if cfg.arm_model == 'mesh_tool':
    for item in list(spec.actuators) + list(spec.lights):
      spec.delete(item)
    # Attach drops <option> anyway and mjlab warns: values from scene.py:227-236.
    spec.option.impratio = 1.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  if not cfg.visual_meshes:
    _strip_visuals(spec)
  return spec


def _add_statics(spec: mujoco.MjSpec, cfg: WorldCfg) -> None:
  spec.worldbody.add_geom(
    name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.95], size=[0, 0, 0.05],
    contype=1, conaffinity=1, rgba=[0.25, 0.3, 0.35, 1])
  if cfg.arm_model == 'spheres_stick':
    # DESIGN D6: the hull of the table frame is a nearly solid block.
    spec.worldbody.add_geom(
      name='table', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -0.488],
      size=[0.7841, 0.762, 0.488], contype=1, conaffinity=1, rgba=[0.5, 0.5, 0.5, 1])
  else:
    table = mujoco.MjSpec.from_file(str(TABLE_XML))
    table.meshdir = str(TABLE_XML.parent)
    if not cfg.visual_meshes:
      _strip_visuals(table)
    spec.attach(table, prefix='table/', frame=spec.worldbody.add_frame())
  for i in range(cfg.num_obstacles):
    body = spec.worldbody.add_body(name=f'box{i}', mocap=True, pos=[0, 0, BOX_PARKED_Z + i])
    # Weld id 0, like the table, the floor and link0: env_spec 5.2 filters those pairs.
    body.add_geom(
      name=f'box{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05],
      contype=1, conaffinity=1, rgba=[0.85, 0.65, 0.2, 1])


class World:
  """W independent copies of one cell. Robots are in `cfg.robot_names` order."""

  def __init__(self, cfg: WorldCfg, num_worlds: int, device) -> None:
    self.cfg = cfg
    self.num_worlds = num_worlds
    self.num_robots = cfg.num_robots
    self.num_obstacles = cfg.num_obstacles
    self.device = device
    self.launches = 0  # Graph launches so far, for profiling.

    names = cfg.robot_names
    entities = {
      name: EntityCfg(
        spec_fn=lambda: _arm_spec(cfg),
        init_state=EntityCfg.InitialStateCfg(
          pos=BASE_POSES[name][0], rot=_base_quat(BASE_POSES[name][1])),
      ) for name in names
    }
    self.scene = MjlabScene(
      SceneCfg(num_envs=num_worlds, env_spacing=0.0, terrain=None, entities=entities,
               spec_fn=lambda spec: _add_statics(spec, cfg)), device=str(device))
    self.sim = KinematicSim(
      num_envs=num_worlds,
      cfg=SimulationCfg(
        nconmax=cfg.nconmax, njmax=0, broadphase='sap_tile',
        broadphase_filter=('plane', 'sphere'),
        # Lets njmax=0 pass put_data's check at qpos0 (mujoco_warp io.py:1950).
        mujoco=MujocoCfg(disableflags=('constraint',))),
      spec=self.scene.spec, device=str(device))
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    # collision() returns early while the flag is set (collision_driver.py:910).
    self.sim.wp_model.opt.disableflags = 0
    self.sim.mj_model.opt.disableflags = 0
    # This replaces warp arrays and re-captures the graph, so it must come last.
    self.sim.expand_model_fields(('geom_size', 'geom_rbound', 'geom_aabb'))

    m = self.sim.mj_model
    if m.nq != 7 * self.num_robots or not np.array_equal(m.jnt_qposadr, np.arange(m.nq)):
      raise ValueError('qpos must be 7 contiguous values per robot, in entity order')
    tcp = 'tool_tip_site' if cfg.arm_model == 'spheres_stick' else 'tool/tool_tip_site'
    site = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'{n}/{tcp}') for n in names]
    base = [m.body(f'{n}/mocap_base').id for n in names]
    boxes = [m.body(f'box{i}').id for i in range(cfg.num_obstacles)]
    as_ids = lambda ids: torch.as_tensor(np.array(ids, dtype=np.int64), device=device)
    self._tcp_site = as_ids(site)
    self._base_mocap = as_ids([m.body_mocapid[b] for b in base])
    self._box_mocap = as_ids([m.body_mocapid[b] for b in boxes])
    self._box_geom = as_ids([m.body_geomadr[b] for b in boxes])
    self._geom_robot = as_ids(self._geom_to_robot(m, names))

    self._nacon = self.sim.data.nacon
    self._ncollision = self.sim.data.ncollision
    self._naconmax = self.sim.wp_data.naconmax
    self._contact = self.sim.data.contact
    self._n_contacts = 0

  @staticmethod
  def _geom_to_robot(m: mujoco.MjModel, names) -> list[int]:
    """Robot index per geom, len(names) for anything else (env_spec 5.1)."""
    out = []
    for g in range(m.ngeom):
      root = m.body_rootid[m.geom_bodyid[g]]
      owner = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root) or '').split('/')[0]
      out.append(names.index(owner) if owner in names else len(names))
    return out

  def set_scene(self, world_ids: torch.Tensor, scene: Scene) -> None:
    """Writes bases and boxes of the given worlds. `scene` is batched over world_ids."""
    d, m = self.sim.data, self.sim.model
    e = world_ids[:, None]
    d.mocap_pos[e, self._base_mocap] = scene.base_pos
    d.mocap_quat[e, self._base_mocap] = rot_to_quat(scene.base_rot)
    d.mocap_pos[e, self._box_mocap] = scene.box_pos
    d.mocap_quat[e, self._box_mocap] = rot_to_quat(scene.box_rot)
    half = scene.box_spans / 2
    m.geom_size[e, self._box_geom] = half
    m.geom_rbound[e, self._box_geom] = half.norm(dim=-1)
    m.geom_aabb[e, self._box_geom, 1] = half

  def set_q(self, q: torch.Tensor, world_ids: torch.Tensor | None = None) -> None:
    """Zero-copy write of joint positions, q is [W or len(world_ids), R, 7]."""
    if world_ids is None:
      self.sim.data.qpos[:] = q.flatten(1)
    else:
      self.sim.data.qpos[world_ids] = q.flatten(1)

  def forward(self) -> None:
    """Kinematics + collision in one captured graph. Raises on contact overflow."""
    self.sim.kinematics_collision()
    self.launches += 1
    nacon, ncollision = int(self._nacon[0]), int(self._ncollision[0])
    # Overflow is silent in this path and makes collisions vanish (mjlab_api 6.4).
    if max(nacon, ncollision) > self._naconmax:
      raise ContactOverflow(
        f'nacon {nacon}, ncollision {ncollision} exceed the pool of {self._naconmax} '
        f'({self.cfg.nconmax} per world): raise WorldCfg.nconmax')
    self._n_contacts = nacon

  def robots_in_collision(self) -> torch.Tensor:
    """[W, R] bool from the last forward(). dist <= 0, both robots of a pair flagged."""
    n = self._n_contacts
    hit = self._contact.dist[:n] <= 0  # Warp stores dist < 0 only, see mjlab_api 4.3.
    world = self._contact.worldid[:n][hit].long()
    robots = self._geom_robot[self._contact.geom[:n][hit].long()]
    flags = torch.zeros(self.num_worlds, self.num_robots + 1, dtype=torch.bool, device=self.device)
    flags[world, robots[:, 0]] = True
    flags[world, robots[:, 1]] = True
    return flags[:, :-1]

  def tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP pos [W, R, 3] and rot [W, R, 3, 3] from the last forward(), as copies."""
    return (self.sim.data.site_xpos[:, self._tcp_site],
            self.sim.data.site_xmat[:, self._tcp_site])


def build_world(cfg: WorldCfg, num_worlds: int, device) -> World:
  return World(cfg, num_worlds, device)
