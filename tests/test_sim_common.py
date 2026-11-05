"""Shared by the sim and env tests: scene loading and independent C MuJoCo worlds."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from roboballet_mjlab.config import HOME_QPOS, ROBOT_ORDER_4, ROBOT_ORDER_8, EnvCfg, WorldCfg
from roboballet_mjlab.paths import data_dir
from roboballet_mjlab.structs import Scene

DEVICE = 'cuda:0'
REPO = Path(__file__).resolve().parents[1]
# The original's data/ (mujoco_world, start_states, paper_test_tasksets), where the package reads it.
DATA = data_dir()
# The directory that holds the original's source tree as roboballet/, its venv as venv/ and the
# reference artefacts under runs/mjlab_port_analysis/. Defaults to the parent of this checkout.
PROJECT = Path(os.environ.get('ROBOBALLET_PROJECT', REPO.parent))
ORIGINAL = PROJECT / 'roboballet'
GOLDEN = Path(__file__).resolve().parent / 'golden'
WORLD_XML = {4: DATA / 'mujoco_world/4_pandas_world.xml',
             8: DATA / 'mujoco_world/8_pandas_world.xml'}
FAR_AWAY = 1000.0  # planning_env.py:57


def stored_scenes(num_robots: int, valid: bool | None = True, device=DEVICE) -> Scene:
  """Stored 10-target start states, in file order. `valid` selects by whether the spheres
  are free at home, None keeps them all."""
  z = np.load(GOLDEN / f'env_scenes_{num_robots}r10t.npz')
  keep = np.ones_like(z['valid']) if valid is None else z['valid'] == valid
  return Scene(*[torch.as_tensor(z[f][keep], device=device) for f in Scene._fields])


def default_scene(cfg: WorldCfg, n: int, num_targets: int = 10, device=DEVICE) -> Scene:
  """Default bases, boxes and targets parked far away, arms at home."""
  from roboballet_mjlab.sim.world import default_bases
  pos, rot = default_bases(cfg, device)
  R, O, T = cfg.num_robots, cfg.num_obstacles, num_targets
  eye = torch.eye(3, device=device)
  return Scene(
    base_pos=pos.repeat(n, 1, 1),
    base_rot=rot.repeat(n, 1, 1, 1),
    box_pos=torch.full((n, O, 3), FAR_AWAY, device=device),
    box_rot=eye.repeat(n, O, 1, 1),
    box_spans=torch.full((n, O, 3), 0.05, device=device),
    target_pos=torch.full((n, T, 3), FAR_AWAY, device=device),
    target_rot=eye.repeat(n, T, 1, 1),
    q0=torch.as_tensor(HOME_QPOS, device=device).repeat(n, R, 1),
  )


def env_cfg(num_robots=4, num_obstacles=30, arm='spheres_stick', nconmax=512, **kw) -> EnvCfg:
  world = WorldCfg(num_robots=num_robots, num_obstacles=num_obstacles, arm_model=arm, nconmax=nconmax)
  return EnvCfg(world=world, **kw)


def robot_names(num_robots: int) -> tuple[str, ...]:
  return ROBOT_ORDER_8 if num_robots > 4 else ROBOT_ORDER_4[:num_robots]


class CWorld:
  """One cell in C MuJoCo, composed without mjlab or anything in sim/world.py.

  'mesh_tool' is the original world XML, 'spheres_stick' puts the asset on mocap
  bodies the same way those XMLs do. Boxes are primitive. Midphase is off because the
  box sizes change per scene and its BVH would go stale (mjlab_api section 9).
  """

  def __init__(self, num_robots: int, num_boxes: int, arm: str = 'spheres_stick') -> None:
    self.names = robot_names(num_robots)
    self.arm = arm
    if arm == 'mesh_tool':
      spec = mujoco.MjSpec.from_file(str(WORLD_XML[8 if num_robots > 4 else 4]))
      spec.delete(spec.body('workpiece'))
      self.tcp_suffix = 'tool/tool_tip_site'
    else:
      spec = mujoco.MjSpec()
      spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.95],
                              size=[0, 0, 0.05])
      spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -0.488],
                              size=[0.7841, 0.762, 0.488])
      for name in self.names:
        arm_spec = mujoco.MjSpec.from_file(str(REPO / 'src/roboballet_mjlab/assets/panda_spheres_stick.xml'))
        for geom in list(arm_spec.geoms):
          if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            arm_spec.delete(geom)
        for mesh in list(arm_spec.meshes):
          arm_spec.delete(mesh)
        body = spec.worldbody.add_body(name=name, mocap=True)
        body.add_frame().attach_body(arm_spec.body('link0'), f'{name}/', '')
      self.tcp_suffix = 'tool_tip_site'
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
    self.tcp_site = np.array([m.site(f'{n}/{self.tcp_suffix}').id for n in self.names])
    self.link7 = np.array([m.body(f'{n}/link7').id for n in self.names])
    self.geom_robot = np.full(m.ngeom, -1)
    for g in range(m.ngeom):
      root = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_rootid[m.geom_bodyid[g]])
      if root in self.names:
        self.geom_robot[g] = self.names.index(root)

  def set_scene(self, base_pos, base_rot, box_pos, box_rot, box_spans) -> None:
    """All numpy, one world, in robot and box order."""
    d, m = self.data, self.model
    quat = np.zeros(4)
    for i, mocap in enumerate(self.base_mocap):
      mujoco.mju_mat2Quat(quat, np.asarray(base_rot[i], dtype=np.float64).ravel())
      d.mocap_pos[mocap] = base_pos[i]
      d.mocap_quat[mocap] = quat
    for i, mocap in enumerate(self.box_mocap):
      mujoco.mju_mat2Quat(quat, np.asarray(box_rot[i], dtype=np.float64).ravel())
      half = np.asarray(box_spans[i], dtype=np.float64) / 2
      d.mocap_pos[mocap] = box_pos[i]
      d.mocap_quat[mocap] = quat
      m.geom_size[self.box_geom[i]] = half
      m.geom_rbound[self.box_geom[i]] = np.linalg.norm(half)
      m.geom_aabb[self.box_geom[i], 3:] = half

  def set_q(self, q) -> None:
    self.data.qpos[self.q_adr] = q

  def forward(self) -> None:
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_collision(self.model, self.data)

  def flags(self) -> np.ndarray:
    """[R] bool, RobotsInCollision of mujoco_simulator.py:595-611."""
    out = np.zeros(len(self.names), dtype=bool)
    con = self.data.contact
    for i in range(self.data.ncon):
      if con.dist[i] <= 0:
        for g in con.geom[i]:
          if self.geom_robot[g] >= 0:
            out[self.geom_robot[g]] = True
    return out

  def tcp(self) -> tuple[np.ndarray, np.ndarray]:
    return (self.data.site_xpos[self.tcp_site].copy(),
            self.data.site_xmat[self.tcp_site].reshape(-1, 3, 3).copy())
