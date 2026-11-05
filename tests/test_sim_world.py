"""Gate G1: the World against C MuJoCo. paper_baseline 10.1 E1, E12, E13; DESIGN section 9."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest
import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, ROBOT_ORDER_8, WorldCfg
from roboballet_mjlab.env.poses import rot_to_quat
from roboballet_mjlab.sim.world import ContactOverflow, build_world, default_bases
from tests.test_sim_common import (
  DEVICE, REPO, CWorld, default_scene, stored_scenes,
)

ASSETS = REPO / 'src/roboballet_mjlab/assets'
LOWER = torch.as_tensor(JOINT_LOWER, device=DEVICE)
UPPER = torch.as_tensor(JOINT_UPPER, device=DEVICE)
HOME = torch.as_tensor(HOME_QPOS, device=DEVICE)


def _home(n, num_robots):
  return HOME.repeat(n, num_robots, 1)


def _random_q(n, num_robots, gen):
  return LOWER + (UPPER - LOWER) * torch.rand(n, num_robots, 7, generator=gen, device=DEVICE)


def _gen(seed=0):
  return torch.Generator(device=DEVICE).manual_seed(seed)


def _pool_scene(num_robots, n, gen):
  pool = stored_scenes(num_robots)
  ids = torch.randint(len(pool.q0), (n,), generator=gen, device=DEVICE)
  return type(pool)(*[x[ids] for x in pool])


@pytest.fixture(scope='module')
def world4():
  return build_world(WorldCfg(), 1024, DEVICE)


@pytest.fixture(scope='module')
def world8():
  return build_world(WorldCfg(num_robots=8), 1024, DEVICE)


@pytest.fixture(scope='module')
def world_mesh():
  return build_world(WorldCfg(arm_model='mesh_tool'), 256, DEVICE)


# Asset.


def test_asset_matches_json():
  spec = mujoco.MjSpec.from_file(str(ASSETS / 'panda_spheres_stick.xml'))
  data = json.load(open(ASSETS / 'spheres.json'))
  for link, spheres in data['spheres'].items():
    assert len(spheres) == (5 if link in ('link5', 'link6', 'link7') else 3)
    for i, sphere in enumerate(spheres):
      geom = spec.geom(f'{link}_s{i}')
      assert geom.size[0] == pytest.approx(sphere['radius'], abs=1e-9)
      assert np.allclose(geom.pos, sphere['center'], atol=1e-9)
  stick = data['stick']
  assert stick['L'] == 0.16 and stick['R'] == 0.015
  for i, z in enumerate(stick['z']):
    geom = spec.geom(f'stick_s{i}')
    assert geom.size[0] == 0.015 and np.allclose(geom.pos, [0, 0, z], atol=1e-9)
  assert sorted([e.bodyname1, e.bodyname2] for e in spec.excludes) == sorted(data['excludes'])
  assert len(data['excludes']) == 16


def test_asset_refers_to_repo_meshes():
  assert not list(ASSETS.glob('**/*.obj')) and not list(ASSETS.glob('**/*.stl'))
  spec = mujoco.MjSpec.from_file(str(ASSETS / 'panda_spheres_stick.xml'))
  assert spec.meshes
  for mesh in spec.meshes:
    assert (ASSETS / spec.meshdir / mesh.file).is_file()


# Composition.


@pytest.mark.parametrize('num_robots', [4, 8])
def test_composition_counts_and_excludes(num_robots, world4, world8):
  world = world4 if num_robots == 4 else world8
  m = world.sim.mj_model
  assert m.nq == 7 * num_robots and m.nu == 0
  assert m.ngeom == 36 * num_robots + 2 + 30  # No meshes in training.
  assert m.nmesh == 0
  assert m.nmocap == num_robots + 30
  assert m.nexclude == 16 * num_robots
  expected = {tuple(e) for e in json.load(open(ASSETS / 'spheres.json'))['excludes']}
  for name in WorldCfg(num_robots=num_robots).robot_names:
    found = set()
    for b1, b2 in zip(m.exclude_signature >> 16, m.exclude_signature & 0xffff):
      n1, n2 = m.body(b1).name, m.body(b2).name
      if n1.startswith(f'{name}/') and n2.startswith(f'{name}/'):
        found.add((n1.split('/')[1], n2.split('/')[1]))
    assert {tuple(sorted(e)) for e in expected} == {tuple(sorted(e)) for e in found}


def test_excluded_pairs_are_absent_from_warp():
  world = build_world(WorldCfg(num_robots=1, num_obstacles=1), 2, DEVICE)
  m = world.sim.mj_model
  pairs = world.sim.wp_model.nxn_geom_pair_filtered.numpy()
  bodies = {tuple(sorted((m.body(m.geom_bodyid[a]).name.split('/')[-1],
                          m.body(m.geom_bodyid[b]).name.split('/')[-1]))) for a, b in pairs}
  for e in json.load(open(ASSETS / 'spheres.json'))['excludes']:
    assert tuple(sorted(e)) not in bodies
  # Live pairs of sphere_model section 4 must remain.
  for a, b in [('link0', 'link5'), ('link2', 'link7'), ('link1', 'stick')]:
    assert tuple(sorted((a, b))) in bodies


def test_visual_meshes_are_optional():
  world = build_world(WorldCfg(visual_meshes=True), 2, DEVICE)
  m = world.sim.mj_model
  assert m.nmesh > 0
  meshes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
  assert meshes and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in meshes)
  # Same collision geometry as the training build.
  plain = build_world(WorldCfg(), 2, DEVICE)
  assert (world.sim.wp_model.nxn_geom_pair_filtered.shape[0]
          >= plain.sim.wp_model.nxn_geom_pair_filtered.shape[0])
  scene = default_scene(WorldCfg(), 2)
  for w in (world, plain):
    w.set_scene(torch.arange(2, device=DEVICE), scene)
    w.set_q(_home(2, 4))
    w.forward()
    assert not w.robots_in_collision().any()
  assert torch.allclose(world.tcp()[0], plain.tcp()[0])


def test_mesh_tool_model():
  world = build_world(WorldCfg(arm_model='mesh_tool', num_obstacles=2, nconmax=64), 2, DEVICE)
  m = world.sim.mj_model
  mesh_collision = [g for g in range(m.ngeom)
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]]
  assert len(mesh_collision) == 4 * 15 + 2  # env_spec 3.4, plus the two table meshes.
  assert m.nexclude == 0
  site = m.site('panda1/tool/tool_tip_site').id
  assert site >= 0
  # env_spec 3.3: hull bounds of the table meshes.
  bounds = {}
  table = [g for g in mesh_collision if m.geom_bodyid[g] and
           m.body(m.body_rootid[m.geom_bodyid[g]]).name.startswith('table')]
  assert len(table) == 2
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  for g in table:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    bounds[g] = (w.min(0), w.max(0))
  low = np.min([b[0] for b in bounds.values()], 0)
  high = np.max([b[1] for b in bounds.values()], 0)
  assert np.allclose(low, [-0.7828, -0.762, -0.9758], atol=1e-3)
  assert np.allclose(high, [0.7841, 0.762, 0.0], atol=1e-3)


# Poses and home.


def test_tcp_in_link7(world4):
  gen = _gen(1)
  n = world4.num_worlds
  world4.set_scene(torch.arange(n, device=DEVICE), default_scene(world4.cfg, n))
  world4.set_q(_random_q(n, 4, gen))
  world4.forward()
  m = world4.sim.mj_model
  link7 = torch.as_tensor([m.body(f'{r}/link7').id for r in world4.cfg.robot_names], device=DEVICE)
  p7 = world4.sim.data.xpos[:, link7]
  r7 = world4.sim.data.xmat[:, link7]
  pos, rot = world4.tcp()
  local = (r7.transpose(-1, -2) @ (pos - p7)[..., None]).squeeze(-1)
  assert (local - torch.tensor([0, 0, 0.26], device=DEVICE)).abs().max() < 2e-5
  # TCP z is link7 z, the stick has no tilt. Yaw is the attachment's 180 degrees.
  rel = r7.transpose(-1, -2) @ rot
  assert (rel - torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=DEVICE))).abs().max() < 2e-3


@pytest.mark.parametrize('kind', ['4r', '8r', 'mesh'])
def test_home_is_free(kind, world4, world8, world_mesh):
  world = {'4r': world4, '8r': world8, 'mesh': world_mesh}[kind]
  n = world.num_worlds
  world.set_scene(torch.arange(n, device=DEVICE), default_scene(world.cfg, n))
  world.set_q(_home(n, world.num_robots))
  world.forward()
  assert world._n_contacts == 0
  assert not world.robots_in_collision().any()


def test_home_tips_match_spec(world4, world_mesh):
  # env_spec section 15, FK rows. Original tool first, then the stick.
  world_mesh.set_scene(torch.arange(256, device=DEVICE), default_scene(world_mesh.cfg, 256))
  world_mesh.set_q(_home(256, 4))
  world_mesh.forward()
  pos, rot = world_mesh.tcp()
  expected = [[0.31328, -0.6285, 0.51928], [-0.31328, -0.6285, 0.51928],
              [-0.31328, 0.6285, 0.51928], [0.31328, 0.6285, 0.51928]]
  assert torch.allclose(pos[0], torch.tensor(expected, device=DEVICE), atol=1e-4)
  panda1 = torch.tensor([[0.76598, 9.26e-5, 0.642865], [7.09e-5, -1.0, 5.95e-5],
                         [0.642865, 0.0, -0.76598]], device=DEVICE)
  assert torch.allclose(rot[0, 0], panda1, atol=1e-3)

  world4.set_scene(torch.arange(1024, device=DEVICE), default_scene(world4.cfg, 1024))
  world4.set_q(_home(1024, 4))
  world4.forward()
  pos, rot = world4.tcp()
  assert torch.allclose(pos[0, 0], torch.tensor([0.5515 - 0.2867, -0.6285, 0.009 + 0.5027],
                                                device=DEVICE), atol=2e-4)
  assert rot[0, 0, 2, 2] < -0.9999  # Stick z points straight down at home.


def test_default_bases_match_original_xml():
  for num_robots in (4, 8):
    cfg = WorldCfg(num_robots=num_robots)
    pos, rot = default_bases(cfg, DEVICE)
    m = CWorld(num_robots, 1, 'mesh_tool').model
    for i, name in enumerate(cfg.robot_names):
      body = m.body(name)
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, body.quat)
      assert np.allclose(pos[i].cpu().numpy(), body.pos, atol=1e-6)
      assert np.allclose(rot[i].cpu().numpy().ravel(), mat, atol=1e-6)
  panda1 = default_bases(WorldCfg(), DEVICE)[1][0]
  assert torch.allclose(panda1, torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=DEVICE)), atol=1e-6)


def test_tcp_returns_copies(world4):
  n = world4.num_worlds
  world4.set_scene(torch.arange(n, device=DEVICE), default_scene(world4.cfg, n))
  world4.set_q(_home(n, 4))
  world4.forward()
  pos, rot = world4.tcp()
  held_pos, held_rot = pos.clone(), rot.clone()
  world4.set_q(_home(n, 4) + 0.1)
  world4.forward()
  assert torch.equal(pos, held_pos) and torch.equal(rot, held_rot)
  assert not torch.equal(world4.tcp()[0], held_pos)


def test_robot_order_of_8_robot_world(world8):
  world = world8
  scene = default_scene(world.cfg, world.num_worlds)
  world.set_scene(torch.arange(world.num_worlds, device=DEVICE), scene)
  world.set_q(_home(world.num_worlds, 8))
  world.forward()
  assert world.cfg.robot_names == ROBOT_ORDER_8
  pos, _ = world.tcp()
  # panda1, panda2, panda5, panda6 are on the y < 0 rails, the rest on y > 0.
  y = pos[0, :, 1]
  assert (y[:4] < 0).all() and (y[4:] > 0).all()
  # Ceiling robots are indices 2, 3, 6, 7.
  z = pos[0, :, 2]
  assert (z[[2, 3, 6, 7]] > 1.0).all() and (z[[0, 1, 4, 5]] < 1.0).all()


# E1: FK of link7 and TCP against C MuJoCo, 10,000 random (q, placement), per robot.


def _angle(a, b):
  cos = ((np.trace(a.transpose(0, 2, 1) @ b, axis1=1, axis2=2) - 1) / 2).clip(-1, 1)
  return np.arccos(cos)


def _random_bases(cfg, n, gen):
  scene = default_scene(cfg, n)
  pos = scene.base_pos.clone()
  pos[..., 1] = -0.8 + 1.6 * torch.rand(n, cfg.num_robots, generator=gen, device=DEVICE)
  return scene._replace(base_pos=pos)


@pytest.mark.parametrize('cfg,n', [
  (WorldCfg(nconmax=128), 2500),
  (WorldCfg(num_robots=8, nconmax=128), 1250),
  (WorldCfg(arm_model='mesh_tool', num_obstacles=1, nconmax=256), 2500),
], ids=['spheres4', 'spheres8', 'mesh4'])
def test_e1_fk(cfg, n):
  gen = _gen(2)
  world = build_world(cfg, n, DEVICE)
  scene = _random_bases(cfg, n, gen)
  q = _random_q(n, cfg.num_robots, gen)
  world.set_scene(torch.arange(n, device=DEVICE), scene)
  world.set_q(q)
  world.forward()
  pos, rot = (x.cpu().numpy() for x in world.tcp())
  m = world.sim.mj_model
  link7 = [m.body(f'{r}/link7').id for r in cfg.robot_names]
  l7_pos = world.sim.data.xpos[:, link7].cpu().numpy()
  l7_rot = world.sim.data.xmat[:, link7].cpu().numpy()

  c = CWorld(cfg.num_robots, cfg.num_obstacles, cfg.arm_model)
  # The original XML has the original chain whatever the arm, so link7 is checked on it too.
  original = CWorld(cfg.num_robots, 1, 'mesh_tool')
  bp, br = scene.base_pos.cpu().numpy(), scene.base_rot.cpu().numpy()
  bxp, bxr, bxs = (x.cpu().numpy() for x in (scene.box_pos, scene.box_rot, scene.box_spans))
  qn = q.cpu().numpy()
  c_pos, c_rot, c_l7p, c_l7r, o_l7p, o_l7r = [], [], [], [], [], []
  for w in range(n):
    c.set_scene(bp[w], br[w], bxp[w], bxr[w], bxs[w])
    c.set_q(qn[w])
    mujoco.mj_kinematics(c.model, c.data)
    p, r = c.tcp()
    c_pos.append(p)
    c_rot.append(r)
    c_l7p.append(c.data.xpos[c.link7].copy())
    c_l7r.append(c.data.xmat[c.link7].reshape(-1, 3, 3).copy())
    original.set_scene(bp[w], br[w], bxp[w][:1], bxr[w][:1], bxs[w][:1])
    original.set_q(qn[w])
    mujoco.mj_kinematics(original.model, original.data)
    o_l7p.append(original.data.xpos[original.link7].copy())
    o_l7r.append(original.data.xmat[original.link7].reshape(-1, 3, 3).copy())
  samples = n * cfg.num_robots
  assert samples >= 10_000
  assert np.abs(pos - np.array(c_pos)).max() <= 1e-4
  assert _angle(rot.reshape(-1, 3, 3), np.array(c_rot).reshape(-1, 3, 3)).max() <= 1e-3
  for ref_p, ref_r in ((c_l7p, c_l7r), (o_l7p, o_l7r)):
    assert np.abs(l7_pos - np.array(ref_p)).max() <= 1e-4
    assert _angle(l7_rot.reshape(-1, 3, 3), np.array(ref_r).reshape(-1, 3, 3)).max() <= 1e-3


# E12: static pairs.


def test_e12_no_weld0_pairs(world4, world_mesh):
  for world in (world4, world_mesh):
    m = world.sim.mj_model
    pairs = world.sim.wp_model.nxn_geom_pair_filtered.numpy()
    weld = m.body_weldid[m.geom_bodyid]
    assert not ((weld[pairs[:, 0]] == 0) & (weld[pairs[:, 1]] == 0)).any()


def _static_scene(cfg, n):
  """Every world holds overlaps that must not count: env_spec 5.3, E12."""
  scene = default_scene(cfg, n)
  pos, spans = scene.box_pos.clone(), scene.box_spans.clone()
  pos[:, 0] = torch.tensor([0.0, 0.0, 0.5], device=DEVICE)  # Box inside box inside box.
  pos[:, 1] = torch.tensor([0.0, 0.0, 0.5], device=DEVICE)
  spans[:, 0] = spans[:, 1] = 0.2
  pos[:, 2] = torch.tensor([0.0, 0.0, -0.1], device=DEVICE)  # Box in the table.
  spans[:, 2] = 0.3
  pos[:, 3] = torch.tensor([0.0, 0.0, -0.95], device=DEVICE)  # Box through the floor.
  spans[:, 3] = 0.1
  # Box inside panda1's link0, well below link1. Base at x = 0.5515, facing -x.
  pos[:, 4] = torch.tensor([0.5515 + 0.03, -0.6285, 0.009 + 0.03], device=DEVICE)
  spans[:, 4] = 0.04
  return scene._replace(box_pos=pos, box_spans=spans)


@pytest.mark.parametrize('kind', ['spheres', 'mesh'])
def test_e12_static_overlaps_do_not_count(kind, world4, world_mesh):
  world = world4 if kind == 'spheres' else world_mesh
  n = world.num_worlds
  scene = _static_scene(world.cfg, n)
  world.set_scene(torch.arange(n, device=DEVICE), scene)
  world.set_q(_home(n, 4))
  world.forward()
  assert world._n_contacts == 0
  assert not world.robots_in_collision().any()
  # The same overlaps are real ones to C MuJoCo's geometry, but filtered there too.
  c = CWorld(4, world.num_obstacles, world.cfg.arm_model)
  c.set_scene(*(x[0].cpu().numpy() for x in (scene.base_pos, scene.base_rot, scene.box_pos,
                                             scene.box_rot, scene.box_spans)))
  c.set_q(_home(1, 4)[0].cpu().numpy())
  c.forward()
  assert c.data.ncon == 0


def test_e12_base_into_table_does_not_count(world4):
  n = world4.num_worlds
  scene = default_scene(world4.cfg, n)
  pos = scene.base_pos.clone()
  pos[..., 2] = -0.05
  world4.set_scene(torch.arange(n, device=DEVICE), scene._replace(base_pos=pos))
  world4.set_q(_home(n, 4))
  world4.forward()
  assert not world4.robots_in_collision().any()


@pytest.mark.parametrize('kind', ['spheres', 'mesh'])
def test_box_on_link4_flags_only_that_robot(kind, world4, world_mesh):
  world = world4 if kind == 'spheres' else world_mesh
  n = world.num_worlds
  world.set_scene(torch.arange(n, device=DEVICE), default_scene(world.cfg, n))
  world.set_q(_home(n, 4))
  world.forward()
  m = world.sim.mj_model
  link4 = m.body('panda2/link4').id
  at = world.sim.data.xpos[0, link4].clone()
  scene = default_scene(world.cfg, n)
  pos = scene.box_pos.clone()
  pos[:n // 2, 0] = at  # Only half of the worlds get the box.
  world.set_scene(torch.arange(n, device=DEVICE), scene._replace(box_pos=pos))
  world.forward()
  flags = world.robots_in_collision()
  assert flags[:n // 2, 1].all() and flags[:n // 2].sum() == n // 2
  assert not flags[n // 2:].any()


def test_robot_pair_flags_both():
  cfg = WorldCfg(num_obstacles=1)
  world = build_world(cfg, 2, DEVICE)
  scene = default_scene(cfg, 2)
  pos = scene.base_pos.clone()
  pos[1, 3, 1] = pos[1, 0, 1]  # World 1: panda4 on top of panda1, same rail.
  world.set_scene(torch.arange(2, device=DEVICE), scene._replace(base_pos=pos))
  world.set_q(_home(2, 4))
  world.forward()
  flags = world.robots_in_collision().cpu().numpy().tolist()
  assert flags == [[False] * 4, [True, False, False, True]]


# E13 and per-world box pose and size parity.


def _copy_world_to_c(c_model, c_data, world, w):
  """Same compiled model as warp: every per-world field goes across."""
  c_model.geom_size[:] = world.sim.model.geom_size[w].cpu().numpy()
  c_model.geom_rbound[:] = world.sim.model.geom_rbound[w].cpu().numpy()
  c_model.geom_aabb[:] = world.sim.model.geom_aabb[w].flatten(0).cpu().numpy().reshape(
    c_model.geom_aabb.shape)
  c_data.qpos[:] = world.sim.data.qpos[w].cpu().numpy()
  c_data.mocap_pos[:] = world.sim.data.mocap_pos[w].cpu().numpy()
  c_data.mocap_quat[:] = world.sim.data.mocap_quat[w].cpu().numpy()
  mujoco.mj_kinematics(c_model, c_data)
  mujoco.mj_collision(c_model, c_data)


def _scatter_states(world, gen, frac_home=0.5):
  """Half near home, half uniform: both contact regimes of the state space."""
  n = world.num_worlds
  R = world.num_robots
  scene = _pool_scene(R, n, gen)
  world.set_scene(torch.arange(n, device=DEVICE), scene)
  q = _random_q(n, R, gen)
  near = HOME + 0.4 * (torch.rand(n, R, 7, generator=gen, device=DEVICE) - 0.5)
  k = int(n * frac_home)
  q[:k] = torch.minimum(torch.maximum(near[:k], LOWER), UPPER)
  world.set_q(q)
  world.forward()
  return scene, q


@pytest.mark.parametrize('num_robots', [4, 8])
def test_e13_contact_sets_match_c(num_robots, world4, world8):
  world = world4 if num_robots == 4 else world8
  _scatter_states(world, _gen(3))
  m = world.sim.mj_model
  flags = m.opt.disableflags
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
  c_data = mujoco.MjData(m)
  try:
    n = world._n_contacts
    wid = world.sim.data.contact.worldid[:n].cpu().numpy()
    geoms = np.sort(world.sim.data.contact.geom[:n].cpu().numpy(), 1)
    dist = world.sim.data.contact.dist[:n].cpu().numpy()
    only_warp = only_c = total = 0
    max_dist = 0.0
    for w in range(world.num_worlds):
      _copy_world_to_c(m, c_data, world, w)
      a = {tuple(sorted(c_data.contact.geom[i].tolist())): c_data.contact.dist[i]
           for i in range(c_data.ncon)}
      sel = wid == w
      b = dict(zip(map(tuple, geoms[sel].tolist()), dist[sel].tolist()))
      total += len(a)
      only_c += len(set(a) - set(b))
      only_warp += len(set(b) - set(a))
      for key in set(a) & set(b):
        max_dist = max(max_dist, abs(a[key] - b[key]))
    assert total > 1000  # The comparison has to mean something.
    assert only_c == 0 and only_warp == 0
    assert max_dist < 1e-5
    print(f'E13 {num_robots}r: {total} contacts identical, max |dist diff| {max_dist:.2e}')
  finally:
    m.opt.disableflags = flags


@pytest.mark.parametrize('num_robots', [4, 8])
def test_e13_labels_match_independent_c_world(num_robots, world4, world8):
  world = world4 if num_robots == 4 else world8
  gen = _gen(4)
  c = CWorld(num_robots, 30)
  labels = disagree = positives = 0
  for _ in range(16):  # 16 x 1024 worlds, per-robot labels.
    scene, q = _scatter_states(world, gen)
    flags = world.robots_in_collision().cpu().numpy()
    arrays = [x.cpu().numpy() for x in (scene.base_pos, scene.base_rot, scene.box_pos,
                                        scene.box_rot, scene.box_spans)]
    qn = q.cpu().numpy()
    for w in range(world.num_worlds):
      c.set_scene(*(a[w] for a in arrays))
      c.set_q(qn[w])
      c.forward()
      ref = c.flags()
      labels += num_robots
      disagree += int((ref != flags[w]).sum())
      positives += int(ref.sum())
  assert labels >= 16 * 1024 * 4
  assert positives > labels / 10
  assert disagree / labels <= 1e-4
  print(f'E13 {num_robots}r: {disagree} of {labels} labels differ, {positives} positive')


@pytest.mark.parametrize('num_robots', [4, 8])
def test_per_world_box_parity(num_robots, world4, world8):
  world = world4 if num_robots == 4 else world8
  gen = _gen(5)
  n = world.num_worlds
  scene = _pool_scene(num_robots, n, gen)
  ids = torch.arange(n, device=DEVICE)
  world.set_scene(ids, scene)
  world.set_q(_home(n, num_robots))
  world.forward()
  # Sizes and poses as written, and they really differ per world.
  g = world._box_geom
  half = scene.box_spans / 2
  assert torch.equal(world.sim.model.geom_size[:, g], half)
  assert torch.equal(world.sim.model.geom_rbound[:, g], half.norm(dim=-1))
  assert torch.equal(world.sim.model.geom_aabb[:, g, 1], half)
  assert (half[0] != half[1]).any() and world.sim.data.geom_xpos[0, g[0]].ne(
    world.sim.data.geom_xpos[1, g[0]]).any()
  # Pose against the Scene.
  assert (world.sim.data.geom_xpos[:, g] - scene.box_pos).abs().max() < 1e-6
  assert (world.sim.data.geom_xmat[:, g] - scene.box_rot).abs().max() < 1e-5
  base = world._base_mocap
  assert (world.sim.data.mocap_pos[:, base] - scene.base_pos).abs().max() == 0
  # Pose and size against C.
  m = world.sim.mj_model
  flags = m.opt.disableflags
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
  c_data = mujoco.MjData(m)
  try:
    for w in range(0, n, 16):
      _copy_world_to_c(m, c_data, world, w)
      xpos = world.sim.data.geom_xpos[w].cpu().numpy()
      xmat = world.sim.data.geom_xmat[w].cpu().numpy().reshape(m.ngeom, 9)
      assert np.abs(xpos - c_data.geom_xpos).max() < 1e-5
      assert np.abs(xmat - c_data.geom_xmat).max() < 1e-5
      assert np.array_equal(c_data.geom_xpos[g.cpu().numpy()].round(4),
                            scene.box_pos[w].double().cpu().numpy().round(4))
  finally:
    m.opt.disableflags = flags
  # A partial write leaves every other world alone.
  before = world.sim.model.geom_size[:, g].clone()
  mocap_before = world.sim.data.mocap_pos.clone()
  other = _pool_scene(num_robots, 3, gen)
  some = torch.tensor([1, 5, 9], device=DEVICE)
  world.set_scene(some, other)
  after = world.sim.model.geom_size[:, g]
  mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
  mask[some] = False
  assert torch.equal(after[mask], before[mask])
  assert torch.equal(after[some], other.box_spans / 2)
  assert torch.equal(world.sim.data.mocap_pos[mask], mocap_before[mask])


def test_rot_to_quat_matches_c():
  gen = _gen(6)
  a = torch.randn(4096, 4, generator=gen, device=DEVICE)
  a = a / a.norm(dim=-1, keepdim=True)
  # Rotations near each branch of the conversion, plus the exact half turns of the bases.
  mats = np.zeros((4096, 9))
  for i, quat in enumerate(a.double().cpu().numpy()):
    mujoco.mju_quat2Mat(mats[i], quat)
  mats[0] = np.diag([-1.0, -1.0, 1.0]).ravel()
  mats[1] = np.diag([-1.0, 1.0, -1.0]).ravel()
  mats[2] = np.diag([1.0, -1.0, -1.0]).ravel()
  mats[3] = np.eye(3).ravel()
  quat = rot_to_quat(torch.as_tensor(mats.reshape(-1, 3, 3), dtype=torch.float32, device=DEVICE))
  back = np.zeros(9)
  for i, q in enumerate(quat.double().cpu().numpy()):
    assert abs(np.linalg.norm(q) - 1) < 1e-6
    mujoco.mju_quat2Mat(back, q)
    assert np.abs(back - mats[i]).max() < 1e-5


# Overflow guard.


def test_overflow_raises_when_forced():
  cfg = WorldCfg(nconmax=1, num_robots=8)
  world = build_world(cfg, 16, DEVICE)
  gen = _gen(7)
  world.set_scene(torch.arange(16, device=DEVICE), _pool_scene(8, 16, gen))
  world.set_q(_random_q(16, 8, gen))
  with pytest.raises(ContactOverflow, match='raise WorldCfg.nconmax'):
    world.forward()


def test_overflow_guard_is_quiet_with_default_pool_and_has_headroom(world8):
  gen = _gen(8)
  peak = 0
  for _ in range(20):
    _scatter_states(world8, gen, frac_home=0.0)  # Uniform q is the heavy case.
    peak = max(peak, int(world8.sim.data.ncollision[0]), world8._n_contacts)
  assert peak * 2 < world8.sim.wp_data.naconmax
  print(f'peak candidates per world {peak / world8.num_worlds:.1f} of {world8.cfg.nconmax}')


def test_overflow_guard_catches_nacon_alone(monkeypatch):
  # mjlab_api 6.4 [CORRECTED]: a multi-contact pair can pass ncollision alone. Spheres
  # cannot produce that, so set the counters by hand and skip the launch that clears them.
  world = build_world(WorldCfg(num_obstacles=1), 2, DEVICE)
  monkeypatch.setattr(world.sim, 'kinematics_collision', lambda: None)
  world.sim.data.ncollision[0] = 0
  world.sim.data.nacon[0] = world.sim.wp_data.naconmax
  world.forward()
  world.sim.data.nacon[0] = world.sim.wp_data.naconmax + 1
  with pytest.raises(ContactOverflow):
    world.forward()


# env_spec section 15: self-collision pin and mesh parity with the original world.


def test_self_collision_pin_on_mesh(world_mesh):
  n = world_mesh.num_worlds
  world_mesh.set_scene(torch.arange(n, device=DEVICE), default_scene(world_mesh.cfg, n))
  q = _home(n, 4)
  q[:, 0] = torch.tensor([0.0, -0.7854, -0.156768, -3.059768, 0.0, 1.3962, 0.0], device=DEVICE)
  world_mesh.set_q(q)
  world_mesh.forward()
  flags = world_mesh.robots_in_collision()
  assert flags[:, 0].all() and not flags[:, 1:].any()


def test_random_collision_parity_on_mesh_and_sphere_rates():
  """env_spec 15, last row: flagged sets on stored scenes, with the original XML as truth."""
  n = 512
  gen = _gen(9)
  cfg = WorldCfg(arm_model='mesh_tool')
  mesh = build_world(cfg, n, DEVICE)
  spheres = build_world(WorldCfg(), n, DEVICE)
  c = CWorld(4, 30, 'mesh_tool')
  disagree = labels = positives = 0
  false_pos = false_neg = free = hit = 0
  pool = stored_scenes(4, valid=True)
  for _ in range(4):
    ids = torch.randint(len(pool.q0), (n,), generator=gen, device=DEVICE)
    scene = type(pool)(*[x[ids] for x in pool])
    q = _random_q(n, 4, gen)
    near = HOME + 0.4 * (torch.rand(n, 4, 7, generator=gen, device=DEVICE) - 0.5)
    q[: n // 2] = torch.minimum(torch.maximum(near[: n // 2], LOWER), UPPER)
    for world in (mesh, spheres):
      world.set_scene(torch.arange(n, device=DEVICE), scene)
      world.set_q(q)
      world.forward()
    flags = mesh.robots_in_collision().cpu().numpy()
    sphere_flags = spheres.robots_in_collision().cpu().numpy()
    arrays = [x.cpu().numpy() for x in (scene.base_pos, scene.base_rot, scene.box_pos,
                                        scene.box_rot, scene.box_spans)]
    qn = q.cpu().numpy()
    for w in range(n):
      c.set_scene(*(a[w] for a in arrays))
      c.set_q(qn[w])
      c.forward()
      ref = c.flags()
      labels += 4
      disagree += int((ref != flags[w]).sum())
      positives += int(ref.sum())
      false_pos += int((sphere_flags[w] & ~ref).sum())
      false_neg += int((~sphere_flags[w] & ref).sum())
      free += int((~ref).sum())
      hit += int(ref.sum())
  print(f'mesh warp vs original XML: {disagree} of {labels} labels differ ({positives} positive)')
  print(f'spheres vs original mesh+tool: FP {false_pos}/{free} = {false_pos / free:.4f}, '
        f'FN {false_neg}/{hit} = {false_neg / hit:.4f}')
  assert positives > 500
  assert disagree / labels <= 1e-3
