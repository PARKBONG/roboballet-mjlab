"""Independent attack on sim/world.py and the sphere assets. DESIGN G1, mjlab_api 4-6 and 9.

Three oracles that share nothing with World: C MuJoCo on `CWorld`, C MuJoCo compiled afresh
with the true box sizes, and a brute-force signed distance over every geom pair.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial import ConvexHull

import roboballet_mjlab.sim.kinematic_sim as sim_module
import roboballet_mjlab.sim.world as world_module
from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, WorldCfg
from roboballet_mjlab.scenes import sampler
from roboballet_mjlab.sim.world import ContactOverflow, build_world
from tests.test_sim_common import (
  DEVICE, REPO, CWorld, default_scene, robot_names, stored_scenes,
)

ASSETS = REPO / 'src/roboballet_mjlab/assets'
SPHERES = json.load(open(ASSETS / 'spheres.json'))
LOWER = torch.as_tensor(JOINT_LOWER, device=DEVICE)
UPPER = torch.as_tensor(JOINT_UPPER, device=DEVICE)
HOME = torch.as_tensor(HOME_QPOS, device=DEVICE)
COMPILED_RBOUND = 0.05 * 3 ** 0.5  # Of the box as compiled, half size 0.05.
BAND = 1e-5  # Metres. Inside it float32 warp and float64 C may disagree. See the band test.
TIE = 1e-6  # Identical stick spheres against identical geometry, rounded differently.


def _gen(seed=0):
  return torch.Generator(device=DEVICE).manual_seed(seed)


def _rand(gen, *shape):
  return torch.rand(*shape, generator=gen, device=DEVICE)


def _uniform_q(n, num_robots, gen):
  return LOWER + (UPPER - LOWER) * _rand(gen, n, num_robots, 7)


def _random_rot(n, m, gen):
  q = torch.randn(n, m, 4, generator=gen, device=DEVICE)
  w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ], -1).view(n, m, 3, 3)


def _hostile(cfg, n, gen):
  """Spans up to 3 m against the 0.1 m compiled, rotated boxes and bases, arms on the floor."""
  scene = default_scene(cfg, n)
  R, O = cfg.num_robots, cfg.num_obstacles
  base_pos, base_rot = scene.base_pos.clone(), scene.base_rot.clone()
  base_pos[..., 1] = -0.8 + 1.6 * _rand(gen, n, R)
  low = _rand(gen, n) < 0.25  # Past the table and low enough for the arm to reach the floor.
  base_pos[low, :, 0] *= 2.5
  base_pos[low, :, 2] = -0.9
  spun = _rand(gen, n) < 0.25
  base_rot[spun] = _random_rot(n, R, gen)[spun]
  spans = 0.02 * (3.0 / 0.02) ** _rand(gen, n, O, 3)
  pos = _rand(gen, n, O, 3) * torch.tensor([3.2, 3.2, 3.6], device=DEVICE)
  pos = pos - torch.tensor([1.6, 1.6, 1.0], device=DEVICE)
  parked = _rand(gen, n, O) > 0.1  # Parked boxes keep their huge sizes on purpose.
  pos[parked] = torch.tensor([0.0, 0.0, 1000.0], device=DEVICE)
  return scene._replace(base_pos=base_pos, base_rot=base_rot, box_pos=pos,
                        box_rot=_random_rot(n, O, gen), box_spans=spans)


def _pool_batch(num_robots, n, gen):
  pool = stored_scenes(num_robots)
  ids = torch.randint(len(pool.q0), (n,), generator=gen, device=DEVICE)
  return type(pool)(*[x[ids] for x in pool])


def _ids(n):
  return torch.arange(n, device=DEVICE)


def _flags(world, q):
  world.set_q(q)
  world.forward()
  return world.robots_in_collision()


def _geom_names(model):
  return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]


def _warp_contacts(world, names):
  """Per world: {sorted name pair: dist}."""
  n = world._n_contacts
  wid = world.sim.data.contact.worldid[:n].cpu().numpy()
  geom = world.sim.data.contact.geom[:n].cpu().numpy()
  dist = world.sim.data.contact.dist[:n].cpu().numpy()
  out = [{} for _ in range(world.num_worlds)]
  for w, (a, b), d in zip(wid, geom, dist):
    out[w][tuple(sorted((names[a], names[b])))] = float(d)
  return out


def _c_contacts(model, data, names):
  return {tuple(sorted((names[data.contact.geom[i][0]], names[data.contact.geom[i][1]]))):
          float(data.contact.dist[i]) for i in range(data.ncon)}


def _scene_row(scene, w):
  return [x[w].cpu().numpy() for x in (scene.base_pos, scene.base_rot, scene.box_pos,
                                       scene.box_rot, scene.box_spans)]


def _write_rows(scene, ids, new):
  """The ledger: what each world must hold, kept apart from the simulator."""
  return type(scene)(*[old.index_copy(0, ids, x) for old, x in zip(scene, new)])


def _fresh_c(num_robots, half):
  """C model compiled with the true box sizes, midphase on: no compile-time bound can be stale."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.95],
                          size=[0, 0, 0.05])
  spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -0.488],
                          size=[0.7841, 0.762, 0.488])
  for name in robot_names(num_robots):
    arm = mujoco.MjSpec.from_file(str(ASSETS / 'panda_spheres_stick.xml'))
    for geom in list(arm.geoms):
      if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        arm.delete(geom)
    for mesh in list(arm.meshes):
      arm.delete(mesh)
    spec.worldbody.add_body(name=name, mocap=True).add_frame().attach_body(
      arm.body('link0'), f'{name}/', '')
  for i, h in enumerate(half):
    spec.worldbody.add_body(name=f'box{i}', mocap=True).add_geom(
      name=f'box{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=h.tolist())
  return spec.compile()


def _fresh_c_flags(num_robots, row, q):
  base_pos, base_rot, box_pos, box_rot, spans = row
  model = _fresh_c(num_robots, spans.astype(np.float64) / 2)
  data = mujoco.MjData(model)
  quat = np.zeros(4)
  names = robot_names(num_robots)
  for i, name in enumerate(names):
    mujoco.mju_mat2Quat(quat, base_rot[i].astype(np.float64).ravel())
    mocap = model.body_mocapid[model.body(name).id]
    data.mocap_pos[mocap], data.mocap_quat[mocap] = base_pos[i], quat
    for j in range(7):
      data.qpos[model.jnt_qposadr[model.joint(f'{name}/joint{j + 1}').id]] = q[i, j]
  for i in range(len(box_pos)):
    mujoco.mju_mat2Quat(quat, box_rot[i].astype(np.float64).ravel())
    mocap = model.body_mocapid[model.body(f'box{i}').id]
    data.mocap_pos[mocap], data.mocap_quat[mocap] = box_pos[i], quat
  mujoco.mj_kinematics(model, data)
  mujoco.mj_collision(model, data)
  flags = np.zeros(num_robots, dtype=bool)
  for i in range(data.ncon):
    for g in data.contact.geom[i]:
      owner = model.body(model.body_rootid[model.geom_bodyid[g]]).name
      if owner in names:
        flags[names.index(owner)] = True
  return flags


def _kept_pairs(model):
  """(a, b) of geom pairs the rules of env_spec 5.2-5.3 keep, from weld ids and spheres.json."""
  body = model.geom_bodyid
  weld = model.body_weldid[body]
  parent = model.body_weldid[model.body_parentid[weld]]
  excluded = np.zeros((model.nbody, model.nbody), dtype=bool)
  labels = [model.body(b).name.split('/') for b in range(model.nbody)]
  pairs = {frozenset(e) for e in SPHERES['excludes']}
  for b1, l1 in enumerate(labels):
    for b2, l2 in enumerate(labels):
      excluded[b1, b2] = len(l1) == len(l2) == 2 and l1[0] == l2[0] and frozenset((l1[1], l2[1])) in pairs
  a, b = np.triu_indices(model.ngeom, 1)
  keep = weld[a] != weld[b]
  keep &= ~((weld[a] > 0) & (weld[b] > 0) & ((parent[a] == weld[b]) | (parent[b] == weld[a])))
  keep &= ~excluded[body[a], body[b]]
  return a[keep], b[keep]


def _pair_distances(world, a, b):
  """Signed distance [W, P] of each pair, from warp's geom poses only."""
  model = world.sim.mj_model
  types = model.geom_type
  sphere, box, plane = (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX,
                        mujoco.mjtGeom.mjGEOM_PLANE)
  swap = types[a] != sphere  # The sphere goes first.
  s, o = np.where(swap, b, a), np.where(swap, a, b)
  assert (types[s] == sphere).all()
  xpos, xmat = world.sim.data.geom_xpos[:], world.sim.data.geom_xmat[:]
  size = world.sim.model.geom_size[:]
  rel = xpos[:, s] - xpos[:, o]
  radius = size[:, s, 0]
  out = torch.zeros_like(radius)
  mask = torch.as_tensor(types[o] == sphere, device=DEVICE)
  out[:, mask] = rel[:, mask].norm(dim=-1) - radius[:, mask] - size[:, o[mask.cpu().numpy()], 0]
  mask = torch.as_tensor(types[o] == box, device=DEVICE)
  o_box = o[mask.cpu().numpy()]
  local = torch.einsum('wpi,wpij->wpj', rel[:, mask], xmat[:, o_box])
  over = local.abs() - size[:, o_box]
  out[:, mask] = (over.clamp(min=0).norm(dim=-1) + over.max(-1).values.clamp(max=0)
                  - radius[:, mask])
  mask = torch.as_tensor(types[o] == plane, device=DEVICE)
  o_plane = o[mask.cpu().numpy()]
  out[:, mask] = (rel[:, mask] * xmat[:, o_plane, :, 2]).sum(-1) - radius[:, mask]
  assert np.isin(types[o], [sphere, box, plane]).all()
  return out


def _assert_matches_oracle(world, a, b):
  """All worlds, all pairs: warp's contact list and flags against brute force. Returns stats."""
  dist = _pair_distances(world, a, b)
  ngeom, num_worlds = world.sim.mj_model.ngeom, world.num_worlds
  lookup = torch.full((ngeom, ngeom), -1, dtype=torch.long, device=DEVICE)
  index = torch.arange(len(a), device=DEVICE)
  lookup[a, b] = index
  lookup[b, a] = index
  n = world._n_contacts
  geom = world.sim.data.contact.geom[:n].long()
  wid = world.sim.data.contact.worldid[:n].long()
  pair = lookup[geom[:, 0], geom[:, 1]]
  assert (pair >= 0).all(), 'a pair env_spec 5.2-5.3 forbids was reported'
  found = torch.zeros(num_worlds, len(a), dtype=torch.bool, device=DEVICE)
  found[wid, pair] = True
  assert found.sum() == n, 'a sphere pair was reported twice'
  assert ((dist < -BAND) & ~found).sum() == 0, 'warp missed a penetrating pair'
  assert ((dist > BAND) & found).sum() == 0, 'warp invented a contact'
  reported = world.sim.data.contact.dist[:n]
  assert (reported - dist[wid, pair]).abs().max() < 1e-4 if n else True
  flags = world.robots_in_collision()
  robot = world._geom_robot
  ra, rb = robot[a], robot[b]
  for r in range(world.num_robots):
    mask = (ra == r) | (rb == r)
    assert flags[(dist[:, mask] < -BAND).any(1), r].all()
    assert (dist[:, mask] < BAND).any(1)[flags[:, r]].all()
  # A contact the stale compiled bound would have culled.
  boxes = torch.zeros(ngeom, dtype=torch.bool, device=DEVICE)
  boxes[world._box_geom] = True
  hit_box = boxes[geom].any(1)
  xpos = world.sim.data.geom_xpos[:]
  gap = (xpos[wid, geom[:, 0]] - xpos[wid, geom[:, 1]]).norm(dim=-1)
  far = int((hit_box & (gap > COMPILED_RBOUND + 0.11)).sum())
  return {'contacts': n, 'beyond_compiled_bound': far, 'flag_rate': flags.float().mean().item()}


# 1. Per-world state.


@pytest.fixture(scope='module', params=[4, 8], ids=['4r', '8r'])
def small(request):
  return build_world(WorldCfg(num_robots=request.param, nconmax=2048), 256, DEVICE)


def test_partial_writes_of_oversized_boxes_match_c_world_by_world(small):
  world = small
  cfg, R, W = world.cfg, world.num_robots, world.num_worlds
  gen = _gen(10)
  ledger = _hostile(cfg, W, gen)
  q = _uniform_q(W, R, gen)
  world.set_scene(_ids(W), ledger)
  world.set_q(q)
  c = CWorld(R, cfg.num_obstacles)
  names_w = _geom_names(world.sim.mj_model)
  names_c = _geom_names(c.model)
  assert set(names_w) <= set(names_c)  # CWorld keeps the stick's visual capsule.
  labels = pairs = beyond = 0
  for round_ in range(6):
    some = torch.randperm(W, generator=gen, device=DEVICE)[:W // 3]  # Unsorted on purpose.
    new = _hostile(cfg, len(some), gen)
    world.set_scene(some, new)
    ledger = _write_rows(ledger, some, new)
    others = torch.randperm(W, generator=gen, device=DEVICE)[:W // 3]
    q_new = _uniform_q(len(others), R, gen)
    world.set_q(q_new, others)
    q = q.index_copy(0, others, q_new)
    world.forward()
    flags = world.robots_in_collision().cpu().numpy()
    pos, rot = (x.cpu().numpy() for x in world.tcp())
    found = _warp_contacts(world, names_w)
    qn = q.cpu().numpy()
    for w in range(W):
      row = _scene_row(ledger, w)
      c.set_scene(*row)
      c.set_q(qn[w])
      c.forward()
      assert np.array_equal(c.flags(), flags[w]), f'round {round_} world {w}'
      ref_pos, ref_rot = c.tcp()
      assert np.abs(pos[w] - ref_pos).max() < 2e-4 and np.abs(rot[w] - ref_rot).max() < 2e-3
      ref = _c_contacts(c.model, c.data, names_c)
      assert set(ref) == set(found[w]), f'round {round_} world {w}'
      for key, d in ref.items():
        assert abs(d - found[w][key]) < 1e-4
      labels += R
      pairs += len(ref)
      half = row[4].max()
      beyond += half > 1
      if w % 16 == 0:  # The in-place C model might itself be stale: compile the truth too.
        assert np.array_equal(_fresh_c_flags(R, row, qn[w]), flags[w]), f'round {round_} world {w}'
  rate = flags.mean()
  print(f'{R}r: {labels} labels, {pairs} pairs identical to C, final flag rate {rate:.2f}')
  assert pairs > 1000 and 0.05 < rate < 0.95 and beyond > 0


@pytest.fixture(scope='module')
def oracle_world4():
  return build_world(WorldCfg(nconmax=2048), 1024, DEVICE)


@pytest.fixture(scope='module')
def oracle_world8():
  return build_world(WorldCfg(num_robots=8, nconmax=2048), 1024, DEVICE)


@pytest.mark.parametrize('num_robots', [4, 8])
def test_every_pair_of_every_world_against_brute_force(num_robots, oracle_world4, oracle_world8):
  world = oracle_world4 if num_robots == 4 else oracle_world8
  gen = _gen(11)
  W, R, cfg = world.num_worlds, world.num_robots, world.cfg
  a, b = _kept_pairs(world.sim.mj_model)
  world.set_scene(_ids(W), _hostile(cfg, W, gen))
  world.set_q(_uniform_q(W, R, gen))
  far = 0
  for _ in range(5):
    some = torch.randperm(W, generator=gen, device=DEVICE)[:W // 2]
    world.set_scene(some, _hostile(cfg, len(some), gen))
    world.set_q(_uniform_q(len(some), R, gen), some)
    world.forward()
    stats = _assert_matches_oracle(world, a, b)
    far += stats['beyond_compiled_bound']
    assert 0.05 < stats['flag_rate'] < 0.95
  print(f'{R}r oracle: {len(a)} pairs x {W} worlds, last {stats}, {far} beyond the compiled bound')
  assert far > 1000


def test_brute_force_holds_at_a_large_batch():
  cfg = WorldCfg(num_robots=8, nconmax=2048)
  W = 4096
  world = build_world(cfg, W, DEVICE)
  gen = _gen(26)
  a, b = _kept_pairs(world.sim.mj_model)
  world.set_scene(_ids(W), _hostile(cfg, W, gen))
  world.set_q(_uniform_q(W, 8, gen))
  world.forward()
  assert _assert_matches_oracle(world, a, b)['beyond_compiled_bound'] > 1000


def test_mesh_arm_with_oversized_boxes_matches_c():
  cfg = WorldCfg(arm_model='mesh_tool', nconmax=2048)
  W = 128
  world = build_world(cfg, W, DEVICE)
  c = CWorld(4, 30, 'mesh_tool')
  gen = _gen(27)
  labels = positives = disagree = 0
  for _ in range(4):
    scene, q = _hostile(cfg, W, gen), _uniform_q(W, 4, gen)
    world.set_scene(_ids(W), scene)
    flags = _flags(world, q).cpu().numpy()
    qn = q.cpu().numpy()
    for w in range(W):
      c.set_scene(*_scene_row(scene, w))
      c.set_q(qn[w])
      c.forward()
      ref = c.flags()
      labels += 4
      positives += int(ref.sum())
      disagree += int((ref != flags[w]).sum())
  print(f'mesh arm, oversized boxes: {disagree} of {labels} labels differ, {positives} positive')
  assert positives > 500 and disagree / labels <= 1e-3


def test_set_scene_touches_only_box_rows(oracle_world4):
  world = oracle_world4
  W = world.num_worlds
  m = world.sim.mj_model
  gen = _gen(12)
  world.set_scene(_ids(W), _hostile(world.cfg, W, gen))
  others = torch.ones(m.ngeom, dtype=torch.bool, device=DEVICE)
  others[world._box_geom] = False
  size = torch.as_tensor(m.geom_size, dtype=torch.float32, device=DEVICE)
  rbound = torch.as_tensor(m.geom_rbound, dtype=torch.float32, device=DEVICE)
  aabb = torch.as_tensor(m.geom_aabb, dtype=torch.float32, device=DEVICE).view(m.ngeom, 2, 3)
  assert torch.equal(world.sim.model.geom_size[:, others], size[others].expand(W, -1, -1))
  assert torch.equal(world.sim.model.geom_rbound[:, others], rbound[others].expand(W, -1))
  assert torch.equal(world.sim.model.geom_aabb[:, others], aabb[others].expand(W, -1, -1, -1))
  assert (world.sim.model.geom_aabb[:, world._box_geom, 0] == 0).all()  # Centres stay at zero.
  norm = world.sim.data.mocap_quat[:].norm(dim=-1)
  assert (norm - 1).abs().max() < 1e-6


def test_one_world_may_far_exceed_nconmax_when_the_pool_has_room():
  cfg = WorldCfg(nconmax=128, num_obstacles=2)
  world = build_world(cfg, 64, DEVICE)
  scene = default_scene(cfg, 64)
  pos, spans = scene.box_pos.clone(), scene.box_spans.clone()
  pos[7, 0] = torch.tensor([0.0, 0.0, 0.5], device=DEVICE)
  spans[7, 0] = 3.0  # Swallows every robot of world 7 and nothing else.
  world.set_scene(_ids(64), scene._replace(box_pos=pos, box_spans=spans))
  world.set_q(HOME.repeat(64, 4, 1))
  world.forward()
  per_world = torch.bincount(world.sim.data.contact.worldid[:world._n_contacts].long(), minlength=64)
  assert per_world[7] > cfg.nconmax and per_world.sum() == per_world[7]
  flags = world.robots_in_collision()
  assert flags[7].all() and flags.sum() == 4
  _assert_matches_oracle(world, *_kept_pairs(world.sim.mj_model))


def test_a_shrunken_box_leaves_no_trace():
  cfg = WorldCfg(num_obstacles=3)
  world = build_world(cfg, 8, DEVICE)
  scene = default_scene(cfg, 8)
  world.set_scene(_ids(8), scene)
  world.set_q(HOME.repeat(8, 4, 1))
  world.forward()
  baseline = int(world.sim.data.ncollision[0])
  huge = scene._replace(box_pos=torch.zeros_like(scene.box_pos),
                        box_spans=torch.full_like(scene.box_spans, 3.0))
  world.set_scene(_ids(8), huge)
  world.forward()
  assert world.robots_in_collision().all() and int(world.sim.data.ncollision[0]) > baseline
  # Grown then parked, in world 3 only: its candidates go, the rest keep theirs.
  three = torch.tensor([3], device=DEVICE)
  world.set_scene(three, type(scene)(*[x[:1] for x in scene]))
  world.forward()
  flags = world.robots_in_collision()
  assert not flags[3].any() and flags[[0, 1, 2, 4, 5, 6, 7]].all()
  world.set_scene(_ids(8), scene)
  world.forward()
  assert int(world.sim.data.ncollision[0]) == baseline and world._n_contacts == 0


def test_a_single_world_detects_collisions_like_c():
  # expand_model_fields returns early at nworld 1 (randomization.py:25): the flag clear must
  # still reach the graph, or nacon stays 0 for good.
  cfg = WorldCfg(nconmax=4096)
  world = build_world(cfg, 1, DEVICE)
  c = CWorld(4, 30)
  gen = _gen(13)
  hits = 0
  for _ in range(32):
    scene = _hostile(cfg, 1, gen)
    q = _uniform_q(1, 4, gen)
    world.set_scene(_ids(1), scene)
    flags = _flags(world, q).cpu().numpy()[0]
    c.set_scene(*_scene_row(scene, 0))
    c.set_q(q[0].cpu().numpy())
    c.forward()
    assert np.array_equal(c.flags(), flags)
    hits += flags.sum()
  assert hits > 8


def test_interleaved_worlds_and_a_dead_world_do_not_disturb_each_other():
  # Training and evaluation hold separate Worlds, DESIGN section 7.
  gen = _gen(14)
  cfg_a, cfg_b = WorldCfg(nconmax=2048), WorldCfg(num_robots=8, num_obstacles=5, nconmax=2048)
  a, b = build_world(cfg_a, 64, DEVICE), build_world(cfg_b, 32, DEVICE)
  scene_a, scene_b = _hostile(cfg_a, 64, gen), _hostile(cfg_b, 32, gen)
  q_a, q_b = _uniform_q(64, 4, gen), _uniform_q(32, 8, gen)
  a.set_scene(_ids(64), scene_a)
  b.set_scene(_ids(32), scene_b)
  ref_a, ref_b = _flags(a, q_a).clone(), _flags(b, q_b).clone()
  pos_a, pos_b = a.tcp()[0], b.tcp()[0]
  assert ref_a.any() and ref_b.any()
  for _ in range(3):
    dead = build_world(cfg_a, 64, DEVICE)
    dead.set_scene(_ids(64), _hostile(cfg_a, 64, gen))
    _flags(dead, _uniform_q(64, 4, gen))
    assert torch.equal(_flags(a, q_a), ref_a)
    assert torch.equal(_flags(b, q_b), ref_b)
    del dead
    gc.collect()
    assert torch.equal(_flags(b, q_b), ref_b) and torch.equal(b.tcp()[0], pos_b)
    assert torch.equal(_flags(a, q_a), ref_a) and torch.equal(a.tcp()[0], pos_a)
    # Memory the dead world freed goes to a fresh allocation, and must not be a live array.
    junk = torch.full((1 << 22,), float('nan'), device=DEVICE)
    assert torch.equal(_flags(a, q_a), ref_a)
    del junk


# 2. Stale arrays, mjlab_api 4.5, 4.6, 5.2, 9.


def test_expanded_fields_are_really_per_world_and_views_are_live(oracle_world4):
  world = oracle_world4
  W = world.num_worlds
  for field in ('geom_size', 'geom_rbound', 'geom_aabb'):
    array = getattr(world.sim.wp_model, field)
    assert array.shape[0] == W  # The non-batched no-op of bug 2 would leave 1.
    assert getattr(world.sim.model, field)._tensor.data_ptr() == array.ptr
  data = world.sim.wp_data
  assert world._nacon._tensor.data_ptr() == data.nacon.ptr
  assert world._ncollision._tensor.data_ptr() == data.ncollision.ptr
  assert world._contact.dist._tensor.data_ptr() == data.contact.dist.ptr
  assert world._contact.geom._tensor.data_ptr() == data.contact.geom.ptr
  assert world._contact.worldid._tensor.data_ptr() == data.contact.worldid.ptr
  assert world.sim.data.qpos._tensor.data_ptr() == data.qpos.ptr
  assert world.sim.data.mocap_pos._tensor.data_ptr() == data.mocap_pos.ptr
  assert world.sim.wp_model.opt.disableflags == 0 and world.sim.mj_model.opt.disableflags == 0
  assert world.sim.kc_graph is not None
  # Contacts stay signed distances: nothing widened the window that `dist <= 0` filters.
  assert world.sim.wp_model.geom_margin.numpy().max() == 0
  assert world.sim.wp_model.geom_gap.numpy().max() == 0


def test_writes_reach_warp_memory_for_one_world_only(oracle_world4):
  world = oracle_world4
  W = world.num_worlds
  g = int(world._box_geom[0])
  scene = default_scene(world.cfg, W)
  world.set_scene(_ids(W), scene)
  one = torch.tensor([5], device=DEVICE)
  spans = scene.box_spans[:1].clone()
  spans[0, 0] = torch.tensor([0.4, 0.6, 0.8], device=DEVICE)
  world.set_scene(one, scene._replace(**{k: v[:1] for k, v in scene._asdict().items()})._replace(box_spans=spans))
  wp.synchronize()
  size = world.sim.wp_model.geom_size.numpy()
  assert np.allclose(size[5, g], [0.2, 0.3, 0.4])
  assert np.allclose(np.delete(size, 5, axis=0)[:, g], 0.025)


def test_no_growth_over_many_forwards():
  # A captured graph with allocations inside must not take more memory per launch.
  cfg = WorldCfg(num_obstacles=4)
  world = build_world(cfg, 64, DEVICE)
  gen = _gen(28)
  world.set_scene(_ids(64), _hostile(cfg, 64, gen))
  q = _uniform_q(64, 4, gen)
  device = wp.get_device(world.sim.wp_device)
  _flags(world, q)
  torch.cuda.synchronize()
  before = wp.get_mempool_used_mem_current(device), torch.cuda.memory_allocated()
  for _ in range(2000):
    world.set_q(q)
    world.forward()
    world.robots_in_collision()
  torch.cuda.synchronize()
  gc.collect()
  assert (wp.get_mempool_used_mem_current(device), torch.cuda.memory_allocated()) == before


def test_nan_joint_reads_as_free_and_shows_only_in_the_tcp():
  # mjlab_api 9: the kinematic path has no NanGuard. A pin of the behaviour, so that a caller
  # knows the collision flags cannot be the thing that catches a NaN policy output.
  cfg = WorldCfg(num_obstacles=1)
  world = build_world(cfg, 2, DEVICE)
  scene = default_scene(cfg, 2)
  world.set_scene(_ids(2), scene)
  q = HOME.repeat(2, 4, 1)
  _flags(world, q)
  at = world.sim.data.xpos[:, world.sim.mj_model.body('panda1/link4').id].clone()
  world.set_scene(_ids(2), scene._replace(box_pos=at[:, None]))  # Robot 0 is inside a box.
  q[0, 0, 0] = float('nan')  # In world 0 its link1 and everything after it are NaN.
  flags = _flags(world, q)
  assert flags[1, 0] and not flags[0, 0]
  assert world.tcp()[0][0, 0].isnan().all() and not world.tcp()[0][1, 0].isnan().any()


# 3. Aliasing.


@pytest.mark.parametrize('num_robots', [1, 4])
def test_returned_values_never_alias_the_simulator(num_robots):
  cfg = WorldCfg(num_robots=num_robots, num_obstacles=2)
  world = build_world(cfg, 16, DEVICE)
  gen = _gen(15)
  scene = _hostile(cfg, 16, gen)
  world.set_scene(_ids(16), scene)
  q1, q2 = _uniform_q(16, num_robots, gen), _uniform_q(16, num_robots, gen)
  flags1 = _flags(world, q1)
  pos1, rot1 = world.tcp()
  held = flags1.clone(), pos1.clone(), rot1.clone()
  # Scribbling on what was returned must not reach warp.
  live_pos = world.sim.data.site_xpos[:].clone()
  pos1 += 1
  rot1 += 1
  flags1 ^= True
  assert torch.equal(world.sim.data.site_xpos[:], live_pos)
  pos1 -= 1
  rot1 -= 1
  flags1 ^= True
  assert torch.allclose(pos1, held[1]) and torch.allclose(rot1, held[2])
  # Nor may a new forward reach what was returned.
  flags2 = _flags(world, q2)
  assert torch.equal(flags1, held[0])
  assert torch.allclose(pos1, held[1]) and torch.allclose(rot1, held[2])
  assert not torch.equal(world.tcp()[0], pos1)
  assert flags2.data_ptr() != flags1.data_ptr()


def test_inputs_are_copied_at_the_call():
  cfg = WorldCfg(num_obstacles=4)
  world = build_world(cfg, 8, DEVICE)
  gen = _gen(16)
  scene = _hostile(cfg, 8, gen)
  q = _uniform_q(8, 4, gen)
  world.set_scene(_ids(8), scene)
  world.set_q(q)
  size = world.sim.model.geom_size[:, world._box_geom].clone()
  held_q = world.sim.data.qpos[:].clone()
  mocap = world.sim.data.mocap_pos[:].clone()
  for x in scene:
    x += 1
  q += 1
  assert torch.equal(world.sim.model.geom_size[:, world._box_geom], size)
  assert torch.equal(world.sim.data.qpos[:], held_q)
  assert torch.equal(world.sim.data.mocap_pos[:], mocap)


# 4. Contact flags.


def test_exact_zero_distance_is_a_contact_in_c_only():
  # Pins mjlab_api 4.3. DESIGN's `dist <= 0` is `dist < 0` here: warp never stores a zero
  # (collision_core.py:254). test_band_of_the_strict_test bounds what that costs.
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom type="plane" size="0 0 1"/>
    <body pos="0 0 0.5"><joint type="slide" axis="0 0 1"/><geom type="sphere" size="0.5"/></body>
    </worldbody></mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_collision(model, data)
  assert data.ncon == 1 and data.contact.dist[0] == 0
  with wp.ScopedDevice(DEVICE):
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(model, data, nconmax=8, njmax=8)
    mjw.kinematics(warp_model, warp_data)
    mjw.collision(warp_model, warp_data)
    assert warp_data.nacon.numpy()[0] == 0


def test_band_of_the_strict_test():
  cfg = WorldCfg(num_obstacles=1)
  W = 512
  world = build_world(cfg, W, DEVICE)
  scene = default_scene(cfg, W)
  world.set_scene(_ids(W), scene)
  world.set_q(HOME.repeat(W, 4, 1))
  world.forward()
  m = world.sim.mj_model
  g = m.geom('panda1/link3_s0').id
  center = world.sim.data.geom_xpos[0, g].double()
  radius = float(m.geom_size[g, 0])
  half = 0.1
  offset = torch.linspace(-4e-6, 4e-6, W, device=DEVICE, dtype=torch.float64)
  pos = center.expand(W, -1).clone()
  pos[:, 2] += radius + half + offset  # Box above the sphere, face at sphere top + offset.
  spans = torch.full((W, 1, 3), 2 * half, device=DEVICE)
  world.set_scene(_ids(W), scene._replace(box_pos=pos[:, None].float(), box_spans=spans))
  world.forward()
  n = world._n_contacts
  geom = world.sim.data.contact.geom[:n]
  box = int(world._box_geom[0])
  sel = ((geom[:, 0] == g) & (geom[:, 1] == box)) | ((geom[:, 1] == g) & (geom[:, 0] == box))
  dist = world.sim.data.contact.dist[:n]
  assert (dist < 0).all()  # Nothing else is ever stored, so `<= 0` cannot see a zero.
  hit = torch.zeros(W, dtype=torch.bool, device=DEVICE)
  hit[world.sim.data.contact.worldid[:n][sel].long()] = True
  assert hit[offset < -1e-6].all() and not hit[offset > 1e-6].any()
  assert hit[offset < 0].any() and (~hit[offset > 0]).any()


def _isolated_arm_scene(cfg, n, world):
  """Bases spread far apart, so only what a test places can touch."""
  scene = default_scene(cfg, n)
  pos = scene.base_pos.clone()
  pos[:, :, 0] = 10.0 * torch.arange(cfg.num_robots, device=DEVICE)
  pos[:, :, 2] = 5.0
  return scene._replace(base_pos=pos)


def test_link0_against_stick_flags_both_robots_and_link0_against_link0_flags_neither():
  # env_spec 5.3: weld 0 on one side counts, weld 0 on both sides does not.
  cfg = WorldCfg(num_robots=2, num_obstacles=1)
  world = build_world(cfg, 2, DEVICE)
  m = world.sim.mj_model
  names = _geom_names(m)
  scene = _isolated_arm_scene(cfg, 2, world)
  world.set_scene(_ids(2), scene)
  world.set_q(HOME.repeat(2, 2, 1))
  world.forward()
  assert world._n_contacts == 0
  link0 = world.sim.data.geom_xpos[0, m.geom('panda1/link0_s2').id]
  tip = world.sim.data.geom_xpos[0, m.geom('panda2/stick_s5').id]
  pos = scene.base_pos.clone()
  pos[1, 1] += link0 - tip  # World 1: robot 2's stick tip into robot 1's link0.
  world.set_scene(_ids(2), scene._replace(base_pos=pos))
  world.forward()
  flags = world.robots_in_collision()
  assert flags.tolist() == [[False, False], [True, True]]
  contacts = _warp_contacts(world, names)[1]
  assert ('panda1/link0_s2', 'panda2/stick_s5') in contacts
  assert not any('link0' in x and 'link0' in y for x, y in contacts)

  # Back to back: the rear link0 spheres overlap by 4 cm, nothing else touches.
  bases = scene.base_pos.clone()
  rot = scene.base_rot.clone()
  bases[:, 0] = torch.tensor([0.0, 0.0, 5.0], device=DEVICE)
  bases[:, 1] = torch.tensor([-0.3, 0.0, 5.0], device=DEVICE)
  rot[:, 0] = torch.eye(3, device=DEVICE)
  rot[:, 1] = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=DEVICE))
  world.set_scene(_ids(2), scene._replace(base_pos=bases, base_rot=rot))
  # Leaning forward, because home leans the upper arm back into the other robot.
  world.set_q(torch.tensor([0.0, 0.6, 0.0, -1.2, 0.0, 1.4, 0.0], device=DEVICE).repeat(2, 2, 1))
  world.forward()
  g1, g2 = m.geom('panda1/link0_s1').id, m.geom('panda2/link0_s1').id
  xpos = world.sim.data.geom_xpos[0]
  assert (xpos[g1] - xpos[g2]).norm() < m.geom_size[g1, 0] + m.geom_size[g2, 0] - 0.02
  assert world._n_contacts == 0 and not world.robots_in_collision().any()


@pytest.mark.parametrize('num_robots', [4, 8])
def test_static_overlaps_never_count(num_robots, oracle_world4, oracle_world8):
  # env_spec 5.3: box-box, box-table, box-floor, link0-box, link0-table, link0-floor all at once,
  # in every world, on top of the permanent table-floor overlap of the compiled model.
  world = oracle_world4 if num_robots == 4 else oracle_world8
  W, R = world.num_worlds, world.num_robots
  scene = default_scene(world.cfg, W)
  pos, spans = scene.box_pos.clone(), scene.box_spans.clone()
  base = scene.base_pos.clone()
  floor = base[:, :, 2] < 1.0
  base[floor] = base[floor] + torch.tensor([0.0, 0.0, -0.02], device=DEVICE)  # link0 in the table.
  link0_box = torch.tensor([0.03, 0.0, 0.03], device=DEVICE)
  pos[:, :R] = base + link0_box  # One small box inside each link0, under link1.
  spans[:, :R] = 0.04
  pos[:, R], spans[:, R] = torch.tensor([0.0, 0.0, -0.1], device=DEVICE), 0.3  # In the table.
  pos[:, R + 1], spans[:, R + 1] = torch.tensor([0.0, 0.0, -0.95], device=DEVICE), 0.2  # Floor.
  pos[:, R + 2:R + 4], spans[:, R + 2:R + 4] = torch.tensor([0.0, 0.0, 0.9], device=DEVICE), 0.1
  world.set_scene(_ids(W), scene._replace(base_pos=base, box_pos=pos, box_spans=spans))
  world.set_q(HOME.repeat(W, R, 1))
  world.forward()
  assert world._n_contacts == 0 and not world.robots_in_collision().any()
  c = CWorld(R, world.num_obstacles)
  c.set_scene(*_scene_row(scene._replace(base_pos=base, box_pos=pos, box_spans=spans), 0))
  c.set_q(HOME.repeat(R, 1).cpu().numpy())
  c.forward()
  assert c.data.ncon == 0


def test_stick_and_own_links_follow_the_exclude_list_exactly():
  # One arm, no boxes: warp's contact list against brute force on every kept pair, and every
  # filtered sphere pair really overlaps somewhere, so its silence means something.
  cfg = WorldCfg(num_robots=1, num_obstacles=1)
  W = 4096
  world = build_world(cfg, W, DEVICE)
  m = world.sim.mj_model
  scene = _isolated_arm_scene(cfg, W, world)
  world.set_scene(_ids(W), scene)
  gen = _gen(17)
  _flags(world, _uniform_q(W, 1, gen))
  a, b = _kept_pairs(m)
  stats = _assert_matches_oracle(world, a, b)
  assert stats['contacts'] > 100
  label = [m.body(body).name.split('/')[-1] for body in m.geom_bodyid]
  n = world._n_contacts
  geom = world.sim.data.contact.geom[:n].cpu().numpy()
  seen = {frozenset((label[x], label[y])) for x, y in geom}
  # Live stick pairs of sphere_model 4 fire, the excluded ones never do.
  for link in ('link0', 'link1', 'link2'):
    assert frozenset((link, 'stick')) in seen
  for link in ('link3', 'link4', 'link5', 'link6', 'link7', 'attachment'):
    assert frozenset((link, 'stick')) not in seen
  for pair in SPHERES['excludes']:
    assert frozenset(pair) not in seen
  # The required excludes hide real sphere overlap, in a sizeable share of these states.
  spheres = np.array([g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])
  xpos = world.sim.data.geom_xpos[:]
  size = world.sim.model.geom_size[:, :, 0]
  for one, two in (('link0', 'link1'), ('link5', 'link7')):
    g1 = spheres[[label[g] == one for g in spheres]]
    g2 = spheres[[label[g] == two for g in spheres]]
    gap = (xpos[:, g1, None] - xpos[:, None, g2]).norm(dim=-1)
    overlap = (gap < size[:, g1, None] + size[:, None, g2]).any(-1).any(-1)
    assert overlap.float().mean() > 0.05, (one, two)


@pytest.mark.parametrize('num_robots', [4, 8])
def test_near_contact_states_match_c(num_robots):
  # Bisect to the boundary of the sphere world itself. C and warp may only differ inside BAND.
  R = num_robots
  cfg = WorldCfg(num_robots=R, nconmax=1024)
  W = 256
  world = build_world(cfg, W, DEVICE)
  c = CWorld(R, 30)
  names, c_names = _geom_names(world.sim.mj_model), _geom_names(c.model)
  gen = _gen(18)
  labels = disagree = 0
  widest = 0.0
  for _ in range(4):
    scene = _pool_batch(R, W, gen)
    world.set_scene(_ids(W), scene)
    q0 = scene.q0
    q1 = _uniform_q(W, R, gen)
    lo, hi = torch.zeros(W, 1, 1, device=DEVICE), torch.ones(W, 1, 1, device=DEVICE)
    for _ in range(22):
      mid = (lo + hi) / 2
      f = _flags(world, q0 + (q1 - q0) * mid).any(1)
      hi[f] = mid[f]
      lo[~f] = mid[~f]
    for t in (lo, hi):
      q = q0 + (q1 - q0) * t
      flags = _flags(world, q).cpu().numpy()
      found = _warp_contacts(world, names)
      qn = q.cpu().numpy()
      for w in range(W):
        c.set_scene(*_scene_row(scene, w))
        c.set_q(qn[w])
        c.forward()
        labels += 1
        ref = _c_contacts(c.model, c.data, c_names)
        disagree += not np.array_equal(c.flags(), flags[w])
        for key in set(ref) ^ set(found[w]):
          widest = max(widest, abs({**ref, **found[w]}[key]))
  # Every state sits on the boundary by construction, so the rate says little. The width does.
  print(f'near contact: {disagree} of {labels} worlds differ, all within {widest:.1e} m')
  assert labels == 2 * 4 * W and widest < BAND


# 5. Overflow.


def _floor_scene(cfg, n):
  """Arms far apart and buried under the floor: every link-floor pair has several contacts."""
  scene = default_scene(cfg, n)
  base = scene.base_pos.clone()
  base[..., :2] *= 6
  base[..., 2] = -1.6
  return scene._replace(base_pos=base)


def _floor_world(nconmax):
  cfg = WorldCfg(arm_model='mesh_tool', nconmax=nconmax, num_obstacles=1)
  world = build_world(cfg, 4, DEVICE)
  world.set_scene(_ids(4), _floor_scene(cfg, 4))
  world.set_q(HOME.repeat(4, 4, 1))
  return world


def test_real_multi_contact_overflow_is_caught_by_nacon_alone():
  roomy = _floor_world(512)
  roomy.forward()
  nacon, ncollision = roomy._n_contacts, int(roomy._ncollision[0])
  assert roomy.robots_in_collision().all()
  assert nacon > ncollision  # Several contacts per pair, for real and not monkeypatched.
  # An exact fit keeps everything.
  fit = nacon // 4 + (nacon % 4 > 0)
  exact = _floor_world(fit)
  exact.forward()
  assert exact._n_contacts == nacon and exact.robots_in_collision().all()
  # One world's worth short: candidates fit, contacts do not.
  short = _floor_world(fit - 1)
  assert ncollision <= short._naconmax < nacon
  with pytest.raises(ContactOverflow):
    short.forward()
  assert int(short._ncollision[0]) <= short._naconmax < int(short._nacon[0])


def test_candidates_alone_overflow_with_no_contact_at_all():
  cfg = WorldCfg(nconmax=2, num_obstacles=1)
  world = build_world(cfg, 8, DEVICE)
  world.set_scene(_ids(8), default_scene(cfg, 8))
  world.set_q(HOME.repeat(8, 4, 1))
  with pytest.raises(ContactOverflow):
    world.forward()
  assert int(world._nacon[0]) == 0 and int(world._ncollision[0]) > world._naconmax


@pytest.mark.parametrize('arm_model', ['spheres_stick', 'mesh_tool'])
def test_pool_boundary_is_exact_on_candidates(arm_model):
  # One world, so the pool can equal the candidate count.
  gen = _gen(19)
  cfg = WorldCfg(arm_model=arm_model, nconmax=4096)
  roomy = build_world(cfg, 1, DEVICE)
  scene = _hostile(cfg, 1, gen)
  q = _uniform_q(1, 4, gen)
  roomy.set_scene(_ids(1), scene)
  ref = _flags(roomy, q)
  count = int(roomy._ncollision[0])
  assert ref.any() and count > roomy._n_contacts
  exact = build_world(WorldCfg(arm_model=arm_model, nconmax=count), 1, DEVICE)
  exact.set_scene(_ids(1), scene)
  assert torch.equal(_flags(exact, q), ref) and int(exact._ncollision[0]) == exact._naconmax
  short = build_world(WorldCfg(arm_model=arm_model, nconmax=count - 1), 1, DEVICE)
  short.set_scene(_ids(1), scene)
  short.set_q(q)
  with pytest.raises(ContactOverflow):
    short.forward()


MESH_OVERFLOW = """
import torch
from roboballet_mjlab.config import HOME_QPOS, WorldCfg
from roboballet_mjlab.sim.world import ContactOverflow, build_world
from tests.test_sim_adversarial import _floor_scene
from tests.test_sim_common import DEVICE, default_scene
home = torch.as_tensor(HOME_QPOS, device=DEVICE)
for robots, boxes, nconmax, worlds, scene_fn in (
    (4, 1, 2, 4, default_scene), (4, 1, 32, 4, default_scene), (1, 1, 1, 1, default_scene),
    (8, 1, 2, 4, default_scene), (8, 30, 1, 1, default_scene), (4, 1, 1, 4, _floor_scene)):
  cfg = WorldCfg(num_robots=robots, arm_model='mesh_tool', nconmax=nconmax, num_obstacles=boxes)
  world = build_world(cfg, worlds, DEVICE)
  world.set_scene(torch.arange(worlds, device=DEVICE), scene_fn(cfg, worlds))
  world.set_q(home.repeat(worlds, robots, 1))
  try:
    world.forward()
    print('NO GUARD', robots, nconmax, worlds, int(world._ncollision[0]), flush=True)
  except ContactOverflow:
    print('GUARD FIRED', flush=True)
"""


def test_mesh_overflow_raises_contact_overflow():
  # In a subprocess: a regression takes the CUDA context with it. collision_convex.py:1040 walks
  # ncollision pairs through an array of naconmax, so far above the pool the launch used to die
  # with CUDA 700 before the guard in World.forward could read a counter. One case has no
  # candidate far from a real contact: arms buried in the floor.
  run = subprocess.run([sys.executable, '-c', MESH_OVERFLOW], capture_output=True, text=True,
                       timeout=600, cwd=REPO, env=os.environ)
  assert run.stdout.count('GUARD FIRED') == 6, run.stdout[-400:] + run.stderr[-400:]


@pytest.mark.parametrize('count,clamped', [(0, 0), (7, 7), (8, 8), (9, 8), (4096, 8)])
def test_clamp_keeps_every_candidate_that_fits_and_restores_the_count(count, clamped):
  # Direct: an off-by-one drops one candidate, which the world tests see only if it is a contact.
  ncollision = wp.array([count], dtype=int, device=DEVICE)
  seen = wp.zeros(1, dtype=int, device=DEVICE)
  wp.launch(sim_module._clamp_candidates, dim=1, inputs=[8, ncollision, seen], device=DEVICE)
  assert ncollision.numpy()[0] == clamped and seen.numpy()[0] == count
  wp.launch(sim_module._restore_candidates, dim=1, inputs=[seen, ncollision], device=DEVICE)
  assert ncollision.numpy()[0] == count


def _sorted_contacts(world):
  """Rows of (world, geom, geom, dist) from the data, in a thread-independent order."""
  n = int(world._nacon[0])
  c = world._contact
  rows = torch.cat([c.worldid[:n, None].double(), c.geom[:n].double(), c.dist[:n, None].double()], 1)
  return rows[np.lexsort(rows.cpu().numpy().T[::-1]).copy()]


def test_clamped_collision_matches_stock_below_the_pool():
  gen = _gen(23)
  cfg = WorldCfg(arm_model='mesh_tool', nconmax=1024)
  n = 64
  world = build_world(cfg, n, DEVICE)
  world.set_scene(_ids(n), _hostile(cfg, n, gen))
  total = 0
  for _ in range(5):
    flags = _flags(world, _uniform_q(n, 4, gen))
    candidates, contacts = int(world._ncollision[0]), _sorted_contacts(world)
    total += len(contacts)
    assert 0 < candidates <= world._naconmax  # Restored, and not silently the clamp.
    with wp.ScopedDevice(world.sim.wp_device):
      mjw.collision(world.sim.wp_model, world.sim.wp_data)
    assert int(world._ncollision[0]) == candidates
    assert torch.equal(_sorted_contacts(world), contacts)
    world._n_contacts = int(world._nacon[0])
    assert torch.equal(world.robots_in_collision(), flags)
  assert total > 0


@pytest.mark.parametrize('cfg,convex', [
  (WorldCfg(), False),
  (WorldCfg(visual_meshes=True), False),
  (WorldCfg(num_robots=8), False),
  (WorldCfg(arm_model='mesh_tool'), True),
])
def test_only_convex_models_leave_the_stock_pipeline(cfg, convex, monkeypatch):
  # DESIGN D9: stock is the ground truth. mesh_tool must not affect the default path.
  calls = []
  monkeypatch.setattr(sim_module, '_clamped_collision', lambda m, d: calls.append(1))
  world = build_world(cfg, 2, DEVICE)
  assert sim_module._has_convex_pairs(world.sim.wp_model) == convex
  assert bool(calls) == convex


# 6. Spheres against the mesh arm with the same stick.


def _mesh_stick_spec(original, cfg):
  spec = original(cfg)
  for body in spec.bodies:
    if body.name.startswith('tool/'):
      spec.delete(body)
      break
  stick = spec.body('attachment').add_body(name='stick')
  for i, z in enumerate(SPHERES['stick']['z']):
    stick.add_geom(name=f'stick_s{i}', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[SPHERES['stick']['R'], 0, 0], pos=[0, 0, z], contype=1, conaffinity=1)
  stick.add_site(name='tool/tool_tip_site', pos=[0, 0, SPHERES['stick']['L']])
  return spec


def _mesh_stick_world(num_robots, n, nconmax=1024, num_obstacles=30):
  """sphere_model's reference: mesh links and table, the very same stick spheres."""
  cfg = WorldCfg(num_robots=num_robots, arm_model='mesh_tool', nconmax=nconmax,
                 num_obstacles=num_obstacles)
  original = world_module._arm_spec
  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(world_module, '_arm_spec', lambda c: _mesh_stick_spec(original, c))
    return build_world(cfg, n, DEVICE)


def _deepest(world):
  """Deepest contact per robot [W, R], 0 where a robot has none."""
  n, R = world._n_contacts, world.num_robots
  wid = world._contact.worldid[:n].long()
  robots = world._geom_robot[world._contact.geom[:n].long()]
  dist = world._contact.dist[:n]
  out = torch.zeros(world.num_worlds * (R + 1), device=DEVICE)
  index = wid[:, None] * (R + 1) + robots
  out.scatter_reduce_(0, index[:, 0], dist, 'amin')
  out.scatter_reduce_(0, index[:, 1], dist, 'amin')
  return out.view(-1, R + 1)[:, :R]


def _copy_to_c(world, model, data, w):
  model.geom_size[:] = world.sim.model.geom_size[w].cpu().numpy()
  model.geom_rbound[:] = world.sim.model.geom_rbound[w].cpu().numpy()
  model.geom_aabb[:] = world.sim.model.geom_aabb[w].flatten(0).cpu().numpy().reshape(
    model.geom_aabb.shape)
  data.qpos[:] = world.sim.data.qpos[w].cpu().numpy()
  data.mocap_pos[:] = world.sim.data.mocap_pos[w].cpu().numpy()
  data.mocap_quat[:] = world.sim.data.mocap_quat[w].cpu().numpy()
  mujoco.mj_kinematics(model, data)
  mujoco.mj_collision(model, data)


def _c_mesh_flags(model, data, geom_robot, num_robots):
  """C truth for the mesh world loaded by _copy_to_c, with the deepest contact per robot."""
  flags, depth = np.zeros(num_robots, dtype=bool), np.zeros(num_robots)
  for i in range(data.ncon):
    if data.contact.dist[i] <= 0:
      for g in data.contact.geom[i]:
        r = geom_robot[g]
        if r < num_robots:
          flags[r] = True
          depth[r] = min(depth[r], data.contact.dist[i])
  return flags, depth


@pytest.mark.parametrize('num_robots,boxes', [(4, True), (4, False), (8, True), (8, False)])
def test_no_false_negative_at_random_or_near_contact(num_robots, boxes):
  R, W, batches = num_robots, 2048, 100
  mesh = _mesh_stick_world(R, W)
  spheres = build_world(WorldCfg(num_robots=R, nconmax=1024), W, DEVICE)
  m = mesh.sim.mj_model
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
  c_data = mujoco.MjData(m)
  geom_robot = mesh._geom_robot.cpu().numpy()
  gen = _gen(20 + R + boxes)
  ids = _ids(W)
  random_labels = random_hit = near_robots = near_worlds = ties = deep_fn = c_checked = 0
  for b in range(batches):
    scene = _pool_batch(R, W, gen)
    if not boxes:
      scene = scene._replace(box_pos=torch.full_like(scene.box_pos, 1000.0))
    for world in (mesh, spheres):
      world.set_scene(ids, scene)
    q1 = _uniform_q(W, R, gen)
    mesh_flags = _flags(mesh, q1)
    fn = mesh_flags & ~_flags(spheres, q1)
    random_labels += mesh_flags.numel()
    random_hit += int(mesh_flags.sum())
    deep_fn += int(fn.sum())
    # Bisect on the mesh arm: from home with all arms, or from anywhere with one arm.
    q0 = scene.q0
    if b % 2:
      q0 = _uniform_q(W, R, gen)
      mover = torch.randint(R, (W,), generator=gen, device=DEVICE)
      others = torch.ones(W, R, 1, device=DEVICE)
      others[ids, mover] = 0
      q1 = q1 * (1 - others) + q0 * others
    use = ~_flags(mesh, q0).any(1) & _flags(mesh, q1).any(1)
    lo, hi = torch.zeros(W, 1, 1, device=DEVICE), torch.ones(W, 1, 1, device=DEVICE)
    for _ in range(18):  # Joint resolution about 2e-5 rad, far under the 0.44 mm sphere margin.
      mid = (lo + hi) / 2
      f = _flags(mesh, q0 + (q1 - q0) * mid).any(1)
      hi[f] = mid[f]
      lo[~f] = mid[~f]
    q = q0 + (q1 - q0) * hi
    mesh_flags = _flags(mesh, q)
    depth = _deepest(mesh)
    sphere_flags = _flags(spheres, q)
    use = use & mesh_flags.any(1)
    near_worlds += int(use.sum())
    near_robots += int(mesh_flags[use].sum())
    missed = mesh_flags & ~sphere_flags & use[:, None]
    # A stick sphere and the table top are the same geometry in both worlds, so at the very
    # boundary the two float32 paths can round apart. Anything deeper would be a real hole.
    ties += int((missed & (depth >= -TIE)).sum())
    deep_fn += int((missed & (depth < -TIE)).sum())
    # And the C mesh arm as truth, since warp's convex path could be blind too.
    if b < 4:
      sphere_np = sphere_flags.cpu().numpy()
      for w in use.nonzero().flatten()[:200].tolist():
        _copy_to_c(mesh, m, c_data, w)
        truth, truth_depth = _c_mesh_flags(m, c_data, geom_robot, R)
        c_checked += 1
        assert not (truth & ~sphere_np[w] & (truth_depth < -TIE)).any(), w
  print(f'{R}r boxes={boxes}: random {random_labels} labels, {random_hit} mesh hits; '
        f'near {near_worlds} worlds, {near_robots} robots, {ties} float ties, {c_checked} on C')
  assert deep_fn == 0
  assert ties <= near_robots * 1e-3
  assert random_hit > 100_000 and near_robots > 10_000 and c_checked > 300


EXCLUDE_LEVELS = 21
EXCLUDE_GAP = 0.04


def _link_index(name):
  return 8 if name == 'stick' else int(name[4:])


def test_excluded_pairs_never_touch_on_the_mesh_arm():
  # sphere_model 4 sampled for this. The relative pose of a pair depends only on the joints
  # between them, at most 4, so walk a grid to the limits and then search downhill from its
  # worst points, reading mesh distances through a gap window.
  W = 8192
  world = _mesh_stick_world(1, W, nconmax=256, num_obstacles=1)
  world.sim.wp_model.geom_gap.fill_(EXCLUDE_GAP / 2)
  scene = default_scene(world.cfg, W)
  base = scene.base_pos.clone()
  base[..., 2] = 5.0  # Away from table and floor.
  world.set_scene(_ids(W), scene._replace(base_pos=base))
  m = world.sim.mj_model
  names = [m.body(b).name.split('/')[-1] for b in range(m.nbody)]
  body = torch.as_tensor(m.geom_bodyid.astype(np.int64), device=DEVICE)
  gen = _gen(21)

  def distance(q, pair):
    i, j = names.index(pair[0]), names.index(pair[1])
    world.set_q(q[:, None])
    world.sim.kinematics_collision()
    n = int(world._nacon[0])
    assert max(n, int(world._ncollision[0])) <= world._naconmax
    geom = world._contact.geom[:n].long()
    b1, b2 = body[geom[:, 0]], body[geom[:, 1]]
    keep = ((b1 == i) & (b2 == j)) | ((b1 == j) & (b2 == i))
    out = torch.full((W,), float('inf'), device=DEVICE)
    return out.scatter_reduce_(0, world._contact.worldid[:n][keep].long(),
                               world._contact.dist[:n][keep], 'amin')

  def lowest(pair):
    first, last = _link_index(pair[0]), _link_index(pair[1])
    joints = list(range(first, min(last, 7)))
    grids = [torch.linspace(LOWER[j], UPPER[j], EXCLUDE_LEVELS, device=DEVICE) for j in joints]
    grid = torch.cartesian_prod(*grids) if len(grids) > 1 else grids[0][:, None]
    best_d, best_q = [], []
    for chunk in grid.split(W):
      q = HOME.repeat(W, 1)
      q[:len(chunk), joints] = chunk
      d = distance(q, pair)[:len(chunk)]
      keep = d.argsort()[:256]
      best_d.append(d[keep])
      best_q.append(q[keep])
    d, q = torch.cat(best_d), torch.cat(best_q)
    sigma = 0.15
    for _ in range(14):
      pick = q[torch.randint(len(q), (W,), generator=gen, device=DEVICE)].clone()
      pick[:, joints] += sigma * torch.randn(W, len(joints), generator=gen, device=DEVICE)
      pick = torch.minimum(torch.maximum(pick, LOWER), UPPER)
      d, q = torch.cat([d, distance(pick, pair)]), torch.cat([q, pick])
      keep = d.argsort()[:W // 8]
      d, q = d[keep], q[keep]
      sigma *= 0.7
    return d.min().item()

  assert lowest(('link0', 'link5')) < 0  # The same search does find a live pair's contact.
  found = {tuple(pair): lowest(pair) for pair in SPHERES['excludes']}
  print('lowest mesh distance per excluded pair:', {k: round(v, 5) for k, v in found.items()})
  assert min(found.values()) > 5e-4, found


@pytest.mark.parametrize('link', [f'link{i}' for i in range(8)])
def test_spheres_cover_the_hull_of_every_link_inside_and_out(link):
  # What strict coverage means: a primitive touching the hull touches a sphere. MuJoCo collides
  # the convex hull, so test that and not the mesh, and the interior too, for a small box.
  spec = world_module._arm_spec(WorldCfg(arm_model='mesh_tool'))
  model = spec.compile()
  body = model.body(link).id
  verts = []
  for g in range(model.ngeom):
    if (model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[g]):
      mesh = model.geom_dataid[g]
      v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, model.geom_quat[g])
      verts.append((v.astype(np.float64), mat.reshape(3, 3), model.geom_pos[g].copy()))
  assert verts
  spheres = SPHERES['spheres'][link]
  center = torch.as_tensor(np.array([s['center'] for s in spheres]), device=DEVICE)
  radius = torch.as_tensor(np.array([s['radius'] for s in spheres]), device=DEVICE)
  rng = np.random.default_rng(22)
  worst = -np.inf
  for v, mat, pos in verts:
    hull = ConvexHull(v @ mat.T + pos)
    points = hull.points
    corners = points[hull.simplices]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    faces = rng.choice(len(corners), 200_000, p=area / area.sum())
    weights = rng.dirichlet(np.ones(3), 200_000)
    surface = (corners[faces] * weights[..., None]).sum(1)
    mid = points[hull.vertices].mean(0)
    inner = mid + (surface - mid) * rng.uniform(0, 1, (200_000, 1)) ** (1 / 3)
    for samples in (points[hull.vertices], surface, inner):
      x = torch.as_tensor(samples, device=DEVICE)
      excess = ((x[:, None] - center[None]).norm(dim=-1) - radius).min(-1).values
      worst = max(worst, excess.max().item())
  print(f'{link}: worst hull point is {-worst * 1000:.2f} mm inside the spheres')
  assert worst < 0


def test_table_box_holds_the_mesh_table_hull():
  # DESIGN D6. Anything of the hull outside the box is table a sphere could miss.
  model = mujoco.MjSpec.from_file(str(world_module.TABLE_XML))
  model.meshdir = str(world_module.TABLE_XML.parent)
  model = model.compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  center, half = np.array([0, 0, -0.488]), np.array([0.7841, 0.762, 0.488])
  checked = 0
  for g in range(model.ngeom):
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[g]:
      mesh = model.geom_dataid[g]
      v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
      world_v = v.astype(np.float64) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
      excess = (np.abs(world_v - center) - half).max()
      print(f'table geom {g}: hull is at most {excess * 1e6:.1f} um outside the box')
      assert excess < 3e-6  # sphere_model's verifier measured 2.1 um in +x, harmless.
      checked += 1
  assert checked == 2


def test_stick_chain_holds_its_physical_capsule():
  stick = SPHERES['stick']
  z = np.array(stick['z'])
  rng = np.random.default_rng(29)
  # Capsule of the MJCF's stick_visual, surface samples plus its two end caps.
  axis = rng.uniform(0.0068, 0.1454, 200_000)
  angle = rng.uniform(0, 2 * np.pi, 200_000)
  wall = np.stack([0.008 * np.cos(angle), 0.008 * np.sin(angle), axis], 1)
  caps = rng.normal(size=(100_000, 3))
  caps = 0.008 * caps / np.linalg.norm(caps, axis=1, keepdims=True)
  caps[:, 2] = np.abs(caps[:, 2])
  top, bottom = caps + [0, 0, 0.1454], caps * [1, 1, -1] + [0, 0, 0.0068]
  points = np.concatenate([wall, top, bottom])
  centers = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
  excess = (np.linalg.norm(points[:, None] - centers[None], axis=-1) - stick['R']).min(1).max()
  print(f'stick: worst capsule point is {-excess * 1e3:.2f} mm inside the chain')
  assert excess <= 1e-5


def test_8r_sampled_placements_are_free_at_home_on_every_model():
  n = 2048
  world = build_world(WorldCfg(num_robots=8, nconmax=1024), n, DEVICE)
  scene = sampler.blank_scene(world, n, 10)
  failed, counts = sampler.sample_rails(world, scene, sampler.SamplerCfg(), _gen(23))
  print(f'8r rails: {int(failed.sum())} failed of {n}, {counts}')
  assert not failed.any()
  assert not _flags_of_scene(world, scene).any()
  mesh = _mesh_stick_world(8, n)
  assert not _flags_of_scene(mesh, scene).any()
  # Same on C for the spheres and the mesh, and the placements really do use the rail.
  c = CWorld(8, 30)
  m = mesh.sim.mj_model
  m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
  c_data = mujoco.MjData(m)
  geom_robot = mesh._geom_robot.cpu().numpy()
  q = HOME.repeat(8, 1).cpu().numpy()
  for w in range(0, n, 8):
    c.set_scene(*_scene_row(scene, w))
    c.set_q(q)
    c.forward()
    assert not c.flags().any(), w
    _copy_to_c(mesh, m, c_data, w)
    assert not _c_mesh_flags(m, c_data, geom_robot, 8)[0].any(), w
  y = scene.base_pos[..., 1]
  assert y.std(0).min() > 0.1
  # Near-misses exist: this filter is doing work, on the rail of a same-side pair.
  assert counts['rail_draws'] > n * 8


def _flags_of_scene(world, scene):
  world.set_scene(_ids(len(scene.q0)), scene)
  return _flags(world, scene.q0)


def test_stored_8r_validity_labels_reproduce_on_warp_and_c():
  pool_all = stored_scenes(8, valid=None)
  n = len(pool_all.q0)
  labels = torch.as_tensor(np.load(REPO / 'tests/golden/env_scenes_8r10t.npz')['valid'],
                           device=DEVICE)
  world = build_world(WorldCfg(num_robots=8, nconmax=1024), n, DEVICE)
  flags = _flags_of_scene(world, pool_all)
  assert torch.equal(~flags.any(1), labels)
  c = CWorld(8, 30)
  flags = flags.cpu().numpy()
  q = pool_all.q0.cpu().numpy()
  for w in range(n):
    c.set_scene(*_scene_row(pool_all, w))
    c.set_q(q[w])
    c.forward()
    assert np.array_equal(c.flags(), flags[w]), w


# 7. Synchronisation of warp and torch.


def test_warp_stream_is_implicitly_ordered_with_torch():
  world = build_world(WorldCfg(num_obstacles=1), 2, DEVICE)
  torch.zeros(1, device=DEVICE)
  cuda = ctypes.CDLL('libcuda.so.1')
  stream = ctypes.c_void_p(wp.get_stream(wp.get_device(world.sim.wp_device)).cuda_stream)
  flags, stream_ctx, current_ctx = ctypes.c_uint(1), ctypes.c_void_p(), ctypes.c_void_p()
  assert cuda.cuStreamGetFlags(stream, ctypes.byref(flags)) == 0
  assert flags.value == 0  # CU_STREAM_DEFAULT: blocking, not CU_STREAM_NON_BLOCKING.
  assert cuda.cuStreamGetCtx(stream, ctypes.byref(stream_ctx)) == 0
  assert cuda.cuCtxGetCurrent(ctypes.byref(current_ctx)) == 0
  assert stream_ctx.value == current_ctx.value
  # Torch runs on the legacy default stream, which orders against blocking streams. That is
  # the only thing that makes World's unsynchronised reads safe.
  assert torch.cuda.current_stream().cuda_stream == 0


def test_reads_wait_for_a_launch_that_returned_early():
  W, R = 16384, 8
  world = build_world(WorldCfg(num_robots=R), W, DEVICE)
  gen = _gen(24)
  device = wp.get_device(world.sim.wp_device)
  scene = _pool_batch(R, W, gen)
  world.set_scene(_ids(W), scene)
  q_a, q_b = _uniform_q(W, R, gen), _uniform_q(W, R, gen)

  def settled(q):
    world.set_q(q)
    wp.synchronize_device(device)
    world.forward()
    wp.synchronize_device(device)
    return world._n_contacts, world.tcp()[0], world.robots_in_collision()

  ref = settled(q_a), settled(q_b)
  assert ref[0][0] != ref[1][0]
  returned, waited, wrong = [], [], 0
  for i in range(120):
    n_ref, pos_ref, flags_ref = ref[i % 2]
    world.set_q(q_b if i % 2 else q_a)
    if i % 3 == 0:
      world.sim.kinematics_collision()  # Two launches queued: a stale read is easier to catch.
    start = time.perf_counter()
    world.sim.kinematics_collision()
    middle = time.perf_counter()
    n = int(world._nacon[0])
    end = time.perf_counter()
    returned.append(middle - start)
    waited.append(end - middle)
    world._n_contacts = n
    wrong += n != n_ref
    wrong += not torch.equal(world.tcp()[0], pos_ref)
    wrong += not torch.equal(world.robots_in_collision(), flags_ref)
  print(f'launch returns in {np.median(returned) * 1e3:.3f} ms, first read waits '
        f'{np.median(waited) * 1e3:.3f} ms, {wrong} stale reads')
  assert wrong == 0
  assert np.median(waited) > 5 * np.median(returned)  # Otherwise this proved nothing.


def test_torch_work_is_finished_before_the_launch_reads_it():
  W, R = 16384, 8
  world = build_world(WorldCfg(num_robots=R), W, DEVICE)
  gen = _gen(25)
  device = wp.get_device(world.sim.wp_device)
  scene_a, scene_b = _pool_batch(R, W, gen), _hostile(world.cfg, W, gen)
  q_a, q_b = _uniform_q(W, R, gen), _uniform_q(W, R, gen)
  heavy = torch.randn(2048, 2048, generator=gen, device=DEVICE)

  def settled(scene, q):
    world.set_scene(_ids(W), scene)
    world.set_q(q)
    wp.synchronize_device(device)
    world.forward()
    wp.synchronize_device(device)
    return world.robots_in_collision(), world.tcp()[0]

  ref = settled(scene_a, q_a), settled(scene_b, q_b)
  for i in range(60):
    scene, q = (scene_b, q_b) if i % 2 else (scene_a, q_a)
    # q arrives from a matmul still running on torch's stream, and goes straight to warp's.
    late = q + 0 * heavy.matmul(heavy).sum()
    world.set_scene(_ids(W), scene)
    world.set_q(late)
    world.forward()
    assert torch.equal(world.robots_in_collision(), ref[i % 2][0]), i
    assert torch.equal(world.tcp()[0], ref[i % 2][1]), i
