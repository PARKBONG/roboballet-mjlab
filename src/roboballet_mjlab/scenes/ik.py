# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Batched damped least squares and collision-free witness search. DESIGN section 6.

The solver follows S/spheres/ik_lib.py. Each restart aims at a random pose inside the
tolerance and is judged on the true target (env_spec.md, verifier notes row 11). A
witness counts only after sim.world.World, the collision model used in training, has
confirmed both the pose and the absence of contacts, with the other robots at home.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
import torch

from roboballet_mjlab.config import JOINT_LOWER, JOINT_UPPER, ArmModel
from roboballet_mjlab.sim.world import MESH_ARM_XML, SPHERE_ARM_MESHDIR, SPHERE_ARM_XML, World
from roboballet_mjlab.structs import Scene

_TCP_SITE = {'spheres_stick': 'tool_tip_site', 'mesh_tool': 'tool/tool_tip_site'}
ARM_XML = {'spheres_stick': SPHERE_ARM_XML, 'mesh_tool': MESH_ARM_XML}
_NEAR_PI_SIN = 1e-4  # Below this the antisymmetric part of a near half turn is f32 noise.


@dataclasses.dataclass(frozen=True)
class IkBudget:
  """The filter strength, stored in every pool, because it selects the target population.

  The original solver is not in the release, so `restarts` is calibrated rather than
  copied: run on the original tool, 2 restarts reproduce the stored 4r/10t target marginals
  best and 4 or more fail E10 on target radius. A stronger search admits targets nearer the
  cell centre, so the mean target radius falls as the budget rises. Search difficulty is not
  policy difficulty, so this says nothing about how hard the targets are to learn. The budget
  was chosen on the 4r/10t E10 statistic, so that pass is an in-sample fit, and 8r/40t is the
  out-of-sample check. See docs/reviews/scenes_status.md.
  """

  restarts: int = 2  # Per target and robot. Restart 0 is home and the exact target.
  iters: int = 40  # Descent steps per restart (S/verify/skeptic_env/sk_world.py:ik_scan).
  damping: float = 0.02
  max_step_rad: float = 0.25  # Largest joint change per step; the whole step is scaled.
  aim_fraction: float = 0.8  # Restarts aim within this fraction of the tolerance.
  pos_tol: float = 0.025  # Strict, as planning_env scores it (target.py:152-164).
  ang_tol: float = math.radians(15.0)
  # Witnesses must clear the tolerance by this factor, so that recomputing them in f64
  # on C MuJoCo from f32 joint values cannot cross a strict comparison.
  margin: float = 0.999
  reach_m: float = 1.3  # mujoco_simulator.py:97, 522-526.


def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, quat)
  return rot.reshape(3, 3)


class ArmKinematics:
  """FK of one arm, with constants read from the same MJCF that World loads."""

  def __init__(self, arm_model: ArmModel, device) -> None:
    spec = mujoco.MjSpec.from_file(str(ARM_XML[arm_model]))
    if arm_model == 'spheres_stick':
      spec.meshdir = str(SPHERE_ARM_MESHDIR)  # Its visual meshes are in the data directory.
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    links = [model.body(f'link{i}') for i in range(1, 8)]
    joints = [model.joint(f'joint{i}') for i in range(1, 8)]
    if any(not np.array_equal(j.axis, [0, 0, 1]) or np.any(j.pos) for j in joints):
      raise ValueError('The chain below assumes each joint is the z axis of its body')
    site = model.site(_TCP_SITE[arm_model]).id
    link7_rot = data.xmat[links[-1].id].reshape(3, 3)
    tcp_pos = link7_rot.T @ (data.site_xpos[site] - data.xpos[links[-1].id])
    tcp_rot = link7_rot.T @ data.site_xmat[site].reshape(3, 3)
    as_f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    self.link_pos = as_f32([link.pos for link in links])
    self.link_rot = as_f32([_quat_to_rot(link.quat) for link in links])
    self.tcp_pos = as_f32(tcp_pos)
    self.tcp_rot = as_f32(tcp_rot)
    self.lower = as_f32(JOINT_LOWER)
    self.upper = as_f32(JOINT_UPPER)

  def fk(self, q: torch.Tensor, base_pos: torch.Tensor, base_rot: torch.Tensor):
    """q [N, 7]. Returns TCP pos [N, 3], rot [N, 3, 3], joint origins and axes [N, 7, 3]."""
    pos, rot = base_pos, base_rot
    cos, sin = q.cos(), q.sin()
    origins, axes = [], []
    for i in range(7):
      pos = pos + (rot @ self.link_pos[i])
      rot = rot @ self.link_rot[i]
      x, y, z = rot[..., 0], rot[..., 1], rot[..., 2]
      origins.append(pos)
      axes.append(z)
      c, s = cos[:, i, None], sin[:, i, None]
      rot = torch.stack([c * x + s * y, c * y - s * x, z], dim=-1)
    tcp_pos = pos + (rot @ self.tcp_pos)
    return tcp_pos, rot @ self.tcp_rot, torch.stack(origins, 1), torch.stack(axes, 1)


def rotation_error(rot_target: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
  """World-frame rotation vector taking rot to rot_target, [N, 3]. Magnitude is the angle.

  Near a half turn the antisymmetric part v vanishes, so the axis comes from the symmetric
  part, (e + e^T) / 2 - cos I = (1 - cos) a a^T, whose largest column is the best conditioned.
  A half turn has no preferred sign: v chooses one if it has any signal, otherwise it is +a.
  """
  e = rot_target @ rot.transpose(-1, -2)
  v = 0.5 * torch.stack(
    [e[:, 2, 1] - e[:, 1, 2], e[:, 0, 2] - e[:, 2, 0], e[:, 1, 0] - e[:, 0, 1]], dim=-1)
  sin = v.norm(dim=-1, keepdim=True)
  cos = 0.5 * (e.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) - 1)
  angle = torch.atan2(sin, cos)
  outer = 0.5 * (e + e.transpose(-1, -2)) - cos[..., None] * torch.eye(3, device=e.device, dtype=e.dtype)
  column = outer.diagonal(dim1=-2, dim2=-1).argmax(-1)
  axis = outer[torch.arange(len(e), device=e.device), :, column]
  axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-7)
  axis = torch.where((axis * v).sum(-1, keepdim=True) < 0, -axis, axis)
  near_pi = (sin < _NEAR_PI_SIN) & (cos < 0)
  return torch.where(near_pi, axis * angle, v * (angle / sin.clamp(min=1e-7)))


def rotation_exp(w: torch.Tensor) -> torch.Tensor:
  """Rodrigues, [N, 3] to [N, 3, 3]."""
  angle = w.norm(dim=-1, keepdim=True).clamp(min=1e-7)
  x, y, z = (w / angle).unbind(-1)
  zero = torch.zeros_like(x)
  k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).view(-1, 3, 3)
  sin, cos = angle.sin()[..., None], angle.cos()[..., None]
  return torch.eye(3, device=w.device) + sin * k + (1 - cos) * (k @ k)


def within_tolerance(pos, rot, target_pos, target_rot, budget: IkBudget) -> torch.Tensor:
  """The predicate of env/ballet_env.py:targets_within, with the margin on both tolerances."""
  pos_err = (pos - target_pos).norm(dim=-1)
  trace = (rot * target_rot).sum((-1, -2))
  return (pos_err < budget.pos_tol * budget.margin) & (
    trace > 1 + 2 * math.cos(budget.ang_tol * budget.margin))


def descend(kin: ArmKinematics, q, aim_pos, aim_rot, base_pos, base_rot, budget: IkBudget):
  """Runs the budgeted damped least squares steps from q, [N, 7]."""
  damping = budget.damping ** 2 * torch.eye(6, device=q.device)
  for _ in range(budget.iters):
    pos, rot, origins, axes = kin.fk(q, base_pos, base_rot)
    error = torch.cat([aim_pos - pos, rotation_error(aim_rot, rot)], dim=-1)
    linear = torch.linalg.cross(axes, pos[:, None] - origins)
    jac = torch.cat([linear, axes], dim=-1).transpose(1, 2)  # [N, 6, 7]
    chol, _ = torch.linalg.cholesky_ex(jac @ jac.transpose(1, 2) + damping)
    dq = (jac.transpose(1, 2) @ torch.cholesky_solve(error[..., None], chol))[..., 0]
    scale = (budget.max_step_rad / dq.abs().amax(-1, keepdim=True).clamp(min=1e-9)).clamp(max=1)
    q = torch.minimum(torch.maximum(q + scale * dq, kin.lower), kin.upper)
  return q


def _unit_gaussian(shape, device, gen) -> torch.Tensor:
  v = torch.randn(shape, device=device, generator=gen)
  return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-7)


def _aim_inside_tolerance(pos, rot, budget: IkBudget, gen):
  """A random pose within aim_fraction of the tolerance (sk_world.py:ik_scan, 'perturb')."""
  n, device = len(pos), pos.device
  radius = torch.rand(n, 1, device=device, generator=gen) ** (1 / 3)
  angle = torch.rand(n, 1, device=device, generator=gen)
  aim_pos = pos + budget.aim_fraction * budget.pos_tol * radius * _unit_gaussian((n, 3), device, gen)
  w = budget.aim_fraction * budget.ang_tol * angle * _unit_gaussian((n, 3), device, gen)
  return aim_pos, rotation_exp(w) @ rot


def scene_subset(scene: Scene, ids: torch.Tensor) -> Scene:
  return Scene(*[field[ids] for field in scene])


def witness_valid(world: World, scene: Scene, scene_ids, robot, q, pos, rot, budget: IkBudget):
  """Checks poses through World, in chunks of its size, with the other robots at q0. [N] bool."""
  out = []
  ids = torch.arange(world.num_worlds, device=q.device)
  for start in range(0, len(q), world.num_worlds):
    chunk = torch.arange(start, min(start + world.num_worlds, len(q)), device=q.device)
    w = ids[: len(chunk)]
    batch = scene_subset(scene, scene_ids[chunk])
    world.set_scene(w, batch)
    q_all = batch.q0.clone()
    q_all[torch.arange(len(chunk)), robot[chunk]] = q[chunk]
    world.set_q(q_all, w)
    world.forward()
    free = ~world.robots_in_collision()[w].any(-1)
    tcp_pos, tcp_rot = world.tcp()
    near = within_tolerance(tcp_pos[w, robot[chunk]], tcp_rot[w, robot[chunk]],
                            pos[chunk], rot[chunk], budget)
    out.append(free & near)
  return torch.cat(out) if out else torch.zeros(0, dtype=torch.bool, device=q.device)


@dataclasses.dataclass
class WitnessResult:
  found: torch.Tensor  # [M] bool
  q: torch.Tensor  # [M, 7]
  robot: torch.Tensor  # [M], -1 where not found.
  restarts_run: int  # Restart problems solved, for the rate reports.


def find_witnesses(world: World, kin: ArmKinematics, scene: Scene, scene_ids, pos, rot,
                   budget: IkBudget, gen: torch.Generator) -> WitnessResult:
  """Searches a collision-free solution per candidate target, pos [M, 3], rot [M, 3, 3].

  Every robot within reach gets each restart, as mujoco_simulator.py:514-583 does for one
  robot at a time. A candidate leaves the search at its first witness.
  """
  device, m = pos.device, len(pos)
  found = torch.zeros(m, dtype=torch.bool, device=device)
  best_q = torch.zeros(m, 7, device=device)
  best_robot = torch.full((m,), -1, device=device)
  base_pos, base_rot = scene.base_pos[scene_ids], scene.base_rot[scene_ids]
  in_reach = (pos[:, None] - base_pos).norm(dim=-1) <= budget.reach_m
  cand, robot = in_reach.nonzero(as_tuple=True)
  home = scene.q0[scene_ids]
  run = 0
  for restart in range(budget.restarts):
    if len(cand) == 0:
      break
    run += len(cand)
    p, r = pos[cand], rot[cand]
    b_pos, b_rot = base_pos[cand, robot], base_rot[cand, robot]
    if restart == 0:
      q, aim_pos, aim_rot = home[cand, robot], p, r
    else:
      u = torch.rand(len(cand), 7, device=device, generator=gen)
      q = kin.lower + u * (kin.upper - kin.lower)
      aim_pos, aim_rot = _aim_inside_tolerance(p, r, budget, gen)
    q = descend(kin, q, aim_pos, aim_rot, b_pos, b_rot, budget)
    tcp_pos, tcp_rot, _, _ = kin.fk(q, b_pos, b_rot)
    hit = within_tolerance(tcp_pos, tcp_rot, p, r, budget).nonzero()[:, 0]
    ok = witness_valid(world, scene, scene_ids[cand[hit]], robot[hit], q[hit], p[hit], r[hit], budget)
    win = hit[ok]
    # Problems are sorted, so this keeps the lowest robot, the one the original tries first
    # (planning_env.py:500-505). Duplicate indices would be nondeterministic on the GPU.
    win = win[torch.cat([torch.ones_like(win[:1], dtype=torch.bool), cand[win[1:]] != cand[win[:-1]]])]
    best_q[cand[win]] = q[win]
    best_robot[cand[win]] = robot[win]
    found[cand[win]] = True
    keep = ~found[cand]
    cand, robot = cand[keep], robot[keep]
  return WitnessResult(found, best_q, best_robot, run)
