# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Rails, boxes and targets, batched on the GPU. DESIGN section 6, env_spec.md 10.

Everything is drawn from one torch.Generator, and every loop is bounded: a world that
runs out of draws is reported as failed and dropped by the caller, never retried forever.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import torch

from roboballet_mjlab.config import HOME_QPOS
from roboballet_mjlab.scenes.ik import ArmKinematics, IkBudget, find_witnesses, scene_subset
from roboballet_mjlab.sim.world import World, default_bases
from roboballet_mjlab.structs import Scene

_PARKED_BASE_X = 500.0  # A floor robot that has no rail position yet.
_PARKED_BOX_Z = 100.0
_CEILING_Z = 1.0  # Bases above this are ceiling robots.


@dataclasses.dataclass(frozen=True)
class SamplerCfg:
  """Distributions from env_spec.md 10 and paper_baseline.md 7.3, and the draw bounds."""

  rail_limit: float = 0.8
  # Stored floor robots stop here on the far side of their rail (env_spec D2, cause unknown).
  floor_rail_cut: float = 0.382
  spans_min: tuple[float, ...] = (0.02, 0.02, 0.1)
  spans_max: tuple[float, ...] = (0.05, 0.05, 0.6)
  box_pos_min: tuple[float, ...] = (-0.3, -0.3, 0.0)
  box_pos_max: tuple[float, ...] = (0.3, 0.3, 1.2)
  surface_offset: float = 0.03
  cone_half_angle: float = math.pi / 8
  max_rail_draws: int = 200  # Per robot. The original has `while True`.
  max_box_draws: int = 100  # Consecutive, planning_env.py:42.
  draws_per_missing: int = 2  # Candidates per missing target per round.
  max_target_rounds: int = 60


class Witnesses(NamedTuple):
  q: torch.Tensor  # [N, T, 7]
  robot: torch.Tensor  # [N, T], -1 for a target with no witness.


def _uniform(lo: torch.Tensor, hi: torch.Tensor, shape, gen) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(shape, device=lo.device, generator=gen)


def random_rotations(n: int, device, gen) -> torch.Tensor:
  """Uniform on SO(3), from a normalised Gaussian quaternion (geometry.py:53-65)."""
  q = torch.randn(n, 4, device=device, generator=gen)
  w, x, y, z = (q / q.norm(dim=-1, keepdim=True).clamp(min=1e-7)).unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ], dim=-1).view(n, 3, 3)


def _parked_boxes(n: int, num_obstacles: int, device) -> tuple[torch.Tensor, ...]:
  pos = torch.zeros(n, num_obstacles, 3, device=device)
  pos[..., 2] = _PARKED_BOX_Z + torch.arange(num_obstacles, device=device, dtype=pos.dtype)
  rot = torch.eye(3, device=device).repeat(n, num_obstacles, 1, 1)
  spans = torch.full((n, num_obstacles, 3), 0.05, device=device)
  return pos, rot, spans


def blank_scene(world: World, n: int, num_targets: int) -> Scene:
  """Default bases, boxes parked, no targets yet, arms at home."""
  device = world.device
  base_pos, base_rot = default_bases(world.cfg, device)
  box_pos, box_rot, box_spans = _parked_boxes(n, world.num_obstacles, device)
  return Scene(
    base_pos=base_pos.repeat(n, 1, 1),
    base_rot=base_rot.repeat(n, 1, 1, 1),
    box_pos=box_pos,
    box_rot=box_rot,
    box_spans=box_spans,
    target_pos=torch.zeros(n, num_targets, 3, device=device),
    target_rot=torch.eye(3, device=device).repeat(n, num_targets, 1, 1),
    q0=torch.as_tensor(HOME_QPOS, device=device).repeat(n, world.num_robots, 1),
  )


def any_robot_collides(world: World, scene: Scene, world_ids: torch.Tensor) -> torch.Tensor:
  """Places the scene in worlds 0..len(world_ids)-1, arms at q0. [len(world_ids)] bool."""
  batch = scene_subset(scene, world_ids)
  w = torch.arange(len(world_ids), device=world_ids.device)
  world.set_scene(w, batch)
  world.set_q(batch.q0, w)
  world.forward()
  return world.robots_in_collision()[w].any(-1)


def rail_ranges(world: World, cfg: SamplerCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """y range per robot [R] and whether it is a floor robot [R]."""
  pos, _ = default_bases(world.cfg, world.device)
  floor = pos[:, 2] < _CEILING_Z
  lo = torch.full((world.num_robots,), -cfg.rail_limit, device=world.device)
  hi = -lo
  low_side = pos[:, 1] < 0
  hi[floor & low_side] = cfg.floor_rail_cut
  lo[floor & ~low_side] = -cfg.floor_rail_cut
  return lo, hi, floor


def sample_rails(world: World, scene: Scene, cfg: SamplerCfg, gen) -> tuple[torch.Tensor, dict]:
  """Writes base_pos in place, sequentially in a per-scene shuffled order. Returns failures [N].

  This is the fit of paper_baseline.md 7.3. A floor robot without a position yet is out of
  the way, because its stored data shows no sign of the partner's default (a default partner
  would cut y at 0.3697, the data reaches 0.3824). A ceiling robot waits at its default, as
  in the released sampler, planning_env.py:588-606, which the ceiling data matches.
  """
  n, num_robots = len(scene.base_pos), world.num_robots
  device = scene.base_pos.device
  lo, hi, floor = rail_ranges(world, cfg)
  order = torch.rand(n, num_robots, device=device, generator=gen).argsort(dim=1)
  default_x = scene.base_pos[0, :, 0].clone()
  # Apart from each other too, or the parked robots would collide.
  scene.base_pos[:, floor, 0] = _PARKED_BASE_X + 10.0 * floor.nonzero()[:, 0].float()
  failed = torch.zeros(n, dtype=torch.bool, device=device)
  draws = 0
  for step in range(num_robots):
    robot = order[:, step]
    scene_ids = torch.arange(n, device=device)
    scene.base_pos[scene_ids, robot, 0] = default_x[robot]
    pending = scene_ids
    for _ in range(cfg.max_rail_draws):
      draws += len(pending)
      r = robot[pending]
      scene.base_pos[pending, r, 1] = _uniform(lo[r], hi[r], (len(pending),), gen)
      pending = pending[any_robot_collides(world, scene, pending)]
      if len(pending) == 0:
        break
    failed[pending] = True
  return failed, {'rail_draws': draws}


def sample_boxes(world: World, scene: Scene, cfg: SamplerCfg, gen) -> tuple[torch.Tensor, dict]:
  """Writes boxes in place, each rejected alone against robots at home. Returns failures [N].

  Only robot contacts count: boxes may overlap each other and the table (env_spec 10.1).
  """
  n, device = len(scene.box_pos), scene.box_pos.device
  spans_lo, spans_hi = (torch.as_tensor(x, device=device) for x in (cfg.spans_min, cfg.spans_max))
  pos_lo, pos_hi = (torch.as_tensor(x, device=device) for x in (cfg.box_pos_min, cfg.box_pos_max))
  parked_pos, parked_rot, parked_spans = (x.clone() for x in _parked_boxes(1, world.num_obstacles, device))
  failed = torch.zeros(n, dtype=torch.bool, device=device)
  draws = 0
  for box in range(world.num_obstacles):
    pending = torch.arange(n, device=device)
    for _ in range(cfg.max_box_draws):
      k = len(pending)
      draws += k
      scene.box_spans[pending, box] = _uniform(spans_lo, spans_hi, (k, 3), gen)
      scene.box_pos[pending, box] = _uniform(pos_lo, pos_hi, (k, 3), gen)
      scene.box_rot[pending, box] = random_rotations(k, device, gen)
      pending = pending[any_robot_collides(world, scene, pending)]
      if len(pending) == 0:
        break
    # Park a box that never fitted, so the failed world cannot poison later boxes.
    scene.box_pos[pending, box] = parked_pos[0, box]
    scene.box_rot[pending, box] = parked_rot[0, box]
    scene.box_spans[pending, box] = parked_spans[0, box]
    failed[pending] = True
  return failed, {'box_draws': draws}


def _random_perpendicular(axis: torch.Tensor, gen) -> torch.Tensor:
  """A unit vector uniform on the circle normal to the unit vectors `axis`, [M, 3]."""
  g = torch.randn(axis.shape, device=axis.device, generator=gen)
  # Twice, because one projection of a g nearly along the axis is not orthogonal in f32.
  for _ in range(2):
    g = g - (g * axis).sum(-1, keepdim=True) * axis
    g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-7)
  return g


def draw_candidates(scene: Scene, scene_ids: torch.Tensor, cfg: SamplerCfg, gen):
  """Candidate target poses on boxes of the given scenes. pos [M, 3], rot [M, 3, 3].

  Area-weighted face, uniform point, offset outward, +z within the cone about the inward
  normal, uniform roll. This follows the stored data, not the released code (env_spec D1).
  """
  m, device = len(scene_ids), scene_ids.device
  spans = scene.box_spans[scene_ids]  # [M, O, 3]
  face_area = spans.prod(-1, keepdim=True) / spans  # [M, O, 3], one per opposite pair.
  pick = torch.multinomial(face_area.flatten(1), 1, generator=gen)[:, 0]
  box, axis = pick // 3, pick % 3
  rows = torch.arange(m, device=device)
  half = spans[rows, box] / 2
  sign = 2.0 * torch.randint(0, 2, (m,), device=device, generator=gen) - 1.0
  local = half * (2 * torch.rand(m, 3, device=device, generator=gen) - 1)
  local[rows, axis] = sign * (half[rows, axis] + cfg.surface_offset)
  box_rot = scene.box_rot[scene_ids, box]
  pos = scene.box_pos[scene_ids, box] + (box_rot @ local[..., None])[..., 0]
  inward = -sign[:, None] * box_rot[rows, :, axis]
  cos = 1 - torch.rand(m, 1, device=device, generator=gen) * (1 - math.cos(cfg.cone_half_angle))
  z = cos * inward + (1 - cos * cos).clamp(min=0).sqrt() * _random_perpendicular(inward, gen)
  z = z / z.norm(dim=-1, keepdim=True)  # 1 - cos^2 is coarse in f32 near the axis.
  x = _random_perpendicular(z, gen)
  return pos, torch.stack([x, torch.linalg.cross(z, x), z], dim=-1)


def _rank_within_scene(sorted_ids: torch.Tensor) -> torch.Tensor:
  """0, 1, 2 ... within each run of equal ids."""
  _, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
  starts = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
  return torch.arange(len(sorted_ids), device=sorted_ids.device) - starts


def sample_targets(world: World, kin: ArmKinematics, scene: Scene, cfg: SamplerCfg,
                   budget: IkBudget, gen) -> tuple[Witnesses, torch.Tensor, dict]:
  """Fills the targets in place. Returns witnesses, failures [N] and draw counts.

  Candidates of a round are drawn before any is judged and taken in draw order, so the
  accepted targets stay independent draws from the filtered distribution.
  """
  n, num_targets = scene.target_pos.shape[:2]
  device = scene.target_pos.device
  witness_q = torch.zeros(n, num_targets, 7, device=device)
  witness_robot = torch.full((n, num_targets), -1, device=device)
  filled = torch.zeros(n, dtype=torch.long, device=device)
  draws, accepted, restarts = 0, 0, 0
  for _ in range(cfg.max_target_rounds):
    active = (filled < num_targets).nonzero()[:, 0]
    if len(active) == 0:
      break
    scene_ids = active.repeat_interleave((num_targets - filled[active]) * cfg.draws_per_missing)
    pos, rot = draw_candidates(scene, scene_ids, cfg, gen)
    result = find_witnesses(world, kin, scene, scene_ids, pos, rot, budget, gen)
    draws += len(scene_ids)
    restarts += result.restarts_run
    idx = result.found.nonzero()[:, 0]
    accepted += len(idx)
    if len(idx) == 0:
      continue
    owner = scene_ids[idx]
    slot = filled[owner] + _rank_within_scene(owner)
    fits = slot < num_targets
    idx, owner, slot = idx[fits], owner[fits], slot[fits]
    scene.target_pos[owner, slot] = pos[idx]
    scene.target_rot[owner, slot] = rot[idx]
    witness_q[owner, slot] = result.q[idx]
    witness_robot[owner, slot] = result.robot[idx]
    filled += torch.bincount(owner, minlength=n)
  counts = {'target_draws': draws, 'target_accepted': accepted, 'ik_restart_problems': restarts}
  return Witnesses(witness_q, witness_robot), filled < num_targets, counts


class Generated(NamedTuple):
  scene: Scene
  witnesses: Witnesses
  counts: dict


def generate(world: World, kin: ArmKinematics, n: int, num_targets: int, cfg: SamplerCfg,
             budget: IkBudget, gen: torch.Generator) -> Generated:
  """n attempts at training scenes. Failed worlds are dropped, and counted."""
  if n > world.num_worlds:
    raise ValueError(f'{n} scenes do not fit in {world.num_worlds} worlds')
  scene = blank_scene(world, n, num_targets)
  rails_failed, rail_counts = sample_rails(world, scene, cfg, gen)
  scene = scene_subset(scene, ~rails_failed)
  boxes_failed, box_counts = sample_boxes(world, scene, cfg, gen)
  scene = scene_subset(scene, ~boxes_failed)
  witnesses, targets_failed, target_counts = sample_targets(world, kin, scene, cfg, budget, gen)
  keep = ~targets_failed
  counts = {
    'attempted': n,
    'rail_failures': int(rails_failed.sum()),
    'box_failures': int(boxes_failed.sum()),
    'target_failures': int(targets_failed.sum()),
    **rail_counts, **box_counts, **target_counts,
  }
  return Generated(scene_subset(scene, keep), Witnesses(witnesses.q[keep], witnesses.robot[keep]), counts)
