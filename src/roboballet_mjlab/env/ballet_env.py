# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Batched RoboBallet environment. env_spec.md sections 4 to 9, DESIGN section 5."""

from __future__ import annotations

import math

import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, EnvCfg
from roboballet_mjlab.sim.world import build_world
from roboballet_mjlab.structs import Obs, RobotState, Scene, StepOut

_MAX_SCENE_DRAWS = 100  # The original gives up after 100 failures as well, planning_env.py:42.


def targets_within(
  tip_pos: torch.Tensor,
  tip_rot: torch.Tensor,
  target_pos: torch.Tensor,
  target_rot: torch.Tensor,
  cfg: EnvCfg,
) -> torch.Tensor:
  """[W, T, R] bool: strict 2.5 cm and 15 degrees over full SO(3), target.py:152-164."""
  near = (tip_pos[:, None] - target_pos[:, :, None]).square().sum(-1) < cfg.position_threshold**2
  # trace(Ra^T Rb) = 1 + 2 cos(angle), so the rotation test needs no acos.
  trace = (tip_rot[:, None] * target_rot[:, :, None]).sum((-1, -2))
  return near & (trace > 1 + 2 * math.cos(cfg.angle_threshold_rad))


def solve_targets(
  within: torch.Tensor, done: torch.Tensor, dwell: torch.Tensor, dwell_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """The ordered double loop of planning_env.py:639-679, without per-world Python.

  Targets go in index order and a solving robot starts to dwell at once, so it is
  skipped for later targets of the same step. Robots go in index order, so the
  lowest-index free robot within tolerance takes the target. Returns new tensors.
  """
  hits = within & ~done[:, :, None]
  hot = hits.any(dim=(0, 2)).nonzero().flatten().tolist()
  if not hot:
    return done, dwell
  done, dwell = done.clone(), dwell.clone()
  for k in hot:
    free = hits[:, k] & (dwell == 0)
    solved = free.any(1)
    robot = free.to(torch.int8).argmax(1)
    done[:, k] |= solved
    dwell[solved, robot[solved]] = dwell_steps
  return done, dwell


class BalletEnv:
  """W worlds stepping in lockstep, with immediate reset of terminated worlds.

  Tensors handed out by `observe` and `step` are never written to afterwards.
  """

  def __init__(
    self, cfg: EnvCfg, num_worlds: int, scene_pool: Scene, device, seed: int,
    debug: bool = False,
  ) -> None:
    self.cfg = cfg
    self.num_worlds = num_worlds
    self.device = device
    self.debug = debug
    self.pool = Scene(*[x.to(device) for x in scene_pool])
    self._check_pool(self.pool)
    self.world = build_world(cfg.world, num_worlds, device)
    self.generator = torch.Generator(device=device)
    self.generator.manual_seed(seed)
    # Worlds that ended a step in collision, and draws rejected by reset. Both stay on
    # the GPU, and the first must stay 0: DESIGN section 5 and E4.
    self.residual_collisions = torch.zeros((), dtype=torch.long, device=device)
    self.invalid_scene_draws = torch.zeros((), dtype=torch.long, device=device)

    as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    self._lower, self._upper = as_tensor(JOINT_LOWER), as_tensor(JOINT_UPPER)
    self._home = as_tensor(HOME_QPOS)
    # Lexicographic pick of deterministic_mode, base_env.py:248. In the 8-robot order
    # this is not the index order.
    names = cfg.world.robot_names
    self._name_rank = as_tensor([-float(sorted(names).index(n)) for n in names])
    self._all = torch.arange(num_worlds, device=device)
    self._scene: Scene
    self._robot: RobotState
    self.reset()

  def _check_pool(self, pool: Scene) -> None:
    world = self.cfg.world
    found = (pool.base_pos.shape[1], pool.box_pos.shape[1], pool.target_pos.shape[1])
    if found != (world.num_robots, world.num_obstacles, self.cfg.num_targets):
      raise ValueError(f'scene pool has (R, O, T) = {found}, config wants '
                       f'{(world.num_robots, world.num_obstacles, self.cfg.num_targets)}')

  def _draw_aug(self, n: int) -> torch.Tensor:
    shape = (n, self.cfg.world.num_obstacles)
    if not self.cfg.augment_obstacles:
      return torch.zeros(shape, dtype=torch.long, device=self.device)
    return torch.randint(4, shape, generator=self.generator, device=self.device)

  def _draw_scene_ids(self, n: int) -> torch.Tensor:
    size = self.pool.base_pos.shape[0]
    return torch.randint(size, (n,), generator=self.generator, device=self.device)

  def _load_scenes(self, ids: torch.Tensor, scene_ids: torch.Tensor) -> None:
    scene = Scene(*[x[scene_ids] for x in self.pool])
    self.world.set_scene(ids, scene)
    self.world.set_q(scene.q0, ids)
    self.world.forward()

  def reset(self, world_ids: torch.Tensor | None = None, scene_ids: torch.Tensor | None = None):
    """Starts new episodes. Scenes colliding at their start pose are redrawn.

    `scene_ids` selects scenes from the pool instead of drawing, and then a colliding
    scene raises. The original catches InvalidStartStateError and redraws, actor.py:189-193.
    """
    ids = self._all if world_ids is None else world_ids
    fixed = scene_ids is not None
    chosen = scene_ids.clone() if fixed else self._draw_scene_ids(len(ids))
    self._load_scenes(ids, chosen)
    if self.cfg.collisions_enabled:
      for _ in range(_MAX_SCENE_DRAWS):
        bad = self.world.robots_in_collision()[ids].any(1).nonzero().flatten()
        if not len(bad):
          break
        if fixed:
          raise ValueError(f'scenes {chosen[bad].tolist()} collide at their start pose')
        self.invalid_scene_draws += len(bad)
        chosen[bad] = self._draw_scene_ids(len(bad))
        self._load_scenes(ids[bad], chosen[bad])
      else:
        raise RuntimeError(f'no valid scene after {_MAX_SCENE_DRAWS} draws: bad scene pool')

    n = len(ids)
    scene = Scene(*[x[chosen] for x in self.pool])
    tip_pos, tip_rot = self.world.tcp()
    robot = RobotState(
      q=scene.q0,
      qd=torch.zeros_like(scene.q0),
      dwell=torch.zeros(n, self.cfg.world.num_robots, dtype=torch.long, device=self.device),
      tip_pos=tip_pos[ids],
      tip_rot=tip_rot[ids],
      target_done=torch.zeros(n, self.cfg.num_targets, dtype=torch.bool, device=self.device),
      score=torch.zeros(n, dtype=torch.float32, device=self.device),
      step=torch.zeros(n, dtype=torch.long, device=self.device),
      aug_code=self._draw_aug(n),
    )
    if world_ids is None:
      self._scene, self._robot = scene, robot
    else:
      self._scene = Scene(*[a.index_copy(0, ids, b) for a, b in zip(self._scene, scene)])
      self._robot = RobotState(*[a.index_copy(0, ids, b) for a, b in zip(self._robot, robot)])

  def observe(self) -> Obs:
    return Obs(self._robot, self._scene)

  def _pick_rollback(self, candidates: torch.Tensor) -> torch.Tensor:
    """One candidate per world: uniform, or lexicographically first (base_env.py:247-250)."""
    if self.cfg.rollback == 'random':
      keys = torch.rand(candidates.shape, generator=self.generator, device=self.device)
    else:
      keys = self._name_rank.expand(candidates.shape)
    robot = keys.masked_fill(~candidates, -math.inf).argmax(1, keepdim=True)
    return torch.zeros_like(candidates).scatter_(1, robot, True) & candidates

  def _move(
    self, q_old: torch.Tensor, q_new: torch.Tensor, qd: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """base_env.py:227-261. Returns final q, final qd and the rolled-back robots [W, R]."""
    self.world.set_q(q_new)
    self.world.forward()
    rolled = torch.zeros(q_new.shape[:2], dtype=torch.bool, device=self.device)
    if not self.cfg.collisions_enabled:
      return q_new, qd, rolled
    colliding = self.world.robots_in_collision()
    for _ in range(self.cfg.world.num_robots):
      candidates = colliding & ~rolled
      if not candidates.any():
        break
      pick = self._pick_rollback(candidates)
      rolled = rolled | pick
      # Whole robot back, all 7 velocities zero, then a full re-check of every robot.
      q_new = torch.where(pick[..., None], q_old, q_new)
      qd = qd.masked_fill(pick[..., None], 0.0)
      self.world.set_q(q_new)
      self.world.forward()
      colliding = self.world.robots_in_collision()
    # The previous state is free, so this cannot fire unless a scene or the sim is wrong.
    residual = colliding.any(1)
    self.residual_collisions += residual.sum()
    if self.debug and residual.any():
      raise AssertionError(f'rollback left worlds {residual.nonzero().flatten().tolist()} colliding')
    return q_new, qd, rolled

  def _score(self, done: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """planning_env.py:680-691. Only `done` matters: shaping defaults to zero, target.py:92."""
    weight = self.cfg.return_to_start_reward_weight
    score = (1.0 - weight) * done.float().mean(1)
    if weight == 0.0:
      return score
    diff = (q - self._home).abs()
    within = (diff < self.cfg.return_to_start_threshold).all(-1)
    shaped = 1.0 - (self.cfg.return_to_start_coeff * diff.square().mean(-1)).clamp(0.0, 1.0)
    rts = torch.where(within, 1.0, shaped).mean(1)
    return score + weight * rts * done.all(1)

  def step(self, action: torch.Tensor) -> StepOut:
    """Order of env_spec 8.1, then immediate reset of terminated worlds. `action` is in [-1, 1]."""
    cfg, st = self.cfg, self._robot
    dt = cfg.timestep_s
    requested = action * cfg.max_joint_velocity

    # Dwell zeroes the request and counts down before the move, planning_env.py:792-796.
    dwelling = st.dwell > 0
    requested = requested.masked_fill(dwelling[..., None], 0.0)
    dwell = st.dwell - dwelling.long()

    # Takeover uses the done flags from before this step and overrides dwell, :800-816.
    to_home = self._home - st.q
    takeover = st.target_done.all(1, keepdim=True) & (
      to_home.abs() < cfg.return_to_start_threshold).all(-1)
    requested = torch.where(takeover[..., None], to_home / dt, requested)

    # The joint-limit clip leaves velocity untouched and velocity is not clipped to
    # 2.175, base_env.py:212-231.
    dv = cfg.max_joint_acceleration * dt
    qd_new = torch.minimum(torch.maximum(requested, st.qd - dv), st.qd + dv)
    q_new = torch.minimum(torch.maximum(st.q + qd_new * dt, self._lower), self._upper)
    q, qd, rolled = self._move(st.q, q_new, qd_new)
    tip_pos, tip_rot = self.world.tcp()

    within = targets_within(tip_pos, tip_rot, self._scene.target_pos, self._scene.target_rot, cfg)
    done, dwell = solve_targets(within, st.target_done, dwell, cfg.dwell_steps)
    score = self._score(done, q)

    rolled_count = rolled.sum(1)
    penalty = cfg.collision_penalty_per_robot * rolled_count
    step = st.step + 1
    finished = done.all(1) & (dwell == 0).all(1)
    if cfg.return_to_start_required:
      finished &= ((q - self._home).abs() < cfg.return_to_start_epsilon).all(-1).all(-1)
    terminal = (step >= cfg.max_steps) | finished

    next_robot = RobotState(
      q=q, qd=qd, dwell=dwell, tip_pos=tip_pos, tip_rot=tip_rot, target_done=done, score=score,
      step=step, aug_code=self._draw_aug(self.num_worlds))
    out = StepOut(
      next_robot=next_robot,
      reward=score - st.score - penalty,
      terminal=terminal,
      collision_robot_steps=rolled_count,
      collision_penalty=penalty,
      acceleration=(qd - st.qd).abs().mean((1, 2)),
    )
    self._robot = next_robot
    ended = terminal.nonzero().flatten()
    if len(ended):
      self.reset(ended)
    return out
