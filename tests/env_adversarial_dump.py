"""Original env side of test_env_adversarial.py: fuzzed scenes, actions and the resulting trace.

Runs in ../venv, needs numpy, scipy and the original package, no torch:
  cd $ROBOBALLET_PROJECT  # holds roboballet/ and venv/
  PYTHONPATH=. venv/bin/python -B roboballet-mjlab/tests/env_adversarial_dump.py CASE SEED OUT

Collisions are off, or replaced by a rule on q that both sides evaluate on the same float32
numbers, so only the task logic and the kinematics can differ.
"""

from __future__ import annotations

import sys
import zlib

import numpy as np

HOME = np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0], dtype=np.float32)
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
ANGLE = 15.0 / 180.0 * np.pi
SOLO, PAIR, SPREAD = -0.4, 0.5, 0.6  # Synthetic collision rule, free within 0.2 of home.
FAR = 9.0  # Metres, out of reach of every target check.

_BASE = dict(R=4, T=10, W=24, steps=700, dt=0.1, dwell=0.5, weight=0.0, required=False, thr=0.25,
             q0='home', bases='rails', actions=('jitter', 'hold', 'bang', 'small'), seek=False,
             rule=False, placed_by=0.6, nconmax=512)
CASES = {
  'reference': dict(),
  'odd_bases_far_q0': dict(bases='any', q0='far', T=6),
  'few_targets_many_resets': dict(T=2, placed_by=0.2),
  'ties_on_stacked_bases': dict(T=6, bases='stacked', actions=('mirror',), nconmax=4096, W=12, steps=400),
  'no_dwell': dict(T=6, dwell=0.0, placed_by=0.3),
  'coarse_step': dict(T=4, dt=0.5, dwell=1.0, steps=300),
  'fine_step': dict(T=3, dt=0.02, dwell=0.03, steps=1500),
  'ceil_dwell': dict(T=5, dwell=0.3),
  'rts_required': dict(T=4, weight=0.25, required=True, q0='near', seek=True, placed_by=0.2),
  'rts_weight_only': dict(T=4, weight=0.25, q0='near', seek=True, placed_by=0.2, thr=0.1),
  'required_no_weight': dict(T=4, required=True, q0='near', seek=True, placed_by=0.2),
  'takeover_no_rts': dict(T=3, q0='near', actions=('small',), placed_by=0.1),
  'eight_robots': dict(R=8, W=12, steps=400),
  'rule': dict(T=6, rule=True, q0='near', actions=('hold', 'bang')),
  'rule_eight_robots': dict(R=8, T=6, rule=True, q0='near', actions=('hold', 'bang'), W=12, steps=400),
  'rule_rts': dict(T=4, weight=0.25, required=True, rule=True, q0='near', seek=True, placed_by=0.2),
}


def case_params(case: str) -> dict:
  return {**_BASE, **CASES[case]}


def rule_flags(q: np.ndarray) -> np.ndarray:
  """[R] bool from float32 q [R, 7]. Per robot and per pair, like real contacts."""
  a, b = q[:, 0], q[:, 2]
  pair = (a[:, None] + a[None]) > PAIR
  spread = (b[:, None] - b[None]) > SPREAD
  np.fill_diagonal(pair, False)
  hit = pair | spread
  return (q[:, 1] > SOLO) | hit.any(0) | hit.any(1)


def sample_actions(rng, mode: str, n: int, R: int) -> np.ndarray:
  if mode == 'jitter':
    a = rng.uniform(-1, 1, (n, R, 7))
  elif mode == 'small':
    a = rng.uniform(-0.03, 0.03, (n, R, 7))
  else:  # Piecewise constant, so the arms travel and hit the joint limits.
    longest = 60 if mode == 'bang' else 30
    a = np.zeros((n, R, 7))
    for r in range(R):
      for j in range(7):
        t = 0
        while t < n:
          hold = rng.integers(2, longest)
          a[t:t + hold, r, j] = rng.choice([-1.0, 1.0]) if mode == 'bang' else rng.uniform(-1, 1)
          t += hold
  if mode == 'mirror':  # Arms on one base share actions, so their tips tie.
    a[:, rng.integers(0, np.arange(R) + 1)] = a.copy()
  return a.astype(np.float32)


def perturb(tcp: np.ndarray, rng, rotation_cls) -> tuple[np.ndarray, np.ndarray]:
  """A pose near `tcp`: mostly inside the tolerance, often right at its boundary."""
  def pick(inside, at, wide, tight):
    u = rng.random()
    if u < 0.6:
      return rng.uniform(0, inside)
    width = wide if u < 0.85 else tight
    return at + rng.uniform(-width, width)
  dist = pick(0.024, 0.025, 1e-3, 1e-5)
  angle = pick(14 / 180 * np.pi, ANGLE, 0.01, 2e-5)
  unit = rng.normal(size=(2, 3))
  unit /= np.linalg.norm(unit, axis=1, keepdims=True)
  rot = tcp[:3, :3] @ rotation_cls.from_rotvec(unit[1] * angle).as_matrix()
  return (tcp[:3, 3] + dist * unit[0]).astype(np.float32), rot.astype(np.float32)


def main(case: str, seed: int, out: str) -> None:
  from scipy.spatial.transform import Rotation

  from roboballet.environments import base_env, planning_env, target

  p = case_params(case)
  R, T, W, dt = p['R'], p['T'], p['W'], p['dt']
  rng = np.random.default_rng([seed, zlib.crc32(case.encode())])  # Cases must not share scenes.

  world = 8 if R > 4 else 4
  extra = {'robot_use_order': ['panda1', 'panda2', 'panda3', 'panda4']} if world == 4 else {}
  base = base_env.BaseEnvConfig(
    world_file_path=f'mujoco_world/{world}_pandas_world.xml', episode_timeout_s=1.0 * T + 2.0,
    timestep_s=dt, num_robots=R, deterministic_mode=True, excluded_objects=[], **extra)
  config = planning_env.PlanningEnvConfig(
    base_config=base, num_obstacles_to_generate=0, num_targets_to_generate=T, dwell_time_s=p['dwell'],
    return_to_start_reward_weight=p['weight'], return_to_start_required=p['required'],
    return_to_start_joint_config_diff_threshold=p['thr'])

  class Env(planning_env.PlanningEnv):
    def _robots_in_collision(self):
      if not p['rule']:
        return set()
      q = np.stack([self._world.GetJointValues(r) for r in self._robots]).astype(np.float32)
      return {r for r, hit in zip(self._robots, rule_flags(q)) if hit}

  planner = Env(config)
  planner._score_and_update_all_targets = lambda: 0.0
  robots = planner._robots
  assert np.array_equal(np.array(planner._start_joint_config, dtype=np.float32)[0], HOME)
  assert np.array_equal(planner._joint_lower_limits[0], LOWER)
  assert np.array_equal(planner._joint_upper_limits[0], UPPER)
  defaults = [planner._world.GetWorldTransform(r).astype(np.float64) for r in robots]

  def state_of(scene, targets) -> planning_env.StartState:
    return planning_env.StartState(
      generated_obstacles=[],
      targets=[target.TargetInfo(position=a, rotation=b) for a, b in targets],
      initial_joint_configurations=scene['q0'], initial_robot_placements=scene['placements'])

  def sample_scene() -> dict:
    place = np.stack(defaults)
    if p['bases'] == 'stacked':
      place[:] = place[0]
    place[:, 1, 3] = rng.uniform(-0.8, 0.8, R) if p['bases'] != 'stacked' else rng.uniform(-0.8, 0.8)
    if p['bases'] == 'any':  # Every rotation code path of rot_to_quat, 180 degrees included.
      for i in range(R):
        place[i, :3, :3] = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
        if rng.random() < 0.4:
          half_turn = np.pi * rng.choice([-1, 1]) * np.eye(3)[rng.integers(3)]
          place[i, :3, :3] = Rotation.from_rotvec(half_turn).as_matrix()
        place[i, :3, 3] = rng.uniform(-1, 1, 3)
    q0 = np.tile(HOME, (R, 1))
    if p['q0'] == 'near':
      keep = rng.random(R) < 0.5
      q0 = q0 + (rng.uniform(-0.2, 0.2, (R, 7)) * keep[:, None]).astype(np.float32)
    elif p['q0'] == 'far':
      q0 = rng.uniform(LOWER, UPPER, (R, 7)).astype(np.float32)
    mode = rng.choice(p['actions'])
    return dict(placements=place.astype(np.float32), q0=np.clip(q0, LOWER, UPPER), mode=mode,
                actions=sample_actions(rng, mode, planner._max_steps, R),
                seek=p['seek'] and rng.random() < 0.8)

  def place_targets(scene) -> list:
    """Walks the scene's actions with scoring stubbed and drops targets on the tips as it goes."""
    far = (np.full(3, FAR, dtype=np.float32), np.eye(3, dtype=np.float32))
    planner._start_state = state_of(scene, [far] * T)
    planner.reset()
    picked, last, rate = [], None, T / (R * p['placed_by'] * planner._max_steps)
    for t in range(planner._max_steps):
      planner.step(scene['actions'][t] * np.float32(2.175))
      for r in rng.permutation(R):
        if len(picked) < T and rng.random() < rate and planner._dwell_steps_left_by_robot[r] == 0:
          if last is None or rng.random() > 0.2:  # Otherwise a second target on the same pose.
            last = perturb(planner._world.GetRobotTcpTransform(robots[r]), rng, Rotation)
          picked.append(last)
          planner._dwell_steps_left_by_robot[r] = planner._dwell_steps
    while len(picked) < T:
      picked.append(perturb(planner._world.GetRobotTcpTransform(robots[rng.integers(R)]), rng, Rotation))
    if rng.random() < 0.15:  # An impossible target: this episode can only time out.
      picked[rng.integers(T)] = far
    return picked

  pool = []
  def start(env) -> None:
    scene = sample_scene()
    scene['targets'] = place_targets(scene)
    pool.append(scene)
    env._start_state = state_of(scene, scene['targets'])
    env.reset()
    env.scene_id, env.t = len(pool) - 1, 0

  envs = [Env(config) for _ in range(W)]
  for env in envs:
    start(env)

  steps = p['steps']
  shape = (W, steps)
  rec = dict(
    a=np.zeros((*shape, R, 7), dtype=np.float32), q=np.zeros((*shape, R, 7), dtype=np.float32),
    qd=np.zeros((*shape, R, 7), dtype=np.float32), tip=np.zeros((*shape, R, 4, 4)), r=np.zeros(shape),
    score=np.zeros(shape), accel=np.zeros(shape), done=np.zeros((*shape, T), dtype=bool),
    dwell=np.zeros((*shape, R), dtype=np.int64), step=np.zeros(shape, dtype=np.int64),
    terminal=np.zeros(shape, dtype=bool), rolled=np.zeros(shape, dtype=np.int64),
    penalty=np.zeros(shape), scene=np.zeros((W, steps + 1), dtype=np.int64))
  rec['scene'][:, 0] = [e.scene_id for e in envs]

  for t in range(steps):
    for w, env in enumerate(envs):
      scene = pool[env.scene_id]
      unit = scene['actions'][env.t]
      if scene['seek'] and env.num_targets_done() == T:  # Heads home once everything is solved.
        q = env.joint_configurations()
        unit = np.clip((HOME - q) / np.float32(dt * 2.175), -1, 1).astype(np.float32)
      rec['a'][w, t] = unit
      before_accel, before_penalty = env.total_acceleration(), env.total_collision_penalty()
      timestep = env.step(unit * np.float32(2.175))
      obs = timestep.observation
      rec['q'][w, t] = obs['robots']['joint_configurations']
      rec['qd'][w, t] = obs['robots']['joint_velocities']
      rec['tip'][w, t] = [env._world.GetRobotTcpTransform(r) for r in robots]
      rec['r'][w, t], rec['score'][w, t] = timestep.reward, env.current_score()
      rec['accel'][w, t] = env.total_acceleration() - before_accel
      rec['penalty'][w, t] = env.total_collision_penalty() - before_penalty
      rec['done'][w, t], rec['dwell'][w, t] = obs['targets']['done'], env._dwell_steps_left_by_robot
      rec['step'][w, t], rec['terminal'][w, t] = obs['step'], env.terminal()
      rec['rolled'][w, t] = len(env._robots_in_collision_last_step)
      env.t += 1
      if env.terminal():
        start(env)
      rec['scene'][w, t + 1] = env.scene_id

  n = len(pool)
  np.savez(
    out, **rec,
    max_steps=planner._max_steps, dwell_steps=planner._dwell_steps,
    base_pos=np.stack([s['placements'][:, :3, 3] for s in pool]),
    base_rot=np.stack([s['placements'][:, :3, :3] for s in pool]),
    box_pos=np.full((n, 1, 3), 100.0, dtype=np.float32),
    box_rot=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1, 1)),
    box_spans=np.full((n, 1, 3), 0.1, dtype=np.float32),
    target_pos=np.array([[a for a, _ in s['targets']] for s in pool]),
    target_rot=np.array([[b for _, b in s['targets']] for s in pool]),
    q0=np.stack([s['q0'] for s in pool]))
  print(f'{case}: {W * steps} steps, {n} episodes, {rec["done"][:, -1].sum()} done at the end')


def extras(seed: int, out: str) -> None:
  """Tables for single functions: augmentation, solve check, score, random rollback."""
  import random

  from scipy.spatial.transform import Rotation

  from roboballet.environments import base_env, planning_env, target

  rng = np.random.default_rng(seed)
  random.seed(seed)
  R, T = 4, 5
  base = base_env.BaseEnvConfig(
    world_file_path='mujoco_world/4_pandas_world.xml', episode_timeout_s=7.0, timestep_s=0.1, num_robots=R,
    robot_use_order=['panda1', 'panda2', 'panda3', 'panda4'], excluded_objects=[])
  config = planning_env.PlanningEnvConfig(
    base_config=base, num_obstacles_to_generate=0, num_targets_to_generate=T,
    return_to_start_reward_weight=0.3,
    return_to_start_joint_config_diff_threshold=0.25)

  class Env(planning_env.PlanningEnv):
    def _robots_in_collision(self):
      q = np.stack([self._world.GetJointValues(r) for r in self._robots]).astype(np.float32)
      return {r for r, hit in zip(self._robots, rule_flags(q)) if hit}

  env = Env(config)
  far = target.TargetInfo(position=np.full(3, FAR), rotation=np.eye(3))
  env._start_state = planning_env.StartState(
    [], [far] * T, np.tile(HOME, (R, 1)), np.stack([env._world.GetWorldTransform(r) for r in env._robots]))
  env.reset()
  table = {}

  # BoxInfo.get_augmentation for every code.
  rots, spans = Rotation.random(64, random_state=seed).as_matrix(), rng.uniform(0.02, 0.6, (64, 3))
  boxes = [planning_env.BoxInfo(spans=s, centroid=np.zeros(3), rotation=r) for s, r in zip(spans, rots)]
  table['box_rot'], table['box_spans'] = rots, spans
  table['aug_rot'] = np.array([[b.get_augmentation(c).rotation for c in range(4)] for b in boxes])
  table['aug_spans'] = np.array([[b.get_augmentation(c).spans for c in range(4)] for b in boxes])

  # in_solution_threshold around its boundary, at every scale down to rounding.
  n = 60000
  tip_pos, tip_rot = rng.uniform(-1, 1, (n, 3)), Rotation.random(n, random_state=seed + 1).as_matrix()
  units = rng.normal(size=(n, 2, 3))
  units /= np.linalg.norm(units, axis=-1, keepdims=True)
  scales = 10.0 ** rng.uniform(-7, -2, (n, 2))
  dist = 0.025 + scales[:, 0] * rng.choice([-1, 1], n)
  angle = ANGLE + scales[:, 1] * rng.choice([-1, 1], n)
  pos = (tip_pos + dist[:, None] * units[:, 0]).astype(np.float32)
  rot = (tip_rot @ Rotation.from_rotvec(angle[:, None] * units[:, 1]).as_matrix()).astype(np.float32)
  scoring = target.TargetScoringConfig()
  pose = np.eye(4)
  solved = np.zeros(n, dtype=bool)
  for i in range(n):
    pose[:3, :3], pose[:3, 3] = tip_rot[i], tip_pos[i]
    goal = target.TargetInfo(position=pos[i], rotation=rot[i])
    solved[i] = target.in_solution_threshold(pose, goal, scoring)
  table.update(tip_pos=tip_pos, tip_rot=tip_rot, target_pos=pos, target_rot=rot, solved=solved)

  # The score, RTS part included, over the done flags and q.
  m = 4000
  around = rng.choice([0.05, 0.2, 0.25, 0.6, 2.0], (m, R, 1))
  score_q = np.clip(HOME + rng.uniform(-1, 1, (m, R, 7)) * around, LOWER, UPPER).astype(np.float32)
  score_done = rng.random((m, T)) < rng.choice([0.5, 1.0], (m, 1))
  scores = np.zeros(m)
  for i in range(m):
    env.set_joint_configurations(score_q[i])
    env._targets = [(far, target.TargetState(done=bool(d))) for d in score_done[i]]
    scores[i] = env._score_and_update_all_targets()
  table.update(score_q=score_q, score_done=score_done, scores=scores)

  # Which robots roll back, from states where several collide at once: base_env.py:239-260.
  env._config = base.replace(deterministic_mode=False)
  states, moves, outcomes = [], [], []
  while len(states) < 8:
    q = np.tile(HOME, (R, 1))
    q[:, 0] = rng.uniform(0.16, 0.25, R)
    q[:, 1] = rng.uniform(-0.46, -0.41, R)
    q[:, 2] = rng.uniform(-0.1, 0.1, R)
    unit = rng.uniform(0.02, 0.35, (R, 7)).astype(np.float32)
    unit[:, 1] *= rng.random() < 0.5
    unit[:, 1] += 1e-3
    def sample(count):
      masks = np.zeros(count, dtype=np.int64)
      for k in range(count):
        env.set_joint_configurations(q)
        env._current_joint_velocities = np.zeros((R, 7), dtype=np.float32)
        env._move_robots(unit * np.float32(2.175))
        masks[k] = sum(1 << env._robots.index(r) for r in env._robots_in_collision_last_step)
      return masks
    if rule_flags(q).any() or len(np.unique(sample(100))) < 3:
      continue
    states.append(q)
    moves.append(unit)
    outcomes.append(sample(4000))
  table.update(states=np.array(states), moves=np.array(moves), outcomes=np.array(outcomes))
  np.savez(out, **table)
  print('extras written')


if __name__ == '__main__':
  if sys.argv[1] == 'extras':
    extras(int(sys.argv[2]), sys.argv[3])
  else:
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
