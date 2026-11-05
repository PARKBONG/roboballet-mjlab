"""Gate G2, E2 and E3: replay goldens of the original env, paper_baseline 10.1."""

from __future__ import annotations

import numpy as np
import torch

from roboballet_mjlab.config import HOME_QPOS
from roboballet_mjlab.env.ballet_env import BalletEnv, solve_targets, targets_within
from roboballet_mjlab.structs import Scene
from tests.test_sim_common import DEVICE, GOLDEN, env_cfg, stored_scenes

Z = np.load(GOLDEN / 'env_nocollision_her_4r10t.npz')
NUM_EPISODES, NUM_STEPS = Z['a'].shape[:2]


def _tensor(x, dtype=torch.float32):
  return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def _pool() -> Scene:
  place = Z['placements']
  return Scene(
    base_pos=_tensor(place[:, :, :3, 3]),
    base_rot=_tensor(place[:, :, :3, :3]),
    box_pos=_tensor(Z['box_pos']),
    box_rot=_tensor(Z['box_rot']),
    box_spans=_tensor(Z['box_spans']),
    target_pos=_tensor(Z['target_pos']),
    target_rot=_tensor(Z['target_rot']),
    q0=_tensor(HOME_QPOS).repeat(NUM_EPISODES, 4, 1),
  )


def test_e2_replay_of_no_collision_golden():
  # mesh_tool gives the original TCP. Goldens used it, with _robots_in_collision stubbed.
  cfg = env_cfg(arm='mesh_tool', collisions_enabled=False, rollback='deterministic')
  env = BalletEnv(cfg, NUM_EPISODES, _pool(), DEVICE, seed=0)
  env.reset(scene_ids=torch.arange(NUM_EPISODES, device=DEVICE))
  worst = dict(q=0.0, qd=0.0, reward=0.0, score=0.0, tip=0.0)
  solves = 0
  for t in range(NUM_STEPS):
    out = env.step(_tensor(Z['a'][:, t]) / cfg.max_joint_velocity)
    robot = out.next_robot
    worst['q'] = max(worst['q'], (robot.q - _tensor(Z['q'][:, t])).abs().max().item())
    worst['qd'] = max(worst['qd'], (robot.qd - _tensor(Z['qd'][:, t])).abs().max().item())
    gold_r = torch.as_tensor(Z['r'][:, t], dtype=torch.float64, device=DEVICE)
    worst['reward'] = max(worst['reward'], (out.reward.double() - gold_r).abs().max().item())
    gold_s = torch.as_tensor(Z['score'][:, t], dtype=torch.float64, device=DEVICE)
    worst['score'] = max(worst['score'], (robot.score.double() - gold_s).abs().max().item())
    worst['tip'] = max(worst['tip'], (robot.tip_pos - _tensor(Z['tip'][:, t, :, :3, 3])).abs().max().item())
    # Identical, not close.
    assert torch.equal(robot.target_done, _tensor(Z['done'][:, t], torch.bool)), f'done at step {t}'
    dwell = torch.as_tensor(np.rint(Z['dwell'][:, t] * cfg.dwell_steps), device=DEVICE).long()
    assert torch.equal(robot.dwell, dwell), f'dwell at step {t}'
    assert torch.equal(robot.step, _tensor(Z['step'][:, t], torch.long)), f'step at {t}'
    assert out.terminal.all() == (t == NUM_STEPS - 1), f'terminal at step {t}'
    assert out.collision_robot_steps.sum() == 0
    solves = robot.target_done.sum().item()
  print(f'E2 worst errors {worst}, targets solved at the end {solves}')
  assert worst['q'] <= 1e-4 and worst['qd'] <= 1e-4 and worst['reward'] <= 1e-6
  assert worst['tip'] <= 1e-4


def test_e3_hindsight_return_and_solved_counts():
  """The original replays stored poses through its scoring, skipping _move_robots."""
  cfg = env_cfg(arm='mesh_tool')
  target_pos, target_rot = _tensor(Z['h_target_pos']), _tensor(Z['h_target_rot'])
  tips = _tensor(Z['tip'])
  done = torch.zeros(NUM_EPISODES, 10, dtype=torch.bool, device=DEVICE)
  dwell = torch.zeros(NUM_EPISODES, 4, dtype=torch.long, device=DEVICE)
  score = torch.zeros(NUM_EPISODES, device=DEVICE)
  total = torch.zeros(NUM_EPISODES, dtype=torch.float64, device=DEVICE)
  assert not Z['h_done'][:, 0].any() and not Z['h_dwell'][:, 0].any()
  for t in range(NUM_STEPS):
    dwell = dwell - (dwell > 0).long()  # Before the move, even though nothing moves here.
    within = targets_within(tips[:, t, :, :3, 3], tips[:, t, :, :3, :3], target_pos, target_rot, cfg)
    done, dwell = solve_targets(within, done, dwell, cfg.dwell_steps)
    new_score = done.float().mean(1)
    # Replay row of env_spec 15: scoring only, so no collision penalty exists here.
    reward = (new_score - score).double()
    score = new_score
    total += reward
    assert torch.equal(done, _tensor(Z['h_done'][:, t + 1], torch.bool)), f'done at step {t}'
    gold_dwell = torch.as_tensor(np.rint(Z['h_dwell'][:, t + 1] * 5), device=DEVICE).long()
    assert torch.equal(dwell, gold_dwell), f'dwell at step {t}'
    gold_r = torch.as_tensor(Z['h_r'][:, t], dtype=torch.float64, device=DEVICE)
    assert (reward - gold_r).abs().max() <= 1e-6
  returns = np.round(total.cpu().numpy(), 5).tolist()
  assert returns.count(0.4) == 18 and returns.count(0.3) == 2
  assert returns == np.round(Z['h_r'].sum(1), 5).tolist()
  assert done.sum(1).cpu().tolist() == Z['h_done'][:, -1].sum(1).tolist()


def test_replay_of_mesh_collision_golden():
  """Rollback against the original code itself: deterministic_mode, meshes, first 5 stored states."""
  gold = np.load(GOLDEN / 'env_mesh_traces_4r10t.npz')
  num = gold['a'].shape[0]
  cfg = env_cfg(arm='mesh_tool', rollback='deterministic')
  pool = Scene(*[x[:num] for x in stored_scenes(4, valid=None)])
  env = BalletEnv(cfg, num, pool, DEVICE, seed=0)
  env.reset(scene_ids=torch.arange(num, device=DEVICE))
  worst_q = worst_qd = worst_r = worst_tip = 0.0
  rollbacks = 0
  for t in range(gold['a'].shape[1]):
    out = env.step(_tensor(gold['a'][:, t]) / cfg.max_joint_velocity)
    robot = out.next_robot
    want = _tensor(gold['rolled'][:, t], torch.long)
    assert torch.equal(out.collision_robot_steps, want), f'rollbacks at step {t}'
    worst_q = max(worst_q, (robot.q - _tensor(gold['q'][:, t])).abs().max().item())
    worst_qd = max(worst_qd, (robot.qd - _tensor(gold['qd'][:, t])).abs().max().item())
    worst_tip = max(worst_tip, (robot.tip_pos - _tensor(gold['tip'][:, t, :, :3, 3])).abs().max().item())
    gold_r = torch.as_tensor(gold['r'][:, t], dtype=torch.float64, device=DEVICE)
    worst_r = max(worst_r, (out.reward.double() - gold_r).abs().max().item())
    assert torch.equal(robot.target_done, _tensor(gold['done'][:, t], torch.bool))
    rollbacks += want.sum().item()
  print(f'mesh golden: {rollbacks} rollbacks identical, worst q {worst_q:.2e}, qd {worst_qd:.2e}, '
        f'reward {worst_r:.2e}, tip {worst_tip:.2e}')
  assert rollbacks == 212
  assert worst_q <= 1e-4 and worst_qd <= 1e-4 and worst_r <= 1e-6 and worst_tip <= 1e-4
