"""Regenerates env_*.npz from the analysis artefacts. Needs numpy and MuJoCo only.

Run from the repository root: PYTHONPATH=src python -m tests.golden.env_make_goldens

env_nocollision_her_4r10t.npz: S/scripts_baseline/out/golden_nocollision_her_4r10t.pkl,
written by S/scripts_baseline/golden2.py on the original env (20 episodes, original
tool TCP, collisions disabled, plus the hindsight replay).
env_mesh_traces_4r10t.npz: S/scripts_baseline/out/golden_traces_4r10t.pkl, written by
S/scripts_baseline/env_probe.py on the original env: the first 5 start states, mesh
collisions on, deterministic_mode.
env_scenes_{4,8}r10t.npz: the first 512 start states of S/bench/states, exported from
data/start_states/*.pb, with `valid` = no sphere contact at home on C MuJoCo.
"""

from __future__ import annotations

import pickle

import numpy as np

from roboballet_mjlab.config import HOME_QPOS
from tests.test_sim_common import GOLDEN, PROJECT, CWorld

S = PROJECT / 'runs/mjlab_port_analysis'
NUM_SCENES = 512


def f32(x):
  return np.asarray(x, dtype=np.float32)


def convert_nocollision() -> None:
  gold = pickle.load(open(S / 'scripts_baseline/out/golden_nocollision_her_4r10t.pkl', 'rb'))
  out = {
    'placements': f32([g['placements'] for g in gold]),
    'box_spans': f32([[b[0] for b in g['boxes']] for g in gold]),
    'box_pos': f32([[b[1] for b in g['boxes']] for g in gold]),
    'box_rot': f32([[b[2] for b in g['boxes']] for g in gold]),
    'target_pos': f32([[t[0] for t in g['start']] for g in gold]),
    'target_rot': f32([[t[1] for t in g['start']] for g in gold]),
    'h_target_pos': f32([[t[0] for t in g['hindsight']['start_targets']] for g in gold]),
    'h_target_rot': f32([[t[1] for t in g['hindsight']['start_targets']] for g in gold]),
    'h_r': np.array([g['hindsight']['r'] for g in gold]),
    'h_done': np.array([g['hindsight']['done'] for g in gold]),
    'h_dwell': f32([g['hindsight']['dwell'] for g in gold]),
  }
  for key in ('a', 'q', 'qd', 'tip', 'dwell'):
    out[key] = f32([g['trace'][key] for g in gold])
  for key in ('r', 'done', 'step', 'score'):
    out[key] = np.array([g['trace'][key] for g in gold])
  np.savez_compressed(GOLDEN / 'env_nocollision_her_4r10t.npz', **out)


def convert_mesh_traces() -> None:
  gold = pickle.load(open(S / 'scripts_baseline/out/golden_traces_4r10t.pkl', 'rb'))
  gold = [gold[i] for i in range(len(gold))]
  out = {key: f32([[row[key] for row in g] for g in gold]) for key in ('a', 'q', 'qd', 'tip', 'dwell')}
  out['r'] = np.array([[row['r'] for row in g] for g in gold])
  out['done'] = np.array([[row['done'] for row in g] for g in gold])
  out['rolled'] = np.array([[len(row['coll']) for row in g] for g in gold])
  np.savez_compressed(GOLDEN / 'env_mesh_traces_4r10t.npz', **out)


def convert_scenes(num_robots: int) -> None:
  z = np.load(S / f'bench/states/{num_robots}r_10t_30o.npz')
  boxes, targets, place = z['boxes'][:NUM_SCENES], z['targets'][:NUM_SCENES], z['placements'][:NUM_SCENES]
  n = len(boxes)
  out = {
    'base_pos': f32(place[:, :, :3, 3]),
    'base_rot': f32(place[:, :, :3, :3]),
    'box_spans': f32(boxes[..., :3]),
    'box_pos': f32(boxes[..., 3:6]),
    'box_rot': f32(boxes[..., 6:15].reshape(n, -1, 3, 3)),
    'target_pos': f32(targets[..., :3]),
    'target_rot': f32(targets[..., 3:12].reshape(n, -1, 3, 3)),
    'q0': np.tile(f32(HOME_QPOS), (n, num_robots, 1)),
  }
  world = CWorld(num_robots, boxes.shape[1])
  valid = np.zeros(n, dtype=bool)
  for i in range(n):
    world.set_scene(out['base_pos'][i], out['base_rot'][i], out['box_pos'][i], out['box_rot'][i],
                    out['box_spans'][i])
    world.set_q(out['q0'][i])
    world.forward()
    valid[i] = not world.flags().any()
  print(f'{num_robots}r: {valid.sum()} of {n} stored scenes are free at home with spheres')
  np.savez_compressed(GOLDEN / f'env_scenes_{num_robots}r10t.npz', valid=valid, **out)


if __name__ == '__main__':
  convert_nocollision()
  convert_mesh_traces()
  convert_scenes(4)
  convert_scenes(8)
