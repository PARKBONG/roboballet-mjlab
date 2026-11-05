"""The small task of gate G11: 1 robot, 1 target, no obstacles, and what a random policy gets on it.

gen_scenes.py draws targets on boxes, so with no boxes it has nothing to draw on. Here a target is
the tip pose of a joint configuration drawn uniformly within the limits, kept if the arm is free of
the table there. So every target is reachable with this arm, and the arm starts at home.

From the repository root, writing OUT/1r_1t_0o_{train,heldout}.pt, then the random baseline:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tests.golden.fasttd3_small_task pool OUT
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tests.golden.fasttd3_small_task random OUT
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from roboballet_mjlab.config import HOME_QPOS, JOINT_LOWER, JOINT_UPPER, EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.scenes.io import Pool, save_pool
from roboballet_mjlab.scenes.sampler import Witnesses
from roboballet_mjlab.sim.world import build_world, default_bases
from roboballet_mjlab.structs import Scene
from roboballet_mjlab.train.evaluator import Evaluator

DEVICE = 'cuda:0'
WORLD = WorldCfg(num_robots=1, num_obstacles=0)
ENV = EnvCfg(world=WORLD, num_targets=1)
BATCH = 1024


def generate(count: int, seed: int) -> Pool:
  generator = torch.Generator(device=DEVICE).manual_seed(seed)
  world = build_world(WORLD, BATCH, DEVICE)
  base_pos, base_rot = default_bases(WORLD, DEVICE)
  low, high = torch.tensor(JOINT_LOWER, device=DEVICE), torch.tensor(JOINT_UPPER, device=DEVICE)
  home = torch.tensor(HOME_QPOS, device=DEVICE)
  ids = torch.arange(BATCH, device=DEVICE)
  scene = Scene(
    base_pos=base_pos.repeat(BATCH, 1, 1), base_rot=base_rot.repeat(BATCH, 1, 1, 1),
    box_pos=torch.zeros(BATCH, 0, 3, device=DEVICE), box_rot=torch.zeros(BATCH, 0, 3, 3, device=DEVICE),
    box_spans=torch.zeros(BATCH, 0, 3, device=DEVICE),
    target_pos=torch.zeros(BATCH, 1, 3, device=DEVICE), target_rot=torch.eye(3, device=DEVICE).repeat(BATCH, 1, 1, 1),
    q0=home.repeat(BATCH, 1, 1))
  world.set_scene(ids, scene)
  world.set_q(scene.q0, ids)
  world.forward()
  if world.robots_in_collision().any():
    raise RuntimeError('the arm collides at home')
  kept_pos, kept_rot, kept_q = [], [], []
  while sum(map(len, kept_pos)) < count:
    q = low + (high - low) * torch.rand(BATCH, 1, 7, generator=generator, device=DEVICE)
    world.set_q(q, ids)
    world.forward()
    free = ~world.robots_in_collision()[:, 0]
    tip_pos, tip_rot = world.tcp()
    kept_pos.append(tip_pos[free])
    kept_rot.append(tip_rot[free])
    kept_q.append(q[free])
  target_pos, target_rot, q = (torch.cat(x)[:count] for x in (kept_pos, kept_rot, kept_q))
  pool = Scene(
    base_pos=base_pos.repeat(count, 1, 1), base_rot=base_rot.repeat(count, 1, 1, 1),
    box_pos=torch.zeros(count, 0, 3, device=DEVICE), box_rot=torch.zeros(count, 0, 3, 3, device=DEVICE),
    box_spans=torch.zeros(count, 0, 3, device=DEVICE), target_pos=target_pos, target_rot=target_rot,
    q0=home.repeat(count, 1, 1))
  witnesses = Witnesses(q, torch.zeros(count, 1, dtype=torch.long, device=DEVICE))
  return Pool(pool, witnesses, {'count': count, 'seed': seed, 'made_by': 'tests/golden/fasttd3_small_task.py'})


def random_baseline(out: Path) -> dict:
  """Evaluator's own rollout, with the policy swapped for uniform actions and for noise only."""
  scenes = {'training_dist': out / '1r_1t_0o_train.pt', 'heldout': out / '1r_1t_0o_heldout.pt'}
  results = {}
  evaluator = Evaluator(ENV, ModelCfg(node_layers=2, node_size=8, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,)),
                        scenes, DEVICE, 1, num_worlds=4096, compile=False)
  generator = torch.Generator(device=DEVICE).manual_seed(1)
  shape = lambda graph: (graph.nodes.shape[0], 1, 7)
  behaviours = {
    'uniform': lambda graph: torch.rand(shape(graph), generator=generator, device=DEVICE) * 2 - 1,
    'still': lambda graph: torch.zeros(shape(graph), device=DEVICE),
    # The mean-zero policy of a fresh agent under FastTD3's own exploration, per-world scale in [0.001, 0.4].
    'noise': lambda graph: (torch.rand(graph.nodes.shape[0], 1, 1, generator=generator, device=DEVICE) * 0.399 + 0.001)
    * torch.randn(shape(graph), generator=generator, device=DEVICE),
  }
  for name, behaviour in behaviours.items():
    evaluator._forward = behaviour
    results[name] = {}
    for set_name, env in evaluator.envs.items():
      per_world = evaluator._rollout(env, None)
      results[name][set_name] = {key: value.mean().item() for key, value in per_world.items()}
  return results


def main() -> None:
  command, out = sys.argv[1], Path(sys.argv[2])
  if command == 'pool':
    out.mkdir(parents=True, exist_ok=True)
    save_pool(out / '1r_1t_0o_train.pt', generate(20_000, 1))
    save_pool(out / '1r_1t_0o_heldout.pt', generate(2_000, 2))
    print('wrote', out)
  else:
    print(json.dumps(random_baseline(out), indent=2))


if __name__ == '__main__':
  main()
