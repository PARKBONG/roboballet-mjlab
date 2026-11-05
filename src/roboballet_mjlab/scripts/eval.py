"""Evaluates a checkpoint on given sets and prints a table. Gate G4 is this script.

  CUDA_VISIBLE_DEVICES=0 uv run python -m roboballet_mjlab.scripts.eval \
    --checkpoint CHECKPOINT.pt --arm-model mesh_tool \
    --training-dist data/start_states/4r_10t_30o.pb \
    --static-paper src/roboballet_mjlab/data/scenes/4r_10t_30o_static_paper.pt

Takes a converted reference checkpoint, a trainer checkpoint or a policy snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import PolicyNet, load_policy, saved_model_cfg
from roboballet_mjlab.config import EnvCfg, WorldCfg
from roboballet_mjlab.train.evaluator import EVAL_SETS, Evaluator, load_scenes

COLUMNS = ('task_rate', 'task_rate_se', 'all_done_fraction', 'num_targets_done', 'final_score', 'episode_length',
           'total_collision_robot_steps', 'total_collision_penalty', 'total_mean_acceleration', 'worlds', 'residual_collisions')


def print_table(results: dict[str, dict[str, float]]) -> None:
  width = max(len(c) for c in COLUMNS)
  names = list(results)
  print(' ' * width, *[f'{name:>14}' for name in names])
  for column in COLUMNS:
    print(f'{column:<{width}}', *[f'{results[name][column]:>14.4f}' for name in names])


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--arm-model', default=WorldCfg.arm_model, choices=('spheres_stick', 'mesh_tool'))
  for name in EVAL_SETS:
    parser.add_argument(f'--{name.replace("_", "-")}', type=Path)
  parser.add_argument('--worlds', type=int, default=1024)
  parser.add_argument('--static-worlds', type=int, default=80)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--no-compile', action='store_true')
  parser.add_argument('--json', type=Path, help='Also write the results here.')
  args = parser.parse_args()

  device = 'cuda'
  torch.set_float32_matmul_precision('highest')
  sets = {name: getattr(args, name) for name in EVAL_SETS}
  first = next((path for path in sets.values() if path is not None), None)
  if first is None:
    parser.error('name at least one set')
  loaded = {name: load_scenes(path, device) if path is not None else None for name, path in sets.items()}
  shape = next(scene for scene in loaded.values() if scene is not None)
  world = WorldCfg(num_robots=shape.base_pos.shape[1], num_obstacles=shape.box_pos.shape[1], arm_model=args.arm_model)
  env_cfg = EnvCfg(world=world, num_targets=shape.target_pos.shape[1])
  model_cfg = saved_model_cfg(str(args.checkpoint))
  topo = Topology(env_cfg, 'cpu')
  policy = PolicyNet(model_cfg, topo)
  step = load_policy(str(args.checkpoint), policy)
  print(f'{args.checkpoint}: grad step {step}, arm {args.arm_model}')
  evaluator = Evaluator(env_cfg, model_cfg, loaded, device, args.seed, args.worlds, args.static_worlds,
                        compile=not args.no_compile)
  results = evaluator.evaluate(policy.to(device), step)
  print_table(results)
  if args.json:
    args.json.write_text(json.dumps({'checkpoint': str(args.checkpoint), 'arm_model': args.arm_model, 'step': step, 'results': results}, indent=2))


if __name__ == '__main__':
  main()
