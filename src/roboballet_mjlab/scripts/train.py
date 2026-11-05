"""Trains the original agent in parity mode. DESIGN sections 7 and 14. Defaults are parity.

Detached, on GPU 0, from the clone:
  CUDA_VISIBLE_DEVICES=0 setsid nohup uv run python \
    -m roboballet_mjlab.scripts.train --run-dir runs/g8a_s1 --seed 1 --arm-model mesh_tool \
    --start-states data/start_states/4r_10t_30o.pb \
    --eval-static-paper src/roboballet_mjlab/data/scenes/4r_10t_30o_static_paper.pt > runs/g8a_s1.log 2>&1 &

SIGTERM saves a checkpoint and exits with status 0. --resume continues from it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from roboballet_mjlab.agent.networks import load_checkpoint
from roboballet_mjlab.config import EnvCfg, ModelCfg, TrainCfg, WorldCfg
from roboballet_mjlab.train.evaluator import Evaluator, load_scenes
from roboballet_mjlab.train.logger import MetricLogger, write_provenance
from roboballet_mjlab.train.trainer import CHECKPOINT_NAME, Trainer

MESH_TOOL_MIN_NCONMAX = 256


def parse_overrides(pairs: list[str], cfg: TrainCfg) -> TrainCfg:
  """KEY=VALUE for any TrainCfg field, for smoke runs. Nothing here is needed for parity."""
  kinds = {field.name: type(getattr(cfg, field.name)) for field in dataclasses.fields(cfg)}
  changes = {}
  for pair in pairs:
    key, value = pair.split('=', 1)
    if key not in kinds:
      raise ValueError(f'TrainCfg has no field {key}')
    kind = kinds[key]
    if kind is bool:
      changes[key] = value.lower() in ('1', 'true', 'yes')
    elif key == 'replay_max_size':
      changes[key] = int(value)
    else:
      changes[key] = kind(value)
  return dataclasses.replace(cfg, **changes)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=TrainCfg.seed)
  parser.add_argument('--total-grad-steps', type=int, default=TrainCfg.total_grad_steps)
  parser.add_argument('--arm-model', default=WorldCfg.arm_model, choices=('spheres_stick', 'mesh_tool'))
  pool = parser.add_mutually_exclusive_group(required=True)
  pool.add_argument('--train-pool', type=Path, help='A .pt pool from gen_scenes.py.')
  pool.add_argument('--start-states', type=Path, help='A .pb of original start states.')
  parser.add_argument('--eval-training-dist', type=Path, help='Default: the training pool, as the original.')
  parser.add_argument('--eval-heldout', type=Path)
  parser.add_argument('--eval-static-dist', type=Path)
  parser.add_argument('--eval-static-paper', type=Path)
  parser.add_argument('--no-eval', action='store_true', help='No evaluator at all, for measuring the training rate.')
  parser.add_argument('--resume', nargs='?', const='', default=None, metavar='CHECKPOINT',
                      help=f'Continue from a checkpoint, by default {CHECKPOINT_NAME} in the run directory.')
  parser.add_argument('--init-weights', type=Path, help='Start from these network weights, converted from an original '
                      'checkpoint or from a trainer checkpoint. Ignored with --resume.')
  parser.add_argument('--wandb-project', help='Also log to this W&B project. Credentials are wandb\'s own business.')
  parser.add_argument('--stop-after-seconds', type=float)
  parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override a TrainCfg field, for smoke runs.')
  args = parser.parse_args()

  device = 'cuda'
  source = args.train_pool or args.start_states
  scenes = load_scenes(source, device)
  world = WorldCfg(num_robots=scenes.base_pos.shape[1], num_obstacles=scenes.box_pos.shape[1], arm_model=args.arm_model)
  if args.arm_model == 'mesh_tool' and world.nconmax < MESH_TOOL_MIN_NCONMAX:
    raise ValueError(f'mesh_tool needs nconmax of at least {MESH_TOOL_MIN_NCONMAX}, have {world.nconmax}')
  env_cfg = EnvCfg(world=world, num_targets=scenes.target_pos.shape[1])
  model_cfg = ModelCfg()
  cfg = parse_overrides(args.set, TrainCfg(seed=args.seed, total_grad_steps=args.total_grad_steps))

  config = {'env': env_cfg, 'train': cfg, 'model': model_cfg, 'source': str(source), 'args': vars(args)}
  args.run_dir.mkdir(parents=True, exist_ok=True)
  write_provenance(args.run_dir, config, device)
  logger = MetricLogger(args.run_dir, args.wandb_project, config={k: dataclasses.asdict(config[k]) for k in ('env', 'train', 'model')})
  logger.notice(f'{len(scenes.q0)} scenes from {source}, {cfg.num_worlds} worlds, batch {cfg.batch_size}, '
                f'bf16 {cfg.bf16}, arm {args.arm_model}')

  evaluator = None
  if not args.no_eval:
    sets = {'training_dist': args.eval_training_dist or scenes, 'heldout': args.eval_heldout,
            'static_dist': args.eval_static_dist, 'static_paper': args.eval_static_paper}
    evaluator = Evaluator(env_cfg, model_cfg, sets, device, cfg.seed, num_worlds=cfg.eval_episodes,
                          compile=cfg.compile, notice=logger.notice)
  trainer = Trainer(env_cfg, cfg, model_cfg, scenes, device, args.run_dir, evaluator, logger, args.stop_after_seconds)
  trainer.install_signal_handlers()
  if args.resume is not None:
    trainer.load_checkpoint(args.resume or args.run_dir / CHECKPOINT_NAME)
  elif args.init_weights is not None:
    start = load_checkpoint(str(args.init_weights), trainer.td3.policy, trainer.td3.q1, trainer.td3.q2)
    trainer.td3.sync_targets()
    trainer.td3.refresh_acting()
    logger.notice(f'network weights from {args.init_weights}, its step {start}, this run starts at 0')
  try:
    status = trainer.run()
  finally:
    (args.run_dir / 'summary.json').write_text(json.dumps(trainer.totals(), indent=2))
    logger.close()
  logger.notice(f'stopped: {status}. {trainer.totals()}')


if __name__ == '__main__':
  main()
