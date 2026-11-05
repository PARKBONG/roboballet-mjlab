"""Trains FastTD3 on the benchmark's task. DESIGN section 12.

Detached, one GPU, from the clone:
  CUDA_VISIBLE_DEVICES=4 setsid nohup uv run python -m roboballet_mjlab.fasttd3.train \
    --run-dir runs/fasttd3_dev/smoke --train-pool data/scenes/4r_10t_30o_train.pt \
    --eval-heldout data/scenes/4r_10t_30o_heldout.pt \
    --eval-static-dist src/roboballet_mjlab/data/scenes/4r_10t_30o_static.pt \
    --eval-static-paper src/roboballet_mjlab/data/scenes/4r_10t_30o_static_paper.pt > runs/fasttd3_dev/smoke.log 2>&1 &

Several GPUs: add --gpus 0,1,2,3, leaving out CUDA_VISIBLE_DEVICES. One process per entry, each with
its own env and replay, gradients averaged. SIGTERM to this process stops all of them.

SIGTERM saves a checkpoint and exits with status 0. --resume continues from it.
--resume PATH names rank 0's file, and each other rank reads rank{r}_ beside it.
--eval-only CHECKPOINT evaluates a checkpoint or a policies/ file and prints a table. scripts/eval.py
also takes both, but only at the default ModelCfg. For any other model use this, with the same --model-set.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.agent.networks import PolicyNet
from roboballet_mjlab.config import EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.fasttd3.trainer import CHECKPOINT_NAME, Trainer, rank_file
from roboballet_mjlab.train.evaluator import Evaluator, load_scenes
from roboballet_mjlab.train.logger import MetricLogger, write_provenance

MESH_TOOL_MIN_NCONMAX = 256


def overrides(pairs: list[str], config):
  """KEY=VALUE for any field of a config dataclass. A tuple takes comma-separated integers."""
  kinds = {field.name: type(getattr(config, field.name)) for field in dataclasses.fields(config)}
  changes = {}
  for pair in pairs:
    key, value = pair.split('=', 1)
    if key not in kinds:
      raise ValueError(f'{type(config).__name__} has no field {key}')
    kind = kinds[key]
    if kind is bool:
      changes[key] = value.lower() in ('1', 'true', 'yes')
    elif kind is tuple:
      changes[key] = tuple(int(x) for x in value.split(','))
    else:
      changes[key] = kind(value)
  return dataclasses.replace(config, **changes)


def launch(gpus: list[str], argv: list[str]) -> int:
  """One process per entry, each seeing its GPU as cuda:0. Any failure ends them all, since the
  others would wait forever in an all-reduce. The same GPU twice uses gloo, for tests: NCCL refuses it."""
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
  backend = 'nccl' if len(set(gpus)) == len(gpus) else 'gloo'
  children = []
  for rank, gpu in enumerate(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, RANK=str(rank), WORLD_SIZE=str(len(gpus)), MASTER_ADDR='127.0.0.1',
               MASTER_PORT=str(port), FASTTD3_BACKEND=backend)
    children.append(subprocess.Popen([sys.executable, '-m', 'roboballet_mjlab.fasttd3.train', *argv], env=env))

  def forward(signum, frame):
    for child in children:
      child.send_signal(signum)

  signal.signal(signal.SIGTERM, forward)
  signal.signal(signal.SIGINT, forward)
  while True:
    codes = [child.poll() for child in children]
    if all(code is not None for code in codes):
      return max(abs(code) for code in codes)
    if any(code not in (None, 0) for code in codes):
      for child in children:
        if child.poll() is None:
          child.kill()
    time.sleep(1.0)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--seed', type=int, default=FastTD3Cfg.seed)
  parser.add_argument('--total-env-rounds', type=int, default=FastTD3Cfg.total_env_rounds)
  parser.add_argument('--arm-model', default=WorldCfg.arm_model, choices=('spheres_stick', 'mesh_tool'))
  pool = parser.add_mutually_exclusive_group(required=True)
  pool.add_argument('--train-pool', type=Path, help='A .pt pool from gen_scenes.py.')
  pool.add_argument('--start-states', type=Path, help='A .pb of original start states.')
  parser.add_argument('--eval-training-dist', type=Path, help='Default: the training pool, as the original.')
  parser.add_argument('--eval-heldout', type=Path)
  parser.add_argument('--eval-static-dist', type=Path)
  parser.add_argument('--eval-static-paper', type=Path)
  parser.add_argument('--no-eval', action='store_true', help='No evaluator at all, for measuring the training rate.')
  parser.add_argument('--eval-only', type=Path, metavar='CHECKPOINT', help='Evaluate this checkpoint, print a table and exit.')
  parser.add_argument('--resume', nargs='?', const='', default=None, metavar='CHECKPOINT',
                      help=f'Continue from a checkpoint, by default {CHECKPOINT_NAME} in the run directory.')
  parser.add_argument('--wandb-project', help='Also log to this W&B project. Credentials are wandb\'s own business.')
  parser.add_argument('--stop-after-seconds', type=float)
  parser.add_argument('--gpus', help='Comma-separated physical GPUs, one process per entry.')
  parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='Override a FastTD3Cfg field.')
  parser.add_argument('--model-set', action='append', default=[], metavar='KEY=VALUE', help='Override a ModelCfg field.')
  return parser.parse_args()


def evaluate_only(args, env_cfg: EnvCfg, model_cfg: ModelCfg, cfg: FastTD3Cfg, sets: dict, device: str) -> None:
  torch.set_float32_matmul_precision('highest')
  policy = PolicyNet(model_cfg, Topology(env_cfg, 'cpu'))
  state = torch.load(args.eval_only, map_location='cpu', weights_only=True)
  policy.load_state_dict(state['policy'], strict=True)
  evaluator = Evaluator(env_cfg, model_cfg, sets, device, cfg.seed, num_worlds=cfg.eval_episodes, compile=cfg.compile)
  results = evaluator.evaluate(policy.to(device), state['step'])
  print(json.dumps({'checkpoint': str(args.eval_only), 'step': state['step'], 'results': results}, indent=2))


def main() -> int:
  args = parse_args()
  if args.gpus and 'RANK' not in os.environ:
    return launch(args.gpus.split(','), sys.argv[1:])
  rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
  if world_size > 1:
    dist.init_process_group(backend=os.environ['FASTTD3_BACKEND'], init_method='env://', rank=rank, world_size=world_size)

  device = 'cuda'
  source = args.train_pool or args.start_states
  scenes = load_scenes(source, device)
  world = WorldCfg(num_robots=scenes.base_pos.shape[1], num_obstacles=scenes.box_pos.shape[1], arm_model=args.arm_model)
  if args.arm_model == 'mesh_tool' and world.nconmax < MESH_TOOL_MIN_NCONMAX:
    raise ValueError(f'mesh_tool needs nconmax of at least {MESH_TOOL_MIN_NCONMAX}, have {world.nconmax}')
  env_cfg = EnvCfg(world=world, num_targets=scenes.target_pos.shape[1])
  model_cfg = overrides(args.model_set, ModelCfg())
  cfg = overrides(args.set, FastTD3Cfg(seed=args.seed, total_env_rounds=args.total_env_rounds))
  sets = {'training_dist': args.eval_training_dist or scenes, 'heldout': args.eval_heldout,
          'static_dist': args.eval_static_dist, 'static_paper': args.eval_static_paper}
  if args.eval_only is not None:
    evaluate_only(args, env_cfg, model_cfg, cfg, sets, device)
    return 0

  logger = None
  evaluator = None
  if rank == 0:
    config = {'env': env_cfg, 'train': cfg, 'model': model_cfg, 'source': str(source), 'args': vars(args), 'world_size': world_size}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(args.run_dir, config, device)
    logger = MetricLogger(args.run_dir, args.wandb_project,
                          config={k: dataclasses.asdict(config[k]) for k in ('env', 'train', 'model')})
    logger.notice(f'{len(scenes.q0)} scenes from {source}, {cfg.num_worlds} worlds and batch {cfg.batch_size} per process, '
                  f'{world_size} processes, bf16 {cfg.bf16}, arm {args.arm_model}')
    if not args.no_eval:
      evaluator = Evaluator(env_cfg, model_cfg, sets, device, cfg.seed, num_worlds=cfg.eval_episodes,
                            compile=cfg.compile, notice=logger.notice)
  trainer = Trainer(env_cfg, cfg, model_cfg, scenes, device, args.run_dir, evaluator, logger, args.stop_after_seconds,
                    rank=rank, world_size=world_size)
  trainer.install_signal_handlers()
  if args.resume is not None:
    trainer.load_checkpoint(rank_file(args.resume, rank) if args.resume else trainer.checkpoint_path())
  try:
    status = trainer.run()
  finally:
    if rank == 0:
      (args.run_dir / 'summary.json').write_text(json.dumps(trainer.totals(), indent=2))
    trainer.logger.close()
  trainer.logger.notice(f'stopped: {status}. {trainer.totals()}')
  if world_size > 1:
    dist.destroy_process_group()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
