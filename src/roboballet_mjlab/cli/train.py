"""Trains an agent on a task: train TASK --agent {td3,fasttd3,sac,ppo}.

Runs the agent's own train module with the task's scenes, arm model and evaluation sets, and the
agent's preset from roboballet_mjlab/tasks. The preset goes first and your --set and --model-set
after it, so yours win. Anything after -- goes to the train module unchanged, for example
-- --no-eval --total-env-rounds 1000 (td3 counts --total-grad-steps instead).

  uv run ballet-train RoboBallet-4R10T-Mesh --agent fasttd3 --seed 1
  uv run ballet-train RoboBallet-4R10T-Mesh --agent fasttd3 --gpus 0,1,2,3
  uv run ballet-train RoboBallet-4R10T-Mesh --agent fasttd3 --run-dir logs/x --resume

SIGTERM saves a checkpoint and exits with status 0. --resume continues from it, in --run-dir.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from roboballet_mjlab.cli._common import add_data_dir, add_task, check_files, run_module, set_data_dir, task_listing
from roboballet_mjlab.tasks import AGENTS, DEFAULT_AGENT, load_task


def agent_listing() -> str:
  lines = ['agents and their presets:']
  for name, preset in AGENTS.items():
    lines.append(f'  {name}: {preset.description}')
    for flag, pairs in (('--set', preset.set), ('--model-set', preset.model_set)):
      if pairs:
        lines.append(f'      {flag} ' + f' {flag} '.join(pairs))
  return '\n'.join(lines)


def parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog='ballet-train', description=__doc__, epilog=f'{task_listing()}\n\n{agent_listing()}',
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  add_task(p)
  p.add_argument('--agent', choices=tuple(AGENTS), default=DEFAULT_AGENT, help=f'Default: {DEFAULT_AGENT}, the main agent.')
  p.add_argument('--run-dir', type=Path, help='Default: logs/TASK/AGENT/DATE_TIME. Required with --resume.')
  p.add_argument('--seed', type=int)
  p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help="Override a field of the agent's config.")
  p.add_argument('--model-set', action='append', default=[], metavar='KEY=VALUE',
                 help='Override a ModelCfg field. Not td3, which is the original at its own model.')
  p.add_argument('--gpus', help='Comma-separated physical GPUs, one process per entry. fasttd3 and sac only.')
  p.add_argument('--stop-after-seconds', type=float)
  p.add_argument('--resume', nargs='?', const='', default=None, metavar='CHECKPOINT',
                 help='Continue from a checkpoint, by default the one in --run-dir.')
  p.add_argument('--wandb-project', help='Also log to this W&B project. Credentials are wandb\'s own business.')
  add_data_dir(p)
  p.add_argument('--dry-run', action='store_true', help='Print the train module\'s command line and stop.')
  return p


def build_argv(args: argparse.Namespace, extra: list[str]) -> tuple[str, list[str]]:
  task, preset = load_task(args.task), AGENTS[args.agent]
  if args.model_set and not preset.takes_model_set:
    raise SystemExit(f'{args.agent} takes no --model-set: it is the original TD3 at its own ModelCfg.')
  if args.gpus and not preset.takes_gpus:
    raise SystemExit(f'{args.agent} takes no --gpus: only fasttd3 and sac run several processes.')
  if args.resume is not None and args.run_dir is None:
    raise SystemExit('--resume needs --run-dir, the run to continue.')
  run_dir = args.run_dir or Path('logs') / args.task / args.agent / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  flag, source = task.train_source()
  argv = ['--run-dir', str(run_dir), '--arm-model', task.arm_model, flag, str(source)]
  for name, path in task.eval_paths().items():
    argv += [f'--eval-{name.replace("_", "-")}', str(path)]
  if args.seed is not None:
    argv += ['--seed', str(args.seed)]
  for pair in (*preset.set, *args.set):
    argv += ['--set', pair]
  for pair in (*preset.model_set, *args.model_set):
    argv += ['--model-set', pair]
  if args.gpus:
    argv += ['--gpus', args.gpus]
  if args.stop_after_seconds is not None:
    argv += ['--stop-after-seconds', str(args.stop_after_seconds)]
  if args.resume is not None:
    argv += ['--resume'] + ([args.resume] if args.resume else [])
  if args.wandb_project:
    argv += ['--wandb-project', args.wandb_project]
  return preset.module, argv + extra


def main(argv: list[str] | None = None) -> int:
  argv = sys.argv[1:] if argv is None else argv
  own, extra = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
  args = parser().parse_args(own)
  set_data_dir(args.data_dir)
  module, module_argv = build_argv(args, extra)
  if not args.dry_run:
    check_files(load_task(args.task))
  return run_module(module, module_argv, args.dry_run)


if __name__ == '__main__':
  raise SystemExit(main())
