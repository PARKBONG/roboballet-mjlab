"""Evaluates a checkpoint on a task's evaluation sets and prints a table: eval TASK --checkpoint PATH.

Runs roboballet_mjlab.scripts.eval with the task's arm model and sets. It takes a trainer checkpoint
of any agent, a policy snapshot from policies/, or a converted checkpoint of the original, and reads
the model's size from the file. training_dist is the task's training scenes.

  uv run ballet-eval RoboBallet-4R10T-Mesh --checkpoint logs/RoboBallet-4R10T-Mesh/fasttd3/DATE/checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from roboballet_mjlab.cli._common import FETCH_HINT, add_data_dir, add_task, run_module, set_data_dir, task_listing
from roboballet_mjlab.tasks import TaskCfg, load_task

EVAL_MODULE = 'roboballet_mjlab.scripts.eval'
SET_NAMES = ('training_dist', 'heldout', 'static_dist', 'static_paper')  # train/evaluator.py's EVAL_SETS.


def add_eval_options(p: argparse.ArgumentParser) -> None:
  p.add_argument('--checkpoint', type=Path, required=True)
  p.add_argument('--sets', nargs='+', choices=SET_NAMES, help="Default: all of the task's sets.")
  p.add_argument('--worlds', type=int, help='Episodes per random set. Default: scripts/eval.py\'s, 1024.')
  p.add_argument('--static-worlds', type=int, help='Episodes per static set. Default: scripts/eval.py\'s, 80.')
  p.add_argument('--seed', type=int)
  p.add_argument('--no-compile', action='store_true')
  p.add_argument('--json', type=Path, help='Also write the results here.')
  add_data_dir(p)
  p.add_argument('--dry-run', action='store_true', help='Print the eval module\'s command line and stop.')


def chosen_sets(task: TaskCfg, args: argparse.Namespace, check: bool) -> dict[str, Path]:
  """The task's sets, or those of --sets. With `check`, a set asked for by name must exist, and of
  the default ones a missing set is left out with a warning, since scripts/eval.py would stop on it."""
  _, source = task.train_source()
  sets = {'training_dist': source, **task.eval_paths()}
  if args.sets:
    unknown = sorted(set(args.sets) - set(sets))
    if unknown:
      raise SystemExit(f'{task.task_id} has no evaluation set {", ".join(unknown)}; it has {", ".join(sets)}.')
    sets = {name: sets[name] for name in args.sets}
  if check:
    if not args.checkpoint.is_file():
      raise SystemExit(f'checkpoint not found: {args.checkpoint}')
    for name, path in list(sets.items()):
      if not path.exists():
        if args.sets:
          raise SystemExit(f'evaluation set {name}: {path} not found. {FETCH_HINT}')
        print(f'[WARN] evaluation set {name}: {path} not found, left out. {FETCH_HINT}', file=sys.stderr)
        del sets[name]
    if not sets:
      raise SystemExit(f'none of the evaluation sets of {task.task_id} is there. {FETCH_HINT}')
  return sets


def eval_argv(task: TaskCfg, args: argparse.Namespace, sets: dict[str, Path]) -> list[str]:
  argv = ['--checkpoint', str(args.checkpoint), '--arm-model', task.arm_model]
  for name, path in sets.items():
    argv += [f'--{name.replace("_", "-")}', str(path)]
  for flag, value in (('--worlds', args.worlds), ('--static-worlds', args.static_worlds), ('--seed', args.seed),
                      ('--json', args.json)):
    if value is not None:
      argv += [flag, str(value)]
  if args.no_compile:
    argv.append('--no-compile')
  return argv


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog='ballet-eval', description=__doc__, epilog=task_listing(),
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  add_task(p)
  add_eval_options(p)
  args = p.parse_args(sys.argv[1:] if argv is None else argv)
  set_data_dir(args.data_dir)
  task = load_task(args.task)
  sets = chosen_sets(task, args, check=not args.dry_run)
  return run_module(EVAL_MODULE, eval_argv(task, args, sets), args.dry_run)


if __name__ == '__main__':
  raise SystemExit(main())
