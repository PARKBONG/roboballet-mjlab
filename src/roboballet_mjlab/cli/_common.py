"""Shared by the commands. Nothing here imports torch or the sim, so --help and --dry-run stay light."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shlex
import sys

from roboballet_mjlab.paths import DATA_ENV, data_dir
from roboballet_mjlab.tasks import TaskCfg, list_tasks, load_task

FETCH_HINT = 'Download it with: uv run ballet-fetch-data (ballet-fetch-data --help lists what it fetches).'


def add_task(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('task', metavar='TASK', choices=list_tasks(), help='One of: ' + ', '.join(list_tasks()) + '.')


def add_data_dir(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--data-dir', type=Path,
                      help=f'Where fetch-data put the downloaded files. Default: ${DATA_ENV}, else data/ in the clone.')


def task_listing() -> str:
  return 'tasks:\n' + '\n'.join(f'  {t}\n      {load_task(t).description}' for t in list_tasks())


def set_data_dir(path: Path | None) -> None:
  """Before any module that reads the data directory is imported. Child processes inherit it."""
  if path is not None:
    os.environ[DATA_ENV] = str(path.expanduser().resolve())


def check_files(task: TaskCfg, eval_names: list[str] | None = None) -> None:
  """Exits on missing training scenes. A missing evaluation set is only reported: the evaluator skips it."""
  _, source = task.train_source()
  if not source.exists():
    raise SystemExit(f'{task.task_id}: training scenes not found at {source}. {FETCH_HINT}')
  for name, path in task.eval_paths().items():
    if (eval_names is None or name in eval_names) and not path.exists():
      print(f'[WARN] evaluation set {name}: {path} not found, it will be skipped. {FETCH_HINT}', file=sys.stderr)


def run_module(module: str, argv: list[str], dry_run: bool = False) -> int:
  """Runs `module`'s main() as if started with `python -m module *argv`, in this process."""
  print(f'[INFO] data directory: {data_dir()}', flush=True)
  print(f'[INFO] {sys.executable} -m {module} {shlex.join(argv)}', flush=True)
  if dry_run:
    return 0
  target = importlib.import_module(module)
  sys.argv = [module, *argv]
  return target.main() or 0
