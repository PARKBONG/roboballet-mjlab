# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""TensorBoard and CSV always, W&B only on request. DESIGN D12. Also the run's provenance record.

Nothing here reads credentials or environment variables that hold them. wandb looks after its
own login, and only when --wandb-project asks for it.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[2]
_PACKAGES = ('torch', 'warp-lang', 'mujoco', 'mujoco-warp', 'mjlab', 'numpy', 'protobuf', 'tensorboard')


class MetricLogger:
  """Scalars by tag and step, so curves overlay the reference's TensorBoard files."""

  def __init__(self, run_dir: Path, wandb_project: str | None = None, config: dict | None = None) -> None:
    self.run_dir = Path(run_dir)
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self._tb = SummaryWriter(log_dir=str(self.run_dir / 'tb'))
    path = self.run_dir / 'metrics.csv'
    new = not path.exists()
    self._file = open(path, 'a', newline='')
    self._csv = csv.writer(self._file)
    if new:
      self._csv.writerow(['step', 'tag', 'value', 'wall_time'])
    self._wandb = None
    if wandb_project:
      import wandb

      self._wandb = wandb.init(project=wandb_project, dir=str(self.run_dir), name=self.run_dir.name, config=config)

  def scalars(self, values: dict[str, float], step: int) -> None:
    now = time.time()
    for tag, value in values.items():
      self._tb.add_scalar(tag, value, global_step=step, walltime=now)
      self._csv.writerow([step, tag, value, now])
    if self._wandb is not None and values:
      self._wandb.log({**values, 'grad_step': step})

  def notice(self, message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)

  def flush(self) -> None:
    self._tb.flush()
    self._file.flush()

  def close(self) -> None:
    self.flush()
    self._tb.close()
    self._file.close()
    if self._wandb is not None:
      self._wandb.finish()


def _git(*args: str) -> str:
  """Read-only. --no-optional-locks keeps git from even refreshing the index's stat cache."""
  try:
    done = subprocess.run(['git', '--no-optional-locks', *args], cwd=REPO, capture_output=True, text=True, timeout=120)
  except (OSError, subprocess.SubprocessError) as error:
    return f'git failed: {error}\n'
  return done.stdout if done.returncode == 0 else f'git failed: {done.stderr}\n'


def _versions() -> dict[str, str]:
  versions = {}
  for name in _PACKAGES:
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = 'absent'
  return versions


def as_plain(config: Any) -> Any:
  """Dataclasses to dicts, for JSON and for comparing a checkpoint's config on resume."""
  return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config


def write_provenance(run_dir: Path, config: dict[str, Any], device: str | torch.device) -> Path:
  """What this run was: full config, code revision and uncommitted patch, machine, versions.

  Each start gets its own directory, so a resumed run keeps the record of every part of it.
  """
  stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
  out = Path(run_dir) / 'provenance' / f'{stamp}_{os.getpid()}'
  out.mkdir(parents=True, exist_ok=True)
  # The first launch stays: a resumed run counts from when training began.
  launch = Path(run_dir) / 'launch_time'
  if not launch.exists():
    launch.write_text(str(psutil.Process().create_time()))
  (out / 'config.json').write_text(json.dumps({k: as_plain(v) for k, v in config.items()}, indent=2, default=str))
  (out / 'git_revision.txt').write_text(_git('rev-parse', 'HEAD'))
  (out / 'git_diff.patch').write_text(_git('diff', 'HEAD', '--', 'roboballet_mjlab/'))
  (out / 'git_untracked.txt').write_text(_git('ls-files', '--others', '--exclude-standard', '--', 'roboballet_mjlab/'))
  on_gpu = torch.device(device).type == 'cuda' and torch.cuda.is_available()
  (out / 'system.json').write_text(json.dumps({
    'hostname': socket.gethostname(),
    'gpu_model': torch.cuda.get_device_name(torch.device(device)) if on_gpu else None,
    'cuda_runtime': torch.version.cuda,
    'python': platform.python_version(),
    'packages': _versions(),
    'argv': sys.argv,
  }, indent=2))
  return out
