"""Plays a checkpoint on a task: play TASK --checkpoint PATH [--video OUT.mp4].

Evaluates the checkpoint on the task's sets as eval does, then, with --video, records one episode
through roboballet_mjlab.scripts.render: the scene asked for, whatever the outcome. --video-only
skips the evaluation. The video is .mp4 or .gif.

  uv run ballet-play RoboBallet-4R10T-Mesh --checkpoint logs/.../policies/policy_00100000.pt --video demo.mp4

Rendering is EGL, on the device MUJOCO_EGL_DEVICE_ID names, or --egl-device.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from roboballet_mjlab.cli._common import run_module, set_data_dir, task_listing, add_task
from roboballet_mjlab.cli.eval import EVAL_MODULE, SET_NAMES, add_eval_options, chosen_sets, eval_argv
from roboballet_mjlab.tasks import load_task


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(prog='ballet-play', description=__doc__, epilog=task_listing(),
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  add_task(p)
  add_eval_options(p)
  p.add_argument('--video', type=Path, metavar='OUT', help='Record an episode to this .mp4 or .gif.')
  p.add_argument('--video-only', action='store_true', help='Record without evaluating first.')
  p.add_argument('--video-set', choices=SET_NAMES, default='training_dist', help='The scenes to record from.')
  p.add_argument('--scene', type=int, default=0, help='Index of the scene in --video-set.')
  p.add_argument('--width', type=int, default=800)
  p.add_argument('--height', type=int, default=600)
  p.add_argument('--label', default='', help='Text before the step count in the caption.')
  p.add_argument('--egl-device', help='MUJOCO_EGL_DEVICE_ID for the renderer.')
  args = p.parse_args(sys.argv[1:] if argv is None else argv)
  if args.video_only and args.video is None:
    p.error('--video-only needs --video')
  set_data_dir(args.data_dir)
  task = load_task(args.task)

  if not args.video_only:
    sets = chosen_sets(task, args, check=not args.dry_run)
    code = run_module(EVAL_MODULE, eval_argv(task, args, sets), args.dry_run)
    if code:
      return code
  if args.video is None:
    return 0

  _, source = task.train_source()
  pool = {'training_dist': source, **task.eval_paths()}.get(args.video_set)
  if pool is None:
    raise SystemExit(f'{task.task_id} has no set {args.video_set}.')
  seed = 1 if args.seed is None else args.seed
  print(f'[INFO] video: scene {args.scene} of {pool}, arm {task.arm_model}, seed {seed}, to {args.video}', flush=True)
  if args.dry_run:
    return 0
  if not args.checkpoint.is_file():
    raise SystemExit(f'checkpoint not found: {args.checkpoint}')
  if not pool.exists():
    raise SystemExit(f'{pool} not found. Download it with: uv run ballet-fetch-data')
  if args.egl_device is not None:
    os.environ['MUJOCO_EGL_DEVICE_ID'] = args.egl_device
  from roboballet_mjlab.scripts import render  # Sets MUJOCO_GL and the EGL device on import.

  render.record(args.checkpoint, pool, args.scene, task.arm_model, seed, args.video, args.width, args.height, args.label)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
