"""Generates a scene pool. DESIGN section 6.

Run, from the clone: CUDA_VISIBLE_DEVICES=3 uv run python \
  -m roboballet_mjlab.scripts.gen_scenes --robots 4 --targets 10 --obstacles 30 --split train \
  --count 50000 --seed 1

Writes scenes/{R}r_{T}t_{O}o_{split}.pt in the data directory (roboballet_mjlab/paths.py), where
it takes the place of a pool of the same name in the package. Shards land in {name}.shards/ as
they finish, each seeded by its own index and holding the fingerprint of what made it, so a
rerun resumes and yields the same pool, and refuses a shard of any other configuration.
Every scene is then rechecked on C MuJoCo, and a scene that fails is dropped and counted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import multiprocessing
import os
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from roboballet_mjlab.config import WorldCfg
from roboballet_mjlab.paths import PACKAGE_DATA, data_dir
from roboballet_mjlab.scenes.ik import ARM_XML, ArmKinematics, IkBudget, find_witnesses
from roboballet_mjlab.scenes.io import Pool, git_revision, load_pool, load_taskset, save_pool
from roboballet_mjlab.scenes.sampler import SamplerCfg, Witnesses, any_robot_collides, blank_scene, generate, sample_targets
from roboballet_mjlab.sim.world import build_world
from roboballet_mjlab.structs import Scene

try:  # The C MuJoCo recheck is test code, tests/test_scenes_common.py: the test branch's, from the clone.
  from tests.test_scenes_common import recheck_chunk, summarise_recheck
except ModuleNotFoundError as missing:
  if missing.name not in ('tests', 'tests.test_scenes_common'):
    raise
  recheck_chunk = summarise_recheck = None

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = data_dir() / 'scenes'
STATIC_BOXES = PACKAGE_DATA / 'static_boxes'  # Frozen OBBs of the original's world, env_spec D4.
WORLDS = 4096
RECHECK_PROCESSES = 16
RECHECK_CHUNK = 128
STATIC_SETS = 10
# The search behind the reported ceiling of the paper's tasksets. It is a floor on true
# reachability (env_spec.md 11.7), so this is well above the filter budget.
CEILING_BUDGET = IkBudget(restarts=1000)
STATIC_LIFT_8R = 0.6  # env_spec D3: lift the boxes with the targets.
# A healthy shard keeps nearly every attempt, so this far past the need means something is wrong.
MAX_ATTEMPTS_PER_SCENE = 4


def pool_name(args) -> str:
  suffix = '' if args.arm_model == 'spheres_stick' else f'_{args.arm_model}'
  return f'{args.robots}r_{args.targets}t_{args.obstacles}o_{args.split}{suffix}'


def shard_seed(args, shard: int) -> int:
  return zlib.crc32(f'{pool_name(args)}/{args.seed}/{shard}'.encode())


def make_generator(args, shard: int, device) -> torch.Generator:
  return torch.Generator(device).manual_seed(shard_seed(args, shard))


def cat_pools(pools: list[Pool]) -> tuple[Scene, Witnesses]:
  scene = Scene(*[torch.cat([p.scene[i] for p in pools]) for i in range(len(Scene._fields))])
  return scene, Witnesses(torch.cat([p.witnesses.q for p in pools]),
                          torch.cat([p.witnesses.robot for p in pools]))


def sum_counts(pools: list[Pool]) -> dict:
  keys = pools[0].meta['counts'].keys()
  return {k: sum(p.meta['counts'][k] for p in pools) for k in keys}


def rates(counts: dict, num_robots: int, num_obstacles: int) -> dict:
  placed = counts['attempted'] - counts['rail_failures']
  return {
    'rail_accept': counts['attempted'] * num_robots / counts['rail_draws'],
    'box_accept': (placed - counts['box_failures']) * num_obstacles / counts['box_draws'],
    'target_accept': counts['target_accepted'] / counts['target_draws'],
    'draws_per_target': counts['target_draws'] / counts['target_accepted'],
    'restart_problems_per_draw': counts['ik_restart_problems'] / counts['target_draws'],
  }


def recheck(pool: Pool, arm_model: str) -> list[dict]:
  """Every scene and witness on C MuJoCo, in separate processes without a GPU."""
  n = len(pool.scene.q0)
  scene = {name: field.cpu().numpy() for name, field in pool.scene._asdict().items()}
  q, robot = pool.witnesses.q.cpu().numpy(), pool.witnesses.robot.cpu().numpy()
  chunks = [
    ({k: v[i:i + RECHECK_CHUNK] for k, v in scene.items()}, q[i:i + RECHECK_CHUNK],
     robot[i:i + RECHECK_CHUNK], arm_model)
    for i in range(0, n, RECHECK_CHUNK)
  ]
  saved = os.environ.get('CUDA_VISIBLE_DEVICES', '')
  os.environ['CUDA_VISIBLE_DEVICES'] = ''  # For the workers: this process already holds its GPU.
  try:
    with multiprocessing.get_context('spawn').Pool(RECHECK_PROCESSES) as workers:
      results = workers.map(recheck_chunk, chunks, chunksize=1)
  finally:
    os.environ['CUDA_VISIBLE_DEVICES'] = saved
  return [row for chunk in results for row in chunk]


def build(args, device):
  cfg = WorldCfg(num_robots=args.robots, num_obstacles=args.obstacles, arm_model=args.arm_model)
  return build_world(cfg, WORLDS, device), ArmKinematics(args.arm_model, device)


def hashed_files(arm_model: str) -> list[Path]:
  """The files a shard's contents depend on, other than through its parameters."""
  return [REPO / 'roboballet_mjlab/scenes/ik.py', REPO / 'roboballet_mjlab/scenes/sampler.py', ARM_XML[arm_model]]


def code_sha256(arm_model: str) -> str:
  digest = hashlib.sha256()
  for path in hashed_files(arm_model):
    digest.update(path.read_bytes())
  return digest.hexdigest()


def fingerprint(args) -> dict:
  """Everything that decides a shard's contents. The pool's count does not, so it is absent."""
  return {
    'seed': args.seed, 'split': args.split, 'robots': args.robots, 'targets': args.targets,
    'obstacles': args.obstacles, 'arm_model': args.arm_model,
    'sampler': dataclasses.asdict(SamplerCfg()), 'budget': dataclasses.asdict(IkBudget()),
    'shard_size': WORLDS, 'code_sha256': code_sha256(args.arm_model),
  }


def base_meta(args, budget: IkBudget) -> dict:
  return {
    'robots': args.robots, 'targets': args.targets, 'obstacles': args.obstacles,
    'split': args.split, 'arm_model': args.arm_model, 'seed': args.seed,
    'budget': dataclasses.asdict(budget), 'sampler': dataclasses.asdict(SamplerCfg()),
    'git_revision': git_revision(REPO.parent), 'code_sha256': code_sha256(args.arm_model),
  }


def _malformed(scene: Scene, witnesses: Witnesses, allow_missing: bool = False) -> torch.Tensor:
  """[N] bool: a nonfinite value anywhere, or a witness robot index that names no robot.

  allow_missing also accepts robot -1, a target without a witness. Only static_paper passes
  it. Its targets are the paper's and were not filtered. Every other split is reachable by
  construction, DESIGN G5, so a missing witness there is a defect.
  """
  finite = torch.stack([field.flatten(1).isfinite().all(1) for field in (*scene, witnesses.q)]).all(0)
  lowest = -1 if allow_missing else 0
  named = ((witnesses.robot >= lowest) & (witnesses.robot < scene.q0.shape[1])).all(1)
  return ~(finite & named)


def _keep_passing(scene: Scene, witnesses: Witnesses, results: list[dict], count: int):
  """The final rejection of every split except static_paper. Returns scene, witnesses, dropped."""
  malformed = _malformed(scene, witnesses).tolist()
  good = [i for i, row in enumerate(results)
          if not malformed[i] and not row['home_collision'] and not row['witness_missing'].any()
          and not row['witness_collision'].any() and not row['witness_far'].any()
          and not row['witness_limits'].any()]
  if len(good) < count:
    raise RuntimeError(f'{len(good)} scenes passed the C MuJoCo recheck, {count} needed. Raise the spare.')
  ids = torch.as_tensor(good[:count], device=scene.q0.device)
  kept = Scene(*[f[ids] for f in scene]), Witnesses(witnesses.q[ids], witnesses.robot[ids])
  return *kept, len(results) - len(good)


def generate_shards(args, shard_dir: Path, needed: int, device) -> list[Pool]:
  """Resumes from finished shards, then generates until `needed` scenes exist.

  Raises on a shard of another configuration, which stays on disk, on a shard without a
  single scene, and once the attempts pass the bound. All of these leave no pool behind.
  """
  shard_dir.mkdir(parents=True, exist_ok=True)
  expected = fingerprint(args)
  limit = MAX_ATTEMPTS_PER_SCENE * max(needed, WORLDS)
  world, kin, pools, total, shard, attempted = None, None, [], 0, 0, 0
  while total < needed:
    path = shard_dir / f'{shard:05d}.pt'
    if path.exists():
      pool = load_pool(path, device)
      found = pool.meta.get('fingerprint')
      if found != expected:
        raise RuntimeError(f'{path} has another generation fingerprint. Use the settings that made '
                           f'it, or remove {shard_dir} to start over.\nShard: {found}\nThis run: {expected}')
    else:
      if world is None:
        world, kin = build(args, device)
      start = time.time()
      result = generate(world, kin, WORLDS, args.targets, SamplerCfg(), IkBudget(),
                        make_generator(args, shard, device))
      torch.cuda.synchronize()
      counts = {**result.counts, 'seconds': time.time() - start}
      pool = Pool(result.scene, result.witnesses, {'counts': counts, 'fingerprint': expected})
      if len(result.scene.q0):
        save_pool(path, pool)
        print(f'shard {shard}: {len(result.scene.q0)} scenes, {counts["seconds"]:.1f} s, '
              f'target accept {counts["target_accepted"] / counts["target_draws"]:.3f}', flush=True)
    if len(pool.scene.q0) == 0:
      raise RuntimeError(f'Shard {shard} of {pool_name(args)} produced no scenes. Counts: {pool.meta["counts"]}')
    pools.append(pool)
    total += len(pool.scene.q0)
    attempted += pool.meta['counts']['attempted']
    shard += 1
    if total < needed and attempted >= limit:
      raise RuntimeError(f'{attempted} attempts gave {total} of {needed} scenes of {pool_name(args)}. '
                         f'Counts: {sum_counts(pools)}')
  return pools


def generate_pool(args, device) -> Pool:
  # Spare scenes for the C MuJoCo recheck. The rate is reported in meta['c_check'].
  needed = args.count + max(32, args.count // 50)
  shard_dir = OUT_DIR / f'{pool_name(args)}.shards'
  pools = generate_shards(args, shard_dir, needed, device)
  counts = sum_counts(pools)
  scene, witnesses = cat_pools(pools)
  scene = Scene(*[f[:needed] for f in scene])
  witnesses = Witnesses(witnesses.q[:needed], witnesses.robot[:needed])
  results = recheck(Pool(scene, witnesses, {}), args.arm_model)
  meta = base_meta(args, IkBudget())
  scene, witnesses, failed = _keep_passing(scene, witnesses, results, args.count)
  meta['c_check'] = {**summarise_recheck(results), 'scenes_failed': failed}
  meta['count'] = args.count
  meta['counts'] = counts
  meta['rates'] = rates(counts, args.robots, args.obstacles)
  meta['generation_s'] = counts['seconds']
  return Pool(scene, witnesses, meta)


def static_boxes(num_robots: int, num_scenes: int, device) -> tuple[torch.Tensor, ...]:
  """Frozen OBBs. The transpose undoes env_spec D4: with it the boxes hold every cube vertex."""
  z = np.load(STATIC_BOXES / f'static_boxes_{num_robots}r.npz')
  as_f32 = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
  pos = as_f32(z['centroid'])
  if num_robots == 8:
    pos = pos + torch.as_tensor([0, 0, STATIC_LIFT_8R], device=device)
  rot = as_f32(z['rotation']).transpose(-1, -2)
  spans = as_f32(z['spans'])
  return (pos.repeat(num_scenes, 1, 1), rot.repeat(num_scenes, 1, 1, 1), spans.repeat(num_scenes, 1, 1))


def static_scene(args, world, num_scenes: int, device) -> Scene:
  box_pos, box_rot, box_spans = static_boxes(args.robots, num_scenes, device)
  scene = blank_scene(world, num_scenes, args.targets)._replace(
    box_pos=box_pos, box_rot=box_rot, box_spans=box_spans)
  if any_robot_collides(world, scene, torch.arange(num_scenes, device=device)).any():
    raise RuntimeError('The static boxes collide with a robot at home')
  return scene


def generate_static(args, device) -> Pool:
  """New target sets on the frozen boxes. Two spare sets cover the recheck."""
  world, kin = build(args, device)
  count = args.count or STATIC_SETS
  scene = static_scene(args, world, count + 2, device)
  start = time.time()
  witnesses, failed, counts = sample_targets(
    world, kin, scene, SamplerCfg(), IkBudget(), make_generator(args, 0, device))
  torch.cuda.synchronize()
  if failed.any():
    raise RuntimeError(f'{int(failed.sum())} static sets could not be filled')
  results = recheck(Pool(scene, witnesses, {}), args.arm_model)
  meta = base_meta(args, IkBudget())
  scene, witnesses, failed = _keep_passing(scene, witnesses, results, count)
  meta['c_check'] = {**summarise_recheck(results), 'scenes_failed': failed}
  meta['count'] = count
  meta['counts'] = counts
  meta['rates'] = {'target_accept': counts['target_accepted'] / counts['target_draws'],
                   'draws_per_target': counts['target_draws'] / counts['target_accepted']}
  meta['generation_s'] = time.time() - start
  meta['box_rotation'] = 'transpose of the stored matrix, env_spec D4'
  return Pool(scene, witnesses, meta)


def generate_static_paper(args, device) -> Pool:
  """The paper's tasksets on the frozen boxes, unfiltered, with their reachability ceiling.

  The one split where a target may lack a witness. Every set is kept, and a witness that was
  not found or fails the recheck is recorded as robot -1. See _malformed.
  """
  world, kin = build(args, device)
  pos, rot = load_taskset(data_dir() / f'paper_test_tasksets/test_{args.targets}targets.pbtxt', device)
  if args.robots == 8:
    pos = pos + torch.as_tensor([0, 0, STATIC_LIFT_8R], device=device)
  num_sets, num_targets = pos.shape[:2]
  scene = static_scene(args, world, num_sets, device)._replace(target_pos=pos, target_rot=rot)
  scene_ids = torch.arange(num_sets, device=device).repeat_interleave(num_targets)
  start = time.time()
  result = find_witnesses(world, kin, scene, scene_ids, pos.flatten(0, 1), rot.flatten(0, 1),
                          CEILING_BUDGET, make_generator(args, 0, device))
  torch.cuda.synchronize()
  witnesses = Witnesses(result.q.view(num_sets, num_targets, 7), result.robot.view(num_sets, num_targets))
  results = recheck(Pool(scene, witnesses, {}), args.arm_model)
  meta = base_meta(args, CEILING_BUDGET)
  meta['c_check'] = summarise_recheck(results)
  if any(row['home_collision'] for row in results):
    raise RuntimeError(f'A static set collides at home on C MuJoCo: {meta["c_check"]}')
  bad = np.stack([row['witness_collision'] | row['witness_far'] | row['witness_limits'] for row in results])
  robot = witnesses.robot.clone()
  robot[torch.as_tensor(bad, device=device)] = -1
  witnesses = Witnesses(witnesses.q, robot)
  if _malformed(scene, witnesses, allow_missing=True).any():
    raise RuntimeError('A static set holds a nonfinite value or an index that names no robot')
  reachable = robot >= 0
  meta['count'] = num_sets
  meta['reachability_ceiling'] = float(reachable.float().mean())
  meta['reachable_per_set'] = reachable.sum(-1).tolist()
  meta['generation_s'] = time.time() - start
  meta['box_rotation'] = 'transpose of the stored matrix, env_spec D4'
  return Pool(scene, witnesses, meta)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--robots', type=int, required=True)
  parser.add_argument('--targets', type=int, required=True)
  parser.add_argument('--obstacles', type=int, required=True)
  parser.add_argument('--split', choices=('train', 'heldout', 'static', 'static_paper'), required=True)
  parser.add_argument('--count', type=int, default=0,
                      help=f'Scenes. Static defaults to {STATIC_SETS} sets, static_paper ignores it.')
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--arm-model', default='spheres_stick', choices=('spheres_stick', 'mesh_tool'),
                      help='mesh_tool exists for acceptance test E10.')
  return parser.parse_args(argv)


def main(argv=None) -> Path:
  args = parse_args(argv)
  if recheck_chunk is None:
    raise SystemExit('Every pool is rechecked on C MuJoCo by tests/test_scenes_common.py, which is on the test '
                     'branch. Run this from the root of a clone of that branch.')
  if args.split in ('train', 'heldout') and args.count <= 0:
    raise SystemExit(f'--count is required for --split {args.split}')
  device = 'cuda:0'
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  generators = {'train': generate_pool, 'heldout': generate_pool,
                'static': generate_static, 'static_paper': generate_static_paper}
  pool = generators[args.split](args, device)
  path = OUT_DIR / f'{pool_name(args)}.pt'
  save_pool(path, pool)
  shutil.rmtree(OUT_DIR / f'{pool_name(args)}.shards', ignore_errors=True)
  print(f'{path}: {len(pool.scene.q0)} scenes, c_check {pool.meta["c_check"]}, '
        f'rates {pool.meta.get("rates")}, generation {pool.meta["generation_s"]:.0f} s', flush=True)
  return path


if __name__ == '__main__':
  main()
