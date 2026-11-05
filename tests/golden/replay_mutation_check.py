"""Injects bugs into temporary copies of replay/ and checks that the tests catch each one.

Run from the repository root, on GPU 4, with TMPDIR pointing somewhere private:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tests.golden.replay_mutation_check

Each mutant gets an overlay package: symlinks to everything in roboballet_mjlab except replay/,
which is a mutated copy. The real files are never written. The throughput tests are left out.
Exit status is 0 only if the unmutated overlay passes and every mutant fails.
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parents[2] / 'src/roboballet_mjlab'
H, R = 'hindsight.py', 'replay.py'
# The tests of the mutated file go first, to stop sooner. The adversarial file is slowest.
ADVERSARIAL = 'tests/test_replay_adversarial.py'
TESTS = {H: ['tests/test_hindsight.py', 'tests/test_replay.py', ADVERSARIAL],
         R: ['tests/test_replay.py', 'tests/test_hindsight.py', ADVERSARIAL]}
WORKERS = 2  # More only adds GPU context switches between processes.
TIMEOUT_S = 600

# name, file, then (old, new) pairs. Each `old` must occur exactly once.
MUTANTS = [
  ('time range includes 0', H, [('% (length[:, None] - 1) + 1', '% length[:, None]')]),
  ('time range includes A', H, [('% (length[:, None] - 1) + 1', '% length[:, None] + 1')]),
  ('arm range misses the last robot', H, [('arm = torch.randint(num_robots, shape', 'arm = torch.randint(num_robots - 1, shape')]),
  ('scaled tip position', H, [('robot.tip_pos[episodes, picks.time, picks.arm]', 'robot.tip_pos[episodes, picks.time, picks.arm] / 5.0')]),
  ('collision penalty kept', H, [('if not train_cfg.hindsight_zero_collision_penalty:', 'if True:')]),
  ('no dwell mutation', H, [('action = action.masked_fill(dwelling[..., None], 0.0)', 'action = action + 0.0')]),
  ('no takeover mutation', H, [('if not all_done.any():', 'if True:')]),
  ('takeover does not override dwell', H, [
    ('takeover = all_done & (to_home.abs() < cfg.return_to_start_threshold).all(-1)',
     'takeover = all_done & (to_home.abs() < cfg.return_to_start_threshold).all(-1) & ~dwelling')]),
  ('takeover skipped when return-to-start weight is zero', H, [('if not all_done.any():', 'if not all_done.any() or cfg.return_to_start_reward_weight == 0.0:')]),
  ('takeover in rad/s, not stored units', H, [('to_home / cfg.timestep_s / cfg.max_joint_velocity', 'to_home / cfg.timestep_s')]),
  ('takeover threshold inclusive of a wider box', H, [('(to_home.abs() < cfg.return_to_start_threshold).all(-1)', '(to_home.abs() < cfg.return_to_start_threshold + 0.05).all(-1)')]),
  ('takeover reads the next row q', H, [('dwelling, done, robot.q[:, t], env_cfg, home)', 'dwelling, done, robot.q[:, t + 1], env_cfg, home)')]),
  ('takeover uses done from after the step', H, [
    ('      action[:, t] = mutated_action(action[:, t], dwelling, done, robot.q[:, t], env_cfg, home)\n', '      pass\n'),
    ('    new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)\n',
     '    action[:, t] = mutated_action(action[:, t], dwelling, done, robot.q[:, t], env_cfg, home)\n'
     '    new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)\n')]),
  ('scoring reads the current row, not the next', H, [('targets_within(robot.tip_pos[:, t + 1], robot.tip_rot[:, t + 1],', 'targets_within(robot.tip_pos[:, t], robot.tip_rot[:, t],')]),
  ('dwell decremented after scoring', H, [
    ('    dwell = (dwell - 1).clamp(min=0)  # Before scoring, planning_env.py:796.\n', ''),
    ('    new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)\n',
     '    dwell = (dwell - 1).clamp(min=0)\n    new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)\n')]),
  ('terminal one step late', H, [('torch.where(finished & (t + 1 < length), t + 1, length)', 'torch.where(finished & (t + 1 < length), t + 2, length)')]),
  ('no early ending when all done', H, [('if everything.any():', 'if False:')]),
  ('ending ignores final dwell', H, [('finished = everything & (dwell == 0).all(1)', 'finished = everything')]),
  ('parked check ignored', H, [('if env_cfg.return_to_start_required:', 'if False:')]),
  ('solved counted a row before the terminal', H, [('[torch.arange(len(batch.length), device=batch.length.device), batch.length]', '[torch.arange(len(batch.length), device=batch.length.device), batch.length - 1]')]),
  ('solved targets can be chosen', H, [('.masked_fill(final_done, 2.0)', '')]),
  ('picks k regardless of solved', H, [('(k - final_done.sum(1, keepdim=True))', 'k')]),
  ('rotation not taken from the tip', H, [('torch.where(chosen[..., None, None], robot.tip_rot[episodes, picks.time, picks.arm], batch.scene.target_rot)', 'batch.scene.target_rot')]),
  ('one augmentation per episode', H, [('aug_code = torch.randint(4, shape, generator=generator, device=device, dtype=torch.int8)',
                                        'aug_code = torch.randint(4, (n, 1, shape[2]), generator=generator, device=device, dtype=torch.int8).expand(shape)')]),
  ('terminal row copies its augmentation', H, [('  stats = batch.stats.clone()\n',
                                                '  aug_code[episodes[:, 0], length] = aug_code[episodes[:, 0], length - 1]\n  stats = batch.stats.clone()\n')]),
  ('statistics flag not set', H, [("  stats[:, STAT['is_hindsight_replay']] = 1.0\n", '')]),
  ('episode flag not set', H, [('torch.ones_like(batch.is_hindsight)', 'batch.is_hindsight')]),
  ('real score reused', H, [('new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)', 'new_score = robot.score[:, t + 1]')]),
  ('return-to-start score from the wrong row', H, [('new_score = score_of(done, robot.q[:, t + 1], env_cfg, home)', 'new_score = score_of(done, robot.q[:, t], env_cfg, home)')]),
  ('queue counts only eligible episodes', H, [('self.finished += len(batch.length)', 'self.finished += int(eligible(batch, self.k).sum())')]),
  ('queue keeps ineligible episodes', H, [('      self.pending.append(select_episodes(batch, keep))', '      self.pending.append(batch)')]),
  ('takeover clamped to the action bounds', H, [('to_home / cfg.timestep_s / cfg.max_joint_velocity, action)', '(to_home / cfg.timestep_s / cfg.max_joint_velocity).clamp(-1, 1), action)')]),

  ('sampling a final state', R, [('is_transition = t < length[ep]', 'is_transition = t > 0')]),
  ('next state is the same row', R, [('next_row = (row + 1) % self.num_rows', 'next_row = row')]),
  ('next state does not wrap', R, [('next_row = (row + 1) % self.num_rows', 'next_row = (row + 1).clamp(max=self.num_rows - 1)')]),
  ('terminal off by one', R, [('self.terminal[dst_t] = t_t == length[ep_t] - 1', 'self.terminal[dst_t] = t_t == length[ep_t]')]),
  ('oldest live transition off by one', R, [('    self.evicted = oldest\n', '    self.evicted = oldest + 1 if oldest else 0\n')]),
  ('whole-episode eviction', R, [('    by_capacity = max(self.evicted, self.inserted - C)\n',
                                  '    by_capacity = max(self.evicted, self.inserted - C)\n'
                                  '    if self.episodes > 1 and by_capacity > int(self.first_transition[(self.episodes - 1) % E]) - C:\n'
                                  '      by_capacity = max(by_capacity, int(self.first_transition[(self.episodes - 1) % E]))\n')]),
  ('row ring has no slack', R, [('self.num_rows = capacity + self.max_episodes', 'self.num_rows = capacity + 1')]),
  ('episode slot wrong after the first batch', R, [('self.row_episode[dst] = slots[ep - first]', 'self.row_episode[dst] = ep - first')]),
  ('extra evictions not counted', R, [('    self.extra_evicted += oldest - by_capacity\n', '')]),
  ('overwritten episodes stay live', R, [('if self.episodes > E:  # Slot', 'if False:  # Slot')]),
  ('trimmed episodes stay live', R, [('oldest = max(by_capacity, first_id) if first else by_capacity', 'oldest = by_capacity')]),
  ('sampled not counted', R, [('    self.sampled += n\n', '')]),
  ('step dtype overflows at T=40', R, [('step=zeros(dtype=_small_int(cfg.max_steps))', 'step=zeros(dtype=torch.int8)')]),
  ('final row not the terminal state', R, [('      store[ended, length] = value[ended].to(store.dtype)\n', '      store[ended, length] = store[ended, length - 1]\n')]),
  ('count not reset at episode end', R, [('    self.count[world_ids] = 0\n', '    self.count[world_ids] = 1\n')]),
  ('totals carried across episodes', R, [('    self.totals[world_ids] = 0.0\n', '')]),
  ('behaviour step not latched', R, [('torch.where(count == 0, behaviour_step, self.behaviour_step)', 'torch.where(count >= 0, behaviour_step, self.behaviour_step)')]),
  ('sample of one aliases storage', R, [('self.action[row], self.reward[row],', 'self.action[row], self.reward[int(row[0]) : int(row[0]) + 1] if len(row) == 1 else self.reward[row],')]),
  ('finished batch aliases the buffer', R, [('self.action[ended], self.reward[ended], length,', 'self.action[ended] if len(ended) > 1 else self.action[int(ended[0]) : int(ended[0]) + 1], self.reward[ended], length,')]),
  ('sampling beyond the live range', R, [('torch.randint(self.size, (n,)', 'torch.randint(min(self.size + 1, self.capacity), (n,)')]),
  ('FIFO position ignores evictions', R, [('self.transition_row[(self.evicted + position) % self.capacity]', 'self.transition_row[position % self.capacity]')]),
  ('statistics of the wrong episode', R, [('self.behaviour_step[episode], self.is_hindsight[episode], self.stats[episode])', 'self.behaviour_step[episode], self.is_hindsight[episode], self.stats[row % self.max_episodes])')]),
  ('oversize batch keeps its head rows', R, [('      stored[first, :skip] = False\n', ''), ('      kept_length[0] -= skip\n', '')]),
  ('action one row late', R, [('self.action[dst_t] = batch.action[ep_t, t_t]', 'self.action[dst_t] = batch.action[ep_t, (t_t + 1).clamp(max=batch.action.shape[1] - 1)]')]),
  ('integer fields stay narrow in samples', R, [('r.dwell[row].long()', 'r.dwell[row]')]),
  ('no graph guard on append', R, [('  @torch.no_grad()\n  def append(', '  def append(')]),
  ('behaviour step passes through f32', R, [('self.behaviour_step = torch.where(count == 0, behaviour_step, self.behaviour_step)',
                                             'self.behaviour_step = torch.where(count == 0, torch.as_tensor(behaviour_step, device=count.device).float().long(), self.behaviour_step)')]),
]


def make_overlay(root: Path, mutant: tuple | None) -> Path:
  overlay = root / 'roboballet_mjlab'
  overlay.mkdir(parents=True)
  (root / 'tests').symlink_to(PACKAGE.parents[1] / 'tests')  # The tests sit beside src/, not in the package.
  for child in PACKAGE.iterdir():
    if child.name not in ('replay', '__pycache__'):
      (overlay / child.name).symlink_to(child)
  replay = overlay / 'replay'
  replay.mkdir()
  for name in ('__init__.py', H, R):
    shutil.copy(PACKAGE / 'replay' / name, replay / name)
  if mutant is not None:
    _, filename, edits = mutant
    path = replay / filename
    text = path.read_text()
    for old, new in edits:
      if text.count(old) != 1:
        raise ValueError(f'{mutant[0]}: found {text.count(old)} matches of {old!r}')
      text = text.replace(old, new)
    path.write_text(text)
  return root


def run_suite(root: Path, filename: str = R) -> tuple[int, str, float]:
  env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
  start = time.time()
  try:
    done = subprocess.run(
      [sys.executable, '-m', 'pytest', '-x', '-q', '--tb=line', '-p', 'no:cacheprovider', f'--rootdir={root}', *TESTS[filename],
       '-k', 'not throughput'],
      cwd=root, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
  except subprocess.TimeoutExpired:
    return -1, 'timed out', time.time() - start
  found = re.search(r'^(?:FAILED|ERROR) (\S+)', done.stdout, re.M)
  summary = found.group(1).split('::')[-1] if found else done.stdout.strip().splitlines()[-1]
  return done.returncode, summary, time.time() - start


def run_mutant(mutant: tuple, base: Path, index: int) -> tuple[str, int, str, float]:
  root = make_overlay(base / f'm{index}', mutant)
  try:
    code, summary, seconds = run_suite(root, mutant[1])
  finally:
    shutil.rmtree(root)
  caught = 'caught  ' if code != 0 else 'SURVIVED'
  print(f'{caught}  {mutant[0]:52s}  {summary}  ({seconds:.0f} s)', flush=True)
  return mutant[0], code, summary, seconds


def main() -> int:
  with tempfile.TemporaryDirectory(prefix='replay_mutants_') as tmp:
    base = Path(tmp)
    root = make_overlay(base / 'clean', None)
    code, summary, seconds = run_suite(root)
    print(f'unmutated overlay: exit {code}, {summary}, {seconds:.0f} s', flush=True)
    if code != 0:
      return 1
    # Syntax of every mutant first, and each `old` must match, before spending any time.
    for i, mutant in enumerate(MUTANTS):
      check = make_overlay(base / f'c{i}', mutant)
      compile((check / 'roboballet_mjlab/replay' / mutant[1]).read_text(), mutant[1], 'exec')
      shutil.rmtree(check)
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
      results = list(pool.map(lambda x: run_mutant(x[1], base, x[0]), enumerate(MUTANTS)))
  survivors = sum(code == 0 for _, code, _, _ in results)
  print(f'{len(results) - survivors} of {len(results)} mutants caught')
  return 1 if survivors else 0


if __name__ == '__main__':
  raise SystemExit(main())
