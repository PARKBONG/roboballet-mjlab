"""Injects bugs into temporary copies of train/ and agent/td3.py and checks that the tests catch each.

Run from the repository root on an otherwise idle GPU, with TMPDIR pointing somewhere private:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m tests.golden.trainer_mutation_check

Each mutant gets an overlay package: symlinks to everything in roboballet_mjlab except train/ and
agent/, which are copies. The real files are never written. Exit status is 0 only if the
unmutated overlay passes and every mutant fails.
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
TR, EV, TD, PR = 'train/trainer.py', 'train/evaluator.py', 'agent/td3.py', 'train/profiler.py'
TESTS = {TR: ['tests/test_trainer.py', 'tests/test_evaluator.py'],
         TD: ['tests/test_trainer.py'],
         PR: ['tests/test_trainer.py'],
         EV: ['tests/test_evaluator.py']}
WORKERS = 2
TIMEOUT_S = 1200

# name, file, then (old, new) pairs. Each `old` must occur exactly once.
MUTANTS = [
  # Latching against actor.py.
  ('fetch before the random check', TR, [
    ('    held = self.held_step[ids]\n', '    self.held_step[ids] = published_step\n    held = self.held_step[ids]\n')]),
  ('sigma from the fresh fetch', TR, [
    ('self.sigma[ids] = self.td3.exploration_sigma(held.double()).float()',
     'self.sigma[ids] = self.td3.exploration_sigma(torch.full_like(held, published_step).double()).float()')]),
  ('random check ignores the held step', TR, [
    ('(self.finished[ids] < self.cfg.random_episodes_per_world) | (held == 0)', '(self.finished[ids] < self.cfg.random_episodes_per_world)')]),
  ('random check ignores the episode count', TR, [
    ('(self.finished[ids] < self.cfg.random_episodes_per_world) | (held == 0)', '(held == 0)')]),
  ('random check one episode longer', TR, [
    ('self.finished[ids] < self.cfg.random_episodes_per_world', 'self.finished[ids] <= self.cfg.random_episodes_per_world')]),
  ('episode counted after the latch', TR, [
    ('      self.latches.finish(ended)\n      self.latches.begin(ended, self.published_step)\n',
     '      self.latches.begin(ended, self.published_step)\n      self.latches.finish(ended)\n')]),
  ('fetch takes the latest step, not the published one', TR, [
    ('      self.latches.begin(ended, self.published_step)\n', '      self.latches.begin(ended, self.td3.grad_steps)\n')]),
  ('latch before the refresh of the same round', TR, [
    ('    if self._finished_since_publish >= self.num_worlds:\n      self.publish()\n    if finished is not None:\n',
     '    if finished is not None and False:\n      pass\n    if finished is not None:\n'),
    ('      self._stagger_at[ended] = -1  # Ended on its own before its offset: already out of step.\n',
     '      self._stagger_at[ended] = -1\n    if self._finished_since_publish >= self.num_worlds:\n      self.publish()\n')]),
  ('published step never moves', TR, [('    self.published_step = self.td3.grad_steps\n    self._finished_since_publish = 0\n', '    self._finished_since_publish = 0\n')]),
  ('behaviour step is the current grad step', TR, [
    ('behaviour_step=self.latches.held_step)', 'behaviour_step=self.td3.grad_steps)')]),
  # Actions.
  ('noise not clipped', TR, [('noisy = (mean + sigma[:, None, None] * normals).clamp(-1.0, 1.0)', 'noisy = mean + sigma[:, None, None] * normals')]),
  ('clip before the noise', TR, [('noisy = (mean + sigma[:, None, None] * normals).clamp(-1.0, 1.0)', 'noisy = mean.clamp(-1.0, 1.0) + sigma[:, None, None] * normals')]),
  ('one sigma for all worlds', TR, [('noisy = (mean + sigma[:, None, None] * normals)', 'noisy = (mean + sigma.max() * normals)')]),
  ('uniform in [0, 1]', TR, [('generator=self.act_rng, device=self.device) * 2.0 - 1.0', 'generator=self.act_rng, device=self.device)')]),
  ('network runs in the random phase', TR, [('    if latches.use_random.all():\n      return uniform  # The original\'s random actors run no network, actor.py:232-239.\n', '')]),
  ('acting with the online policy', TR, [('mean = self.td3.act_behaviour(graph)', 'mean = self.td3.act(graph)')]),
  # Refresh.
  ('refresh every round', TR, [('if self._finished_since_publish >= self.num_worlds:', 'if True:')]),
  ('refresh at twice the interval', TR, [('if self._finished_since_publish >= self.num_worlds:', 'if self._finished_since_publish >= 2 * self.num_worlds:')]),
  ('refresh on the timeout period', TR, [('if self._finished_since_publish >= self.num_worlds:', 'if self.counters.rounds % self.env_cfg.max_steps == 0:')]),
  ('finished count not reset by a refresh', TR, [('    self.published_step = self.td3.grad_steps\n    self._finished_since_publish = 0\n', '    self.published_step = self.td3.grad_steps\n')]),
  ('snapshot holds the acting copy', TR, [("state = {name: value.cpu() for name, value in self.td3.policy.state_dict().items()}", "state = {name: value.cpu() for name, value in self.td3.policy_acting.state_dict().items()}")]),
  ('snapshot overwrites one file', TR, [("f'policy_{step:08d}.pt'", "'policy.pt'")]),
  ('no snapshot of the untrained policy', TR, [("    self._snapshot_if_due()  # The untrained policy, on a new run.\n", "")]),
  ('held step not restored', TR, [("    self.latches.held_step.copy_(state['held_step'])\n", '')]),
  # Replay and rate.
  ('hindsight inserted before the real episode', TR, [
    ('    with profiler.phase(\'replay_insert\'):\n      self.replay.insert(finished)\n    with profiler.phase(\'hindsight\'):\n      relabelled = self.queue.add(finished)\n',
     '    with profiler.phase(\'hindsight\'):\n      relabelled = self.queue.add(finished)\n'),
    ('        self.replay.insert(relabelled)\n', '        self.replay.insert(relabelled)\n    self.replay.insert(finished)\n')]),
  ('hindsight dropped', TR, [('        self.replay.insert(relabelled)\n', '        pass\n')]),
  ('budget counts real rows only', TR, [
    ('(self.replay.inserted - self.min_size)\n    return', '(self.replay.inserted - self.counters.hindsight_episodes * self.env_cfg.max_steps - self.min_size)\n    return')]),
  ('budget ignores min_size', TR, [('(self.replay.inserted - self.min_size)\n    return', '(self.replay.inserted)\n    return')]),
  ('budget one fused call ahead', TR, [('return self.replay.sampled + self.megabatch <= allowed', 'return self.replay.sampled <= allowed')]),
  ('budget one fused call behind', TR, [('return self.replay.sampled + self.megabatch <= allowed', 'return self.replay.sampled + 2 * self.megabatch <= allowed')]),
  ('at most one fused call per round', TR, [('    while self.budget_allows():\n', '    if self.budget_allows():\n')]),
  ('samples per insert 4', TR, [('allowed = self.cfg.samples_per_insert *', 'allowed = 4.0 *')]),
  ('features built per slice, twice for s', TR, [
    ('batch = Batch(build_graph(sample.obs, self.topo), build_graph(sample.next_obs, self.topo),', 'build_graph(sample.obs, self.topo)\n      batch = Batch(build_graph(sample.obs, self.topo), build_graph(sample.next_obs, self.topo),')]),
  ('s\' built from s', TR, [('build_graph(sample.next_obs, self.topo),', 'build_graph(sample.obs, self.topo),')]),
  # Stagger.
  ('partial rows kept', TR, [('      self.buffer.discard(due)\n', '')]),
  ('stagger counted as episodes', TR, [('      self.buffer.discard(due)\n', '      self.buffer.discard(due)\n      self.latches.finish(due)\n')]),
  ('no stagger', TR, [('self._stagger_at = offsets.masked_fill(offsets == 0, -1)', 'self._stagger_at = torch.full_like(offsets, -1)')]),
  ('one offset for all', TR, [('offsets = (self._all * env_cfg.max_steps) // W', 'offsets = (self._all > 0).long()')]),
  ('offset fires after a natural ending', TR, [('      self._stagger_at[ended] = -1  # Ended on its own before its offset: already out of step.\n', '')]),
  # Metrics.
  ('losses averaged', TR, [('losses = self.td3.fused_step(batch, summed=True)', 'losses = self.td3.fused_step(batch)')]),
  ('age measured after the learner call', TR, [('self._age_sum += grad_steps - sample.', 'self._age_sum += grad_steps + self.cfg.fused_steps - sample.')]),
  ('act statistics from the first sample', TR, [('self._stats_sum += sample.stats.mean(0).double()', 'self._stats_sum += sample.stats[0].double()')]),
  ('first report written', TR, [('if n and self._first_report_discarded:', 'if n:')]),
  # Failure handling and checkpoints.
  ('nonfinite ignored', TR, [('if not (math.isfinite(policy) and math.isfinite(critic)):', 'if False:')]),
  ('nonfinite raises without a checkpoint', TR, [('      self.save_checkpoint(reason, failed=True)\n', '')]),
  ('failed checkpoint over the good one', TR, [('path = self.run_dir / (FAILED_CHECKPOINT_NAME if failed else CHECKPOINT_NAME)', 'path = self.run_dir / CHECKPOINT_NAME')]),
  ('not atomic', TR, [("tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')", 'tmp = path')]),
  ('temporary left after an exception', TR, [('    tmp.unlink(missing_ok=True)\n', '')]),
  ('action stream not restored', TR, [("    self.act_rng.set_state(state['act'].cpu())\n", '')]),
  ('env stream not restored', TR, [("    self.env.generator.set_state(state['env'].cpu())\n", '')]),
  ('one seed for every stream', TR, [("return zlib.crc32(f'{self.cfg.seed}/{name}'.encode())", 'return self.cfg.seed')]),
  ('world counters not restored', TR, [("    self.latches.finished.copy_(state['finished'])\n", '')]),
  ('published step not restored', TR, [("    self.published_step = state['published_step']\n", '')]),
  ('resume accepts a failed checkpoint', TR, [("    if state['failed']:\n", '    if False:\n')]),
  ('resume ignores a changed config', TR, [('    self._check_resumable(state[\'config\'])\n', '')]),
  # TD3.
  ('init sets the global', TD, [("    self.grad_hook = grad_hook\n", "    self.grad_hook = grad_hook\n    torch.set_float32_matmul_precision(cfg.matmul_precision)\n")]),
  ('precision not restored on an exception', TD, [
    ('  try:\n    yield\n  finally:\n    torch.set_float32_matmul_precision(previous)\n', '  yield\n  torch.set_float32_matmul_precision(previous)\n')]),
  ('precision restored to high', TD, [('    torch.set_float32_matmul_precision(previous)\n', "    torch.set_float32_matmul_precision('high')\n")]),
  ('fused step without the scope', TD, [
    ('    with matmul_precision(self.cfg.matmul_precision):\n      for i in range(num_steps):\n        losses = self._update(megabatch.slice(i * size, (i + 1) * size))\n        critic, policy = critic + losses.critic, policy + losses.policy\n',
     '    for i in range(num_steps):\n      losses = self._update(megabatch.slice(i * size, (i + 1) * size))\n      critic, policy = critic + losses.critic, policy + losses.policy\n')]),
  ('acting copy is the online policy', TD, [('self.policy_acting = copy.deepcopy(self.policy).requires_grad_(False)', 'self.policy_acting = self.policy')]),
  ('refresh does nothing', TD, [('    torch._foreach_copy_(self._policy_acting_params, self._policy_params)\n', '    pass\n')]),
  ('acting under autocast', TD, [('    return self._policy_acting(graph)\n', '    with self._autocast:\n      return self._policy_acting(graph)\n')]),
  ('acting copy not in the checkpoint', TD, [("      'policy_acting': self.policy_acting.state_dict(),\n", ''),
                                             ("('policy', 'q1', 'q2', 'policy_acting', 'policy_target', 'q1_target', 'q2_target')", "('policy', 'q1', 'q2', 'policy_target', 'q1_target', 'q2_target')")]),
  # Evaluator.
  ('later episodes overwrite the first', EV, [('first = running & out.terminal', 'first = out.terminal')]),
  ('totals keep counting after the first terminal', EV, [
    ("counted['total_collision_robot_steps'] += running * out.collision_robot_steps", "counted['total_collision_robot_steps'] += out.collision_robot_steps")]),
  ('acceleration keeps counting after the first terminal', EV, [
    ("counted['total_mean_acceleration'] += running * out.acceleration", "counted['total_mean_acceleration'] += out.acceleration")]),
  ('length in steps', EV, [('length = torch.where(first, final.step + 1.0, length)', 'length = torch.where(first, final.step * 1.0, length)')]),
  ('task rate over robots', EV, [("metrics['num_targets_done'] / self.env_cfg.num_targets", "metrics['num_targets_done'] / self.env_cfg.world.num_robots")]),
  ('all done means any done', EV, [('(done == self.env_cfg.num_targets)', '(done > 0)')]),
  ('static scenes not forced', EV, [('      self.scene_ids[name] = ids\n', '      self.scene_ids[name] = None\n')]),
  ('static sets one world per scene', EV, [('repeats = max(1, round(static_worlds / size))', 'repeats = 1')]),
  ('evaluation holds the live policy', EV, [('      mine.copy_(theirs)\n', '      mine.data = theirs.data\n')]),
  ('evaluation at the ambient precision', EV, [("with matmul_precision('highest'):", 'with matmul_precision(torch.get_float32_matmul_precision()):')]),
  ('evaluation noise', EV, [('      action = self._forward(build_graph(env.observe(), self.topo))\n',
                             '      action = self._forward(build_graph(env.observe(), self.topo)) + 0.1\n')]),
  ('evaluation draws from the global stream', EV, [('      out = env.step(action)\n', '      out = env.step(action)\n      torch.rand(1)\n')]),
  # Profiler.
  ('phases wait on the GPU in normal running', PR, [('return _Phase(self, name) if self.in_window else _NULL', 'return _Phase(self, name)')]),
  ('.item() left patched', PR, [('    torch.Tensor.item = self._restore_item\n', '')]),
]


def make_overlay(root: Path, mutant: tuple | None) -> Path:
  overlay = root / 'roboballet_mjlab'
  overlay.mkdir(parents=True)
  (root / 'tests').symlink_to(PACKAGE.parents[1] / 'tests')  # The tests sit beside src/, not in the package.
  for child in PACKAGE.iterdir():
    if child.name not in ('train', 'agent', '__pycache__'):
      (overlay / child.name).symlink_to(child)
  for directory in ('train', 'agent'):
    (overlay / directory).mkdir()
    for path in (PACKAGE / directory).glob('*.py'):
      shutil.copy(path, overlay / directory / path.name)
  if mutant is not None:
    _, filename, edits = mutant
    path = overlay / filename
    text = path.read_text()
    for old, new in edits:
      if text.count(old) != 1:
        raise ValueError(f'{mutant[0]}: found {text.count(old)} matches of {old!r}')
      text = text.replace(old, new)
    path.write_text(text)
  return root


def run_suite(root: Path, filename: str) -> tuple[int, str, float]:
  env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
  start = time.time()
  try:
    done = subprocess.run(
      [sys.executable, '-m', 'pytest', '-x', '-q', '--tb=line', '--show-capture=no', '-p', 'no:cacheprovider', f'--rootdir={root}', *TESTS[filename],
       '-k', 'not provenance'],  # The overlay has no .git, and provenance has no mutant.
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
  print(f'{caught}  {mutant[0]:58s}  {summary}  ({seconds:.0f} s)', flush=True)
  return mutant[0], code, summary, seconds


def main() -> int:
  only = sys.argv[1:]
  mutants = [m for m in MUTANTS if not only or any(word in m[0] for word in only)]
  with tempfile.TemporaryDirectory(prefix='trainer_mutants_') as tmp:
    base = Path(tmp)
    for i, mutant in enumerate(mutants):
      check = make_overlay(base / f'c{i}', mutant)
      compile((check / 'roboballet_mjlab' / mutant[1]).read_text(), mutant[1], 'exec')
      shutil.rmtree(check)
    if not only:
      root = make_overlay(base / 'clean', None)
      for filename in (TR, EV):
        code, summary, seconds = run_suite(root, filename)
        print(f'unmutated overlay {filename}: exit {code}, {summary}, {seconds:.0f} s', flush=True)
        if code != 0:
          return 1
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
      results = list(pool.map(lambda x: run_mutant(x[1], base, x[0]), enumerate(mutants)))
  survivors = sum(code == 0 for _, code, _, _ in results)
  print(f'{len(results) - survivors} of {len(results)} mutants caught')
  return 1 if survivors else 0


if __name__ == '__main__':
  raise SystemExit(main())
