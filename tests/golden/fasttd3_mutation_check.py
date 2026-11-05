"""Injects bugs into temporary copies of fasttd3/ and checks that the tests catch each.

Run from the repository root, with one GPU visible:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tests.golden.fasttd3_mutation_check

Each mutant gets an overlay package: symlinks to everything in roboballet_mjlab except fasttd3/, which is
a copy. The real files are never written. Exit status is 0 only if the unmutated overlay passes and
every mutant fails. Names on the command line select mutants by substring.
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
AG, TR, CLI = 'fasttd3/agent.py', 'fasttd3/trainer.py', 'fasttd3/train.py'
AGENT_TESTS = 'tests/test_fasttd3_agent.py'
TRAINER_TESTS = 'tests/test_fasttd3_trainer.py'
DIST_TESTS = 'tests/test_fasttd3_dist.py'
ALL_TESTS = [AGENT_TESTS, TRAINER_TESTS, DIST_TESTS]
SUITES = {AG: [AGENT_TESTS, TRAINER_TESTS], TR: [TRAINER_TESTS, AGENT_TESTS], 'dist': [DIST_TESTS]}
WORKERS = 1  # One mutant at a time: the CPU is shared with timed runs.
TIMEOUT_S = 1800

TARGET_ACTOR = [
  ('    self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)\n',
   '    self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)\n    self.policy_target = copy.deepcopy(self.policy).requires_grad_(False)\n'),
  ('    self._policy = wrap(self.policy)\n', '    self._policy = wrap(self.policy)\n    self._policy_target = wrap(self.policy_target)\n'),
  ('next_action = (self._policy(batch.next_graph) + noise).clamp(-1.0, 1.0)', 'next_action = (self._policy_target(batch.next_graph) + noise).clamp(-1.0, 1.0)'),
  ('    soft_update(self._critic_target_params, self._critic_params, self.cfg.tau)\n',
   '    soft_update(self._critic_target_params, self._critic_params, self.cfg.tau)\n    soft_update(list(self.policy_target.parameters()), self._policy_params, self.cfg.tau)\n'),
]

# name, file, then (old, new) pairs. Each `old` must occur exactly once. The first seven are the brief's.
MUTANTS = [
  ('clip before scaling', AG, [('return normals.mul(sigma).clamp(-clip, clip)', 'return normals.clamp(-clip, clip).mul(sigma)')]),
  ('target actor used for the next action', AG, TARGET_ACTOR),
  ('CDQ picks the higher value', AG, [('expected_value(proj1, support)[:, None] < expected_value(proj2, support)[:, None]', 'expected_value(proj1, support)[:, None] > expected_value(proj2, support)[:, None]')]),
  ('wrong atom spacing', AG, [('delta_z = (v_max - v_min) / (num_atoms - 1)', 'delta_z = (v_max - v_min) / num_atoms')]),
  ('policy updated on i % 2 == 0', TR, [('return i % cfg.policy_frequency == 1', 'return i % cfg.policy_frequency == 0')]),
  ('bootstrapping on terminals', AG, [('bootstrap = (~batch.terminal).to(next_action.dtype)', 'bootstrap = torch.ones_like(batch.reward)')]),
  ('noise scale resampled every step', TR, [
    ('    if self._can_learn():\n      self._learn()\n',
     '    self.noise_scale = draw_scales(self._draw_fresh_scales(), self.cfg)\n    if self._can_learn():\n      self._learn()\n')]),
  # Projection.
  # Removing the clamp of target_z alone changes nothing, since the clamp of b covers it.
  ('neither clamp to the support', AG, [
    ('(rewards[:, None] + bootstrap[:, None] * discount * support).clamp(v_min, v_max)', '(rewards[:, None] + bootstrap[:, None] * discount * support)'),
    ('b = ((target_z - v_min) / delta_z).clamp(0.0, num_atoms - 1)', 'b = (target_z - v_min) / delta_z')]),
  ('b not clamped to the last atom', AG, [('b = ((target_z - v_min) / delta_z).clamp(0.0, num_atoms - 1)', 'b = (target_z - v_min) / delta_z')]),
  ('integer b: masks after the change', AG, [
    ('  l_mask = is_int & (l > 0)\n  u_mask = is_int & (l == 0)\n  l = torch.where(l_mask, l - 1, l)\n  u = torch.where(u_mask, u + 1, u)\n',
     '  l = torch.where(is_int & (l > 0), l - 1, l)\n  u = torch.where(is_int & (l == 0), u + 1, u)\n')]),
  ('integer b: mass lost', AG, [('  l_mask = is_int & (l > 0)\n  u_mask = is_int & (l == 0)\n  l = torch.where(l_mask, l - 1, l)\n  u = torch.where(u_mask, u + 1, u)\n', '')]),
  ('shares swapped', AG, [('(next_dist * (u.float() - b))[..., None]', '(next_dist * (b - l.float()))[..., None]')]),
  ('discount squared', AG, [('bootstrap[:, None] * discount * support', 'bootstrap[:, None] * discount * discount * support')]),
  ('index_add_ in place of the dense sum', AG, [
    ('  empty = next_dist.new_zeros(next_dist.shape[0], num_atoms, num_atoms)\n'
     '  lower = empty.scatter(2, l[..., None], (next_dist * (u.float() - b))[..., None])\n'
     '  upper = empty.scatter(2, u[..., None], (next_dist * (b - l.float()))[..., None])\n'
     '  return (lower + upper).sum(1)\n',
     '  proj = torch.zeros_like(next_dist)\n'
     '  offset = (torch.arange(next_dist.shape[0], device=next_dist.device) * num_atoms)[:, None]\n'
     '  proj.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))\n'
     '  proj.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))\n'
     '  return proj\n')]),
  # Clipped double Q and the losses.
  ('CDQ by atoms rather than by value', AG, [('  return torch.where(lower, proj1, proj2)\n', '  return torch.minimum(proj1, proj2)\n')]),
  ('CDQ on a tie takes the first', AG, [('expected_value(proj1, support)[:, None] < expected_value(proj2, support)', 'expected_value(proj1, support)[:, None] <= expected_value(proj2, support)')]),
  ('next distribution from the online critic', AG, [('F.softmax(self._q1_target(batch.next_graph, next_action), dim=1)', 'F.softmax(self._q1(batch.next_graph, next_action), dim=1)')]),
  ('critic loss averaged over the two', AG, [('  return loss1 + loss2\n', '  return (loss1 + loss2) / 2\n')]),
  ('critic loss on logits', AG, [('target1 * F.log_softmax(logits1, dim=1)', 'target1 * logits1')]),
  ('actor loss on the first critic only', AG, [('value = torch.minimum(value1, value2) if use_cdq else (value1 + value2) / 2.0', 'value = value1')]),
  ('actor loss on logits, not expected values', AG, [('value1 = expected_value(F.softmax(logits1, dim=1), support)', 'value1 = logits1.mean(1)')]),
  ('actor loss with the wrong sign', AG, [('  return -value.mean()\n', '  return value.mean()\n')]),
  # Next action.
  ('no noise on the next action', AG, [('next_action = (self._policy(batch.next_graph) + noise).clamp(-1.0, 1.0)', 'next_action = self._policy(batch.next_graph).clamp(-1.0, 1.0)')]),
  ('next action not clamped', AG, [('next_action = (self._policy(batch.next_graph) + noise).clamp(-1.0, 1.0)', 'next_action = self._policy(batch.next_graph) + noise')]),
  ('next action from s, not s\'', AG, [('next_action = (self._policy(batch.next_graph) + noise)', 'next_action = (self._policy(batch.graph) + noise)')]),
  # Optimiser and target.
  ('Adam in place of AdamW', AG, [('self.critic_optimizer = torch.optim.AdamW(self._critic_params, lr=cfg.lr_critic, **adamw)', 'self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=cfg.lr_critic, weight_decay=0.1)')]),
  ('no weight decay', AG, [('adamw = dict(weight_decay=cfg.weight_decay,', 'adamw = dict(weight_decay=0.0,')]),
  ('gradient clipped', AG, [('    optimizer.step()\n', '    torch.nn.utils.clip_grad_norm_(used, 1.0)\n    optimizer.step()\n')]),
  ('tau hard-coded small', AG, [('soft_update(self._critic_target_params, self._critic_params, self.cfg.tau)', 'soft_update(self._critic_target_params, self._critic_params, 0.005)')]),
  ('tau swapped', AG, [('  torch._foreach_mul_(targets, 1.0 - tau)\n  torch._foreach_add_(targets, online, alpha=tau)\n', '  torch._foreach_mul_(targets, tau)\n  torch._foreach_add_(targets, online, alpha=1.0 - tau)\n')]),
  ('target updated before the critic', AG, [
    ('      critic = self._update_critic(batch)\n      actor = self._update_actor(batch) if update_policy else None\n      self._update_target()\n',
     '      self._update_target()\n      critic = self._update_critic(batch)\n      actor = self._update_actor(batch) if update_policy else None\n')]),
  ('target updated only with the actor', AG, [('      self._update_target()\n', '      if update_policy:\n        self._update_target()\n')]),
  ('actor before the critic', AG, [
    ('      critic = self._update_critic(batch)\n      actor = self._update_actor(batch) if update_policy else None\n',
     '      actor = self._update_actor(batch) if update_policy else None\n      critic = self._update_critic(batch)\n')]),
  ('unused parameter decays', AG, [('    grads = [g for g in grads if g is not None]\n', '    used = params\n    grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]\n')]),
  ('frozen critics do not share weights', AG, [('      frozen.data = live.data\n', '      pass\n')]),
  # Float modes.
  ('learner not in bf16', AG, [('dtype=torch.bfloat16, enabled=cfg.bf16)', 'dtype=torch.bfloat16, enabled=False)')]),
  ('acting under autocast', AG, [('    return self._policy(graph)\n', '    with self._autocast:\n      return self._policy(graph)\n')]),
  ('init sets the global precision', AG, [('    self.grad_hook = grad_hook\n', '    self.grad_hook = grad_hook\n    torch.set_float32_matmul_precision(cfg.matmul_precision)\n')]),
  ('precision not restored on an exception', AG, [('  try:\n    yield\n  finally:\n    torch.set_float32_matmul_precision(previous)\n', '  yield\n  torch.set_float32_matmul_precision(previous)\n')]),
  ('update without the precision scope', AG, [
    ('    with matmul_precision(self.cfg.matmul_precision):\n      critic = self._update_critic(batch)\n      actor = self._update_actor(batch) if update_policy else None\n      self._update_target()\n',
     '    critic = self._update_critic(batch)\n    actor = self._update_actor(batch) if update_policy else None\n    self._update_target()\n')]),
  # Schedule and acting.
  ('first update a round early', TR, [('return global_step > cfg.learning_starts', 'return global_step >= cfg.learning_starts')]),
  ('updates before a batch is in the ring', TR, [('and self.replay.inserted >= self._min_rows\n', 'and self.replay.inserted >= 1\n')]),
  ('one batch for both updates', TR, [
    ('    for i in range(cfg.num_updates):\n      with profiler.phase(\'sampling\'):\n        sample = self.replay.sample(cfg.batch_size, self.sample_rng, cfg.n_step, cfg.gamma)\n',
     '    sample = self.replay.sample(cfg.batch_size, self.sample_rng, cfg.n_step, cfg.gamma)\n    for i in range(cfg.num_updates):\n')]),
  ('three updates per round', TR, [('    for i in range(cfg.num_updates):\n', '    for i in range(cfg.num_updates + 1):\n')]),
  ('s\' built from s', TR, [('build_graph(sample.next_obs, self.topo)', 'build_graph(sample.obs, self.topo)')]),
  ('stored action not clamped', TR, [('return (mean + scales * normals).clamp(-1.0, 1.0)', 'return mean + scales * normals')]),
  ('one noise scale for all worlds', TR, [('return torch.rand(self.num_worlds, 1, 1, generator=self.act_rng, device=self.device)', 'return torch.rand(1, 1, 1, generator=self.act_rng, device=self.device).expand(self.num_worlds, 1, 1)')]),
  ('noise scale never resampled', TR, [('      self.noise_scale = resample_scales(self.noise_scale, out.terminal, self._draw_fresh_scales(), self.cfg)\n', '')]),
  ('all worlds resample when one ends', TR, [('resample_scales(self.noise_scale, out.terminal, self._draw_fresh_scales(), self.cfg)', 'draw_scales(self._draw_fresh_scales(), self.cfg)')]),
  ('stagger reset resamples', TR, [('      self.buffer.discard(due)\n', '      self.buffer.discard(due)\n      self.noise_scale = draw_scales(self._draw_fresh_scales(), self.cfg)\n')]),
  ('noise range wrong', TR, [('return fresh * (cfg.std_max - cfg.std_min) + cfg.std_min', 'return fresh * cfg.std_max')]),
  ('a stale acting copy', TR, [('      mean = self.agent.act(graph)\n', '      self._stale = getattr(self, \'_stale\', None) or __import__(\'copy\').deepcopy(self.agent.policy)\n      mean = self._stale(graph)\n')]),
  # Replay.
  ('hindsight inserted before the real episode', TR, [
    ('    with profiler.phase(\'replay_insert\'):\n      self.replay.insert(finished)\n    if self.queue is None:\n      return\n', '    if self.queue is None:\n      self.replay.insert(finished)\n      return\n'),
    ('        self.replay.insert(relabelled)\n', '        self.replay.insert(relabelled)\n    self.replay.insert(finished)\n')]),
  ('hindsight dropped', TR, [('        self.replay.insert(relabelled)\n', '        pass\n')]),
  ('partial rows kept', TR, [('      self.buffer.discard(due)\n', '')]),
  ('no stagger', TR, [('self._stagger_at = offsets.masked_fill(offsets == 0, -1)', 'self._stagger_at = torch.full_like(offsets, -1)')]),
  # Failure handling and checkpoints.
  ('nonfinite ignored', TR, [('    nonfinite = not all(math.isfinite(x) for x in (*critic, *actor))\n', '    nonfinite = False\n')]),
  ('nonfinite raises without a checkpoint', TR, [('      self.save_checkpoint(reason, failed=True)\n', '')]),
  ('failed checkpoint over the good one', TR, [('name = FAILED_CHECKPOINT_NAME if failed else CHECKPOINT_NAME', 'name = CHECKPOINT_NAME')]),
  ('not atomic', TR, [("tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')", 'tmp = path')]),
  ('temporary left after an exception', TR, [('    tmp.unlink(missing_ok=True)\n', '')]),
  ('action stream not restored', TR, [("    self.act_rng.set_state(state['act'].cpu())\n", '')]),
  ('env stream not restored', TR, [("    self.env.generator.set_state(state['env'].cpu())\n", '')]),
  ('sample stream not restored', TR, [("    self.sample_rng.set_state(state['sample'].cpu())\n", '')]),
  ('noise scales not restored', TR, [("    self.noise_scale = state['noise_scale'].to(self.device)\n", '')]),
  ('counters not restored', TR, [("      rounds=saved['rounds'], ", '      ')]),
  ('resume accepts a failed checkpoint', TR, [("    if state['failed']:\n", '    if False:\n')]),
  ('resume ignores a changed config', TR, [("    self._check_resumable(state['config'])\n", '')]),
  ('a resume waits only for a batch', TR, [('    self._min_rows = max(self.cfg.batch_size, self.cfg.resume_min_rows)\n', '')]),
  ('a fresh run waits as a resume does', TR, [('    self._min_rows = cfg.batch_size  # load_checkpoint raises it.\n', '    self._min_rows = max(cfg.batch_size, cfg.resume_min_rows)\n')]),
  ('a ring below resume_min_rows waits forever', TR, [('and self.replay.inserted >= self._min_rows\n', 'and self.replay.size >= self._min_rows\n')]),
  ('a field newer than the checkpoint refuses it', TR, [("then = saved[section].get(key, getattr(FastTD3Cfg, key, None) if section == 'train' else None)", 'then = saved[section].get(key)')]),
  ('a changed field accepted when the checkpoint lacks it', TR, [("then = saved[section].get(key, getattr(FastTD3Cfg, key, None) if section == 'train' else None)", 'then = saved[section].get(key, value)')]),
  ('no snapshots while learning', TR, [('      self._learn()\n      self._snapshot_if_due()\n', '      self._learn()\n')]),
  ('no snapshot of the untrained policy', TR, [('    self._snapshot_if_due()  # The untrained policy, on a new run.\n', '')]),
  ('a resume rewrites the snapshot it starts from', TR, [("    self._next_snapshot = (self.agent.grad_steps // every + 1) * every if every > 0 else 0\n", '')]),
  ('snapshots from every rank', TR, [('if self.rank != 0 or every <= 0 or step < self._next_snapshot:', 'if every <= 0 or step < self._next_snapshot:')]),
  ('snapshots holding the critics', TR, [("atomic_save({'step': step, 'policy': state, 'model': dataclasses.asdict(self.model_cfg)}, directory", "atomic_save({'step': step, 'policy': state, 'q1': self.agent.q1.state_dict(), 'model': dataclasses.asdict(self.model_cfg)}, directory")]),
  ('a snapshot does not say its model', TR, [("{'step': step, 'policy': state, 'model': dataclasses.asdict(self.model_cfg)}", "{'step': step, 'policy': state}")]),
  ('stale temporaries kept', TR, [('      stale.unlink()\n', '      pass\n')]),
  ('profiler window left open', TR, [('        self.profiler._close_window()\n', '        pass\n')]),
  ('one seed for every stream', TR, [("return zlib.crc32(f'{self.cfg.seed}/{self.rank}/{name}'.encode())", 'return self.cfg.seed')]),
  ('evaluation touches the action stream', TR, [('      results = self.evaluator.evaluate(self.agent.policy, step)\n', '      results = self.evaluator.evaluate(self.agent.policy, step)\n      self._draw_fresh_scales()\n')]),
  # Reads in the hot path.
  ('a loss read back each update', TR, [('      self._critic_updates += 1\n', '      self._critic_updates += 1\n      critic.loss.item()\n')]),
  # Multi-process.
  ('gradients summed, not averaged', 'dist', TR, [('    return flat / world_size\n', '    return flat\n')]),
  ('no all-reduce', 'dist', TR, [('    dist.all_reduce(flat, op=dist.ReduceOp.SUM)\n', '')]),
  ('no broadcast at the start', 'dist', TR, [('    if world_size > 1:\n      self._broadcast_parameters()\n', '')]),
  ('same streams on every rank', 'dist', TR, [("return zlib.crc32(f'{self.cfg.seed}/{self.rank}/{name}'.encode())", "return zlib.crc32(f'{self.cfg.seed}/{name}'.encode())")]),
  ('learning starts by each rank alone', 'dist', TR, [('      dist.all_reduce(flag, op=dist.ReduceOp.MIN)\n      ready = bool(flag)\n', '      ready = bool(flag)\n')]),
  ('nonfinite not agreed', 'dist', TR, [('      dist.all_reduce(flag, op=dist.ReduceOp.MAX)\n      nonfinite = bool(flag)\n', '      nonfinite = bool(flag)\n')]),
  ('stop not agreed', 'dist', TR, [('      dist.all_reduce(flags, op=dist.ReduceOp.MAX)\n', '')]),
  ('env steps not effective', 'dist', TR, [("'env_steps': c.rounds * self.num_worlds * self.world_size,", "'env_steps': c.rounds * self.num_worlds,")]),
  ('every rank reads an explicit path', 'dist', CLI, [('rank_file(args.resume, rank) if args.resume else trainer.checkpoint_path()', 'args.resume or trainer.checkpoint_path()')]),
  ('inserted not effective', 'dist', TR, [("'inserted': self._inserted() * self.world_size,", "'inserted': self._inserted(),")]),
  ('sampled not effective', 'dist', TR, [("'sampled': self._sampled() * self.world_size,", "'sampled': self._sampled(),")]),
  ('checkpoint holds the effective inserted', 'dist', TR, [("'inserted': self._inserted(),\n", "'inserted': totals['inserted'],\n")]),
  ('every rank writes one checkpoint name', 'dist', TR, [('return rank_file(self.run_dir / name, self.rank)', 'return self.run_dir / name')]),
]


def parse(mutant: tuple) -> tuple[str, str, str, list]:
  """(name, suite, file, edits)."""
  if mutant[1] == 'dist':
    return mutant[0], 'dist', mutant[2], mutant[3]
  return mutant[0], mutant[1], mutant[1], mutant[2]


def make_overlay(root: Path, mutant: tuple | None) -> Path:
  overlay = root / 'roboballet_mjlab'
  overlay.mkdir(parents=True)
  (root / 'tests').symlink_to(PACKAGE.parents[1] / 'tests')  # The tests sit beside src/, not in the package.
  for child in PACKAGE.iterdir():
    if child.name not in ('fasttd3', '__pycache__'):
      (overlay / child.name).symlink_to(child)
  (overlay / 'fasttd3').mkdir()
  for path in (PACKAGE / 'fasttd3').glob('*.py'):
    shutil.copy(path, overlay / 'fasttd3' / path.name)
  if mutant is not None:
    name, _, filename, edits = parse(mutant)
    path = overlay / filename
    text = path.read_text()
    for old, new in edits:
      if text.count(old) != 1:
        raise ValueError(f'{name}: found {text.count(old)} matches of {old!r}')
      text = text.replace(old, new)
    path.write_text(text)
  return root


def run_suite(root: Path, tests: list[str]) -> tuple[int, str, float]:
  env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
  start = time.time()
  try:
    done = subprocess.run(
      [sys.executable, '-m', 'pytest', '-x', '-q', '--tb=line', '--show-capture=no', '-p', 'no:cacheprovider', f'--rootdir={root}', *tests],
      cwd=root, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
  except subprocess.TimeoutExpired:
    return -1, 'timed out', time.time() - start
  found = re.search(r'^(?:FAILED|ERROR) (\S+)', done.stdout, re.M)
  summary = found.group(1).split('::')[-1] if found else done.stdout.strip().splitlines()[-1]
  return done.returncode, summary, time.time() - start


def run_mutant(mutant: tuple, base: Path, index: int) -> tuple[str, int, str, float]:
  name, suite, _, _ = parse(mutant)
  root = make_overlay(base / f'm{index}', mutant)
  try:
    code, summary, seconds = run_suite(root, SUITES[suite])
  finally:
    shutil.rmtree(root)
  caught = 'caught  ' if code != 0 else 'SURVIVED'
  print(f'{caught}  {name:50s}  {summary}  ({seconds:.0f} s)', flush=True)
  return name, code, summary, seconds


def main() -> int:
  only = sys.argv[1:]
  mutants = [m for m in MUTANTS if not only or any(word in m[0] for word in only)]
  with tempfile.TemporaryDirectory(prefix='fasttd3_mutants_') as tmp:
    base = Path(tmp)
    for i, mutant in enumerate(mutants):
      check = make_overlay(base / f'c{i}', mutant)
      compile((check / 'roboballet_mjlab' / parse(mutant)[2]).read_text(), parse(mutant)[2], 'exec')
      shutil.rmtree(check)
    if not only:
      root = make_overlay(base / 'clean', None)
      code, summary, seconds = run_suite(root, ALL_TESTS)
      print(f'unmutated overlay: exit {code}, {summary}, {seconds:.0f} s', flush=True)
      if code != 0:
        return 1
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
      results = list(pool.map(lambda x: run_mutant(x[1], base, x[0]), enumerate(mutants)))
  survivors = sum(code == 0 for _, code, _, _ in results)
  print(f'{len(results) - survivors} of {len(results)} mutants caught')
  return 1 if survivors else 0


if __name__ == '__main__':
  raise SystemExit(main())
