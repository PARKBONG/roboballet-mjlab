"""Gradient averaging with two real processes on one GPU, at small sizes. Averaging logic only.

Two ranks on one GPU go through gloo, because NCCL refuses a duplicate GPU. Nothing here measures
or claims multi-GPU throughput. This file is its own worker: python -m tests.test_fasttd3_dist.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist

from roboballet_mjlab.agent.features import Topology
from roboballet_mjlab.config import EnvCfg, WorldCfg
from roboballet_mjlab.fasttd3.agent import Batch, FastTD3
from roboballet_mjlab.fasttd3.config import FastTD3Cfg
from roboballet_mjlab.fasttd3.trainer import Trainer, average_gradients
from roboballet_mjlab.structs import Graph
from tests.test_fasttd3_trainer import MAX_STEPS, SMALL, T, script_args, script_env, small_pool, small_pool_file

REPO = Path(__file__).resolve().parents[1]
ENV = EnvCfg(world=WorldCfg(num_robots=2, num_obstacles=3), num_targets=2)
CFG = FastTD3Cfg(compile=False, bf16=False, matmul_precision='highest', weight_decay=0.1, seed=11)
UPDATES = 4


def flat(module) -> torch.Tensor:
  return torch.cat([p.detach().flatten() for p in module.parameters()])


def rank_batch(rank: int, update: int, topo: Topology, device: str = 'cuda:0') -> Batch:
  """A different batch per rank and update, from a seed the parent can repeat."""
  generator = torch.Generator(device=device).manual_seed(1000 * rank + update)
  rand = lambda *shape: torch.rand(shape, generator=generator, device=device)
  graph = Graph(rand(8, topo.num_nodes, 17), rand(8, topo.num_edges, 30), rand(8, 2))
  next_graph = Graph(rand(8, topo.num_nodes, 17), rand(8, topo.num_edges, 30), rand(8, 2))
  return Batch(graph, next_graph, rand(8, 2, 7) * 2 - 1, rand(8) * 0.2, rand(8) < 0.25)


def worker_average(out: Path) -> None:
  rank, world_size = dist.get_rank(), dist.get_world_size()
  topo = Topology(ENV, 'cuda:0')
  average = average_gradients(world_size)
  record: dict[str, list[torch.Tensor]] = {'before': [], 'after': []}

  def hook(vector):
    record['before'].append(vector.cpu().clone())
    result = average(vector)
    record['after'].append(result.cpu().clone())
    return result

  agent = FastTD3(SMALL, CFG, topo, 'cuda:0', grad_hook=hook, noise_seed=500 + rank)
  # Identical starts come from cfg.seed, whichever rank this is.
  start = (flat(agent.policy).cpu(), flat(agent.q1).cpu(), flat(agent.q2).cpu())
  for update in range(UPDATES):
    agent.update(rank_batch(rank, update, topo), update % 2 == 1)
  torch.save({
    'start': start, 'before': record['before'], 'after': record['after'],
    'policy': flat(agent.policy).cpu(), 'q1': flat(agent.q1).cpu(), 'q2': flat(agent.q2).cpu(),
    'q1_target': flat(agent.q1_target).cpu(), 'q2_target': flat(agent.q2_target).cpu(),
  }, out)


def worker_trainer(out: Path, poison_rank: int | None) -> None:
  """Two trainers on a real env. Each has its own env, replay, seeds and worlds."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  run_dir = out.parent / f'trainer_{rank}'
  cfg = FastTD3Cfg(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight_flush_episodes=4, compile=False, log_every_rounds=25,
                   eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5 + rank)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', run_dir, rank=rank, world_size=world_size)
  # The seeds differ here, so equal starts can only come from the broadcast.
  start = flat(t.agent.policy).cpu()
  learned_at = None
  if poison_rank is None:
    for _ in range(MAX_STEPS + 10):
      t.round()
      if learned_at is None and t.agent.grad_steps:
        learned_at = t.counters.rounds
    torch.save({
      'start': start, 'policy': flat(t.agent.policy).cpu(), 'q1': flat(t.agent.q1).cpu(), 'q2': flat(t.agent.q2).cpu(),
      'q1_target': flat(t.agent.q1_target).cpu(), 'q2_target': flat(t.agent.q2_target).cpu(), 'noise_scale': t.noise_scale.cpu(),
      'grad_steps': t.agent.grad_steps, 'learned_at': learned_at, 'env_steps': t.totals()['env_steps'],
      'first_action_noise': t._draw_normals(torch.zeros(4)).cpu(), 'inserted': t.replay.inserted, 'q': t.env.observe().robot.q.cpu(),
    }, out)
    return
  run_ok = False
  try:
    run = 0
    while True:
      t.round()
      run += 1
      if run == MAX_STEPS and rank == poison_rank:
        t._critic_sum[0] = float('nan')
      if run > MAX_STEPS + 2 * cfg.log_every_rounds:
        run_ok = True
        break
  except FloatingPointError as error:
    torch.save({'raised': str(error), 'failed_exists': t.checkpoint_path(failed=True).exists(), 'rounds': t.counters.rounds}, out)
  if run_ok:
    torch.save({'raised': None}, out)


def worker_uneven(out: Path) -> None:
  """Rank 1 has fewer worlds, so its ring holds a batch later. Neither may start before the other."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  cfg = FastTD3Cfg(num_worlds=8 if rank == 0 else 2, batch_size=60, replay_max_size=4000, hindsight=False, compile=False, log_every_rounds=25,
                   eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', out.parent / f'uneven_{rank}', rank=rank, world_size=world_size)
  sizes_when_learning_began = None
  rounds_when_learning_began = None
  for _ in range(MAX_STEPS + 35):
    t.round()
    if rounds_when_learning_began is None and t.agent.grad_steps:
      rounds_when_learning_began = t.counters.rounds
      sizes_when_learning_began = t.replay.size
  torch.save({'rounds': rounds_when_learning_began, 'size': sizes_when_learning_began, 'grad_steps': t.agent.grad_steps,
              'policy': flat(t.agent.policy).cpu()}, out)


def worker_stop(out: Path) -> None:
  """One signal, on rank 1 only. Both must stop, at the same round, and rank 0 must not hang."""
  rank, world_size = dist.get_rank(), dist.get_world_size()
  cfg = FastTD3Cfg(num_worlds=8, batch_size=16, replay_max_size=4000, hindsight=False, compile=False, log_every_rounds=25,
                   eval_every_rounds=10**9, checkpoint_every_s=1e9, profile=False, seed=5, total_env_rounds=10**6)
  t = Trainer(EnvCfg(num_targets=T), cfg, SMALL, small_pool(), 'cuda:0', out.parent / f'stop_{rank}', rank=rank, world_size=world_size)
  t.install_signal_handlers()
  original = t.round

  def signalled():
    original()
    if rank == 1 and t.counters.rounds == 7:
      os.kill(os.getpid(), signal.SIGTERM)

  t.round = signalled
  status = t.run()
  torch.save({'status': status, 'rounds': t.counters.rounds, 'saved': t.checkpoint_path().exists()}, out)


def spawn_two(*args: str) -> list[subprocess.Popen]:
  with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
  processes = []
  for rank in range(2):
    env = dict(os.environ, RANK=str(rank), WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORKER='1', PYTHONPATH=os.environ.get('PYTHONPATH', str(REPO / 'src')))
    processes.append(subprocess.Popen([sys.executable, '-m', 'tests.test_fasttd3_dist', *args], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
  return processes


def join(processes: list[subprocess.Popen], timeout: float = 240.0) -> None:
  for process in processes:
    try:
      output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
      for other in processes:
        other.kill()
      raise AssertionError('a rank hung, the failure this file exists to catch')
    assert process.returncode == 0, output[-3000:]


def test_gradients_are_averaged_and_the_ranks_stay_identical(tmp_path):
  processes = spawn_two('average', str(tmp_path))
  join(processes)
  a, b = (torch.load(tmp_path / f'average_{rank}.pt', weights_only=True) for rank in range(2))
  # Same weights to start, from the config seed.
  assert all(torch.equal(x, y) for x, y in zip(a['start'], b['start']))
  assert len(a['before']) == UPDATES + UPDATES // 2
  for k, (ga, gb, avg_a, avg_b) in enumerate(zip(a['before'], b['before'], a['after'], b['after'])):
    assert not torch.equal(ga, gb), f'gradient {k}: the batches differ, so the raw gradients must'
    assert torch.equal(avg_a, avg_b), f'gradient {k}'
    assert (avg_a - (ga + gb) / 2).abs().max().item() < 1e-5 * max(ga.abs().max().item(), 1.0), f'gradient {k}'
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(a[name], b[name]), name


def test_one_process_given_the_averaged_gradients_lands_on_the_same_weights(tmp_path):
  """So all the average changes is the gradient: same optimiser and same target update as on one process."""
  processes = spawn_two('average', str(tmp_path))
  join(processes)
  a = torch.load(tmp_path / 'average_0.pt', weights_only=True)
  torch.set_float32_matmul_precision('highest')
  topo = Topology(ENV, 'cuda:0')
  feed = iter(a['after'])
  agent = FastTD3(SMALL, CFG, topo, 'cuda:0', grad_hook=lambda vector: next(feed).to(vector.device), noise_seed=500)
  for update in range(UPDATES):
    agent.update(rank_batch(0, update, topo), update % 2 == 1)
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(a[name], flat(getattr(agent, name)).cpu()), name
  # Against no averaging at all, the weights move elsewhere.
  plain = FastTD3(SMALL, CFG, topo, 'cuda:0', noise_seed=500)
  for update in range(UPDATES):
    plain.update(rank_batch(0, update, topo), update % 2 == 1)
  assert not torch.equal(a['q1'], flat(plain.q1).cpu())


def test_two_trainers_on_a_real_env_stay_in_step(tmp_path):
  processes = spawn_two('trainer', str(tmp_path))
  join(processes)
  a, b = (torch.load(tmp_path / f'trainer_{rank}.pt', weights_only=True) for rank in range(2))
  assert torch.equal(a['start'], b['start'])
  assert a['grad_steps'] == b['grad_steps'] > 0
  assert a['learned_at'] == b['learned_at'], 'all ranks start updating together'
  for name in ('policy', 'q1', 'q2', 'q1_target', 'q2_target'):
    assert torch.equal(a[name], b[name]), name
  # Each has its own env, noise and data, and the counter is effective: two processes count double.
  assert not torch.equal(a['noise_scale'], b['noise_scale'])
  assert not torch.equal(a['q'], b['q'])
  assert a['env_steps'] == (MAX_STEPS + 10) * 8 * 2


def test_learning_starts_when_the_last_rank_is_ready(tmp_path):
  join(spawn_two('uneven', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'uneven_{rank}.pt', weights_only=True) for rank in range(2))
  # Rank 1's two worlds end at rounds 50 and 75, and 60 rows need both. Rank 0 alone would start at 56.
  assert a['rounds'] == b['rounds'] == 75
  assert a['size'] > b['size'] >= 60
  assert a['grad_steps'] == b['grad_steps'] and torch.equal(a['policy'], b['policy'])


def test_a_signal_to_one_rank_stops_both_at_the_same_round(tmp_path):
  join(spawn_two('stop', str(tmp_path)))
  a, b = (torch.load(tmp_path / f'stop_{rank}.pt', weights_only=True) for rank in range(2))
  assert b['status'] == f'signal {int(signal.SIGTERM)}' and a['status'] == 'another rank stopped'
  assert a['rounds'] == b['rounds'] == 25, 'the first report round after the signal'
  assert a['saved'] and b['saved']


def test_a_nonfinite_loss_on_one_rank_stops_both_without_a_hang(tmp_path):
  processes = spawn_two('poison', str(tmp_path))
  join(processes)
  for rank in range(2):
    state = torch.load(tmp_path / f'poison_{rank}.pt', weights_only=True)
    assert state['raised'] is not None and 'nonfinite' in state['raised'], rank
    assert state['failed_exists'], rank


def worker_main(argv: list[str]) -> None:
  mode, out = argv[0], Path(argv[1])
  rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
  dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)
  torch.set_float32_matmul_precision('highest')
  try:
    if mode == 'average':
      worker_average(out / f'average_{rank}.pt')
    elif mode == 'trainer':
      worker_trainer(out / f'trainer_{rank}.pt', None)
    elif mode == 'uneven':
      worker_uneven(out / f'uneven_{rank}.pt')
    elif mode == 'stop':
      worker_stop(out / f'stop_{rank}.pt')
    else:
      worker_trainer(out / f'poison_{rank}.pt', 1)
  finally:
    dist.destroy_process_group()


# The real command, through its --gpus launcher.


def physical_gpu() -> str:
  visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
  assert visible and ',' not in visible, 'run these tests with CUDA_VISIBLE_DEVICES set to one GPU'
  return visible


def launch_args(run_dir: Path, pool: Path, *extra: str) -> list[str]:
  return script_args(run_dir, pool, '--gpus', f'{physical_gpu()},{physical_gpu()}', *extra)


def test_the_launcher_runs_two_ranks_stops_on_sigterm_and_resumes(tmp_path):
  pool = small_pool_file(tmp_path / 'pool.pt')
  run_dir = tmp_path / 'run'
  log = open(tmp_path / 'log.txt', 'w')
  env = script_env()
  process = subprocess.Popen(launch_args(run_dir, pool), env=env, stdout=log, stderr=subprocess.STDOUT)
  try:
    deadline = time.monotonic() + 300
    while not (run_dir / 'metrics.csv').exists() or b'train/qf_loss' not in (run_dir / 'metrics.csv').read_bytes():
      assert process.poll() is None, (tmp_path / 'log.txt').read_text()[-3000:]
      assert time.monotonic() < deadline, 'no report'
      time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=180) == 0, (tmp_path / 'log.txt').read_text()[-3000:]
  finally:
    process.kill()
  zero = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
  one = torch.load(run_dir / 'rank1_checkpoint.pt', map_location='cpu', weights_only=True)
  assert zero['step'] == one['step'] and zero['counters']['rounds'] == one['counters']['rounds']
  assert zero['config']['world_size'] == 2
  for key in zero['policy']:
    assert torch.equal(zero['policy'][key], one['policy'][key]), key
  assert not torch.equal(zero['noise_scale'], one['noise_scale'])
  assert not torch.equal(zero['rng']['act'], one['rng']['act'])
  # Rank 0 alone writes the logs, and the rounds counted are the effective ones.
  values = {}
  for row in csv.DictReader(open(run_dir / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  assert values['counters/env_steps'] == values['counters/rounds'] * 8 * 2
  assert values['counters/sampled'] == values['counters/grad_steps'] * 16 * 2, 'effective as env_steps'
  assert not (run_dir / 'rank1').exists()
  # The effective counters are logged, and each file keeps its rank's own, which a resume counts on.
  summary = json.loads((run_dir / 'summary.json').read_text())
  assert summary['sampled'] == 2 * zero['counters']['sampled'] == 2 * 16 * zero['step']
  assert summary['inserted'] == 2 * zero['counters']['inserted'] > 0

  rounds = zero['counters']['rounds']
  done = subprocess.run(launch_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 30)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
  assert json.loads((run_dir / 'summary.json').read_text())['rounds'] == rounds + 30
  # An explicit path names rank 0's file, and rank 1 reads its own beside it. No rounds, so what they saved is what they read.
  saved = tmp_path / 'saved'
  saved.mkdir()
  shutil.copy(run_dir / 'checkpoint.pt', saved / 'at.pt')
  shutil.copy(run_dir / 'rank1_checkpoint.pt', saved / 'rank1_at.pt')
  read = [torch.load(saved / name, map_location='cpu', weights_only=True) for name in ('at.pt', 'rank1_at.pt')]
  done = subprocess.run(launch_args(run_dir, pool, '--resume', str(saved / 'at.pt'), '--total-env-rounds', str(rounds + 30)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
  wrote = [torch.load(run_dir / name, map_location='cpu', weights_only=True) for name in ('checkpoint.pt', 'rank1_checkpoint.pt')]
  for before, after in zip(read, wrote):
    assert torch.equal(before['rng']['act'], after['rng']['act']) and torch.equal(before['noise_scale'], after['noise_scale'])
    assert before['counters']['inserted'] == after['counters']['inserted'], 'not multiplied by the ranks on the way through'
  assert not torch.equal(wrote[0]['rng']['act'], wrote[1]['rng']['act']), 'each rank its own stream'
  (saved / 'rank1_at.pt').unlink()
  done = subprocess.run(launch_args(run_dir, pool, '--resume', str(saved / 'at.pt'), '--total-env-rounds', str(rounds + 30)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode != 0 and 'rank1_at.pt' in done.stdout + done.stderr
  # A resume refuses a changed process count.
  done = subprocess.run(script_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 60)), env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode != 0 and 'world_size' in done.stderr

  # Rank 1 fails alone here, and rank 0 would wait for it in a collective forever. The launcher ends it.
  (run_dir / 'rank1_checkpoint.pt').unlink()
  process = subprocess.Popen(launch_args(run_dir, pool, '--resume', '--total-env-rounds', str(rounds + 90)), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  output, _ = process.communicate(timeout=180)
  assert process.returncode != 0 and 'rank1_checkpoint.pt' in output, output[-2000:]


if __name__ == '__main__':
  if os.environ.get('WORKER') == '1':
    worker_main(sys.argv[1:])
