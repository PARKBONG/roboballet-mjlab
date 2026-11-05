"""Measures full BalletEnv.step with saturated random actions. Prints, asserts nothing else.

Run alone on an idle GPU, with -s:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m pytest -s tests/test_env_throughput.py
"""

from __future__ import annotations

import time

import pytest
import torch

from roboballet_mjlab.config import EnvCfg, WorldCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from tests.test_sim_common import DEVICE, stored_scenes

WARMUP_STEPS, STEPS = 60, 400


@pytest.mark.parametrize('num_robots,num_worlds', [(4, 64), (4, 1024), (4, 4096), (8, 4096)])
def test_throughput(num_robots, num_worlds):
  cfg = EnvCfg(world=WorldCfg(num_robots=num_robots))
  env = BalletEnv(cfg, num_worlds, stored_scenes(num_robots), DEVICE, seed=51)
  gen = torch.Generator(device=DEVICE).manual_seed(52)
  shape = (num_worlds, num_robots, 7)
  # Generated up front, so that the timing is of the env and not of the action draw.
  actions = torch.randint(2, (WARMUP_STEPS + STEPS, *shape), generator=gen, device=DEVICE) * 2.0 - 1.0
  for i in range(WARMUP_STEPS):
    env.step(actions[i])
  torch.cuda.synchronize()
  launches = env.world.launches
  rollbacks = torch.zeros((), dtype=torch.long, device=DEVICE)
  terminals = torch.zeros((), dtype=torch.long, device=DEVICE)
  start = time.perf_counter()
  cpu_start = time.process_time()
  for i in range(WARMUP_STEPS, WARMUP_STEPS + STEPS):
    out = env.step(actions[i])
    rollbacks += out.collision_robot_steps.sum()
    terminals += out.terminal.sum()
  torch.cuda.synchronize()
  wall = time.perf_counter() - start
  cpu = time.process_time() - cpu_start
  rate = STEPS * num_worlds / wall
  print(f'\n{num_robots}r W={num_worlds}: {rate:,.0f} env-steps/s, {wall / STEPS * 1e3:.2f} ms/step, '
        f'{(env.world.launches - launches) / STEPS:.2f} launches/step, '
        f'{rollbacks.item() / (STEPS * num_worlds):.3f} rollbacks/env-step, '
        f'{terminals.item() / STEPS:.1f} resets/step, CPU/wall {cpu / wall:.2f}')
  assert rate > 0
