# Profiling

Where a training round's time goes, from each trainer's own profiler (`profile/*` and `rate/*` in `metrics.csv`). Means over each run and over its 3 seeds. One B200 per run, 4 robots, 10 targets, 30 obstacles, mesh arm. The original's numbers come from its TensorBoard logs and actor files.

## Rates

| | Original | TD3 port | FastTD3 | rsl-rl SAC | rsl-rl PPO |
|---|---:|---:|---:|---:|---:|
| Grad steps/s | 34.9 (seed 1), 37.6 and 38.1 (seeds 2, 3, to 1M) | 55.0 | 21.4 | 18.0 | 9.2 |
| Samples per grad step | 128 | 128 | 2,048 | 2,048 | 6,144 |
| Env steps/s | 859 (seed 1) | 1,351 | 10,974 | 9,209 | 11,250 |
| Parallel worlds | 64 actors | 64 | 1,024 | 1,024 | 1,024 |
| Host CPU | 64 cores allocated to 68 processes; use not logged | 1.00 core | 0.91 core | 0.82 core | 0.83 core |
| GPU memory, one seed | | 11.9 GiB | 17.2 GiB | 16.9 GiB | 54.3 GiB |

FastTD3, SAC and PPO use the 3-layer GNN of their presets. The original and the TD3 port use the paper's 7-layer GNN. PPO's grad step is one mini-batch of its 5 epochs over 4 mini-batches. Start-up takes 8 to 9.5 cores for about a minute.

## Where a round goes

Milliseconds per round, 3-seed means over every profile window. A FastTD3 round steps 1,024 worlds once and makes 2 updates. A TD3-port round steps 64 worlds once and makes the fused updates its rate limiter allows.

| Part | TD3 port | FastTD3 | rsl-rl SAC | rsl-rl PPO |
|---|---:|---:|---:|---:|
| Critic update | 39.6 (all of TD3) | 54.9 | 51.2 | 67.4 (the whole learner, with its forward passes) |
| Actor update | | 18.7 | 35.9, plus 12.5 action sampling (for the critic target and the actor loss) and 1.0 temperature | |
| Policy forward, acting | 1.6 | 7.4 | 8.0 | 20.1, policy and value |
| Simulation: step, forward, rollback, reset | 3.9 | 7.3 | 9.9 | 7.0 |
| Graph features, replay, hindsight, logging, evaluation, other | 1.9 | 6.7 | 9.8 | 3.4 |
| Round, measured | 47.0 | 94.9 | 128.2 | 97.8 |

The parts are disjoint and add up to the measured round.

**The learner is the bottleneck, not the simulator.** Updates are 78% of a FastTD3 round and 84% of a TD3-port round. Stepping 1,024 worlds in mjlab Warp takes 8% of a FastTD3 round.

## What profiling changed

| Finding | Measured |
|---|---|
| Hindsight relabelling read a GPU result at every one of 120 steps | Before: 44% of a FastTD3 round (39.5 to 46.8% over 4 runs on a shared GPU). After, with all checks in one GPU call and the loop on the CPU: 0.3% of a round in the final runs, about 6 ms a flush of 256 episodes. Output identical, checked against the goldens and CPU against GPU |
| Several GPUs per run | Two GPUs at the same total batch: 1.36 to 1.38x the grad-step rate, in a 5-minute test. Two GPUs at twice the batch: more grad steps to 50% (31,760 against 27,760), the same to 75 and 80%, fewer to 90% (47,760 against 55,760), one seed each. Neither comes near 2x for twice the GPUs, so each run uses one GPU, and spare GPUs run more seeds |
| A smaller batch for more grad steps per second | Batch 512 with 256 worlds: 1.6 to 1.8x the grad-step rate on a shared GPU, the updates alone 2 to 3x. At 40k grad steps, where the runs stopped, they had 0.000 and 0.002, against 0.55 to 0.58 for two batch-2,048 runs. A third batch-2,048 run was also at 0.002 at 40k and took off by 52k |

The original's learner has a capacity of about 90 grad steps/s and ran at 35 to 38 on average, held back by its 64 CPU actors and the replay rate limiter. The TD3 port keeps the same limiter, so its gain is on the simulator and acting side: 1.4 to 1.6x the grad-step rate, averaging one CPU core.
