# Profiling

Where a training round's time goes, from each trainer's own profiler (`profile/*` and `rate/*` in `metrics.csv`), averaged over the run. One B200 per run, 4 robots, 10 targets, 30 obstacles, mesh arm. The original's numbers come from its TensorBoard logs and actor files.

## Rates

| | Original | TD3 port | FastTD3 | rsl-rl SAC | rsl-rl PPO |
|---|---:|---:|---:|---:|---:|
| Grad steps/s | 34.9 | 55.9 | 20.5 | 18.1 | 9.1 |
| Samples per grad step | 128 | 128 | 2,048 | 2,048 | 6,144 |
| Env steps/s | 859 | 1,377 | 10,521 | 9,252 | 11,214 |
| Parallel worlds | 64 actors | 64 | 1,024 | 1,024 | 1,024 |
| Host CPU | 64 actor processes on 64 cores | 1 core | 0.84 core | 0.85 core | 0.81 core |
| GPU memory | | 11.9 GiB | 17.2 GiB | 16.9 GiB | 54.3 GiB |

FastTD3, SAC and PPO use the 3-layer GNN of their presets. The original and the TD3 port use the paper's 7-layer GNN. PPO's grad step is one mini-batch of its 5 epochs over 4 mini-batches.

## Where a round goes

Milliseconds per round, mean. A FastTD3 round steps 1,024 worlds once and makes 2 updates. A TD3-port round steps 64 worlds once and makes the fused updates its rate limiter allows.

| Part | TD3 port | FastTD3 | rsl-rl SAC | rsl-rl PPO |
|---|---:|---:|---:|---:|
| Critic update | 39.2 (all of TD3) | 57.3 | 50.7 | 46.8 (learner) |
| Actor update | | 19.4 | 35.5, plus 12.1 sampling and 1.0 temperature | |
| Policy forward, acting | 1.6 | 7.6 | 8.0 | 8.8, plus 11.4 value |
| Simulation: step, forward, rollback, reset | 3.8 | 7.9 | 9.3 | 7.1 |
| Graph features, replay, hindsight, logging, other | 2.0 | 8.2 | 12.3 | 19.8, of which 15.6 evaluation |
| Round, measured | 46.1 | 99.9 | 125.9 | 98.5 |

The parts are disjoint and add up to the measured round within 3%.

**The learner is the bottleneck, not the simulator.** Updates are 77% of a FastTD3 round and 85% of a TD3-port round. Stepping 1,024 worlds in mjlab Warp takes 8% of a FastTD3 round.

## What profiling changed

| Finding | Before | After |
|---|---|---|
| Hindsight relabelling read a GPU result at every one of 120 steps | 44% of a FastTD3 round; a 256-episode flush took 5.4 to 10.5 s on a shared GPU | One GPU call for all checks, then the loop on the CPU: 0.14 to 0.18 s a flush, 0.5% of a round. Output identical, checked against the goldens and CPU against GPU |
| Several GPUs per run | | Two GPUs at the same total batch: 1.34x the rate. Two GPUs at twice the batch: no fewer grad steps to a given task rate. So each run uses one GPU, and spare GPUs run more seeds |
| Smaller batch for more grad steps per second | | Batch 512: 2 to 3x the rate, but 0.002 task rate at 40k grad steps, where batch 2,048 had 0.4 to 0.6 |

The original's learner has a capacity of about 90 grad steps/s and ran at 34.9 on average, held back by its 64 CPU actors and the replay rate limiter. The TD3 port keeps the same limiter, so its gain is the simulator and acting side: 1.6x the grad-step rate with one CPU core instead of 64.
