# Speed and resources: evidence for the README, verified

> Copied into this repository on 2026-11-06 from the development tree, where the package was named `mjlab_ballet` (here `src/roboballet_mjlab`). Paths in this file refer to that tree. Run directories, scratch directories and `docs/reviews/xval_*` are not part of this repository. The text is unchanged.

Snapshot: 2026-11-05. Tables in section 2 use port `metrics.csv` cut at 13:12:49 KST, so they compare with the collector's read. I re-read the files at 13:26, 13:36, 13:41:16 and 13:48:08, and rows that use those reads say so. The runs are still going, so a number that moved between reads is not an error.

Tags: **[M]** measured, read directly from a log or tool. **[D]** derived, with the inputs named. **[P]** quoted from paper text. **[DOC]** stated in a status document and not re-derived. **unverified**: read once by the collector, or stated somewhere, and I could not re-read it. No number in this file is read from a paper figure.

Paths: `RUN` = `runs/paper_repro/paper_4r10t_s1`. `S` = `runs/mjlab_port_analysis`. `B` = `S/bench/results`. `GT` = `roboballet/mjlab_ballet/docs/reports/gpu_benchmark_tables.md`. `TS` = `roboballet/mjlab_ballet/docs/reviews/trainer_status.md`. `FS` = `roboballet/mjlab_ballet/docs/reviews/fasttd3_status.md`. `DESIGN` = `roboballet/mjlab_ballet/docs/DESIGN.md`. `PB` = `S/reports/paper_baseline.md`. All are under `multirobot-project/`.

Time zero is process launch for both systems, unless a row says otherwise. Reference launch 19:30:46 (`RUN/provenance.txt:1`). Port launch 10:30:39 (`ps lstart`, read 13:29 [M]).

**Two clocks for a reference milestone.** The reference evaluator fetches weights at step S, runs 50 episodes on one CPU core, and writes the TensorBoard row when it finishes. That row lands 153-195 s after the learner reached S at the milestones [D: `eval/training_dist/*` wall_time minus `train/total_step_time_ms` wall_time at the same step]. The port evaluates in-process in about 3 s, so its row carries no such lag. "Eval row" is the wall_time of the row. "Learner clock" is when the learner reached the milestone's step. Learner clock is the fair one for training speed.

## 0. Verdict

| Claim for the README | State | Numbers |
|---|---|---|
| Faster than the original to a task rate, 1 GPU vs 1 GPU + 64 cores | **Small gain at 90% only, so far** | Original time / port time on the learner clock. 50%: 0.88-1.06x, median 1.00x, no gain. 75%: 1.01-1.27x, median 1.09x. 90%: 1.11-1.16x, median 1.15x. 95%: one run of four so far, g8a_ref0 at 3.25 h, 1.07x; the other three pending at 3.29 h (read 13:48). Over every clock and window definition, 90% spans 1.09-1.18x, section 2.3. DESIGN:143 expected 1.4 to 1.9x. Not met to 90%. A whole-run figure does not exist |
| Faster per gradient step | **False early, true later** | 0-100k: reference 64.0 vs port 56-57.5 grad/s. 400-500k, same step range: reference 42.3 vs port 55.1-56.1 grad/s, 1.30-1.33x. The rate crosses in the 100-200k stage, 200-300k on g8a_s1. Cumulative wall clock crosses at 222-228k on three g8a runs and 328k on g8a_s1 |
| Same pace on far less CPU | **The strongest honest claim, with limits** | Port 0.99-1.00 core sustained [M], 7.8-9.5 cores during startup [D], 72 cores available and **never run pinned**. Reference 64 cores allocated. 3 of its 68 processes are logged, 3.3-4.1 cores [D]. The rest are not logged, and 64 is a ceiling rather than a measured use. Both one B200 |
| Far less host memory | **New, weakly attributed** | Port 3.0 GiB peak RSS [M]. Reference: 3 logged processes peak at 27.6 GiB [M]. Machine-wide memory rose 455 GiB across the run, 347 GiB of it in one minute at launch [D, timing only] |
| Simulator is much faster | **True on the sphere arm at large world counts. No like-for-like figure** | The 3-22x ratios are sphere arm against the original's hull arm. With the mesh arm on both sides: 0.39-0.56x env step at W=64, 1.4x with policy at W=64, 5.2-7.5x with policy at about 1,100 worlds, the last from evaluations. See section 3 |
| Simulator speed does not set training speed at parity | True: learner bound | TD3 step is 84-90% of a training round [D from M windows]. Env cost going to zero would give at most 63-69 grad/s against 55 [D] |
| Other agents are faster | **No evidence** | None has passed task rate 0.0001 on 4r/10t/30o [M, 13:40]. Only FastTD3 default has a rate from an unshared GPU |

Things that contradict the brief or a document, raw data winning:

| # | Statement | Raw data |
|---|---|---|
| C1 | The load spike of 214-221 disturbed these runs | **It did not.** The only source is FS:294: started 09:56, load 221 by about 10:05, stopped 10:06. Current runs started 10:30:39, 24 minutes later. It hit the previous `v2_g8*` runs, which were killed at 10:30:35 (`runs/v2_g8a_s1.log`, signal 15) [M]. The value 214 appears in no status document. Load average is not logged anywhere |
| C2 | About 1.2 cores per run | 1.2 is a lifetime mean inflated by startup. **Sustained is 0.99 core** over 13:05:30-13:41:16, 2,121-2,123 CPU-s in 2,146 s [D from 2 `ps`]. See 2.5 |
| C3 | 11.5 GiB GPU memory per run | **11.23 GiB on mesh_tool, 6.51 GiB on spheres** [D from `nvidia-smi` MiB, re-read 13:29]. 11.5 was MiB read as GiB. See 2.5 |
| C4 | DESIGN:143, 1.4 to 1.9x gain at parity | Measured 1.11-1.16x at 90%. DESIGN:43 says a faster env buys "at most about 1.9x". That is a ceiling, and these numbers do not contradict it. DESIGN:143 does not say which milestone it means. The rate ratio at 400-500k is already 1.30-1.33x, and a whole-run ratio is unmeasured |
| C5 | PB 10.2 floors T3, T4 (PB:713-714) | **T3 is not tested by these runs.** It is the learner alone. Inferred from windows it is 62.6-68.3 train/s with sampling and graph build included [D: `td3_step_ms` + `sampling_ms` + `feature_rebuild_ms`, 2.5 grad steps per round], above the floor of 60, and agrees with PB:713's 62-65. **T4 is missed on run means**: 53.3-55.4 grad/s, 1,366-1,419 inserts/s. Its 5-min wording is met only early: best 5 min 1,497-1,573 inserts/s at 10:40-10:46, best 30 min 1,453-1,478 [D] |
| C6 | Reference milestones 3.41 h (PB:350, DESIGN:32) vs 3.46 h (DESIGN:412) | Both reproduce, under different definitions. See 2.3. From launch it is 3.52 h on the eval row, 3.46 h on the learner clock |
| C7 | PB:283 wall clock 14.99 h | 19:30:46 to 10:30:36 is 14.997 h [D]. Rounds to 15.00, not 14.99 |
| C8 | TS:157 "G8a at 1.87M = 9.6 h" | An extrapolation at a pre-fix rate. TS:277 says it was not measured again. A 5-minute late-training probe on pre-fix code gave 55.3 grad/s on `mesh_tool` with the reference weights (TS:152-156) [DOC], so it is a hint and not a measurement of this code |
| C9 | DESIGN:35 and 36 as env-step throughput | Both are prototypes, not the final path. `run_e.sh` was meant to re-run them quietly on the final broadphase, and `B/env_proto2.jsonl`, `B/reset.jsonl` and `B/run_e.log` do not exist [M] |
| C10 | 41,000 env/s "64-core bound" (PB:433) | 645/s x 64, never run at 64 cores. A second measurement gives 479/s per core, so 30,600 (`B/baseline_cpu.jsonl:1`) |
| C11 | PB:715 T5 floor, 95% in <= 2.5 h | **Missed.** At 2.5 h (13:00:39) the 50k window was 0.918-0.923. The first crossing is g8a_ref0 at 3.25 h, and the other three were at 0.9459-0.9499 at 3.29 h [D, read 13:48:08]. The collector's list left this out |
| C12 | PB:315 "learner is actor-limited", repeated as fact | **Not established for the first 3 h.** Actors spend 42-72% of their cycle inside the episode write in that span, section 1. The write includes any block by the Reverb rate limiter, so the actors may be held back rather than holding the learner back |
| C13 | Machine memory "rose smoothly from 24.0 to 25.8%" | Those are 2-hour bin means. The series starts at 5.76% at 19:32:07 and reaches 21.11% by 19:33:07 [M, W&B `memory_percent`] |

## 1. Reference run

| Quantity | Value | Tag | Source |
|---|---|---|---|
| Launch | 11-03 19:30:46 KST | M | `RUN/provenance.txt:1` |
| Stop | 11-04 10:30:36, user termination, step 1,878,880 | M | `RUN/logs/learner/0:687894`, `RUN/logs/replay/0` tail |
| Wall clock | 14.997 h | D | The two rows above |
| Launch to first TB event | 200 s | D | `eval/static_dist` first wall_time minus launch |
| Launch to first logged grad step (2,000) | 268 s | D | `train/total_step_time_ms` first wall_time minus launch |
| Grad steps/s, whole run | 34.94 | D | TB `train/total_step_time_ms`, 2,000 to 1,878,000 over 53,689 s |
| Env steps/s, whole run | 859.4 on the learner's span. 859.3 on the actors' span. PB:295 says 859 | D | `S/scripts_baseline/actor.npz`, 46,288,183 transitions, matching the awk count in `S/verify/baseline/v12_actor.log` |
| Inserts/s, whole run | 894.6 on the learner's span. 896.7 on the actors' span, which is PB:295's 897 | D | Same, plus 2,014,080 hindsight rows |
| Samples per insert | 4.98 | D | 1,878,880 x 128 / 48,302,263 |
| Per actor | 13.4 env/s | D | 859.4 / 64 |
| Processes | 68 plus launcher: 64 actors, 2 evaluators, 1 learner, 1 replay | M | `ls RUN/logs/{actor,evaluator,learner,replay}` |
| CPU pinning | Cores 0-63 of 72, so 68 processes on 64 cores | D, weak | `runs/paper_repro/scripts/run_one.sh:42`, **edited 11-04 11:15, after the run**. `provenance.txt` records the command but not this line. PB:270 says the same. No record from the run itself |
| Nice | 10 | D, weak | Same line, same caveat |
| CPU model | AMD EPYC 9365, 2 sockets x 36 cores, 1 thread per core | M | `lscpu` today. `cpu_count` 72 in `RUN/wandb/*/files/wandb-metadata.json` |
| GPU | One NVIDIA B200, GPU 1 | M | `RUN/provenance.txt:2` (argument `1`), `:604-609` |
| Learner CPU | 1.46, 1.42, 1.34 cores over 0-3, 3-8, 8-15 h | D | W&B `cpu` x 72 / 100. Definition confirmed: process `cpu_percent` / `cpu_count` (`venv/.../wandb/sdk/internal/system/assets/cpu.py:44`). 3,593 records |
| Evaluator CPU, each | 1.31-1.34 cores over 0-3 h, 0.97-0.98 over 8-15 h | D | Same, the 2 evaluator `.wandb` files. The collector left these out |
| Actors and replay CPU | **Not logged** | | No W&B for those processes |
| Learner host memory | RSS 3.1 GiB first record, 9.3 GiB p5, 16.65 GiB max | M | W&B `proc.memory.rssMB` |
| Evaluator host memory | 4.53 and 6.42 GiB max | M | Same |
| Machine-wide memory | 5.76% at 19:32:07, 21.11% at 19:33:07, 24.2% at 20:02, 25.84% at 10:30:22, of 2,263 GiB | M | W&B `memory_percent`, `wandb-metadata.json` total |
| Machine-wide rise | 455 GiB across the run, 347 GiB in the first minute of records | D, timing only | The row above. If each of 64 actors holds what an evaluator holds, 288-410 GiB, which fits. That is an assumption. The first record is 81 s after launch and there is no pre-launch reading |
| Whole-system CPU use | **Not logged** | | No `ps` or sampler, and no `cpu.N.cpu_percent` keys in W&B. 64 is an allocation, not a measurement |
| GPU utilisation and memory | **Not logged** | | No `gpu.*` keys in any of the 3 W&B runs |
| What else ran | **No sign of disturbance, not proof of a free machine** | M, D | No file under `multirobot-project` outside this run was written from 19:35 to 10:30 (`find`, depth 4, venvs excluded). Grad rate in 10-minute bins falls smoothly from 64.1 to 38.1 over the first 4 h, no dips. Other users not checked. Port runs have dips, this one does not |

By stage. Grad steps from TB wall_time. Env and inserts from `actor.npz` episodes ending in that wall window. All [D], all re-computed and matching.

| Grad steps | Wall s | End, h from launch | Grad/s | Env/s | Inserts/s | Per actor env/s | Mean ep. len | `sampling_wall_time_ms` / `total_step_time_ms` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2k-100k | 1,532 | 0.500 | **63.98** | 822.7 | 1,638 | 12.9 | 121.0 | 14.1 / 15.4 |
| 100-200k | 1,913 | 1.031 | **52.26** | 1,029 | 1,339 | 16.1 | 121.0 | 17.1 / 18.8 |
| 200-300k | 2,212 | 1.646 | **45.20** | 1,138 | 1,157 | 17.8 | 120.7 | 20.1 / 21.8 |
| 300-400k | 2,277 | 2.279 | **43.91** | 1,123 | 1,124 | 17.6 | 116.3 | 20.5 / 22.4 |
| 400-500k | 2,362 | 2.935 | **42.34** | 1,084 | 1,084 | 16.9 | 107.2 | 21.4 / 23.3 |
| 500-600k | | | **40.43** | | | | | |
| 400k-1.878M | 45,755 | 14.988 | **32.30** | 827 | 827 | 12.9 | 71.7 | 28.3 / 30.5 |
| 1.86-1.878M | | | **27.7** | | | | | |

About 92% of each step's wall clock is inside the sampling call. **That is not 92% idle.** JAX dispatch is asynchronous (PB:316), so GPU compute overlaps it. Against a learner capacity of 90 steps/s, 11 ms [M, PB:405, on a shared GPU, PB:414 re-timed], the reference used 71% at 2-100k and 47% at 400-500k [D], 39% over the run (PB:411).

Where an actor's time goes. [D, `actor.npz`: `episode_wall_time_s`, `env_sampling_time_s`, `episode_write_wall_time_s`, and the gap between an actor's successive episodes]. New in this file.

| h from launch | Cycle s | Stepping s | Scene sampling s | Write s | Write share | Stepping ms per env step |
|---|---:|---:|---:|---:|---:|---:|
| 0.05-0.50 | 9.54 | 1.65 | 0.85 | 6.89 | 72% | 13.7 |
| 0.50-1.03 | 7.47 | 2.20 | 1.01 | 4.11 | 55% | 18.3 |
| 1.03-1.65 | 6.73 | 2.18 | 1.03 | 3.33 | 50% | 18.2 |
| 1.65-2.28 | 6.57 | 2.39 | 1.12 | 2.84 | 43% | 20.7 |
| 2.28-2.94 | 6.27 | 2.25 | 1.16 | 2.62 | 42% | 21.2 |
| 2.94-6 | 5.66 | 1.95 | 1.34 | 2.06 | 36% | 23.2 |
| 6-10 | 5.31 | 1.77 | 1.59 | 1.56 | 29% | 26.4 |
| 10-15 | 5.40 | 1.76 | 1.81 | 1.37 | 25% | 28.4 |

The write span (`agents/actor.py:618-651`) covers the real episode, hindsight generation and the hindsight episode, plus any block by the rate limiter (`launch/launch.py:145-148`, error buffer 10,240 samples). I cannot split blocked time from CPU time. So actor CPU is somewhere below 64 cores in the early stages, by an amount the logs cannot give. Unloaded, one actor step is 8.43 ms (PB:405, 424-425), so loaded stepping is 1.6 to 3.4x slower, cause unverified (PB:434).

**The collector's inference "a reference with more actors could run faster per step than the port" is untested.** The write share above argues against it early. **The original's learner is faster than the port's**: 10.3-11 ms against 14.6-15.9 ms [D, 2.4]. That part stands.

Paper, for context: 64 actors each on one Ice Lake core, learner on one A100, "6 to 24 hours" to 95% [P, `S/dl_arxiv/paper_pdf.txt:202`]. Training step 33 ms at 4r/10t [P, same file, line 420].

## 2. Port runs

All on git `2a32e5c` plus the uncommitted edit recorded in `runs/g8*/provenance/*/git_diff.patch` [M]. W=64, B=128, SPI=5, fused 20, bf16, compiled, `profile: true`, evaluation every 20,000 steps, checkpoint every 1,800 s (`config.json`) [M].

| Run | GPU (pid) | Arm | Start states | Note |
|---|---|---|---|---|
| g8a_s1 | 0 (3005319) | mesh_tool | Original `.pb` | Seed 1 |
| g8a_s2 | 1 (3005322) | mesh_tool | Original `.pb` | Seed 2. **Same card as the reference** |
| g8a_s3 | 5 (3005325) | mesh_tool | Original `.pb` | Seed 3 |
| g8a_ref0 | 2 (3005329) | mesh_tool | Original `.pb` | Seed 1, reference's initial weights (`--init-weights`) |
| g8b_s1 | 3 (3005331) | spheres_stick | Regenerated pool | **Different task**, so its rates compare and its milestones do not |

### 2.1 Stage rates

From `counters/grad_steps`, `counters/env_steps`, `counters/inserted` against wall_time, all [D]. Each cell is grad/s, env/s, inserts/s. Same step range on both sides.

| Stage | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 |
|---|---|---|---|---|---|---|
| 0-100k | 64.0, 823, 1638 | 56.9, 729, 1457 | 56.2, 720, 1439 | 57.5, 737, 1471 | 57.1, 733, 1461 | 56.3, 723, 1440 |
| 100-200k | 52.3, 1029, 1339 | **47.8**, 935, 1225 | 55.3, 1039, 1416 | 55.5, 1100, 1421 | 55.4, 1109, 1417 | 55.0, 1111, 1407 |
| 200-300k | 45.2, 1138, 1157 | **51.3**, 1296, 1314 | 54.9, 1370, 1405 | 55.4, 1398, 1418 | 55.1, 1392, 1411 | 55.3, 1397, 1415 |
| 300-400k | 43.9, 1123, 1124 | 56.7, 1451, 1453 | 55.0, 1407, 1409 | 55.5, 1420, 1421 | 54.5, 1394, 1395 | 55.5, 1421, 1421 |
| 400-500k | 42.3, 1084, 1084 | 56.1, 1435, 1435 | 55.1, 1412, 1412 | 55.6, 1423, 1423 | 55.2, 1412, 1412 | 55.6, 1424, 1424 |
| 0-500k | 48.4, 1055, 1238 | 53.5, 1165, 1370 | 55.3, 1193, 1416 | 55.9, 1220, 1430 | 55.4, 1213, 1419 | 55.5, 1217, 1421 |
| All, to 13:12:49 | 34.9, 859, 895 (to 1.878M) | 53.4, 1163, 1366 | 54.9, 1194, 1406 | 55.4, 1219, 1419 | 55.0, 1211, 1407 | 55.1, 1218, 1411 |
| All, to 13:41:16 | | 53.1, 1188, 1359 | 54.7, 1221, 1401 | 55.1, 1243, 1412 | 54.3, 1223, 1390 | 54.8, 1240, 1403 |
| Latest step, 13:12:49 | | 504,000 | 523,000 | 524,000 | 520,000 | 525,000 |
| Latest step, 13:41:16 | | 593,000 | 614,000 | 616,000 | 606,000 | 616,000 |

Same step range, 0-500k, g8a against the reference: grad/s **1.11-1.16x**, env/s **1.10-1.16x**, inserts/s **1.11-1.16x** [D]. At 400-500k: grad/s **1.30-1.33x**, env/s **1.30-1.32x** [D]. Bold marks g8a_s1's unexplained slow period, 2.6. `rate/samples_per_insert` median 5.000 in all runs, residual collisions 0 [M].

### 2.2 Wall clock at equal grad step

Hours from launch [D]. This has no learning noise, so it is the cleanest comparison available.

| Grad step | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 | Reference / g8a |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 0.500 | 0.560 | 0.551 | 0.555 | 0.559 | 0.547 | 0.89-0.91x |
| 200k | 1.031 | 1.140 | 1.053 | 1.056 | 1.060 | 1.053 | 0.90-0.98x |
| 300k | 1.646 | 1.681 | 1.559 | 1.557 | 1.564 | 1.555 | 0.98-1.06x |
| 400k | 2.279 | 2.171 | 2.064 | 2.058 | 2.074 | 2.055 | 1.05-1.11x |
| 500k | 2.935 | 2.666 | 2.568 | 2.557 | 2.577 | 2.555 | 1.10-1.15x |

Cumulative crossover: 222k on g8a_s2, 224k on g8a_s3, 228k on g8a_ref0, 328k on g8a_s1 [D]. The port stays ahead after that, by 965-1,359 s at 500k on g8a.

The gap keeps widening only if the rates in 1 and 2.1 persist. The reference falls to 27.7 grad/s late [D, 1.86-1.878M, TB]. **The port's rate at 1.87M on this code has not been measured**, so any total-run ratio is an extrapolation. At 54 grad/s it would be 9.6 h vs 15.0 h, 1.56x, which sits inside DESIGN:143's range. Keep that out of a measured table.

### 2.3 Wall clock to a task rate, `training_dist`

Milestone: first evaluation whose mean over the trailing 50,000 grad steps reaches the rate. Same rule as `mjlab_ballet/scripts/compare_reference.py:78`. Port resolution is one evaluation per 20,000 steps, about 6 minutes, so 0 to +0.1 h, one-sided: a crossing can only be detected late. Reference: 50 episodes per evaluation, 6-8 evaluations in the window. Port: 1,024 per evaluation, 3 in the window, which reach back 40,000 steps rather than 50,000. Tag `eval/training_dist/num_targets_done`. All [D].

| Task rate | Reference step | Reference h, eval row | Reference h, **learner clock** | Sampling noise, +-h | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1, own pool |
|---|---:|---:|---:|---:|---|---|---|---|---|
| 50% | 196,060 | 1.054 | **1.008** | 0.026 | 1.140, 200k | 1.053, 200k | 0.955, 180k | 0.959, 180k | 0.952, 180k |
| 75% | 291,320 | 1.634 | **1.592** | 0.029 | 1.574, 280k | 1.559, 300k | 1.256, 240k | 1.363, 260k | 1.355, 260k |
| 90% | 435,520 | 2.556 | **2.508** | 0.057 | 2.171, 400k | 2.264, 440k | 2.158, 420k | 2.174, 420k | 2.055, 400k |
| 95% | 577,680 | 3.518 | **3.464** | 0.148 | pending | pending | pending | **3.248, 620k** | pending |

Noise is one standard error of the window mean over the local slope [D: residuals of a linear fit over +-60k]. It is one seed and sampling only, so it says nothing about seed spread. On the port the seed spread at 90% is 2.158-2.264 h.

Original time / port time, g8a runs, under each definition [D]:

| Task rate | Eval row (collector's) | **Learner clock** | Median of 4, learner clock | Both interpolated between evaluations | Reference on a 40k window, learner clock | g8a_s2 alone, same card |
|---|---:|---:|---:|---:|---:|---:|
| 50% | 0.92-1.10x | **0.88-1.06x** | 1.00x | 0.94-1.07x | 0.84-1.01x | 0.96x |
| 75% | 1.04-1.30x | **1.01-1.27x** | 1.09x | 1.05-1.28x | 1.01-1.27x | 1.02x |
| 90% | 1.13-1.18x | **1.11-1.16x** | 1.15x | 1.12-1.16x | 1.09-1.14x | 1.11x |

Status at 13:48:08, 3.29 h. Latest 50k window: g8a_s1 0.9459 at 600k, g8a_s2 0.9478, g8a_s3 0.9499, g8a_ref0 0.9547, g8b 0.9496, those four at 620k. **One run has reached 95%: g8a_ref0, step 620,000, 3.248 h, 13:45:33**, which is 1.07x against the reference's learner clock and 1.08x against its eval row, on one seed of four. That run shares the reference's initial weights and is the quickest of the four at this rate. Latest single evaluations at 620k: 0.9487, 0.9528, 0.9592 and g8b 0.9554, with g8a_s1 0.9514 at 600k. 50k window at 500k: g8a 0.919-0.936, reference 0.941. At 580k: g8a_s2, s3, ref0 0.9435-0.9458, reference 0.9547. So the port is **not ahead on steps** and is behind at 580k: 620k or later against the reference's 578k. For the rest to beat the reference they must cross by 3.46 h on the learner clock, 13:58:29, which at 20,000-step cadence leaves the 640k and 660k evaluations. There is only one reference seed, so its noise is unknown (DESIGN:338). The 95% clock will also carry the 112-170 s lost at 13:06-13:18, section 2.6.

Same milestones on the env-step axis, millions [D]. The reference's eval-row figure includes the env steps taken during the evaluator's lag, so use the learner clock:

| Task rate | Reference, eval row | Reference, learner clock | g8a range |
|---|---:|---:|---:|
| 50% | 3.45 | **3.26** | 2.89-3.34 |
| 75% | 5.82 | **5.65** | 4.37-5.76 |
| 90% | 9.52 | **9.34** | 8.43-9.34 |

So no sample-efficiency gain: the reference sits inside the port's range at 50% and 75% and at its top at 90%.

The same reference run under each definition, all reproduced here:

| Definition | 50% | 75% | 90% | 95% | Where |
|---|---:|---:|---:|---:|---|
| Rolling 5 evaluations, from first TB event | 0.953 | 1.536 | 2.402 | 3.409 | PB:347-350, DESIGN:32 |
| 50k window, from first TB event | 0.999 | 1.579 | 2.500 | 3.463 | DESIGN:412 |
| 50k window, from launch, eval row | 1.054 | 1.634 | 2.556 | 3.518 | Collector's |
| **50k window, from launch, learner clock, used here** | **1.008** | **1.592** | **2.508** | **3.464** | This file |
| Interpolated crossing, learner clock | 0.969 | 1.587 | 2.473 | 3.416 | This file |

Pick one for the README. First-event zero hides 200 s on the reference and 153-243 s on the port, so launch is the honest zero.

`static_paper`, hours from launch, 50k window [D]. Reference 10 episodes per evaluation, 30-35 in the window, port 80 worlds, 3 in the window. Tag `eval/static_paper/num_targets_done` (reference `eval/static_dist/*`). The sets are comparable: same tasksets, boxes the same to 0.44 mm between processes (PB:259 [CORRECTED]), and correcting the rotation barely moves the metrics (`differences.md` row 9). **Quote 90% only beside `training_dist`.** The reference is flat around here, 0.874-0.878 at 480-520k [D], so a first crossing is unstable: +-0.117 h on its 3.376 h from sampling alone [D, same method as above], and one seed. At those steps the port's 50k window is 0.878-0.920, ahead of the reference on all four runs at 500k and 520k, by an amount `training_dist` does not show. Why is not established. PB:355 [CORRECTED] warns about short static windows; the windows here are 240-350 episodes.

| Task rate | Reference, eval row | Reference, learner clock | g8a range | Reference / g8a, learner clock |
|---|---:|---:|---:|---:|
| 50% | 1.172 | 1.163 | 1.156-1.255 | 0.93-1.01x |
| 75% | 1.811 | 1.801 | 1.657-1.782 | 1.01-1.09x |
| 90% | 3.387 | 3.376 | 2.456-2.699 | 1.25-1.38x |
| 95% | 6.214 | 6.204 | pending | blank |

### 2.4 Where the port's time goes

Profile windows, 120 rounds with a synchronise at each edge, `profile/*_ms` [M per window, mean over windows]. 13-15 windows at step >= 300k, 3 windows at 20-100k. All re-computed and matching.

| ms per round, step >= 300k | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 |
|---|---:|---:|---:|---:|---:|
| `window_round_ms` | 43.33 | 46.58 | 46.94 | 46.51 | 44.01 |
| `td3_step_ms` | 36.47 | 39.54 | 39.75 | 39.49 | 38.13 |
| `policy_forward_ms` + `feature_build_ms` | 2.07 | 2.09 | 2.11 | 2.11 | 2.09 |
| `world_forward_ms` | 1.59 | 1.63 | 1.70 | 1.60 | 0.43 |
| `rollback_ms` + `env_step_ms` + `reset_ms` | 2.16 | 2.26 | 2.31 | 2.25 | 2.29 |
| TD3 share late / early (20-100k) | 84% / 88% | 85% / 89% | 85% / 89% | 85% / 88% | 87% / 90% |

- One TD3 step is 14.6-15.9 ms late and 15.1-15.6 ms early [D: `td3_step_ms` over grad steps per round, 2.50 late and 4.98-5.00 early], close to the 15.4-16.2 ms Torch bench (`S/reports/algo_spec.md:616-617`, PB:414). JAX at the same point is 10.3-11 ms (`S/reports/algo_spec.md:614`, PB:405), so the port's learner is about 1.5x slower than the original's (`algo_spec.md:632`).
- `train/total_step_time_ms` is 17.4-19.2 ms by stage, and 19.5-20.9 on g8a_s1 at 100-300k [M]. `train/sampling_wall_time_ms` is 1.8-3.0. It is synchronous and absorbs queued acting work, so it is **not comparable** with the reference's tag of the same name.
- `mesh_tool` vs spheres costs about 1.2 ms per round in `world_forward_ms`, 2-3% of the wall clock. It explains why g8a and g8b rates are nearly equal.
- Evaluations cost 80.4-88.8 s on g8a and 146.5 s on g8b at 13:12:49, 0.83-0.91% and 1.51% of elapsed. Per evaluation 3.08-3.29 s and 5.42 s. Checkpoints 2.1-2.2 s in all (`profile/evaluation_s`, `profile/checkpoint_s`, both cumulative) [M]. Net of those, rates rise by 0.5-0.9 grad/s [D].
- The reference's checkpoints every 10,000 steps cost it nothing visible: 21.34 s per 1,000 steps at a 10k boundary against 21.31 s elsewhere [D, TB], since they are async. So that difference between the two runs does not bias the clock.

### 2.5 Resources, re-checked

`ps` at 13:05:30 and 13:09:05 by the collector, by me at 13:29:08 and 13:41:16. `nvidia-smi` by the collector at 13:05:30, by me at 13:29:09.

| Quantity | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 | Tag |
|---|---:|---:|---:|---:|---:|---|
| CPU s at 9,290 s elapsed, 13:05:30 | 11,134 | 11,078 | 11,141 | 11,140 | 11,037 | Collector's read, consistent with mine |
| CPU s at 11,436 s elapsed, 13:41:16 | 13,255 | 13,200 | 13,263 | 13,263 | 13,158 | M |
| Lifetime cores, 13:41:16 | 1.159 | 1.154 | 1.160 | 1.160 | 1.151 | D |
| CPU s in those 2,146 s | 2,121 | 2,122 | 2,122 | 2,123 | 2,121 | D |
| **Sustained cores**, includes the 13:06-13:18 dip | **0.99** | **0.99** | **0.99** | **0.99** | **0.99** | D |
| `profile/process_cpu_fraction`, median | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | M |
| `profile/process_cpu_fraction`, mean since 13:05:30 | 0.985 | 0.985 | 0.986 | 0.985 | 0.984 | M, agrees with `ps` |
| `profile/main_thread_cpu_fraction`, median | 0.599 | 0.601 | 0.602 | 0.602 | 0.590 | M |
| `nvidia-smi` process MiB, 13:29 | 11,498 | 11,498 | 11,496 | 11,498 | **6,664** | M |
| **Process GiB** | **11.23** | **11.23** | **11.23** | **11.23** | **6.51** | D |
| Card used, MiB, 13:29 | 11,512 | 11,512 | 11,510 | 11,512 | 6,678 | M |
| GPU util, single sample, 13:29, % | 72 | 73 | 78 | 73 | 76 | M. Collector's 13:05 sample, 81, 65, 78, 73, 77: unverified |
| `profile/gpu_util_percent`, mean after 15 min, read 13:26, % | 73.4 | 75.2 | 75.6 | 75.1 | 73.9 | M |
| Power, single sample, 13:29, W | 391 | 422 | 420 | 411 | 411 | M. Collector's 13:05 sample, 419, 410, 412, 415, 417: unverified |
| `profile/mem_allocated_gib` / `mem_peak_gib` / `mem_reserved_gib` | 2.55 / 4.01 / 5.36 | same | same | same | 2.57 / 4.03 / 5.36 | M |
| `profile/gpu_mem_used_gib` (NVML, flat) | 11.95 | 11.95 | 11.95 | 11.95 | 7.23 | M |
| `profile/rss_gib`, host, **peak** (`ru_maxrss`, `train/profiler.py:185`) | 2.98 | 2.99 | 2.98 | 2.99 | 2.77 | M |
| `ps` RSS, host, current, 13:41, GiB | 2.74 | 2.77 | 2.76 | 2.78 | 2.55 | M |
| Threads | 206 | 206 | 206 | 206 | 206 | M |
| Affinity, nice | 0-71, 0 | 0-71, 0 | 0-71, 0 | 0-71, 0 | 0-71, 0 | M |

GPU memory will not grow from here: `counters/replay_size` is at its cap of 2,000,000 and NVML is flat [M].

Why 1.2 is not the rate. On g8a_s1 at 13:36:08: main thread 6,558 s, `pt_autograd_0` 4,324 s, and 71 further threads at about 30 s each, 2,064 s in all (`/proc/PID/task/*/stat`) [M]. The collector's read at 13:09 gave 2,065 s for those threads, so they did not move: they are startup only. CPU-s before the first report: 2,006-2,165 in the first 211-277 s, 7.8-9.5 cores on average [D: `ps` minus the integral of `process_cpu_fraction`]. `OMP_NUM_THREADS` is unset in the process environment [M, `/proc/PID/environ`], so the pool presumably defaults to all 72 cores. A README should say "1 core sustained, about 2,100 CPU-s at start".

What 1.0 core means. The process has a full core busy, split about 0.60 main thread and 0.39 autograd thread, while the GPU sits at 73-76%. That is saturation of one core and not a light load. Whether part of it is busy-wait on the GPU is not separated. **It has never been run with `taskset` to one core, so "needs only 1 core" is not measured, and "uses 1.0 core on average" is.**

CPU core-hours to 90%: port 2.7-2.8, used [D: 1.0 core x (2.16-2.26 h less startup) + 0.60 for startup]. Reference 64 x 2.508 = 160, **allocated** [D]. Comparing them as "60x" mixes used with allocated. It needs a measured reference CPU figure, section 5.

GPU sharing: at 13:29 each G8 run was alone on its card, with the other jobs all on GPU 4 (`nvidia-smi --query-compute-apps`) [M]. After the first 15 minutes `profile/gpu_mem_used_gib` min and max differ by at most 0.002 on all 5 [M], so nothing else held memory on those cards at any report.

### 2.6 Disturbances

| When (KST) | What | Evidence | Inside the milestones? |
|---|---|---|---:|
| 09:56-10:06 | **The 221 load spike.** 5 compile-heavy FastTD3 runs, 2 test processes | FS:294 [DOC]. `v2_g8a_s3` and `v2_g8b_s1` show `process_cpu_fraction` 0.53 and 0.35 in 10:00-10:05, and first report at 10:01:40-10:05:23 for a 09:56 start [M]. Today's runs took 3.5-4.6 min [M] | **No.** Hit `v2_g8*` only |
| 10:30-10:33 | FastTD3 variants on the small task finishing, pinned to 56-71 per FS:294 | `fasttd3_dev/variants/*` end 10:32:48-10:33:10 [M]. Pinning [DOC] | Yes, startup only |
| 10:54:22-10:55:09 | All 5 runs dip together: `process_cpu_fraction` 0.65-0.82, 34-46 grad/s | `rate/grad_steps_per_s`, steps 67-71k [M]. `fasttd3_dev/mutation.log` last written 10:53 [M]. Cause not established | Yes. About 15-20 s per run [D] |
| 10:55-12:14 | **g8a_s1 only:** 50.0 grad/s over 70-310k vs 55.3-55.7 on the others. At 200-300k, `td3_step_ms` 40.7-44.0 vs 37.5-42.4 on g8a_s2. `train/total_step_time_ms` 19.2-21.3 vs 17.6-18.3 in 40k bins; `profile/gpu_util_percent` 65-73 vs 75-77 | Those tags, GPU 0 [M]. Memory flat at 11.95, so no second process with memory. CPU fraction stays 0.999, so not preemption of the process. **Cause not established** | **Yes, about 460 s** [D: 4,797 s vs 4,307-4,341 s]. It is why g8a_s1 is the slowest at 50% and 75% |
| 12:15-13:00 | g8a_s1 runs faster than the rest, 56.5-57.5 vs 55 grad/s, as it did at 10:40 (61 vs 58-59) | Same tags [M] | Not explained, and not stable: see the last row |
| 13:06-13:18 | **All 5, longer than the collector saw:** 35-50 grad/s in 2-minute bins, worst 35-40 at 13:12. `process_cpu_fraction` as low as 0.84-0.91 | `counters/grad_steps` [D, read 13:36]. SAC probes written 13:08, `sac_dev/mutation.log` 13:14, `test_full_1.log` 13:19, `ppo_dev/mutation_b.log` 13:26 [M, mtimes]. Parallel evidence workers, this step's nice-19 reads included, not separable. Load average: collector's 12.0 at 13:05, 41.6 "(5 min)" at 13:15 and 11.7 at 13:17 are unverified, and a 5-minute average cannot fall from 41.6 to 11.7 in 2 minutes, so they are not one column. My reads, 1/5/15 min: 10.55, 11.98, 16.90 at 13:29, 8.91, 9.36, 13.20 at 13:38 [M] | After every milestone to 90%. **Before 95%. About 112-170 s per run** [D: against each run's 12:30-13:00 rate, 13:00-13:22], not 60 s |
| 13:18 and 13:26 onward | g8a_ref0 from 13:18, 50.5-52.8 grad/s, and g8a_s1 from 13:26, 50.4-51.8, vs 54-56 on the other three. `td3_step_ms` 40.9-43.2 in most of those windows | `counters/grad_steps`, `profile/td3_step_ms` [M]. Memory flat. Cause not established. Same kind of slowdown as g8a_s1's earlier one, now on a second card | **Yes, for 95%.** Still going at 13:36 |

Other dev work sharing the CPU during the G8 runs, all on GPU 4 (start from `provenance/`, end from `metrics.csv` or logs) [M]. Only FastTD3 reports pinning, to cores 56-71, from 10:06 (FS:294 [DOC]). The collector saw PPO and SAC mains on cores 35, 0 and 42 at its snapshot: unverified.

| Span | Job |
|---|---|
| 10:37-10:47 | `fasttd3_dev/variants/default_s2` |
| 10:57-11:27 | `fasttd3_dev/g11_smoke_default` |
| 11:28-11:58 | `fasttd3_dev/g11_smoke_lr1e4`, `g11_smoke_shallow` |
| until 12:24 | `fasttd3_dev/batch_limit` (directory mtime only) |
| 11:33-11:35, 11:51-11:52 | `ppo_dev/scratch/s1`, `prof2` |
| 11:47-12:27 | `ppo_dev/g14_small_s1` |
| 11:48-12:13, 12:15-12:35 | `sac_dev/g13_small_h1_s1`, `variants/small_h1_lr1e4_s1` |
| 11:54-12:22 | `ppo_dev/variant_near015_s1`, `variant_near015_ent0_s1` |
| 12:28-13:00 | `ppo_dev/g14_smoke_s1`, `g14_smoke_resume` |
| 12:36-13:06 | `sac_dev/g13_smoke_default` |
| 13:01-13:16 | `ppo_dev/variant_near005_ent0_s1` to 13:11, `variant_4r_ent0_s1` to 13:16 |
| 13:19-13:33 | `sac_dev/variants/smoke_lr1e4` |
| Scattered | pytest and mutation suites. Mtimes above confirm mutation and test logs. The collector's "3 pytest processes at 13:05": unverified |

Net: `process_cpu_fraction` is below 0.9 in 4-7 of 504-525 report intervals per run at 13:12:49, 6-10 of 593-616 at 13:41 [M]. CPU contention cost little. These runs are honest but not clean. Load average was never logged, so a reader cannot be shown that it stayed low.

## 3. Raw simulator throughput, separate from training

None of the rows below is a training rate. In training, env plus acting is 5-13% of a round and the rate is set by SPI=5 and the learner.

**Code path.** Final = `mjlab_ballet/sim`, `env/ballet_env.py`, used by the runs. It uses `sap_tile` broadphase, filter `plane`, `sphere` (`mjlab_ballet/sim/world.py:137-138`), stock collision, one-at-a-time random rollback (DESIGN D9). Everything under `S/bench/` is a **prototype**: rough sphere model, its own env, 240 steps or so, not sustained.

**Geometry.** The original collides convex hulls of the arm meshes, 15 collision geoms per robot (`S/reports/env_spec.md:123-138`). Its baseline model has `ngeom` 303 (`B/baseline_cpu.jsonl:1`). **Every prototype below has the sphere arm**, `ngeom` 197, including the column the collector called "meshes": there the table and obstacles are meshes and the arm is still spheres (`S/bench/run_d.sh:52`, `B/warp_mesh.jsonl`). No GPU kinematics-and-collision benchmark of the hull arm exists.

### 3.1 Env steps/s, 4r/10t/30o, one B200

| Worlds | Kin+col, sphere arm, nxn, prototype, quiet | Kin+col, sphere arm, `sap_tile`, plane+sphere, prototype | Kin+col, sphere arm, mesh table and boxes, prototype | Env step, stock + one-at-a-time, sphere arm, prototype, **shared** | Env step, final, spheres | Env step, final, mesh_tool | Env + features + policy, final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | | | | | **23.5k** [D] | **16.0-17.0k** [D] | **13.3k** spheres, **10.5-11.0k** mesh [D] |
| 1,024 | 4.05M | | | | **240k** [D] | | **60.0k** spheres [D] |
| 1,024 + 80 | | | | | | | **39-57k** mesh [D, eval] |
| 2 x 1,024 + 2 x 80 | | | | | | | **53-61k** spheres [D, eval] |
| 4,096 | 4.48M | 6.96M | 2.47M | 619k | **blank** | **blank** | **blank** |
| 16,384 | 4.41M | | | 498k | **blank** | **blank** | **blank** |
| 65,536 | 4.49M | 7.28M | 2.39M | 661k | **blank** | **blank** | **blank** |

All M columns use the median, `env_steps_per_s` in the JSONL. GT's sweep and mesh tables print a from-minimum figure, 7.04M, 7.29M, 2.64M, 2.45M, which the collector had mixed with median figures.

Sources:

| Column | Source | Tag |
|---|---|---|
| nxn | GT:10, 18, 26, 34, `B/warp_final.jsonl` `kincol`, `gpu_before` 5 MiB, 0%, `others` empty | M |
| `sap_tile` | GT:107, 125, `B/warp_sweep.jsonl`, `gpu_before` 5 MiB, 0%. At 4,096 `others` shows a second process on the card during the run, so **not fully quiet**. At 65,536 empty | M |
| Mesh table and boxes | GT:183, 195, `B/warp_mesh.jsonl`, `mesh_both`. `others` shows 1-2 processes on the card, so **not quiet** | M |
| Stock prototype | GT:267, 271, 275, `B/env_proto.jsonl`. `gpu_before` 9,725 MiB, 2,433 MiB, 4,583 MiB at 94% | M, not clean. No sustained run. Custom kernels (`B/env_active.jsonl`: 1.83M, 3.23M, 4.10M) are **not** on the final path, DESIGN D9 |
| W=64 final | `profile/{world_forward,rollback,env_step,reset}_ms`, step >= 300k: 2.72 ms spheres, 3.76-4.00 ms mesh. Adding `policy_forward` and `feature_build` gives 4.81 and 5.83-6.11 ms. Runs in 2.5 | D. Sync at each edge adds overhead |
| W=1024 final | `fasttd3_dev/g11_smoke_default` `profile/*` (FS:158-171 agrees), **spheres only**: 1.19+1.38+0.54+1.15 = 4.26 ms; adding 0.49 and 12.32 gives 17.07 ms | D. GPU alone, f32 at `highest` (`fasttd3/trainer.py:138`) |
| Evaluator rows, final | `profile/evaluation_s` increments for the first 8 evaluations, length 121: 2.31-3.37 s for 1,104 x 120 steps, 4.34-4.99 s for 2,208 x 120 | D. Policy on, card shared only with the same process. Sets run in turn, so the 80-world sets pull this down |

Also on the final path: random actions at W=64, 15,650 spheres, 10,900 mesh_tool env/s (TS:145, [DOC], **pre-fix code**, TS:277). Today's first window, step 1,000: 7.37-11.29 ms per round on g8a, 5.7-8.7k env/s [D], during others' startup. 8 robots, prototype quiet: 1.45M, 1.42M, 1.41M, 1.41M at 1,024-65,536 (GT:42-66). 8 robots on the final path: **blank**.

Policy forward alone caps acting: 19.6 ms at B=1,024 and 75 ms at B=4,096, compiled, default precision, 52-55k graphs/s, some under load (`S/reports/algo_spec.md:638`, prototype). JAX original 23.6 ms at B=4,096, 174k graphs/s (PB:405, GPU shared). Final at W=1,024: 12.32 ms, 83k graphs/s [D]. Inference: adding worlds buys little past about 60k env/s with this GNN until the policy gets cheaper. It has not been run above 1,024 on the final path. Physics in the millions does not carry over.

### 3.2 Against the original, closest available

Only the rows marked mesh arm have the same geometry on both sides. The others compare the port's sphere arm with the original's hull arm, so none of them is like for like.

| Scope | Original, one core | x 64, never run | Port, one GPU | Ratio | Path |
|---|---:|---:|---:|---:|---|
| Kinematics + collision only | 0.080 ms, 12.5k/s (PB:426-427). 0.047 ms, 21.4k/s (`B/baseline_cpu.jsonl:1`) | 800k-1.37M [D] | 4.41-7.28M, sphere arm | **3.2-9.1x** [D] | Prototype. Different geometry |
| Same, mesh table and boxes | | | 2.39-2.47M, sphere arm | **1.7-3.1x** [D] | Prototype. Arm still spheres |
| Env step with rollback and Python | 1.55 ms, 645/s, n=40, unloaded (PB:424, 433). 2.09 ms, 479/s, n=3 (`B/baseline_cpu.jsonl:1`) | 30.6-41.3k [D] | 498-661k, sphere arm | **12-22x** [D] | Prototype, shared |
| Same | | | 240k at W=1,024, spheres | **5.8-7.8x** [D] | Final, largest W measured. Different geometry |
| **Same, mesh arm** | | | **16.0-17.0k at W=64** | **0.39-0.56x** [D] | Final, the only mesh figure. W=64 is the training setting |
| Env + features + policy | 1.55 + 2.04 + 4.84 = 8.43 ms, 119/s (PB:405, 424-425) | 7.6k [D] | 60k at W=1,024, spheres | **7.9x** [D] | Final. Different geometry |
| **Same, mesh arm** | | | **10.5-11.0k at W=64** | **1.4x** [D] | Final |
| **Same, mesh arm** | | | **39-57k at 1,024 + 80 worlds, evaluator** | **5.2-7.5x** [D] | Final. Evaluation, not a training loop |
| Whole actor, real run | 13.4/s mean, 17.8/s best stage [D, section 1] | 859 mean, 1,138 best [D] | 60k, acting only, spheres | **53-70x** [D] | Different scopes: the left side includes resets, hindsight, writes |
| Training at parity, to 500k | | 1,055 [D] | 1,165-1,220 [D] | **1.10-1.16x** [D] | Final, mesh arm, the only row measured end to end |

The "x 64" column overstates the original. Its derived 7.6k is 6.7x above the best it did in 64-actor reality, 1,138 env/s. Do not put 4.48M beside 859. That compares bare collision on a GPU with the full original system.

T1 and T2 of PB:711-712 need >= 5 min sustained on the final path. **Neither has been run.** Derived values on spheres: T1 240k against target 200k. T2 60k, past the floor of 20k and short of the target of 100k.

## 4. Other agents, today's state

All on GPU 4. All 4r/10t/30o, spheres, W=1024, seed 1, from `metrics.csv`, `summary.json`, first to last logged row. No SAC status document exists. `ppo_status.md` is a stub (`TBD_SUMMARY`, line 5). So those two rows rest on raw data alone. "Graphs/s" = grad/s x batch [D].

| Agent, run | Batch | Env/s | Grad/s | Graphs/s | Best task rate, real task | GPU during the measurement |
|---|---:|---:|---:|---:|---:|---|
| Original, g8a, for reference | 128 | 1,163-1,219 at 13:12 | 53.4-55.4 | 6.8-7.1k | 0.9423-0.9550 latest, 13:41 | Alone [M] |
| FastTD3 default, LR 3e-4, `g11_smoke_default`, 10:58-11:27 | 2,048 | **5,928** | **11.58** | 23.7k | **0.0000**, 21 evals, 4 sets | **Alone on GPU 4.** Card memory flat 18.10-18.11 GiB over 51 reports [M]. CPU shared with 5 G8 runs. **The only clean figure here** |
| FastTD3, LR 1e-4, `g11_smoke_lr1e4` | 2,048 | 2,344 | 4.58 | 9.4k | 0.0000, 9 evals | Shared from 11:47: card 29-110 GiB [M] |
| FastTD3, 3-layer, `g11_smoke_shallow` | 2,048 | 2,789 | 5.45 | 11.2k | **0.0001**, 10 evals: `training_dist` at 8,760, `heldout` at 7,760 | Same [M], card 29-105 GiB. FS:184 mentions only `training_dist` |
| SAC default, `g13_smoke_default`, 12:39-13:04 | 2,048 | 1,600 | 3.12 | 6.4k | **0.0000**, 2 evals | Shared: card 21,761-111,029 MiB, 5-11 processes, util 95.6% (`sac_dev/g13_smoke_default_gpu.log`, 61 samples) [M]. Q collapsed: mean 0.973, std 6e-8 at step 5,278 (`sac_dev/probe_collapse_smoke.txt`) |
| SAC, LR 1e-4, `variants/smoke_lr1e4`, 13:19-13:33 | 2,048 | | | | 0.0000 [M, 13:40] | Rates not checked. New since the collector's read |
| PPO, `g14_smoke_s1`, 12:28-12:57 | 6,144 mini | 3,010 | 2.45 | 15.1k (`rate/samples_per_s`) | **0.0001**, `training_dist` at 3,240. 0.0000 on all 4 sets at 4,360 (`ppo_dev/eval_final.json`) | Shared: card up to 102,509 MiB, **3-6 processes**, util 95.4% (`ppo_dev/g14_smoke_gpu.csv`, 61 samples) [M] |
| PPO, entropy 0, `variant_4r_ent0_s1`, finished 13:16 | 6,144 mini | 3,537 | 2.88 | 17.3k | 0.0000 at 1,580 | Shared [M] |

Memory: FastTD3 18.1 GiB card at B=2,048, peak 15.12. SAC peak 15.11. PPO peak 66.83, reserved 77.08, 80,236 MiB in `nvidia-smi` per `sac_dev/g13_smoke_default_gpu.log`, the heaviest by far [M].

Raw data agrees with FS: 10,483,712 env steps, 20,236 grad steps (`summary.json`), FS:129-130's 5,950 env/s and 11.6 grad/s against 5,928 and 11.58 on a slightly different span. Small-task runs are excluded. Multi-GPU: NCCL has never run (FS:278), so **no multi-GPU number exists**.

Env/s from FastTD3 and PPO must not be read as "faster than the original": they have not learned the task, and the grad-step axis is not comparable (DESIGN:253).

## 5. What a clean benchmark must still measure

Conditions, DESIGN 12.2 (lines 252-256) and 17 (lines 412-414):

| Condition | Source |
|---|---|
| Same scene pools, seeds, evaluation protocol, metrics, evaluation world count | DESIGN:252 |
| Report env transitions and wall clock. Grad-step axis not comparable. Record GPU count and model | DESIGN:253 |
| `original` at parity, untuned. Others on a declared sweep, with the run count reported. Paper tasksets never used for tuning | DESIGN:254 |
| 1 GPU vs 1 GPU first, then each agent at its best GPU count | DESIGN:255 |
| At least 3 seeds per configuration, every seed shown | DESIGN:256 |
| Env steps/s, samples/s, grad steps/s, phase wall clock, GPU count and model, at 1 GPU and at best count. Time to 50, 75, 90, 95% | DESIGN:412 |
| Check `nvidia-smi` first, no optimising on a shared GPU | DESIGN:160 |
| >= 5 min sustained, 4r/10t/30o | PB:705 |

Conditions these runs show are also needed, which no document lists yet:

| Condition | Why |
|---|---|
| One milestone definition, one time zero from launch, and the learner's clock rather than the evaluator's row | Five reference figures exist for the same run, 2.3. The evaluator's row lags by 153-195 s on the reference and about 3 s on the port |
| Same evaluation size and cadence on both sides, or interpolate | 50 episodes at about 6,400 steps against 1,024 at 20,000. A noisier series crosses a threshold earlier, so a first-crossing rule favours the reference, most at 95% where the slope is flat |
| Log load average and per-process CPU on a timer | Nothing records it. C1 could only be settled from a status note and process ages |
| Same CPU allowance and nice on both sides, stated | Reference: nice 10, 64 cores, by a script edited later. Port: nice 0, all 72 |
| Run the port pinned to one core, and to one socket | "1.0 core" is an average with 72 available. Thread placement on 2 sockets is uncontrolled. Whether that explains the per-card slow periods is untested |
| Profile off, or its overhead measured | Windows synchronise the GPU and the reference has no such cost |
| Evaluation cost treated the same | Reference evaluators run on separate processes. Port evaluates in the learner's process, 0.8-1.5% |
| Same GPU index, or show per-card spread | g8a_s1 on GPU 0 swung from 45 to 61 grad/s, g8a_ref0 on GPU 2 also slowed, causes unknown |
| Keep evidence work off the machine during timed runs | This evidence pass coincided with a 112-170 s loss per run at 13:06-13:18 |

To measure:

| # | Measurement | Now |
|---|---|---|
| 1 | Original, parity, 3 seeds, free machine: wall clock to 50/75/90/95% | Disturbed, 95% pending |
| 2 | Port rate at 1M to 1.87M on the current code | Not measured. TS:152-157 is pre-fix |
| 3 | Reference CPU core-seconds and GPU memory and utilisation | **Not logged for 65 of 68 processes, cannot be recovered.** Needs a sampled re-run. Stage-specific figures may need a full run, 64 cores for about 15 h per seed (DESIGN:338). More reference seeds are the user's open decision |
| 4 | T1, raw env, final path, W = 64 to 65,536, both arms, random and trained actions, >= 5 min | Blank at 4,096 and above. Mesh arm blank above W=64 outside evaluation |
| 5 | T2, acting, same W list, f32 and bf16 | Derived at 1,024 only, spheres |
| 6 | T3, learner alone on the final path | Inferred 62.6-68.3 train/s from windows, not run alone |
| 7 | T4 and T5, whole system on a free machine, profile off | T4 missed on run means, T5 floor missed, disturbed |
| 8 | T7 reset cost and T8 stability to 1e8 env steps | At 1.3e7 env steps, 0 residual collisions, no late-stage data |
| 9 | FastTD3, SAC, PPO on a free GPU, 3 seeds, wall clock to milestones | No milestone reached by any |
| 10 | FastTD3 above B=2,048 and at W=4,096, clean | FS:281 |
| 11 | Multi-GPU scaling, 1 to 6 B200 | NCCL never run |
| 12 | 8r/40t | No local reference, and no final-path number |
| 13 | Hull-arm kin+col on GPU at large W | No benchmark exists, section 3 |

### Candidate README table

Filled with what exists. **Blank means not measured.** The caption must say: port runs in progress on a shared machine, one reference seed, ratios to 90% only.

| | Original (JAX + CPU MuJoCo) | Port (torch + mjlab), parity |
|---|---:|---:|
| Hardware | 1 B200 + 64 EPYC cores allocated | 1 B200, 72 cores available, not pinned |
| Processes | 68 | 1 |
| CPU in use | not logged (64 allocated, 3 of 68 processes 3.3-4.1 cores) | 1.0 core, about 2,100 CPU-s at start |
| GPU memory | not logged | 11.2 GiB (mesh arm), 6.5 GiB (spheres) |
| Host memory, peak | 455 GiB machine-wide rise, not attributed per process. 27.6 GiB in the 3 logged | 3.0 GiB |
| Seeds | 1 | 4 (3 + 1 with reference init) |
| Grad steps/s, 0-100k | 64.0 | 56-57.5 |
| Grad steps/s, 400-500k | 42.3 | 55-56 |
| Grad steps/s, at 1.87M | 27.7 | |
| Env steps/s, to 500k | 1,055 | 1,165-1,220 |
| Hours to 50% of tasks | 1.01 | 0.96-1.14 |
| Hours to 75% of tasks | 1.59 | 1.26-1.57 |
| Hours to 90% of tasks | 2.51 | 2.16-2.26 |
| Hours to 95% of tasks | 3.46 | 3.25 on 1 of 4, others pending at 3.29 |
| Hours to 1.87M steps | 15.0 | |
| Clean machine | no record, no sign of load | no |

Second table, simulator only, one B200, 4r/10t/30o, with the scope in the row name. The arm column is what a reader would otherwise assume is the same:

| Scope | Original, 64 cores, computed from one core, hull arm | Port, 1 GPU | Port's arm | Status |
|---|---:|---:|---|---|
| Kinematics + collision | 800k-1.37M | 4.4-7.3M | Spheres | Prototype |
| Env step | 31-41k | 240k at 1,024 worlds | Spheres | Final |
| Env step | 31-41k | 16-17k at 64 worlds | Mesh | Final |
| Env step | | | Either | Final at 4,096 and above: not run |
| Env + features + policy | 7.6k | 60k at 1,024 worlds | Spheres | Final |
| Env + features + policy | 7.6k | 39-57k at about 1,100 worlds | Mesh | Final, evaluator |

Plots this data supports: grad steps against wall clock, reference and port (2.2), which shows the crossover at 222-328k. Task rate against wall clock with every seed (2.3). If the reference curve is plotted from TensorBoard wall_time, shift it to the learner clock, or the 153-195 s lag makes the port look better than it is.

A README sentence the numbers can carry today: *"At the paper's settings the port averages 1 CPU core and one GPU where the original was given 64 cores and one GPU, and reaches 90% of tasks about 1.1x faster so far, on one reference seed."* A sentence they cannot carry: any total-run speedup, a 95% ratio until all four runs have crossed, "same as" for speed, "needs only 1 core", any simulator ratio as like for like, or any claim from the other agents.

## Corrections

What I changed from `speed.md`. Numbers not listed here were re-computed from raw data and held, within rounding, or moved only because the runs moved.

Changes that affect a claim:

1. **Reference milestone clock.** The collector compared the reference's evaluator row with the port's. The reference row lags the learner by 153-195 s at the milestones, and the port's by about 3 s. Reference 1.054, 1.634, 2.556, 3.518 h becomes 1.008, 1.592, 2.508, 3.464 h on the learner clock. Ratios 0.92-1.10x, 1.04-1.30x, 1.13-1.18x become 0.88-1.06x, 1.01-1.27x, 1.11-1.16x. The verdict's "90%: 1.13-1.18x" and the candidate table changed. This moves against the port.
2. **Env-step milestones.** Reference 3.45, 5.82, 9.52M includes the lag. Learner clock 3.26, 5.65, 9.34M. So the port is not sample-cheaper.
3. **400-500k rate.** The verdict and candidate table quoted 53-55 grad/s as "400-500k". That was the "400k-latest" row, a different window containing the 13:08-13:12 dip. True 400-500k: 55.1-56.1. This moves in favour of the port.
4. **T3.** "Both floors missed" compared whole-system rates with a learner-alone floor. T3 was not run, and the inferred 62.6-68.3 train/s is above its floor. The summary's "misses the T3 and T4 floors" becomes T4 missed on run means, T3 not tested. Best 5-minute T4 windows do reach 1,497-1,573 inserts/s early.
5. **T5 added.** The 2.5 h floor to 95% is missed, measured. The collector's list left it out.
6. **Simulator "like for like".** Every prototype, and both final W=1,024 figures, is on the sphere arm, and the original is on hull meshes. The "meshes" column has a sphere arm with mesh table and boxes. Added mesh-arm rows: 0.39-0.56x at W=64 env step, 1.4x at W=64 with policy, 5.2-7.5x in the evaluator. The section title changed.
7. **Env step prototype range.** 619-661k and 15-22x left out the 498k in the same table. Now 498-661k, 12-22x.
8. **"Actor-limited".** The "92% waiting on data" reading and the inference about more actors are not supported. Added the actor time split, showing 42-72% of the cycle in writes for the first 2.9 h. Added capacity use, 71% and 47%.
9. **DESIGN:43 and 143.** "at most about 1.9x" is a ceiling and is not contradicted. "Far below the design's expectation" is limited to milestones up to 90%. The rate ratio at 400-500k is 1.30-1.33x.
10. **Candidate "1 B200 + 1 EPYC core".** The run has 72 cores available and was never pinned. Reworded, and 7.8-9.5 cores at startup noted.
11. **Host memory.** "Not attributable" and "rose smoothly from 24.0 to 25.8%" missed the jump from 5.76% to 21.11% in the first minute of records. Added 455 GiB machine-wide, weak. Added evaluator memory and CPU, which W&B has. `profile/rss_gib` is a peak (`ru_maxrss`), so labelled, current RSS 2.55-2.78 GiB.
12. **13:08-13:12 disturbance.** "4.5 min, about 60 s per run" was cut off by the snapshot. It ran 13:06-13:18 and cost 112-170 s per run, before any 95% crossing. Added a further slowdown from 13:18 on g8a_ref0 and from 13:26 on g8a_s1. "GPU 0 looks about 4% quicker" dropped as not stable.
13. **Static set.** The ratios hold: 0.93-1.01x, 1.01-1.09x, 1.25-1.38x on the learner clock. Added that the 90% figure rests on a flat stretch of one reference seed, +-0.117 h, and should only be quoted beside `training_dist`.

Smaller fixes:

14. Cumulative crossover "about 300k" is 222-228k on three runs, 328k on g8a_s1.
15. One TD3 step "14.4-15.9 ms" is 14.6-15.9 late, 15.1-15.6 early. 14.4 does not reproduce.
16. `train/total_step_time_ms` "17.4-18.7" is 17.4-19.2 by stage, 19.5-20.9 on g8a_s1 in its slow period.
17. Evaluation cost "72.8-76.2 s, 128 s, 0.8%, 1.3%" is stale: 80.4-88.8 s, 146.5 s, 0.83-0.91%, 1.51% at 13:12:49. Net rate gain "0.4-0.8" is 0.5-0.9 grad/s.
18. `sap_tile` 7.04M, 7.29M and mesh 2.64M, 2.45M are from-minimum figures beside median ones. Median: 6.96M, 7.28M, 2.47M, 2.39M. Their "quiet" label removed where `others` shows a second process.
19. PPO "6-12 processes" counted tokens. It is 3-6.
20. PPO `variant_4r_ent0_s1` "in progress, 3,398, 2.77, 16.4k" finished at 13:16: 3,537, 2.88, 17.3k. Added `sac_dev/variants/smoke_lr1e4`.
21. Whole-actor rates moved from [M] to [D]. TD3 share from [M] to [D from M windows]. Memory total "2.26 TB" is 2,263 GiB.
22. First-event range on the port 153-242 s is 153-243 s. Port launch 10:30:40 +-5 s is 10:30:39 from `lstart`. Sustained cores 0.97-1.00 from 216 s of `ps` is 0.99 from 2,146 s.
23. `RUN` learner RSS "9.4 to 16.6 GB" is 3.1 GiB first record, 9.3 GiB p5, 16.65 GiB max, in GiB.

Added checks that found no bias: the reference's frequent async checkpoints cost nothing visible, and the reference's rate shows no dips in its first 4 h.

## Weakest points

What a sceptical reader could attack, strongest first:

1. **One reference seed.** Every ratio's range is port seeds against a single reference curve. Sampling noise alone is +-0.03 to +-0.06 h to 90% and +-0.15 h at 95%. Seed spread on the port at 90% is 5%. The whole claimed gain at 90% is 11-16%, so one more reference seed could change it a lot.
2. **The gain is small and depends on definition.** 90% is 1.09-1.18x across the definitions here, 50% is no gain at all, and 95%, the milestone in the paper, has one crossing of four so far, at 1.07x. The port is behind the reference on steps at 580k. Any run that has not crossed by 13:58:29 (learner clock) has no gain at 95%.
3. **A first-crossing rule treats the two sides differently.** The reference's noisier 50-episode evaluations cross a threshold early by chance, and the port's 1,024-world ones do not. The port is also delayed up to 20,000 steps by cadence. Both hurt the port, and both are largest near 95%. So 95% will be hard to compare fairly whichever way it comes out.
4. **"1 core vs 64 cores" compares a measurement with an allocation.** Reference CPU is logged for 3 of 68 processes. Its actors spend 25-72% of their cycle in writes, so 64 is a ceiling. The port was never pinned, had 72 cores for its 7.8-9.5 core startup, and shows a saturated core. Pinning evidence for the reference is a script edited after the run.
5. **The reference ran under worse conditions than a clean original could.** Nice 10, 68 processes on 64 cores, actors stepping 1.6 to 3.4x slower than unloaded (PB:434, cause unverified). The original's learner is 1.5x faster than the port's and used under half its capacity at 400-500k. A reader can argue a better-configured original beats the port. No such run exists, in either direction.
6. **The port runs were not clean either, and the direction is against the port.** Five runs at once beside dev jobs, no load log, unexplained per-card slowdowns of 10% (460 s on g8a_s1), and 112-170 s lost during this evidence collection. `profile: true` adds an unmeasured cost. A clean port may be faster than shown, but that is not measured.
7. **The largest apparent gain is not measured.** The reference falls to 27.7 grad/s by 1.87M while the port holds 55 so far, which would give about 1.56x. The port has only reached 639k. The one late-stage figure, 55.3 grad/s, is pre-fix code for 5 minutes.
8. **Simulator ratios are the easiest thing to knock down.** All 3-22x figures are sphere arm against hull arm, mostly prototypes, 240 steps or a few thousand iterations, partly on a shared GPU. The "x 64" denominator was never run and is 6.7x above what 64 actors really did. On the mesh arm at the training world count the GPU env step is slower than the derived 64-core figure.
9. **The memory comparison rests on timing.** 455 GiB is a machine-wide rise that coincides with launch. It has no per-process log for actors or replay, and no pre-launch reading.
10. **Not one README table is from a single clean experiment.** Reference and port ran on different days. The cards differ except for g8a_s2, which alone gives 0.96x, 1.02x, 1.11x, near the low end at 75% and 90%.
