# RoboBallet on mjlab: design

Status: authoritative. If this file and a report disagree, this file wins. If this file and the original source disagree about behaviour marked "parity", the source wins: report it, do not paper over it.

## 1. Goal

1. Move the RoboBallet environment to mjlab (mujoco_warp on GPU).
2. Represent the robots by collision spheres: no gripper, a straight stick on the flange.
3. Train the RoboBallet algorithm (GNN + TD3 + hindsight) in that environment, using GPU parallelism efficiently and keeping CPU bottlenecks minimal. Profiling is a required deliverable.

Correctness comes before speed. No long training runs until the gates in section 9 pass.

## 2. Evidence

| Report | Scope |
|---|---|
| `docs/reports/env_spec.md` | Behaviour of the original env. Sections 4-9 define `BalletEnv`, section 15 lists acceptance tests. |
| `docs/reports/algo_spec.md` | Graph, network, TD3, hindsight, replay, evaluation. Section 12 is the checklist. The adversarial pass on this report did not finish, so the golden tests of gate G3 are the real check. |
| `docs/reports/mjlab_api.md` | mjlab and mujoco_warp usage, traps, section 3.5 skeleton. Read its `[CORRECTED]` notes. |
| `docs/reports/sphere_model.md` | Sphere fit, stick, excludes, section 8 checklist. |
| `docs/reports/paper_baseline.md` | Paper vs code, reference run, section 7.3 distributions, section 10 acceptance IDs (E*, T*). |
| `docs/reports/gpu_benchmark_tables.md` | Raw simulator benchmark tables. |

Scratch artefacts referenced by the reports are under `S/`, an analysis directory of the development machine that is not part of this repository. Useful: `S/spheres/` (candidate MJCF, `ik_lib.py`, `candidate_spheres.json`), `S/algo/` (`rb_fast.py`, `feat_torch.py`, `dump_net.py`), `S/scripts_baseline/out/` (golden traces, `static_boxes_*.npz`, `tasks_*.npy`), `S/mjlab_probe/direct_env_skeleton.py`, `S/bench/`.

Environment: Python 3.12, mjlab 1.6.0, mujoco and mujoco-warp 3.11.0, warp 1.19.0, torch 2.15.0. JAX exists only in a separate environment for the original, used only to produce golden data. Run as `python -m roboballet_mjlab...` with the package installed. No runtime import from the original `agents/`, `environments/`, `train/` or `launch/`.

## 3. Measured facts that shape the design

| Fact | Number | Source |
|---|---|---|
| Reference system (64 CPU actors) | 859 env steps/s, 34.9 train steps/s, 95% of tasks at 569k steps, 3.41 h | paper_baseline 5 |
| mujoco_warp kinematics + collision, one launch, 4r, 4096 worlds | 0.91 ms, 4.5 M world-steps/s | gpu_benchmark_tables |
| Same, 8r, 4096 worlds | 2.9 ms, 1.4 M world-steps/s | gpu_benchmark_tables |
| Env step with one-at-a-time rollback, stock pipeline, 4r, 4096 worlds | 6.2 ms (shared GPU), about 5 launches | `S/bench/results/env_proto.jsonl` |
| Same with custom pair kernels over an active-world list | 2.2 ms | `S/bench/results/env_active.jsonl` |
| Policy forward, 4r10t, compiled, default precision | 19.6 ms at B=1024, 75 ms at B=4096 | algo_spec 10.4 |
| Learner step, torch compiled `high`, 4r10t | 15.4 ms at B=128 (8.3k graphs/s), 63.7 ms at B=1024 (16k graphs/s) | algo_spec 10.4 |
| Learner step, eager | 53 ms at B=128 | algo_spec 10.4 |

Two conclusions:

- **The learner, not the simulator, limits speed.** At batch 128 and samples-per-insert 5, one GPU needs only about 1,660 inserted transitions/s, and a faster env buys at most about 1.9x. Real acceleration needs a larger batch across several GPUs, which changes the algorithm's denomination of LR, tau and exploration decay. So there are two modes, section 7.
- **The main CPU cost is Python dispatch, not data movement.** Eager to compiled is 3.4x at B=128. Nothing in the design copies through host memory. Profile before optimising further.

## 4. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Direct env: mjlab `Scene` plus a `Simulation` subclass running only `mjw.kinematics` + `mjw.collision`. No managers, no `rsl_rl`. | The original has no physics. Managers force physics, and their term shapes do not fit a graph agent. mjlab_api 3, 10. |
| D2 | PyTorch with dense tensors. | mjlab needs torch. A prototype already matches JAX to rounding. algo_spec 10. |
| D3 | Sphere model from `S/spheres/`: link0-4 3 spheres each, link5-7 5 each, stick 6, so 36 per arm. 16 excludes. Strict coverage, no radius shrink. | 0 false negatives against the mesh arm in about 300k samples. sphere_model 0, 7. |
| D4 | Stick: TCP at 0.160 m on the `attachment` frame +z, sphere radius 15 mm, spacing 25 mm, z = 0.0134 ... 0.1384. | Original flange-to-TCP is 0.15994 m; reachable volume within 0.75%. All sphere measurements used this candidate. env_spec prefers L=0.15 on one metric by a margin that 0.16 was not separately tested against; keep 0.16 as measured. |
| D5 | Keep the full SO(3) target check, 2.5 cm and 15 degrees, even though roll is physically meaningless for a straight stick. | "Apply the algorithm as is", and relaxing gains only 2 to 3 points of acceptance. Flag to the user; do not change silently. |
| D6 | Static scene: one table box `pos 0 0 -0.488, size 0.7841 0.762 0.488`, floor plane z=-0.95, no workpiece. Obstacles are mocap primitive boxes with per-world `geom_size`, `geom_rbound`, `geom_aabb`. | Mesh table costs 15x. Mocap gives weld id 0, reproducing the original's filter of box-box, box-table and box-link0. |
| D7 | Regenerate all scenes (section 6). Do not reuse `.pb` start states with spheres and stick. | 29% collide at home; targets are biased to the original tool. |
| D8 | Follow **stored data** over released code where they disagree: targets 3 cm outside the face with +z inward; rail sampling as fitted in paper_baseline 7.3. | env_spec D1, D2. The stored data produced the reference curve. |
| D9 | Stock collision pipeline with the original one-at-a-time random rollback. Custom kernels only if profiling shows the env above 20% of wall clock. | The env is an order of magnitude below the GNN, and stock is the ground truth. |
| D10 | `sap_tile` broadphase, filter `plane`+`sphere`, `njmax=0`, `disableflags` trick from mjlab_api 3.5, contact pool generous. Check `max(ncollision, nacon) > naconmax` on every launch and raise: overflow is otherwise silent and makes collisions disappear. | mjlab_api 6.4. |
| D11 | One process per GPU, lockstep acting and learning, on-GPU compact replay, features rebuilt at sample time. | No host copies, 45x less replay memory, exact samples-per-insert. |
| D12 | TensorBoard always, W&B optional behind `--wandb-project`. Do not read or print credentials anywhere. | |
| D13 | Replaced by section 14: bf16 learner and full-precision f32 acting, as the original. | The user asked for the closest match to the original first. |

## 5. Package layout and ownership

```
roboballet_mjlab/
  config.py structs.py        contract, do not change without updating this file
  assets/                     panda_spheres_stick.xml, spheres.json, excludes
  sim/world.py                build_world(WorldCfg, num_worlds, device) -> World
  sim/kinematic_sim.py        KinematicSim(Simulation)
  env/ballet_env.py           BalletEnv
  env/poses.py                obstacle augmentation, rigid helpers
  scenes/sampler.py           batched scene generation
  scenes/ik.py                batched damped least squares, torch
  scenes/io.py                .pt pool load and save, .pb and pbtxt import
  agent/features.py           build_graph(Obs) -> Graph
  agent/networks.py           PolicyNet, CriticNet
  agent/td3.py                train step, schedules, target updates
  replay/replay.py            compact ring buffer
  replay/hindsight.py         vectorised relabelling
  train/trainer.py            lockstep loop, multi-GPU
  train/evaluator.py          batched evaluation
  train/profiler.py           phase timers, CPU and GPU utilisation
  train/logger.py             TensorBoard, optional W&B, CSV
  scripts/                    train.py eval.py gen_scenes.py bench.py render.py launch_multi.py
  tests/                      pytest, one file per module, goldens under tests/golden/
```

### World API (`sim/world.py`)

```python
class World:
  num_worlds: int; num_robots: int; num_obstacles: int
  def set_scene(self, world_ids, scene: Scene) -> None      # bases, boxes (mocap, size, rbound, aabb)
  def set_q(self, q, world_ids=None) -> None                # zero-copy write of qpos
  def forward(self) -> None                                 # kinematics + collision, one captured graph, overflow check
  def robots_in_collision(self) -> torch.Tensor             # [W, R] bool, dist <= 0, both robots of a pair flagged
  def tcp(self) -> tuple[torch.Tensor, torch.Tensor]        # pos [W,R,3], rot [W,R,3,3], cloned
```

### Env API (`env/ballet_env.py`)

```python
class BalletEnv:
  def __init__(self, cfg: EnvCfg, num_worlds: int, scene_pool: Scene, device, seed)
  def reset(self, world_ids=None) -> None                   # draw from pool, auto-validate
  def observe(self) -> Obs                                  # current obs, after any auto-reset
  def step(self, action: torch.Tensor) -> StepOut           # action [W,R,7] in [-1,1]
```

`step` performs the order of env_spec 8.1 exactly, computes `StepOut`, then resets terminated worlds. `observe()` afterwards returns the new episode's first observation for those worlds. The original burns one call to reset with the action ignored; here the reset is immediate, which has no effect on stored data.

Semantics not to lose: acceleration clamp before integration; joint-limit clip leaves velocity untouched; velocity not clipped to 2.175; whole-robot rollback and zero velocity; uniform choice among colliding, not-yet-reset robots including stationary innocents; sequential target scoring with the dwell side effect inside the loop; strict thresholds; sticky done; scoring on the post-rollback pose; dwell decremented before the move; terminal evaluated after the step; penalty per rolled-back robot; collision never terminates.

A `step` invariant: the previous state is collision free, so rollback finishes within R rounds. Assert this in debug mode and count violations in training.

`mesh_tool` is a test-only arm model: original meshes, tool, TCP and table meshes, allowing gate G4 to run the reference checkpoint on the original start states. It must not affect the default path. `collisions_enabled=False` supports golden replays.

## 6. Scenes

`scripts/gen_scenes.py` writes pools outside git to `roboballet_mjlab/data/scenes/{R}r_{T}t_{O}o_{split}.pt`.

- Per config: 50,000 `train` (same size as the original), 5,000 `heldout`, plus static sets. First 4r/10t/30o, then 8r/40t/30o, 4r/40t/30o, 8r/10t/30o.
- Rails: paper_baseline 7.3 fit. Sequential shuffled order, floor robots truncated to `[-0.8, 0.382]` or `[-0.382, 0.8]`, ceiling robots on the released sampler, rejection by sphere collision at home. Draws must be bounded; count failures.
- Boxes: spans, centroid and rotation per env_spec 10.1, rejected one at a time against robots at home. Only robot contacts count.
- Targets: area-weighted face, uniform point, 3 cm outside the face, +z within 22.5 degrees of the inward normal, uniform roll, then IK filter: some robot at its placement, others at home, reaches a collision-free pose within tolerance. Solver in `scenes/ik.py` from `S/spheres/ik_lib.py`, aiming inside the tolerance (env_spec verifier notes). Fix and record the budget in the pool's metadata: the original filter's strength is unknowable.
- Verify every pool: 100% home-collision free on C MuJoCo for a random 1,000; every target has its witness IK solution stored in a metadata tensor and rechecked.
- Static evaluation: frozen OBB boxes `S/scripts_baseline/out/static_boxes_{4,8}r.npz`, fixed base placements, new stick-filtered target sets (10 sets), and the paper's tasksets reported separately with their reachability ceiling. The 8r static set in the release is inconsistent, env_spec D3: lift the boxes with the targets.

## 7. Training modes

Common loop, all on one GPU per process:

```
sigma = 0.7 * 0.8 ** (samples_or_steps / 1e5); a = clip(pi(features(obs)) + sigma * N, -1, 1)
out = env.step(a); write row to per-world episode buffers
on terminal: move finished episodes to replay; queue for hindsight; flush hindsight in batches
while grad_steps * B * world_size < SPI * (inserted_total - min_size):
    fused TD3 steps on one sampled megabatch
```

- **Parity mode**: `TrainCfg` defaults. W=64, B=128, SPI=5, one GPU. Curves compare to the reference step for step. This proves the port. Expect 1.4 to 1.9x wall-clock gain, no more.
- **Throughput mode**: W=1024 or more, B=512 to 4096 per GPU, several GPUs via `launch_multi.py`, gradients averaged with all-reduce on a flat vector. `schedule_by_samples=True`: with K = B * world_size / 128, tau becomes `1-(1-tau)^K`, and LR decay and sigma decay use K * steps. `lr_mult` and SPI are the variables to sweep. Compare on wall clock and sample axes, never on the step axis.

Replay: rows hold exactly `RobotState` plus action, reward, terminal, and an episode index into a per-episode table holding `Scene`. `s'` of row i is row i+1, so one obstacle augmentation is shared across the pair. Whole episodes enter the ring at termination. Capacity counts transitions as the original FIFO does. Sampling is uniform with replacement over rows that have a successor in the same episode.

Hindsight (algo_spec 5): exact rule, including `k=int(0.4*T)`, no episode when solved >= k, `t` in `[1, A-1]`, tip pose from stored rows, zero collision penalty, stored actions zeroed for robots dwelling in the hindsight timeline, fresh obstacle augmentation, terminal on last action. Both side effects sit behind `TrainCfg` flags. Pure tensor loop over time, batched over episodes, queued until `hindsight_flush_episodes` to bound Python dispatch.

Random phase: uniform actions until 2048 episodes have finished across all worlds, and until the first gradient step. Stagger initial resets, because aligned episodes correlate batches.

Evaluation: separate `World` with `eval_episodes` worlds. sigma 0, augmentation on, first terminal per world counted. Metrics keep the original names under `eval/{training_dist,heldout,static_dist,static_paper}/`: `num_targets_done`, `final_score`, `episode_length` (observations, so steps + 1), `total_collision_robot_steps`, `total_collision_penalty`, `total_mean_acceleration`. Headline is `num_targets_done / T`. Add `all_done_fraction`.

Checkpoint: atomic, every 15 minutes, full learner state and RNG, not the replay. Resume restores it. Training runs detached with `setsid nohup`, so it survives this session and any usage limit on the agent side.

## 8. Profiling deliverable

`train/profiler.py` records wall time per phase with `torch.cuda.synchronize` at phase edges in profiling windows only: policy forward, feature build, `world.forward`, rollback logic, scoring, reset, episode buffer, replay insert, hindsight, sampling, feature rebuild, TD3 step, logging, evaluation. It also records main-thread CPU time vs wall clock (Python-bound fraction), GPU utilisation through NVML (read-only), memory, `.item()` count, launches per step. Logged to TensorBoard.

`scripts/bench.py` runs the matrix R x W x B, reports T1 to T4 of paper_baseline 10.2. `docs/PROFILING.md` records baseline, each change and its measured effect, and what remains. Do not optimise on a shared GPU measurement: check `nvidia-smi` first. All six GPUs, 0 to 5, are available for this work. The user released GPU 0 on 2026-11-05.

## 9. Gates, in order

| Gate | Pass condition |
|---|---|
| G1 sim | E1, E12, E13 on the sphere world. Home has no contact. TCP in link7 at (0,0,0.26). 16 excludes per arm survive mjlab. Per-world box size and pose match C MuJoCo. Overflow guard fires when forced. Render a PNG and look at it. |
| G2 env | E2 and E3 on `S/scripts_baseline/out/golden_nocollision_her_4r10t.pkl` with an original-TCP hook; E4, E5; rollback outcome table of env_spec 5.4; acceptance table of env_spec 15. |
| G3 agent | E6, E7 (checkpoint `runs/paper_repro/paper_4r10t_s1/model_checkpoints/1870000`), E8, 20-step TD3 golden, parameter counts 8,949,575 and 17,980,164, LR schedule, hindsight rules. |
| G4 end to end, no training | Reference checkpoint in `BalletEnv` with `mesh_tool` and the original start states, through new features, network and evaluator: task rate at least 0.97 on `training_dist`. This exercises the whole stack before any long run. |
| G5 scenes | E10, E11, validity checks of section 6. Fraction of targets reachable is 100% by construction. |
| G6 smoke | 30-minute parity run: no NaN, no overflow, E9, realised SPI 5.0, hindsight fraction near 0.49 at start, loss magnitudes in range. |
| G7 profile | T1 to T4 measured, `docs/PROFILING.md` written, obvious CPU bottlenecks removed. |
| G8 parity learning | 4r/10t/30o parity run crosses paper_baseline 10.4 milestones, on 3 seeds if the first passes. |
| G9 throughput | Sweep K, SPI, `lr_mult` across GPUs. Target: 95% of tasks within 1 h wall clock, final metrics within paper_baseline 10.5. |
| G10 scale | 8r/40t/30o in the best mode. |

## 10. Rules for implementers

- Touch only your own files. No git commands: the lead commits. The working tree has the user's uncommitted W&B edits in `agents/evaluator.py`, `launch/launch.py`, `requirements.txt`, `train/train.py`, and untracked `environments/data_pb2.py`, `train/metric_logger*.py`. Leave them alone.
- Port the code's behaviour, cite `file:line` of the original in comments only where the behaviour is surprising, and keep the code small: no speculative options beyond `config.py`.
- Everything stays on the GPU. Avoid `.item()`, `.cpu()`, per-world Python loops in hot paths. A Python loop over R, T or time is fine where the spec requires order, with early exit when nothing fires.
- Use `CUDA_VISIBLE_DEVICES` to select a free GPU among 0 to 5, and do not leave processes running. Do not touch other users' processes or directories.
- A test that needs goldens from the old venv: generate with the original's JAX environment as `S/algo/dump_net.py` does, commit small `.npz` under `tests/golden/`, keep each under 5 MB.
- Report honestly what passed, what failed and what was not run. A skipped gate is not a passed gate.

## 11. Amendments from the Codex review

`docs/reviews/codex_design_review.md` has the evidence. These override the sections above.

1. **Takeover ignores the RTS flags.** `planning_env.py:801-816` runs whenever all targets are done, whatever `return_to_start_required` and the reward weight say, and it overwrites dwell zeroing. It also runs during the final dwell. `BalletEnv` must do the same. In hindsight, the same mutation path writes into stored actions, so reuse the same code rather than only zeroing dwelling robots. Test: weight 0, required False, all done, dwell nonzero, joints within 0.25 of start.
2. **Replay layout.** Each episode has A actions and A+1 states. The final state is never sampled and does not count toward capacity. Eviction is per oldest transition, as the original FIFO does (`launch/launch.py:154`), and a successor state and the episode's `Scene` stay alive while any surviving transition refers to them. Tests: one-action episode, nonzero final reward, wraparound across a terminal boundary, partial eviction of an episode.
3. **Per-episode metadata.** Keep the behaviour policy's step, a hindsight flag and the episode statistics per episode. Hindsight copies the real episode's statistics, `agents/actor.py:397-400`. Diagnostics average over sampled transitions, `train/train.py:764-767, 862-865`, including `trajectory_network_age`.
4. **Random phase.** Use per-world episode counters, latched at episode start: random if that world's count is below 32 or the step published at that time is 0. Never switch within an episode. The DESIGN's global 2048 is replaced by `random_episodes_per_world` scaled so `worlds * episodes` stays 2048 in throughput mode.
5. **SPI units.** All counters are explicit. Inserts include hindsight. With N ranks each holding a shard: local capacity and local minimum are the global values divided by N, and ranks agree on the permitted step count with an all-reduce. A fused call runs only as many steps as the budget permits.
6. **Throughput clock.** K scaling preserves selected timescales in sampled-transition equivalents. It does not reproduce sequential updates: Adam moments still advance once per update. Treat it as a changed optimiser, keep `lr_mult` explicit, and test K=1 against parity.
7. **Extra tests.** Deterministic nonzero target noise with injected normals beyond the clip and targets near the action bounds. A multi-world partial-reset test: retained `Obs` and `StepOut` snapshots stay unchanged after another world resets or steps, which catches aliasing.
8. **Terminal augmentation.** The original redraws the augmentation for the final stored observation, `agents/actor.py:281`. Bootstrapping is zero there, so this has no learning effect. Draw fresh for the terminal row anyway, since it costs nothing.
9. **Parameters.** Keep every registered parameter, so counts and checkpoint loading are exact. Unused forward computation may be skipped. A fixed flat-gradient vector treats absent gradients as zeros.
10. **G3 splits.** G3a is features, networks and TD3. G3b is hindsight with stored-action checks, tested once replay and env exist. Report them separately.

Update to section 11: the user switched Codex to Full Access. Roles, by the user's decision: Claude implements, Codex cross-validates by reading code and running its own checks. Codex does not implement production modules.

## 12. Benchmark: two agents on one env

Added by the user on 2026-11-05. FastTD3 is a **separate implementation**, by the user's decision. `train/trainer.py` and `agent/td3.py` serve only the original agent and have no FastTD3 branch or plugin interface. FastTD3 lives in its own package with its own config, update and training loop. It may import `env/`, `scenes/`, `structs.py`, `agent/features.py`, the building blocks of `agent/networks.py`, `replay/` and `train/evaluator.py`, and no file on the original's path changes to accommodate it. The shared evaluator is what makes the comparison fair.

| Agent | Module | Config |
|---|---|---|
| `original` | `agent/td3.py`, `train/trainer.py`, `scripts/train.py` | `TrainCfg` in `config.py`, parity mode, no tuning |
| `fasttd3` | `fasttd3/config.py`, `fasttd3/agent.py`, `fasttd3/trainer.py`, `fasttd3/train.py` | `FastTD3Cfg` in `fasttd3/config.py` |

### 12.1 What `fasttd3` is

The FastTD3 update rule on the RoboBallet GNN, with hindsight kept. Reference: github.com/younggyoseo/FastTD3 at commit 229ed59 (MIT licence), `fast_td3/train.py`, `fast_td3/fast_td3.py`, `fast_td3/hyperparams.py`. The SimbaV2 variant is not used.

Kept from the reference:

| Part | Rule | Reference |
|---|---|---|
| Critic | Twin C51, 101 atoms, categorical projection, cross-entropy summed over both | `fast_td3.py:118-158`, `train.py:465-472` |
| Clipped double Q | Use the projected distribution of whichever target has the lower expected value. Actor loss is `-min(Q1, Q2).mean()` on expected values | `train.py:449-456, 505-508` |
| No target actor | Next action is the online actor plus noise. Only the critic has a target | `train.py:427-433` |
| Target noise | `clamp(0.001 * N, -0.5, 0.5)`, so the clip acts after scaling, the opposite of `original` | `train.py:427-430` |
| Bootstrapping | `gamma ** n` with n=1 | `train.py:435` |
| Optimiser | AdamW, weight decay 0.1, LR 3e-4 for both, no grad clipping | `hyperparams.py`, `train.py:291-299` |
| Target update | Critic only, tau 0.1, after every update | `train.py:676` |
| Update schedule | 2 updates per env round, a fresh batch each. Policy updates when `i % 2 == 1`, using that batch | `train.py:644-673` |
| Exploration | Per-world scale uniform in [0.001, 0.4], resampled when that world ends. Noise is scale times a unit normal | `fast_td3.py:285-325` |
| Start | Updates begin after 10 env rounds, no random phase | `train.py:643` |
| Mixed precision | bf16 autocast, compiled | `hyperparams.py` |
| Multi-GPU | One independent env and replay per GPU with averaged gradients, so effective W and B scale with GPU count | `README`, `train_multigpu.py` |

Changed for this task, each with its reason:

| Part | Reference | Here | Why |
|---|---|---|---|
| Networks | MLP 1024-512-256 and 512-256-128 | RoboBallet GNN, critic head outputs 101 logits and has no tanh | The task's agent is a graph network. A smaller `ModelCfg` is a sweep variable: FastTD3's speed relies on shallow networks, and at 7 layers the GNN is the throughput limit |
| Batch | 32768 | 2048 per GPU, swept up to about 8000 | About 14 MB per graph needs about 460 GB at 32768, and each GPU has 178 GiB. Sizes come from algo_spec 10.4 at f32, so measure again under bf16 |
| Worlds | 1024 to 4096 | 1024 per GPU | Measured acting cost is small at this size |
| Replay | 10 M, per env ring | 2 M transitions, the original's size, on the shared compact ring | Swept to 10 M. Sampling stays uniform over transitions |
| Discount | 0.97 to 0.99 | 0.94 | Tuned for this task by the paper. Swept |
| Value support | Per task, typically [-10, 10] | [-1, 1] | The range the original's tanh critic enforces. A step's reward is at most 0.1, so a wide support leaves too few atoms. Swept to [-2.1, 1], the worst discounted penalty at gamma 0.94 |
| Timeouts | Bootstrap on truncation | No bootstrap on any terminal, `disable_bootstrap=True` in the reference's terms | The step count is an input feature, so the horizon is part of the state, and the original treats timeouts this way |
| Observation normalisation | On | Off | Features are already normalised by construction. Swept if the agent is stuck |
| Hindsight | None | Kept, with the same module and rule as `original` | Part of the RoboBallet agent, and rewards are sparse. A run with it off is reported |
| Stored action | Clamped in the env | Clamped to [-1, 1] before storing | Same action range in both agents |

n-step returns above 1 are not implemented until a sweep needs them. The reference's README suggests `num_steps=3` or disabling clipped double Q when an agent is stuck, so those are the first things to try.

### 12.2 Fairness

- Same scene pools, seeds, evaluation protocol and metrics, and the same evaluation world count.
- Report both axes: env transitions and wall clock. Record GPU count and model for every run. The grad-step axis is not comparable.
- `original` runs at parity defaults and is not tuned, because the paper already tuned it. `fasttd3` gets a declared sweep on `training_dist` and `heldout`. Report how many runs it took. Paper tasksets are never used for tuning.
- Compare 1 GPU against 1 GPU first, then each agent's best configuration on the available GPUs.
- At least 3 seeds per reported configuration, with every seed shown rather than only the mean.

### 12.3 Gates

| Gate | Pass condition |
|---|---|
| G11 FastTD3 | Golden tests from the reference code at the pinned commit: projection, clipped double Q choice, both losses, noise, update schedule, AdamW step. A verbatim excerpt with attribution, not an import from a clone. Small-task learning check: 1 robot, 1 target, no obstacles. 30-minute smoke on 4r/10t/30o: no NaN, support not saturated, sane losses |
| G12 Benchmark | `scripts/benchmark.py` runs both agents under 12.2. `docs/BENCHMARK.md` holds curves, tables, time to 50, 75, 90 and 95% of tasks, final metrics on all 4 evaluation sets, and the profile of each agent |

Order: G8 parity comes first, because a failing `original` would invalidate the comparison, and tuning of any kind waits until then. FastTD3 implementation may start once `train/evaluator.py` exists, since it cannot affect the original's path. Then G11, G12. G9, the K-scaling sweep of `original`, moves after G12, and G10 runs on whichever agent wins.

## 13. Accepted differences from the Codex world and env review

`docs/reviews/codex_simenv_review.md` has the evidence. The env matched the original on 63,118 real-contact rollbacks, bit-exact on q and qd. These differences stay, on purpose:

1. **Exact-zero contacts.** mujoco_warp emits a contact only for `dist < 0`, where C MuJoCo also reports `dist == 0`. It needs an f32 distance of exactly zero, and it did not occur in the differential. Fixing it would need a nonzero margin on every launch. Not fixed.
2. **Seed scope.** One generator serves scene choice, augmentation and rollback, so a run reproduces per batch and event schedule, not per world. An unrelated reset can change another world's later random choices without biasing their distribution. This matters only when replaying one world in isolation, so use `rollback='deterministic'` and forced `scene_ids` for that.
3. **Amendment 8 is met in distribution.** The terminal state's augmentation is the step's own fresh draw and is not shared with the next episode. The original's extra redraw changes only RNG consumption, and that row has no bootstrap. No separate draw is added.

## 14. Closest match to the original

By the user's decision on 2026-11-05: first make `original` as close to the released code as possible, reproduce it, and only then tune. Where an earlier section conflicts, this one wins. `TrainCfg` defaults are the parity configuration.

**Commits.** By the user: commit and push often, on branch `mjlab-port`. Each run records the git revision, and a patch of any uncommitted changes under `roboballet_mjlab/`, in its run directory.

**Checkpoints.** By the user: not the original's accumulating set, which reached 69 GB by 1.87M steps. A run directory holds one file, about 450 MB, overwritten every 30 minutes and on stop, 0.4 to 0.6 s per write. Separately, the policy alone is kept every 100,000 grad steps and at step 0, under `policies/`, about 36 MB each and under 1 GB by 1.87M steps, for later evaluation and video. `scripts/eval.py` takes these files. They cannot resume a run.

### 14.1 Float modes

Measured on the reference checkpoint, 1000 graphs, against JAX f32:

| Mode | Action error mean | p99 | max |
|---|---|---:|---:|
| `highest` | 9.6e-7 | 8.4e-6 | 3.2e-4 |
| `high` (TF32) | 5.6e-4 | 4.6e-3 | 0.36 |
| bf16 autocast | 5.6e-3 | 5.0e-2 | 1.16 |

Under autocast `high` and `highest` gave identical values, so every network matmul runs in bf16 there.

- **Learner: bf16**, as the original (`TrainCfg.bf16=True`). Measured 17.5 ms per step at B=128 against 14.3 ms for f32 `high`, about 57 steps/s against the reference's 34.9.
- **Acting and evaluation: f32 at `highest`**, as the original's actors. 1.5 ms per forward at W=64. Evaluation is also the mode G4 needs.
- **Global default is `highest`.** Only the learner call may set `TrainCfg.matmul_precision`, then it restores the previous value. `TD3.__init__` must not change the global setting, because env and feature code run in the same process. Toggling on one compiled module works: returning to `highest` reproduced the JAX numbers.
- **The learner's bf16 rounds as the original's.** Codex found that plain autocast does not. Four things differed, and all four are fixed in `agent/networks.py`, checked against the original's flax MLP under JAX on GPU in `tests/test_agent_bf16.py`:
  - GELU: jax's constants are bf16, 0.0446777 and 0.796875, and every operation rounds. Now bit-exact on all 33,282 bf16 values up to magnitude 8.
  - LayerNorm: flax computes in f32 with `mu2 - mu**2` and `rsqrt`, then rounds the output.
  - Dense: `F.linear` on 2-d input rounds product and bias as one, while flax rounds the product and then the sum. Invisible at init, when biases are zero.
  - `torch.compile` drops the rounding between bf16 operations unless `torch._inductor.config.emulate_precision_casts` is set. It costs nothing measurable: 17.7 ms per step at B=128 with it, 17.8 ms without.

  What remains, on the reference checkpoint: 0.1 to 0.4% of embedding outputs land on the other side of a rounding boundary, from the order of f32 reductions in LayerNorm, and after one dense layer that reaches most elements at about one bf16 ULP. So values are not bit-identical through the full network: action error against the original on GPU is 7.2e-3 at 4r/10t, down from 1.0e-2, and 8.0e-3 at 8r/40t, down from 1.2e-2, where the original on CPU against the original on GPU is 9.3e-3 and 1.1e-2. `tests/test_agent_adversarial.py` holds the port to that bound. The size of the bf16 effect itself also came closer: 9.3e-3 against the original's 9.6e-3, up from 8.0e-3. Parameter drift over 500 updates on the same batches and noise, measured by Codex against fresh original runs: policy RMS 6.0e-4 and critic RMS 4.5e-5, where the original against a repeat of itself gives 4.6e-4 and 4.2e-5. So the port sits on the scale of the original's own sensitivity to rounding, from one control pair. The fix did not measurably reduce this drift, which was 5.9e-4 and 4.2e-5 before it: what it fixed is the arithmetic of each operation, and the size of the bf16 effect. Equal learning curves are for G8 to show.
- Benchmarks and runs use one `TD3` per process. Scripts building several in one process measured up to 2.4 times slower for a reason not established.

### 14.2 Exact rules for the trainer

| Rule | Value | Source |
|---|---|---|
| Worlds, batch, fused | 64, 128, 20. A megabatch of 2560 per fused call, sliced in order. Only full fused calls | `train.py:363-410` |
| Start of learning | When inserted transitions reach `capacity // 10`, 200,000 at T=10, counting hindsight | `launch.py:145-149` |
| Rate | Run a fused call while `grad_steps * 128 + 2560 <= 5 * (inserted - min_size)` | limiter centre, `launch.py:145-149` |
| Random phase | Per world, latched at episode start: random if that world's finished count is below 32 or the published step is 0. Uniform in [-1, 1] | `actor.py:232-239, 597-600` |
| Sigma | Per world, latched at episode start from the step that world fetched at its **previous** episode, then fetches the current one | `actor.py:224, 612` |
| Noise | i.i.d. per joint per step, clipped to [-1, 1] and stored as clipped. Never store the env's internal changes for real episodes | `actor.py:262-264` |
| Hindsight | Real episode inserted first. Stored hindsight actions may exceed 1 through takeover and are not clipped | `actor.py:476, 619-646` |
| Logged losses | `train/policy_loss` and `train/twin_critic_loss` are sums over the 20 steps | `train.py:398-404` |
| Stagger | Reset each world once at a different offset across the first episode, discarding the partial rows with `buffer.discard`, because aligned episodes correlate batches. Discarded rows do not count as episodes | algo_spec 11.3 |

### 14.3 Differences that remain, on purpose

1. **Arm and scenes** in G8b: the user's goal. G8a removes them.
2. **Torch instead of JAX**, with goldens at `highest` and comparable bf16 effect.
3. **Lockstep instead of 64 async actors.** The reference's actors run at varying speeds, and the rate limiter tolerates a band of +-10,240 samples, about 4 fused calls.
4. **Policy refresh.** The original fetches per actor once per episode. One batched forward cannot hold 64 parameter sets cheaply, so the trainer keeps one acting copy, refreshed each time as many episodes have ended as there are worlds. That period follows the real episode length: 120 rounds early, about 61 late. A fixed 120 rounds, the first version, doubled the actual parameter age late in training, 140 updates against the original's 75 in Codex's scheduling model. Two differences remain: a refresh can change a world's weights inside an episode, which the original never does, and the recorded `train/trajectory_network_age` therefore reads about 75 updates higher late on than the weights' actual age.
5. **Hindsight delay.** Hindsight episodes are flushed in batches of `hindsight_flush_episodes=8`, at most 960 transitions late, about 30 ms per flush and 2 ms per round. Why 8: 8 episodes of 120 rows permit 4,800 samples at SPI 5, inside the limiter's band of 10,240. The first value of 64 permitted 38,400 in one round, 15 fused calls with no acting between them, which live Reverb never allows, since it blocks the writer instead.
6. **Limiter.** Reverb allows samples anywhere in a band of +-10,240 around the centre, so it starts with 10,240 samples of credit at 200,000 inserts, and a learner faster than its actors runs at the top of the band, one slower at the bottom. The port runs at the centre, with full fused calls only: first learning comes at 200,520 inserts, 80 updates later than Reverb's credit would allow. Measured by Codex on a live table.
7. **Reused embeddings.** The port embeds the three distinct target and obstacle inputs once and reuses them, while the original embeds every node. Forward is bit-identical. In bf16 the gradient of that MLP rounds differently, relative difference 0.5%, and the original shows the same effect when reused against full, so this is rounding noise rather than a bias. Kept because the full evaluation costs learner time and no effect was measured.
8. **Log windows.** `act/*` and `train/trajectory_network_age` describe the batches of the logged losses here, and one megabatch ahead of them in the original, about 20 updates on the age. `act/env_sampling_time_s` and `act/episode_write_wall_time_s` are zero here, and `act/episode_wall_time_s` includes the learner: compare none of the three with the reference.
9. **Evaluation** runs on a separate env in the same process at fixed intervals, on the same GPU, with more episodes. It has no effect on training data, but it does count in wall clock.

### 14.4 Gate G8 splits in two

- **G8a, same task as the reference.** `mesh_tool` arm and the original start states, nothing else changed. Curves compare step for step with `runs/paper_repro/paper_4r10t_s1`, on the milestones of `paper_baseline` 10.4. If this fails, the cause is in the port and not in the geometry.
- **G8b, the user's task.** Spheres, stick and regenerated scenes. Compare with G8a and with the reference.

Only seed 1 of the reference exists, so reference noise is unknown. Run 3 seeds of G8a where the first passes. Whether to rerun the original stack for more reference seeds is left open: it costs 64 CPU cores for about 15 h per seed.

Any failure is reported rather than papered over.

## 15. Release

By the user on 2026-11-05: the end goal is a GitHub release. It comes after G8 and G12, because its claims rest on their results. The package is not renamed before then, since runs are importing it.

| Item | Decision |
|---|---|
| Repository | `roboballet-mjlab` |
| Python package | `roboballet_mjlab` |
| Branch `main` | The port of RoboBallet to mjlab. FastTD3 is its main agent, and the goal is tens of times faster training than the reference |
| Branch `test` | `main` plus the test scripts that verify sameness with the original |
| Visibility | Private |
| Structure | mjlab's layout and way of running only: `src/roboballet_mjlab/tasks/...`, `pyproject.toml`, `train` and `play` commands. The env stays direct-style |
| Target check | Full SO(3) stays for now, for a straight stick too |

`mjlab train` itself cannot run this env, since `mjlab/scripts/train.py:110` constructs `ManagerBasedRlEnv`. So the release has its own commands of the same form. mjlab's `runner_cls` hook is the same extension point the SAC runner would use.

Speed is judged by wall clock to the task-rate milestones against the reference, 3.46 h to 95%: 10 times is about 21 minutes, 30 times about 7. Not yet measured. The original agent stays in `main` as the baseline, which the benchmark needs. This placement is the lead's default and the user may change it.

Licences are checked in `docs/LICENSES.md`.

The release holds:

| Part | Content | Source |
|---|---|---|
| Same as the original | Task, reward, observations, GNN, TD3, hindsight, replay rules, evaluation, each with how it was checked | Gates G1 to G8, Codex reviews |
| Different from the original | mjlab and Warp instead of the CPU simulator, torch instead of JAX, lockstep instead of async actors, sphere arm and stick, regenerated scenes, FastTD3, each with its measured effect | DESIGN 13, 14.3, 12.1 |
| File layout | What each directory holds and how they depend on each other | DESIGN 5, current tree |
| Demo videos | The trained policy on the paper's tasksets and on random scenes, from the sphere run and the FastTD3 run, with failures not edited out | `scripts/render.py`, extended to record video |
| Results | Curves against the reference, milestones, profile, benchmark, and all seeds shown | `docs/PROFILING.md`, `docs/BENCHMARK.md` |
| Running it | Install, train, evaluate, render, tests, from a clean clone that has actually been tried | README |

To settle at release:

- **Licences.** Checked, with what the release must do, in `docs/LICENSES.md`.
- **Large files.** `data/` is 1.7 GB, `roboballet_mjlab/data/` 840 MB and untracked, a checkpoint about 450 MB. Decide what goes in git, what in release assets, and what `scripts/gen_scenes.py` regenerates.
- **What stays out.** The W&B edits in the original's files, Codex's scratch directories, run directories, and any machine-specific paths in docs and scripts.
- **History.** A new repository or a fork of the original, which decides whether the commits of this branch carry over.

## 16. SAC, a third agent

By the user on 2026-11-05: include rsl-rl, referring to github.com/leggedrobotics/rsl_rl_sac (e0d243a, BSD-3-Clause), released with Sabatini, Li and Hutter, arXiv 2605.24975. TD3 and SAC are siblings rather than a line of descent: both came out in 2018 in the line from DDPG, and SAC's second version took clipped double Q from TD3. The fork implements that second version.

What the code shows:

- It is a fork of rsl_rl at version 4.0.1 under the same names, `rsl-rl-lib` and `rsl_rl`, while mjlab 1.6.0 pins `rsl-rl-lib==5.4.2`, which has only PPO and distillation. The two cannot be installed together.
- Its models are MLPs on flat observation groups, and its replay stores whole observations per env, with no hindsight.
- Its runner and update add RND and symmetry options that this task has no use for.

So the update rule is ported into a separate package, as FastTD3 is, and the fork is not installed:

| Part | From the fork | Here | Why |
|---|---|---|---|
| Update | `algorithms/sac.py`: twin Q with MSE, tanh-Gaussian actor, learned temperature, target entropy, tau, policy frequency, gradient clip | Same | The rule under test |
| Networks | MLP | RoboBallet GNN with a mean and log-std head per robot node | Same reason as FastTD3 |
| Replay | Per-env ring | Shared compact ring and hindsight module | Same as the other agents |
| Timeouts | Bootstraps from the timeout observation, one of the paper's three changes | No bootstrap on any terminal | The step count is an input feature, as in 12.1. A run with the paper's rule can be swept |
| Discount | 0.998 | 0.94 | Tuned for this task by the original paper. Swept |

Layout `sac/{config,agent,trainer,train}.py`, with the same evaluator, logger, fairness rules of 12.2 and benchmark as the others. The policy's mean is what gets evaluated. Gate G13 mirrors G11: goldens from a verbatim excerpt with attribution, a small-task check and a 30-minute smoke.

## 17. PPO, a fourth agent, and the benchmark of all four

By the user on 2026-11-05: run four agents, the TD3 port, FastTD3, rsl-rl SAC and rsl-rl PPO, and report the speed benchmark, the performance and the learning curves.

PPO uses `rsl_rl` as installed, 5.4.2, the version mjlab pins, through a `VecEnv` adapter on `BalletEnv`, so this is the standard rsl-rl path. Layout `ppo/{config,models,vecenv,train}.py`. Its actor and critic are the RoboBallet GNN, given to rsl-rl by `class_name`, with the same tanh mean as the policy. On-policy learning has no use for a replay, so **PPO has no hindsight**. It is the one agent without it, which the results must say beside its curves. Timeouts do not bootstrap, as in the other agents. Gate G14 mirrors G11: tests of the adapter's semantics against the env, a small-task check and a 30-minute smoke.

G12 then covers four agents and reports:

| Report | Content |
|---|---|
| Speed | Env steps/s, samples/s, grad steps/s, wall clock per phase from the profiler, GPU count and model, for each agent at 1 GPU and at its best GPU count. Wall clock to 50, 75, 90 and 95% of tasks against the reference's 1.00, 1.58, 2.50 and 3.46 h |
| Performance | Final metrics on all 4 evaluation sets, at least 1000 episodes each: task rate, all-done fraction, episode length, collisions, acceleration. Every seed shown |
| Learning curves | Task rate and episode length against env steps and against wall clock, from `metrics.csv`, with each agent's logged learning rate alongside |

