# Performance against the original: evidence for the README

> Copied into this repository on 2026-11-06 from the development tree, where the package was named `mjlab_ballet` (here `src/roboballet_mjlab`). Paths in this file refer to that tree. Run directories, scratch directories and `docs/reviews/xval_*` are not part of this repository. Absolute machine paths in the Paths paragraph were replaced by descriptions. The text is otherwise unchanged.

Verified copy of `performance.md`. Base snapshot 2026-11-05 13:16 KST: numbers re-derived from the raw file wherever one exists, with `metrics.csv` filtered to wall_time <= 13:16:00, latest grad step 513k to 533k, latest evaluation 500k to 520k. G8 runs still training. Anything marked **(13:36)** was read at 13:36:13 KST: latest grad step 577k to 599k, latest evaluation 560k to 580k. A number that moved because a run went on is not an error.

**Marks.** [M] measured, recomputed here from raw files or logs. [D] derived, and from what. [T] paper text. [F] read from a paper figure. [S] status document only, no raw file found. **unverified**: I could not re-derive it from a raw file.

**Paths.** `ROOT`=the project directory on the development machine, holding `roboballet/` and `runs/`. `REPO`=ROOT/roboballet. `S`=ROOT/runs/mjlab_port_analysis. `G4`=S/trainer_g4. `RUN`=ROOT/runs/paper_repro/paper_4r10t_s1. `W`=`<scratch>/evidence/perf_work` (saved `compare_reference.out` and the collector's scripts). `V`=`<scratch>/evidence/verify_perf` (this check's scripts, written independently of `W`). `<scratch>` is the evidence session's scratch directory, not kept in this repository. `XP`=G4/original_checks/crossplay_g8a_s1_400k.

## 0. Where the claim stands

| Claim | State | Numbers |
|---|---|---|
| Same weights, same task rate, training scenes | Supported | Port 0.9922, original 0.9931, difference -0.0008 ± 0.0007, 95% CI -0.23 to +0.06 points [D] |
| Same weights, same task rate, paper's tasksets | Supported | Port 0.9880, original 0.9873, difference +0.0007 ± 0.0011, 95% CI -0.14 to +0.28 points [D] |
| Same weights, other four metrics | Supported | Largest of 10 differences is 1.8 SE (all-done on training scenes, -1.0 ± 0.6 points, 95% CI -2.2 to +0.1). 1.9 SE on the 8192-world run alone, 2.0 on 8192 + 4096 [D]. Says "not separated", not "equal" |
| Same weights, a policy the port trained (g8a_s1 at 400k), training scenes, 5 metrics | Supported | Task rate: original 0.9117 ± 0.0022, port 0.9105 ± 0.0009. Largest difference 1.0 SE (acceleration). Section 1 |
| Learning: 50, 75, 90% milestones in steps | Supported, 3 seeds | Median 200k, 280k, 420k; reference 196k, 291k, 436k. Port resolution is 20k, so neither side is shown earlier |
| Learning: 95, 97, 98%, 600k to 1M checks | **No measurement** | No port run has reached 95% (13:36): trailing 50k mean 0.941 to 0.946 at its latest evaluation, 560k to 580k. Reference crossed at 577,680 |
| Final policy after training | **No measurement** | Every row of paper_baseline 10.5 is pending |
| Learning curve matches on acceleration | **Contradicted, 150k to 400k** | At 200-250k reference 37.9, all 4 port runs 51.3 to 56.9. Inside the seeds' range from 400k, through 500-550k (13:36). Section 2.6 |
| Port keeps up at the latest steps | **Mixed, and moving** | Training scenes: 450-500k reference 0.941 ± 0.004, port 0.919 to 0.930; 500-550k reference 0.941 ± 0.005, port 0.929 to 0.942 (13:36). Tasksets the other way: 450-500k reference 0.875 ± 0.006, port 0.908 to 0.927; 500-550k reference 0.886 ± 0.006, port 0.900 to 0.913 (13:36) |
| Faster to 90% in wall clock | Supported, modest | From launch to the learner reaching the crossing step: 2.16 to 2.26 h (3 seeds) against 2.51 h, **1.11 to 1.16x** [D]. Mixes learning luck with rate |
| Faster at equal grad steps | **Slower first, faster later** | From launch, reference / port: 100k 0.89 to 0.91x, 200k 0.90 to 0.98x, 300k 0.98 to 1.06x, 400k 1.05 to 1.11x, 500k 1.10 to 1.15x [D]. **The reference is ahead until about 300k** |
| Faster to 95% or to the end | **No measurement** | Section 4.1 has an estimate, marked as one. Measured instead (13:36): the same 577,680 grad steps took the reference 3.46 h, g8a_s2, s3, ref0 2.98 to 3.02 h, 1.15 to 1.16x. That is a step count, not 95% |

Same-weights agreement is tested at two checkpoints: the reference's at 1.87M and g8a_s1's at 400k. Neither covers 150-350k, where section 2.6 shows the curves disagree.

## 1. Same weights on both simulators

Reference checkpoint at 1,870,000. Original: `actor.run_episode` with the original configs, 32 processes on CPU (`G4/scripts/orig_train.py`, `orig_static.py`). Port: `mjlab_ballet.scripts.eval`, `mesh_tool` arm. ± is one SE. Port SE for task rate is measured. For all-done it is binomial [D from the fraction and worlds]. For length, collisions and acceleration the JSON holds only means, so `(x)` is [D: the original's per-episode SD / sqrt(worlds)], not a measurement of the port's spread.

Weights: `scratchpad/ref_1870000.npz`, read by the original, and `S/ckpt/paper_4r10t_s1_1870000.pt`, read by the port, are identical: 158 policy tensors, max abs difference 0 [M, `V/wt.py`]. Their conversion from the orbax checkpoint: unverified, this venv has no JAX.

Code version: G4 ran 05:26 to 06:53. `train/evaluator.py` changed at 09:40, `agent/networks.py` and `config.py` at 10:26, all after G4. `env/`, `sim/`, `agent/features.py`, `agent/td3.py` are older than G4 by mtime [M]. `trainer_status.md`'s last section records only the `task_rate_se` edit to the evaluator. Git is off limits in this check, so the diff itself is unverified. The 400k check below ran on the code as it is now.

**Training scenes, `training_dist`**

| Metric | Original [M] | Port 8192 worlds [M] | Port 4096 [M] | Port 1024 [M] | Port pooled [D, worlds-weighted] | Port - original [D] | Diff/SE [D] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episodes | 1920 | 8192 | 4096 | 1024 | 13312 | | |
| Task rate | 0.9931 ± 0.0007 | 0.9922 ± 0.0003 | 0.9919 ± 0.0005 | 0.9940 ± 0.0008 | 0.9922 ± 0.0003 | -0.0008 ± 0.0007 | -1.1 |
| All done | 0.9417 ± 0.0054 | 0.9304 ± 0.0028 | 0.9294 ± 0.0040 | 0.9453 ± 0.0071 | 0.9313 ± 0.0022 | -0.0104 ± 0.0058 | -1.8 |
| Episode length | 60.57 ± 0.46 | 61.38 (0.22) | 61.67 (0.32) | 60.06 (0.63) | 61.37 (0.18) | +0.81 ± 0.50 | +1.6 |
| Collision robot-steps | 4.07 ± 0.29 | 4.04 (0.14) | 4.29 (0.20) | 3.98 (0.40) | 4.11 (0.11) | +0.05 ± 0.31 | +0.2 |
| Mean acceleration | 16.45 ± 0.11 | 16.58 (0.06) | 16.67 (0.08) | 16.34 (0.16) | 16.59 (0.04) | +0.14 ± 0.12 | +1.2 |
| Makespan given all done, s | 5.58 ± 0.03 | 5.59 [D] | 5.62 [D] | 5.55 [D] | 5.60 [D] | +0.02 | |
| Source | `G4/original_checks/train_results/p0..p31.json` rows | `G4/g4_mesh_8192.json` results.training_dist | `G4/g4_mesh_large.json` same key | `G4/g4_mesh.json` same key | `W/analysis.py` | | |

**Paper's 10 tasksets**

| Metric | Original [M] | Port 2000 worlds [M] | Port 80 [M] | Port 4000 [S, unverified] | Port pooled 2000+80 [D] | Port - original [D] | Diff/SE [D] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episodes | 2560 | 2000 | 80 | 4000 | 2080 | | |
| Task rate | 0.9873 ± 0.0007 | 0.9880 ± 0.0008 | 0.9900 ± 0.0038 | 0.9876 ± 0.0006 | 0.9880 ± 0.0008 | +0.0007 ± 0.0011 | +0.6 |
| All done | 0.8887 ± 0.0062 | 0.8885 ± 0.0070 | 0.9125 ± 0.0316 | 0.888 | 0.8894 ± 0.0069 | +0.0008 ± 0.0093 | +0.1 |
| Episode length | 68.15 ± 0.51 | 67.82 (0.58) | 66.33 (2.90) | 67.8 | 67.76 (0.57) | -0.38 ± 0.77 | -0.5 |
| Collision robot-steps | 7.97 ± 0.30 | 7.37 (0.34) | 6.09 (1.71) | 7.95 | 7.32 (0.34) | -0.65 ± 0.45 | -1.4 |
| Mean acceleration | 18.44 ± 0.11 | 18.42 (0.13) | 17.92 (0.65) | 18.37 | 18.40 (0.13) | -0.04 ± 0.17 | -0.2 |
| Makespan given all done, s | 6.05 ± 0.04 | 6.02 [D] | 6.01 [D] | | 6.01 [D] | -0.04 | |
| Source | `G4/original_checks/static_results/p0..p31.json` rows | `G4/g4_mesh_large.json` results.static_paper | `G4/g4_mesh.json` results.static_paper | `REPO/mjlab_ballet/docs/reviews/trainer_status.md:117-121` | `W/analysis.py` | | |

- Makespan = (mean length given all 10 - 1) x 0.1 s, the rule of paper_baseline 5.4. Original from rows. Port [D]: length given all = (L - (1-P) x 121) / P. Valid because all 4,480 original episodes obey it: every incomplete episode has length 121, and the formula gives back 56.821 and 61.524 exactly.
- Resolvable at 2 SE [D]: training scenes task rate 0.14 points, all-done 1.2 points, length 1.0, collisions 0.6, acceleration 0.24. Tasksets 0.22 points, 1.9 points, 1.5, 0.9, 0.34 (within-set SEs below: 0.18, 1.5, 0.7, 0.7, 0.14). Smaller true differences are not excluded.
- Episodes are independent enough for these SEs. All 32 static processes reuse the same 80 PRNG keys (`orig_static.py`: `PRNGKey(i + 10 * p)`), but a one-way test on key within set gives F = 0.92 (done), 0.95 (length), 1.15 (collisions), with 70 and 2480 df [M]. SE across process means is 0.00056 against 0.00074 used, so the table is conservative.
- Conservative hides differences as well. The design is balanced over 10 fixed sets, so between-set variance is not sampling error. With within-set SEs (original [M], port scaled from it by sqrt(2560/2080) [D]) the taskset rows read: task rate +0.0007 ± 0.0009 (+0.8), all-done +0.0008 ± 0.0073 (+0.1), length -0.38 ± 0.34 (-1.1), **collisions -0.65 ± 0.37 (-1.8)**, acceleration -0.04 ± 0.07 (-0.5). None passes 2 SE, and collisions now tie the largest.
- Taskset 0 to 9, original, per set [M, `static_results`]: task rate 0.933, 1, 1, 1, 1, 0.988, 1, 0.953, 1, 1. Collision robot-steps 30.44, 1.90, 5.62, 2.88, 2.56, 15.66, 0.00, 11.41, 0.02, 9.21.

**Same weights at 400k: a policy the port trained, training scenes only.** `runs/g8a_s1/policies/policy_00400000.pt` equals `XP/policy.npz`, 158 tensors, max abs difference 0 [M]. Original: `orig_train.py`, 24 CPU processes, `XP/p[0-9]*.json` (480 episodes) and `XP/large/p[0-9]*.json` (1920). Port: `scripts/eval`, two runs of 8192 worlds, seeds 2 and 3, `XP/large/port_8192_s{2,3}.json`, written 13:15 and 13:17, after the collector's snapshot. Source `V/last.py`.

| Metric | Original, 2400 episodes [M] | Port, 16384 worlds [M, pooled D] | Port - original [D] | Diff/SE [D] |
|---|---:|---:|---:|---:|
| Task rate | 0.9117 ± 0.0022 | 0.9105 ± 0.0009 | -0.0012 ± 0.0024 | -0.5 |
| All done | 0.4746 ± 0.0102 | 0.4707 (0.0039) | -0.0039 ± 0.0109 | -0.4 |
| Episode length | 98.06 ± 0.56 | 98.54 (0.21) | +0.49 ± 0.60 | +0.8 |
| Collision robot-steps | 22.28 ± 0.81 | 22.39 (0.31) | +0.11 ± 0.87 | +0.1 |
| Mean acceleration | 40.15 ± 0.27 | 40.44 (0.11) | +0.29 ± 0.29 | +1.0 |

`(x)` as above. This is the only same-weights check on a port-trained policy, and the only one on the current code. It does not reach section 2.6: at 400k the acceleration gap has nearly closed. Static sets were not run at 400k.

**Against `trainer_status.md` "Gate G4".** All means and SEs of the original and of the 8192, 4096, 2000, 1024 and 80 world columns match to the printed digits. Disagreements and gaps:

| Item | Status document | Raw data | Effect |
|---|---|---|---|
| Port 4000-world static column, bold in the document, and the per-taskset port values under it | 0.9876, 7.95, 18.37 | **No raw file, unverified.** `G4/scripts/static_check.py` only prints, no log in `G4/logs` or the scratchpad, not in any `g4_mesh*.json` | Marked [S]. The raw 2000-world run gives 7.37, not 7.95. For the README cite the 2000-world JSON or rerun with `--json` |
| "Reference curve, 1.80 to 1.87M, 54 evals of 10" | 54 evals, collisions 9.35 ± 0.88 | 54 evals needs an upper bound of 1.88M (last step 1,877,140), which gives 9.35 ± 0.89. At 1.87M: 48 evals, 9.19 ± 0.92, 0.9881, 66.78, 17.57 | Label only. `W/refwin.py`, `W/refwin2.py` |
| Reference curve SEs, training column | 4.01 ± 0.39, 16.26 ± 0.15 | 4.01 ± 0.41, 16.26 ± 0.16 with n-1 | Rounding level |
| 8192-world seed 3 | Stated | The JSON has no seed field and no log or script with `--worlds 8192` exists. Seeds 1 and 2 are in `G4/scripts/chain.sh` | Unverified |
| `g4_mesh_first.json` | Not cited | Identical to `g4_mesh.json` in every value | Do not count it as a separate run. Not pooled here |

Not comparable, do not put in this table: the reference's training-window values (static collisions 9.35, acceleration 17.56). At the same weights the original itself gives 7.97 and 18.44, so a single checkpoint can sit about 0.9 from its window mean in acceleration, far more than the 0.11 SE. Any final comparison carries that checkpoint-to-checkpoint variation.

## 2. Learning: reference and port so far

### 2.1 What the runs are

| Run | Arm | Scenes | Init | Static sets logged | Source |
|---|---|---|---|---|---|
| reference | original | `data/start_states/4r_10t_30o.pb`, 50 episodes per eval | seed 1 | `eval/static_dist`, 10 episodes | `RUN/provenance.txt:2`, `REPO/launch/launch.py:311-326` |
| g8a_s1, s2, s3 | `mesh_tool` | same `.pb`, 1024 worlds per eval | seeds 1, 2, 3 | `eval/static_paper`, 80 worlds | `ROOT/runs/g8a_s*/provenance/*/config.json`, `g8a_s1.log:24-28` |
| g8a_ref0 | `mesh_tool` | same | seed 1, **reference's step-0 weights** | same | config.json `init_weights`, `g8a_ref0.log` 10:34:35 |
| g8b_s1 | `spheres_stick` | regenerated pool | seed 1 | `heldout`, `static_dist`, `static_paper` | `g8b_s1.log:19-23` |

**Correction to the brief:** all three `g8a_s*` use the same arm as the original. What sets `g8a_ref0` apart is its starting weights. `g8b_s1` is a different task, so it does not belong in a "same as the original" table.

### 2.2 The tag for the paper's tasksets

The reference logs them as **`eval/static_dist`** (`RUN/tensorboard_log/events...evaluator.static_dist`, name at `launch.py:326`). The port logs the same tasksets as **`eval/static_paper`**. In the port `eval/static_dist` is a different thing: new stick-filtered target sets (`4r_10t_30o_static.pt`, only `g8b_s1` logs it).

`compare_reference.py:49-51` aliases the reference's `static_dist` to `static_paper`, added at 13:05 during this work, so the reference now appears on both rows. **The `eval/static_dist` row must not be read across:** it puts the reference on the paper's tasksets next to `g8b_s1` on easier regenerated sets, 0.874 against 0.973 at 500k, which is not a gain. The `eval/static_paper` row is the valid one.

Same scenes and protocol, checked in code and data (`W/scene_cmp.py`):

| Item | Original `static_dist` | Port `static_paper` | Same? |
|---|---|---|---|
| Targets | `data/paper_test_tasksets/test_10targets.pbtxt`, `agents/evaluator.py:49` | same file, `scripts/gen_scenes.py:285`, stored in `data/scenes/4r_10t_30o_static_paper.pt` | **Yes**, max abs difference 0.0 in position and rotation [M] |
| Base placements | `data/mujoco_world/4_pandas_world.xml:28-37`, no randomising, `launch/hparams.py:160` | `scene.base_pos`, `base_rot` | **Yes** [M] |
| Start joints | keyframe `home`, same file `:43` | `scene.q0` | **Yes** [M] |
| Box centroids and sizes (`paper_baseline` 4.9 [CORRECTED] says the same) | 30 boxes, V-HACD + OBB, fresh per process, `environments/planning_env.py:357-379,417-424,744-751` | frozen `scripts_baseline/out/static_boxes_4r.npz` | **Yes**, centroids within 3.1e-5 m, sorted spans within 9.1e-6 m, over all 32 processes [M]. `paper_baseline` 4.9 says up to 0.44 mm between its processes. These JSONs win |
| Box rotations | stored matrix, the inverted one of `env_spec` D4, goes into collision and features | transpose, `gen_scenes.py:240` | **No, on purpose.** Differs from a process's stored matrix on 4 to 6 boxes by more than 1 degree. In p0: boxes 0, 1, 2, 11 by 32.9, 5.9, 6.3, 1.4 degrees. Boxes 19, 26, 27 only relabel axes by 90 degrees [M]. Effect with the inversion put back: 7.99 and 18.47 against 7.95 and 18.37 [S, unverified] |
| Sigma, feature noise | 0, 0: `agents/evaluator.py:77`, `launch/hparams.py:164,167-168` | 0, none: `train/evaluator.py` | **Yes** |
| Obstacle augmentation | on, random per box per observation, `planning_env.py:941-943` | `augment_obstacles: true`, config.json | **Yes** |
| Episode end, length | `len(observations)`, max 121, `agents/actor.py:290` | `final.step + 1`, `train/evaluator.py:105` | **Yes**, 121 in both tables |
| Episodes per evaluation | 10, one per set, `launch.py:313` | 80, 8 per set, `g8a_s1.log:28` | **No.** Same in expectation, port less noisy per eval |
| Cadence | asynchronous, about every 1,500 steps, 1,244 evals [M] | every 20,000 steps, `config.json eval_every_steps` | **No.** A ±5% window at 400k: 23 evals, 230 episodes, against 3 evals, 240 episodes [M] |

Verdict: same scenes except the deliberate box-rotation fix, same protocol except the number and timing of evaluations.

### 2.3 Task rate by step

Each cell is `compare_reference` window mean over ±5% of the step, ÷ 10. Saved output: `W/compare_reference.out`. All [M]. Reference from TensorBoard, port from `metrics.csv`.

**`eval/training_dist/num_targets_done`**

| Step | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.160 | 0.182 | 0.083 | 0.215 | 0.263 | 0.208 |
| 200,000 | 0.600 | 0.665 | 0.595 | 0.695 | 0.638 | 0.709 |
| 300,000 | 0.821 | 0.816 | 0.810 | 0.829 | 0.844 | 0.862 |
| 400,000 | 0.898 | 0.910 | 0.895 | 0.904 | 0.905 | 0.909 |
| 500,000 | 0.939 | 0.940 (13:36) | 0.933 | 0.923 | 0.928 | 0.929 |
| 600,000 | 0.955 | pending | pending | pending | pending | pending |
| 800,000 | 0.973 | pending | pending | pending | pending | pending |
| 1,000,000 | 0.985 | pending | pending | pending | pending | pending |
| 1,200,000 | 0.987 | pending | pending | pending | pending | pending |
| 1,500,000 | 0.990 | pending | pending | pending | pending | pending |
| 1,870,000 | 0.993 | pending | pending | pending | pending | pending |

**Paper's tasksets: reference `eval/static_dist`, port `eval/static_paper`**

| Step | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1, ceiling 0.890 |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.078 | 0.065 | 0.013 | 0.098 | 0.066 | 0.088 |
| 200,000 | 0.520 | 0.544 | 0.486 | 0.564 | 0.516 | 0.505 |
| 300,000 | 0.753 | 0.776 | 0.705 | 0.729 | 0.816 | 0.593 |
| 400,000 | 0.857 | 0.879 | 0.840 | 0.847 | 0.884 | 0.695 |
| 500,000 | 0.874 | 0.908 (13:36) | 0.920 | 0.901 | 0.903 | 0.701 |
| 600,000 | 0.912 | pending | pending | pending | pending | pending |
| 800,000 | 0.933 | pending | pending | pending | pending | pending |
| 1,000,000 | 0.961 | pending | pending | pending | pending | pending |
| 1,200,000 | 0.973 | pending | pending | pending | pending | pending |
| 1,500,000 | 0.980 | pending | pending | pending | pending | pending |
| 1,870,000 | 0.989 | pending | pending | pending | pending | pending |

The ceiling 0.890 is `meta.reachability_ceiling` of the pool: on the stick arm 11% of the paper's targets have no solution, by design.

Rows to leave out of a README:
- **At or below 20k.** Window artefact: the reference at "2,000" averages evals at steps 0 and 2,420, the port only step 0. The apparent acceleration gap of 21.7 against 0.5 there is this; at step 0 itself they agree, 0.8 against 0.4 to 0.9.
- **100k.** The reference window holds one eval of 50 episodes (0.160 ± 0.018), and 0.207 with paper_baseline's ±25k window (6 evals), 0.213 with ±10k (2 evals).
- 600k stays pending (13:36): the window 570k to 630k holds one evaluation at most.

Number of episodes behind a cell [M]: at 200k the port is one eval of 1024, SE about 0.005; the reference is 2 evals, 100 episodes, 0.600 ± 0.013. At 400k the port is 3 evals, SE about 0.002; the reference 6 evals, 300 episodes, 0.898 ± 0.007. Reference SEs from per-episode lines in `RUN/logs/evaluator/0` and `/1`. Port from `eval/training_dist/task_rate_se`.

### 2.4 Milestones

First step whose trailing 50k mean reaches the rate, and hours since first log. Output of `compare_reference`, [M]. **The steps stand. These hours are not a fair clock**, see "Hours from launch" below.

**`training_dist`**

| Task rate | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 196,060, 1.00 h | 200,000, 1.07 h | 200,000, 1.01 h | 180,000, 0.89 h | 180,000, 0.89 h | 180,000, 0.91 h |
| 75% | 291,320, 1.58 h | 280,000, 1.51 h | 300,000, 1.51 h | 240,000, 1.19 h | 260,000, 1.30 h | 260,000, 1.31 h |
| 90% | 435,520, 2.50 h | 400,000, 2.10 h | 440,000, 2.22 h | 420,000, 2.09 h | 420,000, 2.11 h | 400,000, 2.01 h |
| 95% | 577,680, 3.46 h | pending | pending | pending | pending | pending |
| 97% | 764,080 | pending | pending | pending | pending | pending |
| 98% | 916,500 | pending | pending | pending | pending | pending |

**Paper's tasksets**, same rule applied by me to the tags of 2.2 [D, `W/part2.py`]

| Task rate | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 |
|---:|---:|---:|---:|---:|---:|
| 50% | 221,660, 1.12 h | 220,000, 1.18 h | 240,000, 1.21 h | 220,000, 1.09 h | 220,000, 1.09 h |
| 75% | 324,740, 1.75 h | 320,000, 1.71 h | 340,000, 1.71 h | 320,000, 1.59 h | 320,000, 1.60 h |
| 90% | 565,080, 3.33 h | 500,000, 2.60 h | 480,000, 2.42 h | 480,000, 2.39 h | 520,000, 2.63 h |
| 95% | 934,600, 6.16 h | pending | pending | pending | pending |

- Definitions differ between documents. paper_baseline 10.4 lists the reference at 188k, 285k, 420k, 569k, 739k, 911k and "3.41 h": that is a rolling mean of 5 evals. `compare_reference` gives 196k, 291k, 436k, 578k, 764k, 917k and 3.46 h. I reproduce both from TensorBoard [M, `W/part2.py`]. Use one rule on every run: `compare_reference`'s.
- Port step resolution is 20,000, the eval cadence. Step differences below that are not resolved. The trailing window holds 3072 port episodes against 300 to 400 for the reference [D: 3 x 1024, 6 to 8 x 50]. Mean age of the evaluations in the window: port 20k, reference 21k to 25k, so the rule favours neither by more than the resolution. Two other rules agree: the reference subsampled to the port's 20k grid crosses at 200k, 300k, 440k, 580k; a centred ±25k window gives reference 166k, 267k, 406k against s1, s2, s3 of 166k, 166k, 146k; 255k, 275k, 215k; 366k, 415k, 395k [D, `V/final.py`]. So steps agree within the resolution, and no side is shown earlier.
- **Tasksets 90% is a first passage, not a level reached.** At 13:36 g8a_s2's trailing mean has fallen back to 0.894 at 580k, g8a_ref0 is at 0.900, g8a_s3 has been at 0.900 to 0.907 since 480k. The reference stays above 0.90 from 575,040 [M]. Evaluations of 80 worlds have SE about 0.01 here. Do not quote 480k-520k against 565k as a lead.
- `RUN` static crossings in `paper_baseline` 5.4 carry a [CORRECTED] note: over an equal 250-episode window 0.90 at 565,080 / 3.33 h and 0.95 at 919,320 / 6.03 h. Mine agree at 0.90 and read 934,600 / 6.16 h at 0.95, since the window is 50k steps, not 250 episodes.
- Hours: first log is 3.34 min after launch for the reference (19:30:46 to 19:34:06) and 2.79 to 4.06 min for g8a runs, 2.57 for g8b [M]. All five G8 processes started in the same second, 10:30:39 (`ps -o lstart`, pids in the provenance directory names). The 10:31:14 in the logs comes after about 35 s of loading. Launch to the first logged grad step (2,000): reference 268 s, G8 240 to 296 s, so start-up costs about the same. The GPU index is not in provenance; `speed.md` section 2 has it from `nvidia-smi`, one GPU per run (not re-read here). Machine load is under the rate line below.

**Hours from launch.** The hours in the tables above start at the first log, which leaves out 3.34 min on the reference and 2.8 to 4.1 min on the port, and stop at the line the evaluator writes. On the port that line comes about 3 s after the step, since evaluation blocks the learner. On the reference the evaluator runs on its own for about 2.8 min after fetching the weights: step 435,520 fetched at 22:01:17, written at 22:04:07 (`RUN/logs/evaluator/0`). Below, both clocks start at process launch and stop when the learner reaches the step, from `train/policy_loss` wall_time, linear between 1,000-step logs [D, `V/final.py`].

Equal grad step, no learning noise. This is the cleanest speed comparison there is:

| Step | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | g8b_s1 | Reference / g8a runs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.500 | 0.560 | 0.551 | 0.555 | 0.559 | 0.548 | 0.89 to 0.91 |
| 200,000 | 1.031 | 1.141 | 1.053 | 1.056 | 1.061 | 1.053 | 0.90 to 0.98 |
| 300,000 | 1.646 | 1.682 | 1.560 | 1.558 | 1.564 | 1.556 | 0.98 to 1.06 |
| 400,000 | 2.279 | 2.171 | 2.064 | 2.058 | 2.074 | 2.056 | 1.05 to 1.11 |
| 480,000 | 2.801 | 2.559 | 2.467 | 2.457 | 2.475 | 2.455 | 1.09 to 1.14 |
| 500,000 | 2.935 | 2.667 | 2.568 | 2.557 | 2.578 | 2.555 | 1.10 to 1.15 |
| 550,000 (13:36) | 3.273 | 2.941 | 2.852 | 2.845 | 2.874 | 2.844 | 1.11 to 1.15 |
| 577,680 (13:36) | 3.464 | not yet | 2.991 | 2.982 | 3.021 | 2.982 | 1.15 to 1.16 |

At the crossing, training scenes, same steps as the first table above:

| Task rate | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | Reference / s1, s2, s3 |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 1.008 | 1.141 | 1.053 | 0.955 | 0.959 | 0.88 to 1.06 |
| 75% | 1.592 | 1.575 | 1.560 | 1.257 | 1.363 | 1.01 to 1.27 |
| **90%** | **2.508** | **2.171** | **2.264** | **2.158** | 2.174 | **1.11 to 1.16** |
| 95% (13:36) | 3.464 | pending | pending | pending | pending | |

Tasksets, same way: 50% reference 1.163, port 1.157 to 1.256, 0.93 to 1.01x. 75% 1.801, 1.658 to 1.782, 1.01 to 1.09x. 90% 3.376, 2.457 to 2.699, 1.25 to 1.37x, unstable as noted above.

`speed.md` 2.3 gives 1.13 to 1.18x at 90% from launch. It stops the reference's clock at the evaluator's written line, 2.556 h. Both reproduce [D]. The difference is those 2.8 min.

Grad steps/s by 100k band, from `train/policy_loss` timestamps [D, `W/rate.py`]: reference 64.0, 52.3, 45.2, 43.9, 42.3, then 36.4 over 500k to 1M and 29.8 over 1M to 1.8M. Port g8a_s2 56.4, 55.3, 54.9, 55.0, 55.1. g8a_s1 56.9, 47.8, 51.3, 56.7, 56.1. Last 50k of every port run at the 13:16 snapshot: 48 to 50.

**That last figure is machine load, not a slowdown.** All five runs fall together from 55-57 to 44-46 grad/s over 13:05-13:15 and return to 52-55 after 13:20 [M, `V/rate2.py`, `V/final.py`]. It coincides with this workflow's analysis jobs, including 24 CPU processes and two GPU 4 evaluations of the crossplay, 13:07 to 13:17. That link is timing only, unverified. Cost about 5,500 to 7,200 steps per run, about 2 min [D]. It falls after every 90% crossing on training scenes, and lowers the ratios at 500k and later by about 1%. g8a_s1's own slow period, 70k to 310k, about 460 s, cause not established, is in `speed.md` 2.6 and explains why g8a_s1 is the low end of the ratios up to 400k.

### 2.5 Acceptance, paper_baseline 10.4, median of g8a_s1, s2, s3

Output of `compare_reference --median`, [M]. Limits were written before any run.

| Check | Limit | Reference | Median | State |
|---|---|---:|---:|---|
| task rate 0.50 reached | by 250,000 | 196,060 | 200,000 | PASS |
| task rate 0.75 reached | by 370,000 | 291,320 | 280,000 | PASS |
| task rate 0.90 reached | by 550,000 | 435,520 | 420,000 | PASS |
| task rate 0.95 reached | by 750,000 | 577,680 | | pending |
| task rate 0.97 reached | by 1,000,000 | 764,080 | | pending |
| task rate 0.98 reached | by 1,300,000 | 916,500 | | pending |
| task rate at 200,000 | >= 0.45 | 0.600 | 0.665 | PASS |
| collision steps at 200,000 | <= 40 | 20.3 | 26.9 | PASS |
| task rate at 400,000 | >= 0.8 | 0.898 | 0.904 | PASS |
| episode length at 400,000 | <= 112 | 99.9 | 99.4 | PASS |
| collision steps at 400,000 | <= 40 | 22.8 | 24.4 | PASS |
| task rate at 600,000 | >= 0.9 | 0.955 | | pending |
| episode length at 600,000 | <= 100 | 87.8 | | pending |
| collision steps at 600,000 | <= 25 | 12.8 | | pending |
| task rate at 800,000 | >= 0.94 | 0.973 | | pending |
| episode length at 800,000 | <= 90 | 75.5 | | pending |
| collision steps at 800,000 | <= 18 | 9.18 | | pending |
| task rate at 1,000,000 | >= 0.96 | 0.985 | | pending |
| episode length at 1,000,000 | <= 82 | 70.0 | | pending |
| collision steps at 1,000,000 | <= 13 | 7.37 | | pending |

The limits are wide on purpose, so PASS means "not badly off", not "the same". Reference values differ slightly from 10.4's column (0.602, 120.7, 22, ...) because 10.4 uses a ±25k window.

### 2.6 Where the curves do not agree

**Mean acceleration in evaluation.** Band means [M, `W/bands.py`]. ± is SE across the reference's evals in the band. Port bands hold 2 or 3 evals. Tag `eval/training_dist/total_mean_acceleration`; for the tasksets the reference `eval/static_dist/...`, the port `eval/static_paper/...`.

| Steps | Training scenes: reference | g8a_s1, s2, s3 | g8a_ref0 | Tasksets: reference | g8a_s1, s2, s3 | g8a_ref0 |
|---|---:|---:|---:|---:|---:|---:|
| 100-150k | 58.8 ± 2.0 | 63.6, 65.6, 63.9 | 58.3 | 64.0 ± 0.9 | 68.9, 65.6, 66.8 | 68.1 |
| 150-200k | 47.0 ± 3.6 | 58.4, 59.0, 65.5 | 56.0 | 53.9 ± 1.3 | 60.8, 58.1, 68.1 | 60.0 |
| **200-250k** | **37.9 ± 0.5** | **52.2, 56.9, 53.7** | **51.3** | **39.2 ± 0.7** | **53.0, 56.2, 57.2** | **53.2** |
| 250-300k | 37.0 ± 1.3 | 47.2, 43.7, 50.2 | 46.7 | 37.8 ± 0.7 | 49.8, 44.8, 52.0 | 49.2 |
| 300-350k | 36.2 ± 1.7 | 42.0, 44.3, 41.7 | 41.2 | 37.9 ± 0.9 | 43.3, 46.2, 45.7 | 43.1 |
| 350-400k | 37.5 ± 1.0 | 38.7, 41.0, 41.7 | 39.4 | 41.1 ± 0.7 | 42.4, 47.0, 40.6 | 40.4 |
| 400-450k | 37.0 ± 0.7 | 36.0, 40.1, 37.4 | 36.6 | 39.5 ± 0.6 | 37.2, 43.5, 41.2 | 39.7 |
| 450-500k | 34.6 ± 1.2 | 33.7, 37.6, 35.9 | 36.7 | 38.2 ± 0.4 | 36.7, 39.6, 38.1 | 41.5 |
| 500-550k (13:36) | 33.5 ± 0.8 | 32.8, 33.9, 32.3 | 35.3 | 36.2 ± 0.6 | 37.8, 37.0, 34.1 | 39.2 |

- The reference falls between evals at 166,240 (56.5) and 203,780 (37.1), then stays near 37 until 450k. No port run drops like that: they decline steadily. Same in both evaluation sets, 4 of 4 runs, including the one with the reference's own start weights.
- Acting barely shows it: `act/total_mean_acceleration` over 200-250k is reference 71.7, port 72.5 to 73.6. The sign is the same in every band from 100k to 400k, by 0.04 to 1.5 against all three seeds [M], and exploration noise dominates this tag.
- At equal task rate the port's policy at sigma 0 moves harder in this phase. Converged by 400-500k, and at 1.87M with the same weights the metric agrees (section 1), and the formula is the same (`environments/planning_env.py:843`, `mjlab_ballet/env/ballet_env.py:254`), so the metric's definition is not the cause.
- **Cause not known.** One reference seed cannot separate seed luck from a real difference. A same-weights evaluation of the reference's checkpoints at 200k, 250k and 300k on both simulators would separate them. Those checkpoints exist, every 10k in `RUN/model_checkpoints`. So do `runs/g8a_s1/policies/policy_00200000.pt` and `_00300000.pt`, which `XP/run.sh` could score on the original the same way. It needs compute and was not run here.

**Latest steps** [M, `V/final.py`, read 13:36]. 500-550k holds 3 port evaluations per run, 550-600k holds 1 or 2 and is partial:

| Metric | Band | Reference | g8a_s1, s2, s3 | g8a_ref0 |
|---|---|---:|---:|---:|
| Task rate, training | 450-500k | 0.941 ± 0.004 | 0.930, 0.924, 0.919 | 0.926 |
| | 500-550k | 0.941 ± 0.005 | 0.942, 0.937, 0.929 | 0.933 |
| | 550-600k | 0.950 ± 0.004 | 0.944, 0.948, 0.947 | 0.947 |
| Episode length, training | 450-500k | 92.1 ± 1.2 | 93.9, 94.4, 95.5 | 95.2 |
| | 500-550k | 89.9 ± 1.8 | 90.6, 90.2, 93.1 | 91.9 |
| | 550-600k | 88.6 ± 1.3 | 89.8, 87.2, 87.3 | 89.0 |
| Collision robot-steps, training | 450-500k | 13.3 ± 1.4 | 20.1, 18.5, 22.3 | 19.0 |
| | 500-550k | 13.8 ± 2.1 | 15.8, 19.8, 18.3 | 16.5 |
| | 550-600k | 12.5 ± 1.5 | 12.5, 13.8, 15.5 | 14.3 |
| Task rate, tasksets | 450-500k | 0.875 ± 0.006 | 0.911, 0.927, 0.908 | 0.882 |
| | 500-550k | 0.886 ± 0.006 | 0.913, 0.906, 0.900 | 0.914 |
| | 550-600k | 0.906 ± 0.005 | 0.934, 0.886, 0.913 | 0.895 |

The reference's lead on training scenes at 450-500k rests on one seed and 300 episodes, and task rate is back inside the seeds' range at 500-550k. It stays a little ahead at 550-600k. Collisions stay in its favour. The port's lead on tasksets also shrinks, and the reference is inside the range at 550-600k. Neither lead belongs in a README. Trailing 50k task rate at the latest evaluation: 0.941 at 560k for g8a_s1, then 0.946, 0.943, 0.946 at 580k, against the reference's 0.95 first reached at 577,680. The 95% crossing is the next real test.

**Collisions, training scenes.** The reference is below all three seeds in 5 of the 7 bands from 150k to 500k, and also at 500-550k (13:36) [M, `V/s26.py`]. The final same-weights numbers agree (4.07 and 4.04), so this too is about the policies during learning.

**Inside the spread:** over 150k to 450k the reference's task rate is inside the three seeds' range or at most 0.004 below it, and its episode length is inside or 0.03 steps above [M, `W/bands.py`]. All-done at the windows of 200k, 300k, 400k: reference 0.000, 0.145, 0.417 from evaluator logs, port 0.001 to 0.050, 0.156 to 0.201, 0.409 to 0.471 from `eval/training_dist/all_done_fraction` [M].

## 3. The paper

Copy checked: `S/dl_arxiv/paper.pdf`, arXiv:2509.05397v1, 45 pages. PDF page = printed page (p12 and p31 checked). `S/paper_dl` holds only HTML and text of the same version. The final Science Robotics text was not available, so nothing here checks it.

| Item | Paper value | Page | Mark | Against paper_baseline 2.6 |
|---|---|---:|---|---|
| Actors | 64, asynchronous | 31 | [T] | Same |
| Compute | Learner on one NVIDIA A100, each actor on one Intel Ice Lake Xeon core | 31 | [T] | Same |
| Time to 95% | "6 to 24 hours depending on problem size" | 31 | [T] | Same. **No value per configuration.** 4r/10t being the 6 h end is inference |
| Further improvement | "additional 2 to 6 days" | 31 | [T] | Same |
| Training step, B=128 | 33 ms 4r/10t, 52 ms 4r/40t, 80 ms 8r/40t | 11-12 | [T] | Same. The sentence starts on p11, the numbers are on p12. Whether it includes sampling wait is not stated |
| Max training steps | 10 x 10^6 | 45 | [T] | Same |
| Number of seeds | Not stated in the text. Fig. 3A draws 5 lines per configuration | 41 | [F] | New |
| 33 ms x 628,638 steps to 95% | 5.8 h | | [D from the step time and Fig. 3A median] | New. Fits the 6 h lower end |

Our setups, for the same table: reference on 64 cores of an AMD EPYC 9365 plus one B200 (`S/reports/paper_baseline.md` 5.1; the 64 comes from `run_one.sh`, edited after the run, so unverified from the run's own records, `speed.md` section 1), 3.41 h to 95% with 5-eval rolling, 3.46 h with `compare_reference`'s rule [M]. Port on one B200 (`ROOT/runs/g8a_s1/provenance/*/system.json`), no measurement to 95%. "Paper 6 h against ours" compares different hardware and is not a port effect.

**Fig. 3A, 4 robots, 10 tasks, p41** (y axis "Fraction of Tasks Solved", x axis "Training Steps"). Which scenes it plots and any smoothing are not stated. The figure is vector, so I read the 5 blue paths from the PDF (`W/fig3.py`, calibrated on the grid lines) as well as using the by-eye values of paper_baseline 2.7. All [F].

| Steps | paper_baseline by eye [F] | Fig. 3A paths min, median, max [F] | Reference [M] | Port s1 to s3 [M] |
|---:|---:|---:|---:|---:|
| 100k | | 0.030, 0.078, 0.132 | 0.160 | 0.084 to 0.215 |
| 200k | about 0.50 | 0.443, 0.485, 0.524 | 0.600 | 0.595 to 0.695 |
| 300k | | 0.714, 0.732, 0.800 | 0.821 | 0.810 to 0.829 |
| 400k | about 0.90 | 0.853, 0.873, 0.906 | 0.898 | 0.895 to 0.910 |
| 500k | about 0.92 (5.4) | 0.913, 0.917, 0.926 | 0.939 | 0.923 to 0.940 (13:36) |
| 600k | about 0.93 | 0.935, 0.945, 0.948 | 0.955 | pending |
| 800k | about 0.96 | 0.962, 0.967, 0.970 | 0.973 | pending |
| 1M | about 0.975 | 0.977, 0.977, 0.979 | 0.985 | pending |

Crossings, paths min, median, max [F]: 0.50 at 189k, 204k, 213k. 0.75 at 279k, 312k, 322k. 0.90 at 392k, 444k, 453k. 0.95 at 623k, 629k, 672k. 0.97 at 774k, 854k, 865k.

- By-eye 0.90 at 400k is above the paths' median, and 0.93 at 600k just below their minimum. Use the path values.
- Reference and port are both above the paper's band at 200k and 300k. The reference stays above it at every step except 400k. So the port matches the local reference better than it matches the figure. Different scenes or a smoothed curve could explain this; the paper does not say.
- Reference reaches 95% about 7% earlier in steps than the fastest paper line (578k against 623k).
- The paths continue under the axis clip to 2.3M. At 1.87M they read 0.986, 0.989, 0.992. **Not visible in the printed figure, do not cite as a published value.**

## 4. What is still missing

### 4.1 Rows waiting for G8

- 10.4: crossings 0.95, 0.97, 0.98, and all checks at 600k, 800k, 1M. Latest step 577k to 599k (13:36).
- 10.5: every row, below.
- **Estimate, not measured:** from 13:36 at a constant 52 to 57 steps/s, the range before 13:00 and since 13:20, 1M arrives around 15:35 to 15:55 KST and 1.87M around 19:45 to 20:30 KST. The late-training rate on this code is unmeasured. Reference to 95% at 3.46 h. If the port crossed at the same step it would take 2.98 to 3.02 h, 1.15 to 1.16x, and that step time is now measured (2.4); the crossing is not. `trainer_status.md`'s 9.6 h to 1.87M has not been measured since the changes in its last section. `paper_baseline` 0.3 [CORRECTED] bounds the parity gain to 95% at 1.35 to 1.4x for a 62 to 65 steps/s torch learner, and these runs measure 55 to 57 undisturbed.

### 4.2 paper_baseline 10.5, exact list

Conditions as written: at <= 2M train steps, parity mode, N >= 1,000 fresh episodes on training and held-out scenes. Static cell: frozen OBB set, 10 sets of 10, >= 20 repeats, with "new stick-filtered target sets". That wording fits G8b. For G8a the static row reads "(original targets)", so `static_paper`.

| Metric | Run-A value, 10.5 | Pass vs Run-A | Pass vs Run-B | Original at 1.87M [M] | Port same weights [M, or D] | G8a final | Where it comes from |
|---|---:|---:|---:|---:|---:|---|---|
| Train dist fraction | 0.992 | >= 0.975 | within -1.0 points | 0.9931 | 0.9922 | pending | `eval/training_dist/task_rate` |
| Train P(all 10) | 0.930 | >= 0.85 | within -5 points | 0.9417 | 0.9304 | pending | `.../all_done_fraction` |
| Train makespan given all 10 | 5.66 s (p50 5.3, p90 7.5) | <= 6.5 s | within +10% | 5.58 s | 5.59 s [D] | pending | **Not logged.** [D] from length and all-done |
| Train ep len (all) | 62.1 | <= 72 | within +10% | 60.57 | 61.38 | pending | `.../episode_length` |
| Train collision robot-steps/ep | 4.61 | <= 8 | within +50% | 4.07 | 4.04 | pending | `.../total_collision_robot_steps` |
| Train mean acceleration metric | 16.5 | <= 22 | within +20% | 16.45 | 16.58 | pending | `.../total_mean_acceleration` |
| Static fraction | 0.989 (original targets) | >= 0.96 | within -1.5 points | 0.9873 | 0.9880 | pending | `eval/static_paper/task_rate` |
| Static P(all 10) | 0.897 | >= 0.75 | within -8 points | 0.8887 | 0.8885 | pending | `.../all_done_fraction` |
| Static makespan given all 10 | 5.90 s | <= 7.0 s | within +10% | 6.05 s | 6.02 s [D] | pending | **Not logged.** [D] |
| Static collision robot-steps/ep | 9.1 (sd 6.4) | <= 15 | within +50% | 7.97 | 7.37 | pending | `.../total_collision_robot_steps` |
| Held-out gap | not measured | <= 1 point below train | same | **cannot exist**: the original has no held-out set | not measured | **Not measurable as configured**: g8a logs `heldout: skipped, no pool` (`g8a_s1.log:25`) | G8b only |

Also asked by 10.5: fraction of 50,000 scenes with all targets reachable by the new filter, 100% by construction for the regenerated pool. Not a G8a row.

Run-A values above are training-window means at 1.75 to 1.88M (n 1030 and 903 episodes), not a frozen policy. Re-derived from `RUN/logs/evaluator/{0,1}` and TensorBoard [M, `V/last.py`]: 0.9923, 0.9301, 5.665 s, p50 5.3, p90 7.5, 62.07, 4.61, 16.49; 0.9889, 0.8970, 5.905 s, 9.10 with sd 6.42. For a README the same-checkpoint column is the fair one. G8b for information, not a same-task comparison: `g8b_s1` at 500k window, training 0.929, held-out 0.932, `static_dist` 0.973, `static_paper` 0.701 of a 0.890 ceiling [M, `W/g8b.py`].

Run-B of 10.5 (original stack with a stick) does not exist, so the "vs Run-B" column cannot be applied.

### 4.3 End-of-run evaluation needs a separate command

Yes. Reasons, from code:

- `Trainer.run` (`train/trainer.py:528-547`) has no final evaluation. It evaluates only when 20,000 steps have passed (`:310`, `:533`), then saves a checkpoint (`:545`).
- The periodic evaluation runs 1024 worlds and **80 static worlds, 8 repeats per set**: `Evaluator(..., num_worlds=cfg.eval_episodes)` with the default `static_worlds=80` (`scripts/train.py:90`, `train/evaluator.py:48`). 10.5 asks for at least 20 repeats.
- Runs do not stop at 2M: `total_grad_steps` is 10,000,000 in config.json. Someone must send SIGTERM, or restart with `--resume` and a lower `--total-grad-steps`.
- Port snapshots exist only at 100k multiples (`policy_snapshot_every_steps`), `policies/policy_XXXXXXXX.pt`. So there is no snapshot at 1.87M. Options: 1.8M on both sides (reference checkpoint `RUN/model_checkpoints/1800000` exists, needs conversion and a new original evaluation), 1.87M reference against 1.9M port, or a SIGTERM at 1.87M and `checkpoint.pt`, which `--checkpoint` also accepts.
- `scripts/eval.py` writes means only, so **SE stays missing** for length, collisions and acceleration, as in section 1. `Evaluator.last_per_world` has the per-world values but nothing saves them. Makespan also stays derived.

Command, for each of g8a_s1, s2, s3, ref0, on a GPU not being timed:

```
cd REPO && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ../venv_mjlab/bin/python -m mjlab_ballet.scripts.eval \
  --checkpoint ../runs/g8a_s1/policies/policy_01800000.pt --arm-model mesh_tool \
  --training-dist data/start_states/4r_10t_30o.pb \
  --static-paper mjlab_ballet/data/scenes/4r_10t_30o_static_paper.pt \
  --worlds 8192 --static-worlds 2000 --seed 1 --json OUT.json
```

### 4.4 How strong a one-seed reference is

Only seed 1 of the reference exists (`DESIGN.md` 14.4). Another seed costs 64 CPU cores for about 15 h. Spread of the port at the same windows as `compare_reference` [M; mean, SD, range and last column D from those values]:

| Metric | Step | Reference | g8a_s1 | g8a_s2 | g8a_s3 | g8a_ref0 | Mean, 3 seeds | SD | Range | (Reference - mean) / SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task rate, training | 200k | 0.600 | 0.665 | 0.595 | 0.695 | 0.638 | 0.651 | 0.051 | 0.100 | -1.0 |
| Task rate, training | 400k | 0.898 | 0.910 | 0.895 | 0.904 | 0.905 | 0.903 | 0.008 | 0.015 | -0.7 |
| Task rate, tasksets | 200k | 0.520 | 0.544 | 0.486 | 0.564 | 0.516 | 0.531 | 0.040 | 0.078 | -0.3 |
| Task rate, tasksets | 400k | 0.857 | 0.879 | 0.840 | 0.847 | 0.884 | 0.855 | 0.021 | 0.040 | +0.1 |
| Episode length, training | 400k | 99.9 | 98.1 | 101.6 | 99.4 | 99.7 | 99.7 | 1.8 | 3.5 | +0.1 |
| Collisions, training | 200k | 20.3 | 26.9 | 26.9 | 35.3 | 25.0 | 29.7 | 4.9 | 8.4 | -1.9 |
| Collisions, training | 400k | 22.8 | 21.4 | 25.1 | 24.4 | 19.9 | 23.6 | 2.0 | 3.8 | -0.4 |
| Acceleration, training | 200k | 39.6 | 56.1 | 56.1 | 57.5 | 55.7 | 56.6 | 0.8 | 1.4 | **-20.8** |
| Acceleration, training | 400k | 36.9 | 39.1 | 40.8 | 39.1 | 38.4 | 39.7 | 1.0 | 1.8 | **-2.7** |

The -20.8 is inflated. It divides by an SD of 0.8 from three single evaluations that happen to agree. Over the 200-250k band: reference 37.86, seeds 52.22, 56.92, 53.66, SD 2.41, **-6.8** [D, `V/s_logs.py`]. Report the band, not -20.8. The gap is still far outside the spread.

Seed spread is real, not evaluation noise: at 200k the task-rate SD 0.051 is 10 times the 0.005 SE of one evaluation. It shrinks quickly: 0.051 at 200k, 0.009 at 300k, 0.008 at 400k. `g8a_ref0` shares the reference's start weights and is not closer to it than the nearest other seed: 0.638 against 0.600 at 200k, where `g8a_s2` has 0.595.

Comparison with Fig. 3A's 5 lines [F, D]: SD 0.031 at 200k and 0.021 at 400k, the same order as the port's. They are the only multi-seed view of the original algorithm, on other hardware.

What this supports and what it does not:

- **Supported:** "the reference falls inside the port's seed range" on task rate at 200k (only just: 0.600 against a minimum of 0.595) and 400k.
- **Not supported:** "statistically the same". 1 seed against 3 has a 95% prediction interval of ±4.97 SD [D: t(2) x sqrt(1 + 1/3)], which is ±0.25 at 200k and ±0.04 at 400k. A real gap smaller than that passes unnoticed. Three seeds also pin the SD itself poorly.
- **Reference values at 200k carry their own sampling noise**, ±0.013 from 100 episodes, on top of the unknown seed noise.
- **What gets stronger with no new training:** the same-checkpoint numbers, since their SEs are small and do not depend on seeds. **What only more original seeds can fix:** whether section 2.6 is a seed effect.

### 4.5 README use

| Use now | Wait for the end of G8 | Do not use |
|---|---|---|
| Section 1 tables, 2000-world static column, and the 400k port-trained table | Final row of 4.2, all seeds | Port 4000-world column until there is a raw file |
| Task-rate table and milestones at 50, 75, 90% for g8a seeds, labelled "so far" | 95, 97, 98% and hours | `eval/static_dist` row of `compare_reference` |
| "Hours from launch" equal-step table of 2.4, with the early rows in | Total-time ratio | 1.13 to 1.20x, and hours since first log |
| Fig. 3A band, labelled as read from the figure | Time to 95% | Rows at or below 100k, tasksets 90% as a lead, the 450-500k gaps either way, the -20.8 |
| Section 2.6 acceleration table, if the curve goes in the README it must show this tag or say it is omitted | | g8b_s1 in any "same as the original" table |

## Corrections

Checked: about 600 numbers against raw JSON, CSV, TensorBoard, evaluator logs or the PDF, with scripts in `V`. Changed:

| # | Where | Was | Now | Why |
|---|---|---|---|---|
| 1 | 0, 2.4 | 90% in 2.09 to 2.22 h vs 2.50 h, 1.13 to 1.20x | 2.16 to 2.26 h vs 2.51 h, **1.11 to 1.16x** | The clocks differed. Start at first log leaves out 2.8 to 4.1 min on the port. Stop at the evaluator's line adds 2.8 min on the reference and 3 s on the port. On the reference these two nearly cancel, so only the port's numbers move |
| 2 | 0, 2.4 | Hours table: 100k 0.51 vs 0.49 to 0.50, "no real gap at 100k or 200k" | 100k 0.500 vs 0.551 to 0.560, 200k 1.031 vs 1.053 to 1.141. **The reference leads until about 300k** | The table took the first evaluation at or after the step. The reference's "100,000" is the evaluation at 106,240, written 2.6 min later. Direction reversed |
| 3 | 0, 1 | Same weights tested at 1.87M only | Also g8a_s1's policy at 400k, 2400 vs 16384 episodes, largest difference 1.0 SE | `XP` finished at 13:17, after the snapshot. It strengthens the claim, and is the only such check on a port-trained policy and on the current code |
| 4 | 2.4, 4.1 | "Last 50k of every port run: 48 to 50", 48 used as a lower bound | Load dip 13:05-13:15, all five runs together, recovered after 13:20. Undisturbed 55 to 57 | Measured under load, presented as a rate |
| 5 | 2.4 | Launch to first log 2.2 to 3.5 min, "started within 30 s" | 2.79 to 4.06 min (g8a), 2.57 (g8b). Same second, 10:30:39 | The collector counted from the log's first line, after about 35 s of loading |
| 6 | 1 | SE across process means against "0.00070 used" | 0.00074 | Raw value. Table's 0.0007 is the rounding |
| 7 | 1 | Taskset collisions -1.4 SE, "conservative" | -1.8 with within-set SEs, added beside it | Between-set variance is not sampling error in a balanced design. Largest of 10 stays 1.8 |
| 8 | 2.4 | Tasksets 90% as measured crossings, no caveat | Marked first passage. g8a_s2 back to 0.894 at 580k (13:36) | First-passage on an SE-0.01 series |
| 9 | 2.3 | 0.207 with "±10k" | 0.207 is ±25k. ±10k gives 0.213 | Window mislabelled |
| 10 | 2.3 | g8a_s2 at 100k 0.084, reference tasksets at 1.5M 0.981 | 0.083, 0.980 | Raw 0.08350 and 0.98046. Dividing a rounded print by 10 |
| 11 | 0 | Estimate "in Section 4.4" | Section 4.1 | Wrong cross-reference |
| 12 | 4.4 | Fig. 3A SD at 400k 0.020 | 0.021 | Re-read from the vector paths, `V/fig_mine.py` |
| 13 | 4.4 | -20.8 unqualified | -6.8 on the 200-250k band, beside it | SD from three single evaluations |
| 14 | 1 | Status "54 evals, 9.35 ± 0.88" row | 9.35 ± 0.89 at 1.88M | Rounding level |
| 15 | 2.2 | `paper_baseline` 4.9 cited as agreeing | It says 0.44 mm, these JSONs 0.031 mm | Raw wins. No effect on the verdict |
| 16 | 4.1 | "2.9 to 3.1 h, 1.13 to 1.2x" estimate | 2.98 to 3.02 h at 577,680, 1.15 to 1.16x, measured (13:36) | The runs reached that step. Still not 95% |

Moved because runs went on, not errors: g8a_s1's 500k cells 0.938 to 0.940 and 0.898 to 0.908, latest-band tables, section 0's "Port keeps up" row.

Reproduced with no change: all of section 1's means, SEs, pooling and Diff/SE, per-taskset values, F tests, makespan formula. Scene comparison in 2.2, on a copy of `W/scene_cmp.py`. Code citations sampled in 2.2 and 4.3. All other cells of 2.3. All steps of 2.4. Reference rolling-5 crossings. All of 2.5. All bands of 2.6. Paper pages 11, 12, 31, 41, 45, and Fig. 3A paths from a fresh `pdftocairo` export. The reference evaluator's fixed `PRNGKey(i)` does not pin scenes: scenes come from unseeded `random.choice` (`agents/actor.py:185`), and episode index has no effect, F = 1.02 on 7,080 late episodes [M].

Unverified, and marked so above: the 4000-world static column and the inversion effect 7.99, 18.47; seed 3 of the 8192-world run; the orbax-to-npz conversion of the 1.87M weights; the diff of `train/evaluator.py` and `agent/networks.py` since G4; the reference's 64-core pinning and what else ran on the machine; the cause of the 13:05 dip; the GPU index per run (from `speed.md`, not re-read).

## Weakest points

1. **One reference seed.** "Same learning" can only be "inside the port's range", and at 200k that range only just holds it (0.600 against a minimum of 0.595). 1 against 3 gives a 95% prediction interval of ±0.25 at 200k. A sceptic can say the reference has not been characterised at all, and only more original seeds answer that, at 64 cores for 15 h each.
2. **The acceleration gap, 150k to 400k.** All 4 port runs, both scene sets, 14 units at its widest, cause unknown. It is the one place the curves visibly disagree, on a tag the README's own table might omit. Omitting it would be the first thing a reader finds. No crossplay exists at 200k to 300k, though both sides' policies are on disk.
3. **Same weights does not prove same training.** 1.87M is the reference's policy on the port's simulator. It tests simulator and evaluator, not trainer. 400k is the only port-trained policy tested. The end-of-run comparison does not exist yet, and every row of 10.5 is pending. No port run has reached 95%, and at 580k it trails the reference's 0.95 by 0.004 to 0.007 (13:36).
4. **"Same" at 1.87M is "not separated".** All-done is -1.0 ± 0.6 points, 1.8 to 2.0 SE depending on which runs are pooled, on 1920 original episodes. A 2-point deficit is not excluded. Six of the ten port SEs are not measurements of the port: they borrow the original's SD. `scripts/eval.py` does not save per-world values.
5. **The speed gain is small and depends on when you look.** 0.89x at 100k, 1.11 to 1.16x at 90%, 1.15 to 1.16x at 578k. DESIGN expected 1.4 to 1.9x. A reader who sees only a late row is misled. The reference's rate falls as training goes on and is actor-limited (`speed.md`), so the ratio grows with steps, and the port's late rate is not measured. A total-time claim is still an extrapolation.
6. **Timing conditions are not clean or symmetric.** Port runs shared the machine with this workflow's jobs (13:05-13:17 dip), and g8a_s1 lost about 460 s to an unexplained slow period. The reference's machine load was never logged and its core pinning comes from a script edited afterwards. The port's losses are small and go against it, but nobody can say the same of the reference.
7. **Milestones are first passages on noisy curves.** Reference windows hold 300 to 400 episodes, port windows 3072, port resolution 20k. At 90% the seeds sit 36k before to 5k after the reference, one or two evaluations. Tasksets 90% has already reversed for one seed.
8. **PASS means "not badly off".** 10.4's limits were set wide on purpose: 550k for a reference value of 436k. A README must not present PASS as "matches".
9. **G4 predates three edited files.** `train/evaluator.py`, `agent/networks.py`, `config.py` changed after the 1.87M runs, and the diff was not read here. A rerun on current code, with `--json` and per-world values, would close this and the 4000-world gap together.
10. **Scenes are the same except box rotations on 4 to 6 boxes**, a deliberate fix of the original's inversion. Its measured effect (7.99, 18.47) is only in a status document. Someone comparing static results to the original's code will find this.
11. **The local reference runs above the paper's own band** at 200k, 300k and from 500k to 1M. So "matches the original" means this run on this machine, not the published curve. Fig. 3A's scenes and smoothing are not stated, and only 5 lines read from a vector plot back it.
