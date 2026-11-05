# How each component of the port was checked against the original (verified copy of parity.md)

> Copied into this repository on 2026-11-06 from the development tree, where the package was named `mjlab_ballet` (here `src/roboballet_mjlab`). Paths in this file refer to that tree. Run directories, scratch directories and `docs/reviews/xval_*` are not part of this repository. The text is unchanged.

Verified 2026-11-05 13:22 to 13:40 KST against raw JSON, logs, CSV and TensorBoard. G8 runs are live, so their numbers move. At 13:37:44 the last step in `metrics.csv` and the last evaluation were g8a_s1 582k / 580k, g8a_s2 602k / 600k, g8a_s3 604k / 600k, g8a_ref0 595k / 580k, g8b_s1 604k / 600k, of 1.87M. Window tables were read at 13:25:52, unless a row says 13:37. Everything else in `parity.md` that is not listed under "Corrections" was reopened at its source and stands, or is marked [U].

**Marks.** [M] measured. [D: from X] derived by me or a report from X. [F] read from a paper figure. [U] unverified: no raw source found, only a status document or review says so. "limit" is the acceptance limit of the gate, not a result. "Test limit" means a pytest assertion passed, not a measured maximum.

**Paths.** `MB`=roboballet/mjlab_ballet, `RV`=MB/docs/reviews, `XS`/`XA`/`XT`/`XC`=RV/xval_{simenv,agent,trainer,scenes}, `TL`=runs/mjlab_port_analysis/test_logs, `G4`=runs/mjlab_port_analysis/trainer_g4, `XP`=G4/original_checks/crossplay_g8a_s1_400k, `DS`=MB/data/scenes, `GD`=MB/tests/golden, `RUNS`=multirobot-project/runs, `T`=MB/tests, `EV`=scratchpad/evidence. `DESIGN.md`=MB/docs/DESIGN.md.

**"The original"** is commit 2b2f40a with uncommitted W&B logging and an `--override` flag (RUNS/paper_repro/paper_4r10t_s1/provenance.txt:3-10; I read every non-logging hunk of the diff: the override flag, W&B wiring, the logger class swap, `setuptools<81`, nothing else). The reference run also sets `edge_update_num_layers=6`, `return_to_start_reward_weight=0.0`, `return_to_start_required=False` (provenance.txt:2). Goldens record no commit inside them. Tying them to this code is [D: original sources dated 11-03 at or before the reference start, goldens 11-04 and 11-05; hashes of `agents/actor.py`, `agents/networks.py`, `train/train.py` equal XT/final_source_versions.json and XT/bf16_source_versions.json, rechecked].

**"Four G8a runs" are three seeds plus one re-initialisation.** g8a_ref0 has seed 1 like g8a_s1 and differs only in `init_weights` (RUNS/g8a_ref0/provenance/*/config.json). The median of 10.4 is s1, s2, s3.

## What this supports today

| Claim for the README | State | Why |
|---|---|---|
| Env steps the same as the original (mesh arm, deterministic rollback) | Supported, over 72 original episodes | Row 2. Bit-equal q and qd only under forced rollback choice. Random rollback matches in distribution only |
| Reference weights perform the same in the port | Supported, within noise | Row 9. A true difference of 0.1 point is not excluded |
| A policy trained by the port, scored by the unmodified original evaluator | **Agrees, at one point only:** g8a_s1 at 400k | 0.9117 ± 0.0022 original vs 0.9104 and 0.9106 ± 0.0012 port. Row 8 |
| Same learning curve at 4r/10t | **Close, not the same, to about 600k.** One reference seed | `training_dist` task rate and length mostly inside the port's seed spread. Differences in both directions, section "Contradictions": **evaluation acceleration +20 to +42% at 200k to 300k**, paper tasksets above on all four at 450k to 525k, 0.4 to 4.6 points, `training_dist` up to 1.6 points below at 500k to 550k, `training_dist` collisions above at 450k to 580k, losses offset. SPI has no reference tag, so it cannot "agree" |
| Reaches 95% as early as the reference | **Not so far** | Reference 577,680. No port run has crossed by its last evaluation; trailing 50k means s1 0.9404 at 580k, s2 0.9476 at 600k, s3 0.9484 at 600k, ref0 0.9456 at 580k [D, read 13:38]. Not a failure: limit 750k. Same steps on the reference: 0.9547 ± 0.0027 at 577,680, 0.9517 ± 0.0038 at 602,240 |
| Same final performance from a policy trained by the port | **Not measured** | No run past 604k. Checks at 0.95, 0.97, 0.98 and 600k to 1M are all pending |
| Sphere arm performs the same | **Not measured, and not comparable on the paper tasksets** | `static_paper` ceiling 0.890 on the stick (RV/scenes_status.md:61), 11 of 100 targets without a witness [M DS/test_pools.log] |
| 8r/40t | **Not trained** | G10 not started |
| FastTD3 matches the original's performance | **Nothing supports it.** A different task so far, not yet a final negative | 0.0000 at all 21 evaluations, all 4 sets, default config [M RUNS/fasttd3_dev/g11_smoke_default/metrics.csv]. That run is **30 minutes, 20,236 grad steps, sphere arm and regenerated pool** (g11_smoke_default.log:3), not the reference's task. Best of any FastTD3 config on the real task 0.0001 |

## Component table

| # | Component | Compared against | Number that shows the result, with source | Evidence kind | Reproduces | Raw evidence |
|---|---|---|---|---|---|---|
| 1 | Simulator kinematics and collision (G1) | C MuJoCo, independently compiled, same scene. Original env's TCP on `mesh_tool` | TCP position max 4.77e-7 m, rotation element max 5.09e-7, 18,240 world-steps, **mesh arm** [M XS/differential_*.json:max_errors.tip_*]. Test limits 1e-4 m, 1e-3 rad (T/test_sim_world.py:323-324), its own value not logged. **Sphere arm, not mesh:** labels 0 of 2,048 differ, 256 eight-arm scenes [M XS/contact_results.json:heterogeneous_sizes], limit 1e-4 (test_sim_world.py:508). 128 excludes, 36,036 of 36,036 cross-robot pairs kept [M same:production_eight_arm_asset]. Guard passes at 7,940, raises at 7,939 [M same:physical_overflow_boundaries]. XS/contact_checks.py builds `WorldCfg` without `arm_model`, whose default is `spheres_stick` (MB/config.py:36). On the mesh arm the matching evidence is the per-world `rollback_mask`, row 2. **Exception: gap exactly 0 gives C 1 contact, Warp 0** [M same:exact_tangency] | Golden and independent review (Codex) | T/test_sim_world.py:286,333,453,487,513,588; T/test_sim_adversarial.py:632 | TL/sim_env_0237.log (195 passed); XS/contact_results.json; XS/asset_results.json |
| 2 | Step and reward semantics (G2) | Unmodified original `PlanningEnv`, real 30-box contacts, stored start states, deterministic rollback. Goldens from S/scripts_baseline/out/golden_*.pkl | q and qd max error **0.0**, 0 mismatched world-steps on all 13 fields, 18,240 world-steps, 72 terminal episodes, 63,118 rollbacks [M XS/differential_{4r,8r,enriched_4r,enriched_8r}.json; 72 D: sum of `new_terminals`]. Reward max 2.38e-8, penalty max 2.13e-9 [M same:max_errors]. **Targets solved: 0 in the 4r and 8r runs, 48 + 48 in the enriched runs, whose targets are placed at the home TCP** [M same:new_targets_solved, original.enriched_home_tcp_targets]. Random rollback: 16,384 trials per side at 4r and 8r, min p 0.145 uniformity, 0.235 original vs port, outcome TV at most 0.0186 [M XS/random_summary.json], two-candidate cases only. **Port only, sphere arm, no original on the other side:** E4 10,002,432 world-steps, 0 residual [M XS/long_rollout_results.json]; E5 reward identity error 0.0 over 608 episodes [M XS/reward_contacts_results.json]. Both scripts use the default `WorldCfg` | Golden, statistical, independent review | T/test_env_golden.py:35,65,95; T/test_env_rollout.py:27,76,244; T/test_env_semantics.py; T/test_env_adversarial.py | TL/sim_env_0237.log; XS/ as named; GD/env_*.npz |
| 3 | Observation features (G3a, E6) | Original `compute_features` under JAX on CPU: goldens, then fresh rollouts of the reference checkpoint | E6, 1,000 observations: nodes 1.79e-7, edges 3.13e-7, globals 1.02e-7 [M TL/agent_0848.log:3], limit 1e-4. Fresh 3,616 observations, all 49 columns: 1.79e-7, 3.28e-7, 5.96e-8 [M XA/feature_results.json:errors.*.max_abs]. 8 topologies to 8r40t: at most 3.58e-7 [M TL/agent_0848.log:20-27] | Golden and independent review | T/test_agent_features.py:41,47; T/test_agent_adversarial.py:66 | TL/agent_0848.log; GD/agent_e6_*; XA/feature_results.json |
| 4 | Network (G3a, E7) | Original flax net with checkpoint 1,870,000. JAX CPU f32, JAX GPU bf16 | f32 at `highest`: action max 1.88e-4 compiled, 2.06e-4 eager, mean 9.6e-7, Q1 max 3.3e-7 [M TL/agent_0848.log:5-6], limit 2e-3. This is the acting and evaluation mode. **Training's learner is bf16** (DESIGN.md:295; `bf16: true` in RUNS/g8a_*/provenance). Parameters 8,949,575 and 17,980,164 [M XT/bf16_comparison_results.json; T/test_agent_networks.py:36-37; RV/codex_agent_review.md:147]. **bf16 is not bit-identical:** action mean error 7.15e-3 (4r10t), 7.97e-3 (8r40t), against the original's own CPU-to-GPU 9.34e-3 and 1.08e-2 [M TL/agent_0848.log:64,66,81,83]. That error is about the size of the bf16 effect itself, 9.25e-3 port and 9.56e-3 JAX GPU at 4r10t, so the rounding noise matches in size, not in draw. Largest single action change from bf16: 1.24 port, 1.52 JAX GPU [M :64]. GELU bit-exact on all 65,536 bf16 inputs, eager [M XT/custom_gelu_results.json]; DESIGN.md:299's "33,282" [U] | Golden and independent review | T/test_agent_networks.py:34,107; T/test_agent_bf16.py:51,93,105,147; T/test_agent_adversarial.py:82,211 | TL/agent_0848.log; GD/agent_e7.npz; GD/agent_adv_bf16.npz |
| 5 | TD3 update (G3a, E8) | Original `train.py` update under JAX, same weights, batch and normals. Optax Adam. Fresh original reruns as repeat control | f32, 20 steps, eager: relative loss at most 1.46e-5 (compiled 1.22e-5), test limit 1e-4, policy and critic parameter RMS at step 20 2.08e-8, 1.26e-8 [M TL/agent_0848.log:7,9,10]. Other checks reach 3.15e-5 (:45), magnified hyperparameters 1.6e-4 (:54). E8: 1.1e-6, 1.5e-6 [M :13], limit 1%. Adam, 500 steps: parameters max 4.47e-8, LR gap 9.76e-12, second moment max 5.66e-6 [M XA/adam_results.json]. 500 compiled updates on real transitions: f32 policy RMS 3.48e-4 vs repeat 3.67e-4, ratio 0.95. **bf16 policy RMS 5.98e-4 vs repeat 4.58e-4, ratio 1.31; target policy 7.18e-6 vs 4.74e-6, ratio 1.51**; critic 4.47e-5 vs 4.22e-5, ratio 1.06 [M XT/bf16_comparison_results.json]. One control pair. After those 500 updates actions already differ, port vs original, mean 0.067 at 4r10t, repeat 0.070, f32 0.035 vs 0.042, single actions up to 2.0 on both [M same:*_final_f32_functional]. So "same" here means "same size as the original's own repeat" | Golden and independent review | T/test_agent_td3.py:78,97,208; T/test_agent_adversarial.py:130,144 | TL/agent_0848.log:7-13; GD/agent_td3.npz; XT/bf16_comparison_results.json |
| 6 | Hindsight and replay (G3b) | Unmodified `generate_hindsight_trajectory` and `_write_episode`. A live Reverb table | 234 real and 222 hindsight episodes in 11 goldens [D: recount of `length`, `h_length > 0` in GD/replay_her_*.npz]: exact on discrete fields, reward and score within 1e-6, action within 2e-6 (test limits, T/test_hindsight.py:39-40, maxima not logged). Trained policy: 22 outputs, 3,534 actions, action max 5.96e-8, reward 3.73e-8, q, qd, score, statistics 0.0 [M XA/her_compare_*.json:max_error]. Live Reverb: 30,480 transitions in 10 snapshots, all exact, min uniformity p 0.152 and 0.075 [M XA/replay_live.json:checks]. Mutants 59 of 59 [U RV/replay_status.md:118, no log retained] | Golden and independent review | T/test_hindsight.py:183,187,214,266; T/test_replay.py; T/test_replay_adversarial.py:220,484; GD/replay_mutation_check.py | TL/replay_0816.log (203 passed); GD/replay_her_*.npz; XA/ as named |
| 7 | Scene generation (G5). Used by G8b only: G8a keeps the stored scenes | All 50,000 stored records per file, KS on marginals, pool built on the original tool. C MuJoCo recheck | Worst KS 0.0106 of 17, 4r10t, **in-sample: the IK budget was chosen on it** [M DS/test_e10.log:10]. 0.0098 of 23, 8r40t, out of sample [M DS/test_e10.log:36]. Limit 0.02. Independently 0.010632, 0.009820, on pre-fix pools [M RV/codex_scenes_review.md:373-374]. Witnesses 2,750,000 in training and heldout pools, 0 collisions [M DS/test_pools.log, sum of 4 lines]. E11 random recall 1.0, false positives 0.0646 per robot [M DS/test_e11.log:8162]. Against the original tool, not gated: recall 0.9877, deep 0.9926 [M :8164]. **E11 policy half not run. Target inside a nonowner box 0.3946% vs 0.2304% stored, +71%; 8r 0.4284% vs 0.2393%, +79%; not gated** [M XC/joint_4r_10t_30o.json; RV/codex_scenes_review.md:190]. Sphere pool `target_z` KS 0.0218, above 0.02, information only [M DS/ks_table.md:12] | Statistical and independent review | T/test_scenes_e10.py:47; T/test_scenes_e11.py:164,204; T/test_scenes_pools.py; T/test_scenes_recheck.py | DS/test_all.log (255 passed, sum of 8 suites); DS/test_e10.log; DS/test_e11.log; XC/joint_*.json |
| 8 | Evaluator | Unmodified `actor.run_episode`, JAX CPU, reference checkpoint replaying recorded actions. **Port-trained policy, both evaluators** | 32 controlled episodes: done, length, collisions, penalty error 0.0, score 3.58e-8, acceleration 6.14e-6 [M XT/eval_port_reference_controlled.json:recorded_action_max_errors]. **Closed loop on the same 32 scenes and weights, episodes diverge:** max error 1 target, 31 steps, 50 collision steps, 7.42 acceleration; mean +0.03, -1.75, -1.53, -0.48 [M same:original_closed_loop_*]. So "same" holds in distribution, not per episode. **Port-trained policy, g8a_s1 at 400k, final code:** original 2,400 episodes (480 + 1,920) task rate 0.9117 ± 0.0022, length 98.06 ± 0.56, collisions 22.28 ± 0.81, acceleration 40.15 ± 0.27, all done 0.475 ± 0.010; the 1,920 alone 0.9105 ± 0.0024, 40.24 ± 0.31 [D: recomputed from XP/p*.json, XP/large/p*.json]. Port, 8,192 worlds, two seeds: 0.9104 and 0.9106 ± 0.0012, 98.86 and 98.23, 22.56 and 22.21, 40.55 and 40.32, 0.463 and 0.479 [M XP/large/port_8192_s{2,3}.json]. Same step-0 weights, final code: acceleration 0.8454 vs 0.8488 `training_dist`, 0.5279 vs 0.5306, 0.5319, 0.5279 paper tasksets, three evaluations at step 0 [M RUNS/g8a_ref0/metrics.csv and reference TensorBoard]. A near-still policy, so a weak check. Early policy, v1 code, 48,600 steps, 32 paired episodes: 61.21 port vs 60.69 original [M XT/eval_{port,original}_g8a_s1_controlled.json]. Twin trainers bit-identical after one evaluation, 61 further rounds, 108 updates in all, before the 10:26 trainer edit [M XT/evaluator_isolation_compiled.json]. **Unresolved, not RNG-matched, 64 selected scenes: collision steps 6.44 vs 4.20, length 66.69 vs 64.44, task rate 0.9875 vs 0.9906** [M XT/eval_{port,original}_reference_default.json] | Golden and independent review. Crossplay has no pytest | T/test_evaluator.py:69,96,154,262,376 | TL/trainer_0848.log (pre-dates the last evaluator edit); XT/eval_*.json; XP/ |
| 9 | End to end, no training (G4) | Reference checkpoint through the port's env, features, net and evaluator, against the unmodified original evaluator on the same checkpoint. **Same pool of 50,000 stored start states, independent random draws** (`random.choice`, agents/actor.py:185), not paired scenes | `training_dist` **0.9922 ± 0.0003**, 8,192 worlds [M G4/g4_mesh_8192.json:task_rate] vs **0.9931 ± 0.0007**, 1,920 episodes [D: recomputed from G4/original_checks/train_results/p*.json]. Difference -0.0009 ± 0.0008. Limit 0.97. Reference's own log, 1.80M to 1.87M: 0.9935 ± 0.0008, 12 evals of 50 [D: TensorBoard]. All done 0.9304 vs 0.9417, -0.011 ± 0.006, -1.9 sigma [D: same]. Length 61.38 vs 60.57 ± 0.46, collisions 4.04 vs 4.07 ± 0.29, acceleration 16.58 vs 16.45 ± 0.11. Paper tasksets 0.9880 ± 0.0008, 2,000 worlds [M G4/g4_mesh_large.json] vs 0.9873 ± 0.0007, 2,560 episodes [D: G4/original_checks/static_results/p*.json]. All 32 processes reuse the same 80 RNG keys, yet no two rows are equal; SE by file 0.0006 and by row within set 0.0006, so ± 0.0007 stands. Collision steps 7.37 vs 7.97 ± 0.30; port SE not in the JSON, about 0.34 [D: original's spread over 2,000], so -0.60 ± 0.45. Length 67.82 vs 68.15 ± 0.51, acceleration 18.42 vs 18.44 ± 0.11 | End to end, statistical. Codex: 0.9918 ± 0.0013, 512 worlds [M XT/eval_port_reference_native.json]. Port repeats 0.9919 ± 0.0005, 4,096; 0.9940 ± 0.0008, 1,024 [M G4/g4_mesh_large.json, g4_mesh.json] | No pytest gates 0.97. `scripts/eval.py` (RV/trainer_status.md:58). T/test_evaluator.py:358 asserts only that the table prints | G4/g4_mesh*.json; G4/original_checks/; TL/g4_lead_0816.log (0.9940, 1,024 worlds) |
| 10 | Whole trainer (G6, E9, G8a) | Reference run, seed 1, tag by tag at equal grad step. A literal reading of `actor.py`. A live Reverb limiter | paper_baseline 10.4 on the median of 3 seeds: **8 of 8 decided checks pass, 12 pending** [D: compare_reference on metrics.csv and TensorBoard, EV/parity_compare_reference.txt at 13:10; per run rechecked at 13:35]. Limits rest on Fig. 3A seed spread [F, as quoted by MB/docs/reports/paper_baseline.md:732, figure not reopened] and sit 30 to 43% later than the reference in steps. **g8b_s1, another arm and other scenes, passes the same 8 of 8, so these checks do not separate tasks.** 90% at 400k, 440k, 420k, ref0 420k vs 435,520 [D: same], port granularity 20k. On paper tasksets 500k, 480k, 480k, 520k vs 565,080 [D: same rule]. SPI from step 2k: 4.95 to 5.00 [M tag `rate/samples_per_insert`], limit 5.0 ± 0.1, **port only: the reference logs no such tag**. Age from 100k to the same steps: 39,251 to 40,075 vs 39,154 to 39,656 [M tag `train/trajectory_network_age`, read 13:25]. Window means above the reference by 414 to 431 at 100k, 124 to 152 at 500k. Limit 39k ± 10%. LR at 400k equal to 7 digits [M tags `train/policy_lr`, `train/twin_critic_lr`]. Latches: 128,000 world-events equal, on code before the last fix [M XT/latches_stagger.json]. **Evaluation acceleration differs, see "Contradictions"** | Statistical and independent review | T/test_trainer.py; GD/trainer_mutation_check.py | RUNS/g8a_*/metrics.csv; RUNS/paper_repro/paper_4r10t_s1/tensorboard_log; XT/data_comparison.json; XT/rate_*.json |

## Contradictions and open differences in the data

**Evaluation acceleration, 200k to 300k: +20 to +42% on all four G8a runs, both evaluation sets, where task rate is close.** On `training_dist` all four stay above through 400k, by a smaller margin. It straddles the reference from 450k. At 150k s1 is below the reference on the tasksets. The reference falls fast, then rises from 36 at 300k to 41 at 400k on the tasksets. The port falls steadily. [D: mean over evals within ±25k of the step, tag `eval/{training_dist,static_paper}/total_mean_acceleration`; reference `eval/static_dist`; ± is the SE across its evals. Reference n = 6 or 7 on `training_dist`, 28 to 32 on tasksets; port n = 2 or 3. Read 13:25:52, 550k at 13:37:44.]

| Step | Set | Reference | s1 | s2 | s3 | ref0 |
|---:|---|---:|---:|---:|---:|---:|
| 150k | `training_dist` | 53.13 ± 2.55 | 61.02 | 63.96 | 65.51 | 58.70 |
| 150k | paper tasksets | 62.55 ± 0.92 | 61.58 | 65.34 | 67.87 | 63.96 |
| 200k | `training_dist` | 41.03 ± 2.53 | 55.31 | 55.78 | 58.27 | 53.36 |
| 200k | paper tasksets | 44.60 ± 1.12 | 57.10 | 55.47 | 60.33 | 55.65 |
| 250k | `training_dist` | 37.23 ± 1.05 | 47.13 | 52.25 | 50.53 | 48.08 |
| 250k | paper tasksets | 38.36 ± 0.73 | 48.23 | 49.94 | 54.37 | 51.11 |
| 300k | `training_dist` | 35.02 ± 1.29 | 43.86 | 44.24 | 45.89 | 42.14 |
| 300k | paper tasksets | 36.17 ± 0.69 | 46.04 | 44.88 | 48.96 | 43.69 |
| 350k | `training_dist` | 38.84 ± 1.14 | 39.46 | 41.30 | 40.50 | 39.95 |
| 350k | paper tasksets | 40.67 ± 0.71 | 41.57 | 48.37 | 42.33 | 41.57 |
| 400k | `training_dist` | 36.94 ± 0.65 | 39.11 | 40.83 | 39.07 | 38.37 |
| 400k | paper tasksets | 40.91 ± 0.63 | 41.32 | 45.21 | 41.56 | 40.06 |
| 450k | `training_dist` | 35.33 ± 0.97 | 32.11 | 38.89 | 36.25 | 35.60 |
| 450k | paper tasksets | 38.17 ± 0.47 | 32.83 | 40.94 | 38.94 | 40.91 |
| 500k | `training_dist` | 35.09 ± 1.14 | 33.46 | 35.18 | 33.16 | 36.31 |
| 500k | paper tasksets | 37.76 ± 0.52 | 39.54 | 37.75 | 34.04 | 39.85 |
| 550k | `training_dist` | 31.95 ± 0.24 | 30.82 | 33.07 | 30.88 | 33.62 |
| 550k | paper tasksets | 35.38 ± 0.46 | 33.71 | 36.63 | 34.80 | 38.96 |

Cause not established. What is excluded and what is not:

| Explanation | Evidence | State |
|---|---|---|
| Different initial weights | ref0 starts from the reference's step-0 weights and sits with the port | Excluded |
| Port's evaluator at active policies | Original and port evaluators agree on g8a_s1 at 400k, 40.15 ± 0.27 vs 40.55, 40.32 (row 8) | Excluded at 400k. **No crossplay at 200k or 300k**, where the gap is. `RUNS/g8a_*/policies/policy_00{2,3}00000.pt` and reference `model_checkpoints/{200000,300000}` both exist. Not run here, it needs the JAX venv |
| The reference's one seed | Port spread across 4 runs is SD 2.02 at 200k, 1.54 at 300k on `training_dist`; the gap is 14.65 and 9.01, so 7.3 and 5.9 port SDs [D] | Not excluded, reference seed spread unknown, but it would need a spread several times the port's |
| Exploration-free only | `act/total_mean_acceleration` is only +0.5 to +2.6% at 200k to 300k, same sign on all four [D: tag, ±25k]. Noise dominates that tag, so it says little about the mean policy | Open |

No gate covers this tag, so "8 of 8 pass" does not see it.

**Task rate, equal windows: close, not equal, and not in one direction.** [D: tag `eval/*/num_targets_done` / 10, ±25k, reference ± SE across evals, read 13:25:52, 550k at 13:37:44.]

| Step | Set | Reference | s1 | s2 | s3 | ref0 |
|---:|---|---:|---:|---:|---:|---:|
| 200k | `training_dist` | 0.602 ± 0.015 | 0.664 | 0.599 | 0.704 | 0.642 |
| 300k | `training_dist` | 0.809 ± 0.013 | 0.834 | 0.817 | 0.836 | 0.834 |
| 400k | `training_dist` | 0.898 ± 0.007 | 0.910 | 0.895 | 0.904 | 0.905 |
| 500k | `training_dist` | 0.939 ± 0.004 | 0.940 | 0.933 | 0.923 | 0.928 |
| 550k | `training_dist` | 0.951 ± 0.004 | 0.9395 | 0.9449 | 0.9425 | 0.9435 |
| 200k | paper tasksets | 0.511 ± 0.010 | 0.511 | 0.498 | 0.595 | 0.520 |
| 300k | paper tasksets | 0.750 ± 0.010 | 0.780 | 0.726 | 0.763 | 0.758 |
| 400k | paper tasksets | 0.860 ± 0.008 | 0.879 | 0.840 | 0.847 | 0.884 |
| 450k | paper tasksets | 0.875 ± 0.006 | 0.891 | 0.886 | 0.909 | 0.879 |
| 500k | paper tasksets | 0.874 ± 0.006 | 0.908 | 0.920 | 0.901 | 0.903 |
| 550k | paper tasksets | 0.901 ± 0.004 | 0.9275 | 0.9119 | 0.9019 | 0.9056 |

Each port taskset evaluation is 80 worlds, SE about 0.01 per evaluation [M tag `task_rate_se`], and single evaluations swing more than that: s2 0.9125 at 560k, 0.8588 at 580k. Acting task rate has the most episodes: reference 0.553 ± 0.006 vs 0.612, 0.537, 0.621, 0.584 at 250k, and 0.912 ± 0.001 vs 0.917, 0.913, 0.917, 0.914 at 500k [D: tag `act/num_targets_done`].

| Other difference | Reference | Port | Source | Note |
|---|---:|---:|---|---|
| Hindsight fraction, steps 1k to 16k | 0.4953 ± 0.0006 | 0.4993 to 0.5003 | [D: tag `act/is_hindsight_replay`] | Unexplained. 0.5 is the expected value with no real solves (RV/replay_status.md:117). The port on flush 64 read 0.4954 [M XT/data_comparison.json:data_vs_live], so this tag moves with the hindsight delay. That the reference's value has this cause is not established |
| `training_dist` collision steps, 450k to 580k | 15.57 ± 1.77, 16.17 ± 1.44, 13.78 ± 2.05, 10.40 ± 1.25, 12.75 ± 1.47 at 450k, 500k, 525k, 550k, 580k | 18.60 to 23.24, 16.67 to 18.66, 15.77 to 19.76, 14.65 to 17.29, 14.17 to 14.85 | [D: tag, ±25k] | All four above in every window, up to 5.5 SE. The windows overlap. Not on the other tags: tasksets at 500k 23.25 ± 2.47 vs 15.38 to 23.67; `act/` at 500k 18.95 ± 0.10 vs 18.57 to 19.11. Earlier: within about 2 SE at 200k, 300k, 400k, but 3.8 SE at 250k (s1) and 3.9 SE at 350k (s1). Watch |
| `train/policy_loss`, mean of steps 2k to 4k, seed effect | 0.870 | 1.680, -0.075, 1.436, ref0 0.764 | EV/parity_compare_reference.txt | At exactly 2k: 0.872 vs 1.758, -0.039, 1.531, 0.819 [M tag]. RV/trainer_status.md:183-194 reads it as init seed, its mechanism inferred |
| `train/policy_loss`, 250k to 525k | -2.233 ± 0.014 at 250k, -3.532 ± 0.004 at 500k | s2 within 0.021 throughout. s1, s3, ref0 lower by 0.07 to 0.40; at 500k -3.646, -3.691, -3.617 | [D: tag, ±25k] | **Does not converge by 400k.** Gap narrowing, still open at 525k. g8a_s2 tracks the reference, the other three do not |
| `train/twin_critic_loss`, 200k to 350k | 0.0082, 0.0082, 0.0092, 0.0099 at 200k, 250k, 300k, 350k | 0.0086 to 0.0096, 0.0083 to 0.0100, 0.0097 to 0.0107, 0.0102 to 0.0108 | [D: tag, ±25k] | All four above, +1 to +22%, in the same window as the acceleration gap. Within ±5% from 400k. Link not established |
| Negative-reward share, random phase | 0.1370, 7,680 transitions | 0.1504, 256,000 transitions | [M XT/data_comparison.json] | Fresh small sample. Pre-fix code. Reference tag agrees better: collision steps 45.56 ± 0.41 vs 45.17 to 46.16 [D: tag `act/total_collision_robot_steps`, 1k to 16k] |

## Gates of DESIGN section 9 and later

| Gate | State | Evidence |
|---|---|---|
| G1 sim | Pass by test. **Manual step has no record:** "render and look at it" is only a nonblank-image test (T/test_sim_render.py:1) | TL/sim_env_0237.log |
| G2 env | Pass | TL/sim_env_0237.log; XS/differential_*.json |
| G3a, G3b | Pass | TL/agent_0848.log (95), TL/replay_0816.log (203) |
| G4 | Pass, 0.9922 against 0.97. Not rerun on final code at 1.87M | Row 9 |
| G5 scenes | **Not closed:** E11 policy rollouts and the 10% false-positive limit not run (RV/scenes_status.md:31-32). g8b_s1 policies now exist, so it is runnable and has not been run | DS/test_e11.log |
| G6 smoke | **No dedicated run** (RV/trainer_status.md:255). The live runs meet its terms: no nonfinite value in any tag, residual collisions 0, SPI and age as row 10, hindsight 0.50, critic loss at most 0.0158 from 10k, policy loss -3.29 to -3.75 after 400k, limits 0.02 and -2.5 to -4.5 [D: g8a_* metrics.csv read 13:37; limits MB/docs/reports/paper_baseline.md:756-757] | RUNS/g8a_*/metrics.csv |
| G7 profile | **Not done.** No `docs/PROFILING.md`, no `scripts/bench.py`. T1 to T4 not reported as such. Phase tables exist on old settings (RV/trainer_status.md:141,277) | Listing of MB/docs, MB/scripts |
| G8a | **In progress:** 8 of 20 checks pass per run, 12 pending, 0 fail. Reference has one seed (DESIGN.md:338) | EV/parity_compare_reference.txt; recount 13:35 |
| G8b | **In progress.** No pass limit of its own beyond "compare" (DESIGN.md:336) | RUNS/g8b_s1 |
| G9, G10 | **Not started.** No `scripts/launch_multi.py` | Listing of MB/scripts |
| G11 FastTD3 | **Not passed at default config.** Goldens pass. Small task 0.0000 on 2 seeds at LR 3e-4, 0.989 at LR 1e-4 [M RUNS/fasttd3_dev/variants/{default,default_s2,lr1e4}/metrics.csv, g11_small_s1]. Real task, 30-minute smoke, sphere arm: 0.0000 at 21 evaluations, default; 0.0000 LR 1e-4; 0.0001 at best, 3 layers [M RUNS/fasttd3_dev/g11_smoke_*]. Status test line filled at 13:18: all 91 pass, no retained log seen [U]; 80 of 80 mutants [M RUNS/fasttd3_dev/mutation_final.log] | RV/fasttd3_status.md:5,10,16,209-210 |
| G12 benchmark | **Not run.** No `scripts/benchmark.py`, no `docs/BENCHMARK.md` | Listing of MB/docs, MB/scripts |
| G13 SAC | **No status document at 13:36. Raw runs exist, read 13:36:** small task 0.0000 at 17 evaluations (`g13_small_h1_s1`), 0.957 at best at LR 1e-4; real task smoke 0.0000 at 2 evaluations, LR 1e-4 0.0000 at 3, whose log was still growing | RUNS/sac_dev/g13_small_h1_s1, g13_smoke_default, variants/{small_h1_lr1e4_s1,smoke_lr1e4}/metrics.csv |
| G14 PPO | **Results now exist. Small task fails:** 0.0000 at 15 evaluations; 29.9M env steps per the status. Real task smoke 0.0001 at best, 3 evaluations, 5.48M env steps, shared GPU. Status body written at 13:20, still with placeholders `TBD_TESTS`, `TBD_MUTANTS`, `TBD_NTESTS`, `TBD_FULLRESULT`, `TBD_MUTRESULT`, `TBD_MUTDETAIL` | RUNS/ppo_dev/g14_small_s1, g14_smoke_s1/metrics.csv; RV/ppo_status.md:5,7,9,23,218,219,245 |

E14 is defined against Run-B (paper_baseline.md:703), which did not exist when written and was never compared; g8b_s1 is now the closest. Against Run-A on the same tool, random-phase collision steps are within 1.4% [D: tag `act/total_collision_robot_steps`, 1k to 16k; worst 1.33%, ref0]. g8b_s1 is +3.1%.

## Skipped, not run, or failed

| Item | Source |
|---|---|
| Any policy trained by the port past 604k. Reference noise: only seed 1 exists | Live; DESIGN.md:338 |
| All training-level sameness is 4r/10t. 8r and 40t trainer not run. 500-step learner comparison at 8r not run | RV/trainer_status.md:261; RV/codex_agent_review.md:200 |
| `heldout` not evaluated on any G8a run (flags in provenance config.json). `static_dist` on `mesh_tool` not run | RUNS/g8a_s1/provenance/*/config.json; RV/trainer_status.md:260 |
| Crossplay of port-trained policies at any step but 400k, or any seed but s1. Reference checkpoints below 1.87M through the port's evaluator | Listing of G4/original_checks |
| Pools and E10 for 4r/40t, 8r/10t | RV/scenes_status.md:211-212 |
| Random rollback with 3 to 8 candidates in the conditional test | RV/codex_simenv_review.md:435 onward |
| Full asynchronous actor concurrency, 2M-item live Reverb payloads, late-training age CDFs | RV/codex_trainer_review.md:248-254; RV/codex_agent_review.md:201 |
| bf16 attribution of drift to any one operation. Causal return effect of any difference | RV/codex_trainer_review.md:246-247 |
| Two trainer tests deselected in Codex's run (they call git). W&B against a server | RV/codex_trainer_review.md:185,252 |
| Mutation rerun: 58 of 71 trainer mutants not repeated after test edits | RV/trainer_status.md:218; G4/logs/mutation_rerun.log |
| `.pytest_cache/.../lastfailed` (dated 03:14) lists 7 test IDs: `test_e7_reference_checkpoint` x2, `test_zero_dwell_time...`, `test_fused_step_with_an_empty_budget...`, `test_clamp_keeps_every_candidate...` x3. The agent IDs pass in TL/agent_0848.log. The clamp IDs' only pass is TL/sim_env_0237.log, whose command is not recorded and which cannot be dated against the cache entry. Cleared by rerunning the sim suite when the machine is free | roboballet/.pytest_cache/v/cache/lastfailed |

Codex findings, all fixed or accepted [U as a whole, sampled only]: RV/codex_simenv_review.md:22,66,112,142 (DESIGN 13), RV/codex_agent_review.md:9,62 (DESIGN 14.1; test_agent_bf16.py:147), RV/codex_scenes_review.md findings 1 to 10 (RV/scenes_status.md "Codex findings"), RV/codex_trainer_review.md:9,31,51,63,75,91 (RV/trainer_status.md:266-277).

## Staleness

File dates, plus hashes against those the reviews recorded, recomputed here. No git used.

| Component | Code last changed | Last logged test pass | Last independent review | Verdict |
|---|---|---|---|---|
| Sim, env | `sim/kinematic_sim.py` 02:33, `sim/world.py` 01:32, `env/ballet_env.py` 01:02 | 02:41 | 03:03 | **Current, by file date only:** no review recorded a hash of these files. Differential JSONs are dated 02:48 to 02:56, after the code. `env` and `world` config identical from v1 to g8 [D: diff of provenance config.json] |
| Features, TD3 | 03:17, 05:15. Hashes equal review (XA/versions_features.json; XT/final_source_versions.json) | 09:03 | 05:28, 09:34 | **Current.** XA/adam_results.json is dated 05:13, 2 minutes before `td3.py`; the 09:03 goldens cover the update |
| Network | 10:26. Hash differs from the review's `caf574…`, but **removing `load_policy` gives exactly `caf574…`** [D: sha256, reproduced] | 09:03 | 09:34 | **Current for forward and backward.** `load_policy` itself has no logged run, though XP's port evaluation used it at 13:15 |
| Replay, hindsight | 04:44, 03:35. Hashes equal review (14dab889, 62d01aa0) | 08:23 | 05:28 | **Current.** Default flush later 64 to 8, config only |
| Scenes | 03:01. Pools 03:03 to 03:11 | 03:14 | **02:40, before the fixes** | Implementer's numbers current. Codex's numbers are on pre-fix pools. Status reports tensors bit-identical except witness joints (RV/scenes_status.md "Effect on the pools") [U], not re-reviewed. Pool metadata names a revision without the fixes (:214) |
| Evaluator | 09:40. Differs from reviewed copy only by `task_rate_se` [D: diff with XT/evaluator_se_times_100.py, reproduced] | **09:03, before the edit** | 09:34 | **Stale in form.** New test `test_one_world_has_a_finite_standard_error` has no retained result. Final code has run: XP/large/port_8192_s*.json, 13:15 to 13:17 |
| Trainer | **10:26:** refresh rule changed, snapshots, temp cleanup [D: diff with XT/trainer_reviewed_snapshot.py, reproduced]. Flush 8 in config | **09:03, before the edit** | **09:34, before the edit** | **Stale.** Codex's C1 to C13, the data probe, the scheduling model (140 vs 75) ran on flush 64 and fixed refresh (RV/trainer_status.md:277) [U beyond the status]. Mutants: **no single pass reads 71 of 71**: G4/logs/mutation.log says 69 of 71, mutation_rerun.log 13 of 13 including the two survivors. `test_trainer.py` was compiled by pytest at 10:26, yet no pass is retained. Only the live G8 tags speak for the current code |
| G4 result | Run 06:33 to 06:53, repeat 08:18 | n/a | n/a | **Not rerun on final code at 1.87M.** "Changes since touch bf16 branches and SE only": network side checked via E7 at 09:03; evaluator side [U], no copy of the 06:53 file exists. Final-code evaluator agrees with the original at 400k (row 8) |
| Rate and duration numbers in RV/trainer_status.md:7,141 | Flush 64, fixed refresh | n/a | n/a | **Stale, by its own note** (:277) |

G8 runs on current code: all five at revision 2a32e5c, started 10:30:44 (g8b_s1) and 10:31:13 to 10:31:14, first log 10:33:13 to 10:34:42, only uncommitted diff is `fasttd3/config.py` [M RUNS/g8*/provenance/*/git_revision.txt, git_diff.patch, metrics.csv]. Trainer, config and networks unchanged since 10:26. v1 and v2 run directories are earlier code: do not plot them. `scripts/compare_reference.py`, `plot_curves.py` and `render.py` were edited at 13:05 to 13:08 by someone else; I imported the 13:05 version of `compare_reference`.

## Where a status document and raw data disagree. Raw wins

| Document says | Raw says |
|---|---|
| RV/replay_status.md:170-171: flush 256, `bf16` default False | `hindsight_flush_episodes` 8, `bf16` true (RUNS/g8a_s1/provenance/*/config.json) |
| RV/trainer_status.md:246: bf16 finding "stands, I changed nothing" | Fixed at MB/agent/networks.py:37-65. DESIGN.md:298 is right |
| DESIGN.md:329: reused embeddings, "Forward is bit-identical" | True in bf16 only. f32 max 2.26e-6, not bit-equal [M XT/cache_embed_default_cuda_results.json:cases.f32]. Acting is f32, so this applies to acting |
| RV/trainer_status.md:255: G6 not run | Still no dedicated run, but live data now covers its terms (Gates table) |
| RV/trainer_status.md:117-120: port static at 4,000 worlds 0.9876, 7.95, 18.37 | **No retained output.** G4/scripts/static_check.py has no log. Use the 2,000-world JSON, or rerun |
| RV/trainer_status.md:117-120: reference window "1.80 to 1.87M", 54 evals, collisions 9.35 ± 0.88 | **The label is off by 7k, the number is not.** 54 evals is the window to the last evaluation at 1,877,140: 9.35 ± 0.89. Strictly to 1.87M: 48 evals, 9.19 ± 0.92 [D: TensorBoard `eval/static_dist`] |
| RV/trainer_status.md:216: trainer mutants 71 of 71 | 69 of 71 in G4/logs/mutation.log, then 13 of 13 on rerun. 71 is the union, not a run |
| DESIGN.md:32: reference reaches 95% at 569k, 3.41 h | Same run reads 577,680, 3.46 h under `compare_reference`'s 50k window (EV/parity_compare_reference.txt; DESIGN.md:358; recomputed 3.4628 h). **Use one definition on both sides.** Hours count from the first log: the reference's is 19:34:06, 3 min 20 s after its 19:30:46 launch stamp. The port's omit 2 min 13 s to 3 min 28 s |
| `compare_reference` table `eval/static_dist`: g8b 9.733 vs reference 8.744 at 500k | **Not comparable:** the reference tag is the paper tasksets, g8b's is a regenerated pool (MB/scripts/compare_reference.py:49-51). Compare `eval/static_paper` |
| `compare_reference` row "2,000" of `eval/training_dist/total_mean_acceleration`: 21.7 vs 0.48 to 0.93 | **Not the same step:** its window catches only step 0 on the port, steps 0 and 2,420 on the reference. Do not plot that row |
| Status numbers rechecked against raw, and they agree | 18,240 and 63,118 (sum of XS/differential_*.json), q and qd 0.0, G4 0.9921875, original 0.99307 and 0.98734, goldens 234 and 222, 255 scene tests (sum of DS/test_all.log), KS 0.0106 and 0.0098, witnesses 2,750,000 |

## Differences kept on purpose, with measured effect

| Difference | Effect | Source |
|---|---|---|
| Exact-zero contact | C 1, Warp 0 at gap 0. Never fired in 18,240 differential world-steps | [M XS/contact_results.json]; DESIGN.md:271 |
| One RNG stream | Reset of another world changed first rollback choice in 16 of 32 seeds in both cases tested, and final q in 16 of 32 in case 4. Distribution unbiased in the test above | [M XS/rng_contacts.json]; DESIGN.md:272 |
| Terminal augmentation redraw absent | No bootstrap there, no learning path. RNG sequence differs | RV/codex_simenv_review.md:112; DESIGN.md:273 |
| Torch vs JAX, bf16 | Rows 4, 5. Scale of the original's own repeat variability, from one control pair, except policy 1.31 and target policy 1.51 times that | XT/bf16_comparison_results.json |
| Reused embeddings | bf16 gradient relative L2 0.0049. Original full vs reused 0.0050, a separate one-Dense test | [M XT/cache_embed_default_cuda_results.json; XT/cache_rounding_jax_results.json] |
| Lockstep vs 64 async actors | Not separately measured. SPI, age and early `act/*` tags close | Row 10; RV/codex_trainer_review.md:248 |
| Policy refresh, per `W` finished episodes | **Late effect not measured.** Fixed-period model gave 140 vs 75, pre-fix. A refresh can change weights within an episode. Recorded age reads about 75 high late, by model, not measured [D: DESIGN.md:326]. Measured tag gap to the reference is 124 to 152 at 500k | XT/staleness.json; DESIGN.md:326; tag |
| Limiter at the centre | First learning at 200,520 inserts vs 200,000, 80 updates later [D: 10,240 samples of credit / 128] | [M XT/rate_compare.json; XT/latches_stagger.json]; DESIGN.md:328 |
| Hindsight flush of 8 | At most 960 transitions late [D: 8 × 120], 4,800 samples of catch-up against a 10,240 band | DESIGN.md:327 |
| Stagger discards startup rows | 3,752 rows in a real parity start, no episodes counted. Same on the live runs | [M XT/latches_stagger.json:stagger.discarded; tag `counters/discarded_rows`] |
| Redraw of scenes colliding at start | 9 to 12 draws so far per run. How the original treats these [U] | [M tag `counters/invalid_scene_draws`, g8a_s1, s2, ref0]; MB/env/ballet_env.py:129 |
| Log windows, timing tags | `act/*` one megabatch apart, age about 20 updates. `act/env_sampling_time_s` and `act/episode_write_wall_time_s` are zero. Do not overlay | [M tags; RV/codex_trainer_review.md:75-85]; DESIGN.md:330 |
| Evaluation sampling | Port 1,024 episodes and 80 on tasksets, every 20k. Reference 50 episodes, median gap 6,440, and 10 episodes, median gap 1,480 on tasksets | [M tag cadence, TensorBoard and metrics.csv; launch/launch.py:313; `EvaluatorConfig` in agents/evaluator.py]; DESIGN.md:331 |
| Sphere arm, stick, regenerated scenes (G8b) | Different task. Reference weights on it give 0.725 on `training_dist`, 0.646 on paper tasksets, for information | [M G4/g4_spheres.json] |

## Process note

Collector: at 13:07 one of its commands mistakenly created an empty `/tmp/null` outside the allowed path and hung on stdin. It stopped its own shell and deleted that one file. Absent at 13:36 [M]. Two short JSON reads used system `python3` rather than `venv_mjlab`. `EV/parity_compare_reference.txt` is the unedited output of `compare_reference`.

Verifier: nothing in the repository or run directories changed. Helper scripts and a cache of the reference scalars are in `EV/pv_work/`. Several short JSON and CSV reads used system `python3` rather than `venv_mjlab`, the same slip as above. Repo imports ran with `PYTHONDONTWRITEBYTECODE=1`. No GPU, no tests, no training, no git.

For the speed group, not caused by me: `XP/run.sh` and `run_large.sh` ran 8 then 16 CPU processes at nice 19 from 13:07 to 13:14 and port evaluations on `CUDA_VISIBLE_DEVICES=4` from 13:14 to 13:17, inside the G8 timing window. Whether they shared a GPU with a G8 run was not checked.

## Corrections

Wrong or misleading in `parity.md`:

1. **Age range, different window.** Reference "39,008 to 39,656" spans the whole run to 1.87M against a port range to 534k. Over the same steps it is 39,154 to 39,656. Port is above the reference at every step, 124 to 431.
2. **"SPI agree" has no basis.** The reference logs no SPI tag. It is a port tag against a limit.
3. **"Task rate, length, losses agree" is too strong.** Added a task-rate table. Tasksets 2.7 to 4.6 points above at 500k on all four, `training_dist` up to 1.6 points below at 500k and 0.6 to 1.2 below at 550k. Claim changed to "close".
4. **Policy loss "at 2k" is the mean of 2k, 3k, 4k**, the `compare_reference` window. Exact values added. **"Converges by 150k to 400k" is false:** s1, s3, ref0 stay 0.07 to 0.40 below through 525k.
5. **Critic loss difference at 200k to 350k was absent.** Added, +1 to +22% on all four.
6. **Collision steps at 500k: "17.88 to 20.59" came from truncated windows.** Now 16.67 to 18.66, complete. **"At 200k to 400k within about 2 SE" is false** at 250k and 350k, up to 3.9 SE. Extended through 580k, where all four stay above on this tag.
7. **Acceleration: "All four above on both sets, 150k to 300k" is false at 150k on tasksets** (s1 61.58 vs 62.55). **"Gone by 350k to 500k"** replaced: all four above on `training_dist` through 400k, straddling from 450k. "20 to 40%" is +20 to +42%. Rows for 150k, 250k, 350k, 450k, 550k added. 500k cells moved because windows filled, s1 33.46, ref0 36.31, s1 tasksets 39.54: not errors.
8. **Crossplay omitted.** The original evaluator has scored a port-trained policy at 400k (XP, 13:10 to 13:17, final code). This is the strongest evidence on the evaluator and the only one on a mid-training policy. Added to the claims table, row 8, and the cause table.
9. **95% crossing state omitted.** No port run has crossed where the reference has. Added, read 13:38. Not an error at 13:17, since no run had reached 577,680.
10. **E4 and E5 are sphere arm and port only**, yet sat in a row headed "Unmodified original `PlanningEnv`". **Row 1's labels, excludes and guards are also sphere arm.** Labelled.
11. **Row 2 omitted coverage:** 72 original episodes, and targets solved only on synthetic home-TCP targets.
12. **Row 9 "same start states" is wrong:** same pool, independent random draws. Original evaluator uses `random.choice`.
13. **"71 of 71 mutants (G4/logs/mutation*.log)" is in no log.** Logs read 69 of 71, then 13 of 13.
14. **Reference window "48 evals vs 54":** the status's number is right for the window to 1,877,140. Its label is off by 7k. Softened.
15. **GELU marked [M DESIGN.md]:** the document is not a measurement. Raw is 65,536 of 65,536, XT/custom_gelu_results.json. "33,282" is [U].
16. **Hindsight tolerances marked [M RV/replay_status.md:112]** are test limits, not measured maxima. **Mutants 59 of 59** changed to [U]. **Witnesses** moved from the status to DS/test_pools.log. Parameter citation is RV/codex_agent_review.md:147, not :93.
17. **Row 5 left out the worst ratio:** bf16 target policy 1.51. Added, with the action divergence after 500 updates.
18. **Row 8 left out closed-loop divergence** on the same scenes: up to 31 steps, 50 collision steps, 7.42 acceleration. Added. Step-0 tasksets are three evaluations, 0.5306, 0.5319, 0.5279.
19. **FastTD3 "contradicted" on "4r/10t/30o"** omitted that the run is 30 minutes on the sphere arm and regenerated pool. Reworded.
20. **Gates moved after 13:17, not errors:** `TESTS_RESULT_LINE` gone from fasttd3_status.md at 13:18. `TBD_SUMMARY` gone from ppo_status.md at 13:20, which now reports G14 results, confirmed in raw. SAC raw results exist without a status. The summary's "no G13 or G14 results" no longer holds.
21. **Other added items:** ref0 shares seed 1 with s1. Row 10's tasksets crossing. g8b_s1 passes 8 of 8. Sim and env staleness is by date only. `compare_reference`'s 2,000 row is not one step. Port taskset cadence, 80 worlds. Age model note. Policy loss after 400k now to -3.75, moved with the runs.

Checked and unchanged: all other numbers in rows 1 to 10, the 200k to 400k acceleration cells, step-0 values, cadence, hindsight, SPI, LR, negative-reward share, 8 of 8 and 12 pending, G4 0.9922, 0.9931, 0.9880, 0.9873, all SE, all-done -0.011 ± 0.006, `load_policy` hash, source diffs, E14 1.4%. The ± 0.0007 on the 2,560 static episodes was suspect, since processes reuse RNG keys, and it survived.

## Weakest points

What a sceptical reader can attack, strongest first:

1. **No final result.** Evidence stops at 604k of 1.87M. 12 of 20 checks are pending. "Same performance" cannot be written today beyond "reference weights score the same in the port".
2. **One reference seed.** Every "within noise" is the reference's sampling error plus the port's own seeds. Its seed spread is unknown, and only 3 port seeds exist.
3. **The 10.4 limits are loose and one-sided.** 30 to 43% later in steps than the reference, justified by a paper figure and "tool differs", which does not apply to G8a. A run may be earlier without limit. g8b_s1, a different arm and scenes, passes the same 8. "8 of 8 pass" must not be read as "same".
4. **Acceleration +20 to +42% at 200k to 300k is unexplained**, about 6 to 7 port SDs, and no crossplay exists there. Overlaying that tag exposes it; omitting it would be selective.
5. **Not one direction.** Port is ahead on tasksets, behind on `training_dist` at 500k to 600k, has not crossed 95% where the reference has, and has more `training_dist` collisions from 450k. A README line "same" needs "close, on one reference seed".
6. **"Identical" holds only under controlled conditions.** q and qd 0.0 needs forced rollback. Replay needs recorded actions. Closed loop on the same scenes diverges per episode. 500 updates from equal weights diverge as much as the original's repeat.
7. **Training runs bf16; the tightest numbers are f32.** bf16 is "same size as the original's rounding": policy 1.31 and target policy 1.51 times the repeat, from one control pair, with no attribution.
8. **Small original-side samples.** 72 episodes in the differential; target solving only on synthetic targets, 96 solves. The 10M-step and 608-episode numbers are port-only on the sphere arm. They must not appear as comparisons.
9. **Current trainer has no review, no retained test pass and no late effect measured.** Reviews and mutants predate the 10:26 edit. G4 at 1.87M predates it too. Only live tags and one crossplay speak for final code.
10. **Evaluation is asymmetric.** 50 and 10 episodes per evaluation on the reference, 1,024 and 80 on the port. First-passage crossings favour the noisier side, and port crossings are quantised to 20k.
11. **Everything is 4r/10t, one machine.** No 8r/40t training. `heldout` never run on G8a.
12. **The release's main agent has no result on the real task.** FastTD3, SAC and PPO all read at most 0.0001 on 4r/10t/30o, in runs of about 30 minutes on the sphere arm, so nothing in this file can be cited for them.
