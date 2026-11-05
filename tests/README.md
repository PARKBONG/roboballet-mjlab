# Tests: sameness with the original

These are the port's tests from development, with the package renamed to `roboballet_mjlab`. They
compare the port with the original RoboBallet code: stored goldens made by the original's JAX code,
live runs of the original in its own environment, and the C MuJoCo engine as an independent
reference. `golden/` holds the goldens with the scripts that made them, and the mutation checks.
`evidence/` holds the scripts behind the G8 cross-evaluation numbers.

## Where the tests look

| What | Where | Set by |
|---|---|---|
| The package | `src/roboballet_mjlab` | `PYTHONPATH=src`, or an editable install |
| The original's data: `mujoco_world/`, `start_states/`, `paper_test_tasksets/` | the data directory | `ROBOBALLET_DATA`, else `data/` in this clone, filled by `fetch-data zenodo` |
| Large scene pools (`*_train.pt`, `*_heldout.pt`, `*_train_mesh_tool.pt`) | `scenes/` in the data directory | `fetch-data pools`, `fetch-data test-pools` |
| Small scene pools, frozen static boxes | `src/roboballet_mjlab/data/` | in git |
| The original's source tree, its venv, the reference artefacts | `$ROBOBALLET_PROJECT/roboballet`, `$ROBOBALLET_PROJECT/venv`, `$ROBOBALLET_PROJECT/runs/mjlab_port_analysis` | `ROBOBALLET_PROJECT`, else the parent of this clone |

The old venv is the original's own environment: Python 3.10 with JAX, flax, jraph, orbax, Reverb
and TensorFlow, as the original's requirements give them. The reference artefacts are
`ckpt/paper_4r10t_s1_1870000.pt` (the original's checkpoint of the reference run at step 1,870,000,
converted by `golden/agent_orbax_dump.py`), `scripts_baseline/out/golden_nocollision_her_4r10t.pkl`
and `algo/tb.json` (the reference run's TensorBoard tags). None of them is in this repository.

## Running

From the repository root, because the tests import each other as `tests.*` and some start workers
with `python -m tests.<file>`. pytest is needed on top of the package's dependencies.

```
PYTHONPATH=src python -m pytest -q tests/test_scenes_*.py
PYTHONPATH=src python -m pytest -q tests/test_sim_*.py tests/test_env_golden.py tests/test_env_rollout.py tests/test_env_semantics.py tests/test_env_adversarial.py
PYTHONPATH=src python -m pytest -q tests/test_agent_*.py
PYTHONPATH=src python -m pytest -q tests/test_trainer.py tests/test_evaluator.py
PYTHONPATH=src python -m pytest -q tests/test_replay.py tests/test_hindsight.py tests/test_replay_adversarial.py -k 'not throughput'
PYTHONPATH=src python -m pytest -q tests/test_fasttd3_agent.py tests/test_fasttd3_trainer.py tests/test_fasttd3_dist.py tests/test_sac.py
PYTHONPATH=src python -m pytest -q tests/test_ppo.py
PYTHONPATH=src python -m pytest -s tests/test_env_throughput.py   # alone, on an idle GPU: prints, asserts nothing else
```

All of them were run on one NVIDIA B200, never on CPU only. Most need a CUDA GPU, since
`test_sim_common.py` fixes `DEVICE = 'cuda:0'`. The `test_agent_*` files pick CPU when no GPU is
visible, `test_scenes_e11` compares on C MuJoCo, and `test_fasttd3_agent` has CPU cases beside its
GPU ones. Whether those pass on CPU has not been tried. Five files build CUDA tensors at import, so
without a GPU even `--collect-only` fails on them: `test_env_adversarial`, `test_env_semantics`,
`test_replay_adversarial`, `test_sim_adversarial`, `test_sim_world`. The other 26 files collect 845
tests on CPU, the same node ids as before the rename.

## What each file needs, and how long it took

Data: Z is `fetch-data zenodo`, which the mesh arm, the C MuJoCo worlds, start states and
tasksets read; the sphere arm alone needs no data file. P is `fetch-data pools`, T is
`fetch-data test-pools`. Old venv means the original's source tree and venv under
`ROBOBALLET_PROJECT`. The column comes from reading what each file loads, not from running them
without the data. Times are from the test logs and status notes of development, on a B200 shared
with other jobs, so the same files took up to 2 to 5 times longer on a busy day. Counts are what the files collect now. A logged count that
differs is from an earlier version of the file.

| Files | Tests | Needs, besides the GPU above | Logged time |
|---|---:|---|---|
| `test_scenes_io` | 42 | Z, old venv (the official decoder), a `.git` in this clone | 5.4 s |
| `test_scenes_ik` | 27 | Z | 6.1 s |
| `test_scenes_sampler` | 15 | | 14.4 s |
| `test_scenes_recheck` | 10 | Z | 9.4 s |
| `test_scenes_gen` | 39 | Z, P | 50.1 s |
| `test_scenes_e10` | 46 | Z, T | 48.8 s |
| `test_scenes_e11` | 8 | Z, P | 20.5 s |
| `test_scenes_pools` | 68 | Z, P | 26.5 s |
| `test_sim_world`, `test_sim_adversarial`, `test_sim_render`, `test_env_golden`, `test_env_rollout`, `test_env_semantics`, `test_env_adversarial` | 11 on CPU, the rest only with a GPU | Z, for the mesh-arm and visual-mesh cases of all but `test_env_rollout`. `test_env_adversarial`: old venv, else it skips; it writes traces to `runs/mjlab_port_analysis/review` under `ROBOBALLET_PROJECT` | 195 passed in 284 s (the log does not name the file set) |
| `test_env_throughput` | 4 | an idle GPU | not logged |
| `test_agent_td3`, `_adversarial`, `_networks`, `_features`, `_bf16` | 95 | reference checkpoint (`_adversarial`, `_networks`) | 95 passed in 914 s; 82 without `_bf16` in 601 to 949 s |
| `test_trainer`, `test_evaluator` | 53 + 13 | P, reference checkpoint. `test_trainer`: `algo/tb.json`, a `.git` in this clone, a GPU whose name contains B200 (the provenance test). `test_evaluator`: Z | 60 passed in 931 s; 59 in 3 min 17 s |
| `test_replay`, `test_hindsight`, `test_replay_adversarial` | 112 + 41 + GPU only | `test_hindsight`: the original's `train/stats.py` and the `.pkl` above. `test_replay_adversarial`: Z, P, old venv, else it fails | 203 passed in 185 s and in 417 s |
| `test_fasttd3_agent`, `test_fasttd3_trainer`, `test_fasttd3_dist`, `test_sac` | 50 + 45 + 7 + 90 | | 190 passed in 980 s, and in 860 s; `test_sac` alone 86 passed in 582 to 589 s |
| `test_ppo` | 69 | | 69 passed in 378 s |

Sources, from development and not in this repository: `runs/mjlab_port_analysis/test_logs/` (`sim_env_0237`, `agent_0318` to `agent_0848`,
`trainer_0848`, `replay_0505`, `replay_0816`), the scene logs written by `data/scenes/test_all.sh`
in the source tree, and `docs/reviews/{trainer,fasttd3,sac,ppo}_status.md` of development.

## golden/

The `.npz` goldens are read by the tests. The scripts that made them are kept so that each golden
can be traced to the code that produced it. None of them runs during a test.

| Script | Runs in | Needs | Writes |
|---|---|---|---|
| `agent_orbax_dump.py CKPT_DIR OUT.npz` | old venv, from `ROBOBALLET_PROJECT`, a GPU visible | the original's orbax checkpoint | flat weights; then `PYTHONPATH=src python -m roboballet_mjlab.agent.networks OUT.npz OUT.pt` gives the reference checkpoint |
| `agent_make_goldens.py CKPT_DIR` | old venv, from `ROBOBALLET_PROJECT`, a GPU visible | original source tree and its `data/start_states` | `agent_e6_*`, `agent_e7`, `agent_td3` |
| `agent_adv_make_goldens.py CKPT_DIR` | old venv, from `ROBOBALLET_PROJECT`, a GPU | the same | `agent_adv_*` |
| `agent_bf16_ops_make.py` | old venv, a GPU | | `agent_bf16_ops.npz` |
| `replay_make_goldens.py [case ...]` | old venv, CPU | original source tree and its `data/start_states` | `replay_her_*` |
| `scenes_make_wire_fixture.py` | old venv | `roboballet/data/start_states/4r_10t_30o.pb` under `ROBOBALLET_PROJECT` | `scenes_wire_variants.binpb` |
| `python -m tests.golden.env_make_goldens` | this package | the original's traces under `runs/mjlab_port_analysis` | `env_*` |
| `fasttd3_make_goldens.py [bf16]` | this package, CPU (`bf16` needs a GPU) | nothing: a verbatim excerpt of FastTD3 at commit 229ed59 | `fasttd3_*` |
| `sac_make_goldens.py` | this package, CPU | nothing: a verbatim excerpt of rsl_rl_sac at commit e0d243a | `sac_*` |
| `python -m tests.golden.fasttd3_small_task pool OUT` / `random OUT` | this package, a GPU | | the 1-robot task of gate G11 and its random baseline |
| `python -m tests.golden.{trainer,replay,fasttd3}_mutation_check` | this package, a GPU, `TMPDIR` private | | injects each listed bug into a copy of the code and exits 0 only if the tests catch every one |

Each script's docstring has its full command. The mutation checks were not timed in the logs read
for this file.

## evidence/

| Script | Claim it checks |
|---|---|
| `orig_train.py WEIGHTS.npz EPISODES SEED OUT.json` | The unmodified original scores given weights on the training distribution, to set beside `roboballet_mjlab.scripts.eval` on the same weights. Old venv |
| `orig_static.py WEIGHTS.npz PASSES OUT.json` | The same on the paper's 10 fixed tasksets. Old venv |
| `pt_to_flat.py POLICY.pt OUT.npz` | A policy the port trained converts exactly to the original's parameter layout, or the script refuses |
| `run.sh` | G8 final: port and original scores of the G8a seeds at 200k, 300k and 1.8M and of the reference checkpoint. Needs `ROBOBALLET_PROJECT`, `RUNS`, `OUT`; see its header |
