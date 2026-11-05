# Layout

What each directory and module holds, and what it imports. Taken from DESIGN section 5 and the code as of 2026-11-06. Imports come from the code, not from the design.

## Repository

| Path | Holds |
|---|---|
| `src/roboballet_mjlab/` | The package, below |
| `pyproject.toml` | Package metadata, dependencies, and `train` and `play` commands like mjlab's. mjlab's own `train` cannot run this env, because it builds a `ManagerBasedRlEnv` (`mjlab/scripts/train.py:110`) |
| `tests/` | pytest suites, with goldens in `tests/golden/`. Branch `test` only |
| `data/` | Downloaded and generated data, not in git. See "Data" below |
| `docs/` | `DESIGN.md`, the design and every decision in it. `LICENSES.md`, the licence check. `LAYOUT.md`, this file. `evidence/`, the verified comparisons with the original |
| `LICENSE`, `NOTICE`, `third_party/` | Apache 2.0 with Google's copyright, the credit and list of changes, and third-party licence texts |

Branch `main` holds the port and its four agents. Branch `test` adds `tests/` and the scripts that check sameness with the original.

## Package

Paths are relative to `src/roboballet_mjlab/`. The env is direct-style, with no mjlab managers (DESIGN D1).

| Module | Holds | Imports from the package |
|---|---|---|
| `config.py` | `WorldCfg`, `EnvCfg`, `ModelCfg`, `TrainCfg`. `TrainCfg` defaults are the parity configuration (DESIGN 14) | nothing |
| `structs.py` | Tensors passed between modules: `Scene`, `RobotState`, `Obs`, `StepOut`, `Graph` | nothing |
| `paths.py` | Where files are found at run time: the package directory, and the data directory | nothing |
| `assets/` | `panda_spheres_stick.xml`, the Panda as 36 collision spheres and a stick, read by `sim/world.py` and `scenes/ik.py`. `spheres.json`, the spheres and excludes, read by tests | |
| `data/` | Small files kept in git: the static evaluation pools `scenes/*_static*.pt` and the frozen boxes `static_boxes/` | |
| `sim/world.py` | `build_world` → `World`: an mjlab scene of all worlds, per-world boxes, kinematics and collision, a contact overflow guard | `config`, `structs`, `paths`, `env/poses`, `sim/kinematic_sim` |
| `sim/kinematic_sim.py` | `KinematicSim`, an mjlab `Simulation` that runs only `mjw.kinematics` and `mjw.collision` | nothing |
| `env/ballet_env.py` | `BalletEnv`: reset from a scene pool, observe, step with rollback, target scoring | `config`, `structs`, `sim/world` |
| `env/poses.py` | Obstacle augmentation, rotation helpers | nothing |
| `scenes/sampler.py` | Batched scene generation: rails, boxes, targets | `config`, `structs`, `sim/world`, `scenes/ik` |
| `scenes/ik.py` | Batched damped least squares, witness search | `config`, `structs`, `sim/world` |
| `scenes/io.py` | `.pt` pools, and import of the original's `.pb` start states and `.pbtxt` tasksets | `structs`, `scenes/sampler` |
| `agent/features.py` | `build_graph`, from observation to graph | `config`, `structs` |
| `agent/networks.py` | `PolicyNet`, `CriticNet`, the original's bf16 rounding, checkpoint loading | `config`, `structs`, `agent/features` |
| `agent/td3.py` | The original's TD3 update, schedules, target updates | `config`, `structs`, `agent/features`, `agent/networks` |
| `replay/replay.py` | Compact ring buffer on the GPU. Whole episodes go in, and the oldest transition goes out first | `config`, `structs` |
| `replay/hindsight.py` | Batched hindsight relabelling | `config`, `structs`, `replay/replay`, `env/ballet_env` |
| `train/trainer.py` | The td3 agent's lockstep loop in one process on one GPU. DESIGN 5 planned several GPUs, and the code uses one | `agent/{features,td3}`, `env`, `replay`, `train/{evaluator,logger,profiler}` |
| `train/evaluator.py` | Batched evaluation, shared by all four agents | `agent/{features,networks,td3}`, `env`, `scenes/io` |
| `train/profiler.py`, `train/logger.py` | Phase timers and CPU and GPU use. TensorBoard, CSV and optional W&B | nothing |
| `fasttd3/` | The main agent: `config`, `agent` (twin C51 critic and the update), `trainer`, `train` | `agent/{features,networks}`, `env`, `replay`, `train/{evaluator,logger,profiler}` |
| `sac/` | rsl_rl_sac's SAC update: `config`, `agent`, `trainer`, `train` | same as `fasttd3/` |
| `ppo/` | rsl-rl's PPO as installed: `config`, `models` (GNN actor and critic), `vecenv` (adapter), `train`. No replay and no hindsight | `agent/{features,networks}`, `env`, `train/{evaluator,logger,profiler}` |
| `scripts/` | `train` (td3 agent), `eval`, `gen_scenes`, `render`, `plot_curves`, `compare_reference` | what each one runs |

The four agents share `env/`, `replay/` (all except PPO), `agent/features.py`, the networks and `train/evaluator.py`. No agent imports another agent's package. `fasttd3/`, `sac/` and `ppo/` do not import `train/trainer.py`. They reach `agent/td3.py` only through `train/evaluator.py`, which takes `matmul_precision` from it (DESIGN 12). `fasttd3/` and `sac/` average gradients across GPUs themselves.

Two scripts use `tests/` from the clone, which only branch `test` has. `scripts/gen_scenes.py` imports `tests/test_scenes_common.py` for the C MuJoCo recheck when it is there. `scripts/render.py` reads `tests/golden/env_scenes_{R}r10t.npz`.

## Outside imports

| Package | Used by |
|---|---|
| mjlab, mujoco, mujoco-warp, warp-lang | `sim/`. mujoco is also used by `scenes/ik.py`, and mjlab's viewer by `scripts/render.py` |
| torch | Env, agents, replay, training, scenes |
| rsl-rl-lib, tensordict | `ppo/` |
| protobuf | `scenes/io.py` |
| psutil, wandb | `train/logger.py`. wandb only with `--wandb-project` |
| tensorboard, matplotlib, mediapy, Pillow | `scripts/compare_reference.py`, `scripts/plot_curves.py`, `scripts/render.py` |

## Data

`paths.py` sets the data directory: `$ROBOBALLET_DATA` if that is set, otherwise `data/` beside `src/`. Files kept in git are in the package's own `data/`.

| Path in the data directory | Read by | Comes from |
|---|---|---|
| `mujoco_world/` | `sim/world.py`: mesh arm, table, and the meshes the sphere arm shows | RoboBallet archive, Zenodo 10.5281/zenodo.16546403 |
| `paper_test_tasksets/` | `scripts/gen_scenes.py` | RoboBallet archive |
| `start_states/` | `scenes/io.py`, when a command is given the original's `.pb` start states | RoboBallet archive |
| `scenes/` | Scene pools. `scripts/gen_scenes.py` writes here, and `paths.scene_pool` prefers a pool here over the package's copy | This repository's release assets, or `scripts/gen_scenes.py` |
