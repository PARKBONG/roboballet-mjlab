# roboballet-mjlab

RoboBallet (multi-arm task and motion planning with a graph network and TD3) on mjlab and MuJoCo Warp, with three more agents: FastTD3, rsl-rl SAC and rsl-rl PPO. Unofficial port, see [NOTICE](NOTICE).

Results and figures: being finalized. They will go here as a few short tables and graphs, with the detail in `docs/`.

## Install

```bash
uv sync
uv run ballet-fetch-data          # the original's start states and meshes from Zenodo, and the scene pools
```

## Run

```bash
uv run ballet-train RoboBallet-4R10T-Mesh --agent fasttd3
uv run ballet-eval  RoboBallet-4R10T-Mesh --checkpoint logs/.../checkpoint.pt
uv run ballet-play  RoboBallet-4R10T-Mesh --checkpoint logs/.../policies/policy_00100000.pt --video demo.mp4
```

Agents: `td3` (the original's, reproduced), `fasttd3` (default), `sac`, `ppo`. Tasks: `ballet-train --help`.

The commands carry a `ballet-` prefix because mjlab installs its own `train` and `play` into the same environment, and those cannot run this direct-style env.

## Layout

[docs/LAYOUT.md](docs/LAYOUT.md). The `test` branch adds the tests and scripts that check sameness with the original.

## Licence

Apache 2.0, as the original. Credits and third-party notices: [NOTICE](NOTICE), [third_party/](third_party/).
