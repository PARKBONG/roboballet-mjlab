# Licences for the release

Checked on 2026-11-05 from the files and upstream pages named below. This is an engineering check, not legal advice.

**Result: nothing found blocks a private or a public release.** Everything that would ship is under Apache 2.0, CC-BY 4.0, MIT or BSD terms, and each of these allows redistribution and modification with notices kept. The obligations are listed below. None had been carried out at the time of the check. The last section says where this repository meets each one.

## What would ship

| Part | Source | Licence | Evidence |
|---|---|---|---|
| Original RoboBallet code and anything derived from it | Zenodo 10.5281/zenodo.16546403, Copyright 2025 Google LLC | Apache 2.0 | `LICENSE`, `README.md:66-85`, 30 source file headers, Zenodo record |
| Start states, tasksets, world XML, tool, table meshes | Same archive | Apache 2.0 on the Zenodo record. `README.md:75` says materials other than software are CC-BY 4.0 | The two do not distinguish data the same way, so satisfy both |
| Panda meshes | MuJoCo Menagerie `franka_emika_panda`, through the RoboBallet archive | Apache 2.0, with a `LICENSE` file upstream | 14 of 14 sampled mesh files are byte-identical to Menagerie `main`. `panda.xml`, `hand.xml` and `panda_nohand.xml` differ, which were changed after copying |
| `roboballet_mjlab/assets/panda_spheres_stick.xml` | Derived from the above, with our spheres and stick | Apache 2.0 | File header |
| FastTD3 update rule, verbatim excerpts in `test` | github.com/younggyoseo/FastTD3 at 229ed59, the current upstream head | MIT, Copyright 2025 Younggyo Seo. Its `LICENSE` also bundles notices of LeanRL, TD3, SAC and others | `LICENSE` of the clone |
| SAC update rule | github.com/leggedrobotics/rsl_rl_sac at e0d243a | BSD-3-Clause, Copyright 2026 ETH Zurich and NVIDIA | `LICENSE`, SPDX headers, `licenses/dependencies/` |

## Dependencies, installed and not copied into the repository

mjlab, mujoco, mujoco-warp, warp-lang, tensorboard, mediapy: Apache 2.0. torch: Apache 2.0 and BSD terms. numpy, scipy, protobuf, rsl-rl-lib: BSD. wandb, tyro: MIT. matplotlib: PSF-based. tqdm: MPL-2.0 and MIT, which binds only files that are copied or changed. Read from each package's metadata in the development environment.

## What the release must do

1. Keep the Apache 2.0 `LICENSE` and Google's copyright lines. Mark modified files as changed, as Apache 2.0 section 4(b) requires, and add a `NOTICE` saying this is a modified port.
2. Credit the original under CC-BY: authors, title, the paper's citation and the DOI, with a statement of what changed.
3. Keep "This is not an official Google product", and say the port is unofficial. Licences give no trademark rights, so the names RoboBallet, Franka and Panda are used only to describe what this is.
4. Add `third_party/` with Menagerie's `LICENSE`, FastTD3's whole `LICENSE` and rsl_rl_sac's `LICENSE` and `licenses/dependencies/`. RoboBallet's archive has no licence file for the Panda files.
5. Keep the SPDX headers on anything based on rsl_rl_sac, and cite its paper, arXiv 2605.24975.

## Data

The four `.pb` start-state files are 321 to 524 MB, over GitHub's 100 MB file limit. Recommended: do not copy them. A script fetches the archive from the Zenodo DOI, 1.1 GB, which also keeps its provenance clear. Regenerated scene pools and our own trained policies go in release assets.

## Where this repository meets each item

Added on 2026-11-06, when the release tree was built. `sac/` is in `src/roboballet_mjlab/`. `tests/` is at the repository root, on branch `test` only.

| Item | Where | State |
|---|---|---|
| 1. Apache 2.0 and Google's copyright | `LICENSE`, a byte-identical copy of the original's, whose appendix keeps "Copyright 2025 Google LLC". `NOTICE` says this is a modified port and names which parts of the package port which files of the original | The source files carry no per-file "changed" line. `NOTICE` names the changed parts instead |
| 2. CC-BY credit | `NOTICE`, section "Credit": authors, title, paper citation, both DOIs, and what changed | Done |
| 3. Not an official Google product, unofficial port | First lines of `NOTICE` | Done |
| 4. `third_party/` | `third_party/mujoco_menagerie/franka_emika_panda/LICENSE`, `third_party/FastTD3/LICENSE`, `third_party/rsl_rl_sac/LICENSE` and `third_party/rsl_rl_sac/licenses/dependencies/` (10 files). Byte-identical to upstream, sources in `NOTICE` | Done |
| 5. SPDX headers and the rsl_rl_sac paper | Lines 1-4 of `sac/agent.py`, `sac/config.py`, `sac/trainer.py`, `sac/train.py`, `tests/test_sac.py` and `tests/golden/sac_make_goldens.py` keep the fork's header with `SPDX-License-Identifier: BSD-3-Clause`. arXiv 2605.24975 is cited in `sac/agent.py:8`, `tests/golden/sac_make_goldens.py:8` and `NOTICE` | Done |
