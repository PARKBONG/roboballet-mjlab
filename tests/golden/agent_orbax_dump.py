"""Dumps policy and twin critic of an original orbax checkpoint to a flat .npz. Old venv only.

From $ROBOBALLET_PROJECT, which holds roboballet/ and venv/, with a GPU visible because the checkpoint's sharding names cuda:0:
  CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  LD_LIBRARY_PATH="$(venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib" \
  venv/bin/python roboballet-mjlab/tests/golden/agent_orbax_dump.py CKPT_DIR OUT.npz
Then, from the repository root in the release environment, write the torch file:
  PYTHONPATH=src python -m roboballet_mjlab.agent.networks OUT.npz OUT.pt
"""

import os
import sys

import jax
import numpy as np
from orbax import checkpoint


def flatten(tree) -> dict[str, np.ndarray]:
  leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
  return {'/'.join(str(getattr(p, 'key', p)) for p in path): np.asarray(leaf) for path, leaf in leaves}


def restore(path: str) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
  tree = checkpoint.PyTreeCheckpointer().restore(os.path.abspath(path))
  state = tree['train_state']
  return int(tree['step']), flatten(state['policy_params']), flatten(state['twin_critic_params'])


if __name__ == '__main__':
  step, policy, critic = restore(sys.argv[1])
  np.savez(sys.argv[2], step=step, **{'policy/' + k: v for k, v in policy.items()},
           **{'critic/' + k: v for k, v in critic.items()})
