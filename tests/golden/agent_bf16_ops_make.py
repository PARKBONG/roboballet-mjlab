"""Golden for tests/test_agent_bf16.py: the original's bf16 arithmetic on GPU. Run from the repo root:

  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. LD_LIBRARY_PATH=<python 3.10 lib> \
    ../venv/bin/python -B tests/golden/agent_bf16_ops_make.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from roboballet.agents import networks

assert jax.default_backend() == 'gpu'
rng = np.random.default_rng(11)

# Every bf16 value of magnitude up to 8: well past what a LayerNorm output reaches.
codes = np.arange(2 ** 16, dtype=np.uint16)
values = jnp.asarray(codes).astype(jnp.uint16)
every = np.asarray(jax.lax.bitcast_convert_type(values, jnp.bfloat16), dtype=np.float32)
every = every[np.isfinite(every) & (np.abs(every) <= 8.0)]
gelu_in = jnp.asarray(every, dtype=jnp.bfloat16)
saved = {'gelu_in': every, 'gelu_out': np.asarray(jax.jit(jax.nn.gelu)(gelu_in), dtype=np.float32)}

x = rng.normal(size=(256, 128)).astype(np.float32)
cfg = networks.ModelConfig(use_bf16=True, edge_update_num_layers=6)
model = networks.MLP((64, 64, 7), cfg, last_layer_mlp_scale=0.005)
params = model.init(jax.random.PRNGKey(73), jnp.asarray(x))
# Fresh biases are zero, which hides how a bias enters: trained ones are not.
params = {'params': {scope: {leaf: (value + jnp.asarray(rng.normal(size=value.shape) * 0.3, value.dtype) if leaf in ('bias', 'scale') else value)
                             for leaf, value in leaves.items()} for scope, leaves in params['params'].items()}}
saved['mlp_in'] = x
saved['mlp_out'] = np.asarray(jax.jit(model.apply)(params, jnp.asarray(x)))
weights = rng.normal(size=(7,)).astype(np.float32)
saved['loss_weights'] = weights


def loss(p):
  return (model.apply(p, jnp.asarray(x)) * jnp.asarray(weights)).sum(-1).mean()


grads = jax.jit(jax.grad(loss))(params)
for scope, leaves in params['params'].items():
  for leaf, value in leaves.items():
    saved[f'param/{scope}/{leaf}'] = np.asarray(value, dtype=np.float32)
    saved[f'grad/{scope}/{leaf}'] = np.asarray(grads['params'][scope][leaf], dtype=np.float32)
np.savez_compressed(Path(__file__).parent / 'agent_bf16_ops.npz', **saved)
print({k: v.shape for k, v in saved.items()})
