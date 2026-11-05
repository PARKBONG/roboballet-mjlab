"""The unmodified original training_dist evaluation at the reference checkpoint, on CPU, one process.

Claim checked: the same weights score the same on both simulators. This runs the original's own
`actor.run_episode` with the original configs (4 robots, 10 targets, 30 obstacles, 6 edge layers,
no return to start) on scenes drawn from the training distribution, so its rows can be set beside
`python -m roboballet_mjlab.scripts.eval --training-dist ...` on the same weights. WEIGHTS is a flat
.npz with `policy/...` keys: the reference checkpoint dumped by tests/golden/agent_orbax_dump.py, or
a policy the port trained, written by pt_to_flat.py. Episode i uses PRNGKey(i + 1000 * SEED).

Old venv (JAX), from $ROBOBALLET_PROJECT, the directory that holds the original as roboballet/:
  PYTHONPATH=$PWD venv/bin/python path/to/orig_train.py WEIGHTS.npz EPISODES SEED OUT.json
run.sh starts many of these in parallel and pools their rows.
"""
import json, os, sys, time
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
from roboballet.agents import actor, networks
from roboballet.launch import hparams

weights, episodes, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
z = np.load(weights)
nested = {}
for key in z.files:
  if not key.startswith('policy/'):
    continue
  node = nested
  parts = key.split('/')[1:]
  for p in parts[:-1]:
    node = node.setdefault(p, {})
  node[parts[-1]] = jax.numpy.asarray(z[key])
configs = hparams.make_optimised_configs(num_targets=10, num_obstacles=30, num_robots=4, pregen=True)
for field, value in (('model_config.edge_update_num_layers', 6), ('env_config.return_to_start_reward_weight', 0.0), ('env_config.return_to_start_required', False)):
  configs = hparams.update_configs(configs, field, value)
configs = hparams.update_configs_for_eval(configs)
net = networks.RoboBalletPolicyNet(config=configs.model_config)
net.config = net.config.update_for_cpu_inference()
params = networks.update_params_for_cpu_inference(nested)
rows = []
start = time.time()
for i in range(episodes):
  _, stats = actor.run_episode(
    actor_config=configs.actor_config, planning_env_config=configs.env_config, feature_config=net.config.feature_config, network_apply_fn=net.apply,
    rng_key=jax.random.PRNGKey(i + 1000 * seed), use_random_actions=False, stop_event=None, network_variables_fetch_fn=lambda: params)
  rows.append(dict(done=int(stats.num_targets_done), length=int(stats.episode_length), collisions=float(stats.total_collision_robot_steps),
                   acceleration=float(stats.total_mean_acceleration), score=float(stats.final_score)))
json.dump(dict(rows=rows, seconds=time.time() - start), open(out, 'w'))
