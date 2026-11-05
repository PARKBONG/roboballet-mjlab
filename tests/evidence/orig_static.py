"""The unmodified original static evaluation at the reference checkpoint, on CPU. One process, one fresh decomposition.

Claim checked: the same weights score the same on both simulators, on the paper's 10 fixed
tasksets. This runs the original's `actor.run_episode` on `evaluator.get_predefined_targets(4, 10)`
with the workpiece decomposed into 30 boxes, PASSES times over the 10 sets (set i of pass p uses
PRNGKey(i + 10 * p)), so its rows can be set beside
`python -m roboballet_mjlab.scripts.eval --static-paper src/roboballet_mjlab/data/scenes/4r_10t_30o_static_paper.pt`
on the same weights. It also writes the 30 boxes of the decomposition this process made.
WEIGHTS is a flat .npz with `policy/...` keys and `step`: the reference
checkpoint dumped by tests/golden/agent_orbax_dump.py, or a port policy written by pt_to_flat.py.

Old venv (JAX), from $ROBOBALLET_PROJECT, the directory that holds the original as roboballet/:
  PYTHONPATH=$PWD venv/bin/python path/to/orig_static.py WEIGHTS.npz PASSES OUT.json
"""
import json, os, sys, time
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
from roboballet.agents import actor, evaluator, networks
from roboballet.launch import hparams
from roboballet.environments import planning_env

weights, passes, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
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
step = int(z['step'])

configs = hparams.make_optimised_configs(num_targets=10, num_obstacles=30, num_robots=4, pregen=True)
for field, value in (('model_config.edge_update_num_layers', 6), ('env_config.return_to_start_reward_weight', 0.0), ('env_config.return_to_start_required', False)):
  configs = hparams.update_configs(configs, field, value)
configs = hparams.update_configs_for_eval(configs)
net = networks.RoboBalletPolicyNet(config=configs.model_config)
net.config = net.config.update_for_cpu_inference()
params = networks.update_params_for_cpu_inference(nested)
actor_config = configs.actor_config.replace(start_states_path=None)
target_sets = evaluator.get_predefined_targets(4, 10)
rows = []
start = time.time()
for p in range(passes):
  for i, target_set in enumerate(target_sets):
    env_config = configs.env_config.replace(fixed_targets=target_set, num_obstacles_to_generate=0, num_targets_to_generate=0,
                                            obstacles_by_decomposition={'workpiece': 30})
    _, stats = actor.run_episode(
      actor_config=actor_config, planning_env_config=env_config, feature_config=net.config.feature_config, network_apply_fn=net.apply,
      rng_key=jax.random.PRNGKey(i + 10 * p), use_random_actions=False, stop_event=None, network_variables_fetch_fn=lambda: params)
    rows.append(dict(set=i, done=int(stats.num_targets_done), length=int(stats.episode_length), collisions=float(stats.total_collision_robot_steps),
                     acceleration=float(stats.total_mean_acceleration), score=float(stats.final_score)))
boxes = planning_env.decompose_obstacles(env_config.base_config.world_file_path, 'workpiece', 30)
json.dump(dict(step=step, rows=rows, seconds=time.time() - start, spans=[np.asarray(b.spans).tolist() for b in boxes],
               centroid=[np.asarray(b.centroid).tolist() for b in boxes], rotation=[np.asarray(b.rotation).tolist() for b in boxes]), open(out, 'w'))
