"""Goldens of test_agent_adversarial.py from the original code, on hostile inputs. Old venv only.

Run from $ROBOBALLET_PROJECT, which holds roboballet/ and venv/ (see agent_orbax_dump.py for the environment variables):
  venv/bin/python roboballet-mjlab/tests/golden/agent_adv_make_goldens.py CKPT_DIR

agent_adv_{R}r{T}t{O}o.npz  synthetic observations (out of range values, all flags, all aug
                            codes), the original features, reference checkpoint outputs.
agent_adv_td3.npz           original TD3 steps from trained weights on other topologies, and
                            with hyperparameters big enough to expose update order mistakes.
agent_adv_bf16.npz          reference checkpoint with use_bf16=True, on JAX CPU and GPU.
agent_adv_init.npz          statistics of a real flax init.

It also replays the existing agent_* goldens through the original code and prints the
differences, which is the check that they were not derived some other way.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from scipy.spatial.transform import Rotation

import agent_common
import agent_make_goldens as amg
import agent_orbax_dump
from roboballet.agents import compute_features, networks
from roboballet.environments import planning_env
from roboballet.train import train

OUT = os.path.dirname(os.path.abspath(__file__))
CASES = ((1, 1, 1), (2, 3, 5), (3, 7, 2), (4, 10, 30), (4, 40, 30), (5, 1, 40), (8, 10, 30), (8, 40, 30))
NUM_OBS = 12
DWELL_STEPS = 5
TD3_CASES = ((2, 3, 5), (8, 40, 30))
TD3_BATCH = 4
TD3_STEPS = 3
# Update order, count origin and tau mistakes are invisible at the real hyperparameters:
# tau * lr is 4e-9 per step and the schedule moves 5e-7 per step, below float32 rounding.
MAGNIFIED_CASE = (3, 7, 2)
MAGNIFIED_BATCH = 2
MAGNIFIED_STEPS = 6
MAGNIFIED = dict(
  policy_learning_rate=2e-3, policy_learning_rate_decay_rate=0.5, twin_critic_learning_rate=1e-3,
  twin_critic_learning_rate_decay_rate=0.25, target_sigma=0.2, noise_clip=0.4, policy_tau=0.1, critic_tau=0.3, discount=0.8,
)
MAGNIFIED_STEPS_PER_DECAY = 2


def rotations(rng, n):
  return Rotation.random(n, random_state=rng).as_matrix().astype(np.float32)


def hostile_row(rng, row, R, T, O, q_low, q_high, vmax, max_steps):
  """One observation's raw inputs. Row 0 has no target done, row 1 has all of them done."""
  span = q_high - q_low
  done = rng.random(T) < 0.5
  if row < 2:
    done = np.full(T, row == 1)
  return dict(
    q=rng.uniform(q_low - 0.3 * span, q_high + 0.3 * span, (R, 7)).astype(np.float32),
    qd=rng.uniform(-1.5 * vmax, 1.5 * vmax, (R, 7)).astype(np.float32),
    dwell=rng.integers(0, DWELL_STEPS + 1, R).astype(np.int8),
    target_done=done,
    score=np.float32(rng.uniform(-0.2, 1.1)),
    step=np.int32(rng.integers(0, max_steps + 1)),
    aug_code=rng.integers(0, 4, O).astype(np.int8),
    base_pos=rng.uniform(-1.5, 1.5, (R, 3)).astype(np.float32), base_rot=rotations(rng, R),
    world_tip_pos=rng.uniform(-2.5, 2.5, (R, 3)).astype(np.float32), world_tip_rot=rotations(rng, R),
    target_pos=rng.uniform(-1.5, 1.5, (T, 3)).astype(np.float32), target_rot=rotations(rng, T),
    box_pos=rng.uniform(-1.5, 1.5, (O, 3)).astype(np.float32), box_rot=rotations(rng, O),
    box_spans=rng.uniform(0.02, 1.2, (O, 3)).astype(np.float32),
  )


def original_obs(row, R):
  """The dict make_graph_features takes, built with the original box and pose helpers."""
  boxes = [
    planning_env.BoxInfo(spans=s, centroid=c, rotation=r).get_augmentation(int(code))
    for s, c, r, code in zip(row['box_spans'], row['box_pos'], row['box_rot'], row['aug_code'])
  ]
  base = planning_env._make_pose(row['base_pos'], row['base_rot'])
  tip = planning_env._make_pose(row['world_tip_pos'], row['world_tip_rot'])
  return dict(
    step=np.asarray(row['step'], np.float32),
    robots=dict(
      joint_configurations=row['q'], joint_velocities=row['qd'], base_poses=base,
      tip_relative_poses=(np.linalg.inv(base.astype(np.float64)) @ tip.astype(np.float64)).astype(np.float32),
      robot_index=np.eye(R, dtype=np.float32), dwelling=np.asarray(row['dwell'], np.float32) / DWELL_STEPS,
    ),
    obstacles=dict(
      spans=np.asarray([b.spans for b in boxes], np.float32),
      poses=planning_env._make_pose(np.asarray([b.centroid for b in boxes], np.float32), np.asarray([b.rotation for b in boxes], np.float32)),
    ),
    targets=dict(poses=planning_env._make_pose(row['target_pos'], row['target_rot']), done=row['target_done']),
    current_score=row['score'],
  )


def batched(graphs):
  return jraph.batch(graphs)


def forward_fns(ms):
  policy = networks.RoboBalletPolicyNet(config=ms)
  twin = networks.RoboBalletTwinCriticNet(config=ms)

  def forward(policy_params, critic_params, graph, action):
    c1, c2 = twin.apply(critic_params, graph, action)
    return policy.apply(policy_params, graph), c1.q_values, c2.q_values

  return policy, twin, jax.jit(forward)


def nest(flat):
  root = {}
  for key, value in flat.items():
    node = root
    parts = key.split('/')
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = jnp.asarray(value)
  return root


def initial_state(tc, policy_params, critic_params, key=0):
  opt = train._make_optimisers(tc)
  return train.TD3TrainState(
    step=jnp.array(0, dtype=jnp.uint32), policy_params=policy_params, twin_critic_params=critic_params,
    target_policy_params=policy_params, target_twin_critic_params=critic_params,
    policy_opt_state=opt.policy_optimiser.init(policy_params),
    twin_critic_opt_state=opt.twin_critic_optimiser.init(critic_params),
    random_key=jax.random.PRNGKey(key),
  ), opt


def run_td3(label, R, tc, ms, nodes, edges, globals_, topo_graph, policy_params, critic_params, batch, steps, rng):
  """Real _train_step_impl on the rows in order, s' being the next row, normals exported."""
  policy = networks.RoboBalletPolicyNet(config=ms)
  twin = networks.RoboBalletTwinCriticNet(config=ms)
  count = batch * steps
  actions = rng.uniform(-1.0, 1.0, (count, R, 7)).astype(np.float32)
  rewards = rng.uniform(-0.05, 0.2, count).astype(np.float32)
  terminals = (rng.random(count) < 0.25).astype(np.int8)
  index = np.arange(count)
  args = dict(
    nodes=nodes[index], edges=edges[index], globals=globals_[index][:, None],
    next_nodes=nodes[(index + 1) % len(nodes)], next_edges=edges[(index + 1) % len(nodes)],
    next_globals=globals_[(index + 1) % len(nodes)][:, None], actions=actions, rewards=rewards, terminals=terminals,
  )
  args = {k: jnp.asarray(v) for k, v in args.items()}
  single = jax.tree_util.tree_map(np.asarray, topo_graph)
  topo = networks.GraphTopology(
    senders=jraph.batch([single] * batch).senders, receivers=jraph.batch([single] * batch).receivers,
    n_node=jnp.array([nodes.shape[1]] * batch), n_edge=jnp.array([edges.shape[1]] * batch),
  )
  state, opt = initial_state(tc, policy_params, critic_params)
  step_fn = jax.jit(lambda s, b: train._train_step_impl(s, b, topo, policy.apply, twin.apply, opt, tc))
  out, normals, losses = {}, [], []
  for i in range(steps):
    critic_key, _ = jax.random.split(state.random_key, 2)
    normals.append(np.asarray(jax.random.normal(key=critic_key, shape=(batch, R, 7))))
    state, stats = step_fn(state, {k: v[i * batch:(i + 1) * batch] for k, v in args.items()})
    losses.append((float(stats['twin_critic_loss']), float(stats['policy_loss'])))
    if i in (0, steps - 1):  # Keeps the file under the 5 MB limit of DESIGN 10.
      out.update(amg.snapshot(state, f'{label}/step{i + 1}'))
  out.update({f'{label}/normals': np.stack(normals), f'{label}/losses': np.array(losses), f'{label}/actions': actions,
              f'{label}/rewards': rewards, f'{label}/terminals': terminals, f'{label}/batch': batch, f'{label}/steps': steps})
  print(label, 'losses', np.array(losses).tolist(), flush=True)
  return out


def make_cases(policy_params, critic_params):
  rng = np.random.default_rng(20260511)
  td3_out = {}
  for R, T, O in CASES:
    cfg = amg.make_configs(R, T, O)
    # Only the bounds of the spec matter for scaling, and the original env only builds with
    # all the robots of a world, so take them from a full world with this T and O.
    env = planning_env.PlanningEnv(amg.make_configs(4 if R <= 4 else 8, T, O).env_config)
    spec = env.observation_spec()
    q_spec = spec['robots']['joint_configurations']
    q_low, q_high = np.asarray(q_spec.minimum, np.float64), np.asarray(q_spec.maximum, np.float64)
    vmax = float(spec['robots']['joint_velocities'].maximum)
    max_steps = int(cfg.env_config.base_config.episode_timeout_s / cfg.env_config.base_config.timestep_s)
    rows = [hostile_row(rng, i, R, T, O, q_low, q_high, vmax, max_steps) for i in range(NUM_OBS)]
    fc = cfg.model_config.feature_config
    graphs, tips = [], []
    for row in rows:
      obs = original_obs(row, R)
      tip = np.matmul(obs['robots']['base_poses'], obs['robots']['tip_relative_poses'])
      tips.append((tip[:, :3, 3], tip[:, :3, :3]))
      graphs.append(jax.tree_util.tree_map(np.asarray, compute_features.make_graph_features(obs, spec, fc)))
    stack = {k: np.stack([r[k] for r in rows]) for k in rows[0] if not k.startswith('world_tip')}
    stack['tip_pos'] = np.stack([t[0] for t in tips])
    stack['tip_rot'] = np.stack([t[1] for t in tips])
    nodes, edges = np.stack([g.nodes for g in graphs]), np.stack([g.edges for g in graphs])
    globals_ = np.stack([g.globals[0] for g in graphs])
    action = rng.uniform(-1.0, 1.0, (NUM_OBS, R, 7)).astype(np.float32)
    _, _, forward = forward_fns(cfg.model_config)
    a, q1, q2 = forward(policy_params, critic_params, batched(graphs), jnp.asarray(action))
    np.savez_compressed(
      f'{OUT}/agent_adv_{R}r{T}t{O}o.npz', **stack, nodes=nodes, edges=edges, globals=globals_, senders=graphs[0].senders,
      receivers=graphs[0].receivers, max_steps=max_steps, input_action=action, action=np.asarray(a), q1=np.asarray(q1), q2=np.asarray(q2),
    )
    print((R, T, O), 'q1 range', float(q1.min()), float(q1.max()), 'action sat', float((np.abs(a) > 0.99).mean()), flush=True)
    if (R, T, O) in TD3_CASES:
      tc = train.TrainConfig(train_batch_size=TD3_BATCH, fused_training_steps=TD3_STEPS)
      td3_out.update(run_td3(f'{R}r{T}t{O}o', R, tc, cfg.model_config, nodes, edges, globals_, graphs[0], policy_params, critic_params, TD3_BATCH, TD3_STEPS, rng))
    if (R, T, O) == MAGNIFIED_CASE:
      saved = train._EXP_STEPS_PER_DECAY
      train._EXP_STEPS_PER_DECAY = MAGNIFIED_STEPS_PER_DECAY
      tc = train.TrainConfig(train_batch_size=MAGNIFIED_BATCH, fused_training_steps=MAGNIFIED_STEPS, **MAGNIFIED)
      td3_out.update(run_td3('magnified', R, tc, cfg.model_config, nodes, edges, globals_, graphs[0], policy_params, critic_params, MAGNIFIED_BATCH, MAGNIFIED_STEPS, rng))
      train._EXP_STEPS_PER_DECAY = saved
  schedule_counts = np.array([0, 1, 2, 99_999, 100_000, 1_000_000, 1_878_000, 10_000_000])
  policy_lr, critic_lr = train._make_schedule(5e-5, 0.98), train._make_schedule(5e-5, 0.95)
  td3_out.update(schedule_counts=schedule_counts, schedule_policy=np.array([float(policy_lr(int(c))) for c in schedule_counts]),
                 schedule_critic=np.array([float(critic_lr(int(c))) for c in schedule_counts]))
  np.savez_compressed(f'{OUT}/agent_adv_td3.npz', **td3_out)


def make_bf16(policy_params, critic_params):
  """The bf16 the reference learner ran on, on real rollouts, CPU and GPU."""
  out = {}
  gpu = jax.devices('gpu')[0]
  cpu = jax.devices('cpu')[0]
  for name, R, T, shards, stride, chunk in (('4r10t', 4, 10, 2, 5, 100), ('8r40t', 8, 40, 1, 1, 40)):
    raw = np.load(f'{OUT}/agent_e6_{name}.npz')
    parts = [np.load(f'{OUT}/agent_e6_{name}_expected_{i}.npz') for i in range(shards)]
    nodes, edges, globals_ = [np.concatenate([p[k] for p in parts]) for k in ('nodes', 'edges', 'globals')]
    action = raw['action']
    single = jraph.GraphsTuple(
      nodes=nodes[0], edges=edges[0], globals=globals_[:1], senders=raw['senders'], receivers=raw['receivers'],
      n_node=np.array([nodes.shape[1]]), n_edge=np.array([edges.shape[1]]),
    )
    topo_batch = jraph.batch([single] * chunk)
    topo = networks.GraphTopology(senders=topo_batch.senders, receivers=topo_batch.receivers, n_node=topo_batch.n_node, n_edge=topo_batch.n_edge)
    index = np.arange(0, len(nodes), stride)
    index = index[: len(index) // chunk * chunk]
    out[f'{name}/index'] = index
    cfg = amg.make_configs(R, T)
    for label, device, use_bf16 in (('cpu_bf16', cpu, True), ('gpu_bf16', gpu, True), ('gpu_f32', gpu, False), ('cpu_f32', cpu, False)):
      ms = cfg.model_config.replace(use_bf16=use_bf16)
      policy = networks.RoboBalletPolicyNet(config=ms)
      twin = networks.RoboBalletTwinCriticNet(config=ms)
      pp, cp = jax.device_put(policy_params, device), jax.device_put(critic_params, device)

      @jax.jit
      def forward(n, e, g, a):
        graph = networks.reconstruct_graph_from_batched_data(n, e, g, topo)
        c1, c2 = twin.apply(cp, graph, a)
        return policy.apply(pp, graph), c1.q_values, c2.q_values

      results = []
      for i in range(0, len(index), chunk):
        rows = index[i:i + chunk]
        args = [jax.device_put(jnp.asarray(x), device) for x in (nodes[rows], edges[rows], globals_[rows][:, None], action[rows])]
        results.append([np.asarray(x) for x in forward(*args)])
      for j, field in enumerate(('action', 'q1', 'q2')):
        out[f'{name}/{label}/{field}'] = np.concatenate([r[j] for r in results])
      print(name, label, 'done', flush=True)
  np.savez_compressed(f'{OUT}/agent_adv_bf16.npz', **out)


def make_init():
  """Per leaf statistics of the real initialisers, so the torch init can be compared."""
  cfg = amg.make_configs(4, 10)
  graph = jraph.GraphsTuple(
    nodes=jnp.zeros((44, 17)), edges=jnp.zeros((172, 30)), globals=jnp.zeros((1, 2)),
    senders=jnp.zeros(172, jnp.int32), receivers=jnp.zeros(172, jnp.int32), n_node=jnp.array([44]), n_edge=jnp.array([172]),
  )
  policy = networks.RoboBalletPolicyNet(config=cfg.model_config)
  twin = networks.RoboBalletTwinCriticNet(config=cfg.model_config)
  flat = {
    **{'policy/' + k: v for k, v in agent_orbax_dump.flatten(policy.init(jax.random.key(11), graph)).items()},
    **{'critic/' + k: v for k, v in agent_orbax_dump.flatten(twin.init(jax.random.key(12), graph, jnp.zeros((1, 4, 7)))).items()},
  }
  np.savez_compressed(f'{OUT}/agent_adv_init.npz', keys=np.array(sorted(flat)),
                      std=np.array([flat[k].std() for k in sorted(flat)]), mean=np.array([flat[k].mean() for k in sorted(flat)]),
                      peak=np.array([np.abs(flat[k]).max() for k in sorted(flat)]), size=np.array([flat[k].size for k in sorted(flat)]))


def replay_existing_goldens(policy_params, critic_params):
  """Prints how far the stored agent_* goldens are from a fresh run of the original code."""
  cfg = amg.make_configs(4, 10)
  raw = np.load(f'{OUT}/agent_e6_4r10t.npz')
  parts = [np.load(f'{OUT}/agent_e6_4r10t_expected_{i}.npz') for i in range(2)]
  nodes, edges, globals_ = [np.concatenate([p[k] for p in parts]) for k in ('nodes', 'edges', 'globals')]

  # Same start state, same reset.
  env = planning_env.PlanningEnv(cfg.env_config, start_state=amg.start_states('4r_10t_30o', 1)[0])
  obs = env.reset().observation
  spec = env.observation_spec()
  boxes = [b for _, b in env._generated_obstacles]
  print('replay reset: base', np.abs(obs['robots']['base_poses'][:, :3, 3] - raw['base_pos'][0]).max(),
        'target', np.abs(obs['targets']['poses'][:, :3, 3] - raw['target_pos'][0]).max(),
        'box', np.abs(np.array([b.centroid for b in boxes]) - raw['box_pos'][0]).max(), 'q', np.abs(obs['robots']['joint_configurations'] - raw['q'][0]).max())

  # Stored raw rows through the real feature builder give the stored expected features.
  fc = cfg.model_config.feature_config
  worst = {'nodes': 0.0, 'edges': 0.0, 'globals': 0.0}
  for i in range(0, 1000, 7):
    episode = raw['episode'][i]
    row = dict(
      q=raw['q'][i], qd=raw['qd'][i], dwell=raw['dwell'][i], target_done=raw['target_done'][i], score=raw['score'][i], step=raw['step'][i],
      aug_code=raw['aug_code'][i], base_pos=raw['base_pos'][episode], base_rot=raw['base_rot'][episode], world_tip_pos=raw['tip_pos'][i], world_tip_rot=raw['tip_rot'][i],
      target_pos=raw['target_pos'][episode], target_rot=raw['target_rot'][episode], box_pos=raw['box_pos'][episode], box_rot=raw['box_rot'][episode], box_spans=raw['box_spans'][episode],
    )
    graph = compute_features.make_graph_features(original_obs(row, 4), spec, fc)
    worst['nodes'] = max(worst['nodes'], float(np.abs(np.asarray(graph.nodes) - nodes[i]).max()))
    worst['edges'] = max(worst['edges'], float(np.abs(np.asarray(graph.edges) - edges[i]).max()))
    worst['globals'] = max(worst['globals'], float(np.abs(np.asarray(graph.globals)[0] - globals_[i]).max()))
  print('replay E6 raw -> real features, 143 rows, max error', worst)

  # Stored graphs through the real nets give the stored E7 outputs.
  e7 = np.load(f'{OUT}/agent_e7.npz')
  topo = networks.get_graph_topology(cfg.env_config, fc, batch_size=100)
  ms = cfg.model_config
  policy = networks.RoboBalletPolicyNet(config=ms)
  twin = networks.RoboBalletTwinCriticNet(config=ms)
  graph = networks.reconstruct_graph_from_batched_data(jnp.asarray(nodes[300:400]), jnp.asarray(edges[300:400]), jnp.asarray(globals_[300:400])[:, None], topo)
  a = policy.apply(policy_params, graph)
  c1, c2 = twin.apply(critic_params, graph, jnp.asarray(raw['action'][300:400]))
  print('replay E7 100 rows, max error', float(np.abs(np.asarray(a) - e7['action'][300:400]).max()),
        float(np.abs(np.asarray(c1.q_values) - e7['q1'][300:400]).max()), float(np.abs(np.asarray(c2.q_values) - e7['q2'][300:400]).max()))

  # First step of agent_td3.npz.
  golden = np.load(f'{OUT}/agent_td3.npz')
  index = golden['index'][:amg.TD3_BATCH]
  args = dict(nodes=nodes[index], edges=edges[index], globals=globals_[index][:, None], next_nodes=nodes[index + 1], next_edges=edges[index + 1],
              next_globals=globals_[index + 1][:, None], actions=raw['action'][index], rewards=raw['reward'][index], terminals=raw['terminal'][index])
  args = {k: jnp.asarray(v) for k, v in args.items()}
  single = amg.single_graph(nodes, edges, globals_, raw)
  pv = amg.unflatten_like(jax.eval_shape(policy.init, jax.random.key(0), single), agent_common.deterministic_leaf)
  cv = amg.unflatten_like(jax.eval_shape(twin.init, jax.random.key(0), single, jnp.zeros((1, 4, 7))), agent_common.deterministic_leaf)
  tc = train.TrainConfig(train_batch_size=amg.TD3_BATCH, fused_training_steps=amg.TD3_STEPS)
  state, opt = initial_state(tc, pv, cv)
  topo16 = networks.get_graph_topology(cfg.env_config, fc, batch_size=amg.TD3_BATCH)
  state, stats = train._train_step_impl(state, args, topo16, policy.apply, twin.apply, opt, tc)
  snap = amg.snapshot(state, 's')
  print('replay TD3 step 1: losses', float(stats['twin_critic_loss']), float(stats['policy_loss']), 'golden', golden['losses'][0].tolist(),
        'pol rms', float(np.sqrt(np.mean((snap['s/pol/sample'] - golden['step1/pol/sample']) ** 2))),
        'cri rms', float(np.sqrt(np.mean((snap['s/cri/sample'] - golden['step1/cri/sample']) ** 2))), flush=True)


def main():
  random.seed(0)
  np.random.seed(0)
  step, pol_flat, cri_flat = agent_orbax_dump.restore(sys.argv[1])
  assert step == 1870000
  jax.config.update('jax_default_device', jax.devices('cpu')[0])
  policy_params, critic_params = nest(pol_flat), nest(cri_flat)
  replay_existing_goldens(policy_params, critic_params)
  make_init()
  make_cases(policy_params, critic_params)
  make_bf16(policy_params, critic_params)
  print('DONE')


if __name__ == '__main__':
  main()
