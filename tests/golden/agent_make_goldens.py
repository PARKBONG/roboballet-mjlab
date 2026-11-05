"""Regenerates the agent_*.npz goldens with the original JAX code. Old venv only, no torch.

Run from $ROBOBALLET_PROJECT, which holds roboballet/ and venv/ (see agent_orbax_dump.py for the environment variables):
  venv/bin/python roboballet-mjlab/tests/golden/agent_make_goldens.py CKPT_DIR

agent_e6_{4r10t,8r40t}.npz  raw observations, scenes and aug codes from real rollouts.
agent_e6_4r10t_expected_{0,1}.npz, agent_e6_8r40t_expected_0.npz  original graph features.
agent_e7.npz    reference checkpoint policy and critic outputs, JAX CPU float32.
agent_td3.npz   20 original TD3 steps from weights in agent_common, real normals exported.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import jraph
import numpy as np

import agent_common
import agent_orbax_dump
from roboballet.agents import compute_features, networks
from roboballet.environments import data_pb2, planning_env
from roboballet.launch import hparams
from roboballet.train import train

OUT = os.path.dirname(os.path.abspath(__file__))
EXPECTED_SHARDS = 2
TD3_BATCH = 16
TD3_STEPS = 20


def make_configs(R, T, O=30):
  cfg = hparams.make_optimised_configs(num_targets=T, num_obstacles=O, num_robots=R, pregen=True)
  cfg = hparams.update_configs(cfg, 'model_config.edge_update_num_layers', 6)
  cfg = hparams.update_configs(cfg, 'env_config.return_to_start_reward_weight', 0.0)
  cfg = hparams.update_configs(cfg, 'env_config.return_to_start_required', False)
  return hparams.update_configs(cfg, 'model_config.use_bf16', False)


def start_states(name, k):
  out = []
  with open(f'roboballet/data/start_states/{name}.pb', 'rb', buffering=1 << 20) as f:
    while len(out) < k:
      assert f.read(1) == b'\x0a'
      length, shift = 0, 0
      while True:
        c = f.read(1)[0]
        length |= (c & 0x7f) << shift
        shift += 7
        if not c & 0x80:
          break
      out.append(planning_env.StartState.from_proto(data_pb2.StartState.FromString(f.read(length))))
  return out


class AugRecorder:
  """Records the aug codes the unmodified env draws in observation()."""

  def __init__(self):
    self.codes = []
    self._original = planning_env.BoxInfo.get_augmentation
    recorder, original = self, self._original

    def recording(box, selection):
      recorder.codes.append(selection)
      return original(box, selection)

    planning_env.BoxInfo.get_augmentation = recording

  def last(self, num_obstacles):
    return list(self.codes[-num_obstacles:])


def scene_arrays(env, obs, num_obstacles):
  boxes = [b for _, b in env._generated_obstacles]
  assert len(boxes) == num_obstacles
  return dict(
    base_pos=obs['robots']['base_poses'][:, :3, 3], base_rot=obs['robots']['base_poses'][:, :3, :3],
    box_pos=np.array([b.centroid for b in boxes], np.float32), box_rot=np.array([b.rotation for b in boxes], np.float32),
    box_spans=np.array([b.spans for b in boxes], np.float32),
    target_pos=obs['targets']['poses'][:, :3, 3], target_rot=obs['targets']['poses'][:, :3, :3],
  )


def obs_arrays(obs, codes, dwell_steps):
  tip = np.matmul(obs['robots']['base_poses'], obs['robots']['tip_relative_poses'])
  return dict(
    q=obs['robots']['joint_configurations'], qd=obs['robots']['joint_velocities'],
    dwell=np.rint(obs['robots']['dwelling'] * dwell_steps).astype(np.int8),
    tip_pos=tip[:, :3, 3], tip_rot=tip[:, :3, :3], target_done=obs['targets']['done'],
    score=obs['current_score'], step=obs['step'].astype(np.int32), aug_code=np.array(codes, np.int8),
  )


def rollouts(cfg, states, total_obs, policy_fn, recorder, random_steps=40):
  """Alternates policy and uniform-action episodes until `total_obs` observations."""
  fc = cfg.model_config.feature_config
  vmax = cfg.env_config.base_config.max_joint_velocity
  O = cfg.env_config.num_obstacles_to_generate
  dwell_steps = int(np.ceil(cfg.env_config.dwell_time_s / cfg.env_config.base_config.timestep_s))
  rows, scenes, graphs = [], [], []
  for episode, state in enumerate(states):
    use_policy = policy_fn is not None and episode % 2 == 0
    env = planning_env.PlanningEnv(cfg.env_config, start_state=state)
    spec = env.observation_spec()
    ts = env.reset()
    scenes.append(scene_arrays(env, ts.observation, O))
    steps = 0
    while True:
      obs = ts.observation
      codes = recorder.last(O)
      for box, code, spans in zip([b for _, b in env._generated_obstacles], codes, obs['obstacles']['spans']):
        assert np.array_equal(np.asarray(box.get_augmentation(code).spans, np.float32), spans)
      del recorder.codes[-O:]
      graph = compute_features.make_graph_features(obs, spec, fc)
      row = obs_arrays(obs, codes, dwell_steps)
      row.update(episode=episode)
      graphs.append(jax.tree_util.tree_map(np.asarray, graph))
      budget_left = total_obs - len(rows) - 1
      over = budget_left == 0 or (not use_policy and steps >= random_steps)
      if over or ts.last():
        row.update(action=np.zeros(env.action_spec().shape, np.float32), reward=0.0, terminal=0, has_next=False)
        rows.append(row)
        break
      if use_policy:
        action = np.asarray(policy_fn(graph)) * vmax
      else:
        action = np.random.uniform(-vmax, vmax, size=env.action_spec().shape).astype(np.float32)
      ts = env.step(action.copy())
      row.update(action=np.asarray(action / vmax, np.float32), reward=float(ts.reward), terminal=int(ts.last()), has_next=True)
      rows.append(row)
      steps += 1
    if len(rows) >= total_obs:
      break
  assert len(rows) == total_obs, (len(rows), total_obs)
  stack = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
  stack.update({k: np.stack([s[k] for s in scenes]) for k in scenes[0]})
  stack['reward'] = stack['reward'].astype(np.float32)
  stack['terminal'] = stack['terminal'].astype(np.int8)
  stack['senders'] = graphs[0].senders
  stack['receivers'] = graphs[0].receivers
  stack['max_steps'] = int(cfg.env_config.base_config.episode_timeout_s / cfg.env_config.base_config.timestep_s)
  nodes = np.stack([g.nodes for g in graphs])
  edges = np.stack([g.edges for g in graphs])
  globals_ = np.stack([g.globals[0] for g in graphs])
  return stack, nodes, edges, globals_


def save_e6(name, stack, nodes, edges, globals_, shards):
  np.savez_compressed(f'{OUT}/agent_e6_{name}.npz', **stack)
  for i, idx in enumerate(np.array_split(np.arange(len(nodes)), shards)):
    np.savez_compressed(f'{OUT}/agent_e6_{name}_expected_{i}.npz', nodes=nodes[idx], edges=edges[idx], globals=globals_[idx])


def make_e7(cfg, stack, nodes, edges, globals_, policy_params, critic_params, chunk=100):
  ms = cfg.model_config
  policy = networks.RoboBalletPolicyNet(config=ms)
  twin = networks.RoboBalletTwinCriticNet(config=ms)
  topo = networks.get_graph_topology(cfg.env_config, ms.feature_config, batch_size=chunk)

  @jax.jit
  def forward(n, e, g, a):
    graph = networks.reconstruct_graph_from_batched_data(n, e, g, topo)
    c1, c2 = twin.apply(critic_params, graph, a)
    return policy.apply(policy_params, graph), c1.q_values, c2.q_values

  out = [forward(jnp.asarray(nodes[i:i + chunk]), jnp.asarray(edges[i:i + chunk]), jnp.asarray(globals_[i:i + chunk])[:, None],
                 jnp.asarray(stack['action'][i:i + chunk])) for i in range(0, len(nodes), chunk)]
  np.savez_compressed(f'{OUT}/agent_e7.npz', action=np.concatenate([np.asarray(o[0]) for o in out]),
                      q1=np.concatenate([np.asarray(o[1]) for o in out]), q2=np.concatenate([np.asarray(o[2]) for o in out]))


def unflatten_like(tree, flat_fn):
  leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
  values = [jnp.asarray(flat_fn('/'.join(str(getattr(p, 'key', p)) for p in path), leaf.shape)) for path, leaf in leaves]
  return jax.tree_util.tree_unflatten(treedef, values)


def snapshot(state, label):
  out = {}
  for short, tree in (('pol', state.policy_params), ('cri', state.twin_critic_params),
                      ('tpol', state.target_policy_params), ('tcri', state.target_twin_critic_params)):
    sample, norm = agent_common.sample_leaves(agent_orbax_dump.flatten(tree))
    out[f'{label}/{short}/sample'] = sample
    out[f'{label}/{short}/norm'] = norm
  return out


def make_td3(cfg, stack, nodes, edges, globals_):
  """Original TD3 on real transitions, with deterministic weights both sides can derive."""
  ms = cfg.model_config
  policy = networks.RoboBalletPolicyNet(config=ms)
  twin = networks.RoboBalletTwinCriticNet(config=ms)
  rng = np.random.RandomState(1234)
  valid = np.flatnonzero(stack['has_next'])
  index = rng.choice(valid, size=TD3_STEPS * TD3_BATCH, replace=True)
  args = dict(
    nodes=nodes[index], edges=edges[index], globals=globals_[index][:, None],
    next_nodes=nodes[index + 1], next_edges=edges[index + 1], next_globals=globals_[index + 1][:, None],
    actions=stack['action'][index], rewards=stack['reward'][index], terminals=stack['terminal'][index],
  )
  args = {k: jnp.asarray(v) for k, v in args.items()}
  topo = networks.get_graph_topology(cfg.env_config, ms.feature_config, batch_size=TD3_BATCH)
  graph0 = single_graph(nodes, edges, globals_, stack)
  pv = unflatten_like(jax.eval_shape(policy.init, jax.random.key(0), graph0), agent_common.deterministic_leaf)
  cv = unflatten_like(jax.eval_shape(twin.init, jax.random.key(0), graph0, jnp.zeros((1, 4, 7))), agent_common.deterministic_leaf)

  def initial_state(tc, key=0, target_policy=None, target_critic=None):
    opt = train._make_optimisers(tc)
    return train.TD3TrainState(
      step=jnp.array(0, dtype=jnp.uint32), policy_params=pv, twin_critic_params=cv,
      target_policy_params=pv if target_policy is None else target_policy,
      target_twin_critic_params=cv if target_critic is None else target_critic,
      policy_opt_state=opt.policy_optimiser.init(pv), twin_critic_opt_state=opt.twin_critic_optimiser.init(cv),
      random_key=jax.random.PRNGKey(key)), opt

  # Per-step chain, recording the normals the original draws inside _critic_loss.
  tc = train.TrainConfig(train_batch_size=TD3_BATCH, fused_training_steps=TD3_STEPS)
  state, opt = initial_state(tc)
  slice_batch = lambda i: {k: v[i * TD3_BATCH:(i + 1) * TD3_BATCH] for k, v in args.items()}
  step_fn = jax.jit(lambda s, b: train._train_step_impl(s, b, topo, policy.apply, twin.apply, opt, tc))
  losses, normals, golden = [], [], {}
  for i in range(TD3_STEPS):
    critic_key, _ = jax.random.split(state.random_key, 2)
    normals.append(np.asarray(jax.random.normal(key=critic_key, shape=(TD3_BATCH, 4, 7))))
    state, stats = step_fn(state, slice_batch(i))
    losses.append((float(stats['twin_critic_loss']), float(stats['policy_loss'])))
    if i in (0, TD3_STEPS - 1):
      golden.update(snapshot(state, f'step{i + 1}'))
  golden.update(index=index, normals=np.stack(normals), losses=np.array(losses))

  # The same steps through the original fused entry point: its logging sums the losses.
  state, opt = initial_state(tc)
  fused_state, fused_stats = jax.jit(lambda s: train._fused_train_step_impl(
    TD3_STEPS, s, args, topo, policy.apply, twin.apply, opt, tc))(state)
  golden['fused_sums'] = np.array([float(fused_stats['twin_critic_loss']), float(fused_stats['policy_loss'])])
  golden.update(snapshot(fused_state, 'fused'))

  # Targets near the action bounds, normals beyond the clip, and a target critic
  # steep enough that the clipping shows in the loss. DESIGN 11.7.
  for label, sigma, bias in (('near_a', 0.05, 3.0), ('near_b', 5e-4, 5.0)):
    flat_pv = agent_orbax_dump.flatten(pv)
    target_policy = unflatten_like(pv, lambda key, shape: near_bound_leaf(key, flat_pv, bias))
    flat_cv = agent_orbax_dump.flatten(cv)
    target_critic = unflatten_like(cv, lambda key, shape: steep_critic_leaf(key, flat_cv))
    tc_near = train.TrainConfig(train_batch_size=TD3_BATCH, fused_training_steps=1, target_sigma=sigma)
    state, opt = initial_state(tc_near, target_policy=target_policy, target_critic=target_critic)
    critic_key, _ = jax.random.split(state.random_key, 2)
    eps = jax.random.normal(key=critic_key, shape=(TD3_BATCH, 4, 7))
    batch = slice_batch(0)
    next_graph = networks.reconstruct_graph_from_batched_data(batch['next_nodes'], batch['next_edges'], batch['next_globals'], topo)
    pi_next = policy.apply(target_policy, next_graph)  # train.py:222-227
    noisy = jnp.clip(pi_next + sigma * jnp.clip(eps, -tc_near.noise_clip, tc_near.noise_clip), -1.0, 1.0)
    state, stats = jax.jit(lambda s, b: train._train_step_impl(s, b, topo, policy.apply, twin.apply, opt, tc_near))(state, batch)
    golden.update({f'{label}/normals': np.asarray(eps), f'{label}/noisy_action': np.asarray(noisy),
                   f'{label}/losses': np.array([float(stats['twin_critic_loss']), float(stats['policy_loss'])]),
                   f'{label}/sigma': sigma, f'{label}/bias': bias})
    golden.update(snapshot(state, label))
  np.savez_compressed(f'{OUT}/agent_td3.npz', **golden)


def single_graph(nodes, edges, globals_, stack):
  return jraph.GraphsTuple(
    nodes=jnp.asarray(nodes[0]), edges=jnp.asarray(edges[0]), globals=jnp.asarray(globals_[:1]),
    senders=jnp.asarray(stack['senders']), receivers=jnp.asarray(stack['receivers']),
    n_node=jnp.array([nodes.shape[1]]), n_edge=jnp.array([edges.shape[1]]))


def near_bound_leaf(key, flat, bias):
  leaf = flat[key]
  if key.endswith('policy_head/Dense_2/bias'):
    return (bias * np.array([1, -1, 1, -1, 1, -1, 1])).astype(np.float32)
  return leaf


def steep_critic_leaf(key, flat):
  leaf = flat[key]
  return leaf * 50.0 if key.endswith('qvalue/Dense_2/kernel') else leaf


def main():
  ckpt = sys.argv[1]
  random.seed(0)
  np.random.seed(0)
  step, pol_flat, cri_flat = agent_orbax_dump.restore(ckpt)
  assert step == 1870000
  jax.config.update('jax_default_device', jax.devices('cpu')[0])
  recorder = AugRecorder()

  def nest(flat):
    root = {}
    for key, value in flat.items():
      node = root
      parts = key.split('/')
      for part in parts[:-1]:
        node = node.setdefault(part, {})
      node[parts[-1]] = jnp.asarray(value)
    return root

  policy_params, critic_params = nest(pol_flat), nest(cri_flat)
  cfg = make_configs(4, 10)
  policy = networks.RoboBalletPolicyNet(config=cfg.model_config)
  policy_fn = jax.jit(lambda g: policy.apply(policy_params, g))
  stack, nodes, edges, globals_ = rollouts(cfg, start_states('4r_10t_30o', 60), 1000, lambda g: policy_fn(g)[0], recorder)
  print('4r10t', {k: v.shape for k, v in stack.items() if hasattr(v, 'shape')})
  print('episodes', stack['episode'].max() + 1, 'done', stack['target_done'].sum(1).max(), 'dwell>0', (stack['dwell'] > 0).sum(),
        'reward>0', (stack['reward'] > 0).sum(), 'reward<0', (stack['reward'] < 0).sum(), 'terminal', stack['terminal'].sum())
  save_e6('4r10t', stack, nodes, edges, globals_, EXPECTED_SHARDS)
  make_e7(cfg, stack, nodes, edges, globals_, policy_params, critic_params)
  make_td3(cfg, stack, nodes, edges, globals_)

  cfg8 = make_configs(8, 40)
  stack8, nodes8, edges8, globals8 = rollouts(cfg8, start_states('8r_40t_30o', 4), 40, None, recorder, random_steps=9)
  print('8r40t', nodes8.shape, edges8.shape)
  save_e6('8r40t', stack8, nodes8, edges8, globals8, 1)
  print('DONE')


if __name__ == '__main__':
  main()
