"""Independent checks of agent/ on hostile inputs, goldens from the original code only.

See tests/golden/agent_adv_make_goldens.py for how the agent_adv_* files were made.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import CriticNet, PolicyNet, flax_names, load_checkpoint
from roboballet_mjlab.agent.td3 import TD3, Batch
from roboballet_mjlab.config import EnvCfg, ModelCfg, TrainCfg, WorldCfg
from roboballet_mjlab.structs import Graph, Obs, RobotState, Scene
from tests.golden import agent_common
from tests.golden.agent_load import CHECKPOINT, DEVICE, GOLDEN, flat_nets, load_e6

CASES = [(1, 1, 1), (2, 3, 5), (3, 7, 2), (4, 10, 30), (4, 40, 30), (5, 1, 40), (8, 10, 30), (8, 40, 30)]
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
E6_TOLERANCE = 1e-4
E7_TOLERANCE = 2e-3
PARAM_RMS_TOLERANCE = 5e-8
LOSS_TOLERANCE = 1e-4


def env_cfg(R: int, T: int, O: int, **kwargs) -> EnvCfg:
  return EnvCfg(world=WorldCfg(num_robots=R, num_obstacles=O), num_targets=T, **kwargs)


def load_case(case: tuple[int, int, int]) -> dict[str, np.ndarray]:
  R, T, O = case
  with np.load(GOLDEN / f'agent_adv_{R}r{T}t{O}o.npz') as z:
    return {k: z[k] for k in z.files}


def obs_of(raw: dict[str, np.ndarray]) -> Obs:
  t = lambda k: torch.as_tensor(raw[k], device=DEVICE)
  scene = Scene(t('base_pos'), t('base_rot'), t('box_pos'), t('box_rot'), t('box_spans'), t('target_pos'), t('target_rot'), torch.zeros_like(t('q')))
  robot = RobotState(t('q'), t('qd'), t('dwell'), t('tip_pos'), t('tip_rot'), t('target_done'), t('score'), t('step'), t('aug_code').long())
  return Obs(robot, scene)


def graph_of(raw: dict[str, np.ndarray]) -> Graph:
  return Graph(*[torch.as_tensor(raw[k], device=DEVICE) for k in ('nodes', 'edges', 'globals')])


def reference_nets(case: tuple[int, int, int]) -> tuple[PolicyNet, CriticNet, CriticNet]:
  if not CHECKPOINT.exists():
    pytest.fail(f'{CHECKPOINT} is missing: see agent/networks.py for the conversion.')
  topo = Topology(env_cfg(*case), DEVICE)
  policy, q1, q2 = PolicyNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo)
  assert load_checkpoint(str(CHECKPOINT), policy, q1, q2) == 1_870_000
  return policy.to(DEVICE), q1.to(DEVICE), q2.to(DEVICE)


@pytest.fixture(autouse=True)
def highest_precision():
  torch.set_float32_matmul_precision('highest')


@pytest.mark.parametrize('case', CASES)
def test_features_on_hostile_observations(case):
  raw = load_case(case)
  topo = Topology(env_cfg(*case), DEVICE)
  assert topo.max_steps == raw['max_steps']
  assert np.array_equal(topo.senders.cpu().numpy(), raw['senders'])
  assert np.array_equal(topo.receivers.cpu().numpy(), raw['receivers'])
  got = build_graph(obs_of(raw), topo)
  errors = {n: (a - b).abs().max().item() for n, a, b in zip(Graph._fields, got, graph_of(raw))}
  print('features', case, errors)
  assert all(a.shape == b.shape for a, b in zip(got, graph_of(raw)))
  assert max(errors.values()) <= E6_TOLERANCE
  assert max(errors.values()) <= 2e-6


# Compiling is slow, so only a small graph with edges of each kind and the largest.
@pytest.mark.parametrize('case,compiled', [*[(case, False) for case in CASES], ((2, 3, 5), True), ((8, 40, 30), True)])
def test_trained_weights_on_hostile_graphs(case, compiled):
  raw = load_case(case)
  nets = [torch.compile(n) if compiled else n for n in reference_nets(case)]
  graph = graph_of(raw)
  action = torch.as_tensor(raw['input_action'], device=DEVICE)
  with torch.no_grad():
    a, v1, v2 = nets[0](graph), nets[1](graph, action), nets[2](graph, action)
  da, d1, d2 = [np.abs(x.cpu().numpy() - raw[k]) for x, k in ((a, 'action'), (v1, 'q1'), (v2, 'q2'))]
  print(f'forward {case} compiled={compiled} action max {da.max():.2e} mean {da.mean():.2e} q1 {d1.max():.2e} q2 {d2.max():.2e} q range {raw["q1"].min():.3f} {raw["q1"].max():.3f}')
  assert max(da.max(), d1.max(), d2.max()) <= E7_TOLERANCE


def td3_for(case, label: str, golden: dict[str, np.ndarray], **train_kwargs) -> tuple[TD3, Batch]:
  raw = load_case(case)
  batch_size, steps = int(golden[f'{label}/batch']), int(golden[f'{label}/steps'])
  cfg = TrainCfg(batch_size=batch_size, fused_steps=steps, matmul_precision='highest', compile=False, bf16=False, **train_kwargs)
  td3 = TD3(ModelCfg(), cfg, Topology(env_cfg(*case), DEVICE), DEVICE)
  load_checkpoint(str(CHECKPOINT), td3.policy, td3.q1, td3.q2)
  td3.sync_targets()
  g = graph_of(raw)
  count = batch_size * steps
  rows = torch.arange(count, device=DEVICE)
  next_rows = (rows + 1) % len(g.nodes)
  t = lambda k: torch.as_tensor(golden[f'{label}/{k}'], device=DEVICE)
  batch = Batch(Graph(*[x[rows] for x in g]), Graph(*[x[next_rows] for x in g]), t('actions'), t('rewards'), t('terminals'))
  normals = iter(t('normals'))
  td3._draw_normals = lambda like: next(normals)
  return td3, batch


def rms_errors(td3: TD3, golden, label: str) -> dict[str, float]:
  """Empty for a step the golden keeps no snapshot of."""
  out = {}
  if f'{label}/pol/sample' not in golden:
    return out
  for net, flat in flat_nets(td3).items():
    sample, _ = agent_common.sample_leaves(flat)
    out[net] = float(np.sqrt(np.mean((sample - golden[f'{label}/{net}/sample']) ** 2)))
  return out


@pytest.fixture(scope='module')
def td3_golden() -> dict[str, np.ndarray]:
  with np.load(GOLDEN / 'agent_adv_td3.npz') as z:
    return {k: z[k] for k in z.files}


@pytest.mark.parametrize('case', [(2, 3, 5), (8, 40, 30)])
def test_td3_from_trained_weights_on_other_topologies(case, td3_golden):
  label = '{}r{}t{}o'.format(*case)
  td3, data = td3_for(case, label, td3_golden)
  size, steps = td3.cfg.batch_size, td3.cfg.fused_steps
  for i in range(steps):
    out = td3.train_step(data.slice(i * size, (i + 1) * size))
    got = np.array([out.critic.item(), out.policy.item()])
    errors = rms_errors(td3, td3_golden, f'{label}/step{i + 1}')
    relative = np.abs(got / td3_golden[f'{label}/losses'][i] - 1)
    print(f'td3 {case} step {i + 1} rel loss {relative} rms {errors}')
    assert relative.max() <= LOSS_TOLERANCE
    assert max(errors.values(), default=0.0) <= 3 * PARAM_RMS_TOLERANCE


def test_td3_with_magnified_hyperparameters(td3_golden):
  """Tau, LR, decay and noise big enough that update order and count origin dominate."""
  td3, data = td3_for(
    (3, 7, 2), 'magnified', td3_golden, lr_policy=2e-3, lr_policy_decay=0.5, lr_critic=1e-3, lr_critic_decay=0.25,
    decay_every_steps=2.0, target_noise_sigma=0.2, target_noise_clip=0.4, tau_policy=0.1, tau_critic=0.3, gamma=0.8,
  )
  size, steps = td3.cfg.batch_size, td3.cfg.fused_steps
  for i in range(steps):
    out = td3.train_step(data.slice(i * size, (i + 1) * size))
    got = np.array([out.critic.item(), out.policy.item()])
    errors = rms_errors(td3, td3_golden, f'magnified/step{i + 1}')
    relative = np.abs(got / td3_golden['magnified/losses'][i] - 1)
    print(f'magnified step {i + 1} losses {got} rel {relative} rms {errors}')
    assert relative.max() <= 2e-3
    assert max(errors.values(), default=0.0) <= 5e-6


def test_schedules_against_the_original_optax_schedule(td3_golden):
  td3 = TD3(SMALL, TrainCfg(compile=False, matmul_precision='highest'), Topology(EnvCfg(), DEVICE), DEVICE)
  for count, policy, critic in zip(td3_golden['schedule_counts'], td3_golden['schedule_policy'], td3_golden['schedule_critic']):
    assert td3.policy_lr(int(count)) == pytest.approx(policy, rel=2e-5)
    assert td3.critic_lr(int(count)) == pytest.approx(critic, rel=2e-5)


def test_init_against_a_real_flax_init():
  with np.load(GOLDEN / 'agent_adv_init.npz') as z:
    stats = {str(k): (z['std'][i], z['mean'][i], z['peak'][i], z['size'][i]) for i, k in enumerate(z['keys'])}
  td3 = TD3(ModelCfg(), TrainCfg(compile=False, matmul_precision='highest', seed=7), Topology(EnvCfg(), 'cpu'), 'cpu')
  names = {**flax_names(td3.policy, 'policy/params')}
  state = {**td3.policy.state_dict()}
  for prefix, net in (('critic/params/critic_1', td3.q1), ('critic/params/critic_2', td3.q2)):
    for name, key in flax_names(net, prefix).items():
      names[f'{prefix}|{name}'] = key
      state[f'{prefix}|{name}'] = net.state_dict()[name]
  assert set(names.values()) == set(stats)
  for name, key in names.items():
    std, mean, peak, size = stats[key]
    leaf = state[name].numpy().astype(np.float64)
    assert leaf.size == size, key
    if std == 0:
      assert leaf.std() == 0 and leaf.mean() == pytest.approx(mean), key
      continue
    # Both are samples of one distribution, so allow the sampling error of each.
    assert leaf.std() == pytest.approx(std, rel=6.0 / np.sqrt(size) + 1e-3), key
    assert abs(leaf.mean() - mean) <= 8.0 * std / np.sqrt(size), key
    assert np.abs(leaf).max() <= peak * (1 + 4.0 / np.log(size + 2.0)), key


@pytest.fixture(scope='module')
def bf16_golden() -> dict[str, np.ndarray]:
  with np.load(GOLDEN / 'agent_adv_bf16.npz') as z:
    return {k: z[k] for k in z.files}


def forward_rows(nets, e6, index, chunk, bf16):
  action = torch.as_tensor(e6.raw['action'], device=DEVICE)
  out = []
  with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=bf16):
    for i in range(0, len(index), chunk):
      rows = torch.as_tensor(index[i:i + chunk], device=DEVICE)
      graph = Graph(*[x[rows] for x in e6.graph])
      out.append((nets[0](graph), nets[1](graph, action[rows]), nets[2](graph, action[rows])))
  return [torch.cat(x).float().cpu().numpy() for x in zip(*out)]


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('name,case,shards,chunk', [('4r10t', (4, 10, 30), 2, 100), ('8r40t', (8, 40, 30), 1, 40)])
def test_bf16_forward_on_trained_weights(bf16_golden, name, case, shards, chunk, compiled):
  """The learner's arithmetic. The f32 forward must stay at f32 and bf16 must show its effect."""
  e6 = load_e6(name, shards=shards)
  index = bf16_golden[f'{name}/index']
  nets = [torch.compile(n) if compiled else n for n in reference_nets(case)]
  f32, bf16 = forward_rows(nets, e6, index, chunk, False), forward_rows(nets, e6, index, chunk, True)
  table = {}
  for label, reference in (('jax_cpu_bf16', 'cpu_bf16'), ('jax_gpu_bf16', 'gpu_bf16')):
    for j, field in enumerate(('action', 'q1')):
      theirs = bf16_golden[f'{name}/{reference}/{field}']
      f32_ref = bf16_golden[f'{name}/cpu_f32/{field}']
      table[f'{label} {field}'] = dict(
        torch_vs_jax_bf16=float(np.abs(bf16[j] - theirs).mean()),
        jax_effect=float(np.abs(theirs - f32_ref).mean()),
        torch_effect=float(np.abs(bf16[j] - f32[j]).mean()),
        torch_effect_max=float(np.abs(bf16[j] - f32[j]).max()),
        jax_effect_max=float(np.abs(theirs - f32_ref).max()),
      )
  print(f'bf16 {name} compiled={compiled}')
  for key, value in table.items():
    print(' ', key, {k: f'{v:.2e}' for k, v in value.items()})
  gap = float(np.abs(bf16_golden[f'{name}/cpu_bf16/action'] - bf16_golden[f'{name}/gpu_bf16/action']).mean())
  print('  jax cpu vs gpu bf16 action mean', f'{gap:.2e}')
  assert np.abs(f32[0] - bf16_golden[f'{name}/cpu_f32/action']).max() <= E7_TOLERANCE
  # The test of how much bf16 matters is only meaningful if the port's bf16 moves actions
  # by about as much as the original's, DESIGN D13 and algo_spec 10.3.
  entry = table['jax_gpu_bf16 action']
  assert entry['torch_effect'] >= entry['jax_effect'] / 3.0
  # Every bf16 operation now rounds as the original's, tests/test_agent_bf16.py. What remains is which
  # values land on a rounding boundary, from the order of f32 reductions. The original on CPU and on GPU
  # differ by this much as well, so the port must be no further from the original on GPU than that.
  assert entry['torch_vs_jax_bf16'] <= gap


def trained_td3(compile: bool, bf16: bool) -> tuple[TD3, Graph]:
  cfg = TrainCfg(batch_size=8, fused_steps=1, matmul_precision='highest', compile=compile, bf16=bf16)
  td3 = TD3(ModelCfg(), cfg, Topology(EnvCfg(), DEVICE), DEVICE)
  load_checkpoint(str(CHECKPOINT), td3.policy, td3.q1, td3.q2)
  td3.sync_targets()
  return td3, Graph(*[x[:8] for x in load_e6('4r10t', shards=2).graph])


@pytest.mark.parametrize('act_first', [True, False])
def test_compiled_act_is_float32_whatever_ran_under_autocast(act_first):
  """Actors stay in float32 under bf16. A compiled graph must not leak autocast either way."""
  td3, graph = trained_td3(True, True)
  e6 = load_e6('4r10t', shards=2)
  batch = Batch(graph, Graph(*[x[1:9] for x in e6.graph]), torch.as_tensor(e6.raw['action'][:8], device=DEVICE),
                torch.as_tensor(e6.raw['reward'][:8], device=DEVICE), torch.as_tensor(e6.raw['terminal'][:8], device=DEVICE))
  if act_first:
    td3.act(graph)
  td3.train_step(batch)
  acted = td3.act(graph)
  plain = PolicyNet(ModelCfg(), Topology(EnvCfg(), DEVICE)).to(DEVICE)
  plain.load_state_dict(td3.policy.state_dict())
  with torch.no_grad():
    exact = plain(graph)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
      coarse = plain(graph)
  assert (acted - exact).abs().max() <= 1e-5
  assert (acted - coarse).abs().max() > 1e-4  # bf16 really is different here.


def _rotation_about_z(angle: float) -> torch.Tensor:
  c, s = np.cos(angle), np.sin(angle)
  return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)


def test_features_do_not_see_a_rigid_motion_of_the_whole_world():
  raw = load_case((4, 10, 30))
  obs = obs_of(raw)
  topo = Topology(env_cfg(4, 10, 30), DEVICE)
  rot, shift = _rotation_about_z(0.7), torch.tensor([0.3, -0.2, 0.1], device=DEVICE)
  moved_pos = lambda p: (p.double() @ rot.double().T + shift.double()).float()
  moved_rot = lambda r: (rot.double() @ r.double()).float()
  robot, scene = obs
  moved = Obs(
    robot._replace(tip_pos=moved_pos(robot.tip_pos), tip_rot=moved_rot(robot.tip_rot)),
    scene._replace(
      base_pos=moved_pos(scene.base_pos), base_rot=moved_rot(scene.base_rot), box_pos=moved_pos(scene.box_pos),
      box_rot=moved_rot(scene.box_rot), target_pos=moved_pos(scene.target_pos), target_rot=moved_rot(scene.target_rot),
    ),
  )
  a, b = build_graph(obs, topo), build_graph(moved, topo)
  assert torch.equal(a.nodes, b.nodes) and torch.equal(a.globals, b.globals)
  assert (a.edges - b.edges).abs().max() <= 3e-6


def test_every_aug_code_describes_the_same_physical_box():
  """Scaled basis vectors agree up to order and sign, centroids exactly."""
  raw = load_case((2, 3, 5))
  obs = obs_of(raw)
  topo = Topology(env_cfg(2, 3, 5), DEVICE)
  robot, scene = obs
  graphs = [build_graph(Obs(robot._replace(aug_code=torch.full_like(robot.aug_code, code)), scene), topo) for code in range(4)]
  rows = topo.num_robot_edges + 2 * 3
  base = graphs[0].edges[:, rows:, 18:].reshape(-1, 3, 4)
  for g in graphs[1:]:
    other = g.edges[:, rows:, 18:].reshape(-1, 3, 4)
    assert torch.equal(base[..., 3], other[..., 3])
    columns = base[..., :3].transpose(-1, -2)
    match = ((columns[:, :, None] - other[..., :3].transpose(-1, -2)[:, None]).abs().amax(-1)
             .minimum((columns[:, :, None] + other[..., :3].transpose(-1, -2)[:, None]).abs().amax(-1)))
    assert match.amin(-1).max() <= 1e-6
    assert not torch.equal(base[..., :3], other[..., :3])


def test_relative_translations_are_in_units_of_five_metres():
  raw = load_case((2, 3, 5))
  obs = obs_of(raw)
  topo = Topology(env_cfg(2, 3, 5), DEVICE)
  robot, scene = obs
  # Moving target 1 by 5 m along the first axis of robot 0's tip frame adds 1 to x there.
  moved_pos = scene.target_pos.clone()
  moved_pos[:, 1] = moved_pos[:, 1] + 5.0 * robot.tip_rot[:, 0, :, 0]
  a = build_graph(obs, topo).edges
  b = build_graph(Obs(robot, scene._replace(target_pos=moved_pos)), topo).edges
  row = topo.num_robot_edges + 1
  assert (b[:, row, 9] - a[:, row, 9] - 1.0).abs().max() <= 2e-6
  assert (b[:, row, 10:12] - a[:, row, 10:12]).abs().max() <= 2e-6


def _perturb(graph: Graph, row: int, column_start: int, column_stop: int) -> Graph:
  edges = graph.edges.clone()
  edges[:, row, column_start:column_stop] += 0.5
  return graph._replace(edges=edges)


def test_messages_reach_only_the_nodes_they_should():
  """Directions: a target edge reaches its receiver, a robot edge its sender and receiver."""
  case = (4, 10, 30)
  raw = load_case(case)
  policy, q1, _ = reference_nets(case)
  topo = Topology(env_cfg(*case), DEVICE)
  graph = graph_of(raw)
  with torch.no_grad():
    base = policy(graph)
    moved = lambda g: (policy(g) - base).abs().amax((0, 2)) > 1e-5
    target_row = topo.num_robot_edges + 2 * 10 + 4  # Target 4 to robot 2.
    assert moved(_perturb(graph, target_row, 9, 18)).tolist() == [False, False, True, False]
    obstacle_row = topo.num_robot_edges + 4 * 10 + 3 * 30 + 7  # Obstacle 7 to robot 3.
    assert moved(_perturb(graph, obstacle_row, 18, 30)).tolist() == [False, False, False, True]
    robot_row = next(i for i in range(12) if topo.senders[i] == 1 and topo.receivers[i] == 3)
    assert moved(_perturb(graph, robot_row, 0, 9)).tolist() == [False, True, False, True]
    action = torch.as_tensor(raw['input_action'], device=DEVICE)
    assert (q1(_perturb(graph, target_row, 9, 18), action) - q1(graph, action)).abs().max() > 1e-5
    # A critic's actions must only enter on robot nodes, so a target embedding stays fixed.
    n = q1.gnn.embed_nodes(torch.cat([graph.nodes[:, :4], action], -1), graph.nodes)
    n2 = q1.gnn.embed_nodes(torch.cat([graph.nodes[:, :4], -action], -1), graph.nodes)
    assert torch.equal(n[:, 4:], n2[:, 4:]) and not torch.equal(n[:, :4], n2[:, :4])


def _permute(obs: Obs, robots: torch.Tensor, targets: torch.Tensor, boxes: torch.Tensor) -> Obs:
  robot, scene = obs
  return Obs(
    robot._replace(
      q=robot.q[:, robots], qd=robot.qd[:, robots], dwell=robot.dwell[:, robots], tip_pos=robot.tip_pos[:, robots],
      tip_rot=robot.tip_rot[:, robots], target_done=robot.target_done[:, targets], aug_code=robot.aug_code[:, boxes],
    ),
    scene._replace(
      base_pos=scene.base_pos[:, robots], base_rot=scene.base_rot[:, robots], box_pos=scene.box_pos[:, boxes], box_rot=scene.box_rot[:, boxes],
      box_spans=scene.box_spans[:, boxes], target_pos=scene.target_pos[:, targets], target_rot=scene.target_rot[:, targets],
    ),
  )


@pytest.mark.parametrize('case', [(4, 10, 30), (8, 40, 30)])
def test_nets_are_equivariant_to_relabelling(case):
  raw = load_case(case)
  R, T, O = case
  obs = obs_of(raw)
  topo = Topology(env_cfg(*case), DEVICE)
  policy, q1, _ = reference_nets(case)
  generator = torch.Generator(device=DEVICE).manual_seed(5)
  robots = torch.randperm(R, generator=generator, device=DEVICE)
  targets = torch.randperm(T, generator=generator, device=DEVICE)
  boxes = torch.randperm(O, generator=generator, device=DEVICE)
  action = torch.as_tensor(raw['input_action'], device=DEVICE)
  with torch.no_grad():
    g, g2 = build_graph(obs, topo), build_graph(_permute(obs, robots, targets, boxes), topo)
    a, a2 = policy(g)[:, robots], policy(g2)
    v, v2 = q1(g, action), q1(g2, action[:, robots])
  print(f'equivariance {case} action {(a - a2).abs().max().item():.2e} q {(v - v2).abs().max().item():.2e}')
  assert (a - a2).abs().max() <= E7_TOLERANCE and (v - v2).abs().max() <= E7_TOLERANCE


def test_batch_of_one_equals_the_batch():
  case = (8, 40, 30)
  raw = load_case(case)
  policy, q1, _ = reference_nets(case)
  graph = graph_of(raw)
  action = torch.as_tensor(raw['input_action'], device=DEVICE)
  with torch.no_grad():
    a, v = policy(graph), q1(graph, action)
    one = [policy(Graph(*[x[i:i + 1] for x in graph])) for i in range(len(a))]
    single = [q1(Graph(*[x[i:i + 1] for x in graph]), action[i:i + 1]) for i in range(len(a))]
  assert (a - torch.cat(one)).abs().max() <= 1e-4 and (v - torch.cat(single)).abs().max() <= 1e-4


def test_zero_dwell_time_matches_the_original_scaling():
  """planning_env.py:979-986 reports dwelling zeros when dwell_steps is 0, so the column is -1."""
  raw = load_case((2, 3, 5))
  cfg = dataclasses.replace(env_cfg(2, 3, 5), dwell_time_s=0.0)
  topo = Topology(cfg, DEVICE)
  assert topo.dwell_steps == 0
  obs = obs_of(raw)
  robot = obs.robot._replace(dwell=torch.zeros_like(obs.robot.dwell))
  nodes = build_graph(Obs(robot, obs.scene), topo).nodes
  assert nodes.isfinite().all()
  assert (nodes[:, :2, 14] == -1.0).all()


def test_fused_step_with_an_empty_budget_is_a_no_op():
  """DESIGN 11.5: a fused call runs only as many steps as the budget permits, maybe none."""
  raw = load_case((2, 3, 5))
  graph = graph_of(raw)
  action = torch.as_tensor(raw['input_action'], device=DEVICE)
  zeros = torch.zeros(len(action), device=DEVICE)
  td3 = TD3(SMALL, TrainCfg(batch_size=2, fused_steps=2, compile=False, matmul_precision='highest'), Topology(env_cfg(2, 3, 5), DEVICE), DEVICE)
  params, rng = [p.clone() for p in td3.policy.parameters()], td3.rng.get_state()
  for summed in (False, True):
    out = td3.fused_step(Batch(graph, graph, action, zeros, zeros), num_steps=0, summed=summed)
    assert out.critic.item() == 0.0 and out.policy.item() == 0.0
  assert td3.grad_steps == 0
  assert all(torch.equal(a, b) for a, b in zip(params, td3.policy.parameters()))
  assert torch.equal(rng, td3.rng.get_state())


def _small_batch(batch_size: int, steps: int) -> Batch:
  raw = load_case((2, 3, 5))
  graph = graph_of(raw)
  count = batch_size * steps
  action = torch.as_tensor(raw['input_action'][:count], device=DEVICE)
  reward = torch.linspace(0.0, 0.1, count, device=DEVICE)
  next_graph = Graph(*[x[1:count + 1] for x in graph])
  return Batch(Graph(*[x[:count] for x in graph]), next_graph, action, reward, torch.zeros(count, device=DEVICE))


def test_real_target_noise_is_standard_normal_and_changes_every_step():
  """Every golden injects normals, so nothing else looks at the real draw."""
  td3 = TD3(SMALL, TrainCfg(compile=False, matmul_precision='highest'), Topology(env_cfg(2, 3, 5), DEVICE), DEVICE)
  like = torch.zeros(4096, 2, 7, device=DEVICE)
  a, b = td3._draw_normals(like), td3._draw_normals(like)
  assert a.shape == like.shape and a.dtype == torch.float32
  assert not torch.equal(a, b)
  assert abs(a.mean().item()) < 0.02 and abs(a.std().item() - 1.0) < 0.02
  assert not torch.equal(a[0], a[1]) and not torch.equal(a[:, 0], a[:, 1])


@pytest.mark.parametrize('bf16', [False, True])
def test_compiled_learner_resumes_identically(tmp_path, bf16):
  """The critic view used by the policy loss must still see the loaded parameters."""
  data = _small_batch(2, 4)
  cfg = TrainCfg(batch_size=2, fused_steps=2, compile=True, bf16=bf16, matmul_precision='highest')
  topo = Topology(env_cfg(2, 3, 5), DEVICE)
  first = TD3(SMALL, cfg, topo, DEVICE)
  first.fused_step(data.slice(0, 4))
  torch.save(first.state_dict(), tmp_path / 'learner.pt')
  second = TD3(SMALL, dataclasses.replace(cfg, seed=99), topo, DEVICE)
  second.load_state_dict(torch.load(tmp_path / 'learner.pt', map_location=DEVICE, weights_only=True))
  for frozen, live in zip(second.q1_frozen.parameters(), second.q1.parameters()):
    assert frozen.data_ptr() == live.data_ptr()
  a, b = first.fused_step(data.slice(4, 8)), second.fused_step(data.slice(4, 8))
  assert (a.critic - b.critic).abs() <= 1e-5 * a.critic.abs() and (a.policy - b.policy).abs() <= 1e-5 * a.policy.abs()
  for x, y in zip(first.policy.parameters(), second.policy.parameters()):
    assert (x - y).abs().max() <= 1e-5


def test_megabatch_shorter_than_the_steps_is_refused_or_harmless():
  """The original's dynamic_slice clamps. An empty slice must never poison the parameters."""
  data = _small_batch(2, 2)
  td3 = TD3(SMALL, TrainCfg(batch_size=2, fused_steps=3, compile=False, matmul_precision='highest'), Topology(env_cfg(2, 3, 5), DEVICE), DEVICE)
  try:
    td3.fused_step(data)
  except (ValueError, IndexError, RuntimeError):
    return
  assert all(p.isfinite().all() for p in td3.policy.parameters())
  assert all(p.isfinite().all() for p in td3.q1.parameters())
