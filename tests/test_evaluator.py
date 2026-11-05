"""The evaluator: isolation from training, metric definitions, first terminal per world."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import CriticNet, PolicyNet, load_checkpoint
from roboballet_mjlab.config import EnvCfg, ModelCfg, WorldCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.paths import scene_pool
from roboballet_mjlab.structs import Scene
from tests.golden.agent_load import CHECKPOINT
from tests.test_sim_common import DEVICE, default_scene, stored_scenes
from tests.test_trainer import SMALL, make_trainer, small_pool
from roboballet_mjlab.train.evaluator import Evaluator, load_scenes
from roboballet_mjlab.train.trainer import Trainer

REPO = Path(__file__).resolve().parents[1]
T = 2
MAX_STEPS = 40  # int((2 + 2) / 0.1)
ALL_DONE_LENGTH = 7  # Solved on step 1, then dwell 5, so 6 actions and 7 observations.
SAME_ROBOT_LENGTH = 12  # One robot takes both: one on step 1, the other as its dwell ends on step 6, then dwell 5.
METRICS = ('num_targets_done', 'final_score', 'episode_length', 'total_collision_robot_steps',
           'total_collision_penalty', 'total_mean_acceleration', 'network_step')


def crafted_pool(far: Scene | None = None) -> Scene:
  """Three kinds of episode, with arms held still, so the outcome is known in advance.

  Scene 0: both targets on tips at home, ending after dwell. Scene 1: one on a tip, one far
  away, running to the timeout with one done. Scene 2: both far away, nothing done. Scene 3:
  both on one tip, ending later than scene 0, so a second episode can differ from a first.
  """
  cfg = EnvCfg(num_targets=T)
  far = default_scene(cfg.world, 4, num_targets=T) if far is None else far
  probe = BalletEnv(cfg, 4, far, DEVICE, seed=1)
  probe.reset(scene_ids=torch.arange(4, device=DEVICE))
  tip_pos, tip_rot = probe.observe().robot.tip_pos, probe.observe().robot.tip_rot
  target_pos, target_rot = far.target_pos.clone(), far.target_rot.clone()
  target_pos[0, 0], target_rot[0, 0] = tip_pos[0, 0], tip_rot[0, 0]
  target_pos[0, 1], target_rot[0, 1] = tip_pos[0, 1], tip_rot[0, 1]
  target_pos[1, 0], target_rot[1, 0] = tip_pos[1, 0], tip_rot[1, 0]
  target_pos[3, 0], target_rot[3, 0] = tip_pos[3, 0], tip_rot[3, 0]
  target_pos[3, 1], target_rot[3, 1] = tip_pos[3, 0], tip_rot[3, 0]
  return far._replace(target_pos=target_pos, target_rot=target_rot)


def still(graph) -> torch.Tensor:
  return torch.zeros(graph.nodes.shape[0], 4, 7, device=DEVICE)


@pytest.fixture(scope='module')
def crafted() -> Evaluator:
  cfg = EnvCfg(num_targets=T)
  evaluator = Evaluator(cfg, SMALL, {'static_paper': crafted_pool()}, DEVICE, seed=5, static_worlds=32, compile=False)
  evaluator._forward = still
  return evaluator


def test_metric_definitions_on_crafted_worlds(crafted):
  policy = PolicyNet(SMALL, crafted.topo).to(DEVICE)
  results = crafted.evaluate(policy, network_step=123)['static_paper']
  per_world = crafted.last_per_world['static_paper']
  assert results['worlds'] == 32
  ids = crafted.scene_ids['static_paper']
  assert ids.tolist() == [0, 1, 2, 3] * 8
  done = {0: 2.0, 1: 1.0, 2: 0.0, 3: 2.0}
  length = {0: float(ALL_DONE_LENGTH), 1: MAX_STEPS + 1.0, 2: MAX_STEPS + 1.0, 3: float(SAME_ROBOT_LENGTH)}
  score = {0: 1.0, 1: 0.5, 2: 0.0, 3: 1.0}
  for world, scene in enumerate(ids.tolist()):
    assert per_world['num_targets_done'][world] == done[scene]
    assert per_world['episode_length'][world] == length[scene], 'observations, so steps + 1'
    assert per_world['final_score'][world] == score[scene]
    assert per_world['total_collision_robot_steps'][world] == 0
    assert per_world['total_mean_acceleration'][world] == 0
  assert results['num_targets_done'] == pytest.approx(1.25)
  assert results['task_rate'] == pytest.approx(0.625), 'num_targets_done / T'
  # Sample standard deviation of 2, 1, 0, 2 repeated 8 times, over T = 2 and the root of 32 worlds.
  assert results['task_rate_se'] == pytest.approx((22 / 31) ** 0.5 / 2 / 32 ** 0.5)
  assert results['all_done_fraction'] == pytest.approx(0.5)
  assert results['final_score'] == pytest.approx(0.625)
  assert results['episode_length'] == pytest.approx((ALL_DONE_LENGTH + SAME_ROBOT_LENGTH + 2 * (MAX_STEPS + 1)) / 4)
  assert results['network_step'] == 123
  assert set(METRICS) <= set(results)


def test_world_that_resets_is_not_counted_again(crafted):
  """Scene 0 ends after 6 actions, then its world runs on: a new draw from the pool, even an
  episode that ends in the rollout with another length, must change nothing."""
  policy = PolicyNet(SMALL, crafted.topo).to(DEVICE)
  env = crafted.envs['static_paper']
  ends = []
  step = env.step

  def counting(action):
    out = step(action)
    ends.append(out.terminal.clone())
    return out

  env.step = counting
  try:
    crafted.evaluate(policy, 0)
    per_world = crafted.last_per_world['static_paper']
    terminals = torch.stack(ends).cpu().numpy()
    ids = crafted.scene_ids['static_paper'].tolist()
    differs = 0
    for world in range(env.num_worlds):
      when = terminals[:, world].nonzero()[0]
      if len(when) >= 2:
        differs += int(when[1] - when[0]) != int(when[0] + 1)
    assert differs > 0, 'a second episode that ended with another length must occur, or this checks nothing'
    expected = {0: ALL_DONE_LENGTH, 1: MAX_STEPS + 1, 2: MAX_STEPS + 1, 3: SAME_ROBOT_LENGTH}
    assert per_world['episode_length'].tolist() == [float(expected[k]) for k in ids]
  finally:
    env.step = step


def _slow_reference(env: BalletEnv, act, topo, scene_ids) -> dict[str, list[float]]:
  """A per-world Python loop, written from agents/evaluator.py and actor.py:283-293."""
  env.reset(scene_ids=scene_ids)
  W = env.num_worlds
  out = {k: [0.0] * W for k in ('num_targets_done', 'final_score', 'episode_length', 'total_collision_robot_steps',
                                'total_collision_penalty', 'total_mean_acceleration')}
  finished = [False] * W
  for _ in range(env.cfg.max_steps):
    action = act(build_graph(env.observe(), topo))
    step = env.step(action)
    for w in range(W):
      if finished[w]:
        continue
      out['total_collision_robot_steps'][w] += step.collision_robot_steps[w].item()
      out['total_collision_penalty'][w] += step.collision_penalty[w].item()
      out['total_mean_acceleration'][w] += step.acceleration[w].item()
      if step.terminal[w]:
        finished[w] = True
        out['num_targets_done'][w] = step.next_robot.target_done[w].sum().item()
        out['final_score'][w] = step.next_robot.score[w].item()
        out['episode_length'][w] = step.next_robot.step[w].item() + 1
    if all(finished):
      break
  assert all(finished)
  return out


def test_rollout_against_a_slow_per_world_loop_with_collisions_and_early_ends():
  """Real scenes and a violent scripted policy, so collision totals are nonzero, with worlds
  ending at different steps. Same seed and same draws on both sides, so the numbers match."""
  cfg = EnvCfg(num_targets=T)
  stored = Scene(*[x[:4] for x in stored_scenes(4)])
  far = torch.full_like(stored.target_pos[:, :T], 1000.0)
  pool = crafted_pool(stored._replace(target_pos=far, target_rot=stored.target_rot[:, :T]))
  evaluator = Evaluator(cfg, SMALL, {'static_paper': pool}, DEVICE, seed=11, static_worlds=16, compile=False)
  gen = torch.Generator(device=DEVICE).manual_seed(3)
  calls = [0]

  def scripted(graph):
    # Worlds of scenes 0 and 3 stay still and end early. Everything else swings hard through
    # 30 boxes, those worlds too once they have reset, so that anything counted after a first
    # terminal shows up in the collision and acceleration totals.
    b = graph.nodes.shape[0]
    swing = torch.randint(2, (b, 4, 7), generator=gen, device=DEVICE) * 2.0 - 1.0
    calls[0] += 1
    kinds = torch.arange(b, device=DEVICE) % 4
    return swing * ((kinds == 1) | (kinds == 2) | (calls[0] > SAME_ROBOT_LENGTH - 1))[:, None, None]

  evaluator._forward = scripted
  env = evaluator.envs['static_paper']
  ids = evaluator.scene_ids['static_paper']
  state, script_state = env.generator.get_state(), gen.get_state()
  reference = _slow_reference(env, scripted, evaluator.topo, ids)
  env.generator.set_state(state)
  gen.set_state(script_state)
  calls[0] = 0
  got = evaluator._rollout(env, ids)
  for key, values in reference.items():
    assert got[key].tolist() == pytest.approx(values, rel=1e-5, abs=1e-5), key
  assert sum(reference['total_collision_robot_steps']) > 0, 'collisions must have fired'
  assert sum(reference['total_collision_penalty']) > 0
  assert sum(reference['total_mean_acceleration']) > 0
  assert set(reference['episode_length']) == {float(ALL_DONE_LENGTH), float(SAME_ROBOT_LENGTH), MAX_STEPS + 1.0}, 'early and timed-out worlds'


def test_static_sets_force_every_scene_and_repeat_to_about_80_worlds():
  pool = stored_scenes(4)
  ten = Scene(*[x[:10] for x in pool])
  cfg = EnvCfg()
  evaluator = Evaluator(cfg, SMALL, {'static_dist': ten, 'static_paper': ten}, DEVICE, seed=2, compile=False)
  for name in ('static_dist', 'static_paper'):
    assert evaluator.envs[name].num_worlds == 80
    assert torch.equal(evaluator.scene_ids[name], torch.arange(10, device=DEVICE).repeat(8))
  env = evaluator.envs['static_paper']
  env.reset(scene_ids=evaluator.scene_ids['static_paper'])
  assert torch.equal(env.observe().scene.box_pos[:10], ten.box_pos)
  assert torch.equal(env.observe().scene.box_pos[70:80], ten.box_pos)


def test_absent_set_is_skipped_with_a_notice(tmp_path):
  notices = []
  cfg = EnvCfg()
  evaluator = Evaluator(cfg, SMALL, {'training_dist': stored_scenes(4), 'heldout': None, 'static_dist': tmp_path / 'absent.pt'}, DEVICE,
                        seed=2, num_worlds=4, compile=False, notice=notices.append)
  assert set(evaluator.envs) == {'training_dist'}
  text = ' '.join(notices)
  assert 'heldout: skipped' in text and 'static_dist: skipped' in text and 'static_paper: skipped' in text
  with pytest.raises(ValueError, match='unknown'):
    Evaluator(cfg, SMALL, {'test': stored_scenes(4)}, DEVICE, seed=2, compile=False)


def test_evaluator_streams_are_in_the_trainer_checkpoint(tmp_path):
  a = make_trainer(tmp_path / 'a', num_worlds=4)
  a.evaluator = _eval_for(a)
  a.evaluator.evaluate(a.td3.policy, 0)
  path = a.save_checkpoint('test')
  b = make_trainer(tmp_path / 'b', num_worlds=4)
  b.evaluator = _eval_for(b)
  assert not torch.equal(a.evaluator.envs['training_dist'].generator.get_state(), b.evaluator.envs['training_dist'].generator.get_state())
  b.load_checkpoint(path)
  for name in a.evaluator.envs:
    assert torch.equal(a.evaluator.envs[name].generator.get_state(), b.evaluator.envs[name].generator.get_state()), name
  a.logger.close()
  b.logger.close()


def _state_of(t: Trainer) -> dict[str, list[torch.Tensor]]:
  d, m = t.env.world.sim.data, t.env.world.sim.model
  return {
    'robot': [x.clone() for x in t.env.observe().robot],
    'scene': [x.clone() for x in t.env.observe().scene],
    'sim': [d.qpos.clone(), d.mocap_pos.clone(), d.mocap_quat.clone(), d.site_xpos.clone(), m.geom_size.clone()],
    'env_rng': [t.env.generator.get_state()],
    'streams': [t.act_rng.get_state(), t.sample_rng.get_state(), t.hindsight_rng.get_state(), t.td3.rng.get_state()],
    'global_rng': [torch.get_rng_state(), *torch.cuda.get_rng_state_all()],
    'policy': [p.clone() for net in (t.td3.policy, t.td3.policy_acting, t.td3.q1, t.td3.q2, t.td3.policy_target) for p in net.parameters()],
    'latches': [t.latches.finished.clone(), t.latches.held_step.clone(), t.latches.sigma.clone(), t.latches.use_random.clone()],
    'buffer': [x.clone() for x in t.buffer.robot] + [t.buffer.action.clone(), t.buffer.count.clone(), t.buffer.totals.clone()],
    'replay': [x.clone() for x in t.replay.robot] + [t.replay.action.clone(), t.replay.reward.clone(), t.replay.transition_row.clone()]
              + [torch.tensor([t.replay.inserted, t.replay.sampled, t.replay.evicted, t.replay.episodes])],
  }


def _assert_same(a, b):
  for key in a:
    assert len(a[key]) == len(b[key])
    for i, (x, y) in enumerate(zip(a[key], b[key])):
      assert torch.equal(x, y), f'{key}[{i}]'


def _eval_for(t: Trainer) -> Evaluator:
  return Evaluator(t.env_cfg, t.model_cfg, {'training_dist': small_pool(), 'static_paper': Scene(*[x[:3] for x in small_pool()])},
                   DEVICE, seed=t.cfg.seed, num_worlds=8, static_worlds=6, compile=False)


def test_evaluation_mid_training_touches_nothing_in_training(tmp_path):
  """Bit-exact on the state, then a twin that never evaluates stays bit-exact for many rounds.

  The second part is what catches a leak through something the first part does not list.
  """
  a = make_trainer(tmp_path / 'a', num_worlds=8, seed=4)
  b = make_trainer(tmp_path / 'b', num_worlds=8, seed=4)
  evaluator = _eval_for(a)
  a.evaluator = None  # Called by hand below, at a chosen moment.
  for _ in range(120):
    a.round()
    b.round()
  assert a.td3.grad_steps > 0, 'learning must be under way'
  torch.manual_seed(77)
  random.seed(7)
  before = _state_of(a)
  np_state = np.random.get_state()
  python_state = random.getstate()
  eval_rng = evaluator.envs['training_dist'].generator.get_state()
  results = evaluator.evaluate(a.td3.policy, a.td3.grad_steps)
  assert set(results) == {'training_dist', 'static_paper'}
  _assert_same(before, _state_of(a))
  assert random.getstate() == python_state and all(np.array_equal(x, y) for x, y in zip(np_state, np.random.get_state()))
  assert torch.get_float32_matmul_precision() == 'highest'
  # The evaluator's own stream did move, so it was not a no-op.
  assert not torch.equal(evaluator.envs['training_dist'].generator.get_state(), eval_rng)
  for _ in range(60):
    a.round()
    b.round()
  _assert_same(_state_of(a) | {'global_rng': []}, _state_of(b) | {'global_rng': []})
  a.logger.close()
  b.logger.close()


def test_two_evaluations_differ_by_stream_but_each_stream_is_seeded():
  pool = small_pool()
  cfg = EnvCfg(num_targets=3)
  make = lambda seed: Evaluator(cfg, SMALL, {'training_dist': pool}, DEVICE, seed=seed, num_worlds=8, compile=False)
  policy = PolicyNet(SMALL, Topology(cfg, DEVICE)).to(DEVICE)
  torch.nn.init.zeros_(policy.policy_head[-1].weight)
  one = make(1)
  two = make(1)
  three = make(2)
  a = one.evaluate(policy, 0)['training_dist']
  b = two.evaluate(policy, 0)['training_dist']
  assert a == b
  assert not torch.equal(one.envs['training_dist'].observe().scene.base_pos, three.envs['training_dist'].observe().scene.base_pos)


def test_evaluator_runs_f32_and_ignores_a_leaked_autocast_free_policy():
  """The forward is the plain policy: float32 at highest, whatever the global was."""
  pool = small_pool()
  cfg = EnvCfg(num_targets=3)
  evaluator = Evaluator(cfg, SMALL, {'training_dist': pool}, DEVICE, seed=1, num_worlds=4, compile=False)
  seen = []

  def spy(graph):
    seen.append((torch.get_float32_matmul_precision(), torch.is_autocast_enabled(), graph.nodes.dtype))
    return torch.zeros(graph.nodes.shape[0], 4, 7, device=DEVICE)

  evaluator._forward = spy
  torch.set_float32_matmul_precision('high')
  try:
    evaluator.evaluate(PolicyNet(SMALL, evaluator.topo).to(DEVICE), 0)
    assert torch.get_float32_matmul_precision() == 'high', 'and the caller keeps its own setting'
  finally:
    torch.set_float32_matmul_precision('highest')
  assert seen and all(s == ('highest', False, torch.float32) for s in seen)


def test_evaluator_policy_is_a_private_copy():
  pool = small_pool()
  cfg = EnvCfg(num_targets=3)
  evaluator = Evaluator(cfg, SMALL, {'training_dist': pool}, DEVICE, seed=1, num_worlds=4, compile=False)
  policy = PolicyNet(SMALL, evaluator.topo).to(DEVICE)
  evaluator.load_policy(policy)
  mine, theirs = next(evaluator.policy.parameters()), next(policy.parameters())
  assert torch.equal(mine, theirs) and mine.data_ptr() != theirs.data_ptr() and not mine.requires_grad
  with torch.no_grad():
    theirs += 1.0
  assert not torch.equal(mine, theirs)
  evaluator.load_policy(policy)
  assert torch.equal(mine, theirs)


def test_loads_both_kinds_of_scene_file_and_the_reference_checkpoint(tmp_path):
  """The start states of G4 and the pools of G8b, through the loader the scripts use."""
  pool_path = scene_pool('4r_10t_30o_static_paper')
  pool = load_scenes(pool_path, DEVICE)
  assert pool.base_pos.shape == (10, 4, 3) and pool.target_pos.shape == (10, 10, 3)
  cfg = EnvCfg(world=WorldCfg(arm_model='mesh_tool'))
  topo = Topology(cfg, 'cpu')
  policy, q1, q2 = PolicyNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo), CriticNet(ModelCfg(), topo)
  assert load_checkpoint(str(CHECKPOINT), policy, q1, q2) == 1_870_000


def test_eval_script_on_the_reference_checkpoint_prints_the_table(tmp_path):
  """The G4 command in small: sphere arm, a forced static pool, a pool drawn at random, a missing set."""
  out = tmp_path / 'out.json'
  env = {**os.environ, 'PYTHONPATH': str(Path(os.environ.get('PYTHONPATH') or REPO / 'src').resolve())}
  done = subprocess.run(
    [sys.executable, '-m', 'roboballet_mjlab.scripts.eval', '--checkpoint', str(CHECKPOINT), '--heldout', str(scene_pool('4r_10t_30o_heldout')),
     '--static-paper', str(scene_pool('4r_10t_30o_static_paper')), '--worlds', '8', '--static-worlds', '10', '--no-compile', '--json', str(out)],
    cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
  assert 'grad step 1870000' in done.stdout and 'static_dist: skipped' in done.stdout and 'task_rate' in done.stdout
  saved = json.loads(out.read_text())
  assert saved['step'] == 1_870_000 and set(saved['results']) == {'heldout', 'static_paper'}
  assert saved['results']['heldout']['worlds'] == 8 and saved['results']['static_paper']['worlds'] == 10
  for metrics in saved['results'].values():
    assert set(METRICS) <= set(metrics) and metrics['network_step'] == 1_870_000


def test_one_world_has_a_finite_standard_error():
  from roboballet_mjlab.train.evaluator import task_rate_se

  assert task_rate_se(torch.tensor([1.0]), 3) == 0.0

