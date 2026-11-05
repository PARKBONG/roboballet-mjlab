"""rsl-rl's PPO on BalletEnv: the adapter against the env, returns against rsl-rl's own code. DESIGN 17, gate G14.

Everything runs through the installed rsl_rl: OnPolicyRunner, PPO, RolloutStorage. The mutation check
at runs/ppo_dev/mutation_check.py injects bugs into a copy of ppo/ and confirms that these tests fail.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zlib

import pytest
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
import torch

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import CriticNet, PolicyNet, load_policy
from roboballet_mjlab.config import EnvCfg, ModelCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.ppo.config import PPOCfg
from roboballet_mjlab.ppo.models import BalletActor, BalletCritic, BalletPPO, ValueNet
from roboballet_mjlab.ppo.train import CHECKPOINT_NAME, FAILED_CHECKPOINT_NAME, POLICY_DIR, Trainer, atomic_save
from roboballet_mjlab.ppo.vecenv import EPISODE_STATS, BalletVecEnv, obs_to_tensordict, stagger_offsets, tensordict_to_obs
from roboballet_mjlab.structs import Obs, RobotState, Scene
from tests.test_sim_common import DEVICE, stored_scenes
from roboballet_mjlab.train.evaluator import Evaluator

REPO = Path(__file__).resolve().parents[1]
SMALL = ModelCfg(node_layers=2, node_size=16, edge_embed_layers=2, edge_update_layers=2, edge_size=8, global_layers=2, head_sizes=(8,))
T = 3
MAX_STEPS = 50  # Of T = 3.
ENV = EnvCfg(num_targets=T)
R, ACT = 4, 7
FULL_MODEL_WORLDS, FULL_MODEL_ITERATIONS = 64, 4


def small_pool(targets: int = T) -> Scene:
  scenes = stored_scenes(4)
  return scenes._replace(target_pos=scenes.target_pos[:, :targets], target_rot=scenes.target_rot[:, :targets])


def small_cfg(**changes) -> PPOCfg:
  base = dict(num_worlds=8, num_steps_per_env=6, num_learning_epochs=2, num_mini_batches=2, compile=False, profile=False,
              log_every_rounds=12, eval_every_rounds=10**9, checkpoint_every_s=1e9)
  return PPOCfg(**{**base, **changes})


def make_trainer(run_dir, evaluator=None, **changes) -> Trainer:
  torch.set_float32_matmul_precision('highest')
  return Trainer(ENV, small_cfg(**changes), SMALL, small_pool(), DEVICE, run_dir, evaluator=evaluator)


def make_runner(num_worlds: int = 8, steps: int = 6, stagger: bool = True, **changes) -> tuple[BalletVecEnv, OnPolicyRunner]:
  torch.set_float32_matmul_precision('highest')
  vec = BalletVecEnv(ENV, num_worlds, small_pool(), DEVICE, seed=3, stagger=stagger)
  cfg = small_cfg(num_worlds=num_worlds, num_steps_per_env=steps, **changes)
  return vec, OnPolicyRunner(vec, cfg.runner_dict(ENV, SMALL), log_dir=None, device=DEVICE)


def spy_env_step(vec: BalletVecEnv) -> list:
  """Every action BalletEnv receives and its StepOut."""
  log: list = []
  original = vec.env.step

  def step(action):
    out = original(action)
    log.append((action.clone(), out))
    return out

  vec.env.step = step
  return log


def set_value_bias(critic: BalletCritic, value: float) -> None:
  """A value far from zero, so a bootstrap could not hide in the rounding."""
  with torch.no_grad():
    critic.net.value[-1].bias.fill_(value)


def flat(module) -> torch.Tensor:
  return torch.cat([p.detach().flatten() for p in module.parameters()])


# The adapter against the env.


def test_the_adapter_is_a_vecenv_with_flat_actions():
  vec = BalletVecEnv(ENV, 4, small_pool(), DEVICE, seed=1, stagger=False)
  assert isinstance(vec, VecEnv)
  assert (vec.num_envs, vec.num_actions, vec.max_episode_length) == (4, R * ACT, MAX_STEPS)


def test_observations_rewards_dones_and_resets_equal_the_env_on_the_same_actions():
  """A twin env with the same seed gets the same actions, clamped, over more than two timeouts. Every byte agrees."""
  W, seed = 8, 11
  vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=seed, stagger=False)
  twin = BalletEnv(ENV, W, small_pool(), DEVICE, seed=seed)
  generator = torch.Generator(device=DEVICE).manual_seed(5)
  ended = 0
  for _ in range(2 * MAX_STEPS + 15):
    actions = torch.randn(W, R * ACT, generator=generator, device=DEVICE) * 1.5
    obs, reward, dones, extras = vec.step(actions)
    out = twin.step(actions.view(W, R, ACT).clamp(-1.0, 1.0))
    assert torch.equal(reward, out.reward)
    assert torch.equal(dones, out.terminal.long())
    assert extras == {}
    expected = twin.observe()
    for name, field in {**expected.robot._asdict(), **expected.scene._asdict()}.items():
      assert torch.equal(obs[name], field), name
    ended += int(dones.sum())
  assert ended >= 2 * W, 'the run crossed the timeouts of every world twice'


def test_observations_are_the_compact_fields_without_a_copy():
  vec = BalletVecEnv(ENV, 4, small_pool(), DEVICE, seed=1, stagger=False)
  fields = vec.get_observations()
  raw = vec.env.observe()
  assert set(fields.keys()) == {*RobotState._fields, *Scene._fields}
  assert fields.batch_size == torch.Size([4])
  assert fields['q'] is raw.robot.q and fields['aug_code'] is raw.robot.aug_code and fields['target_pos'] is raw.scene.target_pos
  back = tensordict_to_obs(fields)
  assert isinstance(back, Obs) and back.robot.dwell is raw.robot.dwell and back.scene.box_rot is raw.scene.box_rot
  assert fields['target_done'].dtype == torch.bool and fields['step'].dtype == torch.long


def test_an_observation_handed_out_is_not_changed_by_later_steps():
  """A rollout keeps the current row's observation until the step after it."""
  vec = BalletVecEnv(ENV, 8, small_pool(), DEVICE, seed=1)
  fields = vec.get_observations()
  kept = {name: fields[name].clone() for name in fields.keys()}
  generator = torch.Generator(device=DEVICE).manual_seed(5)
  for _ in range(MAX_STEPS + 2):
    vec.step(torch.randn(8, R * ACT, generator=generator, device=DEVICE))
  for name, value in kept.items():
    assert torch.equal(fields[name], value), name


def test_actions_reach_the_env_clipped_and_robot_by_robot():
  W = 4
  vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=1, stagger=False)
  log = spy_env_step(vec)
  actions = torch.linspace(-3.0, 3.0, W * R * ACT, device=DEVICE).view(W, R * ACT)
  before = actions.clone()
  vec.step(actions)
  sent = log[0][0]
  assert sent.shape == (W, R, ACT)
  assert sent.min() == -1.0 and sent.max() == 1.0
  assert torch.equal(sent, before.view(W, R, ACT).clamp(-1.0, 1.0))
  # World 1, robot 2, joint 3 of the flat vector.
  assert sent[1, 2, 3] == before[1, 2 * ACT + 3].clamp(-1.0, 1.0)
  assert torch.equal(actions, before), 'the rollout keeps the sample as it was drawn'


def test_the_rollout_keeps_the_sample_while_the_env_gets_it_clipped():
  vec, runner = make_runner()
  log = spy_env_step(vec)
  stored = []
  original = runner.alg.storage.add_transition

  def add(transition):
    stored.append(transition.actions.clone())
    original(transition)

  runner.alg.storage.add_transition = add
  runner.learn(1)
  assert len(stored) == 6
  samples = torch.stack(stored)
  assert samples.abs().max() > 1.0, 'std 1: some sample leaves the range'
  sent = torch.stack([a for a, _ in log])
  assert torch.equal(sent, samples.view(6, 8, R, ACT).clamp(-1.0, 1.0))
  assert torch.equal(runner.alg.storage.actions, samples), 'the sample, for its log probability'


def test_dones_hold_a_finish_as_well_as_a_timeout():
  """Each target is on a robot's own tip at the start: zero actions solve them at the first step, and the dwell ends the episode."""
  W = 4
  pool = small_pool()
  probe = BalletEnv(ENV, len(pool.q0), pool, DEVICE, seed=1)
  probe.reset(scene_ids=torch.arange(len(pool.q0), device=DEVICE))
  start = probe.observe().robot
  pool = pool._replace(target_pos=start.tip_pos[:, :T].clone(), target_rot=start.tip_rot[:, :T].clone())
  vec = BalletVecEnv(ENV, W, pool, DEVICE, seed=1, stagger=False)
  twin = BalletEnv(ENV, W, pool, DEVICE, seed=1)
  length = None
  for i in range(MAX_STEPS):
    _, reward, dones, extras = vec.step(torch.zeros(W, R * ACT, device=DEVICE))
    out = twin.step(torch.zeros(W, R, ACT, device=DEVICE))
    assert torch.equal(dones, out.terminal.long()) and torch.equal(reward, out.reward)
    if i == 0:
      assert torch.allclose(reward, torch.ones(W, device=DEVICE), atol=1e-6), 'all three targets at once'
    if dones.any():
      length = i + 1
      break
    assert 'time_outs' not in extras
  assert length == ENV.dwell_steps + 1 and dones.all(), 'a finish well before the timeout'
  assert (out.next_robot.step < MAX_STEPS).all()


def test_dones_hold_every_timeout_and_extras_never_name_one():
  W = 8
  vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=1, stagger=False)
  log = spy_env_step(vec)
  seen = 0
  for i in range(MAX_STEPS):
    _, _, dones, extras = vec.step(torch.zeros(W, R * ACT, device=DEVICE))
    assert 'time_outs' not in extras
    timed_out = log[-1][1].next_robot.step >= MAX_STEPS
    assert torch.equal(dones.bool() & timed_out, timed_out)
    seen += int(timed_out.sum())
  assert seen == W


def test_the_episode_length_buffer_is_the_step_input_and_cannot_be_randomised():
  vec = BalletVecEnv(ENV, 4, small_pool(), DEVICE, seed=1, stagger=False)
  vec.step(torch.zeros(4, R * ACT, device=DEVICE))
  assert vec.episode_length_buf.tolist() == [1, 1, 1, 1]
  with pytest.raises(AttributeError):
    vec.episode_length_buf = torch.zeros(4, dtype=torch.long, device=DEVICE)


def test_episode_statistics_against_a_literal_sum():
  W = 8
  vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=1)
  log = spy_env_step(vec)
  generator = torch.Generator(device=DEVICE).manual_seed(5)
  totals = torch.zeros(W, 4, device=DEVICE)
  finished = []
  for _ in range(2 * MAX_STEPS):
    vec.step(torch.randn(W, R * ACT, generator=generator, device=DEVICE))
    out = log[-1][1]
    totals += torch.stack([out.reward, out.collision_penalty, out.collision_robot_steps.float(), out.acceleration], 1)
    for world in out.terminal.nonzero().flatten().tolist():
      finished.append([out.next_robot.score[world].item(), *totals[world].tolist(),
                       out.next_robot.target_done[world].sum().item(), out.next_robot.step[world].item() + 1.0])
      totals[world] = 0.0
  stats = vec.take_episode_stats()
  assert list(stats) == list(EPISODE_STATS)
  expected = torch.tensor(finished, dtype=torch.float64).mean(0).tolist()
  assert list(stats.values()) == pytest.approx(expected, abs=1e-4)
  assert vec.episodes == len(finished) > W
  assert vec.take_episode_stats() == {}, 'a second call has nothing new'


# No bootstrap at a timeout, on rsl-rl's own return computation.


def literal_gae(rewards, values, dones, last_values, gamma, lam) -> torch.Tensor:
  """ppo.py:compute_returns written out per world and step, with no terminal bootstrapped."""
  steps, worlds = rewards.shape[:2]
  returns = torch.zeros_like(rewards)
  for world in range(worlds):
    advantage = 0.0
    for t in reversed(range(steps)):
      next_value = last_values[world] if t == steps - 1 else values[t + 1, world]
      alive = 1.0 - float(dones[t, world])
      delta = rewards[t, world] + alive * gamma * next_value - values[t, world]
      advantage = delta + alive * gamma * lam * advantage
      returns[t, world] = advantage + values[t, world]
  return returns


def run_rollout(alg: PPO, vec: BalletVecEnv, steps: int, keep_extras=lambda extras, out: extras):
  """The body of OnPolicyRunner.learn up to the returns, on_policy_runner.py:81-105."""
  log = spy_env_step(vec)
  obs = vec.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      actions = alg.act(obs)
      obs, rewards, dones, extras = vec.step(actions)
      alg.process_env_step(obs, rewards, dones, keep_extras(extras, log[-1][1]))
    alg.compute_returns(obs)
  return obs, log


@pytest.mark.parametrize('steps', [MAX_STEPS, MAX_STEPS + 7])
def test_a_timeout_row_returns_its_reward_alone_in_rsl_rl(steps):
  """`steps` = MAX_STEPS puts the timeout on the last row, where compute_returns would use last_values."""
  W = 8
  vec, runner = make_runner(W, steps, stagger=False)
  set_value_bias(runner.alg.critic, 0.5)
  obs, log = run_rollout(runner.alg, vec, steps)
  storage = runner.alg.storage
  rewards = torch.stack([out.reward for _, out in log])
  assert torch.equal(storage.rewards[..., 0], rewards), 'nothing added to the reward of any row'
  timed_out = torch.stack([out.next_robot.step >= MAX_STEPS for _, out in log])
  assert timed_out.any(0).all(), 'every world timed out in the rollout'
  assert storage.values.abs().min() > 0.4, 'values that a bootstrap would show'
  assert torch.allclose(storage.returns[timed_out][..., 0], rewards[timed_out], atol=1e-5)
  assert torch.equal(storage.dones[..., 0].bool(), torch.stack([out.terminal for _, out in log]))
  with torch.no_grad():
    last = runner.alg.critic(obs)
  assert torch.allclose(storage.returns, literal_gae(storage.rewards, storage.values, storage.dones, last, 0.94, 0.95), atol=2e-5)
  assert not (storage.returns[~timed_out].abs() < 1e-9).all(), 'the others do bootstrap'


def test_rsl_rl_would_bootstrap_if_the_adapter_named_a_timeout():
  """The same rollout with time_outs in extras, so the test above is able to tell the two apart."""
  W, steps = 8, MAX_STEPS
  vec, runner = make_runner(W, steps, stagger=False)
  set_value_bias(runner.alg.critic, 0.5)
  named = lambda extras, out: {**extras, 'time_outs': (out.next_robot.step >= MAX_STEPS).long()}
  _, log = run_rollout(runner.alg, vec, steps, keep_extras=named)
  storage = runner.alg.storage
  rewards = torch.stack([out.reward for _, out in log])
  timed_out = torch.stack([out.next_robot.step >= MAX_STEPS for _, out in log])
  added = storage.rewards[..., 0] - rewards
  assert torch.allclose(added[timed_out], 0.94 * storage.values[timed_out][..., 0], atol=1e-5)
  assert (storage.returns[timed_out][..., 0] - rewards[timed_out]).abs().min() > 0.4


# Stagger.


def test_each_world_resets_once_at_its_offset_before_any_rollout():
  W = 64
  resets = []
  original = BalletEnv.reset

  def spy(self, world_ids=None, scene_ids=None):
    resets.append(None if world_ids is None else world_ids.tolist())
    return original(self, world_ids, scene_ids)

  BalletEnv.reset = spy
  try:
    vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=2)
  finally:
    BalletEnv.reset = original
  offsets = ((torch.arange(W) * MAX_STEPS) // W).tolist()
  assert resets[0] is None, 'the constructor starts every world'
  reset_worlds = [world for ids in resets[1:] for world in ids]
  # A world can also end on its own during those rounds, here at a timeout of world 0's kind only.
  assert sorted(reset_worlds) == sorted(set(reset_worlds)), 'no world twice'
  assert set(reset_worlds) == {world for world, offset in enumerate(offsets) if offset > 0}
  assert vec.stagger_rounds == vec.rounds == max(offsets)
  steps = vec.env.observe().robot.step
  assert steps.tolist() == [vec.stagger_rounds - offset for offset in offsets]
  assert vec.discarded_rows == W * vec.stagger_rounds, 'no rollout holds any row of the stagger'
  assert vec.episodes == 0 and vec.take_episode_stats() == {}, 'stagger rows count as no episode'


def test_the_offsets_are_the_other_agents_rule():
  """train/trainer.py and fasttd3/trainer.py compute (world * max_steps) // W, and world 0 has none."""
  offsets = stagger_offsets(1024, 120, 'cpu')
  assert offsets[0] == -1 and offsets[8] == -1 and offsets[9] == 1 and offsets[1023] == 119


def test_episodes_end_on_different_rounds_and_no_reset_falls_inside_a_rollout():
  W, rounds = 64, MAX_STEPS
  vec = BalletVecEnv(ENV, W, small_pool(), DEVICE, seed=2)
  log = spy_env_step(vec)
  resets = []
  original = vec.env.reset

  def spy(world_ids=None, scene_ids=None):
    resets.append(world_ids.clone())
    return original(world_ids, scene_ids)

  vec.env.reset = spy
  ended_at = []
  for _ in range(rounds):
    _, _, dones, _ = vec.step(torch.zeros(W, R * ACT, device=DEVICE))
    ended_at.append(int(dones.sum()))
  assert sum(ended_at) == W, 'every world once in the episode, standing still'
  assert max(ended_at) <= W // MAX_STEPS + 2, f'spread over the rounds: {ended_at}'
  assert sum(n > 0 for n in ended_at) >= 0.9 * rounds
  terminals = [out.terminal.nonzero().flatten() for _, out in log if out.terminal.any()]
  assert len(resets) == len(terminals) and all(torch.equal(a, b) for a, b in zip(resets, terminals)), 'only terminals reset'


def test_a_rollout_after_the_stagger_has_no_row_with_a_made_up_terminal():
  W, steps = 16, MAX_STEPS
  vec, runner = make_runner(W, steps)
  _, log = run_rollout(runner.alg, vec, steps)
  storage = runner.alg.storage
  rows = torch.stack([out.next_robot.step for _, out in log])
  assert torch.equal(storage.dones[..., 0].bool(), rows >= MAX_STEPS), 'nothing ended but by the clock, with no targets solved'
  assert (storage.dones.sum(1)[..., 0] <= W // 2).all(), 'not aligned'


# Models.


def small_obs(num_worlds: int = 6):
  vec = BalletVecEnv(ENV, num_worlds, small_pool(), DEVICE, seed=4)
  return vec, vec.get_observations()


def make_actor(obs, seed=1, compile=False, bf16=False, model=SMALL) -> BalletActor:
  groups = {'actor': list(obs.keys())}
  return BalletActor(obs, groups, 'actor', R * ACT, ENV, model, seed, compile, bf16,
                     {'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}).to(DEVICE)


def make_critic(obs, seed=1, compile=False, bf16=False, model=SMALL) -> BalletCritic:
  groups = {'critic': list(obs.keys())}
  return BalletCritic(obs, groups, 'critic', 1, ENV, model, seed, compile, bf16).to(DEVICE)


def test_the_actor_mean_is_policy_net_on_the_features_of_the_env():
  vec, obs = small_obs()
  actor = make_actor(obs)
  assert isinstance(actor.net, PolicyNet)
  graph = build_graph(vec.env.observe(), Topology(ENV, DEVICE))
  with torch.no_grad():
    mean = actor(obs)
    direct = actor.net(graph)
  assert mean.shape == (6, R * ACT)
  assert torch.equal(mean, direct.flatten(1))
  # A mean far from zero, to see the tanh's limit.
  with torch.no_grad():
    actor.net.policy_head[-1].weight.zero_()
    actor.net.policy_head[-1].bias.fill_(50.0)
    assert torch.equal(actor(obs), actor.net(graph).flatten(1))
    assert actor(obs).abs().max() <= 1.0 and actor(obs).min() > 0.99


def test_the_initial_policy_is_the_seeds_first_draws_as_in_td3_and_fasttd3():
  _, obs = small_obs(2)
  actor = make_actor(obs, seed=7)
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(7)
    expected = PolicyNet(SMALL, Topology(ENV, DEVICE))
  assert torch.equal(flat(actor.net), flat(expected.to(DEVICE)))


def test_the_distribution_is_rsl_rl_gaussian_with_a_learned_std_per_joint():
  _, obs = small_obs()
  actor = make_actor(obs)
  std = actor.distribution.std_param
  assert std.requires_grad and std.shape == (R * ACT,) and (std == 1.0).all()
  with torch.no_grad():
    std.copy_(torch.linspace(0.5, 1.5, R * ACT, device=DEVICE))
    torch.manual_seed(3)
    sample = actor(obs, stochastic_output=True)
    mean = actor.output_mean
    expected = torch.distributions.Normal(mean, std)
    assert torch.equal(mean, actor(obs))
    assert torch.allclose(actor.get_output_log_prob(sample), expected.log_prob(sample).sum(-1))
    assert torch.allclose(actor.output_entropy, expected.entropy().sum(-1))
    torch.manual_seed(3)
    assert torch.equal(sample, expected.sample())
    assert not torch.equal(sample, mean)
  old = (mean, std.expand_as(mean))
  new = (mean + 0.1, std.expand_as(mean))
  kl = actor.get_kl_divergence(old, new)
  assert torch.allclose(kl, (0.1 ** 2 / (2 * std ** 2)).sum().expand(6), atol=1e-4)


def test_parameters_are_the_nets_alone_compiled_or_not():
  _, obs = small_obs(2)
  torch.set_float32_matmul_precision('highest')
  eager = make_actor(obs)
  compiled = make_actor(obs, compile=True)
  assert sum(p.numel() for p in compiled.parameters()) == sum(p.numel() for p in eager.parameters())
  assert sum(p.numel() for p in compiled.parameters()) == sum(p.numel() for p in compiled.net.parameters()) + R * ACT
  assert set(compiled.state_dict()) == set(eager.state_dict())
  # Gradients off first, as acting comes first, then on, then off again, on one compiled module.
  with torch.inference_mode():
    first = compiled(obs)
    assert torch.allclose(first, eager(obs), atol=1e-6)
  second = compiled(obs)
  second.sum().backward()
  assert next(compiled.net.policy_head.parameters()).grad is not None
  with torch.inference_mode():
    assert torch.equal(compiled(obs), first)


def test_full_model_counts():
  topo = Topology(ENV, 'cpu')
  assert sum(p.numel() for p in PolicyNet(ModelCfg(), topo).parameters()) == 8_949_575
  critic = sum(p.numel() for p in CriticNet(ModelCfg(), topo).parameters())
  # No reward head, and no action on the nodes: 7 inputs of the first dense layer.
  value = sum(p.numel() for p in ValueNet(ModelCfg(), topo).parameters())
  head = sum(p.numel() for p in CriticNet(ModelCfg(), topo).rewards.parameters())
  assert value == critic - head - 7 * ModelCfg().node_size


def test_the_value_is_critic_net_without_the_action_and_without_the_tanh():
  """CriticNet's weights in ValueNet, the action's columns dropped, and a zero action on CriticNet."""
  vec, obs = small_obs()
  topo = Topology(ENV, DEVICE)
  torch.manual_seed(5)
  critic = CriticNet(SMALL, topo).to(DEVICE)
  value = ValueNet(SMALL, topo).to(DEVICE)
  state = {k.replace('qvalue.', 'value.'): v for k, v in critic.state_dict().items() if not k.startswith('rewards.')}
  state['gnn.embed_node.0.weight'] = state['gnn.embed_node.0.weight'][:, :17]
  value.load_state_dict(state, strict=True)
  graph = build_graph(vec.env.observe(), topo)
  with torch.no_grad():
    q = critic(graph, torch.zeros(6, R, ACT, device=DEVICE))
    v = value(graph)
  assert v.shape == (6, 1)
  assert torch.allclose(torch.tanh(v[:, 0]), q, atol=1e-6)
  assert not torch.allclose(v[:, 0], q, atol=0.0)


def test_the_value_has_no_limit_and_trains_through_the_updated_global():
  _, obs = small_obs()
  critic = make_critic(obs)
  set_value_bias(critic, -2.5)
  value = critic(obs)
  assert value.shape == (6, 1) and value.max() < -2.0, 'below tanh\'s -1, as returns can be'
  value.sum().backward()
  assert critic.net.gnn.update_global[0].weight.grad.abs().max() > 0
  assert critic.net.gnn.embed_node[0].weight.grad.abs().max() > 0


def test_returns_can_leave_the_range_a_tanh_would_impose():
  """-15 / 120 per step with every robot rolled back, at gamma 0.94, for T = 10."""
  cfg = EnvCfg()
  assert -cfg.collision_penalty_per_robot * cfg.world.num_robots / (1 - 0.94) < -2.0


def test_a_batch_of_rows_goes_through_the_model_as_rsl_rl_slices_it():
  vec, runner = make_runner()
  run_rollout(runner.alg, vec, 6)
  storage = runner.alg.storage
  batch = next(storage.mini_batch_generator(2, 1))
  assert batch.observations.batch_size == torch.Size([24])
  graph = runner.alg.actor.graph(batch.observations)
  assert graph.nodes.shape[0] == 24 and batch.observations['aug_code'].dtype == torch.long
  rows = storage.observations.flatten(0, 1)
  index = (rows['q'][:, None] == batch.observations['q'][None]).all((2, 3)).float().argmax(0)
  by_hand = build_graph(tensordict_to_obs(rows[index]), runner.alg.actor.topo)
  assert torch.equal(graph.edges, by_hand.edges) and torch.equal(graph.nodes, by_hand.nodes) and torch.equal(graph.globals, by_hand.globals)
  with torch.no_grad():
    mean = runner.alg.actor(batch.observations)
  assert torch.allclose(mean, batch.old_distribution_params[0], atol=1e-6), 'the stored mean, before any update'


# It learns.


class ScoredVecEnv(BalletVecEnv):
  """The env's dynamics and dones with a dense reward: each robot's first joint command should be the sign of
  the summed target offsets along its tip's x axis, which only the edge features tell it."""

  @staticmethod
  def wanted(obs: Obs) -> torch.Tensor:
    robot, scene = obs
    delta = scene.target_pos[:, None] - robot.tip_pos[:, :, None]
    return torch.sign((delta * robot.tip_rot[:, :, None, :, 0]).sum(-1).sum(2))

  def step(self, actions):
    wanted = self.wanted(self.env.observe())
    fields, _, dones, extras = super().step(actions)
    first = actions.view(self.num_envs, self.num_robots, ACT)[..., 0].clamp(-1.0, 1.0)
    return fields, -(first - wanted).square().mean(1), dones, extras


def test_ppo_through_the_adapter_learns_a_reward_that_depends_on_the_observation():
  """The sparse task itself is too slow to be a test, runs/ppo_dev has those. This one takes 15 iterations."""
  W = 64
  torch.set_float32_matmul_precision('highest')
  torch.manual_seed(1)
  vec = ScoredVecEnv(ENV, W, small_pool(), DEVICE, seed=3)
  model = ModelCfg(node_layers=2, node_size=32, edge_embed_layers=2, edge_update_layers=2, edge_size=16, global_layers=2, head_sizes=(16,), head_last_scale=0.01)
  runner = OnPolicyRunner(vec, small_cfg(num_worlds=W, num_steps_per_env=16, num_learning_epochs=5, num_mini_batches=4).runner_dict(ENV, model), log_dir=None, device=DEVICE)

  def agreement() -> float:
    with torch.no_grad():
      mean = runner.alg.actor(vec.get_observations()).view(W, R, ACT)[..., 0]
    return (torch.sign(mean) == vec.wanted(vec.env.observe())).float().mean().item()

  assert 0.3 < agreement() < 0.7, 'a coin before training'
  runner.learn(15)
  assert agreement() > 0.8


# Precision.


def record_modes(trainer: Trainer) -> list:
  seen = []

  def observer(module, args):
    graph = args[0]
    seen.append(('actor' if module is trainer.actor.net else 'critic', torch.get_float32_matmul_precision(), torch.is_grad_enabled(),
                 torch.is_autocast_enabled('cuda'), graph.nodes.dtype))

  trainer.actor.net.register_forward_pre_hook(observer)
  trainer.critic.net.register_forward_pre_hook(observer)
  return seen


def test_acting_is_f32_at_highest_and_only_the_learner_call_runs_at_its_own(tmp_path):
  t = make_trainer(tmp_path)
  seen = record_modes(t)
  env_modes = []
  original = t.vec.env.step

  def step(action):
    env_modes.append(torch.get_float32_matmul_precision())
    return original(action)

  t.vec.env.step = step
  t.iteration()
  assert set(env_modes) == {'highest'}
  acting = [m for m in seen if not m[2]]
  learning = [m for m in seen if m[2]]
  assert len(acting) == 2 * 6 + 1, 'actor and critic per row, and the last value'
  assert {m[1] for m in acting} == {'highest'} and not any(m[3] for m in acting)
  assert {m[4] for m in seen} == {torch.float32}
  assert len(learning) == 2 * 2 * 2
  assert {m[1] for m in learning} == {'high'} and not any(m[3] for m in learning)
  assert torch.get_float32_matmul_precision() == 'highest'


def test_the_learner_can_run_bf16_and_acting_stays_f32(tmp_path):
  t = make_trainer(tmp_path, bf16=True)
  dtypes = {'acting': set(), 'learning': set()}

  def observer(module, args):
    dtypes['learning' if torch.is_grad_enabled() else 'acting'].add(args[0].dtype if isinstance(args[0], torch.Tensor) else None)

  # The first dense layer, which sees the type autocast casts to.
  t.actor.net.gnn.embed_edge[0].register_forward_pre_hook(observer)
  t.iteration()
  assert dtypes == {'acting': {torch.float32}, 'learning': {torch.bfloat16}}
  assert torch.get_float32_matmul_precision() == 'highest'


def test_the_global_precision_comes_back_to_what_it_was_also_through_an_exception(tmp_path, monkeypatch):
  t = make_trainer(tmp_path)
  obs = t.vec.get_observations()
  with torch.inference_mode():
    for _ in range(6):
      actions = t.alg.act(obs)
      obs, rewards, dones, extras = t.vec.step(actions)
      t.alg.process_env_step(obs, rewards, dones, extras)
    t.alg.compute_returns(obs)
  seen = []

  def fails(self):
    seen.append(torch.get_float32_matmul_precision())
    raise RuntimeError('in the learner')

  monkeypatch.setattr(PPO, 'update', fails)
  with pytest.raises(RuntimeError, match='in the learner'):
    t.alg.update()
  assert seen == ['high'] and torch.get_float32_matmul_precision() == 'highest'
  torch.set_float32_matmul_precision('medium')
  try:
    with pytest.raises(RuntimeError):
      t.alg.update()
    assert torch.get_float32_matmul_precision() == 'medium'
  finally:
    torch.set_float32_matmul_precision('highest')


def test_features_give_the_same_bytes_after_a_learner_call(tmp_path):
  """At `high`, a batched 3x3 matmul differs by up to 7e-4, agent/features.py. Nothing may leak."""
  t = make_trainer(tmp_path)
  obs = t.vec.env.observe()
  before = build_graph(obs, t.actor.topo)
  canary = torch.linalg.inv(obs.robot.tip_rot) @ obs.robot.tip_rot
  t.iteration()
  after = build_graph(obs, t.actor.topo)
  assert all(torch.equal(a, b) for a, b in zip(before, after))
  assert torch.equal(canary, torch.linalg.inv(obs.robot.tip_rot) @ obs.robot.tip_rot)


# The evaluator.


def small_evaluator(**changes) -> Evaluator:
  return Evaluator(ENV, SMALL, {'training_dist': small_pool(), 'heldout': small_pool()}, DEVICE, 1, num_worlds=8, compile=False, **changes)


def test_the_evaluator_takes_the_actor_and_runs_its_mean(tmp_path):
  evaluator = small_evaluator()
  t = make_trainer(tmp_path, evaluator=evaluator)
  t.iteration()
  results = t.evaluate()
  assert set(results) == {'training_dist', 'heldout'}
  assert torch.equal(flat(evaluator.policy), flat(t.actor.net)), 'the mean, not the std'
  env = evaluator.envs['training_dist']
  env.reset()
  fields = obs_to_tensordict(env.observe())
  with torch.no_grad():
    assert torch.equal(evaluator._forward(build_graph(env.observe(), evaluator.topo)).flatten(1), t.actor(fields))


def test_evaluation_mid_training_changes_nothing_in_training(tmp_path):
  evaluator = small_evaluator()
  t = make_trainer(tmp_path, evaluator=evaluator)
  t.iteration()
  weights = flat(t.actor), flat(t.critic)
  env_rng, torch_rng, cuda_rng = t.vec.env.generator.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()
  robot = t.vec.env.observe().robot
  t.evaluate()
  assert torch.equal(flat(t.actor), weights[0]) and torch.equal(flat(t.critic), weights[1])
  assert torch.equal(t.vec.env.generator.get_state(), env_rng)
  assert torch.equal(torch.get_rng_state(), torch_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
  assert t.vec.env.observe().robot.q is robot.q
  assert torch.get_float32_matmul_precision() == 'highest'


def test_evaluation_comes_at_the_rounds_fasttd3_uses(tmp_path):
  t = make_trainer(tmp_path, evaluator=small_evaluator(), eval_every_rounds=20)
  calls = []
  original = t.evaluator.evaluate

  def evaluate(policy, step):
    calls.append(t.rounds())
    return original(policy, step)

  t.evaluator.evaluate = evaluate
  start = t.rounds()
  for _ in range(6):
    t.iteration()
  # The first report-free check is at iteration end, the first round at or past a multiple of 20.
  assert calls[0] == start + 6
  for earlier, later in zip(calls, calls[1:]):
    assert (earlier // 20 + 1) * 20 <= later < (earlier // 20 + 1) * 20 + 6


# One iteration, through the runner.


def test_defaults_are_the_plans():
  cfg = PPOCfg()
  assert (cfg.num_worlds, cfg.num_steps_per_env, cfg.num_learning_epochs, cfg.num_mini_batches) == (1024, 24, 5, 4)
  assert (cfg.schedule, cfg.desired_kl, cfg.clip_param, cfg.entropy_coef, cfg.gamma, cfg.lam) == ('adaptive', 0.01, 0.2, 0.01, 0.94, 0.95)
  assert (cfg.eval_every_rounds, cfg.eval_episodes, cfg.checkpoint_every_s, cfg.log_every_rounds) == (2000, 1024, 900.0, 200)
  assert cfg.profile_every_rounds % cfg.num_steps_per_env == 0
  built = cfg.runner_dict(ENV, SMALL)
  assert built['algorithm']['class_name'] == 'roboballet_mjlab.ppo.models:BalletPPO'
  assert built['actor']['class_name'] == 'roboballet_mjlab.ppo.models:BalletActor'
  assert built['critic']['class_name'] == 'roboballet_mjlab.ppo.models:BalletCritic'
  assert built['actor']['distribution_cfg'] == {'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}
  assert built['algorithm']['gamma'] == 0.94 and built['algorithm']['lam'] == 0.95 and built['algorithm']['rnd_cfg'] is None


def test_the_runner_builds_rsl_rl_with_these_classes_and_arguments():
  vec, runner = make_runner()
  alg = runner.alg
  assert type(alg) is BalletPPO and isinstance(alg, PPO)
  assert type(alg.actor) is BalletActor and type(alg.critic) is BalletCritic
  assert (alg.gamma, alg.lam, alg.clip_param, alg.entropy_coef, alg.desired_kl, alg.schedule) == (0.94, 0.95, 0.2, 0.01, 0.01, 'adaptive')
  assert isinstance(alg.optimizer, torch.optim.Adam) and alg.learning_rate == 1e-3
  held = {id(p) for group in alg.optimizer.param_groups for p in group['params']}
  assert held == {id(p) for p in [*alg.actor.parameters(), *alg.critic.parameters()]}
  assert set(alg.storage.observations.keys()) == set(vec.get_observations().keys())
  assert alg.storage.observations['aug_code'].dtype == torch.long and alg.storage.observations['target_done'].dtype == torch.bool


def test_an_iteration_is_the_rollout_then_every_mini_batch(tmp_path):
  t = make_trainer(tmp_path)
  log = spy_env_step(t.vec)
  sizes = []
  t.actor.net.register_forward_pre_hook(lambda module, args: sizes.append((torch.is_grad_enabled(), args[0].nodes.shape[0])))
  rounds = t.rounds()
  steps = []
  optimizer_step = t.alg.optimizer.step

  def step():
    steps.append(1)
    return optimizer_step()

  t.alg.optimizer.step = step
  t.iteration()
  assert len(log) == 6 and t.rounds() == rounds + 6
  assert sizes[:6] == [(False, 8)] * 6
  assert sizes[6:] == [(True, 24)] * 4, 'mini-batch of 8 * 6 / 2, 2 epochs of 2'
  assert len(steps) == 4 and t.grad_steps() == 4
  assert t.counters.iterations == 1 and t.mini_batch_size() == 24


def test_one_iteration_trains_the_mean_the_std_and_the_critic_and_nothing_else(tmp_path):
  t = make_trainer(tmp_path)
  before = {name: p.clone() for name, p in [*t.actor.named_parameters(prefix='actor'), *t.critic.named_parameters(prefix='critic')]}
  t.iteration()
  after = dict([*t.actor.named_parameters(prefix='actor'), *t.critic.named_parameters(prefix='critic')])
  changed = {name for name, p in after.items() if not torch.equal(p, before[name])}
  # PolicyNet registers an update_global it never runs, DESIGN 11.9, so that one has no gradient.
  assert {name for name in after if name not in changed} == {name for name in after if name.startswith('actor.net.gnn.update_global')}
  assert 'actor.distribution.std_param' in changed
  assert t.alg.actor is t.actor and t.alg.critic is t.critic


def test_the_stagger_acts_uniformly_from_a_stream_of_its_own():
  seen = []
  original = BalletEnv.step

  def spy(self, action):
    seen.append(action.clone())
    return original(self, action)

  BalletEnv.step = spy
  try:
    vec = BalletVecEnv(ENV, 64, small_pool(), DEVICE, seed=2)
  finally:
    BalletEnv.step = original
  again = BalletVecEnv(ENV, 64, small_pool(), DEVICE, seed=2)
  assert len(seen) == vec.stagger_rounds == again.stagger_rounds
  actions = torch.stack(seen)
  assert actions.min() < -0.99 and actions.max() > 0.99
  assert actions.mean().abs() < 0.02 and (actions.std() - 3 ** -0.5).abs() < 0.01
  generator = torch.Generator(device=DEVICE).manual_seed(zlib.crc32(b'2/stagger'))
  assert torch.equal(actions[0], torch.rand(64, R, ACT, generator=generator, device=DEVICE) * 2.0 - 1.0)
  assert torch.equal(vec.env.observe().robot.q, again.env.observe().robot.q), 'one seed, one start'


def test_the_rollout_stores_the_observation_each_action_was_chosen_from():
  vec, runner = make_runner()
  seen = []
  act = runner.alg.act

  def spy(obs):
    seen.append({name: obs[name].clone() for name in obs.keys()})
    return act(obs)

  runner.alg.act = spy
  run_rollout(runner.alg, vec, 6)
  storage = runner.alg.storage
  assert len(seen) == 6
  for row, fields in enumerate(seen):
    for name, value in fields.items():
      assert torch.equal(storage.observations[name][row], value), (row, name)
  assert torch.equal(seen[1]['q'], storage.observations['q'][1]) and not torch.equal(seen[0]['q'], seen[1]['q'])


def test_the_learning_rate_is_adapted_and_logged_as_it_is(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=6, learning_rate=1e-3, num_mini_batches=4, num_learning_epochs=3)
  for _ in range(3):
    t.iteration()
  t.logger.flush()
  assert t.alg.learning_rate != 1e-3, 'adaptive'
  assert t.alg.optimizer.param_groups[0]['lr'] == t.alg.learning_rate
  rows = list(csv.DictReader(open(tmp_path / 'metrics.csv')))
  last = [float(row['value']) for row in rows if row['tag'] == 'train/policy_lr'][-1]
  assert last == t.alg.learning_rate
  assert [float(row['value']) for row in rows if row['tag'] == 'train/critic_lr'][-1] == t.alg.learning_rate
  assert [float(row['value']) for row in rows if row['tag'] == 'train/kl']


def test_a_report_writes_the_names_of_the_others_and_rsl_rl_terms(tmp_path):
  t = make_trainer(tmp_path, evaluator=small_evaluator(), profile=True, profile_every_rounds=12, profile_window_rounds=6, eval_every_rounds=6)
  t.profiler.first_window = 6
  for _ in range(5):
    t.iteration()
  t.logger.flush()
  tags = {row['tag'] for row in csv.DictReader(open(tmp_path / 'metrics.csv'))}
  for name in ('rate/env_steps_per_s', 'rate/grad_steps_per_s', 'rate/samples_per_s', 'counters/env_steps', 'counters/grad_steps', 'counters/rounds',
               'train/policy_lr', 'train/critic_lr', 'train/value_loss', 'train/surrogate_loss', 'train/entropy', 'train/action_std', 'train/kl',
               'train/explained_variance', 'train/mean_std_over_states', 'train/value_std_over_states',
               'train/returns_min', 'train/returns_max', 'act/policy_action_abs', 'act/policy_action_saturated', 'act/action_clipped',
               'eval/training_dist/task_rate', 'eval/training_dist/env_steps', 'eval/heldout/episode_length', 'eval/heldout/total_collision_robot_steps',
               'profile/env_step_ms', 'profile/policy_forward_ms', 'profile/value_forward_ms', 'profile/learner_ms', 'profile/learner_policy_forward_ms',
               'profile/feature_build_ms', 'profile/world_forward_ms', 'profile/mem_peak_gib', 'profile/iteration_s', 'profile/learn_s', 'profile/collect_s'):
    assert name in tags, name
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  logged = EventAccumulator(str(tmp_path / 'tb')).Reload().Tags()['scalars']
  assert 'train/policy_lr' in logged and 'eval/training_dist/task_rate' in logged and 'rate/env_steps_per_s' in logged


def logged(run_dir: Path, tag: str) -> list[float]:
  return [float(row['value']) for row in csv.DictReader(open(run_dir / 'metrics.csv')) if row['tag'] == tag]


def variance(xs: list[float]) -> float:
  mean = sum(xs) / len(xs)
  return sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


def test_a_report_reads_explained_variance_and_state_dependence_from_the_rollout(tmp_path):
  t = make_trainer(tmp_path, log_every_rounds=24)
  t.iteration()
  t.logger.flush()
  storage = t.alg.storage
  returns, values = storage.returns.flatten().tolist(), storage.values.flatten().tolist()
  means = storage.distribution_params[0].cpu()
  assert means.shape == (6, 8, R * ACT)
  assert logged(tmp_path, 'train/explained_variance') == pytest.approx([1 - variance([r - v for r, v in zip(returns, values)]) / variance(returns)], abs=1e-4)
  per_joint = [math.sqrt(variance(means[:, :, joint].flatten().tolist())) for joint in range(R * ACT)]
  assert logged(tmp_path, 'train/mean_std_over_states') == pytest.approx([sum(per_joint) / len(per_joint)], rel=1e-4)
  assert logged(tmp_path, 'train/value_std_over_states') == pytest.approx([math.sqrt(variance(values))], rel=1e-4)
  # A report is the mean of the iterations since the last one, and only of those.
  seen, due = [], t._next_report
  while t.rounds() < due:
    t.iteration()
    seen.append(t.rollout_health())
  t.logger.flush()
  assert len(seen) >= 2
  assert logged(tmp_path, 'train/mean_std_over_states')[1] == pytest.approx(torch.stack(seen)[:, 1].mean().item(), rel=1e-4)


def test_networks_that_do_not_depend_on_the_state_read_as_such(tmp_path):
  t = make_trainer(tmp_path)
  with torch.no_grad():
    t.actor.net.policy_head[-1].weight.zero_()
    t.critic.net.value[-1].weight.zero_()
  t.iteration()
  t.logger.flush()
  assert logged(tmp_path, 'train/mean_std_over_states') == pytest.approx([0.0], abs=1e-5)
  assert logged(tmp_path, 'train/value_std_over_states') == pytest.approx([0.0], abs=1e-5)
  assert logged(tmp_path, 'train/explained_variance') == pytest.approx([0.0], abs=1e-3)
  # All returns equal and a constant value would otherwise read as everything explained.
  t.alg.storage.returns.zero_()
  t.alg.storage.values.fill_(0.5)
  assert t.rollout_health()[0].item() == 0.0


# The review's measurement, smaller: the full model, 20 updates an iteration, at rsl-rl's adaptive defaults and at a low fixed rate.
# The rates are not a recommendation, and the first case stands whatever PPOCfg's defaults become.


@pytest.mark.parametrize('schedule, rate, alive', [('adaptive', 1e-3, False), ('fixed', 1e-4, True)])
def test_the_report_tells_a_full_model_that_kept_its_state_dependence_from_one_that_lost_it(tmp_path, schedule, rate, alive):
  iterations = FULL_MODEL_ITERATIONS
  torch.set_float32_matmul_precision('highest')
  cfg = small_cfg(num_worlds=FULL_MODEL_WORLDS, num_steps_per_env=24, num_learning_epochs=5, num_mini_batches=4, log_every_rounds=1, schedule=schedule, learning_rate=rate)
  t = Trainer(ENV, cfg, ModelCfg(), small_pool(), DEVICE, tmp_path)
  for _ in range(iterations):
    t.iteration()
  t.logger.flush()
  means, values = logged(tmp_path, 'train/mean_std_over_states'), logged(tmp_path, 'train/value_std_over_states')
  assert len(means) == len(values) == iterations
  if alive:  # Not the value as well: on seed 1 its std dips 60 times for one iteration at this rate and comes back.
    assert means[-1] > means[0] / 10, means
    assert t.alg.learning_rate == rate and logged(tmp_path, 'train/policy_lr') == [rate] * iterations
    assert not logged(tmp_path, 'train/kl'), 'no KL under a fixed schedule'
  else:
    assert means[-1] < means[0] / 100, means
    assert values[-1] < values[0] / 100, values


def test_a_rate_counts_every_world_and_the_counters_add_up(tmp_path):
  t = make_trainer(tmp_path)
  start = t.rounds()
  for _ in range(3):
    t.iteration()
  totals = t.totals()
  assert totals['rounds'] == start + 18 and totals['env_steps'] == (start + 18) * 8
  assert totals['grad_steps'] == 12 and totals['iterations'] == 3
  assert totals['discarded_rows'] == t.vec.discarded_rows > 0
  assert totals['residual_collisions'] == 0


def test_every_stream_has_its_own_seed(tmp_path):
  t = make_trainer(tmp_path)
  assert t._seed('env') != t._seed('torch') and t._seed('env') != t.cfg.seed


def test_the_global_generators_hold_their_own_seed_after_the_models_are_built(tmp_path):
  t = make_trainer(tmp_path)
  cuda, cpu = torch.cuda.get_rng_state().clone(), torch.get_rng_state().clone()
  torch.manual_seed(t._seed('torch'))
  assert torch.equal(torch.cuda.get_rng_state(), cuda) and torch.equal(torch.get_rng_state(), cpu)


def test_one_seed_gives_one_run_and_another_seed_another(tmp_path):
  """rsl-rl draws from the global generators, so those are seeded too. Dirtied here first, to see it."""
  runs = []
  for name, seed in (('a', 1), ('b', 1), ('c', 2)):
    torch.manual_seed(len(runs) * 1000 + 5)
    t = make_trainer(tmp_path / name, seed=seed)
    t.iteration()
    runs.append((flat(t.actor).clone(), flat(t.critic).clone()))
  assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])
  assert not torch.equal(runs[0][0], runs[2][0])


# Checkpoints.


def test_snapshots_hold_the_mean_for_load_policy(tmp_path):
  t = make_trainer(tmp_path, policy_snapshot_every_rounds=12)
  t._snapshot_if_due()
  first = flat(t.actor.net).clone()
  for _ in range(3):
    t.iteration()
  names = sorted(p.name for p in (tmp_path / POLICY_DIR).iterdir())
  assert names[0] == 'policy_00000000.pt' and len(names) >= 2
  policy = PolicyNet(SMALL, Topology(ENV, DEVICE))
  assert load_policy(tmp_path / POLICY_DIR / names[0], policy) == 0
  assert torch.equal(flat(policy).to(DEVICE), first)
  path = t.save_checkpoint('test')
  assert load_policy(path, policy) == t.grad_steps(), 'a checkpoint fits as well'
  assert torch.equal(flat(policy).to(DEVICE), flat(t.actor.net))
  state = torch.load(tmp_path / POLICY_DIR / names[-1], map_location='cpu', weights_only=True)
  assert set(state) == {'step', 'policy'}


def test_snapshot_period_can_be_turned_off(tmp_path):
  t = make_trainer(tmp_path, policy_snapshot_every_rounds=0)
  t.iteration()
  assert not (tmp_path / POLICY_DIR).exists()


def test_save_and_resume_restores_everything(tmp_path):
  first = make_trainer(tmp_path / 'a', evaluator=small_evaluator(), eval_every_rounds=6)
  for _ in range(4):
    first.iteration()
  path = first.save_checkpoint('test')
  state = torch.load(path, map_location='cpu', weights_only=True)
  assert state['format'] == 'ppo-1' and state['step'] == first.grad_steps() and state['failed'] is False
  assert state['learner']['optimizer_state_dict']['param_groups'][0]['lr'] == first.alg.learning_rate
  torch_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
  env_rng = first.vec.env.generator.get_state().clone()

  second = make_trainer(tmp_path / 'b', evaluator=small_evaluator(), eval_every_rounds=6)
  assert not torch.equal(flat(second.actor), flat(first.actor))
  fresh_rounds = second.vec.rounds
  second.load_checkpoint(path)
  assert torch.equal(flat(second.actor), flat(first.actor)) and torch.equal(flat(second.critic), flat(first.critic))
  assert second.alg.learning_rate == first.alg.learning_rate
  assert second.alg.optimizer.param_groups[0]['lr'] == first.alg.learning_rate
  for theirs, mine in zip(first.alg.optimizer.state.values(), second.alg.optimizer.state.values()):
    assert torch.equal(theirs['exp_avg'], mine['exp_avg']) and torch.equal(theirs['exp_avg_sq'], mine['exp_avg_sq'])
    assert theirs['step'] == mine['step']
  assert torch.equal(torch.get_rng_state(), torch_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
  assert torch.equal(second.vec.env.generator.get_state(), env_rng)
  assert second.counters.iterations == 4 and second.grad_steps() == first.grad_steps()
  assert second.rounds() == first.rounds() + fresh_rounds, 'a stagger is real env steps too'
  assert second.totals()['real_episodes'] == first.totals()['real_episodes']
  assert second._next_eval == first._next_eval
  assert second.runner.current_learning_iteration == first.runner.current_learning_iteration
  assert torch.equal(second.evaluator.state_dict()['heldout'], first.evaluator.state_dict()['heldout'])

  second.iteration()
  assert second.counters.iterations == 5 and second.rounds() == first.rounds() + fresh_rounds + 6
  assert not torch.equal(flat(second.actor), flat(first.actor)), 'and it learns from there'
  assert all(p.isfinite().all() for p in second.actor.parameters())


def test_two_resumes_of_one_checkpoint_are_the_same_run(tmp_path):
  first = make_trainer(tmp_path / 'a')
  first.iteration()
  path = first.save_checkpoint('test')
  results = []
  for name in ('b', 'c'):
    resumed = make_trainer(tmp_path / name)
    resumed.load_checkpoint(path)
    resumed.iteration()
    results.append((flat(resumed.actor).clone(), flat(resumed.critic).clone(), resumed.alg.learning_rate))
  assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
  assert results[0][2] == results[1][2]


def test_resume_refuses_a_changed_config_but_not_a_changed_cadence(tmp_path):
  first = make_trainer(tmp_path / 'a')
  first.iteration()
  path = first.save_checkpoint('test')
  with pytest.raises(ValueError, match='gamma'):
    make_trainer(tmp_path / 'b', gamma=0.9).load_checkpoint(path)
  with pytest.raises(ValueError, match='num_mini_batches'):
    make_trainer(tmp_path / 'c', num_mini_batches=4).load_checkpoint(path)
  make_trainer(tmp_path / 'd', total_env_rounds=999, eval_every_rounds=7, log_every_rounds=9).load_checkpoint(path)


def test_resume_refuses_another_agents_checkpoint(tmp_path):
  path = tmp_path / 'other.pt'
  torch.save({'format': 'fasttd3-1'}, path)
  with pytest.raises(ValueError, match='not a PPO checkpoint'):
    make_trainer(tmp_path / 'a').load_checkpoint(path)


def poison(trainer: Trainer, module) -> None:
  """A weight goes NaN in an update, as it would through an exploding gradient."""
  update = trainer.alg.update

  def bad_update():
    losses = update()
    with torch.no_grad():
      next(iter(module.parameters())).data[0] = float('nan')
    return losses

  trainer.alg.update = bad_update


@pytest.mark.parametrize('which', ['actor', 'critic'])
def test_a_nan_weight_saves_a_failed_checkpoint_and_stops_before_acting_on_it(tmp_path, which):
  t = make_trainer(tmp_path)
  poison(t, getattr(t, which))
  log = spy_env_step(t.vec)
  with pytest.raises(FloatingPointError):
    t.iteration()
  assert len(log) == 6, 'no action of the NaN weights reached the env'
  path = tmp_path / FAILED_CHECKPOINT_NAME
  assert path.exists() and not (tmp_path / CHECKPOINT_NAME).exists()
  state = torch.load(path, map_location='cpu', weights_only=True)
  assert state['failed'] is True and 'nonfinite' in state['reason']
  with pytest.raises(ValueError, match='failed checkpoint'):
    make_trainer(tmp_path / 'next').load_checkpoint(path)


@pytest.mark.parametrize('field', ['rewards', 'returns', 'actions'])
def test_nan_in_the_rollout_stops_too(tmp_path, field):
  t = make_trainer(tmp_path)
  update = t.alg.update

  def bad_update():
    losses = update()
    getattr(t.alg.storage, field)[3, 2] = float('nan')
    return losses

  t.alg.update = bad_update
  with pytest.raises(FloatingPointError):
    t.iteration()


def test_nan_in_a_stored_observation_stops_too(tmp_path):
  t = make_trainer(tmp_path)
  update = t.alg.update

  def bad_update():
    losses = update()
    t.alg.storage.observations['qd'][3, 2, 0, 0] = float('nan')
    return losses

  t.alg.update = bad_update
  with pytest.raises(FloatingPointError):
    t.iteration()


def test_a_nan_loss_also_stops(tmp_path):
  t = make_trainer(tmp_path)
  update = t.alg.update

  def bad_update():
    update()
    t.alg.last_losses = {'value': float('nan'), 'surrogate': 0.0, 'entropy': 0.0}
    return t.alg.last_losses

  t.alg.update = bad_update
  with pytest.raises(FloatingPointError):
    t.iteration()


def test_atomic_save_keeps_the_old_file_when_writing_fails(tmp_path):
  path = tmp_path / 'x.pt'
  atomic_save({'a': 1}, path)
  with pytest.raises(Exception):
    atomic_save({'a': lambda: 0}, path)
  assert torch.load(path, weights_only=True) == {'a': 1}
  assert [p.name for p in tmp_path.iterdir()] == ['x.pt']


def test_run_stops_on_a_signal_and_on_the_time_limit_and_saves(tmp_path):
  t = make_trainer(tmp_path / 'a', total_env_rounds=10**9)
  update = t.alg.update

  def signalled():
    losses = update()
    t._stop_signal = signal.SIGTERM
    return losses

  t.alg.update = signalled
  assert t.run() == f'signal {int(signal.SIGTERM)}'
  assert torch.load(tmp_path / 'a' / CHECKPOINT_NAME, map_location='cpu', weights_only=True)['reason'] == f'signal {int(signal.SIGTERM)}'
  assert t.counters.iterations == 1

  t = make_trainer(tmp_path / 'b', total_env_rounds=10**9)
  t.stop_after_s = 0.0
  assert t.run() == 'stop_after_seconds' and t.counters.iterations == 1

  t = make_trainer(tmp_path / 'c')
  object.__setattr__(t.cfg, 'total_env_rounds', t.rounds() + 7)
  assert t.run() == 'complete' and t.counters.iterations == 2, 'until the budget is passed'


def test_periodic_checkpoint(tmp_path):
  t = make_trainer(tmp_path, checkpoint_every_s=0.0)
  t.iteration()
  assert t._poll() is None
  assert (tmp_path / CHECKPOINT_NAME).exists()


def test_the_first_evaluation_and_snapshot_see_the_untrained_policy(tmp_path):
  t = make_trainer(tmp_path, evaluator=small_evaluator(), total_env_rounds=0, policy_snapshot_every_rounds=12)
  t.run()
  assert (tmp_path / POLICY_DIR / 'policy_00000000.pt').exists()
  rows = [row for row in csv.DictReader(open(tmp_path / 'metrics.csv')) if row['tag'] == 'eval/training_dist/network_step']
  assert [float(row['value']) for row in rows] == [0.0]


# The script.


def script_env() -> dict:
  """PYTHONPATH as it came, so an overlay of the mutation check reaches the subprocess too."""
  return dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH', str(REPO / 'src')), PYTHONUNBUFFERED='1')


def small_pool_file(path: Path) -> Path:
  from roboballet_mjlab.scenes.io import Pool, save_pool
  from roboballet_mjlab.scenes.sampler import Witnesses

  save_pool(path, Pool(small_pool(), Witnesses(torch.zeros(1), torch.zeros(1)), {'test': True}))
  return path


def script_args(run_dir: Path, pool: Path, *extra: str) -> list[str]:
  sets = ['num_worlds=8', 'num_steps_per_env=6', 'num_learning_epochs=2', 'num_mini_batches=2', 'compile=False', 'log_every_rounds=12',
          'eval_every_rounds=30', 'eval_episodes=8', 'profile=False', 'policy_snapshot_every_rounds=30']
  models = ['node_layers=2', 'node_size=16', 'edge_embed_layers=2', 'edge_update_layers=2', 'edge_size=8', 'global_layers=2', 'head_sizes=8']
  args = [sys.executable, '-m', 'roboballet_mjlab.ppo.train', '--run-dir', str(run_dir), '--train-pool', str(pool)]
  for item in sets:
    args += ['--set', item]
  for item in models:
    args += ['--model-set', item]
  return args + list(extra)


def wait_for(path: Path, process: subprocess.Popen, timeout: float = 300.0) -> None:
  deadline = time.monotonic() + timeout
  while not path.exists():
    assert process.poll() is None, f'exited early with {process.returncode}'
    assert time.monotonic() < deadline, f'no {path}'
    time.sleep(0.5)


def test_script_records_provenance_stops_on_sigterm_and_resumes(tmp_path):
  pool = small_pool_file(tmp_path / 'pool.pt')
  run_dir = tmp_path / 'run'
  log = open(tmp_path / 'log.txt', 'w')
  process = subprocess.Popen(script_args(run_dir, pool), env=script_env(), stdout=log, stderr=subprocess.STDOUT)
  try:
    wait_for(run_dir / 'tb', process)
    deadline = time.monotonic() + 300
    while b'counters/rounds' not in (run_dir / 'metrics.csv').read_bytes():  # Past the first report.
      assert process.poll() is None and time.monotonic() < deadline
      time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=120) == 0
  finally:
    process.kill()
  state = torch.load(run_dir / CHECKPOINT_NAME, map_location='cpu', weights_only=True)
  assert state['format'] == 'ppo-1' and state['reason'] == f'signal {int(signal.SIGTERM)}'
  provenance = next((run_dir / 'provenance').iterdir())
  assert {p.name for p in provenance.iterdir()} >= {'config.json', 'git_revision.txt', 'git_diff.patch', 'git_untracked.txt', 'system.json'}
  config = json.loads((provenance / 'config.json').read_text())
  assert config['train']['gamma'] == 0.94 and config['model']['node_size'] == 16
  summary = json.loads((run_dir / 'summary.json').read_text())
  assert summary['rounds'] == state['counters']['rounds'] and summary['iterations'] >= 1
  assert (run_dir / POLICY_DIR / 'policy_00000000.pt').exists()

  done = subprocess.run(script_args(run_dir, pool, '--resume', '--total-env-rounds', str(summary['rounds'] + 50)), env=script_env(),
                        capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  assert 'resumed from' in done.stdout
  later = json.loads((run_dir / 'summary.json').read_text())
  assert later['iterations'] > summary['iterations'] and later['rounds'] >= summary['rounds'] + 6
  assert len(list((run_dir / 'provenance').iterdir())) == 2

  done = subprocess.run(script_args(run_dir, pool, '--eval-only', str(run_dir / CHECKPOINT_NAME)), env=script_env(), capture_output=True, text=True, timeout=300)
  assert done.returncode == 0, done.stderr[-2000:]
  results = json.loads(done.stdout[done.stdout.index('{\n'):])
  assert results['results']['training_dist']['worlds'] == 8.0


def test_end_to_end_compiled_small_model(tmp_path):
  """As training runs: compiled, profiler on. Finite, counters agree, memory flat, and the env keeps working outside inference mode."""
  t = make_trainer(tmp_path, compile=True, profile=True, num_worlds=16, num_steps_per_env=8, profile_every_rounds=32, profile_window_rounds=16, log_every_rounds=16,
                   evaluator=None)
  t.profiler.first_window = 16
  for _ in range(4):
    t.iteration()
  memory = torch.cuda.memory_allocated()
  for _ in range(4):
    t.iteration()
  t.logger.flush()
  assert torch.cuda.memory_allocated() - memory < 32 * 2**20
  assert t.grad_steps() == 8 * 4 and t.counters.iterations == 8
  assert all(p.isfinite().all() for p in [*t.actor.parameters(), *t.critic.parameters()])
  values: dict[str, float] = {}
  for row in csv.DictReader(open(tmp_path / 'metrics.csv')):
    values[row['tag']] = float(row['value'])
  assert values['profile/window_round_ms'] > 0 and values['profile/learner_policy_forward_ms'] > 0 and values['profile/env_step_ms'] > 0
  assert values['counters/rounds'] <= t.rounds() and math.isfinite(values['train/value_loss'])
  assert torch.get_float32_matmul_precision() == 'highest'
  # Rows of the runner's loop live in inference mode. A reset and an evaluation of this policy must still work after it.
  t.vec.env.reset(torch.tensor([0, 1], device=DEVICE))
  evaluator = Evaluator(ENV, SMALL, {'training_dist': small_pool()}, DEVICE, 1, num_worlds=4, compile=True)
  assert 'training_dist' in evaluator.evaluate(t.actor.net, t.grad_steps())
