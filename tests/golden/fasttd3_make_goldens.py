"""Generates tests/golden/fasttd3_*.npz from a verbatim excerpt of the FastTD3 reference.

Source: https://github.com/younggyoseo/FastTD3 at commit 229ed59, files fast_td3/fast_td3.py,
fast_td3/train.py and fast_td3/hyperparams.py, copyright (c) 2025 Younggyo Seo. Blocks between
BEGIN VERBATIM and END VERBATIM are copied, shifted only by a uniform change of indentation, except
that two say on their BEGIN line what they replace. Everything else here is harness. The tests read the
stored values and never import from a clone or from this file.

The reference's LICENSE opens with "This software is part of the BAIR Commons HIC Repository as of
calendar year 2025." and also bundles the notices of code it builds on: LeanRL under MIT, Copyright (c)
2024 LeanRL developers, as its README.md:307 says, and others. docs/LICENSES.md item 4 ships that whole
file at release. The notice for this excerpt:

MIT License

Copyright (c) 2025 Younggyo Seo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Run on the CPU, from the repository root:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/golden/fasttd3_make_goldens.py
The bf16 trace needs a GPU, because under CPU autocast the reference's projection raises:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python tests/golden/fasttd3_make_goldens.py bf16
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

OUT = Path(__file__).resolve().parent
device = torch.device('cpu')

# Values of BaseArgs in fast_td3/hyperparams.py, written as a dict.
REFERENCE_DEFAULTS = dict(
  critic_learning_rate=3e-4,
  actor_learning_rate=3e-4,
  num_steps=1,
  tau=0.1,
  batch_size=32768,
  policy_noise=0.001,
  std_min=0.001,
  std_max=0.4,
  learning_starts=10,
  policy_frequency=2,
  noise_clip=0.5,
  num_updates=2,
  num_atoms=101,
  use_cdq=True,
  use_grad_norm_clipping=False,
  amp=True,
  weight_decay=0.1,
)


class DistributionalQNetwork(nn.Module):
  def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
    """fast_td3.py:58-101 without the sim_type branch, which is off by default."""
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_obs + n_act, hidden_dim, device=device),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2, device=device),
        nn.ReLU(),
    )
    self.fc_head = nn.Sequential(
        nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
        nn.ReLU(),
        nn.Linear(hidden_dim // 4, num_atoms, device=device),
    )
    self.v_min = v_min
    self.v_max = v_max
    self.num_atoms = num_atoms

  # BEGIN VERBATIM fast_td3/fast_td3.py:103-154
  def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
      x = torch.cat([obs, actions], 1)
      x = self.net(x)
      x = self.fc_head(x)
      return x

  def projection(
      self,
      obs: torch.Tensor,
      actions: torch.Tensor,
      rewards: torch.Tensor,
      bootstrap: torch.Tensor,
      discount: float,
      q_support: torch.Tensor,
      device: torch.device,
  ) -> torch.Tensor:
      delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
      batch_size = rewards.shape[0]

      target_z = (
          rewards.unsqueeze(1)
          + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
      )
      target_z = target_z.clamp(self.v_min, self.v_max)
      b = (target_z - self.v_min) / delta_z
      l = torch.floor(b).long()
      u = torch.ceil(b).long()

      is_int = l == u
      l_mask = is_int & (l > 0)
      u_mask = is_int & (l == 0)

      l = torch.where(l_mask, l - 1, l)
      u = torch.where(u_mask, u + 1, u)

      next_dist = F.softmax(self.forward(obs, actions), dim=1)
      proj_dist = torch.zeros_like(next_dist)
      offset = (
          torch.linspace(
              0, (batch_size - 1) * self.num_atoms, batch_size, device=device
          )
          .unsqueeze(1)
          .expand(batch_size, self.num_atoms)
          .long()
      )
      proj_dist.view(-1).index_add_(
          0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
      )
      proj_dist.view(-1).index_add_(
          0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
      )
      return proj_dist
  # END VERBATIM


class Critic(nn.Module):
  def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device=None):
    """fast_td3.py:158-200 without the sim_type arguments."""
    super().__init__()
    self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
    self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
    self.register_buffer(
        "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
    )
    self.device = device

  # BEGIN VERBATIM fast_td3/fast_td3.py:202-236
  def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
      return self.qnet1(obs, actions), self.qnet2(obs, actions)

  def projection(
      self,
      obs: torch.Tensor,
      actions: torch.Tensor,
      rewards: torch.Tensor,
      bootstrap: torch.Tensor,
      discount: float,
  ) -> torch.Tensor:
      """Projection operation that includes q_support directly"""
      q1_proj = self.qnet1.projection(
          obs,
          actions,
          rewards,
          bootstrap,
          discount,
          self.q_support,
          self.q_support.device,
      )
      q2_proj = self.qnet2.projection(
          obs,
          actions,
          rewards,
          bootstrap,
          discount,
          self.q_support,
          self.q_support.device,
      )
      return q1_proj, q2_proj

  def get_value(self, probs: torch.Tensor) -> torch.Tensor:
      """Calculate value from logits using support"""
      return torch.sum(probs * self.q_support, dim=1)
  # END VERBATIM


class Actor(nn.Module):
  def __init__(self, n_obs, n_act, num_envs, init_scale, hidden_dim, std_min=0.05, std_max=0.8, device=None):
    """fast_td3.py:240-296 without the sim_type branch."""
    super().__init__()
    self.n_act = n_act
    self.net = nn.Sequential(
        nn.Linear(n_obs, hidden_dim, device=device),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2, device=device),
        nn.ReLU(),
    )
    self.fc_head = nn.Sequential(
        nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
        nn.ReLU(),
    )
    self.fc_mu = nn.Sequential(
        nn.Linear(hidden_dim // 4, n_act, device=device),
        nn.Tanh(),
    )
    nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
    nn.init.constant_(self.fc_mu[0].bias, 0.0)

    # BEGIN VERBATIM fast_td3/fast_td3.py:288-296
    noise_scales = (
        torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
    )
    self.register_buffer("noise_scales", noise_scales)

    self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
    self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
    self.n_envs = num_envs
    self.device = device
    # END VERBATIM

  # BEGIN VERBATIM fast_td3/fast_td3.py:298-328
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
      x = obs
      x_net = self.net(x)
      x_head = self.fc_head(x_net)
      action = self.fc_mu(x_head)
      return action

  def explore(
      self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
  ) -> torch.Tensor:
      # If dones is provided, resample noise for environments that are done
      if dones is not None and dones.sum() > 0:
          # Generate new noise scales for done environments (one per environment)
          new_scales = (
              torch.rand(self.n_envs, 1, device=obs.device)
              * (self.std_max - self.std_min)
              + self.std_min
          )

          # Update only the noise scales for environments that are done
          dones_view = dones.view(-1, 1) > 0
          self.noise_scales.copy_(
              torch.where(dones_view, new_scales, self.noise_scales)
          )

      act = self(obs)
      if deterministic:
          return act

      noise = torch.randn_like(act) * self.noise_scales
      return act + noise
  # END VERBATIM


def make_update_fns(args, envs, qnet, qnet_target, actor, q_optimizer, actor_optimizer, scaler, amp_device_type, amp_dtype,
                    amp_enabled, policy_noise, noise_clip, action_low, action_high):
  """The closures of train.py:406-533, with the closure's names as arguments."""

  # BEGIN VERBATIM fast_td3/train.py:406-533
  def update_main(data, logs_dict):
      with autocast(
          device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
      ):
          observations = data["observations"]
          next_observations = data["next"]["observations"]
          if envs.asymmetric_obs:
              critic_observations = data["critic_observations"]
              next_critic_observations = data["next"]["critic_observations"]
          else:
              critic_observations = observations
              next_critic_observations = next_observations
          actions = data["actions"]
          rewards = data["next"]["rewards"]
          dones = data["next"]["dones"].bool()
          truncations = data["next"]["truncations"].bool()
          if args.disable_bootstrap:
              bootstrap = (~dones).float()
          else:
              bootstrap = (truncations | ~dones).float()

          clipped_noise = torch.randn_like(actions)
          clipped_noise = clipped_noise.mul(policy_noise).clamp(
              -noise_clip, noise_clip
          )

          next_state_actions = (actor(next_observations) + clipped_noise).clamp(
              action_low, action_high
          )
          discount = args.gamma ** data["next"]["effective_n_steps"]

          with torch.no_grad():
              qf1_next_target_projected, qf2_next_target_projected = (
                  qnet_target.projection(
                      next_critic_observations,
                      next_state_actions,
                      rewards,
                      bootstrap,
                      discount,
                  )
              )
              qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
              qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
              if args.use_cdq:
                  qf_next_target_dist = torch.where(
                      qf1_next_target_value.unsqueeze(1)
                      < qf2_next_target_value.unsqueeze(1),
                      qf1_next_target_projected,
                      qf2_next_target_projected,
                  )
                  qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
              else:
                  qf1_next_target_dist, qf2_next_target_dist = (
                      qf1_next_target_projected,
                      qf2_next_target_projected,
                  )

          qf1, qf2 = qnet(critic_observations, actions)
          qf1_loss = -torch.sum(
              qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
          ).mean()
          qf2_loss = -torch.sum(
              qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
          ).mean()
          qf_loss = qf1_loss + qf2_loss

      q_optimizer.zero_grad(set_to_none=True)
      scaler.scale(qf_loss).backward()
      scaler.unscale_(q_optimizer)

      if args.use_grad_norm_clipping:
          critic_grad_norm = torch.nn.utils.clip_grad_norm_(
              qnet.parameters(),
              max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
          )
      else:
          critic_grad_norm = torch.tensor(0.0, device=device)
      scaler.step(q_optimizer)
      scaler.update()

      logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
      logs_dict["qf_loss"] = qf_loss.detach()
      logs_dict["qf_max"] = qf1_next_target_value.max().detach()
      logs_dict["qf_min"] = qf1_next_target_value.min().detach()
      return logs_dict

  def update_pol(data, logs_dict):
      with autocast(
          device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
      ):
          critic_observations = (
              data["critic_observations"]
              if envs.asymmetric_obs
              else data["observations"]
          )

          qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
          qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
          qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
          if args.use_cdq:
              qf_value = torch.minimum(qf1_value, qf2_value)
          else:
              qf_value = (qf1_value + qf2_value) / 2.0
          actor_loss = -qf_value.mean()

      actor_optimizer.zero_grad(set_to_none=True)
      scaler.scale(actor_loss).backward()
      scaler.unscale_(actor_optimizer)
      if args.use_grad_norm_clipping:
          actor_grad_norm = torch.nn.utils.clip_grad_norm_(
              actor.parameters(),
              max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
          )
      else:
          actor_grad_norm = torch.tensor(0.0, device=device)
      scaler.step(actor_optimizer)
      scaler.update()
      logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
      logs_dict["actor_loss"] = actor_loss.detach()
      return logs_dict

  @torch.no_grad()
  def soft_update(src, tgt, tau: float):
      src_ps = [p.data for p in src.parameters()]
      tgt_ps = [p.data for p in tgt.parameters()]

      torch._foreach_mul_(tgt_ps, 1.0 - tau)
      torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
  # END VERBATIM

  return update_main, update_pol, soft_update


def run_schedule(args, global_steps, rb, normalize_obs, normalize_reward, update_main, update_pol, soft_update, qnet, qnet_target,
                 logs_dict, envs, env_type, on_step=lambda global_step: None):
  """The loop of train.py:578, :643-676, with the env step left out."""
  for global_step in range(global_steps):
    on_step(global_step)
    # BEGIN VERBATIM fast_td3/train.py:643-676
    if global_step > args.learning_starts:
        for i in range(args.num_updates):
            data = rb.sample(max(1, args.batch_size // args.num_envs))
            data["observations"] = normalize_obs(data["observations"])
            data["next"]["observations"] = normalize_obs(
                data["next"]["observations"]
            )
            if envs.asymmetric_obs:
                data["critic_observations"] = normalize_critic_obs(
                    data["critic_observations"]
                )
                data["next"]["critic_observations"] = normalize_critic_obs(
                    data["next"]["critic_observations"]
                )
            raw_rewards = data["next"]["rewards"]
            if env_type in ["mtbench"] and args.reward_normalization:
                # Multi-task reward normalization
                task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                data["next"]["rewards"] = normalize_reward(
                    raw_rewards, task_ids=task_indices
                )
            else:
                data["next"]["rewards"] = normalize_reward(raw_rewards)

            logs_dict = update_main(data, logs_dict)
            if args.num_updates > 1:
                if i % args.policy_frequency == 1:
                    logs_dict = update_pol(data, logs_dict)
            else:
                if global_step % args.policy_frequency == 0:
                    logs_dict = update_pol(data, logs_dict)

            soft_update(qnet, qnet_target, args.tau)
    # END VERBATIM


# Harness from here on.


def flat(module: nn.Module) -> np.ndarray:
  return torch.cat([p.detach().flatten() for p in module.parameters()]).cpu().numpy()


def identity(x):
  return x


def projection_cases() -> dict[str, np.ndarray]:
  """Edge cases of the projection: atoms on, at and past the support, and terminals."""
  out: dict[str, np.ndarray] = {}
  supports = [(-1.0, 1.0, 101), (-2.1, 1.0, 101), (-10.0, 10.0, 101), (-250.0, 250.0, 101), (-1.0, 1.0, 51), (-1.0, 1.0, 11), (-0.5, 1.5, 5)]
  discounts = [0.94, 0.99, 1.0, 0.0]
  case = 0
  for v_min, v_max, atoms in supports:
    for gamma in discounts:
      torch.manual_seed(1000 + case)
      span = v_max - v_min
      delta = span / (atoms - 1)
      rewards = torch.tensor([
        0.0, 0.1, -0.125, 0.2,                  # Ordinary steps.
        v_min, v_max,                           # Exactly the ends: b is 0 or atoms - 1 when terminal.
        v_min + delta, v_max - delta,           # On an atom next to an end.
        v_min + 7 * delta if atoms > 8 else v_min + delta,  # On an interior atom.
        v_min - 5 * span, v_max + 5 * span,     # Everything past the support.
        v_min - 0.25 * span, v_max + 0.25 * span,  # Some atoms past it.
        0.5 * delta, -0.5 * delta, 1e-7,        # Between atoms, and nearly on one.
      ], dtype=torch.float32)
      rewards = torch.cat([rewards, rewards])
      n = rewards.shape[0]
      # Each reward once bootstrapping, once terminal.
      bootstrap = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
      logits = torch.randn(n, atoms) * 3.0
      logits[0] = 0.0                           # Uniform.
      logits[1] = -100.0
      logits[1, 0] = 100.0                      # All mass on the first atom.
      logits[2] = -100.0
      logits[2, -1] = 100.0                     # All mass on the last atom.

      class Fixed(DistributionalQNetwork):
        def forward(self, obs, actions):
          return logits

      net = Fixed(1, 1, atoms, v_min, v_max, 4)
      support = torch.linspace(v_min, v_max, atoms)
      discount = torch.full((n,), gamma)
      proj = net.projection(None, None, rewards, bootstrap, discount, support, device).detach()
      prefix = f'case{case}_'
      out.update({
        f'{prefix}logits': logits.numpy(), f'{prefix}rewards': rewards.numpy(), f'{prefix}bootstrap': bootstrap.numpy(),
        f'{prefix}proj': proj.numpy(), f'{prefix}support': support.numpy(),
        f'{prefix}params': np.array([v_min, v_max, atoms, gamma], dtype=np.float64),
      })
      case += 1
  out['num_cases'] = np.array(case)
  return out


class FixedBuffer:
  """rb.sample of train.py:645, handing out the batches a test can read back."""

  def __init__(self, batches):
    self.batches, self.next = batches, 0

  def sample(self, n):
    batch = self.batches[self.next]
    self.next += 1
    return {'observations': batch['observations'], 'actions': batch['actions'],
            'next': {**batch['next']}}


def update_trace(name: str, seed: int, amp: bool = False, **changes) -> dict[str, np.ndarray]:
  """Reference nets through the reference's loop, after the 12th env step, with every result kept.
  With amp, on the GPU under bf16 autocast at train.py:45's matmul precision."""
  global device
  device = torch.device('cuda' if amp else 'cpu')  # The verbatim code reads it.
  if amp:
    torch.set_float32_matmul_precision('high')
  torch.manual_seed(seed)
  n_obs, n_act, hidden, batch = 6, 3, 32, 64
  args = SimpleNamespace(
    gamma=0.94, disable_bootstrap=True, use_cdq=True, use_grad_norm_clipping=False, max_grad_norm=0.0, tau=0.1,
    num_updates=2, policy_frequency=2, learning_starts=10, batch_size=batch, num_envs=1, reward_normalization=False,
    critic_learning_rate=3e-4, actor_learning_rate=3e-4, weight_decay=0.1, policy_noise=0.001, noise_clip=0.5,
    v_min=-1.0, v_max=1.0, num_atoms=101)
  for key, value in changes.items():
    setattr(args, key, value)
  actor = Actor(n_obs, n_act, 4, changes.get('init_scale', 0.01), hidden, 0.001, 0.4).to(device)
  critic_kwargs = dict(n_obs=n_obs, n_act=n_act, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max, hidden_dim=hidden)
  qnet = Critic(**critic_kwargs).to(device)
  qnet_target = Critic(**critic_kwargs).to(device)
  qnet_target.load_state_dict(qnet.state_dict())
  # train.py:291-300.
  q_optimizer = optim.AdamW(list(qnet.parameters()), lr=torch.tensor(args.critic_learning_rate, device=device), weight_decay=args.weight_decay)
  actor_optimizer = optim.AdamW(list(actor.parameters()), lr=torch.tensor(args.actor_learning_rate, device=device), weight_decay=args.weight_decay)
  scaler = GradScaler(enabled=False)
  envs = SimpleNamespace(asymmetric_obs=False)
  update_main, update_pol, soft_update = make_update_fns(
    args, envs, qnet, qnet_target, actor, q_optimizer, actor_optimizer, scaler, device.type, torch.bfloat16, amp,
    args.policy_noise, args.noise_clip, -1.0, 1.0)

  out: dict[str, np.ndarray] = {
    'init_actor': flat(actor), 'init_q1': flat(qnet.qnet1), 'init_q2': flat(qnet.qnet2),
  }
  rounds = changes.get('global_steps', 16)
  updates = (rounds - args.learning_starts - 1) * args.num_updates
  batches = []
  for _ in range(updates):
    obs = torch.randn(batch, n_obs)
    next_obs = torch.randn(batch, n_obs)
    actions = torch.rand(batch, n_act) * 2 - 1
    rewards = (torch.rand(batch) * 0.4 - 0.2) * (torch.rand(batch) < 0.5)
    dones = torch.rand(batch) < 0.25
    truncations = dones & (torch.rand(batch) < 0.5)  # Timeouts must not bootstrap.
    batches.append({'observations': obs.to(device), 'actions': actions.to(device), 'next': {
      'observations': next_obs.to(device), 'rewards': rewards.to(device), 'dones': dones.long().to(device),
      'truncations': truncations.long().to(device),
      'effective_n_steps': torch.ones(batch, dtype=torch.long, device=device)}})  # Long, fast_td3_utils.py:156.
  log: dict[str, list] = {k: [] for k in ('normals', 'qf_loss', 'qf_max', 'qf_min', 'actor_loss', 'actor', 'q1', 'q2', 'q1_target', 'q2_target')}

  counter = {'main': 0}

  def recorded_main(data, logs_dict):
    seed_now = 90_000 + counter['main']
    counter['main'] += 1
    torch.manual_seed(seed_now)
    log['normals'].append(torch.randn_like(data['actions']).cpu().numpy())
    torch.manual_seed(seed_now)
    logs_dict = update_main(data, logs_dict)
    log['qf_loss'].append(logs_dict['qf_loss'].item())
    log['qf_max'].append(logs_dict['qf_max'].item())
    log['qf_min'].append(logs_dict['qf_min'].item())
    logs_dict.pop('actor_loss', None)
    return logs_dict

  def recorded_soft(src, tgt, tau):
    soft_update(src, tgt, tau)
    log['actor'].append(flat(actor))
    log['q1'].append(flat(qnet.qnet1))
    log['q2'].append(flat(qnet.qnet2))
    log['q1_target'].append(flat(qnet_target.qnet1))
    log['q2_target'].append(flat(qnet_target.qnet2))

  logs_dict: dict[str, torch.Tensor] = {}

  def recorded_pol(data, logs_dict):
    logs_dict = update_pol(data, logs_dict)
    log['actor_loss'].append(logs_dict['actor_loss'].item())
    return logs_dict

  run_schedule(args, rounds, FixedBuffer(batches), identity, identity, recorded_main, recorded_pol, recorded_soft, qnet, qnet_target,
               logs_dict, envs, 'x')
  stack = lambda key: np.array(log[key])
  out.update({key: stack(key) for key in ('normals', 'qf_loss', 'qf_max', 'qf_min', 'actor_loss', 'actor', 'q1', 'q2', 'q1_target', 'q2_target')})
  for name_, values in (('observations', [b['observations'] for b in batches]), ('actions', [b['actions'] for b in batches]),
                        ('next_observations', [b['next']['observations'] for b in batches]),
                        ('rewards', [b['next']['rewards'] for b in batches]), ('dones', [b['next']['dones'] for b in batches])):
    out[f'batch_{name_}'] = torch.stack(values).cpu().numpy()
  scalars = {key: value for key, value in vars(args).items() if isinstance(value, (bool, int, float))}
  out.update({f'arg_{key}': np.array(value) for key, value in scalars.items()})
  out['hidden'] = np.array(hidden)
  return {f'{name}_{key}': value for key, value in out.items()}


def loss_cases() -> dict[str, np.ndarray]:
  """The clipped double Q choice and both losses on stored inputs, including ties and a clear winner each way."""
  torch.manual_seed(21)
  out: dict[str, np.ndarray] = {}
  batch, atoms = 256, 101
  support = torch.linspace(-1.0, 1.0, atoms)
  qnet_target = SimpleNamespace(get_value=lambda probs: torch.sum(probs * support, dim=1))
  qnet = qnet_target
  qf1_next_target_projected = torch.softmax(torch.randn(batch, atoms) * 2.0, 1)
  qf2_next_target_projected = torch.softmax(torch.randn(batch, atoms) * 2.0, 1)
  qf2_next_target_projected[:8] = qf1_next_target_projected[:8]  # Ties of the same distribution.
  # Ties of different distributions: half on each of two atoms symmetric about zero, exactly in f32.
  qf1_next_target_projected[8:12] = 0.0
  qf1_next_target_projected[8:12, 25] = qf1_next_target_projected[8:12, 75] = 0.5
  qf2_next_target_projected[8:12] = 0.0
  qf2_next_target_projected[8:12, 30] = qf2_next_target_projected[8:12, 70] = 0.5
  qf1 = torch.randn(batch, atoms) * 2.0
  qf2 = torch.randn(batch, atoms) * 2.0
  for use_cdq in (True, False):
    args = SimpleNamespace(use_cdq=use_cdq)
    # BEGIN VERBATIM fast_td3/train.py:447-461
    qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
    qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
    if args.use_cdq:
        qf_next_target_dist = torch.where(
            qf1_next_target_value.unsqueeze(1)
            < qf2_next_target_value.unsqueeze(1),
            qf1_next_target_projected,
            qf2_next_target_projected,
        )
        qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
    else:
        qf1_next_target_dist, qf2_next_target_dist = (
            qf1_next_target_projected,
            qf2_next_target_projected,
        )
    # END VERBATIM
    # BEGIN VERBATIM fast_td3/train.py:464-470
    qf1_loss = -torch.sum(
        qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
    ).mean()
    qf2_loss = -torch.sum(
        qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
    ).mean()
    qf_loss = qf1_loss + qf2_loss
    # END VERBATIM
    # BEGIN VERBATIM fast_td3/train.py:503-509
    qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
    qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
    if args.use_cdq:
        qf_value = torch.minimum(qf1_value, qf2_value)
    else:
        qf_value = (qf1_value + qf2_value) / 2.0
    actor_loss = -qf_value.mean()
    # END VERBATIM
    name = 'cdq' if use_cdq else 'no_cdq'
    out[f'{name}_dist1'], out[f'{name}_dist2'] = qf1_next_target_dist.numpy(), qf2_next_target_dist.numpy()
    out[f'{name}_qf_loss'], out[f'{name}_actor_loss'] = qf_loss.numpy(), actor_loss.numpy()
  out['proj1'], out['proj2'] = qf1_next_target_projected.numpy(), qf2_next_target_projected.numpy()
  out['value1'], out['value2'] = qf1_next_target_value.numpy(), qf2_next_target_value.numpy()
  out['logits1'], out['logits2'], out['support'] = qf1.numpy(), qf2.numpy(), support.numpy()
  return out


def noise_cases() -> dict[str, np.ndarray]:
  """train.py:427-430 at several sizes, and fast_td3.py:288-328 over a run of dones."""
  torch.manual_seed(7)
  out: dict[str, np.ndarray] = {}
  normals = torch.randn(4096, 7)
  normals[:8, 0] = torch.tensor([0.0, 0.3, 0.49, 0.5, 0.51, 3.0, -0.7, -6.0])
  sizes = [(0.001, 0.5), (0.2, 0.5), (1.0, 0.5), (2.0, 0.5), (0.2, 0.05)]
  for k, (policy_noise, noise_clip) in enumerate(sizes):
    # BEGIN VERBATIM fast_td3/train.py:427-430, with normals, drawn above, in place of torch.randn_like(actions).
    clipped_noise = normals
    clipped_noise = clipped_noise.mul(policy_noise).clamp(
        -noise_clip, noise_clip
    )
    # END VERBATIM
    out[f'clipped_{k}'] = clipped_noise.numpy()
    out[f'clipped_{k}_args'] = np.array([policy_noise, noise_clip])
  out['normals'] = normals.numpy()
  out['num_sizes'] = np.array(len(sizes))

  num_envs, n_act = 16, 7
  actor = Actor(5, n_act, num_envs, 0.01, 32, 0.001, 0.4)
  out['initial_scales'] = actor.noise_scales.numpy().copy()
  obs = torch.randn(num_envs, 5)
  steps = []
  for t in range(8):
    dones = torch.rand(num_envs) < 0.3 if t != 2 else torch.zeros(num_envs, dtype=torch.bool)
    before = actor.noise_scales.clone()
    state = torch.get_rng_state()
    fresh = torch.rand(num_envs, 1) if dones.sum() > 0 else torch.full((num_envs, 1), float('nan'))
    # explore draws its normal after any scale draw, from the same stream.
    normal = torch.randn(num_envs, n_act)
    torch.set_rng_state(state)
    action = actor.explore(obs, dones=dones).detach()
    steps.append((dones.numpy(), before.numpy(), fresh.numpy(), normal.numpy(), actor(obs).detach().numpy(), actor.noise_scales.numpy().copy(), action.numpy()))
  for key, value in zip(('dones', 'before', 'fresh', 'normal', 'mean', 'after', 'action'), zip(*steps)):
    out[f'explore_{key}'] = np.array(value)
  return out


def schedule_cases() -> dict[str, np.ndarray]:
  """Which calls the verbatim loop makes, in order: (global_step, kind). Kind 0 sample, 1 critic, 2 actor, 3 target."""
  out: dict[str, np.ndarray] = {}
  for name, changes in (('default', {}), ('one_update', {'num_updates': 1}), ('three_updates', {'num_updates': 3}),
                        ('frequency_three', {'num_updates': 6, 'policy_frequency': 3})):
    args = SimpleNamespace(**{**dict(learning_starts=10, num_updates=2, policy_frequency=2, batch_size=2048, num_envs=1024,
                                     tau=0.1, reward_normalization=False), **changes})
    calls: list[tuple[int, int]] = []
    now = {'step': 0}

    class Recording:
      def sample(self, n):
        calls.append((now['step'], 0))
        return {'observations': None, 'next': {'observations': None, 'rewards': None}}

    def recorder(kind):
      def record(*_):
        calls.append((now['step'], kind))
        return _[1] if kind != 3 else None
      return record

    run_schedule(args, 40, Recording(), identity, identity, recorder(1), recorder(2), recorder(3), None, None, {},
                 SimpleNamespace(asymmetric_obs=False), 'x', on_step=lambda s: now.update(step=s))
    out[name] = np.array(calls)
    out[f'{name}_args'] = np.array([args.learning_starts, args.num_updates, args.policy_frequency])
  return out


def adamw_cases() -> dict[str, np.ndarray]:
  """AdamW as train.py:291-300 builds it, three steps."""
  torch.manual_seed(11)
  params = [nn.Parameter(torch.randn(5, 4)), nn.Parameter(torch.randn(3))]
  out = {'before_0': params[0].detach().numpy().copy(), 'before_1': params[1].detach().numpy().copy()}
  # BEGIN VERBATIM fast_td3/train.py:291-295, for a list of parameters in place of the critic's, and 3e-4 and 0.1 in place of the args.
  q_optimizer = optim.AdamW(
      list(params),
      lr=torch.tensor(3e-4, device=device),
      weight_decay=0.1,
  )
  # END VERBATIM
  for step in range(3):
    grads = [torch.randn_like(p) for p in params]
    for p, g in zip(params, grads):
      p.grad = g
    q_optimizer.step()
    out[f'grad_{step}_0'], out[f'grad_{step}_1'] = grads[0].numpy(), grads[1].numpy()
    out[f'after_{step}_0'] = params[0].detach().numpy().copy()
    out[f'after_{step}_1'] = params[1].detach().numpy().copy()
  return out


def main() -> None:
  if sys.argv[1:] == ['bf16']:
    np.savez_compressed(OUT / 'fasttd3_update_bf16.npz', **update_trace('bf16', 3, amp=True))
    return
  np.savez_compressed(OUT / 'fasttd3_projection.npz', **projection_cases())
  np.savez_compressed(OUT / 'fasttd3_update.npz', **update_trace('default', 3), **update_trace('no_cdq', 4, use_cdq=False),
                      **update_trace('loud', 5, policy_noise=2.0, noise_clip=0.5),
                      # A saturated actor, so that noise takes the next action past its bounds and the clamp acts.
                      **update_trace('edge', 6, policy_noise=2.0, noise_clip=0.5, init_scale=2.0))
  np.savez_compressed(OUT / 'fasttd3_losses.npz', **loss_cases())
  np.savez_compressed(OUT / 'fasttd3_noise.npz', **noise_cases())
  np.savez_compressed(OUT / 'fasttd3_schedule.npz', **schedule_cases())
  np.savez_compressed(OUT / 'fasttd3_adamw.npz', **adamw_cases(),
                      **{f'default_{key}': np.array(value) for key, value in REFERENCE_DEFAULTS.items()})
  for path in sorted(OUT.glob('fasttd3_*.npz')):
    print(path.name, path.stat().st_size)


if __name__ == '__main__':
  main()
