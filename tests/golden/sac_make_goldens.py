# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Generates tests/golden/sac_*.npz from a verbatim excerpt of rsl_rl_sac.

Source: https://github.com/leggedrobotics/rsl_rl_sac at commit e0d243a, files rsl_rl/algorithms/sac.py
and rsl_rl/models/sac_mlp_model.py, released with Sabatini, Li and Hutter, arXiv 2605.24975. Blocks
between BEGIN VERBATIM and END VERBATIM are copied, shifted only by a uniform change of indentation.
Everything else here is harness. The tests read the stored values and never import from a clone or
from this file.

Copyright (c) 2026, ETH Zurich
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Run on the CPU, from the repository root:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/golden/sac_make_goldens.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal as _TorchNormal

OUT = Path(__file__).resolve().parent
_GENERATOR = torch.Generator().manual_seed(0)

# sac.py:37-54 and sac_mlp_model.py:43-46, the constructor's defaults.
FORK_DEFAULTS = dict(
  replay_buffer_size=1_000_000, num_learning_epochs=1, num_mini_batches=1, mini_batch_size=256, actor_learning_rate=1e-3,
  critic_learning_rate=1e-3, alpha_learning_rate=1e-3, auto_alpha=True, alpha=0.05, tau=0.005, gamma=0.998,
  target_entropy_scale=1.0, max_grad_norm=1.0, policy_frequency=2, n_steps=1, init_noise_std=1.0, log_std_min=-20.0, log_std_max=2.0,
)


class Normal(_TorchNormal):
  """torch's Normal with its standard normals drawn from a known stream and kept for the tests."""

  drawn: list[torch.Tensor] = []

  def rsample(self, sample_shape=torch.Size()):
    eps = torch.randn(self._extended_shape(sample_shape), generator=_GENERATOR)
    Normal.drawn.append(eps.clone())
    return self.loc + eps * self.scale


def elu_mlp(in_size: int, out, hidden: list[int]) -> nn.Sequential:
  """rsl_rl/modules/mlp.py:56-79, without its registry of activations."""
  layers = []
  for size in hidden:
    layers += [nn.Linear(in_size, size), nn.ELU()]
    in_size = size
  if isinstance(out, int):
    layers.append(nn.Linear(in_size, out))
  else:
    layers += [nn.Linear(in_size, out[0] * out[1]), nn.Unflatten(dim=-1, unflattened_size=out)]
  return nn.Sequential(*layers)


class Actor(nn.Module):
  def __init__(self, obs_dim, output_dim, hidden, init_noise_std, log_std_min=-20.0, log_std_max=2.0):
    super().__init__()
    self.mlp = elu_mlp(obs_dim, [2, output_dim], hidden) if hidden is not None else nn.Sequential()  # Empty: mean and log-std given.
    self.output_dim = output_dim
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    # BEGIN VERBATIM rsl_rl/models/sac_mlp_model.py:85-96
    last_linear = None
    for module in reversed(self.mlp):
        if isinstance(module, nn.Linear):
            last_linear = module
            break
    if last_linear is not None:
        # Mean output head → near-zero initial actions
        torch.nn.init.normal_(last_linear.weight[:output_dim], mean=0.0, std=1e-3)
        torch.nn.init.zeros_(last_linear.bias[:output_dim])
        # Log-std output head → log(init_noise_std)
        torch.nn.init.zeros_(last_linear.weight[output_dim:])
        torch.nn.init.constant_(last_linear.bias[output_dim:], torch.log(torch.tensor(init_noise_std + 1e-7)))
    # END VERBATIM
    # BEGIN VERBATIM rsl_rl/models/sac_mlp_model.py:100-102
    self.register_buffer("action_bias", torch.zeros(output_dim))
    self.register_buffer("action_range", torch.ones(output_dim))
    self.register_buffer("log_action_range", torch.zeros(1))
    # END VERBATIM

  def get_latent(self, obs):
    return obs

  # BEGIN VERBATIM rsl_rl/models/sac_mlp_model.py:133-162
  def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
      """Sample an action and compute its corrected log-probability.

      The log-probability includes corrections for Tanh squashing and action scaling.

      Args:
          obs: Observation dictionary.

      Returns:
          Tuple of (scaled_action, log_prob).
      """
      latent = self.get_latent(obs)
      self._update_distribution(latent)
      x_t = self.distribution.rsample()
      tanh_x = torch.tanh(x_t)
      action = self.action_range * tanh_x + self.action_bias

      # Log-probability with Tanh Jacobian correction and action scale correction
      log_prob = self.distribution.log_prob(x_t).sum(dim=-1, keepdim=True)
      log_prob -= torch.log(1 - tanh_x.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
      log_prob -= self.log_action_range

      return action, log_prob

  def _update_distribution(self, latent: torch.Tensor) -> None:
      """Update the Gaussian distribution with log-std clamping."""
      mean_and_log_std = self.mlp(latent)
      mean, log_std = torch.unbind(mean_and_log_std, dim=-2)
      std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
      self.distribution = Normal(mean, std)
  # END VERBATIM


class Critic(nn.Module):
  def __init__(self, in_size, hidden):
    super().__init__()
    self.critic1 = elu_mlp(in_size, 1, hidden)
    self.critic2 = elu_mlp(in_size, 1, hidden)
    # BEGIN VERBATIM rsl_rl/models/sac_mlp_model.py:238-243
    self.critic1_target = copy.deepcopy(self.critic1)
    self.critic2_target = copy.deepcopy(self.critic2)
    for param in self.critic1_target.parameters():
        param.requires_grad = False
    for param in self.critic2_target.parameters():
        param.requires_grad = False
    # END VERBATIM

  def get_latent(self, obs):
    return obs

  # BEGIN VERBATIM rsl_rl/models/sac_mlp_model.py:270-316
  def evaluate_all_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      """Compute Q1 and Q2 for the given observations and actions.

      Args:
          obs: Observation dictionary.
          actions: Action tensor.

      Returns:
          Tuple of (Q1, Q2) value estimates.
      """
      latent = self.get_latent(obs)
      latent = torch.cat([latent, actions], dim=-1)
      return self.critic1(latent), self.critic2(latent)

  def evaluate_all_target_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      """Compute target Q1 and target Q2 for the given observations and actions.

      Uses the frozen target networks (never updated by gradient descent).

      Args:
          obs: Observation dictionary.
          actions: Action tensor.

      Returns:
          Tuple of (Q1_target, Q2_target) value estimates.
      """
      latent = self.get_latent(obs)
      latent = torch.cat([latent, actions], dim=-1)
      return self.critic1_target(latent), self.critic2_target(latent)

  def init_target_networks(self) -> None:
      """Initialize the target networks with the current critic network parameters."""
      self.critic1_target.load_state_dict(self.critic1.state_dict())
      self.critic2_target.load_state_dict(self.critic2.state_dict())

  def soft_update_target_networks(self, tau: float) -> None:
      """Soft-update the target networks using Polyak averaging.

      New target parameters are computed as: ``target = tau * online + (1 - tau) * target``.

      Args:
          tau: Interpolation parameter for soft updates.
      """
      for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
      for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
          target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
  # END VERBATIM


class Algorithm:
  """The attributes of sac.py:32-176 that one update reads. No RND, no symmetry, one process."""

  def __init__(self, actor, critic, args):
    self.device = 'cpu'
    self.actor, self.critic = actor, critic
    self.is_multi_gpu, self.rnd, self.symmetry = False, None, None
    self.gamma, self.tau, self.auto_alpha = args['gamma'], args['tau'], args['auto_alpha']
    self.alpha = args['alpha']
    self.actor_learning_rate = args['actor_learning_rate']
    self.critic_learning_rate = args['critic_learning_rate']
    self.alpha_learning_rate = args['alpha_learning_rate']
    self.policy_frequency = args['policy_frequency']
    self.update_step = 0
    self.max_grad_norm = args['max_grad_norm']
    target_entropy_scale = args['target_entropy_scale']
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:151
    self.target_entropy = -target_entropy_scale * self.actor.output_dim
    # END VERBATIM
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:154-156
    if self.auto_alpha:
        self.log_alpha = torch.log(torch.tensor(self.alpha, device=self.device)).detach().clone().requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_learning_rate)
    # END VERBATIM
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:164-165
    self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
    self.critic_parameters = [p for p in self.critic.parameters() if p.requires_grad]
    # END VERBATIM
    # resolve_optimizer("adam") of sac.py:168-173 is torch.optim.Adam, rsl_rl/utils/utils.py:77-78.
    self.actor_optimizer = optim.Adam(self.actor_parameters, lr=self.actor_learning_rate)
    self.critic_optimizer = optim.Adam(self.critic_parameters, lr=self.critic_learning_rate)
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:176
    self.critic.init_target_networks()
    # END VERBATIM

  def one_update(self, obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch, bootstrap_batch, effective_n_steps):
    """The body of the loop of sac.py:233-420, without its RND and symmetry branches. Returns its locals."""
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:275-303
    # 1) Critic update
    with torch.no_grad():
        bootstrap_mask = bootstrap_batch + 1 - dones_batch
        if torch.any(bootstrap_mask > 1):
            raise ValueError("bootstrap_mask has values greater than 1. Check bootstrapping logic.")

        new_actions, next_log_prob = self.actor.sample_action_logp(next_obs_batch)
        next_state_entropy = -self.log_alpha.exp() * next_log_prob

        q1_target, q2_target = self.critic.evaluate_all_target_q(next_obs_batch, new_actions)
        min_target_q = torch.min(q1_target, q2_target)
        q_target_next = min_target_q + next_state_entropy
        n_step_discount = torch.pow(self.gamma, effective_n_steps.to(dtype=q_target_next.dtype))
        target_q = rewards_batch + n_step_discount * bootstrap_mask * q_target_next

    q1_pred, q2_pred = self.critic.evaluate_all_q(obs_batch, actions_batch)

    critic1_loss = nn.functional.mse_loss(q1_pred, target_q)
    critic2_loss = nn.functional.mse_loss(q2_pred, target_q)

    total_critic_loss = 0.5 * (critic1_loss + critic2_loss)
    self.critic_optimizer.zero_grad()
    total_critic_loss.backward()

    if self.is_multi_gpu:
        self.reduce_parameters(self.critic_parameters)

    torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
    self.critic_optimizer.step()
    # END VERBATIM

    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:305-326
    ###########################################################################
    # Sample new actions for actor and alpha update
    new_actions, log_prob = self.actor.sample_action_logp(obs_batch)

    ###########################################################################
    # 2) Alpha update
    if self.auto_alpha:
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()

        if self.is_multi_gpu:
            if self.log_alpha.grad is not None:
                torch.distributed.all_reduce(self.log_alpha.grad, op=torch.distributed.ReduceOp.SUM)
                self.log_alpha.grad /= self.gpu_world_size

        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
    else:
        alpha_loss = torch.tensor(0.0, device=self.device)

    entropy = self.log_alpha.exp().detach() * log_prob
    # END VERBATIM

    actor_step = self.update_step % self.policy_frequency == 0
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:328-337
    ###########################################################################
    # 3) Actor update
    if self.update_step % self.policy_frequency == 0:
        # Freeze critic parameters for actor update
        for p in self.critic_parameters:
            p.requires_grad_(False)

        q1, q2 = self.critic.evaluate_all_q(obs_batch, new_actions)
        q_new = torch.min(q1, q2)
        actor_loss = (entropy - q_new).mean()
    # END VERBATIM

        # BEGIN VERBATIM rsl_rl/algorithms/sac.py:374-388
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.is_multi_gpu:
            self.reduce_parameters(self.actor_parameters)

        torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
        self.actor_optimizer.step()

        # Unfreeze critic parameters after actor update
        for p in self.critic_parameters:
            p.requires_grad_(True)
    else:
        actor_loss = torch.tensor(0.0, device=self.device)
        symmetry_loss = torch.tensor(0.0, device=self.device)
        # END VERBATIM

    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:390-393
    ###########################################################################
    # 4) Soft update target networks
    with torch.no_grad():
        self.critic.soft_update_target_networks(self.tau)
    # END VERBATIM
    self.update_step += 1  # sac.py:420.
    return locals()


def flat(module: nn.Module) -> np.ndarray:
  return torch.cat([p.detach().flatten() for p in module.parameters()]).numpy().copy()


def as_np(x: torch.Tensor) -> np.ndarray:
  return x.detach().numpy().copy()


def update_trace(name: str, seed: int, updates: int = 12, **changes) -> dict[str, np.ndarray]:
  """Follows the verbatim update. Keeps weights, normals, batches and every loss."""
  torch.manual_seed(seed)
  _GENERATOR.manual_seed(seed + 1000)
  args = {**FORK_DEFAULTS, **changes}
  obs_dim, action_dim, batch_size, hidden = 6, 4, 16, [32, 32]
  actor = Actor(obs_dim, action_dim, hidden, args['init_noise_std'], args['log_std_min'], args['log_std_max'])
  critic = Critic(obs_dim + action_dim, hidden)
  algorithm = Algorithm(actor, critic, args)
  out = {f'{name}_arg_{key}': np.array(value) for key, value in args.items()}
  out[f'{name}_init_actor'] = flat(actor.mlp)
  out[f'{name}_init_q1'], out[f'{name}_init_q2'] = flat(critic.critic1), flat(critic.critic2)
  out[f'{name}_init_log_alpha'] = as_np(algorithm.log_alpha)
  out[f'{name}_target_entropy'] = np.array(algorithm.target_entropy)
  keep = {key: [] for key in ('obs', 'next_obs', 'action', 'reward', 'done', 'normals', 'target', 'next_log_prob', 'log_prob', 'critic1_loss',
                              'critic2_loss', 'critic_loss', 'alpha_loss', 'log_alpha', 'actor_loss', 'actor_step', 'critic_norm', 'actor_norm',
                              'actor', 'q1', 'q2', 'q1_target', 'q2_target')}
  norms: list[torch.Tensor] = []
  clip = torch.nn.utils.clip_grad_norm_

  def recording_clip(*a, **k):
    norm = clip(*a, **k)
    norms.append(norm.detach().clone())
    return norm

  torch.nn.utils.clip_grad_norm_ = recording_clip
  try:
    for _ in range(updates):
      obs, next_obs = torch.randn(batch_size, obs_dim), torch.randn(batch_size, obs_dim)
      action = torch.rand(batch_size, action_dim) * 2 - 1
      reward = torch.randn(batch_size, 1) * args.get('reward_scale', 1.0)
      done = (torch.rand(batch_size, 1) < 0.25).float()
      bootstrap = torch.zeros_like(done)  # No timeout ever bootstraps here, DESIGN 16.
      Normal.drawn.clear()
      norms.clear()
      result = algorithm.one_update(obs, action, reward, next_obs, done, bootstrap, torch.ones(batch_size, 1, dtype=torch.long))
      keep['obs'].append(obs.numpy()), keep['next_obs'].append(next_obs.numpy()), keep['action'].append(action.numpy())
      keep['reward'].append(reward[:, 0].numpy()), keep['done'].append(done[:, 0].numpy())
      keep['normals'].append(torch.stack(Normal.drawn).numpy())
      keep['target'].append(as_np(result['target_q'])[:, 0])
      keep['next_log_prob'].append(as_np(result['next_log_prob'])[:, 0])
      keep['log_prob'].append(as_np(result['log_prob'])[:, 0])
      keep['critic1_loss'].append(result['critic1_loss'].item()), keep['critic2_loss'].append(result['critic2_loss'].item())
      keep['critic_loss'].append(result['total_critic_loss'].item()), keep['alpha_loss'].append(result['alpha_loss'].item())
      keep['log_alpha'].append(algorithm.log_alpha.item()), keep['actor_loss'].append(result['actor_loss'].item())
      keep['actor_step'].append(result['actor_step'])
      keep['critic_norm'].append(norms[0].item()), keep['actor_norm'].append(norms[1].item() if len(norms) > 1 else float('nan'))
      keep['actor'].append(flat(actor.mlp)), keep['q1'].append(flat(critic.critic1)), keep['q2'].append(flat(critic.critic2))
      keep['q1_target'].append(flat(critic.critic1_target)), keep['q2_target'].append(flat(critic.critic2_target))
  finally:
    torch.nn.utils.clip_grad_norm_ = clip
  out.update({f'{name}_{key}': np.array(value) for key, value in keep.items()})
  return out


def log_prob_cases() -> dict[str, np.ndarray]:
  """sample_action_logp with a given mean, raw log-std and normal, including saturation and both clamps."""
  _GENERATOR.manual_seed(7)
  B, d = 256, 7
  draw = torch.Generator().manual_seed(8)
  mean = torch.randn(B, d, generator=draw) * torch.tensor([0.0, 0.1, 1.0, 3.0, 8.0, 20.0, 60.0])
  log_std = torch.rand(B, d, generator=draw) * 34.0 - 28.0  # Past both clamps, -20 and 2.
  log_std[:64] = torch.rand(64, d, generator=draw) * 4.0 - 3.0  # Ordinary values.
  log_std[-8:] = 0.0
  mean[-8:] = 0.0
  actor = Actor(1, d, None, 1.0)
  action, log_prob = actor.sample_action_logp(torch.stack([mean, log_std], -2))
  return dict(mean=mean.numpy(), log_std=log_std.numpy(), normals=torch.stack(Normal.drawn)[-1].numpy(), action=as_np(action),
              log_prob=as_np(log_prob)[:, 0], log_std_min=np.array(actor.log_std_min), log_std_max=np.array(actor.log_std_max))


def loss_cases() -> dict[str, np.ndarray]:
  """The four losses alone, on chosen values, with timeouts present in one of the masks."""
  draw = torch.Generator().manual_seed(9)
  B = 512
  q1_target, q2_target = (torch.randn(B, 1, generator=draw) * 3.0 for _ in range(2))
  next_log_prob = torch.randn(B, 1, generator=draw) * 10.0 - 5.0
  rewards_batch = torch.randn(B, 1, generator=draw) * 0.2
  dones_batch = (torch.rand(B, 1, generator=draw) < 0.3).float()
  timeout = dones_batch * (torch.rand(B, 1, generator=draw) < 0.5).float()
  effective_n_steps = torch.ones(B, 1, dtype=torch.long)
  q1_pred, q2_pred = (torch.randn(B, 1, generator=draw) for _ in range(2))
  q1, q2 = (torch.randn(B, 1, generator=draw) for _ in range(2))
  log_prob = torch.randn(B, 1, generator=draw) * 10.0 - 5.0
  out = dict(q1_target=q1_target, q2_target=q2_target, next_log_prob=next_log_prob, reward=rewards_batch, done=dones_batch, timeout=timeout,
             q1_pred=q1_pred, q2_pred=q2_pred, q1=q1, q2=q2, log_prob=log_prob)
  out = {key: value.numpy()[:, 0] for key, value in out.items()}

  class Self:
    gamma = 0.94
    log_alpha = torch.log(torch.tensor(0.05))
    target_entropy = -28.0

  self = Self()
  for key, bootstrap_batch in (('no_timeouts', torch.zeros_like(dones_batch)), ('fork_timeouts', timeout)):
    # BEGIN VERBATIM rsl_rl/algorithms/sac.py:277, :282, :285-288
    bootstrap_mask = bootstrap_batch + 1 - dones_batch
    next_state_entropy = -self.log_alpha.exp() * next_log_prob
    min_target_q = torch.min(q1_target, q2_target)
    q_target_next = min_target_q + next_state_entropy
    n_step_discount = torch.pow(self.gamma, effective_n_steps.to(dtype=q_target_next.dtype))
    target_q = rewards_batch + n_step_discount * bootstrap_mask * q_target_next
    # END VERBATIM
    out[f'target_{key}'] = target_q.numpy()[:, 0]
  target_q = torch.as_tensor(out['target_no_timeouts'])[:, None]
  # BEGIN VERBATIM rsl_rl/algorithms/sac.py:292-295
  critic1_loss = nn.functional.mse_loss(q1_pred, target_q)
  critic2_loss = nn.functional.mse_loss(q2_pred, target_q)

  total_critic_loss = 0.5 * (critic1_loss + critic2_loss)
  # END VERBATIM
  # BEGIN VERBATIM rsl_rl/algorithms/sac.py:312
  alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
  # END VERBATIM
  # BEGIN VERBATIM rsl_rl/algorithms/sac.py:326
  entropy = self.log_alpha.exp().detach() * log_prob
  # END VERBATIM
  # BEGIN VERBATIM rsl_rl/algorithms/sac.py:336-337
  q_new = torch.min(q1, q2)
  actor_loss = (entropy - q_new).mean()
  # END VERBATIM
  out.update(critic1_loss=critic1_loss.numpy(), critic2_loss=critic2_loss.numpy(), critic_loss=total_critic_loss.numpy(), alpha_loss=alpha_loss.numpy(),
             actor_loss=actor_loss.numpy(), log_alpha=self.log_alpha.numpy(), target_entropy=np.array(self.target_entropy), gamma=np.array(self.gamma))
  return out


def init_cases() -> dict[str, np.ndarray]:
  """The initialization of the last layer, the paper's first change, at the default and at its Table C9's sigma."""
  out = {}
  for name, sigma in (('default', 1.0), ('paper', 0.15)):
    torch.manual_seed(11)
    actor = Actor(6, 7, [64, 64], sigma)
    last = [m for m in actor.mlp if isinstance(m, nn.Linear)][-1]
    out[f'{name}_sigma'] = np.array(sigma)
    out[f'{name}_weight'], out[f'{name}_bias'] = as_np(last.weight), as_np(last.bias)
  return out


def main() -> None:
  np.savez_compressed(OUT / 'sac_log_prob.npz', **log_prob_cases())
  np.savez_compressed(OUT / 'sac_losses.npz', **loss_cases())
  np.savez_compressed(OUT / 'sac_init.npz', **init_cases())
  # 'default' is SACCfg's update rule, gamma 0.94. 'fork' is the constructor's own defaults, policy frequency 2.
  # 'every' and 'third' vary the frequency, 'tight' makes both clips act, 'paper' takes the update's values of its Table C9.
  np.savez_compressed(OUT / 'sac_update.npz',
                      **update_trace('default', 3, gamma=0.94),
                      **update_trace('fork', 4),
                      **update_trace('every', 5, gamma=0.94, policy_frequency=1),
                      **update_trace('third', 6, gamma=0.94, policy_frequency=3),
                      **update_trace('tight', 8, gamma=0.94, max_grad_norm=0.05),
                      **update_trace('paper', 7, gamma=0.97, tau=0.003, actor_learning_rate=2e-4, critic_learning_rate=2e-4,
                                     alpha_learning_rate=2e-5, alpha=0.001, target_entropy_scale=0.167, policy_frequency=1,
                                     init_noise_std=0.15))
  for path in sorted(OUT.glob('sac_*.npz')):
    print(path.name, path.stat().st_size)


if __name__ == '__main__':
  main()
