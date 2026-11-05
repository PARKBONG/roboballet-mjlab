"""Loads the agent goldens as torch tensors for the test_agent_* files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from roboballet_mjlab.agent.networks import flax_names
from roboballet_mjlab.agent.td3 import TD3, Batch
from roboballet_mjlab.structs import Graph, Obs, RobotState, Scene
from tests.golden import agent_common

GOLDEN = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('ROBOBALLET_PROJECT', GOLDEN.parents[2]))  # As tests/test_sim_common.py.
CHECKPOINT = PROJECT / 'runs/mjlab_port_analysis/ckpt/paper_4r10t_s1_1870000.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class E6(NamedTuple):
  obs: Obs
  graph: Graph
  raw: dict[str, np.ndarray]


def load_e6(name: str, shards: int, device: str = DEVICE) -> E6:
  z = np.load(GOLDEN / f'agent_e6_{name}.npz')
  raw = {k: z[k] for k in z.files}
  t = lambda x: torch.as_tensor(x, device=device)
  episode = raw['episode']
  scene_rows = {f: raw[f][episode] for f in ('base_pos', 'base_rot', 'box_pos', 'box_rot', 'box_spans', 'target_pos', 'target_rot')}
  scene = Scene(
    **{k: t(v) for k, v in scene_rows.items()},
    q0=t(np.zeros_like(raw['q'])),
  )
  robot = RobotState(
    q=t(raw['q']), qd=t(raw['qd']), dwell=t(raw['dwell']), tip_pos=t(raw['tip_pos']), tip_rot=t(raw['tip_rot']),
    target_done=t(raw['target_done']), score=t(raw['score']), step=t(raw['step']), aug_code=t(raw['aug_code']).long(),
  )
  parts = [np.load(GOLDEN / f'agent_e6_{name}_expected_{i}.npz') for i in range(shards)]
  graph = Graph(*[t(np.concatenate([p[k] for p in parts])) for k in ('nodes', 'edges', 'globals')])
  return E6(Obs(robot, scene), graph, raw)


def transitions(e6: E6, index: np.ndarray) -> Batch:
  """Transitions from the golden features themselves, so features cannot affect TD3 checks."""
  t = lambda x: torch.as_tensor(x, device=e6.graph.nodes.device)
  i = t(index)
  return Batch(
    Graph(*[x[i] for x in e6.graph]), Graph(*[x[i + 1] for x in e6.graph]),
    t(e6.raw['action'][index]), t(e6.raw['reward'][index]), t(e6.raw['terminal'][index]),
  )


def flat_nets(td3: TD3) -> dict[str, dict[str, np.ndarray]]:
  """Current parameters by flax key and in flax layout, for sample_leaves."""
  def flat(net, prefix):
    state = net.state_dict()
    out = {}
    for name, key in flax_names(net, prefix).items():
      leaf = state[name].cpu().numpy()
      out[key] = leaf.T if key.endswith('kernel') else leaf
    return out

  return {
    'pol': flat(td3.policy, 'params'), 'tpol': flat(td3.policy_target, 'params'),
    'cri': {**flat(td3.q1, 'params/critic_1'), **flat(td3.q2, 'params/critic_2')},
    'tcri': {**flat(td3.q1_target, 'params/critic_1'), **flat(td3.q2_target, 'params/critic_2')},
  }


def deterministic_init(td3: TD3) -> None:
  """Loads the agent_common weights, the starting point of agent_td3.npz."""
  def load(net, prefix):
    state = net.state_dict()
    for name, key in flax_names(net, prefix).items():
      leaf = agent_common.deterministic_leaf(key, state[name].T.shape if key.endswith('kernel') else state[name].shape)
      state[name].copy_(torch.from_numpy(leaf.T if key.endswith('kernel') else leaf))

  load(td3.policy, 'params')
  load(td3.q1, 'params/critic_1')
  load(td3.q2, 'params/critic_2')
  td3.sync_targets()
