"""Task ids and agent presets, registered the way mjlab registers its tasks.

A task id names what an agent trains on: the scenes, the arm model and the evaluation sets. The
env is BalletEnv, direct style: mjlab's own train builds a ManagerBasedRlEnv and cannot run it,
so the train, play and eval commands of this package read this registry instead.

Two kinds of defaults, kept apart:
- The config dataclasses (config.py, fasttd3/config.py, sac/config.py, ppo/config.py) hold their
  defaults unchanged: TrainCfg is the original's TD3 in parity mode, and the others are each
  reference implementation's own values, as DESIGN records.
- AGENTS below holds a preset per agent: the overrides the train command applies on top of those
  defaults, before the user's own --set and --model-set. They are what the tuning runs on
  RoboBallet-4R10T-Mesh chose. td3 has none: it is the original's TD3, reproduced.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from roboballet_mjlab.config import ArmModel
from roboballet_mjlab.paths import data_dir, scene_pool


@dataclasses.dataclass(frozen=True)
class TaskCfg:
  """`start_states` is a file in the data directory's start_states/, `train_pool` a pool name for
  paths.scene_pool. Exactly one is set. `eval_sets` maps evaluator set names to pool names; with no
  training_dist entry the evaluator uses the training scenes, as the original does."""

  task_id: str
  description: str
  arm_model: ArmModel
  start_states: str | None = None
  train_pool: str | None = None
  eval_sets: tuple[tuple[str, str], ...] = ()

  def train_source(self) -> tuple[str, Path]:
    """The train modules' flag for the training scenes, and the file."""
    if self.start_states is not None:
      return '--start-states', data_dir() / 'start_states' / self.start_states
    return '--train-pool', scene_pool(self.train_pool)

  def eval_paths(self) -> dict[str, Path]:
    return {name: scene_pool(pool) for name, pool in self.eval_sets}


@dataclasses.dataclass(frozen=True)
class AgentPreset:
  """The train module an agent runs, the overrides applied before the user's, and which of the
  command's options that module takes."""

  module: str
  description: str
  set: tuple[str, ...] = ()
  model_set: tuple[str, ...] = ()
  takes_model_set: bool = True
  takes_gpus: bool = True


_REGISTRY: dict[str, TaskCfg] = {}


def register_task(cfg: TaskCfg) -> None:
  if cfg.task_id in _REGISTRY:
    raise ValueError(f"Task '{cfg.task_id}' is already registered")
  if (cfg.start_states is None) == (cfg.train_pool is None):
    raise ValueError(f'{cfg.task_id}: give start_states or train_pool, not both')
  _REGISTRY[cfg.task_id] = cfg


def list_tasks() -> list[str]:
  return sorted(_REGISTRY)


def load_task(task_id: str) -> TaskCfg:
  return _REGISTRY[task_id]


register_task(TaskCfg(
  task_id='RoboBallet-4R10T-Mesh',
  description="4 robots, 10 targets, 30 obstacles. The original's mesh arm and tool on its own start states: "
              'the parity setting of the paper.',
  arm_model='mesh_tool',
  start_states='4r_10t_30o.pb',
  eval_sets=(('static_paper', '4r_10t_30o_static_paper'),),
))
register_task(TaskCfg(
  task_id='RoboBallet-4R10T-Spheres',
  description='4 robots, 10 targets, 30 obstacles. The sphere-and-stick arm on the regenerated scene pools.',
  arm_model='spheres_stick',
  train_pool='4r_10t_30o_train',
  eval_sets=(('heldout', '4r_10t_30o_heldout'), ('static_dist', '4r_10t_30o_static'),
             ('static_paper', '4r_10t_30o_static_paper')),
))
register_task(TaskCfg(
  task_id='RoboBallet-8R40T-Mesh',
  description="8 robots, 40 targets, 30 obstacles. The original's mesh arm and tool on its own start states.",
  arm_model='mesh_tool',
  start_states='8r_40t_30o.pb',
  eval_sets=(('static_paper', '8r_40t_30o_static_paper'),),
))
register_task(TaskCfg(
  task_id='RoboBallet-8R40T-Spheres',
  description='8 robots, 40 targets, 30 obstacles. The sphere-and-stick arm on the regenerated scene pools.',
  arm_model='spheres_stick',
  train_pool='8r_40t_30o_train',
  eval_sets=(('heldout', '8r_40t_30o_heldout'), ('static_dist', '8r_40t_30o_static'),
             ('static_paper', '8r_40t_30o_static_paper')),
))

# The 3-layer GNN of the tuning runs. ModelCfg's default is the paper's: 7, 6, 6 and 7 layers.
_SMALL_MODEL = ('node_layers=3', 'edge_embed_layers=3', 'edge_update_layers=3', 'global_layers=3')

DEFAULT_AGENT = 'fasttd3'
AGENTS: dict[str, AgentPreset] = {
  'td3': AgentPreset(
    module='roboballet_mjlab.scripts.train',
    description="The original's TD3 with hindsight, reproduced: TrainCfg and ModelCfg defaults, no overrides.",
    takes_model_set=False,  # Its train module builds ModelCfg() as the original does.
    takes_gpus=False,
  ),
  'fasttd3': AgentPreset(
    module='roboballet_mjlab.fasttd3.train',
    description='FastTD3, the main agent. Preset: the 3-layer model, rates 2e-4 with a cosine to 0.1 over 100k '
                'grad steps, 3-step returns, and the exploration cap 0.4 cosine-decayed to 0.1 over the same steps.',
    set=('lr_policy=2e-4', 'lr_critic=2e-4', 'n_step=3', 'lr_decay_steps=100000', 'lr_end_fraction=0.1',
         'noise_decay_steps=100000', 'noise_end_fraction=0.25'),
    model_set=_SMALL_MODEL,
  ),
  'sac': AgentPreset(
    module='roboballet_mjlab.sac.train',
    description="rsl_rl_sac's SAC. Preset: the 3-layer model, the tuning runs' rates, temperature and noise, "
                '5-step returns and a cosine over 150k grad steps.',
    set=('lr_actor=2e-4', 'lr_critic=2e-4', 'lr_alpha=2e-5', 'alpha=0.001', 'tau=0.003', 'target_entropy_scale=0.167',
         'policy_frequency=1', 'init_noise_std=0.15', 'log_std_max=-1.0', 'n_step=5', 'lr_decay_steps=150000',
         'lr_end_fraction=0.1'),
    model_set=_SMALL_MODEL,
  ),
  'ppo': AgentPreset(
    module='roboballet_mjlab.ppo.train',
    description="rsl-rl's PPO. Preset: the 3-layer model and a fixed learning rate of 1e-4.",
    set=('schedule=fixed', 'learning_rate=1e-4'),
    model_set=_SMALL_MODEL,
    takes_gpus=False,  # One process: its train module has no --gpus.
  ),
}
