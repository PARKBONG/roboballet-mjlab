"""Draws a cell from four cameras to one PNG, or records a policy solving a scene to a video.

Picture: CUDA_VISIBLE_DEVICES=1 uv run python -m roboballet_mjlab.scripts.render
Video, .mp4 or .gif, with any checkpoint or policy snapshot:
  CUDA_VISIBLE_DEVICES=1 uv run python -m roboballet_mjlab.scripts.render \
    --policy runs/g8a_s1/policies/policy_00400000.pt --arm-model mesh_tool \
    --pool data/start_states/4r_10t_30o.pb --scene 0 --out demo.mp4
The picture reads its scenes from tests/golden, which is on the test branch.
The scene is the one asked for, whatever the outcome: a failed episode records as it happened.

EGL is llvmpipe on the CPU on this host. Device 6 is the only one that does not walk
the NVIDIA nodes, GPU 0 among them (mjlab_api 8.2).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import mediapy
import numpy as np
import torch
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from PIL import Image, ImageDraw, ImageFont

from roboballet_mjlab.agent.features import Topology, build_graph
from roboballet_mjlab.agent.networks import PolicyNet, load_policy, saved_model_cfg
from roboballet_mjlab.config import HOME_QPOS, EnvCfg, WorldCfg
from roboballet_mjlab.env.ballet_env import BalletEnv
from roboballet_mjlab.sim.world import build_world
from roboballet_mjlab.structs import Scene
from roboballet_mjlab.train.evaluator import load_scenes

SPHERES, BOTH, MESHES = (1, 0, 0, 1, 0, 0), (1, 0, 1, 1, 0, 0), (1, 0, 1, 0, 0, 0)  # Geom groups: 2 meshes, 3 spheres.
PENDING, SOLVED = (0.95, 0.35, 0.1, 1.0), (0.1, 0.8, 0.25, 1.0)
HOLD_LAST_S = 1.0
GOLDEN = Path(__file__).resolve().parents[3] / 'tests/golden'  # The clone's, test branch.


def _load_scene(num_robots: int, index: int, device) -> Scene:
  z = np.load(GOLDEN / f'env_scenes_{num_robots}r10t.npz')
  i = np.flatnonzero(z['valid'])[index]
  return Scene(*[torch.as_tensor(z[f][i:i + 1], device=device) for f in Scene._fields])


def render(num_robots: int, scene_index: int, out: Path, width: int, height: int) -> None:
  device = 'cuda:0'
  cfg = WorldCfg(num_robots=num_robots, visual_meshes=True)
  world = build_world(cfg, 1, device)
  scene = _load_scene(num_robots, scene_index, device)
  world.set_scene(torch.zeros(1, dtype=torch.long, device=device), scene)
  world.set_q(torch.as_tensor(HOME_QPOS, device=device).repeat(1, num_robots, 1))
  world.forward()
  if world.robots_in_collision().any():
    raise RuntimeError('this scene collides at home')
  tips = world.tcp()[0][0].cpu().numpy()
  targets = scene.target_pos[0].cpu().numpy()

  def overlay(vis):
    # Blue: TCP, the stick's working point. Green: targets.
    for tip in tips:
      vis.add_sphere(tip, 0.012, (0.1, 0.3, 1.0, 1.0))
    for target in targets:
      vis.add_sphere(target, 0.015, (0.1, 0.9, 0.2, 1.0))

  mid = 0.9 if num_robots > 4 else 0.3
  views = [
    dict(geom_group=BOTH, distance=3.6, elevation=-25, azimuth=120, lookat=(0, 0, mid)),
    dict(geom_group=SPHERES, distance=3.6, elevation=-25, azimuth=120, lookat=(0, 0, mid)),
    dict(geom_group=BOTH, distance=3.2, elevation=-89, azimuth=90, lookat=(0, 0, 0)),
    dict(geom_group=BOTH, distance=0.9, elevation=-5, azimuth=90,
         lookat=tuple(tips[0] + np.array([0.05, 0.0, 0.16]))),
  ]
  frames = []
  for view in views:
    vc = ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD, width=width, height=height, max_extra_envs=0,
      enable_shadows=False, enable_reflections=False, **view)
    renderer = OffscreenRenderer(
      model=world.sim.mj_model, cfg=vc, scene=world.scene, sim_model=world.sim.model,
      expanded_fields=world.sim.expanded_fields)
    renderer.initialize()
    renderer.update(world.sim.data, debug_vis_callback=overlay)
    frames.append(renderer.render())
    renderer.close()
  mosaic = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
  out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_image(str(out), mosaic)


@torch.no_grad()
def rollout(policy_path: Path, pool_path: Path, scene_index: int, arm_model: str, seed: int, device) -> tuple[WorldCfg, EnvCfg, Scene, list, int]:
  """One episode of the scene, as the states to draw: the env resets a world as soon as it ends."""
  pool = load_scenes(pool_path, device)
  world_cfg = WorldCfg(num_robots=pool.base_pos.shape[1], num_obstacles=pool.box_pos.shape[1], arm_model=arm_model)
  env_cfg = EnvCfg(world=world_cfg, num_targets=pool.target_pos.shape[1])
  topo = Topology(env_cfg, device)
  policy = PolicyNet(saved_model_cfg(str(policy_path)), topo)
  step = load_policy(str(policy_path), policy)
  policy.to(device).requires_grad_(False)
  env = BalletEnv(env_cfg, 1, pool, device, seed)
  env.reset(scene_ids=torch.tensor([scene_index], device=device))
  observed = env.observe()
  scene = observed.scene
  states = [(observed.robot.q, observed.robot.target_done, observed.robot.tip_pos, 0)]
  for _ in range(env_cfg.max_steps):
    out = env.step(policy(build_graph(env.observe(), topo)))
    final = out.next_robot
    collisions = states[-1][3] + int(out.collision_robot_steps.item())
    states.append((final.q, final.target_done, final.tip_pos, collisions))
    if out.terminal.item():
      break
  return world_cfg, env_cfg, scene, states, step


def caption(frame: np.ndarray, text: str, height: int) -> np.ndarray:
  image = Image.fromarray(frame)
  font = ImageFont.load_default(size=max(12, height // 28))
  draw = ImageDraw.Draw(image)
  box = draw.textbbox((12, 10), text, font=font)
  draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(255, 255, 255))
  draw.text((12, 10), text, fill=(11, 11, 11), font=font)
  return np.asarray(image)


def record(policy_path: Path, pool_path: Path, scene_index: int, arm_model: str, seed: int, out: Path, width: int, height: int, label: str) -> None:
  device = 'cuda:0'
  torch.set_float32_matmul_precision('highest')
  world_cfg, env_cfg, scene, states, step = rollout(policy_path, pool_path, scene_index, arm_model, seed, device)
  world = build_world(dataclasses.replace(world_cfg, visual_meshes=True), 1, device)
  world.set_scene(torch.zeros(1, dtype=torch.long, device=device), scene)
  targets = scene.target_pos[0].cpu().numpy()
  vc = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD, width=width, height=height, max_extra_envs=0, enable_shadows=False,
    enable_reflections=False, geom_group=MESHES, distance=3.2, elevation=-30, azimuth=120, lookat=(0, 0, 0.3))
  renderer = OffscreenRenderer(
    model=world.sim.mj_model, cfg=vc, scene=world.scene, sim_model=world.sim.model, expanded_fields=world.sim.expanded_fields)
  renderer.initialize()
  frames = []
  for t, (q, done, _, collisions) in enumerate(states):
    world.set_q(q)
    world.forward()
    solved = done[0].cpu().numpy()

    def overlay(vis, solved=solved):
      for target, is_solved in zip(targets, solved):
        vis.add_sphere(target, 0.02, SOLVED if is_solved else PENDING)

    renderer.update(world.sim.data, debug_vis_callback=overlay)
    text = f'{label}{step:,} grad steps   tasks {int(solved.sum())}/{len(solved)}   {t * env_cfg.timestep_s:.1f} s   collision steps {collisions}'
    frames.append(caption(renderer.render(), text, height))
  renderer.close()
  fps = round(1 / env_cfg.timestep_s)
  frames += [frames[-1]] * round(HOLD_LAST_S * fps)
  out.parent.mkdir(parents=True, exist_ok=True)
  if out.suffix == '.gif':
    images = [Image.fromarray(f).quantize(colors=128, dither=Image.Dither.NONE) for f in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0, optimize=True)
  else:
    mediapy.write_video(str(out), frames, fps=fps)
  final_done, length = int(states[-1][1].sum().item()), len(states) - 1
  print(f'{out}: scene {scene_index}, step {step}, tasks {final_done}/{env_cfg.num_targets}, length {length}, collision steps {states[-1][3]}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--robots', type=int, default=4)
  parser.add_argument('--scene', type=int, default=0, help='Index among the valid stored scenes.')
  parser.add_argument('--out', type=Path, default=Path('docs/world_4r.png'))
  parser.add_argument('--width', type=int, default=800)
  parser.add_argument('--height', type=int, default=600)
  parser.add_argument('--policy', type=Path, help='Record this policy to --out, a .mp4 or .gif, instead of drawing a cell.')
  parser.add_argument('--pool', type=Path, help='Scene pool for --policy.')
  parser.add_argument('--arm-model', default=WorldCfg.arm_model, choices=('spheres_stick', 'mesh_tool'))
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--label', default='', help='Text before the step count in the caption.')
  args = parser.parse_args()
  if args.policy is not None:
    if args.pool is None:
      parser.error('--policy needs --pool')
    record(args.policy, args.pool, args.scene, args.arm_model, args.seed, args.out, args.width, args.height, args.label)
  else:
    render(args.robots, args.scene, args.out, args.width, args.height)
