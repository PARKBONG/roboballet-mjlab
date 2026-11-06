"""Learning curves of runs against the reference run, as PNG files for the docs.

  uv run python -m roboballet_mjlab.scripts.plot_curves \
    --reference runs/paper_repro/paper_4r10t_s1/tensorboard_log \
    "TD3 port=runs/g8a_s1,runs/g8a_s2,runs/g8a_s3" "spheres and stick=runs/g8b_s1"

Directories after one name are seeds of one agent: they share a colour and every seed is drawn. Every run
gets the same smoothing: the mean over a window of grad steps. Runs evaluate at different intervals and
with different episode counts, so the raw points are not comparable one by one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from roboballet_mjlab.scripts.compare_reference import CROSSINGS, NUM_TARGETS, Series, learner_hours, load_reference, load_run

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
REFERENCE_COLOR = '#0b0b0b'
# Categorical slots in fixed order, checked with the dataviz validator. A run keeps its slot.
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
LINE_WIDTH = 1.5  # 2 px at 96 dpi.
HALF_WINDOW = 15_000.0
GRID_STEP = 5_000.0
# Runs that finish in a few hundred thousand grad steps evaluate every 4k, so they get a window to scale.
SHORT_RUN, SHORT_HALF_WINDOW, SHORT_GRID_STEP = 500_000.0, 6_000.0, 1_000.0

PANELS = (
  ('eval/training_dist/num_targets_done', 'Tasks done, training scenes', 1 / NUM_TARGETS, 0.0),
  ('eval/training_dist/episode_length', 'Episode length, steps', 1.0, 0.0),
  # The untrained policy at step 0 reaches about 360 on one arm, which would flatten the rest of the panel.
  ('eval/training_dist/total_collision_robot_steps', 'Collision robot-steps per episode, after step 0', 1.0, 1.0),
  ('train/policy_loss', 'Policy loss, sum over 20 updates', 1.0, 0.0),
  ('train/twin_critic_loss', 'Twin critic loss, sum over 20 updates', 1.0, 0.0),
  ('act/is_hindsight_replay', 'Hindsight share of stored episodes', 1.0, 0.0),
)
README_PANELS = (
  ('eval/training_dist/num_targets_done', 'Tasks done, random training scenes', 1 / NUM_TARGETS, 0.0),
  ('eval/static_paper/num_targets_done', "Tasks done, the paper's 10 tasksets", 1 / NUM_TARGETS, 0.0),
  ('eval/training_dist/episode_length', 'Episode length, steps (lower is faster)', 1.0, 0.0),
  ('eval/training_dist/total_collision_robot_steps', 'Collision robot-steps per episode, after 100k steps', 1.0, 100_000.0),
)
# Fig. 3A of the paper, 4 robots and 10 tasks: median of its 5 lines, read from the vector paths
# (docs/evidence/performance.verified.md).
PAPER_FIG_3A = ((1e5, 0.078), (2e5, 0.485), (3e5, 0.732), (4e5, 0.873), (5e5, 0.917), (6e5, 0.945), (8e5, 0.967), (1e6, 0.977))
HOUR_TICKS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16)


def smooth(steps: np.ndarray, values: np.ndarray, end: float) -> tuple[np.ndarray, np.ndarray]:
  """Interpolates onto a grid and then averages, so evaluations 20k steps apart do not draw as stairs."""
  short = end < SHORT_RUN
  step, window = (SHORT_GRID_STEP, SHORT_HALF_WINDOW) if short else (GRID_STEP, HALF_WINDOW)
  grid = np.arange(0.0, end + step, step)
  grid = grid[grid >= steps.min()]
  filled = np.interp(grid, steps, values)
  half = int(window // step)
  total = np.concatenate([[0.0], np.cumsum(filled)])
  left, right = np.maximum(np.arange(len(grid)) - half, 0), np.minimum(np.arange(len(grid)) + half + 1, len(grid))
  return grid, (total[right] - total[left]) / (right - left)


def style(ax: plt.Axes, title: str) -> None:
  ax.set_facecolor(SURFACE)
  ax.set_title(title, loc='left', fontsize=10, color=INK, pad=8)
  ax.grid(True, axis='y', color=GRID, linewidth=0.75)
  ax.set_axisbelow(True)
  for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
  ax.spines['bottom'].set_color(GRID)
  ax.tick_params(colors=MUTED, labelsize=8, length=0)


def draw(ax: plt.Axes, series: Series, tag: str, scale: float, color: str, width: float, x: str, label: str | None = None,
         from_step: float = 0.0) -> None:
  if tag not in series:
    return
  steps, values, _ = series[tag]
  keep = steps >= from_step
  steps, values = steps[keep], values[keep]
  if not len(steps):
    return
  grid, mean = smooth(steps, values * scale, float(steps.max()))
  if x == 'hours':
    grid = learner_hours(series, grid)
  else:
    grid = grid / 1e6
  ax.plot(grid, mean, color=color, linewidth=width, solid_capstyle='round', solid_joinstyle='round', label=label)


def draw_all(ax: plt.Axes, sets: dict[str, list[Series]], tag: str, scale: float, x: str, from_step: float = 0.0) -> None:
  for i, (name, group) in enumerate(sets.items()):
    color = REFERENCE_COLOR if i == 0 else SERIES_COLORS[(i - 1) % len(SERIES_COLORS)]
    for j, series in enumerate(group):
      draw(ax, series, tag, scale, color, LINE_WIDTH, x, name if j == 0 else None, from_step)


def curves_figure(sets: dict[str, list[Series]], out: Path) -> None:
  fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE, sharex=True)
  for ax, (tag, title, scale, from_step) in zip(axes.flat, PANELS):
    style(ax, title)
    draw_all(ax, sets, tag, scale, 'steps', from_step)
  rate_axis = axes.flat[0]
  deadlines = [(step / 1e6, rate) for rate, step in CROSSINGS]
  rate_axis.plot(*zip(*deadlines), linestyle='none', marker='o', markersize=5, markerfacecolor=SURFACE,
                 markeredgecolor=MUTED, markeredgewidth=1.0, label='latest step that passes', zorder=3)
  rate_axis.set_ylim(0, 1.02)
  for ax in axes[1]:
    ax.set_xlabel('Grad steps, millions', fontsize=9, color=MUTED)
  handles, labels = rate_axis.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.012, 0.995), ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
  fig.text(0.985, 0.012, f'Each line is the mean over +-{HALF_WINDOW / 1000:.0f}k grad steps', ha='right', fontsize=8, color=MUTED)
  fig.tight_layout(rect=(0, 0.02, 1, 0.95))
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  plt.close(fig)


def wall_clock_figure(sets: dict[str, list[Series]], out: Path) -> None:
  """Task rate against hours on a log axis, so a run many times faster still fits beside the reference."""
  fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
  style(ax, 'Tasks done on random training scenes, against wall clock')
  draw_all(ax, sets, 'eval/training_dist/num_targets_done', 1 / NUM_TARGETS, 'hours')
  ax.axhline(0.95, color=MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
  ax.annotate('95% of tasks', (HOUR_TICKS[0], 0.95), xytext=(2, 3), textcoords='offset points', fontsize=8, color=MUTED)
  ax.set_xscale('log')
  ax.set_xticks(HOUR_TICKS, [str(t) for t in HOUR_TICKS])
  ax.minorticks_off()
  ax.set_xlim(HOUR_TICKS[0], HOUR_TICKS[-1])
  ax.set_ylim(0, 1.02)
  ax.grid(True, axis='x', color=GRID, linewidth=0.75)
  ax.set_xlabel('Hours from launch to the learner reaching the step, log scale', fontsize=9, color=MUTED)
  ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='upper left', bbox_to_anchor=(0.0, 0.85))
  fig.tight_layout()
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  plt.close(fig)


def benchmark_figure(sets: dict[str, list[Series]], out: Path) -> None:
  """Task rate and the policy's learning rate against hours, log scales, every agent."""
  fig, (rate_axis, lr_axis) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
  style(rate_axis, 'Tasks done on random training scenes')
  draw_all(rate_axis, sets, 'eval/training_dist/num_targets_done', 1 / NUM_TARGETS, 'hours')
  rate_axis.axhline(0.95, color=MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
  rate_axis.set_ylim(0, 1.02)
  style(lr_axis, 'Policy learning rate')
  for i, (name, group) in enumerate(sets.items()):
    color = REFERENCE_COLOR if i == 0 else SERIES_COLORS[(i - 1) % len(SERIES_COLORS)]
    for series in group:
      if 'train/policy_lr' in series:
        steps, values, _ = series['train/policy_lr']
        lr_axis.plot(learner_hours(series, steps), values, color=color, linewidth=LINE_WIDTH)
  lr_axis.set_yscale('log')
  for ax in (rate_axis, lr_axis):
    ax.set_xscale('log')
    ax.set_xticks(HOUR_TICKS, [str(t) for t in HOUR_TICKS])
    ax.minorticks_off()
    ax.set_xlim(HOUR_TICKS[0], HOUR_TICKS[-1])
    ax.grid(True, axis='x', color=GRID, linewidth=0.75)
    ax.set_xlabel('Hours from launch, log scale', fontsize=9, color=MUTED)
  handles, labels = rate_axis.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.012, 0.995), ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  plt.close(fig)


def same_performance_figure(sets: dict[str, list[Series]], out: Path) -> None:
  """The four curves a README reader needs, against grad steps, with the paper's own points."""
  fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), facecolor=SURFACE, sharex=True)
  for ax, (tag, title, scale, from_step) in zip(axes.flat, README_PANELS):
    style(ax, title)
    draw_all(ax, sets, tag, scale, 'steps', from_step)
  rate_axis = axes.flat[0]
  rate_axis.plot([s / 1e6 for s, _ in PAPER_FIG_3A], [r for _, r in PAPER_FIG_3A], linestyle='none', marker='o', markersize=5,
                 markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.0, label='paper Fig. 3A, median of 5 lines', zorder=3)
  for ax in axes.flat[:2]:
    ax.set_ylim(0, 1.02)
  for ax in axes.flat[2:]:
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Grad steps, millions', fontsize=9, color=MUTED)
  handles, labels = rate_axis.get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.012, 0.995), ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('runs', nargs='+', help='NAME=DIRECTORY[,DIRECTORY...], one directory per seed')
  parser.add_argument('--reference', type=Path, nargs='+', required=True,
                      help="The original runs' tensorboard_log directories, one per seed, as for compare_reference.")
  parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[3] / 'docs/figures')
  parser.add_argument('--prefix', default='g8')
  args = parser.parse_args()
  references = [load_reference(path) for path in args.reference]
  sets: dict[str, list[Series]] = {f'original code, {len(references)} seed{"s" if len(references) > 1 else ""}': references}
  for pair in args.runs:
    name, directories = pair.split('=', 1)
    sets[name] = [load_run(Path(d)) for d in directories.split(',')]
  args.out.mkdir(parents=True, exist_ok=True)
  curves_figure(sets, args.out / f'{args.prefix}_learning_curves.png')
  wall_clock_figure(sets, args.out / f'{args.prefix}_wall_clock.png')
  same_performance_figure(sets, args.out / f'{args.prefix}_same_performance.png')
  benchmark_figure(sets, args.out / f'{args.prefix}_benchmark.png')
  print(args.out / f'{args.prefix}_learning_curves.png')


if __name__ == '__main__':
  main()
