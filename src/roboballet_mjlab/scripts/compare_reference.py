"""Overlays runs on the reference run, on the tags both log. DESIGN 14.4.

  uv run python -m roboballet_mjlab.scripts.compare_reference \
    --reference runs/paper_repro/paper_4r10t_s1/tensorboard_log runs/g8a_s1 runs/g8a_s2

The reference is a run of the original code: its tensorboard_log directory, with the launcher's
provenance.txt beside it. It is not in this repository.

Each cell is the mean over a window of grad steps around the step, so runs that evaluate at different
intervals compare. Milestones are the first step whose trailing mean reaches the task rate, timed from process
launch to the learner reaching that step. The reference's evaluator writes its row about 3 minutes after the
learner passes the step, so the row's time is not used.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

NUM_TARGETS = 10
STEPS = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000, 600_000,
         800_000, 1_000_000, 1_200_000, 1_500_000, 1_870_000)
TAGS = (
  'eval/training_dist/num_targets_done', 'eval/training_dist/episode_length',
  'eval/training_dist/total_collision_robot_steps', 'eval/training_dist/total_mean_acceleration',
  'eval/static_dist/num_targets_done', 'eval/static_paper/num_targets_done',
  'train/policy_loss', 'train/twin_critic_loss', 'train/trajectory_network_age',
  'act/num_targets_done', 'act/episode_length', 'act/total_reward', 'act/total_collision_robot_steps',
  'act/is_hindsight_replay', 'act/network_step',
)
MILESTONES = (0.5, 0.75, 0.9, 0.95, 0.97, 0.98)
MILESTONE_EPISODES = 500
# Acceptance of paper_baseline 10.4, written before any run of this trainer. Rate and the step it must be reached by.
CROSSINGS = ((0.50, 250_000), (0.75, 370_000), (0.90, 550_000), (0.95, 750_000), (0.97, 1_000_000), (0.98, 1_300_000))
# Step, minimum task rate, maximum episode length, maximum collision robot-steps.
FIXED = ((200_000, 0.45, None, 40), (400_000, 0.80, 112, 40), (600_000, 0.90, 100, 25), (800_000, 0.94, 90, 18), (1_000_000, 0.96, 82, 13))

REFERENCE_EPISODES_PER_EVAL = 50
LEARNER_TAGS = ('train/total_step_time_ms', 'counters/grad_steps')
KST = timezone(timedelta(hours=9))


class Series(dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]):
  """Tag to steps, values and wall times, plus when the process was launched."""

  launch: float = float('nan')
  episodes_per_eval: int = 0


def load_reference(reference: Path) -> Series:
  """`reference` is the original run's tensorboard_log directory, with provenance.txt beside it."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  series = Series()
  # `date` in the launcher, first line of provenance.txt.
  first_line = (reference.parent / 'provenance.txt').read_text().splitlines()[0]
  series.launch = datetime.strptime(first_line.replace('KST ', ''), '%a %b %d %I:%M:%S %p %Y').replace(tzinfo=KST).timestamp()
  series.episodes_per_eval = REFERENCE_EPISODES_PER_EVAL
  for path in sorted(reference.glob('events.out.tfevents.*')):
    events = EventAccumulator(str(path), size_guidance={'scalars': 0})
    events.Reload()
    for tag in events.Tags()['scalars']:
      rows = events.Scalars(tag)
      series[tag] = tuple(np.array(x, dtype=float) for x in zip(*((r.step, r.value, r.wall_time) for r in rows)))
  # The original logs the paper's tasksets as static_dist. Here that name is a regenerated pool (paper_baseline 4.9).
  for tag in [t for t in series if t.startswith('eval/static_dist/')]:
    series[tag.replace('static_dist', 'static_paper')] = series[tag]
  return series


def load_run(run_dir: Path) -> Series:
  rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
  with open(run_dir / 'metrics.csv', newline='') as f:
    for row in csv.DictReader(f):
      rows[row['tag']].append((float(row['step']), float(row['value']), float(row['wall_time'])))
  # A resumed run repeats no step, but keep the last value of any step that appears twice.
  series = Series()
  for tag, values in rows.items():
    latest = {step: (value, wall) for step, value, wall in values}
    steps = np.array(sorted(latest))
    series[tag] = (steps, np.array([latest[s][0] for s in steps]), np.array([latest[s][1] for s in steps]))
  series.launch = launch_of(run_dir)
  worlds = series.get('eval/training_dist/worlds')
  series.episodes_per_eval = int(worlds[1][-1]) if worlds is not None else 0
  return series


def launch_of(run_dir: Path) -> float:
  """write_provenance records it. Without that, the earliest provenance stamp, about 35 s after launch."""
  recorded = run_dir / 'launch_time'
  if recorded.exists():
    return float(recorded.read_text())
  stamps = sorted(p.name.split('_')[0] for p in (run_dir / 'provenance').iterdir())
  return datetime.strptime(stamps[0], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).timestamp()


def learner_hours(series: Series, steps: np.ndarray | float) -> np.ndarray:
  """Hours from launch at which the learner reached these grad steps."""
  learner = next((series[t] for t in LEARNER_TAGS if t in series), None)
  if learner is None:
    return np.full(np.shape(steps), np.nan)
  return (np.interp(steps, learner[0], learner[2]) - series.launch) / 3600


def window_mean(series: Series, tag: str, step: float) -> float:
  if tag not in series:
    return float('nan')
  steps, values, _ = series[tag]
  half = max(0.05 * step, 2_500.0)
  chosen = values[np.abs(steps - step) <= half]
  return float(chosen.mean()) if len(chosen) else float('nan')


def milestone(series: Series, tag: str, rate: float) -> tuple[float, float]:
  """First step, and hours from launch on the learner's clock, at which the trailing mean reaches the rate.

  The trailing window is 50k grad steps, or the last 3 evaluations where those hold enough episodes. A run
  that crosses in few grad steps would otherwise be held back by a window as long as its whole curve.
  """
  if tag not in series:
    return float('nan'), float('nan')
  steps, values, _ = series[tag]
  short = 3 * series.episodes_per_eval >= MILESTONE_EPISODES
  for i, step in enumerate(steps):
    if step < 20_000:
      continue
    start = max(i - 2, 0) if short else int(np.searchsorted(steps, step - 50_000))
    if values[start:i + 1].mean() / NUM_TARGETS >= rate:
      return float(step), float(learner_hours(series, step))
  return float('nan'), float('nan')


def acceptance(series: Series) -> list[tuple[str, str, float | None, str]]:
  """Each check of 10.4 as (name, limit, value, PASS, FAIL or pending)."""
  train = 'eval/training_dist/'
  latest = series[train + 'num_targets_done'][0].max() if train + 'num_targets_done' in series else 0.0
  rows = []
  for rate, deadline in CROSSINGS:
    step, _ = milestone(series, train + 'num_targets_done', rate)
    if np.isnan(step):
      state = 'FAIL' if latest >= deadline else 'pending'
      rows.append((f'task rate {rate:.2f} reached', f'by {deadline:,}', None, state))
    else:
      rows.append((f'task rate {rate:.2f} reached', f'by {deadline:,}', step, 'PASS' if step <= deadline else 'FAIL'))
  for step, min_rate, max_length, max_collisions in FIXED:
    done = latest >= step * 1.05  # The whole window is in.
    checks = [('task rate', 'num_targets_done', 1 / NUM_TARGETS, min_rate, 1), ('episode length', 'episode_length', 1, max_length, -1),
              ('collision steps', 'total_collision_robot_steps', 1, max_collisions, -1)]
    for label, tag, scale, limit, sign in checks:
      if limit is None:
        continue
      value = window_mean(series, train + tag, step) * scale
      name = f'{label} at {step:,}'
      limit_text = f'{">=" if sign > 0 else "<="} {limit}'
      if not done:
        rows.append((name, limit_text, None, 'pending'))
      else:
        rows.append((name, limit_text, value, 'PASS' if (value - limit) * sign >= 0 else 'FAIL'))
  return rows


def fmt(value: float) -> str:
  if np.isnan(value):
    return ''
  magnitude = abs(value)
  if magnitude >= 1000:
    return f'{value:,.0f}'
  if magnitude >= 10:
    return f'{value:.1f}'
  return f'{value:.3f}' if magnitude >= 0.01 or value == 0 else f'{value:.2e}'


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('runs', type=Path, nargs='+')
  parser.add_argument('--reference', type=Path, required=True,
                      help="The original run's tensorboard_log directory, with provenance.txt beside it.")
  parser.add_argument('--median', nargs='*', default=[], help='Run names forming the median of 10.4, which needs at least 3 seeds.')
  args = parser.parse_args()
  sets = {'reference': load_reference(args.reference)}
  for run in args.runs:
    sets[run.name] = load_run(run)
  names = list(sets)
  latest = {name: max((s[0].max() for s in sets[name].values() if len(s[0])), default=0.0) for name in names}
  print('| Latest grad step | ' + ' | '.join(names) + ' |')
  print('|---|' + '---:|' * len(names))
  print('| | ' + ' | '.join(f'{latest[n]:,.0f}' for n in names) + ' |')

  for tag in TAGS:
    if not any(tag in sets[n] for n in names):
      continue
    rows = [(s, [window_mean(sets[n], tag, s) if s <= latest[n] * 1.01 else float('nan') for n in names]) for s in STEPS]
    rows = [(s, cells) for s, cells in rows if not all(np.isnan(cells))]
    if not rows:
      continue
    print(f'\n**{tag}**\n')
    print('| Step | ' + ' | '.join(names) + ' |')
    print('|---:|' + '---:|' * len(names))
    for step, cells in rows:
      print(f'| {step:,} | ' + ' | '.join(fmt(c) for c in cells) + ' |')

  tag = 'eval/training_dist/num_targets_done'
  print('\n**Milestones on training_dist: grad step, hours from launch on the learner clock**\n')
  print('| Task rate | ' + ' | '.join(names) + ' |')
  print('|---:|' + '---:|' * len(names))
  for rate in MILESTONES:
    cells = []
    for n in names:
      step, hours = milestone(sets[n], tag, rate)
      cells.append('' if np.isnan(step) else f'{step:,.0f}, {hours:.2f} h')
    print(f'| {rate:.0%} | ' + ' | '.join(cells) + ' |')

  print('\n**Acceptance, paper_baseline 10.4, per run**\n')
  print('| Check | Limit | ' + ' | '.join(names) + ' |')
  print('|---|---|' + '---:|' * len(names))
  tables = {}
  for n in names:
    tables[n] = acceptance(sets[n])
  for i, (name, limit, _, _) in enumerate(tables[names[0]]):
    cells = []
    for n in names:
      _, _, value, state = tables[n][i]
      cells.append(state if value is None else f'{fmt(value)} {state}')
    print(f'| {name} | {limit} | ' + ' | '.join(cells) + ' |')
  if args.median:
    print(f'\n**Median of {", ".join(args.median)}**, the pass condition when there are at least 3 seeds\n')
    print('| Check | Limit | Median | State |')
    print('|---|---|---:|---|')
    for i, (name, limit, _, _) in enumerate(tables[names[0]]):
      rows = [tables[n][i] for n in args.median]
      values = [r[2] for r in rows]
      if any(v is None for v in values):
        state = 'FAIL' if sum(r[3] == 'FAIL' for r in rows) > len(rows) / 2 else 'pending'
        print(f'| {name} | {limit} | | {state} |')
        continue
      value = float(np.median(values))
      if limit.startswith('by'):
        state = 'PASS' if value <= float(limit[3:].replace(',', '')) else 'FAIL'
      elif limit.startswith('>='):
        state = 'PASS' if value >= float(limit[3:]) else 'FAIL'
      else:
        state = 'PASS' if value <= float(limit[3:]) else 'FAIL'
      print(f'| {name} | {limit} | {fmt(value)} | {state} |')


if __name__ == '__main__':
  main()
