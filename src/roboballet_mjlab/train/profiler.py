# Modified from RoboBallet (Zenodo 10.5281/zenodo.16546403), Copyright 2025 Google LLC, Apache 2.0. See NOTICE.
"""Phase timers, CPU against wall, GPU utilisation and memory, launches. DESIGN section 8.

Outside a window a phase costs one Python call and no GPU wait: torch.cuda.synchronize runs at
phase edges only inside short windows, so normal running keeps its asynchronous dispatch and
the window's own rate is pessimistic. Each phase records exclusive time.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import resource
import time
from typing import Any, Callable

import torch

_NULL = contextlib.nullcontext()
EVERY_ROUNDS = 6000
WINDOW_ROUNDS = 120
FIRST_WINDOW = 240  # Past the first episode, and clear of the env's own warm-up.


class _Utilization(ctypes.Structure):
  _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class _Memory(ctypes.Structure):
  _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class Nvml:
  """Read-only NVML through ctypes, since pynvml is not in the venv. Reports nothing if absent."""

  def __init__(self, device: torch.device) -> None:
    self._lib = self._handle = None
    if device.type != 'cuda':
      return
    try:
      lib = ctypes.CDLL('libnvidia-ml.so.1')
      if lib.nvmlInit_v2() != 0:
        return
      uuid = str(torch.cuda.get_device_properties(device).uuid)
      handle = ctypes.c_void_p()
      if lib.nvmlDeviceGetHandleByUUID(f'GPU-{uuid}'.encode(), ctypes.byref(handle)) != 0:
        return
      self._lib, self._handle = lib, handle
    except (OSError, AttributeError):
      pass

  def read(self) -> dict[str, float]:
    if self._lib is None:
      return {}
    rates, memory = _Utilization(), _Memory()
    if self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(rates)) != 0:
      return {}
    if self._lib.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(memory)) != 0:
      return {}
    return {'gpu_util_percent': float(rates.gpu), 'gpu_mem_util_percent': float(rates.memory),
            'gpu_mem_used_gib': memory.used / 2**30}


class _Phase:
  __slots__ = ('profiler', 'name')

  def __init__(self, profiler: Profiler, name: str) -> None:
    self.profiler, self.name = profiler, name

  def __enter__(self) -> None:
    self.profiler._edge()
    self.profiler._stack.append(self.name)

  def __exit__(self, *exc: Any) -> None:
    self.profiler._edge()
    self.profiler._stack.pop()


class Profiler:
  def __init__(
    self, enabled: bool, device: str | torch.device, launches: Callable[[], int] = lambda: 0,
    every_rounds: int = EVERY_ROUNDS, window_rounds: int = WINDOW_ROUNDS, first_window: int = FIRST_WINDOW,
  ) -> None:
    self.enabled = enabled
    self.device = torch.device(device)
    self.launches = launches
    self.every_rounds, self.window_rounds, self.first_window = every_rounds, window_rounds, first_window
    self.in_window = False
    self._stack: list[str] = []
    self._totals: dict[str, float] = {}
    self._last_edge = 0.0
    self._window_left = 0
    self._window_report: dict[str, float] = {}
    self._items = 0
    self._nvml = Nvml(self.device) if enabled else None
    self._reset_interval(0)

  def _reset_interval(self, rounds: int) -> None:
    self._interval = (time.perf_counter(), time.thread_time(), time.process_time(), self.launches(), rounds)

  def _sync(self) -> None:
    if self.device.type == 'cuda':
      torch.cuda.synchronize(self.device)

  def _edge(self) -> None:
    self._sync()
    now = time.perf_counter()
    top = self._stack[-1] if self._stack else 'other'
    self._totals[top] = self._totals.get(top, 0.0) + now - self._last_edge
    self._last_edge = now

  def phase(self, name: str) -> _Phase | contextlib.nullcontext:
    return _Phase(self, name) if self.in_window else _NULL

  def wrap(self, owner: Any, method: str, name: str) -> None:
    """Times a method of one object, so env internals get phases without touching the env."""
    original = getattr(owner, method)

    @functools.wraps(original)
    def timed(*args, **kwargs):
      if not self.in_window:
        return original(*args, **kwargs)
      with _Phase(self, name):
        return original(*args, **kwargs)

    setattr(owner, method, timed)

  def begin_round(self, round_index: int) -> None:
    """Opens or closes a window. Call at the top of a round, outside any phase."""
    if not self.enabled:
      return
    if self.in_window:
      self._window_left -= 1
      if self._window_left == 0:
        self._close_window()
    elif round_index >= self.first_window and (round_index - self.first_window) % self.every_rounds == 0:
      self._open_window(round_index)

  def _open_window(self, round_index: int) -> None:
    self._sync()
    self.in_window = True
    self._window_left = self.window_rounds
    self._totals, self._stack = {}, []
    self._window_start = (time.perf_counter(), time.thread_time(), self.launches())
    self._last_edge = self._window_start[0]
    self._items = 0
    # Python-level .item() only. int(tensor) and tolist() go through other paths.
    original = torch.Tensor.item

    def counted(tensor: torch.Tensor):
      self._items += 1
      return original(tensor)

    self._restore_item = original
    torch.Tensor.item = counted

  def _close_window(self) -> None:
    self._edge()
    torch.Tensor.item = self._restore_item
    wall, cpu, launches = self._window_start
    wall = time.perf_counter() - wall
    cpu = time.thread_time() - cpu
    n = self.window_rounds
    report = {f'profile/{name}_ms': total / n * 1e3 for name, total in self._totals.items()}
    report['profile/window_round_ms'] = wall / n * 1e3
    report['profile/window_main_thread_cpu_fraction'] = cpu / wall
    report['profile/window_launches_per_round'] = (self.launches() - launches) / n
    report['profile/window_item_calls_per_round'] = self._items / n
    self._window_report = report
    self.in_window = False

  def report(self, rounds: int) -> dict[str, float]:
    """The last finished window, and this interval's CPU, GPU and memory. Clears the window."""
    if not self.enabled:
      return {}
    wall0, thread0, process0, launches0, rounds0 = self._interval
    wall = max(time.perf_counter() - wall0, 1e-9)
    out = dict(self._window_report)
    self._window_report = {}
    out['profile/main_thread_cpu_fraction'] = (time.thread_time() - thread0) / wall
    out['profile/process_cpu_fraction'] = (time.process_time() - process0) / wall
    if rounds > rounds0:
      out['profile/launches_per_round'] = (self.launches() - launches0) / (rounds - rounds0)
    self._reset_interval(rounds)
    out['profile/rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if self.device.type == 'cuda':
      out['profile/mem_allocated_gib'] = torch.cuda.memory_allocated(self.device) / 2**30
      out['profile/mem_reserved_gib'] = torch.cuda.memory_reserved(self.device) / 2**30
      out['profile/mem_peak_gib'] = torch.cuda.max_memory_allocated(self.device) / 2**30
      out.update({f'profile/{k}': v for k, v in self._nvml.read().items()})
    return out
