"""G1: the render script produces a picture. Looking at it is a manual step, see DESIGN 9."""

from __future__ import annotations

import mediapy
import numpy as np

from roboballet_mjlab.scripts.render import render


def test_render_writes_a_nonblank_mosaic(tmp_path):
  out = tmp_path / 'world.png'
  render(num_robots=4, scene_index=0, out=out, width=160, height=120)
  image = mediapy.read_image(str(out))
  assert image.shape[:2] == (240, 320)
  assert np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0] > 50
