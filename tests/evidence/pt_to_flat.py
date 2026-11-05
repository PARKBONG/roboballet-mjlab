"""A policy trained by the port as the flat .npz that orig_train.py and orig_static.py read, so the unmodified original can score it.

Claim checked: a policy the port trained performs the same when the original runs it. The
conversion is exact or refuses: the port's parameter names must equal the original network's flax
names one to one, kernels are transposed back to flax's layout, and nothing else changes. It
prints the grad step and the number of tensors (158 for the full model).

Release environment, from the repository root:
  PYTHONPATH=src python tests/evidence/pt_to_flat.py POLICY.pt OUT.npz
"""
import sys

import numpy as np
import torch

from roboballet_mjlab.agent.networks import _reference_nets, flax_names

checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
names = flax_names(_reference_nets()[0], 'policy/params')
if set(names) != set(checkpoint['policy']):
  raise ValueError('parameters differ')
flat = {key: (checkpoint['policy'][name].numpy().T if key.endswith('kernel') else checkpoint['policy'][name].numpy()) for name, key in names.items()}
np.savez(sys.argv[2], step=checkpoint['step'], **flat)
print(checkpoint['step'], len(flat))
