"""
lpcvc2026.modules.soup
======================
Model-soup weight interpolation utilities.

Exports:
  load_sd    - Load state_dict from a checkpoint file
  avg_sd     - Average multiple state_dicts (uniform model soup)
  blend      - Linear interpolation: alpha*sd1 + (1-alpha)*sd2
  blend3     - 3-way blend: a*sda + b*sdb + c*sdc  (a+b+c must equal 1)
  grid3      - Generate (a, b, c) grids where a+b+c=1
"""

import os
import torch


def load_sd(path):
    """Load state_dict from .pt checkpoint.

    Handles both raw state_dicts and {'state_dict': ...} wrappers.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt.get('state_dict', ckpt)


def avg_sd(sds):
    """Uniform average of a list of state_dicts.

    All keys must match. Returns a new state_dict with float32 values.
    """
    result = {}
    for k in sds[0].keys():
        result[k] = sum(sd[k].float() for sd in sds) / len(sds)
    return result


def blend(sd1, sd2, alpha):
    """2-way linear blend: alpha*sd1 + (1-alpha)*sd2."""
    return {k: alpha * sd1[k].float() + (1 - alpha) * sd2[k].float()
            for k in sd1}


def blend3(sda, sdb, sdc, a, b, c):
    """3-way blend: a*sda + b*sdb + c*sdc.

    a + b + c should equal 1.0 (not enforced, caller's responsibility).
    """
    return {k: a * sda[k].float() + b * sdb[k].float() + c * sdc[k].float()
            for k in sda}


def grid3(a_vals, b_vals, c_min=0.05, c_max=0.9):
    """Generate valid (a, b, c) triples where a+b+c=1 and c_min <= c <= c_max.

    Args:
      a_vals: iterable of values for the first component
      b_vals: iterable of values for the second component
      c_min:  minimum value for c (default 0.05)
      c_max:  maximum value for c (default 0.9)

    Returns: list of (a, b, c) tuples
    """
    grids = []
    for a in a_vals:
        for b in b_vals:
            c = round(1.0 - a - b, 4)
            if c_min <= c <= c_max:
                grids.append((a, b, c))
    return grids


def load_checkpoints(base_dir, spec):
    """Bulk-load checkpoints from a spec dict.

    spec: dict mapping key_name -> path (absolute or relative to base_dir)
    Returns: dict of {key: state_dict}, skips missing files with a warning.
    """
    sd = {}
    for key, path in spec.items():
        full_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        if os.path.exists(full_path):
            sd[key] = load_sd(full_path)
            print(f"  Loaded {key}  ({full_path})")
        else:
            print(f"  WARNING: not found — {full_path}")
    return sd
