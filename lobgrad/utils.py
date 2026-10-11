"""Reproducibility helpers for lobgrad.

Usage:
    from lobgrad.utils import set_seed
    set_seed(42)                        # once at the top of the script/notebook
    out = jt_model.generate(ctx_q, ctx_a, steps=20, n_paths=50, seed=42)

Caveats:
    - Same machine + same GPU + same CUDA driver: outputs are bit-identical.
    - Cross-GPU / cross-driver: outputs may differ in the last few ULPs even with
      seeds matched (PyTorch CUDA non-determinism in atomic ops).
    - `deterministic=True` calls torch.use_deterministic_algorithms(..., warn_only=True),
      so unsupported CUDA ops warn instead of erroring. Pass deterministic=False
      to skip determinism-mode entirely (faster, but no reproducibility guarantee).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
