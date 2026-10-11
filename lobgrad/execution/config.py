"""
config.py
---------
ExecutionConfig — hyperparameters for the certified optimal execution experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExecutionConfig:
    """
    Parameters governing the execution experiment.

    Order parameters
    ----------------
    order_size   : total shares (BTC) to execute. Used by the open-loop
                   schedule constructors. The MPC path uses ``mpc_q_units``
                   instead (which can be auto-calibrated from depth).
    horizon      : number of time steps over which to execute
    side         : "sell" (hit bids) or "buy" (lift asks)

    Evaluation
    ----------
    n_windows    : number of evaluation windows sampled from test set
    context_len  : context length fed to the generative model

    MPC (closed-loop)
    -----------------
    mpc_n_rollouts  : K rollouts per cost-to-go estimate (per step, per
                      candidate quantity)
    mpc_n_candidates: size of the candidate-quantity grid, including 0 and Q_t
    mpc_q_units     : total quantity to acquire over the horizon. If None, a
                      caller-supplied calibration (e.g. 1.0 × avg top-5 depth)
                      should be plugged in before running the benchmark.
    mpc_default_policy: how to complete the remaining quantity inside each
                      rollout when computing cost-to-go. "twap" is the only
                      currently supported option (constant-rate completion).

    Market parameters
    -----------------
    price_level  : tick size δ (USDT). Must match the data the simulator was
                   trained on.
    levels       : number of LOB levels per side (k).
    """

    # Order
    order_size:    float = 1.0       # BTC — used by open-loop legacy code
    horizon:       int   = 30        # steps (30 × 5s = 2.5 min)
    side:          str   = "buy"     # "buy" (lift asks — acquire Q at min cost)

    # Evaluation
    n_windows:     int   = 500       # paired windows for distributional metrics reuse
    context_len:   int   = 100       # context length the generator conditions on

    # MPC
    mpc_n_rollouts:    int   = 50            # K rollouts per cost-to-go estimate
    mpc_n_candidates:  int   = 11            # candidate-q grid size (incl. 0 and Q_t)
    mpc_q_units:       float | None = None   # total quantity; auto-calibrate if None
    mpc_default_policy: str  = "twap"        # rollout-internal completion policy

    # Market — must match BenchmarkConfig used to train the simulator
    price_level:   float = 0.1       # tick size δ (USDT)
    levels:        int   = 10        # k
