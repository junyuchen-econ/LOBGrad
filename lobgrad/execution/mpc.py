"""
mpc.py
------
Receding-horizon model-predictive control (MPC) for the optimal-execution
downstream task.

At each step t the agent chooses an immediate quantity ``q_t`` from a
discrete grid. Cost-to-go for each candidate is estimated by averaging
``walk_book`` cost over K rollouts drawn from a ``RolloutSource``, with the
remaining quantity ``Q_t - q_t`` completed at TWAP rate over the rest of the
horizon. The agent commits only ``q_t``; at step t+1 it observes the real
LOB, re-plans from scratch.

The same algorithm and same hyperparameters apply to every rollout source —
only the source of forecasted LOBs differs across baselines. This is the
controlled-experiment claim: any cost difference between rows is attributable
to forecast quality.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .cost import book_walk_cost
from .rollouts import RolloutSource


# =====================================================================
# 1-step decision
# =====================================================================
def mpc_step(
    *,
    real_mid_t:        float,
    real_queue_raw_t:  np.ndarray,    # (2k,) — real absolute sizes at step t
    Q_remaining:       float,
    H_remaining:       int,           # steps remaining including current step
    history:           dict,          # {"queue_norm": (T_so_far, 2k), "anchors": (T_so_far,)}
                                      # — observations up to AND INCLUDING the current step
    t:                 int,           # index of the next step to forecast.
                                      # Caller passes (current episode step + 1) so
                                      # source rollouts cover positions [t, t+H_forecast).
    source:            RolloutSource,
    n_rollouts:        int,
    n_candidates:      int,
    side:              str,
    price_level:       float,
    levels:            int,
) -> float:
    """Decide the immediate quantity ``q_t`` at the current step.

    Returns
    -------
    q_t : float in [0, Q_remaining]

    Mechanism
    ---------
    1. Build a candidate-q grid of size ``n_candidates`` linearly spanning
       [0, Q_remaining] (inclusive on both ends).
    2. For each candidate q:
         immediate = walk current real LOB to fill q
         if Q_remaining - q == 0: cost-to-go = 0
         else: sample n_rollouts forecast paths of length (H_remaining - 1),
               complete the remainder at TWAP rate on each rollout, average.
       total = immediate + cost-to-go
    3. Argmin over the grid; return the corresponding q.

    Notes
    -----
    - ``H_remaining`` includes the current step. The forecast horizon for
      cost-to-go is ``H_remaining - 1`` (the steps *after* the current one).
    - When ``H_remaining == 1`` the agent must take everything now; the grid
      collapses and we return ``Q_remaining`` directly.
    """
    if H_remaining <= 0:
        raise ValueError(f"H_remaining must be > 0, got {H_remaining}")
    if Q_remaining <= 1e-12:
        return 0.0
    if H_remaining == 1:
        # Last step — must clear remainder. Skip the grid + rollout entirely.
        return float(Q_remaining)

    # Candidate grid: 0, Q/(n-1), 2Q/(n-1), ..., Q.
    candidates = np.linspace(0.0, Q_remaining, n_candidates)

    # Cost-to-go horizon (forecast steps).
    H_forecast = H_remaining - 1

    # Rollouts are drawn ONCE per step and re-used across all candidates,
    # so the candidates are compared against the same noise realizations
    # (paired comparison; lower variance, fairer pick).
    mid_paths, queue_paths = source.sample(
        t       = t,
        history = history,
        K       = n_rollouts,
        H       = H_forecast,
    )
    K_actual = mid_paths.shape[0]   # may be 1 for deterministic sources

    total_costs = np.empty(len(candidates), dtype=np.float64)
    for i, q in enumerate(candidates):
        immediate = book_walk_cost(
            shares      = float(q),
            mid         = float(real_mid_t),
            queue       = real_queue_raw_t,
            side        = side,
            price_level = price_level,
            levels      = levels,
        )

        Q_left = float(Q_remaining - q)
        if Q_left <= 1e-12:
            total_costs[i] = immediate
            continue

        # Depth-proportional completion: allocate Q_left at each future step
        # weighted by predicted same-side depth so more quantity is placed
        # when liquidity is deep (lower price impact per share).
        per_path = np.zeros(K_actual, dtype=np.float64)
        for k in range(K_actual):
            if side == "buy":
                depths = queue_paths[k, :H_forecast, levels:].sum(axis=1)
            else:
                depths = queue_paths[k, :H_forecast, :levels].sum(axis=1)
            depths = np.maximum(depths, 1e-6)
            weights = depths / depths.sum()
            c = 0.0
            for s in range(H_forecast):
                c += book_walk_cost(
                    shares      = float(Q_left * weights[s]),
                    mid         = float(mid_paths[k, s]),
                    queue       = queue_paths[k, s],
                    side        = side,
                    price_level = price_level,
                    levels      = levels,
                )
            per_path[k] = c

        total_costs[i] = immediate + per_path.mean()

    best = int(np.argmin(total_costs))
    return float(candidates[best])


# =====================================================================
# Episode runners
# =====================================================================
def run_mpc_episode(
    *,
    real_mid:        np.ndarray,        # (T,) raw USDT — actual realized future
    real_queue_raw:  np.ndarray,        # (T, 2k) raw absolute sizes
    history_init:    dict,              # {"queue_norm": (ctx, 2k), "anchors": (ctx,)}
                                        # — observations PRECEDING the window
    Q_total:         float,
    source:          RolloutSource,
    n_rollouts:      int,
    n_candidates:    int,
    side:            str,
    price_level:     float,
    levels:          int,
) -> Dict[str, np.ndarray]:
    """Run one execution episode (one window) using MPC with the given source.

    The agent observes ``real_mid[t]`` and ``real_queue_raw[t]`` at step t,
    decides ``q_t`` via ``mpc_step``, pays the immediate cost on the real
    LOB, then advances. The model only ever influences decisions — costs
    are always charged against the real future.

    Returns dict with:
        actions (T,)        : q_t at each step
        costs   (T,)        : cost paid at each step
        total_cost (scalar) : sum of step costs
        Q_remaining (T+1,)  : Q remaining at start of each step (final = 0)
    """
    T = len(real_mid)
    if real_queue_raw.shape[0] != T:
        raise ValueError("real_mid and real_queue_raw must have the same length")

    actions     = np.zeros(T, dtype=np.float64)
    costs       = np.zeros(T, dtype=np.float64)
    Q_track     = np.zeros(T + 1, dtype=np.float64)
    Q_track[0]  = Q_total

    # We update history as the episode progresses so that the model sees the
    # actually-observed real LOB at every replanning step.
    queue_norm_hist = list(history_init["queue_norm"])   # list of (2k,)
    anchors_hist    = list(history_init["anchors"])

    Q_t = Q_total
    for t in range(T):
        # NOTE: this variant does NOT update history during the episode (kept
        # for callers that only need the prefix). For evolving history during
        # the episode, use ``run_mpc_episode_with_norm`` instead — that's the
        # function the benchmark actually drives. The ``t + 1`` here matches
        # the forecast-index convention used by RolloutSource.sample.
        H_remaining = T - t   # steps including this one
        history_t = {
            "queue_norm": np.asarray(queue_norm_hist),
            "anchors":    np.asarray(anchors_hist),
        }

        q_t = mpc_step(
            real_mid_t        = float(real_mid[t]),
            real_queue_raw_t  = real_queue_raw[t],
            Q_remaining       = Q_t,
            H_remaining       = H_remaining,
            history           = history_t,
            t                 = t + 1,                # ← index of next forecast step
            source            = source,
            n_rollouts        = n_rollouts,
            n_candidates      = n_candidates,
            side              = side,
            price_level       = price_level,
            levels            = levels,
        )

        c_t = book_walk_cost(
            shares      = q_t,
            mid         = float(real_mid[t]),
            queue       = real_queue_raw[t],
            side        = side,
            price_level = price_level,
            levels      = levels,
        )

        actions[t]     = q_t
        costs[t]       = c_t
        Q_t            = max(Q_t - q_t, 0.0)
        Q_track[t + 1] = Q_t

        # Update history with the just-observed real snapshot for the next replan.
        # Note: queue_norm_hist tracks the *normalized* queue (matching the
        # source's expectations). The caller is responsible for passing
        # normalized snapshots in real_queue_norm_window if they want this
        # update — see ``run_mpc_episode_with_norm`` below.

    return {
        "actions":      actions,
        "costs":        costs,
        "total_cost":   float(costs.sum()),
        "Q_remaining":  Q_track,
    }


def run_mpc_episode_with_norm(
    *,
    real_mid:           np.ndarray,        # (T,) raw USDT
    real_queue_raw:     np.ndarray,        # (T, 2k) raw absolute
    real_queue_norm:    np.ndarray,        # (T, 2k) normalized — for history updates
    history_init_q:     np.ndarray,        # (ctx, 2k) normalized prefix history
    history_init_a:     np.ndarray,        # (ctx,) raw USDT prefix history
    Q_total:            float,
    source:             RolloutSource,
    n_rollouts:         int,
    n_candidates:       int,
    side:               str,
    price_level:        float,
    levels:             int,
) -> Dict[str, np.ndarray]:
    """Same as ``run_mpc_episode`` but maintains the normalized history
    correctly between steps. Use this when the source needs an evolving
    history (e.g. ModelRollouts) — the history grows by one normalized
    snapshot per step using ``real_queue_norm[t]``.
    """
    T = len(real_mid)

    actions     = np.zeros(T, dtype=np.float64)
    costs       = np.zeros(T, dtype=np.float64)
    Q_track     = np.zeros(T + 1, dtype=np.float64)
    Q_track[0]  = Q_total

    # Pre-allocate history buffers; we'll grow by appending each step's true snapshot.
    qn_hist = list(history_init_q)
    a_hist  = list(history_init_a)

    Q_t = Q_total
    for t in range(T):
        # Append the just-observed real snapshot to history BEFORE replanning,
        # so the source sees all observations through (and including) step t.
        # The source's ``t`` parameter then refers to the index of the next
        # step to forecast, i.e. (t + 1) — forecasts cover positions
        # t+1, t+2, ..., T-1 (matching the cost-to-go horizon T-t-1).
        qn_hist.append(real_queue_norm[t])
        a_hist.append(float(real_mid[t]))

        history_t = {
            "queue_norm": np.asarray(qn_hist),
            "anchors":    np.asarray(a_hist),
        }
        H_remaining = T - t

        q_t = mpc_step(
            real_mid_t        = float(real_mid[t]),
            real_queue_raw_t  = real_queue_raw[t],
            Q_remaining       = Q_t,
            H_remaining       = H_remaining,
            history           = history_t,
            t                 = t + 1,                # ← index of next forecast step
            source            = source,
            n_rollouts        = n_rollouts,
            n_candidates      = n_candidates,
            side              = side,
            price_level       = price_level,
            levels            = levels,
        )

        c_t = book_walk_cost(
            shares      = q_t,
            mid         = float(real_mid[t]),
            queue       = real_queue_raw[t],
            side        = side,
            price_level = price_level,
            levels      = levels,
        )

        actions[t]      = q_t
        costs[t]        = c_t
        Q_t             = max(Q_t - q_t, 0.0)
        Q_track[t + 1]  = Q_t

    return {
        "actions":     actions,
        "costs":       costs,
        "total_cost":  float(costs.sum()),
        "Q_remaining": Q_track,
    }


def run_twap_episode(
    *,
    real_mid:        np.ndarray,        # (T,) raw USDT
    real_queue_raw:  np.ndarray,        # (T, 2k) raw absolute
    Q_total:         float,
    side:            str,
    price_level:     float,
    levels:          int,
) -> Dict[str, np.ndarray]:
    """Deterministic TWAP episode: q_t = Q_total / T at every step.

    No model, no rollouts, no decision. Floor baseline.
    """
    T = len(real_mid)
    q_per_step = Q_total / T

    actions = np.full(T, q_per_step, dtype=np.float64)
    costs = np.array([
        book_walk_cost(
            shares      = q_per_step,
            mid         = float(real_mid[t]),
            queue       = real_queue_raw[t],
            side        = side,
            price_level = price_level,
            levels      = levels,
        )
        for t in range(T)
    ])
    Q_track = Q_total - np.concatenate([[0.0], np.cumsum(actions)])

    return {
        "actions":     actions,
        "costs":       costs,
        "total_cost":  float(costs.sum()),
        "Q_remaining": Q_track,
    }
