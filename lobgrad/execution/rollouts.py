"""
rollouts.py
-----------
Sources of forecasted LOB sequences for the MPC cost-to-go estimator.

Every ``RolloutSource`` exposes a single method:

    sample(t, history, K, H) -> (mid_paths, queue_raw_paths)

where ``mid_paths`` has shape ``(K, H)`` raw-USDT mid-prices and
``queue_raw_paths`` has shape ``(K, H, 2k)`` raw absolute queue sizes (≥ 0,
laid out bid[0..k-1] | ask[k..2k-1] — matching ``cost.book_walk_cost``).

Four implementations:

    ModelRollouts        : wraps a generative model
                           (any object with .generate(ctx_q, ctx_a, steps,
                           n_paths, seed=) -> (queue_norm, mid_paths))
    BootstrapRollouts    : random contiguous slices from training-set history
                           (no learned model — tests "does learning help?")
    RealFutureRollouts   : the actually realized future LOB
                           (perfect-information ceiling — defines regret = 0)
    PersistenceRollouts  : current snapshot, repeated H times
                           (no-information sanity check; collapses to TWAP
                           under our cost model)

The MPC algorithm itself (see ``mpc.py``) doesn't care which source it is
calling — only the rollouts differ between baselines, the policy is identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..data.normalizer import Normalizer


class RolloutSource(ABC):
    """Abstract source of K rollouts of length H."""

    @abstractmethod
    def sample(
        self,
        t:       int,
        history: dict,
        K:       int,
        H:       int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mid_paths        : (K, H) raw-USDT mid-prices
        queue_raw_paths  : (K, H, 2k) raw absolute queue sizes (≥ 0)

        Parameters
        ----------
        t        : current step within the evaluation window (0-indexed).
                   Some sources (RealFuture) use this to index the future.
        history  : {"queue_norm": (T_so_far, 2k), "anchors": (T_so_far,)}
                   Real observations up to (but not including) step t. Sources
                   that condition on history (Model) use this; others ignore it.
        K        : number of rollouts to draw.
        H        : horizon of each rollout.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for tables / logs."""
        ...


# =====================================================================
# ModelRollouts — wraps a generative model
# =====================================================================
class ModelRollouts(RolloutSource):
    """Rollouts come from a generative model conditioned on recent history.

    The generator is expected to expose
        ``generate(context_queue, context_anchors, steps, n_paths, seed=None)
            -> (queue_norm: (n_paths, steps, 2k), mid_paths: (n_paths, steps))``
    matching ``JointTimeGradGenerator`` and friends.
    """

    def __init__(
        self,
        generator,
        normalizer:  Normalizer,
        context_len: int,
        label:       str,
    ):
        self.generator   = generator
        self.normalizer  = normalizer
        self.context_len = context_len
        self._label      = label

    def sample(self, t, history, K, H):
        ctx_q = history["queue_norm"][-self.context_len:]   # (context_len, 2k)
        ctx_a = history["anchors"][-self.context_len:]      # (context_len,)
        queue_norm, mid_paths = self.generator.generate(
            context_queue   = ctx_q,
            context_anchors = ctx_a,
            steps           = H,
            n_paths         = K,
        )
        # Inverse-transform queue → raw absolute sizes (≥ 0).
        queue_raw = np.stack([
            np.abs(self.normalizer.inverse_transform(queue_norm[k]))
            for k in range(K)
        ]).astype(np.float64)
        return mid_paths.astype(np.float64), queue_raw

    @property
    def name(self) -> str:
        return self._label


# =====================================================================
# BootstrapRollouts — random training-set slices
# =====================================================================
class BootstrapRollouts(RolloutSource):
    """Rollouts are random contiguous slices of the training-set LOB history.

    This is the "no learned model" baseline: marginal training distribution
    is preserved, but the rollout is *independent of the current state* —
    so any improvement of a learned model over Bootstrap is specifically
    attributable to state-conditional forecasting.

    Training queue is inverse-transformed to raw absolute sizes once at
    construction; per-call sampling is just slicing.
    """

    def __init__(
        self,
        train_queue_norm: np.ndarray,   # (T_train, 2k) normalized
        train_anchors:    np.ndarray,   # (T_train,) raw USDT
        normalizer:       Normalizer,
        rng:              Optional[np.random.Generator] = None,
    ):
        if len(train_queue_norm) != len(train_anchors):
            raise ValueError("train_queue_norm and train_anchors must have same length")
        self.train_anchors    = np.asarray(train_anchors, dtype=np.float64)
        self.train_queue_raw  = np.abs(
            normalizer.inverse_transform(np.asarray(train_queue_norm))
        ).astype(np.float64)
        self._rng = rng if rng is not None else np.random.default_rng(42)

    def sample(self, t, history, K, H):
        T = len(self.train_anchors)
        if T < H:
            raise ValueError(f"training history too short for rollouts of length {H}")
        starts = self._rng.integers(0, T - H + 1, size=K)
        mid   = np.stack([self.train_anchors[s:s+H] for s in starts])           # (K, H)
        queue = np.stack([self.train_queue_raw[s:s+H] for s in starts])         # (K, H, 2k)
        return mid, queue

    @property
    def name(self) -> str:
        return "Bootstrap"


# =====================================================================
# RealFutureRollouts — the actual realized future (oracle)
# =====================================================================
class RealFutureRollouts(RolloutSource):
    """K=1 deterministic 'rollout' equal to the realized future LOB.

    Bound to a single window's data (window_anchors, window_queue_raw) at
    construction. At step t, returns the next H steps from the bound arrays.
    This is the perfect-information ceiling that defines regret = 0.
    """

    def __init__(
        self,
        window_anchors:   np.ndarray,   # (T_window,) raw USDT
        window_queue_raw: np.ndarray,   # (T_window, 2k) raw absolute sizes
    ):
        if len(window_anchors) != len(window_queue_raw):
            raise ValueError("window_anchors and window_queue_raw must have same length")
        self.window_anchors   = np.asarray(window_anchors, dtype=np.float64)
        self.window_queue_raw = np.asarray(window_queue_raw, dtype=np.float64)

    def sample(self, t, history, K, H):
        end = t + H
        avail = len(self.window_anchors) - t
        if avail >= H:
            mid   = self.window_anchors[t:end]            # (H,)
            queue = self.window_queue_raw[t:end]          # (H, 2k)
        else:
            # Pad by repeating the last available snapshot — only triggers if
            # the caller asked for an out-of-range horizon. Shouldn't happen
            # in normal MPC use (caller bounds H to the remaining horizon).
            mid_avail   = self.window_anchors[t:t+avail]
            queue_avail = self.window_queue_raw[t:t+avail]
            pad_n       = H - avail
            if avail > 0:
                mid   = np.concatenate([mid_avail,   np.full(pad_n, mid_avail[-1])])
                queue = np.concatenate([queue_avail, np.tile(queue_avail[-1:], (pad_n, 1))])
            else:
                # Caller asked for steps entirely past the window — degenerate.
                mid   = np.zeros(H)
                queue = np.zeros((H, self.window_queue_raw.shape[1]))
        # K is ignored — perfect-info rollouts are deterministic. Broadcast to K=1.
        return mid[None, :], queue[None, :, :]

    @property
    def name(self) -> str:
        return "Oracle"


# =====================================================================
# PersistenceRollouts — current snapshot repeated H times
# =====================================================================
class PersistenceRollouts(RolloutSource):
    """K=1 rollout where every step equals the most recent observed snapshot.

    This is the no-information forecast. Under our cost model (where buying
    is super-linear in quantity) this collapses to TWAP behavior and exists
    primarily as a sanity check that the MPC pipeline matches the analytical
    reduction.
    """

    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def sample(self, t, history, K, H):
        last_q_norm = history["queue_norm"][-1]                                  # (2k,)
        last_q_raw  = np.abs(self.normalizer.inverse_transform(last_q_norm))     # (2k,)
        last_mid    = float(history["anchors"][-1])

        mid   = np.full((1, H), last_mid, dtype=np.float64)
        queue = np.tile(last_q_raw[None, None, :], (1, H, 1)).astype(np.float64)
        return mid, queue

    @property
    def name(self) -> str:
        return "Persistence"
