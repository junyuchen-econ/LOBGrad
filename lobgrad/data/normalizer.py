"""
normalizer.py
-------------
Normalisation for the 2k-dimensional queue representation.

Two representation modes (set at construction via `mode`):

  "signed"   (default)
      Raw input has bids stored as negative values and asks as positive.
      Transform  : sign(Q) * log(|Q| + 1)   — preserves sign, maps 0 → 0
      Inverse    : sign(v) * (exp(|v|) - 1)

  "unsigned"
      Raw input has all volumes as non-negative values (bids and asks ≥ 0).
      Transform  : log(Q + 1)
      Inverse    : exp(v) - 1

Two queue content modes (controlled externally via BenchmarkConfig.queue_mode):

  "absolute"  — model raw queue sizes directly (what the normalizer transforms)
  "delta"     — model step-wise changes in queue sizes

  For delta mode use the `to_delta()` / `from_delta()` helpers below which
  convert between absolute-normalised features and delta-normalised features.
  Delta features always use signed log compression (changes can be negative
  even when the absolute representation is unsigned).

No fitted parameters are needed; fit() is a no-op kept for API consistency.

Layout (per snapshot):
  indices 0..k-1   : bid queues  (shallow → deep)
  indices k..2k-1  : ask queues  (shallow → deep)
"""

from __future__ import annotations

import numpy as np


class Normalizer:
    """
    Normaliser for 2k-dim queue vectors.

    Parameters
    ----------
    levels : int
        Number of levels per side (k).
    mode : {"signed", "unsigned"}
        "signed"   – bids stored as negative values; uses sign(Q)*log(|Q|+1).
        "unsigned" – all volumes non-negative;       uses log(Q+1).
    """

    def __init__(self, levels: int = 5, mode: str = "signed"):
        if mode not in ("signed", "unsigned"):
            raise ValueError(f"mode must be 'signed' or 'unsigned', got '{mode}'")
        self.levels = levels
        self.mode   = mode

    # ------------------------------------------------------------------
    # Core absolute transform
    # ------------------------------------------------------------------

    def fit(self, snapshots: np.ndarray) -> "Normalizer":
        """No-op — transform is parameter-free. Kept for API consistency."""
        return self

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Forward transform of absolute queue sizes.

        signed   : sign(Q) * log(|Q| + 1)
        unsigned : log(Q + 1)   (assumes Q ≥ 0)
        """
        out     = np.asarray(snapshots, dtype=np.float32).copy()
        squeeze = out.ndim == 1
        if squeeze:
            out = out[np.newaxis, :]

        if self.mode == "signed":
            out = np.sign(out) * np.log(np.abs(out) + 1.0)
        else:
            out = np.log(np.maximum(out, 0.0) + 1.0)

        return out[0] if squeeze else out

    def inverse_transform(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Inverse transform back to raw absolute queue sizes.

        signed   : sign(v) * (exp(|v|) - 1)
        unsigned : exp(v) - 1
        """
        out     = np.asarray(snapshots, dtype=np.float32).copy()
        squeeze = out.ndim == 1
        if squeeze:
            out = out[np.newaxis, :]

        if self.mode == "signed":
            out = np.sign(out) * np.expm1(np.abs(out))
        else:
            out = np.expm1(np.maximum(out, 0.0))

        return out[0] if squeeze else out

    def fit_transform(self, snapshots: np.ndarray) -> np.ndarray:
        return self.transform(snapshots)

    # ------------------------------------------------------------------
    # Delta-mode helpers
    # ------------------------------------------------------------------

    def to_delta(self, snapshots: np.ndarray) -> np.ndarray:
        """
        Convert absolute-normalised features → delta-normalised features.

        Steps
        -----
        1. Inverse-transform to raw absolute queue sizes.
        2. Compute step-wise differences  ΔQ_t = Q_t − Q_{t-1}.
           The first step is set to zero (no previous state).
        3. Apply signed log compression to the changes:
              sign(ΔQ) * log(|ΔQ| + 1)
           (always signed, regardless of the absolute mode, because changes
           can be negative even when absolute volumes are unsigned).

        Parameters
        ----------
        snapshots : (T, 2k) absolute-normalised features

        Returns
        -------
        (T, 2k) delta-normalised features
        """
        raw = self.inverse_transform(snapshots)          # (T, 2k) raw sizes
        delta_raw = np.zeros_like(raw, dtype=np.float32)
        delta_raw[1:] = raw[1:] - raw[:-1]              # ΔQ_t  (first row = 0)
        return np.sign(delta_raw) * np.log(np.abs(delta_raw) + 1.0)

    def from_delta(
        self,
        delta_features: np.ndarray,
        initial_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct absolute-normalised features from delta-normalised features.

        Parameters
        ----------
        delta_features : (T, 2k)  delta-normalised (output of to_delta / model)
        initial_raw    : (2k,)    raw absolute queue sizes at t = −1
                                  (last step of the conditioning context)

        Returns
        -------
        (T, 2k) absolute-normalised features
        """
        raw_deltas = np.sign(delta_features) * np.expm1(np.abs(delta_features))
        raw_abs = initial_raw[np.newaxis, :] + np.cumsum(raw_deltas, axis=0)
        if self.mode == "unsigned":
            raw_abs = np.maximum(raw_abs, 0.0)
        return self.transform(raw_abs)
