"""
base.py
-------
Abstract base class for all LOB generators used in the benchmark.

Feature representation: 2k-dimensional signed queue vector.
  indices 0..k-1   : bid queues (≤ 0), shallow→deep
  indices k..2k-1  : ask queues (≥ 0), shallow→deep
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class LOBGenerator(ABC):

    @abstractmethod
    def fit(
        self,
        snapshots:  np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """
        Train / calibrate on normalised training data.

        Parameters
        ----------
        snapshots  : (T, 2k)  normalised signed queue features
        timestamps : (T,)     Unix timestamps in ms
        """

    @abstractmethod
    def generate(
        self,
        context: np.ndarray,
        steps:   int,
        n_paths: int,
        rng:     Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate future LOB snapshots conditioned on a context window.

        Parameters
        ----------
        context : (context_len, 2k) normalised
        steps   : number of future snapshots to generate
        n_paths : number of independent sample paths
        rng     : optional numpy RNG for reproducibility

        Returns
        -------
        features : (n_paths, steps, 2k)  normalised signed queue features
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
