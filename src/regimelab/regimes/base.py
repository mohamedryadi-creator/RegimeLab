"""The common interface for all regime detection models.

Fixing this interface early keeps the methodology flexible: heuristic filters,
HMMs, and Markov-switching models all plug into the same backtesting and
validation machinery.

Contract
--------
- ``fit(X)`` may only be called on training data; within walk-forward validation
  the validation module controls what ``X`` contains.
- ``predict_regimes(X)`` must be *causal*: the regime label at date ``t`` may only
  depend on data up to and including ``t`` (filtered, not smoothed, probabilities
  for HMM-type models — smoothing uses future data and is for diagnostics only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RegimeModel(ABC):
    """Abstract base class for regime detection models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "RegimeModel":
        """Fit the model on training data ``X`` (features indexed by date)."""

    @abstractmethod
    def predict_regimes(self, X: pd.DataFrame) -> pd.Series:
        """Return a causal regime label (or probability) per date in ``X``."""

    @property
    def n_regimes(self) -> int:
        """Number of regimes the model distinguishes, if defined."""
        raise NotImplementedError
