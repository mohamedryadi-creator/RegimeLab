"""The common interface for all regime detection models.

Fixing this interface early keeps the methodology flexible: heuristic filters,
HMMs, and Markov-switching models all plug into the same backtesting and
validation machinery.

Contract
--------
- ``X`` is a price DataFrame with a ``DatetimeIndex`` (single column for now —
  regime models are per-asset). Models derive whatever features they need
  (returns, rolling volatility, ...) internally, so composition stays uniform.
- ``fit(X)`` may only be called on training data; within walk-forward
  validation, :func:`regimelab.validation.walkforward.walk_forward_predict`
  controls what ``X`` contains.
- ``predict_regimes(X)`` must be *causal*: the label at date ``t`` may only
  depend on data up to and including ``t`` (filtered, not smoothed,
  probabilities for HMM-type models — smoothing uses future data and is for
  diagnostics only). Causality is enforced by truncation tests.
- Labels are a float Series: regime identifiers (0.0, 1.0, ...) with ``NaN``
  during warm-up, when the model cannot yet emit a label. Consumers decide what
  to do with NaN; :class:`regimelab.strategies.regime_gated.RegimeGatedStrategy`
  maps it to a conservative default exposure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RegimeModel(ABC):
    """Abstract base class for regime detection models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "RegimeModel":
        """Fit the model on training prices ``X``."""

    @abstractmethod
    def predict_regimes(self, X: pd.DataFrame) -> pd.Series:
        """Return a causal regime label per date in ``X`` (NaN during warm-up)."""

    @property
    def n_regimes(self) -> int:
        """Number of regimes the model distinguishes, if defined."""
        raise NotImplementedError


def single_price_column(X: pd.DataFrame) -> pd.Series:
    """Extract the single price column regime models currently operate on."""
    if isinstance(X, pd.Series):
        return X
    if X.shape[1] != 1:
        raise ValueError(
            f"Regime models are per-asset: expected one price column, got {list(X.columns)}."
        )
    return X.iloc[:, 0]
