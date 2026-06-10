"""Heuristic regime filters.

Deliberately simple, transparent rules. They serve as cheap benchmarks: a
statistical model that cannot beat these out-of-sample is not earning its
complexity.

Both models emit float labels per the :mod:`regimelab.regimes.base` contract
(NaN during warm-up) and are causal by construction (trailing windows only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimelab.features.engineering import moving_average, rolling_volatility
from regimelab.regimes.base import RegimeModel, single_price_column


class VolatilityThresholdRegime(RegimeModel):
    """Calm (0) vs. turbulent (1) based on trailing realized volatility.

    ``fit`` estimates the threshold as the ``quantile`` of trailing volatility
    over the *training* sample; ``predict_regimes`` labels a date turbulent when
    its trailing volatility exceeds that fixed threshold. The threshold is the
    only fitted state, so out-of-sample predictions cannot leak test data.
    """

    n_regimes = 2

    def __init__(self, vol_window: int = 20, quantile: float = 0.8):
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")
        self.vol_window = vol_window
        self.quantile = quantile
        self.threshold_: float | None = None

    def _volatility(self, X: pd.DataFrame) -> pd.Series:
        prices = single_price_column(X)
        returns = prices.pct_change(fill_method=None)
        return rolling_volatility(returns, self.vol_window)

    def fit(self, X: pd.DataFrame) -> "VolatilityThresholdRegime":
        vol = self._volatility(X).dropna()
        if vol.empty:
            raise ValueError(
                f"Training sample too short for vol_window={self.vol_window}."
            )
        self.threshold_ = float(vol.quantile(self.quantile))
        return self

    def predict_regimes(self, X: pd.DataFrame) -> pd.Series:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before predict_regimes().")
        vol = self._volatility(X)
        labels = (vol > self.threshold_).astype(float)
        labels[vol.isna()] = np.nan
        return labels.rename("regime")


class TrendRegime(RegimeModel):
    """Downtrend (0) vs. uptrend (1): price above its trailing moving average.

    Stateless — ``fit`` is a no-op kept for interface uniformity.
    """

    n_regimes = 2

    def __init__(self, window: int = 200):
        self.window = window

    def fit(self, X: pd.DataFrame) -> "TrendRegime":
        return self

    def predict_regimes(self, X: pd.DataFrame) -> pd.Series:
        prices = single_price_column(X)
        ma = moving_average(prices, self.window)
        labels = (prices > ma).astype(float)
        labels[ma.isna()] = np.nan
        return labels.rename("regime")
