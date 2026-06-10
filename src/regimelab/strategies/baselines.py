"""Baseline strategies.

These define the reference results that any regime-aware strategy must beat
out-of-sample and after costs. Both take a price DataFrame as ``data`` and
return long-only weights summing to at most 1 across assets. The backtest
engine applies the execution lag — strategies do not shift their own signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regimelab.strategies.base import Strategy


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    """Constant full investment, equal-weighted across assets."""

    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = data
        weight = 1.0 / prices.shape[1]
        return pd.DataFrame(weight, index=prices.index, columns=prices.columns)


@dataclass(frozen=True)
class TrendFollowing(Strategy):
    """Long an asset when its price is above its trailing moving average, else cash.

    ``window`` is a research parameter, not a tuning knob: experiments must
    report sensitivity across a range (e.g. 100-300 days), not a single
    optimized value.
    """

    window: int = 200

    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = data
        ma = prices.rolling(self.window).mean()
        # NaN MA (warm-up) compares False -> position 0: flat until enough history.
        signal = prices.gt(ma).astype(float)
        return signal / prices.shape[1]
