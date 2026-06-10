"""Feature engineering.

Point-in-time discipline: every feature value at date ``t`` depends only on data
up to and including ``t``. Rolling windows are trailing, never centered; no
full-sample statistics. ``tests/test_lookahead.py::test_features_are_causal``
enforces this contract for every feature listed in ``CAUSAL_FEATURES``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimelab.backtest.metrics import TRADING_DAYS_PER_YEAR


def rolling_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Trailing realized volatility of ``returns`` over ``window`` observations."""
    vol = returns.rolling(window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Trailing ``window``-period total return."""
    return prices.pct_change(window, fill_method=None)


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Trailing simple moving average."""
    return prices.rolling(window).mean()


# Registry of (factory, input) pairs used by the causality test; every new
# feature must be added here so it is automatically checked for look-ahead.
CAUSAL_FEATURES = [
    (lambda s: rolling_volatility(s, window=20), "returns"),
    (lambda s: momentum(s, window=20), "prices"),
    (lambda s: moving_average(s, window=20), "prices"),
]
