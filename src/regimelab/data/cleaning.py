"""Data cleaning and quality checks.

Cleaning is conservative and fails loudly: impossible values raise, suspicious
values warn. Every transformation here is a research decision recorded in
reports/decisions.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# A single-day move beyond this fraction is flagged as a possible data error
# (real crashes can exceed it — hence a warning, not an exception).
JUMP_THRESHOLD = 0.5


def clean_prices(prices: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Validate and clean a raw price DataFrame.

    - sorts the index and drops duplicate dates (keeping the first);
    - raises on non-positive prices;
    - warns on single-day moves beyond ``JUMP_THRESHOLD``;
    - forward-fills gaps up to ``max_ffill`` periods (holidays, stale quotes);
    - drops rows that are still incomplete (typically the head, before all
      assets have history).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")

    df = prices.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if (df <= 0).any().any():
        bad = df.columns[(df <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found in columns {bad}.")

    rel_change = df.pct_change(fill_method=None).abs()
    jumps = rel_change > JUMP_THRESHOLD
    if jumps.any().any():
        dates = df.index[jumps.any(axis=1)].tolist()
        warnings.warn(
            f"Single-day moves beyond {JUMP_THRESHOLD:.0%} on {dates}; "
            "verify these are real market moves, not data errors.",
            stacklevel=2,
        )

    df = df.ffill(limit=max_ffill).dropna()
    if df.empty:
        raise ValueError("No complete rows remain after cleaning.")
    return df


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Convert prices to returns.

    ``kind="simple"`` (arithmetic) is what the backtest engine consumes —
    portfolio returns aggregate linearly across assets. ``kind="log"`` is for
    statistical modeling (regime models), where time-additivity matters.
    """
    if kind == "simple":
        returns = prices.pct_change(fill_method=None)
    elif kind == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}.")
    return returns.iloc[1:]
