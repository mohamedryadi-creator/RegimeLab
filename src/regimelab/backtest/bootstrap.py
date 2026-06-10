"""Block-bootstrap inference for performance metrics.

A point difference between two Sharpe ratios is not a finding: daily returns
are autocorrelated and heavy-tailed, so resampling must preserve local
dependence. This module uses the *circular block bootstrap*: indices are drawn
in contiguous blocks (wrapping at the boundary) and percentile intervals are
formed over the bootstrap distribution.

For strategy comparisons, resampling is *paired*: the same blocks of dates are
applied to both return series, preserving their cross-correlation — the two
strategies live through the same resampled history. ``prob_nonpositive`` is the
fraction of bootstrap draws where the difference is <= 0; it is a descriptive
bootstrap quantity, not a formal p-value, and is labeled accordingly in
reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regimelab.backtest.metrics import TRADING_DAYS_PER_YEAR


def _sharpe(x: np.ndarray, periods_per_year: int) -> float:
    sd = x.std(ddof=1)
    if sd == 0.0 or np.isnan(sd):
        return np.nan
    return float(x.mean() / sd * np.sqrt(periods_per_year))


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices of length ``n``."""
    n_blocks = -(-n // block_size)  # ceil
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


@dataclass(frozen=True)
class SharpeCI:
    sharpe: float
    ci_low: float
    ci_high: float
    n_boot: int
    block_size: int


@dataclass(frozen=True)
class SharpeDiff:
    sharpe_a: float
    sharpe_b: float
    diff: float
    ci_low: float
    ci_high: float
    prob_nonpositive: float
    n_boot: int
    block_size: int


def sharpe_confidence_interval(
    returns: pd.Series,
    n_boot: int = 2000,
    block_size: int = 21,
    ci: float = 0.95,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> SharpeCI:
    """Percentile CI for the annualized Sharpe ratio of one return series."""
    x = returns.dropna().to_numpy()
    _validate(len(x), block_size, ci)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = _sharpe(x[_block_indices(len(x), block_size, rng)], periods_per_year)
    lo, hi = np.nanquantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return SharpeCI(
        sharpe=_sharpe(x, periods_per_year),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        block_size=block_size,
    )


def sharpe_difference(
    a: pd.Series,
    b: pd.Series,
    n_boot: int = 2000,
    block_size: int = 21,
    ci: float = 0.95,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> SharpeDiff:
    """Paired percentile CI for ``sharpe(a) - sharpe(b)``.

    Both series must share an identical index (the same backtest period);
    each bootstrap draw applies the same date blocks to both.
    """
    if not a.index.equals(b.index):
        raise ValueError("Series must share an identical index for a paired comparison.")
    xa, xb = a.to_numpy(), b.to_numpy()
    _validate(len(xa), block_size, ci)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(xa), block_size, rng)
        diffs[i] = _sharpe(xa[idx], periods_per_year) - _sharpe(xb[idx], periods_per_year)
    lo, hi = np.nanquantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return SharpeDiff(
        sharpe_a=_sharpe(xa, periods_per_year),
        sharpe_b=_sharpe(xb, periods_per_year),
        diff=_sharpe(xa, periods_per_year) - _sharpe(xb, periods_per_year),
        ci_low=float(lo),
        ci_high=float(hi),
        prob_nonpositive=float(np.mean(diffs <= 0.0)),
        n_boot=n_boot,
        block_size=block_size,
    )


def _validate(n: int, block_size: int, ci: float) -> None:
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if n <= 2 * block_size:
        raise ValueError(f"Need more than {2 * block_size} observations, got {n}.")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1).")
