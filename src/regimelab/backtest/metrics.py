"""Performance and risk metrics.

All metrics are small pure functions of a net (simple) return series. Conventions:

- Returns are *simple* periodic returns (the backtest engine produces these).
- Annualization uses an explicit ``periods_per_year`` (252 for daily data).
- Sharpe/Sortino assume a zero risk-free rate unless ``rf`` (annualized) is given;
  this choice is recorded in reports/decisions.md.
- An empty or constant series yields ``nan`` rather than raising, so summary
  tables can always be built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualized return of a simple-return series."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0.0:
        return -1.0
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Annualized Sharpe ratio; ``rf`` is an annualized rate, converted to per-period."""
    r = returns.dropna() - rf / periods_per_year
    if len(r) < 2:
        return np.nan
    sd = float(r.std(ddof=1))
    if sd == 0.0 or np.isnan(sd):
        return np.nan
    return float(r.mean()) / sd * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Annualized Sortino ratio with downside deviation from the full sample.

    Downside deviation is sqrt(mean(min(r, 0)^2)) — zeros count, so a series
    that is rarely negative is rewarded.
    """
    r = returns.dropna() - rf / periods_per_year
    if len(r) < 2:
        return np.nan
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if downside == 0.0:
        return np.nan
    return float(r.mean()) / downside * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown (<= 0) of the compounded wealth curve at each date.

    The peak includes the initial capital of 1.0, so a series that only
    declines is already in drawdown from the first period.
    """
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if len(returns.dropna()) == 0:
        return np.nan
    return float(drawdown_series(returns).min())


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of positive periods among periods with a nonzero return.

    Zero-return periods (e.g. flat positions) are excluded so that sitting in
    cash does not mechanically alter the ratio.
    """
    active = returns.dropna()
    active = active[active != 0.0]
    if len(active) == 0:
        return np.nan
    return float((active > 0.0).mean())


def summary(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    name: str = "strategy",
) -> pd.DataFrame:
    """One-row summary table; the uniform output format for all experiments."""
    row = {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_ratio": hit_ratio(returns),
        "n_periods": int(returns.dropna().shape[0]),
    }
    if turnover is not None:
        row["avg_annual_turnover"] = float(turnover.mean() * periods_per_year)
    return pd.DataFrame(row, index=pd.Index([name], name="strategy"))
