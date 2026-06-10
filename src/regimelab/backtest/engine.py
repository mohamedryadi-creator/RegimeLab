"""The vectorized backtesting engine.

This is the single place where the execution lag lives: target positions decided
at the close of ``t`` are held during ``t + lag`` (default ``lag=1``) and earn
that period's return. Strategies never lag their own signals.

Conventions and simplifications (documented, deliberate):
- Positions are portfolio weights; the portfolio is implicitly rebalanced back
  to its target every period (standard vectorized approximation at daily
  frequency — intra-period weight drift is ignored).
- ``turnover(t) = sum_assets |held(t) - held(t-1)|``, with the initial position
  counted as a trade. Costs are charged at ``t``.
- NaNs in inputs fail loudly: a NaN position or return is a bug upstream, not
  something to silently fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regimelab.backtest.costs import CostModel


@dataclass(frozen=True)
class BacktestResult:
    """Everything downstream analysis needs, from one place."""

    net_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame  # weights actually held during each period
    turnover: pd.Series
    costs: pd.Series

    @property
    def total_cost_drag(self) -> float:
        return float(self.costs.sum())


def run_backtest(
    target_positions: pd.Series | pd.DataFrame,
    asset_returns: pd.Series | pd.DataFrame,
    cost_model: CostModel | None = None,
    lag: int = 1,
) -> BacktestResult:
    """Backtest ``target_positions`` against ``asset_returns``.

    ``target_positions[t]`` is the weight decided at the close of ``t``; it is
    held during ``t + lag``. ``lag`` must be >= 1 — a zero lag would let
    positions earn the same period's return they were computed from.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1; lag=0 introduces look-ahead bias.")

    tp = target_positions
    ar = asset_returns
    if isinstance(tp, pd.Series):
        tp = tp.to_frame()
    if isinstance(ar, pd.Series):
        ar = ar.to_frame()
    if tp.shape[1] == 1 and ar.shape[1] == 1:
        tp = tp.set_axis(ar.columns, axis=1)
    elif set(tp.columns) == set(ar.columns):
        tp = tp[ar.columns]
    else:
        raise ValueError(
            f"target_positions columns {list(tp.columns)} do not match "
            f"asset_returns columns {list(ar.columns)}."
        )

    if tp.isna().any().any():
        raise ValueError("target_positions contains NaN; strategies must be explicit (use 0).")
    if ar.isna().any().any():
        raise ValueError("asset_returns contains NaN; clean the data first.")

    # Lag on the strategy's own index, then align to the return index. Dates
    # with no prior decision (warm-up) are flat by construction.
    held = tp.shift(lag).reindex(ar.index).fillna(0.0)

    gross = (held * ar).sum(axis=1)

    trades = held.diff()
    trades.iloc[0] = held.iloc[0]
    turnover = trades.abs().sum(axis=1)

    costs = cost_model(turnover) if cost_model is not None else turnover * 0.0
    net = gross - costs

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        positions=held,
        turnover=turnover,
        costs=costs,
    )
