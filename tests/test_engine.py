"""Backtest engine invariants."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.costs import ProportionalCost
from regimelab.backtest.engine import run_backtest


def make_returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx, name="X")


def test_zero_positions_zero_returns():
    r = make_returns()
    pos = pd.Series(0.0, index=r.index)
    result = run_backtest(pos, r)
    assert (result.net_returns == 0).all()
    assert (result.turnover == 0).all()


def test_buy_and_hold_reproduces_asset_return():
    r = make_returns()
    # Decide full investment one day before the return series starts, so the
    # lagged position is already 1 on the first return date.
    pos_index = r.index.insert(0, r.index[0] - pd.offsets.BDay(1))
    pos = pd.Series(1.0, index=pos_index)
    result = run_backtest(pos, r)
    assert (1 + result.net_returns).prod() == pytest.approx((1 + r).prod())
    # One initial trade, then zero turnover.
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert (result.turnover.iloc[1:] == 0).all()


def test_costs_reduce_net_returns_monotonically():
    r = make_returns()
    rng = np.random.default_rng(1)
    pos = pd.Series(rng.integers(0, 2, len(r)).astype(float), index=r.index)
    total = {
        bps: run_backtest(pos, r, cost_model=ProportionalCost(bps)).net_returns.sum()
        for bps in (0, 5, 10)
    }
    assert total[0] > total[5] > total[10]
    gross = run_backtest(pos, r, cost_model=ProportionalCost(5)).gross_returns.sum()
    assert total[0] == pytest.approx(gross)


def test_engine_matches_manual_computation():
    r = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range("2024-01-01", periods=3))
    pos = pd.Series([1.0, 0.5, 0.0], index=r.index)
    result = run_backtest(pos, r, cost_model=ProportionalCost(10))
    held = [0.0, 1.0, 0.5]
    gross = [0.0, 1.0 * -0.02, 0.5 * 0.03]
    turnover = [0.0, 1.0, 0.5]
    net = [g - t * 10e-4 for g, t in zip(gross, turnover, strict=True)]
    assert result.positions.iloc[:, 0].tolist() == pytest.approx(held)
    assert result.gross_returns.tolist() == pytest.approx(gross)
    assert result.turnover.tolist() == pytest.approx(turnover)
    assert result.net_returns.tolist() == pytest.approx(net)


def test_nan_inputs_fail_loudly():
    r = make_returns(10)
    pos = pd.Series(1.0, index=r.index)
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(pos.where(pos.index != r.index[3]), r)
    with pytest.raises(ValueError, match="NaN"):
        run_backtest(pos, r.where(r.index != r.index[3]))


def test_mismatched_columns_fail():
    r = make_returns(10).to_frame()
    pos = pd.DataFrame(
        {"A": 1.0, "B": 0.0}, index=r.index
    )
    with pytest.raises(ValueError, match="columns"):
        run_backtest(pos, r)
