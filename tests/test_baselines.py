"""Baseline strategies on toy series where correct positions are known by hand."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.engine import run_backtest
from regimelab.strategies.baselines import BuyAndHold, TrendFollowing


def test_buy_and_hold_full_investment_equal_weight():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    pos = BuyAndHold().target_positions(prices)
    assert pos.sum(axis=1).tolist() == pytest.approx([1.0] * len(idx))
    assert (pos == 0.5).all().all()


def test_trend_following_uptrend_and_downtrend():
    idx = pd.bdate_range("2020-01-01", periods=40)
    up = pd.DataFrame({"X": np.linspace(100, 140, 40)}, index=idx)
    down = pd.DataFrame({"X": np.linspace(140, 100, 40)}, index=idx)
    strat = TrendFollowing(window=10)

    pos_up = strat.target_positions(up)
    # Flat during MA warm-up, then long for the whole uptrend.
    assert (pos_up["X"].iloc[: 9] == 0.0).all()
    assert (pos_up["X"].iloc[9:] == 1.0).all()

    pos_down = strat.target_positions(down)
    assert (pos_down["X"] == 0.0).all()


def test_trend_following_no_nans_ever():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        {"X": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))}, index=idx
    )
    pos = TrendFollowing(window=200).target_positions(prices)
    assert not pos.isna().any().any()


def test_trend_following_runs_through_engine():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=400)
    prices = pd.DataFrame(
        {"X": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400)))}, index=idx
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    pos = TrendFollowing(window=50).target_positions(prices)
    result = run_backtest(pos, returns)
    assert len(result.net_returns) == len(returns)
    assert result.turnover.sum() > 0
