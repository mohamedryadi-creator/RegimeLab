"""RegimeGatedStrategy: exposure mapping, NaN handling, engine integration."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.engine import run_backtest
from regimelab.strategies import BuyAndHold, RegimeGatedStrategy


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=8)
    return pd.DataFrame({"X": 100.0}, index=idx)


def test_exposure_mapping(prices):
    regimes = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], index=prices.index)
    strat = RegimeGatedStrategy(BuyAndHold(), regimes, exposure={0.0: 1.0, 1.0: 0.0})
    pos = strat.target_positions(prices)
    assert pos["X"].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def test_partial_exposure(prices):
    regimes = pd.Series(1.0, index=prices.index)
    strat = RegimeGatedStrategy(BuyAndHold(), regimes, exposure={0.0: 1.0, 1.0: 0.25})
    pos = strat.target_positions(prices)
    assert (pos["X"] == 0.25).all()


def test_nan_and_missing_regimes_use_default(prices):
    regimes = pd.Series(
        [np.nan, 0.0, 0.0, 0.0], index=prices.index[:4]
    )  # NaN warm-up, and no labels at all for the last 4 dates
    strat = RegimeGatedStrategy(BuyAndHold(), regimes, default_exposure=0.0)
    pos = strat.target_positions(prices)
    assert pos["X"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_unknown_label_uses_default(prices):
    regimes = pd.Series(2.0, index=prices.index)  # label not in the exposure map
    strat = RegimeGatedStrategy(BuyAndHold(), regimes, exposure={0.0: 1.0, 1.0: 0.5})
    pos = strat.target_positions(prices)
    assert (pos["X"] == 0.0).all()


def test_gating_avoids_flagged_crash_through_engine():
    """Gated strategy sidesteps a crash its regime labels flag one day early."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.01, 0.01, -0.20, -0.20, 0.01, 0.01], index=idx, name="X")
    prices = pd.DataFrame({"X": 100 * (1 + r).cumprod()}, index=idx)
    # Labels turn turbulent at t=2; with the engine's one-day lag the gated
    # strategy is flat at t=3 and so dodges the second crash day only.
    regimes = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], index=idx)
    gated = RegimeGatedStrategy(BuyAndHold(), regimes).target_positions(prices)
    plain = BuyAndHold().target_positions(prices)
    res_gated = run_backtest(gated, r)
    res_plain = run_backtest(plain, r)
    assert res_gated.net_returns.iloc[3] == 0.0  # flat during second crash day
    assert res_gated.net_returns.iloc[2] == pytest.approx(-0.20)  # too late for the first
    assert (1 + res_gated.net_returns).prod() > (1 + res_plain.net_returns).prod()
