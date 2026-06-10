"""BandedStrategy: band semantics, zero-band equivalence, causality, turnover."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.engine import run_backtest
from regimelab.strategies import BandedStrategy, BuyAndHold, ProbabilityScaledStrategy


def weights_fixture(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    prices = pd.DataFrame({"X": 100.0}, index=idx)
    weights = pd.Series(values, index=idx)
    return prices, ProbabilityScaledStrategy(BuyAndHold(), weights)


def test_band_semantics_hand_example():
    prices, base = weights_fixture([0.0, 0.5, 0.55, 0.4, 1.0])
    pos = BandedStrategy(base, band=0.2).target_positions(prices)
    # start flat; trade to 0.5 (|0.5|>0.2); hold through 0.55 and 0.4
    # (moves of 0.05 and 0.10 are within the band); trade to 1.0.
    assert pos["X"].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.5, 1.0])


def test_zero_band_reproduces_base():
    rng = np.random.default_rng(0)
    prices, base = weights_fixture(rng.uniform(0, 1, 200))
    banded = BandedStrategy(base, band=0.0).target_positions(prices)
    pd.testing.assert_frame_equal(banded, base.target_positions(prices))


def test_banded_positions_are_causal():
    rng = np.random.default_rng(1)
    prices, base = weights_fixture(rng.uniform(0, 1, 150))
    strat = BandedStrategy(base, band=0.15)
    full = strat.target_positions(prices)
    for cut in (50, 100, 150):
        truncated = strat.target_positions(prices.iloc[:cut])
        pd.testing.assert_frame_equal(truncated, full.iloc[:cut])


def test_turnover_nonincreasing_in_band():
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame({"X": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    weights = pd.Series(rng.uniform(0, 1, n), index=idx)
    base = ProbabilityScaledStrategy(BuyAndHold(), weights)
    returns = prices.pct_change(fill_method=None).iloc[1:]

    turnovers = []
    for band in (0.0, 0.1, 0.2, 0.4):
        pos = BandedStrategy(base, band).target_positions(prices)
        turnovers.append(run_backtest(pos, returns).turnover.sum())
    assert turnovers == sorted(turnovers, reverse=True)
    assert turnovers[-1] < turnovers[0]


def test_negative_band_rejected():
    with pytest.raises(ValueError, match="band"):
        BandedStrategy(BuyAndHold(), band=-0.1)
