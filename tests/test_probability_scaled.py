"""ProbabilityScaledStrategy: scaling, clipping, NaN handling, per-asset weights."""

import numpy as np
import pandas as pd
import pytest

from regimelab.strategies import BuyAndHold, ProbabilityScaledStrategy, RegimeGatedStrategy


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame({"X": 100.0}, index=idx)


def test_continuous_scaling(prices):
    weights = pd.Series([1.0, 0.8, 0.5, 0.2, 0.0, 1.0], index=prices.index)
    strat = ProbabilityScaledStrategy(BuyAndHold(), weights)
    pos = strat.target_positions(prices)
    assert pos["X"].tolist() == pytest.approx([1.0, 0.8, 0.5, 0.2, 0.0, 1.0])


def test_weights_clipped_to_unit_interval(prices):
    weights = pd.Series([1.4, -0.2, 0.5, 0.5, 0.5, 0.5], index=prices.index)
    pos = ProbabilityScaledStrategy(BuyAndHold(), weights).target_positions(prices)
    assert pos["X"].iloc[0] == 1.0
    assert pos["X"].iloc[1] == 0.0


def test_nan_and_missing_weights_use_default(prices):
    weights = pd.Series([np.nan, 0.7], index=prices.index[:2])
    pos = ProbabilityScaledStrategy(BuyAndHold(), weights).target_positions(prices)
    assert pos["X"].tolist() == pytest.approx([0.0, 0.7, 0.0, 0.0, 0.0, 0.0])


def test_binary_weights_match_hard_gate(prices):
    """With 0/1 weights, probability scaling and the regime gate coincide."""
    labels = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], index=prices.index)
    weights = 1.0 - labels  # exposure 1 in calm (0), 0 in turbulent (1)
    soft = ProbabilityScaledStrategy(BuyAndHold(), weights).target_positions(prices)
    hard = RegimeGatedStrategy(BuyAndHold(), labels).target_positions(prices)
    pd.testing.assert_frame_equal(soft, hard)


def test_per_asset_weights():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 0.5, 0.0, 1.0], "B": [0.0, 1.0, 1.0, 0.5]}, index=idx)
    pos = ProbabilityScaledStrategy(BuyAndHold(), weights).target_positions(prices)
    # BuyAndHold weights are 0.5 each; scaled per asset.
    assert pos["A"].tolist() == pytest.approx([0.5, 0.25, 0.0, 0.5])
    assert pos["B"].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.25])


def test_per_asset_weights_require_matching_columns():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    weights = pd.DataFrame({"A": 1.0, "C": 1.0}, index=idx)
    with pytest.raises(ValueError, match="columns"):
        ProbabilityScaledStrategy(BuyAndHold(), weights).target_positions(prices)
