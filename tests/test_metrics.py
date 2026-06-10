"""Metrics tested against hand-computed examples."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest import metrics

# Hand example: two periods, +10% then -5%.
# wealth: 1.10 -> 1.045; max drawdown = 1.045/1.10 - 1 = -0.05
R = pd.Series([0.10, -0.05])


def test_annualized_return():
    expected = 1.045 ** (252 / 2) - 1
    assert metrics.annualized_return(R) == pytest.approx(expected)


def test_annualized_volatility():
    expected = np.std([0.10, -0.05], ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(R) == pytest.approx(expected)


def test_sharpe_ratio():
    expected = 0.025 / np.std([0.10, -0.05], ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(R) == pytest.approx(expected)


def test_sortino_ratio():
    downside = np.sqrt((0.0**2 + 0.05**2) / 2)
    expected = 0.025 / downside * np.sqrt(252)
    assert metrics.sortino_ratio(R) == pytest.approx(expected)


def test_max_drawdown():
    assert metrics.max_drawdown(R) == pytest.approx(-0.05)


def test_max_drawdown_monotone_decline():
    r = pd.Series([-0.10, -0.10, -0.10])
    assert metrics.max_drawdown(r) == pytest.approx(0.9**3 - 1)


def test_hit_ratio_excludes_flat_periods():
    r = pd.Series([0.02, 0.0, -0.01, 0.0, 0.03])
    assert metrics.hit_ratio(r) == pytest.approx(2 / 3)


def test_degenerate_series_give_nan_not_errors():
    empty = pd.Series(dtype=float)
    constant = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(metrics.annualized_return(empty))
    assert np.isnan(metrics.sharpe_ratio(constant))  # zero volatility
    assert np.isnan(metrics.sortino_ratio(constant))
    assert np.isnan(metrics.hit_ratio(constant))


def test_summary_is_one_row():
    table = metrics.summary(R, turnover=pd.Series([1.0, 0.0]), name="toy")
    assert table.shape[0] == 1
    assert table.index[0] == "toy"
    assert table.loc["toy", "avg_annual_turnover"] == pytest.approx(0.5 * 252)
