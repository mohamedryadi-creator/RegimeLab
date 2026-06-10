"""Smoke tests: every plotting helper produces a figure from realistic inputs."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="requires the notebooks extra")
matplotlib.use("Agg", force=True)

from regimelab import plotting  # noqa: E402


@pytest.fixture
def net_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.DataFrame(
        {"a": rng.normal(3e-4, 0.01, 300), "b": rng.normal(2e-4, 0.006, 300)}, index=idx
    )


def test_wealth_and_drawdown_curves(net_returns, tmp_path):
    fig = plotting.wealth_curves(net_returns)
    out = plotting.save_figure(fig, tmp_path / "sub" / "wealth.png")
    assert out.exists() and out.stat().st_size > 0
    assert plotting.drawdown_curves(net_returns) is not None


def test_regime_overlay(net_returns):
    prices = 100 * (1 + net_returns["a"]).cumprod()
    labels = (net_returns["a"].rolling(20).std() > 0.009).astype(float)
    prob = labels.rolling(5).mean()
    fig = plotting.regime_overlay(prices, labels, turbulent_probability=prob)
    assert len(fig.axes) >= 2


def test_bootstrap_forest():
    table = pd.DataFrame(
        {"diff": [0.17, 0.01], "ci_low": [-0.12, -0.2], "ci_high": [0.5, 0.22]},
        index=["a vs b", "c vs d"],
    )
    assert plotting.bootstrap_forest(table) is not None


def test_sensitivity_heatmap_and_subperiod_bars():
    pivot = pd.DataFrame(
        [[0.7, 0.8], [0.71, 0.75]], index=[10, 20], columns=[0.7, 0.8]
    )
    assert plotting.sensitivity_heatmap(pivot) is not None
    sharpe = pd.DataFrame(
        {"bh": [0.2, 0.9], "gated": [0.3, 1.0]}, index=["2005..2009", "2010..2019"]
    )
    assert plotting.subperiod_bars(sharpe) is not None
