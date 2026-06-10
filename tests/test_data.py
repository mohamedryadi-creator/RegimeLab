"""Data loading and cleaning tests (no network: cache-based)."""

import numpy as np
import pandas as pd
import pytest

from regimelab.data.cleaning import clean_prices, to_returns
from regimelab.data.loaders import load_prices, load_risk_free


@pytest.fixture
def cached_prices(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(np.linspace(100, 110, 10), index=idx, name="close")
    close.to_csv(tmp_path / "TEST.csv", index_label="date")
    return tmp_path, close


def test_load_prices_from_cache(cached_prices):
    cache_dir, close = cached_prices
    prices = load_prices(["TEST"], cache_dir=cache_dir)
    assert list(prices.columns) == ["TEST"]
    assert prices["TEST"].tolist() == pytest.approx(close.tolist())


def test_load_prices_date_slicing(cached_prices):
    cache_dir, close = cached_prices
    prices = load_prices(["TEST"], start="2020-01-06", end="2020-01-09", cache_dir=cache_dir)
    assert prices.index.min() >= pd.Timestamp("2020-01-06")
    assert prices.index.max() <= pd.Timestamp("2020-01-09")


def test_load_prices_empty_range_raises(cached_prices):
    cache_dir, _ = cached_prices
    with pytest.raises(ValueError, match="No price data"):
        load_prices(["TEST"], start="2030-01-01", cache_dir=cache_dir)


def test_load_risk_free_conversion_and_lag(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=4)
    # Annualized percent yields, varying so the one-day lag is observable.
    yields = pd.Series([2.52, 5.04, 5.04, 5.04], index=idx, name="close")
    yields.to_csv(tmp_path / "IRX.csv", index_label="date")

    rf = load_risk_free(cache_dir=tmp_path, ticker="^IRX")
    # First date is consumed by the lag; day 2 accrues day 1's known rate.
    assert rf.index[0] == idx[1]
    assert rf.iloc[0] == pytest.approx(2.52 / 100 / 252)
    assert rf.iloc[1] == pytest.approx(5.04 / 100 / 252)


def test_load_risk_free_tolerates_near_zero_yields(tmp_path):
    idx = pd.bdate_range("2012-01-01", periods=3)
    pd.Series([0.0, 0.01, 0.0], index=idx, name="close").to_csv(
        tmp_path / "IRX.csv", index_label="date"
    )
    rf = load_risk_free(cache_dir=tmp_path)  # must not raise (unlike clean_prices)
    assert (rf >= 0).all()


def test_clean_prices_rejects_nonpositive():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"X": [100.0, -1.0, 101.0]}, index=idx)
    with pytest.raises(ValueError, match="Non-positive"):
        clean_prices(prices)


def test_clean_prices_fills_short_gaps_and_drops_incomplete_head():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "X": [np.nan, np.nan, 100.0, np.nan, 102.0, 103.0],
            "Y": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        },
        index=idx,
    )
    cleaned = clean_prices(prices, max_ffill=2)
    # Head rows where X has no history yet are dropped; the interior gap is filled.
    assert cleaned.index[0] == idx[2]
    assert cleaned.loc[idx[3], "X"] == pytest.approx(100.0)


def test_clean_prices_warns_on_extreme_jump():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"X": [100.0, 100.0, 250.0]}, index=idx)
    with pytest.warns(UserWarning, match="Single-day moves"):
        clean_prices(prices)


def test_to_returns_conventions():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=idx)
    simple = to_returns(prices, kind="simple")
    log = to_returns(prices, kind="log")
    assert simple["X"].tolist() == pytest.approx([0.10, -0.10])
    assert log["X"].tolist() == pytest.approx([np.log(1.10), np.log(0.90)])
    assert len(simple) == 2  # first observation consumed by differencing
    with pytest.raises(ValueError, match="kind"):
        to_returns(prices, kind="weekly")
