"""Data loading and cleaning tests (no network: cache-based)."""

import numpy as np
import pandas as pd
import pytest

from regimelab.data.cleaning import clean_prices, to_returns
from regimelab.data.loaders import load_prices


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
