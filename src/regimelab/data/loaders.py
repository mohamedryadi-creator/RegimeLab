"""Data loaders.

The core library is source-agnostic: ``load_prices`` returns adjusted close
prices as a DataFrame (DatetimeIndex, one column per ticker) regardless of
origin. Raw data is cached as one CSV per ticker under ``data/raw/`` with a JSON
sidecar recording the source and download date, so every experiment can state
exactly which data snapshot it used.

If a cached CSV exists it is used as-is (no network). Downloading requires the
``data`` extra (yfinance); alternatively, place a CSV at
``data/raw/<TICKER>.csv`` with columns ``date,close`` and the loader will use it.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers``, from cache or by downloading.

    Returns a DataFrame indexed by date with one column per ticker, restricted
    to ``[start, end]`` if given.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    series = {}
    for ticker in tickers:
        path = cache / f"{ticker}.csv"
        if refresh or not path.exists():
            _download(ticker, path)
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if "close" not in frame.columns:
            raise ValueError(f"{path} must have a 'close' column, found {list(frame.columns)}.")
        series[ticker] = frame["close"].rename(ticker)

    prices = pd.DataFrame(series).sort_index()
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end)]
    if prices.empty:
        raise ValueError(f"No price data for {tickers} in [{start}, {end}].")
    return prices


def load_risk_free(
    cache_dir: str | Path = "data/raw",
    ticker: str = "^IRX",
    periods_per_year: int = 252,
    refresh: bool = False,
) -> pd.Series:
    """Daily cash (risk-free) return series from an annualized T-bill yield index.

    ``^IRX`` quotes the 13-week T-bill discount yield in percent (e.g. ``4.85``).
    Conversion to a per-period simple return is ``y / 100 / periods_per_year`` —
    a documented simplification (ignores discount vs. bond-equivalent compounding;
    the error is < 0.1% annualized at historical yield levels).

    Point-in-time convention: the series is forward-filled and **lagged one
    day** — the rate known at the close of ``t`` accrues over ``t+1``, matching
    the engine's execution lag. Deliberately *not* routed through
    ``clean_prices``: near-zero yields (2009-2015, 2020-21) are real data.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{ticker.replace('^', '')}.csv"
    if refresh or not path.exists():
        _download(ticker, path)
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "close" not in frame.columns:
        raise ValueError(f"{path} must have a 'close' column, found {list(frame.columns)}.")
    rate = frame["close"]
    # Bond-market holidays (e.g. Columbus Day) have no T-bill quote while
    # equities trade: reindex to a continuous business-day calendar so the
    # prior day's rate carries over before lagging.
    calendar = pd.bdate_range(rate.index.min(), rate.index.max()).union(rate.index)
    rate = rate.reindex(calendar).ffill()
    rf = (rate / 100.0 / periods_per_year).shift(1).rename("rf")
    return rf.dropna()


def _download(ticker: str, path: Path) -> None:
    """Download the full adjusted-close history for ``ticker`` to ``path``."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            f"No cached data at {path} and yfinance is not installed. "
            f'Install the data extra (pip install -e ".[data]") or place a CSV '
            f"with columns date,close at {path}."
        ) from exc

    history = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if history.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}.")
    close = history["Close"]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.rename("close").to_csv(path, index_label="date")

    sidecar = {
        "ticker": ticker,
        "source": "yfinance (auto_adjust=True)",
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": int(close.shape[0]),
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
