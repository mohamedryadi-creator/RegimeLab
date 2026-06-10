"""Look-ahead bias tests.

These are the project's core methodological guarantees:
- features are causal (no future information leaks into past values);
- the engine lags positions (a same-day signal cannot earn same-day returns);
- walk-forward training data strictly precedes test data.
"""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.engine import run_backtest
from regimelab.features.engineering import CAUSAL_FEATURES
from regimelab.validation.walkforward import walk_forward_splits


def _toy_series(n=120, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
    prices = 100 * (1 + returns).cumprod()
    return prices, returns


@pytest.mark.parametrize("feature_fn,input_kind", CAUSAL_FEATURES)
def test_features_are_causal(feature_fn, input_kind):
    """feature(x[:t]) == feature(x)[:t] for every feature in the registry."""
    prices, returns = _toy_series()
    x = prices if input_kind == "prices" else returns
    full = feature_fn(x)
    for cut in (40, 80, len(x)):
        truncated = feature_fn(x.iloc[:cut])
        pd.testing.assert_series_equal(truncated, full.iloc[:cut])


def test_engine_lags_positions():
    """A signal equal to the *same day's* return sign would be perfect with no
    lag; the engine's execution lag must neutralize it."""
    _, returns = _toy_series(n=500)
    clairvoyant = np.sign(returns)
    result = run_backtest(clairvoyant, returns)
    perfect = returns.abs().sum()
    assert result.gross_returns.sum() < 0.5 * perfect
    # held(t) must equal signal(t-1), exactly.
    expected_held = clairvoyant.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        result.positions.iloc[:, 0], expected_held, check_names=False
    )


def test_engine_rejects_zero_lag():
    _, returns = _toy_series(n=50)
    pos = pd.Series(1.0, index=returns.index)
    with pytest.raises(ValueError, match="look-ahead"):
        run_backtest(pos, returns, lag=0)


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walkforward_train_precedes_test(scheme):
    idx = pd.bdate_range("2015-01-01", periods=1000)
    splits = walk_forward_splits(idx, train_size=250, test_size=60, scheme=scheme)
    assert len(splits) > 0
    for train, test in splits:
        assert len(train) > 0 and len(test) > 0
        assert train.max() < test.min()
    # Test blocks tile the out-of-sample period without overlap or gaps.
    covered = splits[0][1]
    for _, test in splits[1:]:
        covered = covered.append(test)
    pd.testing.assert_index_equal(covered, idx[250:])


def test_walkforward_embargo_creates_gap():
    idx = pd.bdate_range("2015-01-01", periods=600)
    splits = walk_forward_splits(idx, train_size=250, test_size=60, embargo=10)
    for train, test in splits:
        gap = idx.get_loc(test.min()) - idx.get_loc(train.max())
        assert gap > 10


def test_walkforward_rolling_window_size():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    splits = walk_forward_splits(idx, train_size=250, test_size=60, scheme="rolling")
    assert all(len(train) == 250 for train, _ in splits)
