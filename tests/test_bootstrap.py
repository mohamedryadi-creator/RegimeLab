"""Block-bootstrap inference tests."""

import numpy as np
import pandas as pd
import pytest

from regimelab.backtest.bootstrap import (
    _block_indices,
    sharpe_confidence_interval,
    sharpe_difference,
)


def make_series(n=1000, mean=0.0004, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_block_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = _block_indices(100, 21, rng)
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < 100
    # Within a block, indices are consecutive (mod n).
    assert (np.diff(idx[:21]) % 100 == 1).all()


def test_ci_brackets_point_estimate():
    r = make_series()
    res = sharpe_confidence_interval(r, n_boot=500, seed=1)
    assert res.ci_low <= res.sharpe <= res.ci_high
    assert res.ci_low < res.ci_high


def test_deterministic_given_seed():
    r = make_series()
    a = sharpe_confidence_interval(r, n_boot=200, seed=42)
    b = sharpe_confidence_interval(r, n_boot=200, seed=42)
    assert a == b


def test_identical_series_diff_is_exactly_zero():
    r = make_series()
    res = sharpe_difference(r, r, n_boot=200)
    assert res.diff == 0.0
    assert res.ci_low == res.ci_high == 0.0
    assert res.prob_nonpositive == 1.0  # every draw equals zero


def test_clearly_better_series_has_small_prob_nonpositive():
    base = make_series(seed=3)
    better = base + 0.0008  # same risk, materially higher mean
    res = sharpe_difference(better, base, n_boot=500, seed=0)
    assert res.diff > 0
    assert res.prob_nonpositive < 0.05
    assert res.ci_low > 0


def test_paired_requires_identical_index():
    a = make_series(n=500)
    b = make_series(n=400)
    with pytest.raises(ValueError, match="identical index"):
        sharpe_difference(a, b)


def test_validation():
    r = make_series(n=30)
    with pytest.raises(ValueError, match="observations"):
        sharpe_confidence_interval(r, block_size=21)
    with pytest.raises(ValueError, match="block_size"):
        sharpe_confidence_interval(make_series(), block_size=0)
