"""GaussianHMMRegime: recovery, causality, determinism, state identification."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn", reason="requires the models extra")

from regimelab.regimes.statistical import GaussianHMMRegime  # noqa: E402
from regimelab.validation.walkforward import (  # noqa: E402
    walk_forward_predict,
    walk_forward_splits,
)


def simulate_two_state(n=2000, seed=7, p_stay=0.98):
    """Persistent 2-state process: calm N(5e-4, 0.005), turbulent N(-5e-4, 0.02)."""
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < p_stay else 1 - states[t - 1]
    mu = np.where(states == 0, 5e-4, -5e-4)
    sigma = np.where(states == 0, 0.005, 0.02)
    log_returns = rng.normal(mu, sigma)
    idx = pd.bdate_range("2015-01-01", periods=n)
    prices = pd.DataFrame({"X": 100 * np.exp(np.cumsum(log_returns))}, index=idx)
    return prices, pd.Series(states, index=idx, dtype=float)


@pytest.fixture(scope="module")
def fitted():
    prices, true_states = simulate_two_state()
    model = GaussianHMMRegime(n_regimes=2, n_restarts=3, max_iter=100, seed=0)
    model.fit(prices.iloc[:1000])
    return model, prices, true_states


def test_recovers_simulated_regimes(fitted):
    model, prices, true_states = fitted
    labels = model.predict_regimes(prices)
    # Skip the first date (no return) and allow filtering lag at switches.
    accuracy = (labels.iloc[1:] == true_states.iloc[1:]).mean()
    assert accuracy > 0.85


def test_states_ordered_by_variance(fitted):
    model, _, _ = fitted
    assert model.vars_[0] < model.vars_[1]
    # States must be clearly separated and in the right ballpark of the
    # simulated 0.005/0.02 vols; exact recovery is not expected from EM on a
    # finite sample (boundary observations bias the calm state upward).
    assert model.vars_[1] / model.vars_[0] > 3
    assert np.sqrt(model.vars_[0]) == pytest.approx(0.005, rel=0.6)
    assert np.sqrt(model.vars_[1]) == pytest.approx(0.02, rel=0.6)


def test_predictions_are_causal(fitted):
    """Filtered labels must not change when future data is removed."""
    model, prices, _ = fitted
    full = model.predict_regimes(prices)
    for cut in (300, 900, 1500):
        truncated = model.predict_regimes(prices.iloc[:cut])
        pd.testing.assert_series_equal(truncated, full.iloc[:cut])


def test_filtered_probabilities_are_distributions(fitted):
    model, prices, _ = fitted
    probs = model.filtered_probabilities(prices)
    assert probs.iloc[0].isna().all()  # warm-up: first price has no return
    valid = probs.dropna()
    assert np.allclose(valid.sum(axis=1), 1.0)
    assert (valid.to_numpy() >= 0).all()


def test_fit_is_deterministic_given_seed():
    prices, _ = simulate_two_state(n=800)
    kwargs = dict(n_regimes=2, n_restarts=2, max_iter=50, seed=123)
    a = GaussianHMMRegime(**kwargs).fit(prices)
    b = GaussianHMMRegime(**kwargs).fit(prices)
    np.testing.assert_allclose(a.transmat_, b.transmat_)
    pd.testing.assert_series_equal(a.predict_regimes(prices), b.predict_regimes(prices))


def test_regimes_are_persistent(fitted):
    """Expected durations far above 1 period: regimes, not relabeled noise."""
    model, _, _ = fitted
    durations = model.expected_durations()
    assert (durations > 5).all()


def test_requires_fit_and_validates_args():
    prices, _ = simulate_two_state(n=300)
    with pytest.raises(RuntimeError, match="fit"):
        GaussianHMMRegime().predict_regimes(prices)
    with pytest.raises(ValueError, match="n_regimes"):
        GaussianHMMRegime(n_regimes=1)
    with pytest.raises(ValueError, match="too short"):
        GaussianHMMRegime(n_regimes=2).fit(prices.iloc[:50])


def test_walk_forward_integration():
    prices, _ = simulate_two_state(n=900)
    splits = walk_forward_splits(prices.index, train_size=400, test_size=250)
    model = GaussianHMMRegime(n_regimes=2, n_restarts=2, max_iter=50, seed=0)
    labels = walk_forward_predict(model, prices, splits)
    pd.testing.assert_index_equal(labels.index, prices.index[400:])
    assert labels.notna().all()  # test dates always have trailing history
    assert set(labels.unique()) <= {0.0, 1.0}
