"""Heuristic regime models: correctness, causality, and walk-forward discipline."""

import numpy as np
import pandas as pd
import pytest

from regimelab.regimes import (
    RegimeModel,
    TrendRegime,
    VolatilityThresholdRegime,
    regime_model_from_spec,
)
from regimelab.validation.walkforward import walk_forward_predict, walk_forward_splits


def make_prices(n=600, seed=0, vol_break=None):
    """Geometric random walk; optionally triple the volatility after ``vol_break``."""
    rng = np.random.default_rng(seed)
    sigma = np.full(n, 0.005)
    if vol_break is not None:
        sigma[vol_break:] = 0.015
    returns = rng.normal(0, 1, n) * sigma
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"X": 100 * np.exp(np.cumsum(returns))}, index=idx)


def test_vol_regime_detects_volatility_break():
    prices = make_prices(n=600, vol_break=300)
    model = VolatilityThresholdRegime(vol_window=20, quantile=0.8)
    model.fit(prices.iloc[:300])
    labels = model.predict_regimes(prices)
    # Comfortably after the break, trailing vol must exceed the calm-sample threshold.
    assert (labels.iloc[340:] == 1.0).all()
    # The calm period is mostly labeled calm (the 0.8 quantile flags ~20%).
    assert labels.iloc[20:300].mean() < 0.3


def test_vol_regime_warmup_is_nan():
    prices = make_prices(n=100)
    model = VolatilityThresholdRegime(vol_window=20).fit(prices)
    labels = model.predict_regimes(prices)
    # First return is undefined and the vol window needs 20 returns.
    assert labels.iloc[:20].isna().all()
    assert labels.iloc[20:].notna().all()


def test_vol_regime_requires_fit():
    prices = make_prices(n=100)
    with pytest.raises(RuntimeError, match="fit"):
        VolatilityThresholdRegime().predict_regimes(prices)


def test_vol_regime_rejects_short_training_sample():
    prices = make_prices(n=10)
    with pytest.raises(ValueError, match="too short"):
        VolatilityThresholdRegime(vol_window=20).fit(prices)


def test_trend_regime_up_and_down():
    idx = pd.bdate_range("2020-01-01", periods=40)
    up = pd.DataFrame({"X": np.linspace(100, 140, 40)}, index=idx)
    model = TrendRegime(window=10).fit(up)
    labels_up = model.predict_regimes(up)
    assert labels_up.iloc[:9].isna().all()
    assert (labels_up.iloc[9:] == 1.0).all()

    down = pd.DataFrame({"X": np.linspace(140, 100, 40)}, index=idx)
    labels_down = model.predict_regimes(down)
    assert (labels_down.iloc[9:] == 0.0).all()


@pytest.mark.parametrize(
    "model",
    [VolatilityThresholdRegime(vol_window=20, quantile=0.8), TrendRegime(window=20)],
    ids=["volatility", "trend"],
)
def test_regime_predictions_are_causal(model):
    """predict(x[:t]) == predict(x)[:t]: labels never depend on future data."""
    prices = make_prices(n=300)
    model.fit(prices.iloc[:150])
    full = model.predict_regimes(prices)
    for cut in (100, 200, 300):
        truncated = model.predict_regimes(prices.iloc[:cut])
        pd.testing.assert_series_equal(truncated, full.iloc[:cut])


def test_regime_models_reject_multiasset_input():
    idx = pd.bdate_range("2020-01-01", periods=50)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    with pytest.raises(ValueError, match="per-asset"):
        TrendRegime(window=10).predict_regimes(prices)


def test_n_regimes():
    assert VolatilityThresholdRegime().n_regimes == 2
    assert TrendRegime().n_regimes == 2


def test_regime_model_from_spec():
    model = regime_model_from_spec(
        {"name": "v", "type": "volatility", "vol_window": 10, "quantile": 0.9,
         "exposure": {0: 1.0, 1: 0.0}}
    )
    assert isinstance(model, VolatilityThresholdRegime)
    assert model.vol_window == 10 and model.quantile == 0.9

    assert isinstance(regime_model_from_spec({"type": "trend", "window": 50}), TrendRegime)

    with pytest.raises(ValueError, match="Unknown regime model type"):
        regime_model_from_spec({"type": "wavelet"})


class RecordingModel(RegimeModel):
    """Mock that records what data fit() saw; predicts a constant regime."""

    def __init__(self):
        self.fit_ends = []

    def fit(self, X):
        self.fit_ends.append(X.index.max())
        return self

    def predict_regimes(self, X):
        return pd.Series(0.0, index=X.index, name="regime")


def test_walk_forward_predict_fits_only_on_past():
    prices = make_prices(n=500)
    splits = walk_forward_splits(prices.index, train_size=200, test_size=100)
    model = RecordingModel()
    labels = walk_forward_predict(model, prices, splits)

    assert len(model.fit_ends) == len(splits)
    for fit_end, (_, test) in zip(model.fit_ends, splits, strict=True):
        assert fit_end < test.min()
    # Predictions cover exactly the out-of-sample period, in order.
    pd.testing.assert_index_equal(labels.index, prices.index[200:])
