"""Walk-forward validation splits.

The split generator is model-agnostic: it yields ``(train_index, test_index)``
pairs of dates, and the experiment runner fits regime models only on the train
slice. This is the structural defense against look-ahead in model fitting.

Schemes:
- ``"expanding"``: train on everything before the test block (more data per fit,
  mixes eras);
- ``"rolling"``: train on the most recent ``train_size`` observations (adapts
  faster, less data per fit).

``embargo`` drops the last ``embargo`` observations before each test block from
the training set, so rolling features computed near the boundary cannot leak
test-period information into training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

    from regimelab.regimes.base import RegimeModel


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    scheme: str = "expanding",
    embargo: int = 0,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate walk-forward ``(train, test)`` index pairs over ``index``.

    The first test block starts at position ``train_size``; subsequent blocks
    advance by ``test_size``. The final block may be shorter than ``test_size``.
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}.")
    if train_size <= 0 or test_size <= 0 or embargo < 0:
        raise ValueError("train_size and test_size must be > 0, embargo >= 0.")
    if len(index) <= train_size:
        raise ValueError(
            f"Need more than train_size={train_size} observations, got {len(index)}."
        )
    if embargo >= train_size:
        raise ValueError("embargo must be smaller than train_size.")

    splits = []
    test_start = train_size
    while test_start < len(index):
        test = index[test_start : test_start + test_size]
        train_end = test_start - embargo
        if scheme == "expanding":
            train = index[:train_end]
        else:
            train = index[max(0, train_end - train_size) : train_end]
        splits.append((train, test))
        test_start += test_size
    return splits


def walk_forward_predict_proba(
    model: "RegimeModel",
    X: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.DataFrame:
    """Stitched out-of-sample *filtered* state probabilities.

    Same refit-on-train discipline as :func:`walk_forward_predict`, but returns
    the per-state probability DataFrame for models that expose
    ``filtered_probabilities`` (e.g. the Gaussian HMM). Used for continuous
    position scaling, where exposure is a probability-weighted average rather
    than a hard regime gate.
    """
    if not hasattr(model, "filtered_probabilities"):
        raise TypeError(
            f"{type(model).__name__} does not expose filtered_probabilities(); "
            "probability scaling requires a probabilistic regime model."
        )
    predictions = []
    for train, test in splits:
        model.fit(X.loc[train])
        probs = model.filtered_probabilities(X.loc[X.index <= test.max()])
        predictions.append(probs.loc[test])
    return pd.concat(predictions)


def walk_forward_predict_per_asset(
    model_factory: "Callable[[], RegimeModel]",
    prices: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.DataFrame:
    """Per-asset out-of-sample regime labels (one fresh model per asset).

    Regime models are per-asset by design (see regimes.base); for a multi-asset
    universe each column gets its own independently fitted model. Returns a
    DataFrame of labels with one column per asset.
    """
    return pd.DataFrame(
        {
            col: walk_forward_predict(model_factory(), prices[[col]], splits)
            for col in prices.columns
        }
    )


def walk_forward_proba_per_asset(
    model_factory: "Callable[[], RegimeModel]",
    prices: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.DataFrame:
    """Per-asset out-of-sample filtered probabilities.

    Returns a DataFrame with MultiIndex columns ``(asset, state)``.
    """
    frames = {
        col: walk_forward_predict_proba(model_factory(), prices[[col]], splits)
        for col in prices.columns
    }
    return pd.concat(frames, axis=1)


def walk_forward_predict(
    model: "RegimeModel",
    X: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
) -> pd.Series:
    """Stitch together out-of-sample regime predictions across walk-forward splits.

    For each split the model is refitted on the train slice only, then asked to
    label the test dates. Prediction uses data up to the end of the test block
    (trailing features need history that may start before the block), which is
    safe because ``predict_regimes`` is causal — labels at test dates cannot
    depend on later data.

    Returns one label series covering all test blocks: the model's genuinely
    out-of-sample regime history.
    """
    predictions = []
    for train, test in splits:
        model.fit(X.loc[train])
        labels = model.predict_regimes(X.loc[X.index <= test.max()])
        predictions.append(labels.loc[test])
    return pd.concat(predictions)
