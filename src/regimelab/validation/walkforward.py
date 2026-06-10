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
