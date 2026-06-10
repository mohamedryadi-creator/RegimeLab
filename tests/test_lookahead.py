"""Look-ahead bias tests.

This file establishes the testing pattern for point-in-time correctness. The
tests are written first as specifications and skipped until the corresponding
phase implements the functionality — they must be un-skipped, not deleted.

Planned checks:
- features: truncating the input series must not change feature values on the
  remaining dates (no future information leaks into the past);
- backtest engine: positions are lagged — a signal equal to *tomorrow's* return
  sign must not produce a perfect backtest;
- walk-forward splits: every train date strictly precedes every test date.
"""

import pytest


@pytest.mark.skip(reason="Phase 2: implemented together with the feature functions.")
def test_features_are_causal():
    """feature(x[:t]) == feature(x)[:t] for all t."""


@pytest.mark.skip(reason="Phase 2: implemented together with the backtest engine.")
def test_engine_lags_positions():
    """A clairvoyant signal must be neutralized by the engine's execution lag."""


@pytest.mark.skip(reason="Phase 2: implemented together with the split generator.")
def test_walkforward_train_precedes_test():
    """max(train_dates) < min(test_dates) for every split."""
