"""Feature engineering.

Point-in-time discipline
------------------------
Every feature at date ``t`` must be computable from information available at the
close of date ``t`` (or earlier). Rolling windows must not be centered; any
normalization (z-scores, ranks) must use rolling/expanding statistics, never
full-sample statistics. Tests in ``tests/test_lookahead.py`` enforce this contract.

Candidate features (to be selected in Phases 2–4):
- rolling realized volatility at several horizons;
- rolling returns / momentum (e.g. 1, 3, 6, 12 months);
- moving-average spreads (price vs. long MA) for trend;
- drawdown from rolling peak;
- (later) downside volatility, autocorrelation, cross-asset signals.

TODO (Phase 2+): implement features as small pure functions, one per feature,
each returning a Series aligned on the input index.
"""

from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility of ``returns`` over ``window`` observations.

    TODO: implement in Phase 2.
    """
    raise NotImplementedError("Features are implemented from Phase 2 onward.")
