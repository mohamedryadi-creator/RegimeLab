"""Data cleaning and quality checks.

Cleaning must be conservative and fully logged: every dropped or filled observation
is a research decision. Quality checks should fail loudly rather than silently
producing a plausible-looking series.

TODO (Phase 1):
- handle missing values (decide: forward-fill limits vs. dropping dates);
- detect obvious data errors (zero/negative prices, extreme single-day jumps);
- align calendars across assets for the multi-asset case;
- compute simple and log returns with explicit conventions.
"""

from __future__ import annotations

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw price DataFrame.

    TODO: implement in Phase 1.
    """
    raise NotImplementedError("Cleaning is implemented in Phase 1.")


def to_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Convert prices to returns (``kind`` in {"log", "simple"}).

    TODO: implement in Phase 1 (trivial, but conventions must be fixed and tested).
    """
    raise NotImplementedError("Return computation is implemented in Phase 1.")
