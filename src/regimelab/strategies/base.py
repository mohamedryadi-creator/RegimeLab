"""The common interface for strategies.

Contract
--------
``target_positions`` returns the *desired* position for each date, computed from
information available at that date's close. The backtest engine — not the
strategy — is responsible for applying the execution lag (positions decided at
the close of ``t`` earn the return of ``t+1``). Keeping the lag in one place
makes look-ahead bias testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading/allocation strategies."""

    @abstractmethod
    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return target positions (assets in columns) for each date in ``data``.

        ``data`` contains whatever the strategy needs (prices, features, regime
        labels), already restricted to point-in-time information by the caller.
        """
