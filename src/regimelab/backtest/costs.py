"""Transaction cost models.

A cost model is any callable mapping a turnover series (sum of absolute weight
changes per period) to a cost series in return units. Proportional costs are the
working assumption for liquid index products at daily frequency; every experiment
should sweep the cost level rather than trust a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class CostModel(Protocol):
    def __call__(self, turnover: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class ProportionalCost:
    """Cost = ``bps`` basis points per unit of turnover (one-way)."""

    bps: float

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("Transaction costs cannot be negative.")

    def __call__(self, turnover: pd.Series) -> pd.Series:
        return turnover * (self.bps * 1e-4)
