"""No-trade band (hysteresis) around a base strategy's target positions.

The study's mechanistic diagnosis of HMM cost-fragility is filtered-state
flicker: the filtered probability moves a little almost every day, and a
probability-scaled strategy rebalances on each move. A no-trade band is the
transparent remedy: hold the current weight while the target stays within
``band`` of it, and trade *to the target* only when it drifts further.

Point-in-time safety: the held weight at ``t`` is a forward recursion over
targets up to ``t`` only (verified by truncation tests). The band is a research
parameter — experiments must report a sweep, never an in-sample "best band".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regimelab.strategies.base import Strategy


@dataclass(frozen=True, eq=False)
class BandedStrategy(Strategy):
    """Follow ``base`` targets, but only trade when they move more than ``band``.

    With ``band=0`` this reproduces ``base`` exactly. Bands are applied per
    asset; the recursion starts flat (weight 0 before the first breach).
    """

    base: Strategy
    band: float

    def __post_init__(self) -> None:
        if self.band < 0:
            raise ValueError("band must be >= 0.")

    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        target = self.base.target_positions(data)
        t = target.to_numpy(dtype=float)
        out = np.empty_like(t)
        prev = np.zeros(t.shape[1])
        for i in range(t.shape[0]):
            breach = np.abs(t[i] - prev) > self.band
            prev = np.where(breach, t[i], prev)
            out[i] = prev
        return pd.DataFrame(out, index=target.index, columns=target.columns)
