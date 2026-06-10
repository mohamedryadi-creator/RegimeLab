"""Regime-gated strategy wrapper.

Composes any base strategy with a *precomputed* point-in-time regime label
series: the base strategy's target positions are scaled by the exposure mapped
to that date's regime. Passing labels (rather than a model) keeps the wrapper
agnostic to how they were produced — in experiments they come from
:func:`regimelab.validation.walkforward.walk_forward_predict`, so models are
only ever fitted on past data.

Dates whose regime is NaN (warm-up) or missing from the label series get
``default_exposure`` — conservative (flat) by default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from regimelab.strategies.base import Strategy


@dataclass(frozen=True, eq=False)
class RegimeGatedStrategy(Strategy):
    """Scale ``base`` positions by the exposure assigned to each date's regime.

    ``exposure`` maps regime labels to position scales, e.g. ``{0: 1.0, 1: 0.0}``
    for "fully invested in calm, flat in turbulence". Binary gating is the
    Phase 4 choice; continuous scaling (e.g. by regime probability) is a
    Phase 5+ variant and fits the same interface.
    """

    base: Strategy
    regimes: pd.Series
    exposure: Mapping[float, float] = field(default_factory=lambda: {0.0: 1.0, 1.0: 0.0})
    default_exposure: float = 0.0

    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        positions = self.base.target_positions(data)
        scale = (
            self.regimes.map(dict(self.exposure))
            .reindex(positions.index)
            .fillna(self.default_exposure)
        )
        return positions.mul(scale, axis=0)
