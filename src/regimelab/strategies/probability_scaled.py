"""Continuous position scaling by filtered regime probabilities.

The continuous counterpart of :class:`RegimeGatedStrategy`: instead of a hard
on/off gate on the most likely regime, positions are scaled by a point-in-time
exposure weight, typically

    weight_t = sum_k exposure(k) * P(state_t = k | r_1..t)

computed from *filtered* probabilities (smoothed probabilities condition on
future data and must never reach this class). The weights are precomputed by
the experiment runner — e.g. via ``walk_forward_proba_per_asset`` — so the
strategy stays agnostic to where they came from, exactly like the gated
variant.

Hypothesis this enables testing: soft scaling should trade less than hard
gating (no flip on marginal probability moves around 0.5) while keeping most
of the risk reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regimelab.strategies.base import Strategy


@dataclass(frozen=True, eq=False)
class ProbabilityScaledStrategy(Strategy):
    """Scale ``base`` positions by a precomputed exposure weight in [0, 1].

    ``exposure_weights`` is a Series (one weight for the whole portfolio) or a
    DataFrame with one column per asset. Weights are clipped to [0, 1]; dates
    with no weight (warm-up, missing) get ``default_exposure``.
    """

    base: Strategy
    exposure_weights: pd.Series | pd.DataFrame
    default_exposure: float = 0.0

    def target_positions(self, data: pd.DataFrame) -> pd.DataFrame:
        positions = self.base.target_positions(data)
        weights = self.exposure_weights
        if isinstance(weights, pd.DataFrame):
            if set(weights.columns) != set(positions.columns):
                raise ValueError(
                    f"Per-asset weight columns {list(weights.columns)} must "
                    f"match position columns {list(positions.columns)}."
                )
            scale = (
                weights[positions.columns]
                .reindex(positions.index)
                .clip(0.0, 1.0)
                .fillna(self.default_exposure)
            )
            return positions * scale
        scale = (
            weights.reindex(positions.index).clip(0.0, 1.0).fillna(self.default_exposure)
        )
        return positions.mul(scale, axis=0)
