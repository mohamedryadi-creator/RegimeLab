"""Regime detection: heuristic filters and statistical models.

All regime detectors implement the :class:`regimelab.regimes.base.RegimeModel`
interface so that strategies and experiments can treat them interchangeably.
"""

from collections.abc import Mapping

from regimelab.regimes.base import RegimeModel
from regimelab.regimes.heuristic import TrendRegime, VolatilityThresholdRegime

__all__ = [
    "RegimeModel",
    "TrendRegime",
    "VolatilityThresholdRegime",
    "regime_model_from_spec",
]


def regime_model_from_spec(spec: Mapping) -> RegimeModel:
    """Build a regime model from a config mapping with a ``type`` key.

    Remaining keys are passed to the model constructor; ``name`` and
    ``exposure`` keys (consumed by experiment runners) are ignored here.
    The HMM import is lazy so configs without it never need the models extra.
    """
    kwargs = dict(spec)
    kind = kwargs.pop("type", None)
    kwargs.pop("name", None)
    kwargs.pop("exposure", None)
    kwargs.pop("probability_scaled", None)
    kwargs.pop("bands", None)
    if kind == "volatility":
        return VolatilityThresholdRegime(**kwargs)
    if kind == "trend":
        return TrendRegime(**kwargs)
    if kind == "hmm":
        from regimelab.regimes.statistical import GaussianHMMRegime

        return GaussianHMMRegime(**kwargs)
    raise ValueError(f"Unknown regime model type: {kind!r}.")
