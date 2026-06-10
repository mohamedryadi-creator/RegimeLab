"""Regime detection: heuristic filters and statistical models.

All regime detectors implement the :class:`regimelab.regimes.base.RegimeModel`
interface so that strategies and experiments can treat them interchangeably.
"""

from regimelab.regimes.base import RegimeModel
from regimelab.regimes.heuristic import TrendRegime, VolatilityThresholdRegime

__all__ = ["RegimeModel", "TrendRegime", "VolatilityThresholdRegime"]
