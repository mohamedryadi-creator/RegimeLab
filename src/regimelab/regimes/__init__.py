"""Regime detection: heuristic filters and statistical models.

All regime detectors implement the :class:`regimelab.regimes.base.RegimeModel`
interface so that strategies and experiments can treat them interchangeably.
"""

from regimelab.regimes.base import RegimeModel

__all__ = ["RegimeModel"]
