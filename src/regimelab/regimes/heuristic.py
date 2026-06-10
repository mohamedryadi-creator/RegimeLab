"""Heuristic regime filters.

These are deliberately simple, transparent rules (e.g. "high-vol regime when
realized volatility exceeds its rolling median", "bear regime when price is below
its 200-day moving average"). They serve as cheap benchmarks: a statistical model
that cannot beat a heuristic filter out-of-sample is not earning its complexity.

TODO (Phase 4):
- VolatilityThresholdRegime (rolling vol vs. rolling quantile threshold);
- TrendRegime (price vs. long moving average);
- both implementing :class:`regimelab.regimes.base.RegimeModel`.
"""
