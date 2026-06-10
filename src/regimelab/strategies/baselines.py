"""Baseline strategies.

Baselines come first (Phase 3): they define the reference results that any
regime-aware strategy must beat out-of-sample and after costs.

Planned baselines:
- BuyAndHold: constant full investment; zero turnover by construction.
- TrendFollowing: long when price is above a long moving average, otherwise in
  cash (window is a parameter; sensitivity to it is part of the analysis).
- (later) regime-aware variants that scale or gate these baselines using a
  RegimeModel's output.

TODO (Phase 3): implement both as Strategy subclasses with tests on toy series
where the correct positions are known by hand.
"""
