"""The backtesting engine.

Design (to be implemented in Phase 2)
-------------------------------------
A *vectorized* backtester over target positions is sufficient for this project's
scope (daily/weekly frequency, no intraday execution modeling) and is much easier
to test than an event-driven engine.

Core loop, conceptually:
    positions_held(t) = target_positions(t - 1)          # explicit execution lag
    gross_return(t)   = positions_held(t) · asset_returns(t)
    turnover(t)       = |positions_held(t) - positions_held(t - 1)|
    net_return(t)     = gross_return(t) - cost(turnover(t))

Invariants to test:
- zero positions => zero returns regardless of asset moves;
- buy-and-hold with zero costs reproduces the asset's compounded return;
- shifting all signals by +1 day must never *improve* results on average
  (look-ahead canary);
- doubling costs never increases net performance.

The engine returns a result object bundling net/gross return series, positions,
turnover, and cost drag, so metrics and reports always come from one place.

TODO (Phase 2): implement ``run_backtest(positions, returns, cost_model)``.
"""
