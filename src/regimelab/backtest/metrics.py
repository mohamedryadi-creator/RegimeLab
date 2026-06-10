"""Performance and risk metrics.

Planned metrics (Phase 2), each a small pure function of a net return series
(plus positions/turnover where relevant), tested against hand-computed examples:

- annualized return and volatility (annualization factor explicit, not implicit);
- Sharpe ratio (decide and document: rf = 0 vs. actual short rate);
- Sortino ratio;
- maximum drawdown, drawdown durations, time under water;
- hit ratio;
- average turnover and total cost drag;
- (for comparisons) simple block-bootstrap confidence intervals on Sharpe —
  a point estimate difference between two strategies is not a finding.

TODO (Phase 2): implement; ``summary(returns, positions)`` returns a tidy
one-row DataFrame so experiment outputs are uniform.
"""
