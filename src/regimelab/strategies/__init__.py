"""Trading and allocation strategies.

All strategies implement :class:`regimelab.strategies.base.Strategy`: they map
available information to target positions (long-only weights in [0, 1] summing
to at most 1 across assets). The backtest engine applies the execution lag.
"""

from regimelab.strategies.base import Strategy
from regimelab.strategies.baselines import BuyAndHold, TrendFollowing
from regimelab.strategies.probability_scaled import ProbabilityScaledStrategy
from regimelab.strategies.regime_gated import RegimeGatedStrategy

__all__ = [
    "BuyAndHold",
    "ProbabilityScaledStrategy",
    "RegimeGatedStrategy",
    "Strategy",
    "TrendFollowing",
]
