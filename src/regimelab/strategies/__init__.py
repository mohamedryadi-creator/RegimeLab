"""Trading and allocation strategies.

All strategies implement :class:`regimelab.strategies.base.Strategy`: they map
available information to target positions. Position sizing conventions (e.g.
weights in [0, 1] for long-only, [-1, 1] if shorting is allowed) are fixed in
Phase 2 together with the backtest engine.
"""

from regimelab.strategies.base import Strategy

__all__ = ["Strategy"]
