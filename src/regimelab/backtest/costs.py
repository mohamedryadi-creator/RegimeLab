"""Transaction cost models.

Starting point (Phase 2): proportional costs, ``cost(t) = c * turnover(t)`` with
``c`` in basis points per unit of turnover. This is simple, transparent, and
sufficient for liquid index products at daily frequency.

Possible later refinement: separate spread and impact components. Decide only if
results turn out to be sensitive to the cost model — cost *sensitivity analysis*
(sweeping c) is part of every experiment regardless.

TODO (Phase 2): implement ProportionalCost; keep a small CostModel protocol so
the engine does not hard-code the functional form.
"""
