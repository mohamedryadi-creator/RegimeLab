"""Smoke tests: the package installs, imports, and exposes its interfaces."""

import pytest

import regimelab
from regimelab.regimes import RegimeModel
from regimelab.strategies import Strategy


def test_version():
    assert regimelab.__version__


def test_submodules_import():
    """Every subpackage must stay importable as the skeleton fills in."""
    import regimelab.backtest.costs  # noqa: F401
    import regimelab.backtest.engine  # noqa: F401
    import regimelab.backtest.metrics  # noqa: F401
    import regimelab.data.cleaning  # noqa: F401
    import regimelab.data.loaders  # noqa: F401
    import regimelab.features.engineering  # noqa: F401
    import regimelab.regimes.heuristic  # noqa: F401
    import regimelab.regimes.statistical  # noqa: F401
    import regimelab.strategies.baselines  # noqa: F401
    import regimelab.validation.walkforward  # noqa: F401


def test_interfaces_are_abstract():
    """RegimeModel and Strategy are contracts, not usable classes."""
    with pytest.raises(TypeError):
        RegimeModel()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]
