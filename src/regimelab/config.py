"""Experiment configuration handling.

Every experiment must be fully described by a config file (in ``configs/``) plus a
random seed, so that results are reproducible without any notebook state.

TODO (Phase 6, but the dataclass may grow earlier):
- Decide on the config format (YAML is the working assumption).
- Add validation of config contents (dates, universe, cost parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """A minimal, immutable description of one experiment run.

    Fields will be extended as the methodology is fixed; keep this the single
    source of truth for anything that can change a result.
    """

    name: str
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a config file.

    TODO: implement YAML loading once the config schema is decided.
    """
    raise NotImplementedError("Config loading is implemented in Phase 6.")
