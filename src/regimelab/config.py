"""Experiment configuration handling.

Every experiment is fully described by a YAML config file (in ``configs/``) plus
the code version: name, seed, and a free-form ``params`` mapping whose keys are
validated by the experiment script that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """A minimal, immutable description of one experiment run."""

    name: str
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "name" not in raw:
        raise ValueError(f"{path} must be a mapping with at least a 'name' key.")
    unknown = set(raw) - {"name", "seed", "params"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys in {path}: {sorted(unknown)}.")
    return ExperimentConfig(
        name=str(raw["name"]),
        seed=int(raw.get("seed", 0)),
        params=dict(raw.get("params") or {}),
    )
