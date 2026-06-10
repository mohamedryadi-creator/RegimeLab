"""Config loading tests."""

from pathlib import Path

import pytest

from regimelab.config import load_config


def test_load_config_roundtrip(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: demo\nseed: 7\nparams:\n  universe: [SPY]\n  cost_bps: 5\n")
    cfg = load_config(path)
    assert cfg.name == "demo"
    assert cfg.seed == 7
    assert cfg.params["universe"] == ["SPY"]


def test_load_config_defaults(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: minimal\n")
    cfg = load_config(path)
    assert cfg.seed == 0
    assert cfg.params == {}


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: x\ntypo_key: 1\n")
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_config(path)


def test_example_config_is_valid():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "example.yaml")
    assert cfg.name
