"""Robustness checks on the head-to-head comparison results.

Usage:
    python experiments/run_regime_comparison.py configs/comparison_spy.yaml
    python experiments/run_robustness.py configs/comparison_spy.yaml

Reads the net return series produced by the comparison run (at the headline
cost level) and adds the analysis that turns point estimates into findings:

1. paired block-bootstrap confidence intervals on Sharpe differences
   (each gated strategy vs. its ungated base, and core gated variants pairwise);
2. sub-period Sharpe and max drawdown (is the result one lucky era?);
3. a parameter-sensitivity grid for the volatility gate (is q80/20d a peak
   or a plateau?), recomputed walk-forward at the headline cost level;
4. nuisance-parameter sensitivity: bootstrap block lengths and the
   walk-forward scheme/cadence — conclusions should not hinge on either.

If the config sets ``risk_free``, Sharpe-based quantities use excess returns
(the comparison run already credited the cash leg to net returns).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from regimelab.backtest.bootstrap import sharpe_difference
from regimelab.backtest.costs import ProportionalCost
from regimelab.backtest.engine import run_backtest
from regimelab.backtest.metrics import max_drawdown, sharpe_ratio
from regimelab.config import load_config
from regimelab.data.cleaning import clean_prices, to_returns
from regimelab.data.loaders import load_prices, load_risk_free
from regimelab.regimes import VolatilityThresholdRegime
from regimelab.strategies import BuyAndHold, RegimeGatedStrategy
from regimelab.validation.walkforward import (
    walk_forward_predict_per_asset,
    walk_forward_splits,
)

REPO_ROOT = Path(__file__).parent.parent

_BANDED = re.compile(r"p_b\d")  # banded variants: in vs-base rows, not pairwise


def bootstrap_table(exc: pd.DataFrame, seed: int) -> pd.DataFrame:
    bh_gated = [c for c in exc.columns if c.endswith("_bh")]
    core_bh = [c for c in bh_gated if not _BANDED.search(c)]
    trend_base = [c for c in exc.columns if c.startswith("trend_")]
    trend_gated = [c for c in exc.columns if c.endswith("_trend")]

    pairs = [(c, "buy_and_hold") for c in bh_gated]
    pairs += [(c, trend_base[0]) for c in trend_gated if trend_base]
    pairs += list(combinations(core_bh, 2))

    rows = []
    for a, b in pairs:
        res = sharpe_difference(exc[a], exc[b], n_boot=2000, seed=seed)
        rows.append({"comparison": f"{a} vs {b}", **asdict(res)})
    return pd.DataFrame(rows).set_index("comparison")


def blocklen_table(
    exc: pd.DataFrame, seed: int, block_sizes: tuple[int, ...] = (5, 21, 63)
) -> pd.DataFrame:
    """Are the bootstrap conclusions stable to the block-length choice?"""
    core_bh = [c for c in exc.columns if c.endswith("_bh") and not _BANDED.search(c)]
    rows = []
    for c in core_bh:
        for block in block_sizes:
            res = sharpe_difference(
                exc[c], exc["buy_and_hold"], n_boot=2000, block_size=block, seed=seed
            )
            rows.append(
                {
                    "comparison": f"{c} vs buy_and_hold",
                    "block_size": block,
                    "diff": res.diff,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "prob_nonpositive": res.prob_nonpositive,
                }
            )
    return pd.DataFrame(rows)


def subperiod_table(
    net: pd.DataFrame, exc: pd.DataFrame, subperiods: list[list[str]]
) -> pd.DataFrame:
    rows = []
    for start, end in subperiods:
        era_net = net.loc[str(start) : str(end)]
        era_exc = exc.loc[str(start) : str(end)]
        label = f"{era_net.index.min().date()}..{era_net.index.max().date()}"
        for col in net.columns:
            rows.append(
                {
                    "era": label,
                    "strategy": col,
                    "sharpe": sharpe_ratio(era_exc[col]),
                    "max_drawdown": max_drawdown(era_net[col]),
                }
            )
    return pd.DataFrame(rows).set_index(["era", "strategy"])


def load_universe(p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_prices(
        p.get("universe", ["SPY"]),
        start=p.get("start"),
        end=p.get("end"),
        cache_dir=REPO_ROOT / "data" / "raw",
    )
    prices = clean_prices(prices)
    return prices, to_returns(prices, kind="simple")


def _gated_bh_result(prices, returns, splits, cost_bps, rf, vol_window=20, quantile=0.8):
    def factory():
        return VolatilityThresholdRegime(vol_window=vol_window, quantile=quantile)

    regimes = walk_forward_predict_per_asset(factory, prices, splits)
    strat = RegimeGatedStrategy(BuyAndHold(), regimes, exposure={0.0: 1.0, 1.0: 0.0})
    oos_returns = returns.loc[returns.index.isin(regimes.index)]
    return run_backtest(
        strat.target_positions(prices),
        oos_returns,
        cost_model=ProportionalCost(cost_bps),
        cash_returns=rf,
    )


def vol_gate_sensitivity(
    p: dict, prices, returns, cost_bps: float, rf: pd.Series | None
) -> pd.DataFrame:
    splits = walk_forward_splits(
        prices.index,
        train_size=int(p.get("train_size", 1260)),
        test_size=int(p.get("test_size", 252)),
        scheme=p.get("scheme", "expanding"),
    )
    rows = []
    for window in p.get("sens_vol_windows", [10, 20, 60]):
        for quantile in p.get("sens_vol_quantiles", [0.7, 0.8, 0.9]):
            result = _gated_bh_result(
                prices, returns, splits, cost_bps, rf,
                vol_window=int(window), quantile=float(quantile),
            )
            rows.append(
                {
                    "vol_window": int(window),
                    "quantile": float(quantile),
                    "sharpe": sharpe_ratio(result.net_returns, rf=rf if rf is not None else 0.0),
                    "max_drawdown": max_drawdown(result.net_returns),
                    "annual_turnover": float(result.turnover.mean() * 252),
                }
            )
    return pd.DataFrame(rows)


def walkforward_sensitivity(
    p: dict, prices, returns, cost_bps: float, rf: pd.Series | None
) -> pd.DataFrame:
    """Headline vol gate (20d/q80) under alternative validation schedules."""
    rows = []
    for scheme in ("expanding", "rolling"):
        for test_size in (126, 252):
            splits = walk_forward_splits(
                prices.index,
                train_size=int(p.get("train_size", 1260)),
                test_size=test_size,
                scheme=scheme,
            )
            result = _gated_bh_result(prices, returns, splits, cost_bps, rf)
            rows.append(
                {
                    "scheme": scheme,
                    "test_size": test_size,
                    "sharpe": sharpe_ratio(result.net_returns, rf=rf if rf is not None else 0.0),
                    "max_drawdown": max_drawdown(result.net_returns),
                    "annual_turnover": float(result.turnover.mean() * 252),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg.params
    cost_bps = float(p.get("cost_bps", 5.0))
    out_dir = REPO_ROOT / "experiments" / "outputs" / cfg.name
    net_path = out_dir / "net_returns.csv"
    if not net_path.exists():
        sys.exit(
            f"{net_path} not found - run "
            f"'python experiments/run_regime_comparison.py {args.config}' first."
        )
    net = pd.read_csv(net_path, index_col=0, parse_dates=True)

    rf = None
    exc = net
    if p.get("risk_free"):
        rf = load_risk_free(cache_dir=REPO_ROOT / "data" / "raw", ticker=p["risk_free"])
        exc = net.sub(rf.reindex(net.index), axis=0)

    pd.set_option("display.width", 200)
    cols = ["sharpe_a", "sharpe_b", "diff", "ci_low", "ci_high", "prob_nonpositive"]

    boot = bootstrap_table(exc, seed=cfg.seed)
    boot.to_csv(out_dir / "bootstrap_sharpe_diffs.csv")
    print(f"Paired block-bootstrap Sharpe differences (95% CI, @{cost_bps:g}bps, "
          f"{'excess' if rf is not None else 'raw'} returns):")
    print(boot[cols].round(3))

    blocks = blocklen_table(exc, seed=cfg.seed)
    blocks.to_csv(out_dir / "bootstrap_blocklen_sensitivity.csv", index=False)
    print("\nBlock-length sensitivity (prob_nonpositive by block size):")
    print(
        blocks.pivot(index="comparison", columns="block_size", values="prob_nonpositive")
        .round(3)
    )

    if "subperiods" in p:
        subperiods = p["subperiods"]
    else:  # fall back to date thirds of the out-of-sample window
        parts = np.array_split(net.index, 3)
        subperiods = [[str(part.min().date()), str(part.max().date())] for part in parts]
    sub = subperiod_table(net, exc, subperiods)
    sub.to_csv(out_dir / "subperiod_metrics.csv")
    print("\nSub-period Sharpe:")
    print(sub["sharpe"].unstack("strategy").round(3))
    print("\nSub-period max drawdown:")
    print(sub["max_drawdown"].unstack("strategy").round(3))

    prices, returns = load_universe(p)
    sens = vol_gate_sensitivity(p, prices, returns, cost_bps, rf)
    sens.to_csv(out_dir / "vol_gate_sensitivity.csv", index=False)
    print(f"\nVol-gate sensitivity (Sharpe of gated buy-and-hold @{cost_bps:g}bps):")
    print(sens.pivot(index="vol_window", columns="quantile", values="sharpe").round(3))

    wf = walkforward_sensitivity(p, prices, returns, cost_bps, rf)
    wf.to_csv(out_dir / "walkforward_sensitivity.csv", index=False)
    print("\nWalk-forward schedule sensitivity (vol gate 20d/q80):")
    print(wf.round(3).to_string(index=False))

    print(f"\nWrote bootstrap_sharpe_diffs.csv, bootstrap_blocklen_sensitivity.csv, "
          f"subperiod_metrics.csv, vol_gate_sensitivity.csv, walkforward_sensitivity.csv "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
