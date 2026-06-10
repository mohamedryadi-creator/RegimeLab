"""Robustness checks on the head-to-head comparison results.

Usage:
    python experiments/run_regime_comparison.py configs/comparison_spy.yaml
    python experiments/run_robustness.py configs/comparison_spy.yaml

Reads the net return series produced by the comparison run (at the headline
cost level) and adds the analysis that turns point estimates into findings:

1. paired block-bootstrap confidence intervals on Sharpe differences
   (each gated strategy vs. its ungated base, and gated variants pairwise);
2. sub-period Sharpe and max drawdown (is the result one lucky era?);
3. a parameter-sensitivity grid for the volatility gate (is q80/20d a peak
   or a plateau?), recomputed walk-forward at the headline cost level.
"""

from __future__ import annotations

import argparse
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
from regimelab.data.loaders import load_prices
from regimelab.regimes import VolatilityThresholdRegime
from regimelab.strategies import BuyAndHold, RegimeGatedStrategy
from regimelab.validation.walkforward import (
    walk_forward_predict_per_asset,
    walk_forward_splits,
)

REPO_ROOT = Path(__file__).parent.parent


def bootstrap_table(net: pd.DataFrame, seed: int) -> pd.DataFrame:
    bh_gated = [c for c in net.columns if c.endswith("_bh")]
    trend_base = [c for c in net.columns if c.startswith("trend_")]
    trend_gated = [c for c in net.columns if c.endswith("_trend")]

    pairs = [(c, "buy_and_hold") for c in bh_gated]
    pairs += [(c, trend_base[0]) for c in trend_gated if trend_base]
    pairs += list(combinations(bh_gated, 2))

    rows = []
    for a, b in pairs:
        res = sharpe_difference(net[a], net[b], n_boot=2000, seed=seed)
        rows.append({"comparison": f"{a} vs {b}", **asdict(res)})
    return pd.DataFrame(rows).set_index("comparison")


def subperiod_table(net: pd.DataFrame, subperiods: list[list[str]]) -> pd.DataFrame:
    rows = []
    for start, end in subperiods:
        era = net.loc[str(start) : str(end)]
        label = f"{era.index.min().date()}..{era.index.max().date()}"
        for col in net.columns:
            rows.append(
                {
                    "era": label,
                    "strategy": col,
                    "sharpe": sharpe_ratio(era[col]),
                    "max_drawdown": max_drawdown(era[col]),
                }
            )
    return pd.DataFrame(rows).set_index(["era", "strategy"])


def vol_gate_sensitivity(p: dict, cost_bps: float) -> pd.DataFrame:
    prices = load_prices(
        p.get("universe", ["SPY"]),
        start=p.get("start"),
        end=p.get("end"),
        cache_dir=REPO_ROOT / "data" / "raw",
    )
    prices = clean_prices(prices)
    returns = to_returns(prices, kind="simple")
    splits = walk_forward_splits(
        prices.index,
        train_size=int(p.get("train_size", 1260)),
        test_size=int(p.get("test_size", 252)),
        scheme=p.get("scheme", "expanding"),
    )
    rows = []
    for window in p.get("sens_vol_windows", [10, 20, 60]):
        for quantile in p.get("sens_vol_quantiles", [0.7, 0.8, 0.9]):
            def factory(window=window, quantile=quantile):
                return VolatilityThresholdRegime(
                    vol_window=int(window), quantile=float(quantile)
                )

            regimes = walk_forward_predict_per_asset(factory, prices, splits)
            strat = RegimeGatedStrategy(BuyAndHold(), regimes, exposure={0.0: 1.0, 1.0: 0.0})
            oos_returns = returns.loc[returns.index.isin(regimes.index)]
            result = run_backtest(
                strat.target_positions(prices), oos_returns, cost_model=ProportionalCost(cost_bps)
            )
            rows.append(
                {
                    "vol_window": int(window),
                    "quantile": float(quantile),
                    "sharpe": sharpe_ratio(result.net_returns),
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

    pd.set_option("display.width", 200)

    boot = bootstrap_table(net, seed=cfg.seed)
    boot.to_csv(out_dir / "bootstrap_sharpe_diffs.csv")
    cols = ["sharpe_a", "sharpe_b", "diff", "ci_low", "ci_high", "prob_nonpositive"]
    print(f"Paired block-bootstrap Sharpe differences (95% CI, @{cost_bps:g}bps):")
    print(boot[cols].round(3))

    if "subperiods" in p:
        subperiods = p["subperiods"]
    else:  # fall back to date thirds of the out-of-sample window
        parts = np.array_split(net.index, 3)
        subperiods = [[str(part.min().date()), str(part.max().date())] for part in parts]
    sub = subperiod_table(net, subperiods)
    sub.to_csv(out_dir / "subperiod_metrics.csv")
    print("\nSub-period Sharpe:")
    print(sub["sharpe"].unstack("strategy").round(3))
    print("\nSub-period max drawdown:")
    print(sub["max_drawdown"].unstack("strategy").round(3))

    sens = vol_gate_sensitivity(p, cost_bps)
    sens.to_csv(out_dir / "vol_gate_sensitivity.csv", index=False)
    print(f"\nVol-gate sensitivity (Sharpe of gated buy-and-hold @{cost_bps:g}bps):")
    print(sens.pivot(index="vol_window", columns="quantile", values="sharpe").round(3))

    print(f"\nWrote bootstrap_sharpe_diffs.csv, subperiod_metrics.csv, "
          f"vol_gate_sensitivity.csv to {out_dir}")


if __name__ == "__main__":
    main()
