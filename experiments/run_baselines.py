"""Run the baseline strategies (buy-and-hold, trend-following) from a config.

Usage:
    python experiments/run_baselines.py configs/baselines_spy.yaml

Downloads/loads the configured universe, backtests both baselines net of
proportional costs, sweeps the cost level as a sensitivity check, and writes
summary tables and net return series to experiments/outputs/<name>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from regimelab.backtest.costs import ProportionalCost
from regimelab.backtest.engine import run_backtest
from regimelab.backtest.metrics import summary
from regimelab.config import load_config
from regimelab.data.cleaning import clean_prices, to_returns
from regimelab.data.loaders import load_prices
from regimelab.strategies.baselines import BuyAndHold, TrendFollowing

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg.params
    universe = p.get("universe", ["SPY"])
    cost_bps = float(p.get("cost_bps", 5.0))
    cost_sweep = [float(c) for c in p.get("cost_sweep_bps", [0.0, cost_bps, 4 * cost_bps])]
    trend_window = int(p.get("trend_window", 200))

    prices = load_prices(
        universe,
        start=p.get("start"),
        end=p.get("end"),
        cache_dir=REPO_ROOT / "data" / "raw",
    )
    prices = clean_prices(prices)
    returns = to_returns(prices, kind="simple")
    print(f"Universe {universe}: {len(prices)} days, {prices.index.min().date()} "
          f"to {prices.index.max().date()}")

    strategies = {
        "buy_and_hold": BuyAndHold(),
        f"trend_{trend_window}d": TrendFollowing(window=trend_window),
    }

    out_dir = REPO_ROOT / "experiments" / "outputs" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    net_series = {}
    for name, strat in strategies.items():
        positions = strat.target_positions(prices)
        for bps in cost_sweep:
            result = run_backtest(positions, returns, cost_model=ProportionalCost(bps))
            label = f"{name}@{bps:g}bps"
            tables.append(summary(result.net_returns, result.turnover, name=label))
            if bps == cost_bps:
                net_series[name] = result.net_returns

    table = pd.concat(tables)
    table.to_csv(out_dir / "summary.csv")
    pd.DataFrame(net_series).to_csv(out_dir / "net_returns.csv")

    pd.set_option("display.width", 140)
    print(f"\n{table.round(3)}")
    print(f"\nWrote {out_dir / 'summary.csv'} and net_returns.csv")


if __name__ == "__main__":
    main()
