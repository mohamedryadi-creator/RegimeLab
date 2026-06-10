"""Head-to-head: baselines vs. regime-gated variants, out-of-sample.

Usage:
    python experiments/run_regime_comparison.py configs/regimes_spy.yaml

The volatility regime model is refitted on a walk-forward schedule and only its
out-of-sample labels are used for gating. All strategies — including the
ungated baselines — are evaluated on the identical out-of-sample period and
cost levels, so differences are attributable to the regime signal alone.
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
from regimelab.regimes import VolatilityThresholdRegime
from regimelab.strategies import BuyAndHold, RegimeGatedStrategy, TrendFollowing
from regimelab.validation.walkforward import walk_forward_predict, walk_forward_splits

REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg.params
    cost_bps = float(p.get("cost_bps", 5.0))
    cost_sweep = [float(c) for c in p.get("cost_sweep_bps", [0.0, cost_bps, 4 * cost_bps])]
    trend_window = int(p.get("trend_window", 200))

    prices = load_prices(
        p.get("universe", ["SPY"]),
        start=p.get("start"),
        end=p.get("end"),
        cache_dir=REPO_ROOT / "data" / "raw",
    )
    prices = clean_prices(prices)
    returns = to_returns(prices, kind="simple")

    # Out-of-sample regime labels via walk-forward refitting.
    splits = walk_forward_splits(
        prices.index,
        train_size=int(p.get("train_size", 1260)),
        test_size=int(p.get("test_size", 252)),
        scheme=p.get("scheme", "expanding"),
    )
    vol_model = VolatilityThresholdRegime(
        vol_window=int(p.get("vol_window", 20)),
        quantile=float(p.get("vol_quantile", 0.8)),
    )
    regimes = walk_forward_predict(vol_model, prices, splits)
    oos = regimes.index
    print(
        f"Out-of-sample: {oos.min().date()} to {oos.max().date()} "
        f"({len(oos)} days, {len(splits)} refits); "
        f"turbulent fraction: {regimes.mean():.1%}"
    )

    risk_off = {0.0: 1.0, 1.0: 0.0}
    strategies = {
        "buy_and_hold": BuyAndHold(),
        f"trend_{trend_window}d": TrendFollowing(window=trend_window),
        "volgate_bh": RegimeGatedStrategy(BuyAndHold(), regimes, exposure=risk_off),
        f"volgate_trend_{trend_window}d": RegimeGatedStrategy(
            TrendFollowing(window=trend_window), regimes, exposure=risk_off
        ),
    }

    out_dir = REPO_ROOT / "experiments" / "outputs" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    net_series = {}
    oos_returns = returns.loc[returns.index.isin(oos)]
    for name, strat in strategies.items():
        # Positions are computed on the full price history (warm-ups resolved),
        # but performance is measured on the common out-of-sample period only.
        positions = strat.target_positions(prices)
        for bps in cost_sweep:
            result = run_backtest(positions, oos_returns, cost_model=ProportionalCost(bps))
            tables.append(summary(result.net_returns, result.turnover, name=f"{name}@{bps:g}bps"))
            if bps == cost_bps:
                net_series[name] = result.net_returns

    table = pd.concat(tables)
    table.to_csv(out_dir / "summary.csv")
    pd.DataFrame(net_series).to_csv(out_dir / "net_returns.csv")
    regimes.to_csv(out_dir / "regimes_oos.csv")

    pd.set_option("display.width", 160)
    print(f"\n{table.round(3)}")
    print(f"\nWrote summary.csv, net_returns.csv, regimes_oos.csv to {out_dir}")


if __name__ == "__main__":
    main()
