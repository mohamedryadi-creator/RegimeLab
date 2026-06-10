"""Head-to-head: baselines vs. regime-gated variants, out-of-sample.

Usage:
    python experiments/run_regime_comparison.py configs/comparison_spy.yaml

Regime models are declared in the config (``regime_models`` list, each with a
``type``, a ``name``, constructor parameters, and an ``exposure`` map from
regime label to position scale). Every model is refitted on the same
walk-forward schedule and only its out-of-sample labels are used for gating.
All strategies — including the ungated baselines — are evaluated on the
identical out-of-sample period and cost levels, so differences are
attributable to the regime signals alone.
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
from regimelab.data.loaders import load_prices, load_risk_free
from regimelab.regimes import regime_model_from_spec
from regimelab.strategies import (
    BandedStrategy,
    BuyAndHold,
    ProbabilityScaledStrategy,
    RegimeGatedStrategy,
    TrendFollowing,
)
from regimelab.validation.walkforward import (
    walk_forward_predict_per_asset,
    walk_forward_proba_per_asset,
    walk_forward_splits,
)

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

    # Optional cash leg: uninvested weight earns the (lagged) T-bill rate, and
    # Sharpe/Sortino are computed on excess returns.
    rf = None
    if p.get("risk_free"):
        rf = load_risk_free(cache_dir=REPO_ROOT / "data" / "raw", ticker=p["risk_free"])
        print(f"Cash leg: {p['risk_free']}, mean annualized rate "
              f"{rf.reindex(returns.index).mean() * 252:.2%}")

    splits = walk_forward_splits(
        prices.index,
        train_size=int(p.get("train_size", 1260)),
        test_size=int(p.get("test_size", 252)),
        scheme=p.get("scheme", "expanding"),
    )

    out_dir = REPO_ROOT / "experiments" / "outputs" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies: dict[str, object] = {
        "buy_and_hold": BuyAndHold(),
        f"trend_{trend_window}d": TrendFollowing(window=trend_window),
    }
    oos = None
    for spec in p.get("regime_models", []):
        spec = dict(spec)
        name = spec.get("name", spec["type"])
        if spec["type"] == "hmm":
            spec.setdefault("seed", cfg.seed)
        exposure = {float(k): float(v) for k, v in spec.get("exposure", {0: 1.0, 1: 0.0}).items()}

        def factory(spec=spec):
            return regime_model_from_spec(spec)

        # One independently fitted model per asset (regime models are per-asset).
        regimes = walk_forward_predict_per_asset(factory, prices, splits)
        regimes.to_csv(out_dir / f"regimes_{name}.csv")
        oos = regimes.index
        for col in regimes.columns:
            shares = regimes[col].value_counts(normalize=True).sort_index()
            dist = ", ".join(f"{int(k)}: {v:.1%}" for k, v in shares.items())
            print(f"{name}/{col}: OOS regime shares {{{dist}}}, {len(splits)} refits")

        strategies[f"{name}_bh"] = RegimeGatedStrategy(BuyAndHold(), regimes, exposure)
        strategies[f"{name}_trend"] = RegimeGatedStrategy(
            TrendFollowing(window=trend_window), regimes, exposure
        )

        bands = [float(b) for b in spec.get("bands", [])]
        if spec.get("probability_scaled") or bands:
            # Continuous variant: exposure_t = sum_k exposure(k) * P(state k | past),
            # from walk-forward *filtered* probabilities (never smoothed).
            probs = walk_forward_proba_per_asset(factory, prices, splits)
            probs.to_csv(out_dir / f"probs_{name}.csv")
            weights = pd.DataFrame(
                {
                    asset: sum(
                        exposure.get(float(state), 0.0) * probs[(asset, state)]
                        for state in probs[asset].columns
                    )
                    for asset in prices.columns
                }
            )
            scaled = ProbabilityScaledStrategy(BuyAndHold(), weights)
            if spec.get("probability_scaled"):
                strategies[f"{name}p_bh"] = scaled
            for band in bands:
                # No-trade band on the scaled exposure; reported as a sweep,
                # never a selected "best" band.
                label = f"{name}p_b{int(round(band * 100)):02d}_bh"
                strategies[label] = BandedStrategy(scaled, band)

    if oos is None:
        oos = prices.index[len(splits[0][0]) :] if splits else prices.index
    print(f"Out-of-sample window: {oos.min().date()} to {oos.max().date()} ({len(oos)} days)")

    tables = []
    net_series = {}
    oos_returns = returns.loc[returns.index.isin(oos)]
    for name, strat in strategies.items():
        # Positions are computed on the full price history (warm-ups resolved),
        # but performance is measured on the common out-of-sample period only.
        positions = strat.target_positions(prices)
        for bps in cost_sweep:
            result = run_backtest(
                positions, oos_returns, cost_model=ProportionalCost(bps), cash_returns=rf
            )
            tables.append(
                summary(
                    result.net_returns,
                    result.turnover,
                    name=f"{name}@{bps:g}bps",
                    rf=rf if rf is not None else 0.0,
                )
            )
            if bps == cost_bps:
                net_series[name] = result.net_returns

    table = pd.concat(tables)
    table.to_csv(out_dir / "summary.csv")
    pd.DataFrame(net_series).to_csv(out_dir / "net_returns.csv")
    if rf is not None:
        rf.reindex(oos_returns.index).to_csv(out_dir / "risk_free.csv")

    pd.set_option("display.width", 160)
    print(f"\n{table.round(3)}")
    print(f"\nWrote summary.csv, net_returns.csv, regimes_*.csv to {out_dir}")


if __name__ == "__main__":
    main()
