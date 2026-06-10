# RegimeLab

A reproducible Python research platform for **market regime detection** and its impact on
**simple systematic strategies**, with an emphasis on out-of-sample robustness and
transaction costs.

## Research question

> Can market regime detection models (heuristic filters, Hidden Markov Models,
> Markov-switching models) improve the *robustness* — not just the in-sample performance —
> of simple trading or allocation strategies, once we account for transaction costs,
> turnover, and proper out-of-sample validation?

The goal is **not** to build a trading bot or to maximize backtest performance. It is to
build a rigorous research framework where regime-aware strategies can be compared fairly
against simple baselines (buy-and-hold, trend-following), and where common methodological
errors (look-ahead bias, in-sample overfitting, ignored costs) are structurally prevented
or explicitly tested for.

## Project structure

```
RegimeLab/
├── pyproject.toml          # Package metadata, dependencies, tooling config
├── src/regimelab/          # The research library (installable package)
│   ├── data/               # Data loading and cleaning
│   ├── features/           # Feature engineering for financial time series
│   ├── regimes/            # Regime detection: heuristic filters and statistical models
│   ├── strategies/         # Trading/allocation strategies (baselines and regime-aware)
│   ├── backtest/           # Backtesting engine, transaction costs, performance metrics
│   ├── validation/         # Train/test splits and walk-forward validation
│   └── config.py           # Experiment configuration loading
├── tests/                  # Unit tests, including look-ahead bias checks
├── configs/                # Versioned experiment configurations (YAML)
├── experiments/            # Experiment runner scripts and their outputs
├── notebooks/              # Exploration and figures ONLY — no core logic
├── data/                   # Local datasets (gitignored; structure only is versioned)
└── reports/                # Research notes and the eventual short research report
```

### Design principles

1. **Library vs. experiments.** All logic lives in `src/regimelab/` and is unit-tested.
   Notebooks and experiment scripts only *call* the library. Results must be reproducible
   from a config file and a seed, never from notebook state.
2. **Point-in-time discipline.** Every component that transforms a time series must be
   explicit about what information is available at each date. Signals are lagged before
   they become positions; regime models are fitted only on past data within walk-forward
   splits. Tests enforce this.
3. **Costs and turnover are first-class.** A strategy result without transaction costs and
   turnover statistics is considered incomplete.
4. **Baselines first.** No regime model is evaluated before buy-and-hold and a simple
   trend-following baseline are implemented and validated.
5. **Flexible methodology.** Interfaces (`RegimeModel`, `Strategy`) are fixed early;
   concrete methodological choices (universe, frequency, model family, validation scheme)
   are deliberately left open and recorded as decisions in `reports/decisions.md` when made.

## Open methodological decisions

These are intentionally **not** decided yet:

| Decision | Main options |
|---|---|
| Asset universe | Single equity index (e.g. S&P 500) vs. multi-asset (equities, bonds, gold) |
| Data frequency | Daily (recommended starting point) vs. weekly |
| Regime definition | Volatility-based, trend-based, or model-implied (latent states) |
| Statistical model | Gaussian HMM vs. Markov-switching regression vs. both |
| Number of regimes | Fixed (2–3) vs. selected by information criteria |
| Validation scheme | Single train/test split vs. expanding walk-forward vs. rolling walk-forward |
| Cost model | Fixed bps per trade vs. spread + impact components |
| Strategy use of regimes | Binary filter (risk-on/off) vs. continuous position scaling |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Optional extras: `pip install -e ".[models,data,notebooks]"` once we commit to the
modeling and data-source choices.

## Development roadmap

- [ ] **Phase 0 — Skeleton (this commit).** Package structure, interfaces, test scaffolding.
- [ ] **Phase 1 — Data layer.** Choose data source and universe; implement loading,
      cleaning, and a cached local data format; validate data quality.
- [ ] **Phase 2 — Metrics and backtest engine.** Vectorized backtester with explicit
      signal lagging, transaction costs, turnover; full metrics suite (Sharpe, Sortino,
      max drawdown, volatility, hit ratio, drawdown durations); tests against hand-computed
      examples.
- [ ] **Phase 3 — Baselines.** Buy-and-hold and trend-following; establish the reference
      results that any regime model must beat *after costs, out-of-sample*.
- [ ] **Phase 4 — Heuristic regimes.** Simple volatility / trend filters as a cheap
      benchmark for the statistical models.
- [ ] **Phase 5 — Statistical regime models.** HMM and/or Markov-switching models with
      walk-forward refitting; regime stability and persistence diagnostics.
- [ ] **Phase 6 — Experiments and report.** Config-driven experiment runner, robustness
      checks (cost sensitivity, sub-period analysis), short research report in `reports/`.

## Status

Phase 0. The repository is a skeleton: interfaces and module layout are in place, the
implementations are intentionally `TODO`.
