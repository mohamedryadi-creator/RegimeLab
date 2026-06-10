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

## Methodological decisions

Decisions made so far are recorded in [`reports/decisions.md`](reports/decisions.md)
(universe: SPY to start; daily frequency; yfinance with an immutable local cache;
proportional costs with a mandatory 0/5/20 bps sensitivity sweep; execution lag
centralized in the engine; rf = 0 in Sharpe).

Still intentionally **open**:

| Decision | Main options |
|---|---|
| Regime definition | Volatility-based, trend-based, or model-implied (latent states) |
| Statistical model | Gaussian HMM vs. Markov-switching regression vs. both |
| Number of regimes | Fixed (2–3) vs. selected by information criteria |
| Walk-forward scheme per experiment | Expanding vs. rolling; refit frequency; embargo length |
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

- [x] **Phase 0 — Skeleton.** Package structure, interfaces, test scaffolding.
- [x] **Phase 1 — Data layer.** SPY via yfinance with an immutable CSV cache and
      metadata sidecars; conservative cleaning with loud failures; return conventions.
- [x] **Phase 2 — Metrics and backtest engine.** Vectorized backtester with centralized
      execution lag, proportional costs, turnover; metrics suite (Sharpe, Sortino, max
      drawdown, volatility, hit ratio) tested against hand-computed examples;
      walk-forward split generator (expanding/rolling, embargo).
- [x] **Phase 3 — Baselines.** Buy-and-hold and trend-following, plus a config-driven
      runner (`experiments/run_baselines.py`) with a cost sensitivity sweep.
- [ ] **Phase 4 — Heuristic regimes.** Volatility-threshold and trend filters as
      `RegimeModel`s; regime-gated baseline variants; cheap benchmark for Phase 5.
- [ ] **Phase 5 — Statistical regime models.** Gaussian HMM (and/or Markov-switching)
      with walk-forward refitting, filtered probabilities only, seed-controlled EM
      restarts, regime identification across refits; persistence diagnostics.
- [ ] **Phase 6 — Experiments and report.** Head-to-head experiment (baselines vs.
      heuristic vs. statistical regimes, identical splits and costs), robustness checks
      (cost sweep, sub-periods, parameter sensitivity), bootstrap intervals on Sharpe
      differences; short research report in `reports/`.

## Running the baselines

```bash
python experiments/run_baselines.py configs/baselines_spy.yaml
```

The first run downloads SPY history into `data/raw/` (requires
`pip install -e ".[data]"`); subsequent runs use the cache. Outputs land in
`experiments/outputs/baselines_spy/`.

## Status

Phases 0–3 complete: data layer, metrics, backtest engine, validation splits, and
baselines are implemented and tested (including look-ahead bias tests). Next:
heuristic regime filters (Phase 4), then statistical regime models (Phase 5).
