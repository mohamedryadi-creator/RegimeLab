# Methodological decision log

Each entry records a decision, the options considered, and the rationale.
Keeping this log prevents silent methodology drift and feeds directly into the
final report's methodology section.

| Date | Decision | Options considered | Choice & rationale |
|---|---|---|---|
| 2026-06-10 | Repository architecture | flat scripts vs. installable library + experiments | Library (`src/` layout) with config-driven experiments, for testability and reproducibility |
| 2026-06-10 | Initial universe | single equity index vs. multi-asset | Start with SPY (S&P 500 ETF) only; the framework is multi-asset-ready (all code paths take DataFrames), extension is a config change |
| 2026-06-10 | Data source | yfinance vs. committed CSV snapshot vs. paid API | yfinance with immutable local CSV cache + JSON sidecar (source, download date); experiments run off the cache, so results are reproducible from the snapshot |
| 2026-06-10 | Frequency | daily vs. weekly | Daily: enough observations for regime models, standard for this literature |
| 2026-06-10 | Return conventions | simple vs. log | Simple returns for backtesting (portfolio returns aggregate linearly); log returns for statistical modeling |
| 2026-06-10 | Execution lag | signal lagging inside strategies vs. centralized in engine | Centralized in `run_backtest` (`lag>=1` enforced); strategies never shift their own signals — makes look-ahead testable in one place |
| 2026-06-10 | Cost model | proportional bps vs. spread+impact | Proportional (5 bps base) with a mandatory sensitivity sweep (0/5/20 bps) in every experiment; refine only if results prove cost-sensitive |
| 2026-06-10 | Risk-free rate in Sharpe | rf = 0 vs. actual short rate | rf = 0 for now (documented in metrics); revisit if multi-asset extension adds a cash leg |
| 2026-06-10 | Validation primitives | single split vs. walk-forward | Walk-forward split generator (expanding and rolling schemes, optional embargo); the scheme used per experiment is chosen in Phase 5 |
| | Regime model family & K | HMM vs. Markov-switching; fixed K vs. IC-selected | *open (Phase 5)* |
| | Strategy use of regimes | binary gate vs. continuous scaling | *open (Phase 4–5)* |
