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
| 2026-06-10 | Regime model input | feature matrix vs. price series | `RegimeModel.fit/predict` take a (single-column) price DataFrame and derive features internally — uniform composition; revisit if Phase 5 needs cross-asset features |
| 2026-06-10 | Regime label convention | int labels vs. float with NaN warm-up | Float labels (0.0, 1.0, ...) with NaN during warm-up; consumers must handle NaN explicitly (gated strategy defaults to flat) |
| 2026-06-10 | Strategy use of regimes (Phase 4) | binary gate vs. continuous scaling | Binary risk-off gate ({calm: 1, turbulent: 0}) for heuristics; continuous scaling by regime probability deferred to Phase 5+ |
| 2026-06-10 | Sharpe-difference inference | t-tests vs. block bootstrap; independent vs. paired | Paired circular block bootstrap (21d blocks, 2,000 draws, percentile 95% CI): respects autocorrelation and cross-correlation; `prob_nonpositive` reported as descriptive, not a formal p-value |
| 2026-06-10 | Exposure maps | fixed a priori vs. learned | Fixed a priori (flat in highest-variance state) to avoid an extra overfitting channel; learning exposures would require nested validation |
| 2026-06-10 | Heuristic walk-forward | expanding vs. rolling; refit frequency | Expanding scheme, ~5y initial train (1260 days), yearly refits (252) — sensitivity to this belongs in Phase 6 robustness checks |
| 2026-06-10 | Statistical model family | Gaussian HMM vs. Markov-switching regression | Gaussian HMM (hmmlearn, diag covariance) on log returns; Markov-switching regression dropped for scope — the HMM answers the regime question directly |
| 2026-06-10 | Causal HMM inference | hmmlearn predict/predict_proba vs. manual forward filter | Manual forward recursion for filtered P(state_t \| r_1..t): hmmlearn's are Viterbi/smoothed and condition on future data; causality enforced by truncation tests |
| 2026-06-10 | EM stability | single fit vs. multiple restarts | n_restarts seeded fits, keep best in-sample log-likelihood; fit metadata (score, restart, convergence) recorded in `fit_info_` |
| 2026-06-10 | State identification across refits | none vs. variance ordering | Relabel states by ascending variance after every fit: regime 0 is always calmest — exposure maps stay meaningful across walk-forward refits |
| 2026-06-10 | Number of regimes | fixed K vs. IC-selected | Compare K=2 and K=3 explicitly as experiment variants rather than auto-selecting; K=2 gates well, K=3's extreme state fires too rarely (2.5% of OOS days) |
