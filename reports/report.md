# Does regime detection make simple strategies more robust?

**RegimeLab research report — June 2026.**
All results are out-of-sample, net of transaction costs, reproducible via the
commands in the appendix.

## Summary

On the S&P 500 ETF (SPY, 2005–2026, walk-forward validation, 5 bps proportional
costs), gating buy-and-hold with either a simple volatility filter or a 2-state
Gaussian HMM raised the Sharpe ratio from 0.64 to ≈0.82 and cut the maximum
drawdown from −55% to −34% (vol gate) or −23% (HMM). However, paired
block-bootstrap intervals show that **none of the Sharpe improvements are
statistically distinguishable from zero at the 95% level** over this 21-year
sample, and the Sharpe advantage is concentrated in the 2005–2009 crisis era.
The robust, consistent benefit of regime detection is **drawdown control** —
crash insurance — not higher risk-adjusted returns. The statistical model
(HMM) matched but did not beat the heuristic filter on Sharpe; it controlled
drawdowns better at the price of ~3× the turnover, making it more
cost-fragile.

## 1. Research question

Can market regime detection improve the *robustness* — not the headline
performance — of simple strategies, out-of-sample and after costs? And does a
statistical regime model (HMM) earn its complexity over a transparent
heuristic?

## 2. Data and protocol

- **Data.** SPY adjusted closes, 2000–2026, daily (yfinance snapshot, cached
  immutably with download metadata; see `data/README.md`).
- **Validation.** Expanding walk-forward: ~5 years initial training
  (1,260 days), refit every 252 days, 22 refits. All models see only past data
  at every fit; only out-of-sample regime labels are used. Out-of-sample
  window: **2005-01-07 to 2026-06-09** (5,388 days), identical for every
  strategy.
- **Execution.** Positions decided at close *t* earn the return of *t+1*; the
  lag lives in the backtest engine, never in strategies, and is enforced by
  tests (a clairvoyant signal is neutralized; features and regime labels pass
  truncation causality tests).
- **Costs.** Proportional, 5 bps per unit turnover (headline), swept over
  {0, 5, 20} bps.

## 3. Strategies

| Name | Description |
|---|---|
| `buy_and_hold` | Fully invested at all times |
| `trend_200d` | Long when price > 200-day MA, else cash |
| `vol20q80_bh` | Buy-and-hold, flat when 20d realized vol > the 80% quantile of training-sample vol (walk-forward refit) |
| `hmm2_bh` | Buy-and-hold, flat when a 2-state Gaussian HMM's *filtered* state is the high-variance one |
| `hmm3_bh` | Buy-and-hold, flat only in the most turbulent of 3 HMM states |
| `*_trend` | Same gates applied to the trend strategy |

HMM details: EM on log returns with 3 seeded restarts (best in-sample
likelihood kept), states relabeled by ascending variance at every refit,
prediction via a causal forward recursion (filtered probabilities — smoothed
inference would leak future data).

## 4. Results (out-of-sample, 5 bps)

| Strategy | Sharpe | Ann. return | Ann. vol | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| buy_and_hold | 0.64 | 10.9% | 19.0% | −55% | 0.05× |
| trend_200d | 0.72 | 8.0% | 11.6% | −22% | 6.3× |
| vol20q80_bh | 0.82 | 10.1% | 12.8% | −34% | 3.4× |
| hmm2_bh | 0.81 | 9.0% | 11.4% | −23% | 9.4× |
| hmm3_bh | 0.65 | 10.0% | 17.1% | −52% | 4.1× |
| vol20q80_trend | 0.77 | 8.2% | 11.1% | −22% | 5.9× |
| hmm2_trend | 0.78 | 7.7% | 10.3% | −19% | 7.9× |

### 4.1 Are the Sharpe differences real?

Paired circular block bootstrap (21-day blocks, 2,000 draws, identical date
blocks applied to both series). `P(≤0)` is the bootstrap fraction of draws
where the difference is non-positive (descriptive, not a formal p-value):

| Comparison | ΔSharpe | 95% CI | P(≤0) |
|---|---|---|---|
| vol20q80_bh vs buy_and_hold | +0.18 | [−0.12, +0.50] | 0.14 |
| hmm2_bh vs buy_and_hold | +0.17 | [−0.16, +0.52] | 0.19 |
| hmm3_bh vs buy_and_hold | +0.00 | [−0.15, +0.17] | 0.53 |
| vol20q80_bh vs hmm2_bh | +0.01 | [−0.22, +0.22] | 0.49 |
| hmm2_bh vs hmm3_bh | +0.17 | [−0.13, +0.49] | 0.16 |

**Every interval includes zero.** Twenty-one years of daily data are not
enough to claim a Sharpe improvement of ~0.17 with 95% confidence — a sobering
and central result. The vol gate and the 2-state HMM are statistically
indistinguishable from each other (ΔSharpe +0.01).

### 4.2 Is it one lucky era?

Sub-period Sharpe (max drawdown in parentheses):

| Era | buy_and_hold | vol20q80_bh | hmm2_bh |
|---|---|---|---|
| 2005–2009 | 0.15 (−55%) | 0.26 (−34%) | 0.50 (−23%) |
| 2010–2019 | 0.93 (−19%) | 1.00 (−19%) | 0.93 (−17%) |
| 2020–2026 | 0.81 (−34%) | 0.99 (−16%) | 0.86 (−16%) |

The Sharpe advantage comes mostly from the 2008 crisis (and, for the vol gate,
the 2020 crash); in the calm 2010s the gates neither helped nor hurt. The
**drawdown reduction, by contrast, appears in every era** — consistent with
the crash-insurance interpretation.

### 4.3 Parameter and cost sensitivity

Vol-gate Sharpe across the parameter grid (gated buy-and-hold, 5 bps) is a
plateau, not a peak — every cell beats buy-and-hold's 0.64:

| vol window \ quantile | 0.7 | 0.8 | 0.9 |
|---|---|---|---|
| 10d | 0.71 | 0.78 | 0.75 |
| 20d | 0.70 | 0.82 | 0.79 |
| 60d | 0.72 | 0.71 | 0.75 |

Costs: from 0 → 20 bps, the vol gate loses 0.05 Sharpe (0.83 → 0.78) while the
HMM loses 0.16 (0.85 → 0.69) — the HMM's filtered state flickers (9.4× annual
turnover vs. 3.4×), so its edge erodes ~3× faster with costs. At 20 bps the
heuristic is clearly preferable.

### 4.4 K = 3

The 3-state HMM gated on its most turbulent state was flat on only 2.5% of
out-of-sample days — too rare to change anything (Sharpe 0.65 ≈ ungated 0.64).
Most of the useful signal lives in the 2-state calm/turbulent split; the third
state mainly relabels moderate volatility.

## 5. Limitations

- Single asset and single market (SPY); conclusions may not transfer.
- yfinance data; survivorship is not an issue for an index ETF, but early-year
  adjusted prices are vendor-dependent.
- The cost model is proportional only (no spread/impact decomposition), though
  the sweep brackets plausible levels for SPY.
- Sharpe-difference inference uses percentile intervals with one block length
  (21 days); block-length sensitivity was not explored.
- Exposure maps (which regimes are "risk-off") were fixed a priori, not
  learned — deliberately, to avoid an extra overfitting channel.

## 6. Conclusions

1. Regime gating is best understood as **drawdown insurance**: −55% → −23/34%
   max drawdown, consistently across eras, while keeping ≈ 90% of
   buy-and-hold's annualized return.
2. Sharpe improvements (~+0.17) are real in-sample-of-history but **not
   statistically resolvable** even with 21 years of daily data — claims of
   regime-based Sharpe enhancement should be treated with suspicion at this
   sample size.
3. The **HMM did not earn its complexity** over the volatility heuristic on
   risk-adjusted returns (ΔSharpe +0.01), though it is genuinely better at
   drawdown control. Its higher turnover makes it strictly worse once costs
   reach 20 bps.
4. Methodological discipline (centralized execution lag, filtered-only HMM
   inference, walk-forward refits, state relabeling) was load-bearing: each of
   these, if relaxed, inflates results.

## Appendix: reproducibility

```bash
pip install -e ".[models,data,dev]"
pytest                                                          # 73 tests
python experiments/run_regime_comparison.py configs/comparison_spy.yaml
python experiments/run_robustness.py configs/comparison_spy.yaml
```

Outputs land in `experiments/outputs/comparison_spy/`: `summary.csv` (all
strategies × cost levels), `net_returns.csv`, `regimes_*.csv` (OOS labels),
`bootstrap_sharpe_diffs.csv`, `subperiod_metrics.csv`,
`vol_gate_sensitivity.csv`. Methodological decisions and their rationale:
`reports/decisions.md`.
