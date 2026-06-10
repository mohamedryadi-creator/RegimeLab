# Notebooks

Exploration and figure-making **only**. Rules:

- no core logic here — anything worth keeping moves into `src/regimelab/` with tests;
- notebooks only load files from `experiments/outputs/` and call `regimelab.plotting`;
- figures are saved to `reports/figures/` (committed) for use in the report;
- restart-and-run-all before committing a notebook.

| Notebook | Inputs (regenerate first) | Figures |
|---|---|---|
| `01_strategy_performance` | `run_regime_comparison.py configs/comparison_spy.yaml` | `spy_wealth`, `spy_drawdowns` |
| `02_regimes_and_probabilities` | same | `spy_hmm2_regimes` |
| `03_robustness` | `run_robustness.py configs/comparison_spy.yaml` | `spy_bootstrap`, `spy_sensitivity`, `spy_subperiods` |
| `04_multi_asset` | both runners with `configs/multiasset.yaml` | `multiasset_wealth`, `multiasset_bootstrap` |

Headless re-execution:

```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```
