"""Statistical regime models (Hidden Markov Models, Markov-switching models).

Open decisions (Phase 5):
- Model family: Gaussian HMM on returns/features (hmmlearn) vs. Markov-switching
  regression (statsmodels) vs. both for comparison.
- Number of regimes: fixed small K (2-3, interpretable) vs. selected by AIC/BIC.
- Refit frequency within walk-forward validation, and how to handle label
  switching across refits (regimes must be identified, e.g. ordered by variance).

Implementation constraints:
- Use *filtered* state probabilities for any signal that feeds a strategy
  (smoothed probabilities use future data — diagnostics only).
- Seeds must be controlled; EM is sensitive to initialization, so use multiple
  restarts and record the chosen one.

Dependencies live in the ``models`` extra (``pip install -e ".[models]"``).

TODO (Phase 5): implement GaussianHMMRegime and/or MarkovSwitchingRegime,
both implementing :class:`regimelab.regimes.base.RegimeModel`.
"""
