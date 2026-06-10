"""Statistical regime models (Hidden Markov Models).

Methodological choices (recorded in reports/decisions.md):

- Gaussian HMM on *log returns* derived internally from the price input,
  fitted by EM (hmmlearn) with seed-controlled restarts; the restart with the
  best in-sample log-likelihood is kept and recorded in ``fit_info_``.
- **Filtered, never smoothed.** hmmlearn's ``predict`` (Viterbi) and
  ``predict_proba`` (forward-backward) condition on the full sequence — future
  data. Prediction here uses a manual forward recursion computing
  ``P(state_t | r_1..t)``, so labels are causal and pass truncation tests.
- **State identification.** EM label order is arbitrary and changes across
  walk-forward refits; states are relabeled by ascending variance, so regime 0
  is always the calmest and regime ``K-1`` the most turbulent.

Requires the ``models`` extra (``pip install -e ".[models]"``); the dependency
is imported lazily so the core package works without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimelab.regimes.base import RegimeModel, single_price_column

_LOG_EPS = 1e-300  # additive floor before log() to tolerate exact-zero probabilities


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf slices stay -inf without NaN
    out = np.log(np.sum(np.exp(a - m), axis=axis)) + np.squeeze(m, axis=axis)
    return out


class GaussianHMMRegime(RegimeModel):
    """K-state Gaussian HMM on log returns with causal (filtered) prediction.

    Parameters
    ----------
    n_regimes : number of hidden states (2-3 recommended for interpretability).
    n_restarts : EM restarts with seeds ``seed .. seed+n_restarts-1``; EM is
        sensitive to initialization, so single-run fits are not trusted.
    max_iter : EM iteration cap per restart.
    seed : base random seed; fitting is fully deterministic given it.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        n_restarts: int = 5,
        max_iter: int = 200,
        seed: int = 0,
    ):
        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2.")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1.")
        self._n_regimes = n_regimes
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.seed = seed
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.fit_info_: dict | None = None

    @property
    def n_regimes(self) -> int:
        return self._n_regimes

    def _log_returns(self, X: pd.DataFrame) -> pd.Series:
        prices = single_price_column(X)
        return np.log(prices).diff()

    def fit(self, X: pd.DataFrame) -> "GaussianHMMRegime":
        from hmmlearn.hmm import GaussianHMM  # models extra; lazy on purpose

        r = self._log_returns(X).dropna()
        if len(r) < 50 * self._n_regimes:
            raise ValueError(
                f"Training sample too short: {len(r)} returns for "
                f"{self._n_regimes} regimes (need >= {50 * self._n_regimes})."
            )
        obs = r.to_numpy().reshape(-1, 1)

        best, best_score, best_restart = None, -np.inf, -1
        for i in range(self.n_restarts):
            hmm = GaussianHMM(
                n_components=self._n_regimes,
                covariance_type="diag",
                n_iter=self.max_iter,
                random_state=self.seed + i,
            )
            hmm.fit(obs)
            score = hmm.score(obs)
            if score > best_score:
                best, best_score, best_restart = hmm, score, i

        variances = np.array([np.atleast_2d(c)[0, 0] for c in best.covars_])
        order = np.argsort(variances)  # regime 0 = calmest, K-1 = most turbulent
        self.means_ = best.means_.ravel()[order]
        self.vars_ = variances[order]
        self.startprob_ = best.startprob_[order]
        self.transmat_ = best.transmat_[np.ix_(order, order)]
        self.fit_info_ = {
            "log_likelihood": float(best_score),
            "best_restart": best_restart,
            "converged": bool(best.monitor_.converged),
            "n_obs": len(r),
        }
        return self

    def _check_fitted(self) -> None:
        if self.transmat_ is None:
            raise RuntimeError("Call fit() before predicting.")

    def filtered_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """``P(state_t | r_1..t)`` per date — the causal forward recursion.

        Rows are NaN during warm-up (the first price has no return). Columns
        are regime ids 0..K-1 ordered by ascending variance.
        """
        self._check_fitted()
        r = self._log_returns(X)
        valid = r.dropna()
        obs = valid.to_numpy()

        log_emission = -0.5 * (
            np.log(2.0 * np.pi * self.vars_)
            + (obs[:, None] - self.means_) ** 2 / self.vars_
        )  # (T, K)
        log_trans = np.log(self.transmat_ + _LOG_EPS)

        filtered = np.empty_like(log_emission)
        alpha = np.log(self.startprob_ + _LOG_EPS) + log_emission[0]
        alpha -= _logsumexp(alpha)
        filtered[0] = np.exp(alpha)
        for t in range(1, len(obs)):
            alpha = log_emission[t] + _logsumexp(alpha[:, None] + log_trans, axis=0)
            alpha -= _logsumexp(alpha)
            filtered[t] = np.exp(alpha)

        probs = pd.DataFrame(filtered, index=valid.index, columns=range(self._n_regimes))
        return probs.reindex(X.index)

    def predict_regimes(self, X: pd.DataFrame) -> pd.Series:
        """Most likely current regime under the filtered distribution."""
        probs = self.filtered_probabilities(X)
        valid = probs.dropna()
        labels = pd.Series(
            valid.to_numpy().argmax(axis=1).astype(float), index=valid.index
        )
        return labels.reindex(X.index).rename("regime")

    def expected_durations(self) -> pd.Series:
        """Expected regime durations in periods, ``1 / (1 - p_stay)`` — a
        persistence diagnostic: near-1 durations mean the model found noise,
        not regimes."""
        self._check_fitted()
        stay = np.diag(self.transmat_)
        return pd.Series(1.0 / (1.0 - stay), index=range(self._n_regimes), name="duration")
