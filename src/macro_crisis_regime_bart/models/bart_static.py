"""Static monotone probit BART-like model for phase-1 Bayesian layer."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import BaseCrisisModel
from .samplers.truncated_normal import sample_probit_latent_z
from .samplers.monotone_tree_engine import MonotoneStumpEnsemble


@dataclass
class StaticBARTDiagnostics:
    acceptance_rate: float
    n_saved_draws: int


class StaticProbitMonotoneBART(BaseCrisisModel):
    """Bayesian probit model with monotone additive stump ensemble.

    This implementation uses Albert-Chib latent z augmentation and stochastic
    backfitting updates over monotone stumps with NA routing.
    """

    def __init__(
        self,
        n_trees: int = 30,
        n_mcmc: int = 120,
        burn_in: int = 40,
        thin: int = 4,
        seed: int = 42,
        monotonicity: list[int] | None = None,
        missing: str = "native",
    ) -> None:
        self.n_trees = n_trees
        self.n_mcmc = n_mcmc
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.monotonicity = monotonicity
        self.missing = missing

        self.rng = np.random.default_rng(seed)
        self.ensemble: MonotoneStumpEnsemble | None = None
        self.draw_rules: list[list] = []
        self.feature_names_: list[str] | None = None
        self.feature_medians_: np.ndarray | None = None

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is None:
                self.feature_names_ = list(X.columns)
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _handle_missing(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.missing == "native":
            return X
        if self.missing == "raise":
            if np.isnan(X).any():
                raise ValueError("Missing values found with missing='raise'")
            return X
        if self.missing == "median":
            if fit:
                self.feature_medians_ = np.nanmedian(X, axis=0)
            if self.feature_medians_ is None:
                raise ValueError("Median statistics unavailable. Fit model first.")
            out = X.copy()
            mask = np.isnan(out)
            out[mask] = np.take(self.feature_medians_, np.where(mask)[1])
            return out
        raise ValueError(f"Unknown missing policy: {self.missing}")

    def fit(self, X, y) -> "StaticProbitMonotoneBART":
        Xv = self._handle_missing(self._to_array(X), fit=True)
        yv = np.asarray(y).astype(int)
        p = Xv.shape[1]
        mono = self.monotonicity or [0] * p
        self.ensemble = MonotoneStumpEnsemble(n_trees=self.n_trees, monotonicity=mono, rng=self.rng)

        eta = np.zeros(len(yv), dtype=float)
        self.draw_rules = []
        for it in range(self.n_mcmc):
            z = sample_probit_latent_z(yv, eta, self.rng)
            self.ensemble.backfit_step(Xv, z)
            eta = self.ensemble.predict(Xv)

            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                self.draw_rules.append(deepcopy(self.ensemble.rules))
        return self

    def _predict_from_rules(self, Xv: np.ndarray, rules: list) -> np.ndarray:
        assert self.ensemble is not None
        out = np.zeros(Xv.shape[0], dtype=float)
        for rule in rules:
            out += self.ensemble.predict_tree(Xv, rule)
        return norm.cdf(out)

    def predict_proba_samples(self, X) -> np.ndarray | None:
        if not self.draw_rules:
            return None
        Xv = self._handle_missing(self._to_array(X), fit=False)
        draws = [self._predict_from_rules(Xv, rules) for rules in self.draw_rules]
        return np.vstack(draws)

    def predict_proba(self, X) -> np.ndarray:
        samples = self.predict_proba_samples(X)
        if samples is None:
            raise ValueError("Model has no retained posterior draws. Did you call fit?")
        return samples.mean(axis=0)

    def posterior_summary(self, X, quantiles: tuple[float, float] = (0.1, 0.9)) -> pd.DataFrame:
        samples = self.predict_proba_samples(X)
        if samples is None:
            raise ValueError("Model has no retained posterior draws")
        ql, qh = quantiles
        return pd.DataFrame(
            {
                "mean": samples.mean(axis=0),
                "std": samples.std(axis=0),
                f"q{int(100*ql)}": np.quantile(samples, ql, axis=0),
                f"q{int(100*qh)}": np.quantile(samples, qh, axis=0),
            }
        )

    def diagnostics_summary(self) -> StaticBARTDiagnostics:
        if self.ensemble is None:
            raise ValueError("Model not fitted")
        return StaticBARTDiagnostics(
            acceptance_rate=self.ensemble.acceptance_rate(),
            n_saved_draws=len(self.draw_rules),
        )

    def benchmark_fit_predict(self, X_train, y_train, X_test) -> np.ndarray:
        self.fit(X_train, y_train)
        return self.predict_proba(X_test)


# Backward-compatible alias used by configs/factory names
StaticPMBARTModel = StaticProbitMonotoneBART
