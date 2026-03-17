"""RS-1: latent global regime + country effects + regime intercept + baseline monotone BART."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import BaseCrisisModel
from .bart_static import StaticProbitMonotoneBART
from .samplers.truncated_normal import sample_probit_latent_z
from .samplers.ffbs import ffbs_sample, aggregate_time_log_emissions


@dataclass
class RS1Diagnostics:
    n_saved_draws: int
    avg_transition: np.ndarray


class RegimeSwitchingProbitMonotoneBARTPhase1(BaseCrisisModel):
    def __init__(
        self,
        n_regimes: int = 2,
        n_trees: int = 20,
        n_mcmc: int = 120,
        burn_in: int = 40,
        thin: int = 4,
        seed: int = 42,
        monotonicity: list[int] | None = None,
        missing: str = "native",
        alpha_prior_var: float = 4.0,
        delta_prior_var: float = 4.0,
        transition_prior: float = 5.0,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_mcmc = n_mcmc
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.alpha_prior_var = alpha_prior_var
        self.delta_prior_var = delta_prior_var
        self.transition_prior = transition_prior
        self.rng = np.random.default_rng(seed)

        self.f0 = StaticProbitMonotoneBART(
            n_trees=n_trees,
            n_mcmc=1,
            burn_in=0,
            thin=1,
            seed=seed,
            monotonicity=monotonicity,
            missing=missing,
        )
        self.country_to_idx: dict[str, int] = {}
        self.time_to_idx: dict[int, int] = {}
        self.saved: dict[str, list] = {k: [] for k in ["alpha", "delta", "A", "s", "f_rules"]}

    def _map_ids(self, country_ids, time_ids, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(country_ids)
        t = np.asarray(time_ids)
        if fit:
            cvals = list(pd.Index(c).unique())
            tvals = list(np.sort(pd.Index(t).unique()))
            self.country_to_idx = {v: i for i, v in enumerate(cvals)}
            self.time_to_idx = {int(v): i for i, v in enumerate(tvals)}
        ci = np.array([self.country_to_idx.get(v, -1) for v in c], dtype=int)
        ti = np.array([self.time_to_idx.get(int(v), -1) for v in t], dtype=int)
        return ci, ti

    def fit(self, X, y, country_ids, time_ids):
        Xv = self.f0._handle_missing(self.f0._to_array(X), fit=True)
        yv = np.asarray(y).astype(int)
        ci, ti = self._map_ids(country_ids, time_ids, fit=True)
        N, T = len(yv), len(self.time_to_idx)
        C, K = len(self.country_to_idx), self.n_regimes

        self.f0.ensemble = self.f0.ensemble or None
        if self.f0.ensemble is None:
            p = Xv.shape[1]
            mono = self.f0.monotonicity or [0] * p
            from .samplers.monotone_tree_engine import MonotoneStumpEnsemble

            self.f0.ensemble = MonotoneStumpEnsemble(self.f0.n_trees, mono, rng=self.rng)

        alpha = np.zeros(C)
        delta = np.linspace(-0.2, 0.2, K)
        s = np.zeros(T, dtype=int)
        A = np.full((K, K), 1.0 / K)
        eta = np.zeros(N)

        self.saved = {k: [] for k in ["alpha", "delta", "A", "s", "f_rules"]}

        for it in range(self.n_mcmc):
            z = sample_probit_latent_z(yv, eta, self.rng)

            # alpha | rest
            resid_alpha = z - delta[s[ti]] - self.f0.ensemble.predict(Xv)
            for cidx in range(C):
                m = ci == cidx
                n = m.sum()
                v = 1.0 / (n + 1.0 / self.alpha_prior_var)
                mu = v * resid_alpha[m].sum()
                alpha[cidx] = self.rng.normal(mu, np.sqrt(v))

            # delta | rest
            resid_delta = z - alpha[ci] - self.f0.ensemble.predict(Xv)
            for k in range(K):
                m = s[ti] == k
                n = m.sum()
                v = 1.0 / (n + 1.0 / self.delta_prior_var)
                mu = v * resid_delta[m].sum()
                delta[k] = self.rng.normal(mu, np.sqrt(v))

            # label ordering to mitigate switching
            order = np.argsort(delta)
            inv = np.zeros(K, dtype=int)
            inv[order] = np.arange(K)
            delta = delta[order]
            A = A[order][:, order]
            s = inv[s]

            # f0 | rest
            target_f = z - alpha[ci] - delta[s[ti]]
            self.f0.ensemble.backfit_step(Xv, target_f)
            f0 = self.f0.ensemble.predict(Xv)

            # s | rest via FFBS
            mu = np.zeros((N, K))
            for k in range(K):
                mu[:, k] = alpha[ci] + delta[k] + f0
            log_emissions = aggregate_time_log_emissions(z, mu, ti)
            s, _ = ffbs_sample(log_emissions, A, np.full(K, 1.0 / K), self.rng)

            # A | s
            counts = np.zeros((K, K), dtype=float)
            for tt in range(1, T):
                counts[s[tt - 1], s[tt]] += 1.0
            for k in range(K):
                A[k] = self.rng.dirichlet(counts[k] + self.transition_prior)

            eta = alpha[ci] + delta[s[ti]] + f0

            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                self.saved["alpha"].append(alpha.copy())
                self.saved["delta"].append(delta.copy())
                self.saved["A"].append(A.copy())
                self.saved["s"].append(s.copy())
                self.saved["f_rules"].append(deepcopy(self.f0.ensemble.rules))

        self.f0.draw_rules = self.saved["f_rules"]
        return self

    def _predict_draw(self, Xv: np.ndarray, ci: np.ndarray, ti: np.ndarray, d: int) -> np.ndarray:
        alpha = self.saved["alpha"][d]
        delta = self.saved["delta"][d]
        s = self.saved["s"][d]
        f = self.f0._predict_from_rules(Xv, self.saved["f_rules"][d])
        # convert probit prob back to latent for additive components
        latent_f = norm.ppf(np.clip(f, 1e-6, 1 - 1e-6))

        alpha_obs = np.where(ci >= 0, alpha[ci], alpha.mean())
        t_state = np.where(ti >= 0, s[ti], np.argmax(np.bincount(s)))
        latent = alpha_obs + delta[t_state] + latent_f
        return norm.cdf(latent)

    def predict_proba_samples(self, X, country_ids=None, time_ids=None):
        if not self.saved["alpha"]:
            return None
        Xv = self.f0._handle_missing(self.f0._to_array(X), fit=False)
        if country_ids is None:
            country_ids = np.array([next(iter(self.country_to_idx))] * len(Xv))
        if time_ids is None:
            time_ids = np.array([next(iter(self.time_to_idx))] * len(Xv))
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        draws = [self._predict_draw(Xv, ci, ti, d) for d in range(len(self.saved["alpha"]))]
        return np.vstack(draws)

    def predict_proba(self, X, country_ids=None, time_ids=None):
        samples = self.predict_proba_samples(X, country_ids=country_ids, time_ids=time_ids)
        return samples.mean(axis=0)

    def regime_posterior_summary(self) -> pd.DataFrame:
        s_draws = np.array(self.saved["s"])
        T = s_draws.shape[1]
        rows = []
        for t in range(T):
            p = np.bincount(s_draws[:, t], minlength=self.n_regimes) / s_draws.shape[0]
            row = {"time_index": t}
            row.update({f"p_regime_{k}": p[k] for k in range(self.n_regimes)})
            rows.append(row)
        return pd.DataFrame(rows)

    def component_summary_df(self, X, country_ids, time_ids) -> pd.DataFrame:
        Xv = self.f0._handle_missing(self.f0._to_array(X), fit=False)
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        rows = []
        for d in range(len(self.saved["alpha"])):
            alpha = self.saved["alpha"][d]
            delta = self.saved["delta"][d]
            s = self.saved["s"][d]
            fprob = self.f0._predict_from_rules(Xv, self.saved["f_rules"][d])
            flatent = norm.ppf(np.clip(fprob, 1e-6, 1 - 1e-6))
            a = np.where(ci >= 0, alpha[ci], alpha.mean())
            st = np.where(ti >= 0, s[ti], np.argmax(np.bincount(s)))
            dlt = delta[st]
            latent = a + dlt + flatent
            rows.append(pd.DataFrame({"alpha": a, "delta": dlt, "f0": flatent, "latent_total": latent, "probability": norm.cdf(latent)}))
        out = pd.concat(rows, keys=range(len(rows)), names=["draw", "row"]).reset_index(level=0)
        return out.groupby(out.index).mean(numeric_only=True)

    def diagnostics_summary(self) -> RS1Diagnostics:
        A = np.array(self.saved["A"])
        return RS1Diagnostics(n_saved_draws=len(A), avg_transition=A.mean(axis=0))


RS1Model = RegimeSwitchingProbitMonotoneBARTPhase1
