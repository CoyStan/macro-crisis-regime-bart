"""RS-2: regime-switching probit BART with baseline and regime-specific deviation ensembles."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from .base import BaseCrisisModel
from .bart_static import StaticProbitMonotoneBART
from .samplers.truncated_normal import sample_probit_latent_z
from .samplers.ffbs import ffbs_sample, aggregate_time_log_emissions
from .samplers.monotone_tree_engine import MonotoneStumpEnsemble


class RegimeSwitchingProbitMonotoneBART(BaseCrisisModel):
    def __init__(
        self,
        n_regimes: int = 2,
        n_trees_baseline: int = 20,
        n_trees_deviation: int = 8,
        deviation_scale: float = 0.5,
        n_mcmc: int = 120,
        burn_in: int = 40,
        thin: int = 4,
        seed: int = 42,
        monotonicity_baseline: list[int] | None = None,
        monotonicity_deviation: list[int] | None = None,
        missing: str = "native",
    ) -> None:
        self.n_regimes = n_regimes
        self.n_mcmc = n_mcmc
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.deviation_scale = deviation_scale
        self.rng = np.random.default_rng(seed)

        self.f0 = StaticProbitMonotoneBART(
            n_trees=n_trees_baseline,
            n_mcmc=1,
            burn_in=0,
            thin=1,
            seed=seed,
            monotonicity=monotonicity_baseline,
            missing=missing,
        )
        self.g_ensembles: list[MonotoneStumpEnsemble] = []
        self.n_trees_deviation = n_trees_deviation
        self.monotonicity_deviation = monotonicity_deviation

        self.country_to_idx: dict[str, int] = {}
        self.time_to_idx: dict[int, int] = {}
        self.saved: dict[str, list] = {k: [] for k in ["alpha", "delta", "A", "s", "f_rules", "g_rules"]}

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

        p = Xv.shape[1]
        mono0 = self.f0.monotonicity or [0] * p
        self.f0.ensemble = MonotoneStumpEnsemble(self.f0.n_trees, mono0, rng=self.rng)
        monog = self.monotonicity_deviation or [0] * p
        self.g_ensembles = [MonotoneStumpEnsemble(self.n_trees_deviation, monog, tau_leaf=0.1, rng=self.rng) for _ in range(K)]

        alpha = np.zeros(C)
        delta = np.linspace(-0.3, 0.3, K)
        s = np.zeros(T, dtype=int)
        A = np.full((K, K), 1.0 / K)
        eta = np.zeros(N)

        self.saved = {k: [] for k in ["alpha", "delta", "A", "s", "f_rules", "g_rules"]}

        for it in range(self.n_mcmc):
            z = sample_probit_latent_z(yv, eta, self.rng)

            f0 = self.f0.ensemble.predict(Xv)
            gk = np.column_stack([g.predict(Xv) for g in self.g_ensembles])
            g_active = gk[np.arange(N), s[ti]]

            resid_alpha = z - delta[s[ti]] - f0 - g_active
            for cidx in range(C):
                m = ci == cidx
                n = m.sum()
                v = 1.0 / (n + 0.25)
                mu = v * resid_alpha[m].sum()
                alpha[cidx] = self.rng.normal(mu, np.sqrt(v))

            resid_delta = z - alpha[ci] - f0 - g_active
            for k in range(K):
                m = s[ti] == k
                n = m.sum()
                v = 1.0 / (n + 0.25)
                mu = v * resid_delta[m].sum()
                delta[k] = self.rng.normal(mu, np.sqrt(v))

            order = np.argsort(delta)
            inv = np.zeros(K, dtype=int)
            inv[order] = np.arange(K)
            delta = delta[order]
            A = A[order][:, order]
            s = inv[s]
            self.g_ensembles = [self.g_ensembles[o] for o in order]

            target_f0 = z - alpha[ci] - delta[s[ti]] - gk[np.arange(N), s[ti]]
            self.f0.ensemble.backfit_step(Xv, target_f0)
            f0 = self.f0.ensemble.predict(Xv)

            for k in range(K):
                m = s[ti] == k
                target_g = np.zeros(N)
                target_g[m] = (z - alpha[ci] - delta[k] - f0)[m] * self.deviation_scale
                self.g_ensembles[k].backfit_step(Xv, target_g)
            gk = np.column_stack([g.predict(Xv) for g in self.g_ensembles])

            mu = np.zeros((N, K))
            for k in range(K):
                mu[:, k] = alpha[ci] + delta[k] + f0 + gk[:, k]
            log_emissions = aggregate_time_log_emissions(z, mu, ti)
            s, _ = ffbs_sample(log_emissions, A, np.full(K, 1.0 / K), self.rng)

            counts = np.zeros((K, K), dtype=float)
            for tt in range(1, T):
                counts[s[tt - 1], s[tt]] += 1.0
            for k in range(K):
                A[k] = self.rng.dirichlet(counts[k] + 5.0)

            eta = alpha[ci] + delta[s[ti]] + f0 + gk[np.arange(N), s[ti]]

            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                self.saved["alpha"].append(alpha.copy())
                self.saved["delta"].append(delta.copy())
                self.saved["A"].append(A.copy())
                self.saved["s"].append(s.copy())
                self.saved["f_rules"].append(deepcopy(self.f0.ensemble.rules))
                self.saved["g_rules"].append([deepcopy(g.rules) for g in self.g_ensembles])
        return self

    def _latent_components_draw(self, Xv: np.ndarray, ci: np.ndarray, ti: np.ndarray, d: int):
        alpha = self.saved["alpha"][d]
        delta = self.saved["delta"][d]
        s = self.saved["s"][d]
        fprob = self.f0._predict_from_rules(Xv, self.saved["f_rules"][d])
        f0 = norm.ppf(np.clip(fprob, 1e-6, 1 - 1e-6))

        gk = np.zeros((Xv.shape[0], self.n_regimes))
        for k in range(self.n_regimes):
            rules = self.saved["g_rules"][d][k]
            for rule in rules:
                gk[:, k] += self.g_ensembles[k].predict_tree(Xv, rule)

        a = np.where(ci >= 0, alpha[ci], alpha.mean())
        st = np.where(ti >= 0, s[ti], np.argmax(np.bincount(s)))
        dlt = delta[st]
        gactive = gk[np.arange(len(st)), st]
        latent = a + dlt + f0 + gactive
        return a, dlt, f0, gactive, latent

    def predict_proba_samples(self, X, country_ids=None, time_ids=None):
        if not self.saved["alpha"]:
            return None
        Xv = self.f0._handle_missing(self.f0._to_array(X), fit=False)
        if country_ids is None:
            country_ids = np.array([next(iter(self.country_to_idx))] * len(Xv))
        if time_ids is None:
            time_ids = np.array([next(iter(self.time_to_idx))] * len(Xv))
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        draws = []
        for d in range(len(self.saved["alpha"])):
            _, _, _, _, latent = self._latent_components_draw(Xv, ci, ti, d)
            draws.append(norm.cdf(latent))
        return np.vstack(draws)

    def predict_proba(self, X, country_ids=None, time_ids=None):
        return self.predict_proba_samples(X, country_ids=country_ids, time_ids=time_ids).mean(axis=0)

    def regime_posterior_summary(self) -> pd.DataFrame:
        s_draws = np.array(self.saved["s"])
        rows = []
        for t in range(s_draws.shape[1]):
            p = np.bincount(s_draws[:, t], minlength=self.n_regimes) / s_draws.shape[0]
            row = {"time_index": t}
            row.update({f"p_regime_{k}": p[k] for k in range(self.n_regimes)})
            rows.append(row)
        return pd.DataFrame(rows)

    def component_summary_df(self, X, country_ids, time_ids) -> pd.DataFrame:
        Xv = self.f0._handle_missing(self.f0._to_array(X), fit=False)
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        mats = []
        for d in range(len(self.saved["alpha"])):
            a, dlt, f0, g, latent = self._latent_components_draw(Xv, ci, ti, d)
            mats.append(np.column_stack([a, dlt, f0, g, latent, norm.cdf(latent)]))
        arr = np.stack(mats)
        mean = arr.mean(axis=0)
        return pd.DataFrame(mean, columns=["alpha", "delta", "f0", "g_active", "latent_total", "probability"])

    def diagnostics_summary(self) -> dict:
        return {
            "n_saved_draws": len(self.saved["alpha"]),
            "avg_transition": np.mean(np.array(self.saved["A"]), axis=0).tolist() if self.saved["A"] else None,
            "baseline_acceptance": self.f0.ensemble.acceptance_rate() if self.f0.ensemble else None,
            "deviation_acceptance": [g.acceptance_rate() for g in self.g_ensembles],
        }

    def plot_regime_diagnostics(self, output_path: str) -> None:
        rp = self.regime_posterior_summary()
        plt.figure(figsize=(8, 3))
        for k in range(self.n_regimes):
            plt.plot(rp["time_index"], rp[f"p_regime_{k}"], label=f"Regime {k}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_country_component_timeseries(self, component_df: pd.DataFrame, time_ids, output_path: str, column: str = "probability") -> None:
        plt.figure(figsize=(9, 3))
        plt.plot(np.asarray(time_ids), component_df[column].to_numpy())
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def feature_component_dependence_df(self, X: pd.DataFrame, feature: str, country_ids, time_ids, grid_points: int = 20) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame for dependence export")
        grid = np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), grid_points)
        rows = []
        Xb = X.copy()
        for v in grid:
            Xb[feature] = v
            comp = self.component_summary_df(Xb, country_ids=country_ids, time_ids=time_ids)
            rows.append({"feature": feature, "value": v, "probability": comp["probability"].mean(), "latent_total": comp["latent_total"].mean()})
        return pd.DataFrame(rows)

    def plot_feature_component_dependence(self, dep_df: pd.DataFrame, output_path: str, y_col: str = "probability") -> None:
        plt.figure(figsize=(6, 3))
        plt.plot(dep_df["value"], dep_df[y_col])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


RS2Model = RegimeSwitchingProbitMonotoneBART
