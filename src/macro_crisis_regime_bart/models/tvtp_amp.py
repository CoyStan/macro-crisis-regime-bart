"""Final paper model: sticky TVTP regime-switching monotone probit BART with amplification."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from .base import BaseCrisisModel
from .bart_static import StaticProbitMonotoneBART
from .samplers.truncated_normal import sample_probit_latent_z
from .samplers.ffbs import aggregate_time_log_emissions, ffbs_sample_tvtp, tvtp_transition_log_probs
from .samplers.pg import logistic_pg_gaussian_update
from .samplers.slice_sampler import slice_sample_positive


@dataclass
class TVTPDiagnostics:
    runtime_sec: float
    n_saved_draws: int
    avg_stress_occupancy: float
    empty_regime_frequency: float
    eta2_mean: float
    lambda2_mean: float


class TVTPAmplifiedRegimeSwitchingProbitMonotoneBART(BaseCrisisModel):
    """K=2 TVTP amplification model.

    z_it = alpha_i + delta_{s_t} + lambda_{s_t} f0(x_it) + eps_it
    with lambda_1=1, lambda_2=exp(eta_2), eta_2>0 and logistic TVTP transitions.
    """

    def __init__(
        self,
        n_trees: int = 25,
        seed: int = 42,
        monotonicity: list[int] | None = None,
        missing: str = "native",
        n0_warmup: int = 20,
        n1_warmup: int = 30,
        n_burn: int = 60,
        n_draws: int = 40,
        thin: int = 2,
        n_bart_sweeps_per_iter: int = 1,
        n_bart_sweeps_during_warmup: int = 2,
        eta_prior_mu: float = 0.0,
        eta_prior_sd: float = 0.5,
        tvtp_prior_sd: float = 1.5,
        pg_trunc: int = 120,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n0_warmup = n0_warmup
        self.n1_warmup = n1_warmup
        self.n_burn = n_burn
        self.n_draws = n_draws
        self.thin = thin
        self.n_bart_sweeps_per_iter = n_bart_sweeps_per_iter
        self.n_bart_sweeps_during_warmup = n_bart_sweeps_during_warmup
        self.eta_prior_mu = eta_prior_mu
        self.eta_prior_sd = eta_prior_sd
        self.tvtp_prior_sd = tvtp_prior_sd
        self.pg_trunc = pg_trunc

        self.f0_model = StaticProbitMonotoneBART(
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
        self.time_order_: np.ndarray | None = None
        self.W_time_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

        self.saved: dict[str, list] = {k: [] for k in ["alpha", "delta", "eta2", "lambda2", "beta", "s", "f_rules"]}
        self.runtime_sec_: float = 0.0
        self.regime_init_source_: str = "global_covariates_only"

    def _map_ids(self, country_ids, time_ids, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(country_ids)
        t = np.asarray(time_ids).astype(int)
        if fit:
            cvals = list(pd.Index(c).unique())
            tvals = np.sort(pd.Index(t).unique())
            self.country_to_idx = {v: i for i, v in enumerate(cvals)}
            self.time_to_idx = {int(v): i for i, v in enumerate(tvals)}
            self.time_order_ = tvals
        ci = np.array([self.country_to_idx.get(v, -1) for v in c], dtype=int)
        ti = np.array([self.time_to_idx.get(int(v), -1) for v in t], dtype=int)
        return ci, ti

    def _prepare_W(self, W_time, fit: bool) -> np.ndarray:
        if W_time is None:
            raise ValueError("W_time is required for TVTP model")
        if isinstance(W_time, pd.DataFrame):
            if "time_id" not in W_time.columns:
                raise ValueError("W_time DataFrame must include time_id")
            wcols = [c for c in W_time.columns if c != "time_id"]
            Wdf = W_time[["time_id", *wcols]].drop_duplicates("time_id").sort_values("time_id")
            tids = Wdf["time_id"].to_numpy().astype(int)
            W = Wdf[wcols].to_numpy(dtype=float)
        else:
            W = np.asarray(W_time, dtype=float)
            if self.time_order_ is None:
                raise ValueError("time mapping unavailable")
            tids = self.time_order_.astype(int)
            if W.shape[0] != len(tids):
                raise ValueError("W_time rows must align with unique sorted time periods")

        if fit:
            if self.time_order_ is None:
                raise ValueError("time mapping unavailable")
            if not np.array_equal(tids, self.time_order_.astype(int)):
                raise ValueError("W_time time_id must match sorted unique time_ids in training data")
            self.W_time_ = W.copy()
            return W

        if self.time_order_ is None:
            raise ValueError("Model not fitted")
        map_w = {int(t): W[i] for i, t in enumerate(tids)}
        out = np.vstack([map_w[int(t)] for t in self.time_order_])
        return out

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _init_regime_from_W(self, W: np.ndarray) -> np.ndarray:
        stress_score = W[:, 0] if W.shape[1] > 0 else np.zeros(W.shape[0])
        thr = np.median(stress_score)
        s = (stress_score > thr).astype(int)
        return s

    def _recenter_f0(self, f0: np.ndarray, delta: np.ndarray, lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply sample-centering identification for baseline f0.

        f0 <- f0 - m; delta_s <- delta_s + lambda_s * m.
        """
        m = float(np.mean(f0))
        return f0 - m, delta + lambdas * m, m

    def _eta2_logpost(self, eta2: float, z: np.ndarray, alpha_obs: np.ndarray, delta2: float, f0: np.ndarray, mask2: np.ndarray) -> float:
        if eta2 <= 0:
            return -np.inf
        lam2 = np.exp(eta2)
        r = z[mask2] - alpha_obs[mask2] - delta2 - lam2 * f0[mask2]
        ll = -0.5 * np.sum(r * r)
        lp = -0.5 * ((eta2 - self.eta_prior_mu) / self.eta_prior_sd) ** 2
        return float(ll + lp)

    def fit(self, X, y, country_ids, time_ids, W_time=None):
        t0 = time.time()
        Xv = self.f0_model._handle_missing(self.f0_model._to_array(X), fit=True)
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        yv = np.asarray(y).astype(int)
        ci, ti = self._map_ids(country_ids, time_ids, fit=True)
        W = self._prepare_W(W_time, fit=True)

        N = len(yv)
        C = len(self.country_to_idx)
        T = len(self.time_to_idx)
        if (ti < 0).any():
            raise ValueError("All observations must map to valid time index")

        from .samplers.monotone_tree_engine import MonotoneStumpEnsemble

        p = Xv.shape[1]
        mono = self.f0_model.monotonicity or [0] * p
        self.f0_model.ensemble = MonotoneStumpEnsemble(self.f0_model.n_trees, mono, rng=self.rng)

        # initialization
        z = np.where(yv == 1, 0.5, -0.5).astype(float)
        s = self._init_regime_from_W(W)  # uses only W_t, not y
        alpha = np.zeros(C)
        delta = np.array([-0.1, 0.1], dtype=float)
        eta2 = 1e-3
        beta = np.zeros((2, W.shape[1] + 1), dtype=float)

        self.saved = {k: [] for k in ["alpha", "delta", "eta2", "lambda2", "beta", "s", "f_rules"]}

        total_iters = self.n0_warmup + self.n1_warmup + self.n_burn + self.n_draws * self.thin
        retain_start = self.n0_warmup + self.n1_warmup + self.n_burn

        for it in range(total_iters):
            phaseA = it < self.n0_warmup
            phaseB = self.n0_warmup <= it < self.n0_warmup + self.n1_warmup
            warmup = phaseA or phaseB or (it < retain_start)

            lambdas = np.array([1.0, np.exp(eta2)])
            lambda_obs = lambdas[s[ti]]
            eta = alpha[ci] + delta[s[ti]] + lambda_obs * self.f0_model.ensemble.predict(Xv)
            z = sample_probit_latent_z(yv, eta, self.rng)

            # alpha update with sum-to-zero normalization
            f0_cur = self.f0_model.ensemble.predict(Xv)
            for cidx in range(C):
                m = ci == cidx
                resid = z[m] - delta[s[ti[m]]] - lambdas[s[ti[m]]] * f0_cur[m]
                v = 1.0 / (m.sum() + 0.5)
                mu = v * resid.sum()
                alpha[cidx] = self.rng.normal(mu, np.sqrt(v))
            alpha -= alpha.mean()

            # delta update under ordering constraint
            for k in range(2):
                m = s[ti] == k
                resid = z[m] - alpha[ci[m]] - lambdas[k] * f0_cur[m]
                v = 1.0 / (m.sum() + 0.5)
                mu = v * resid.sum()
                delta[k] = self.rng.normal(mu, np.sqrt(v))
            delta = np.sort(delta)

            # f0 update with multiple sweeps (no response scaling by lambda)
            sweeps = self.n_bart_sweeps_during_warmup if warmup else self.n_bart_sweeps_per_iter
            for _ in range(sweeps):
                target_f = z - alpha[ci] - delta[s[ti]]
                self.f0_model.ensemble.backfit_step(Xv, target_f)

            f0 = self.f0_model.ensemble.predict(Xv)
            f0, delta, _ = self._recenter_f0(f0, delta, lambdas)

            # eta2/lambda2 update only in full chain (phase C)
            if not phaseA and not phaseB:
                mask2 = s[ti] == 1
                if mask2.any():
                    alpha_obs = alpha[ci]
                    eta2 = slice_sample_positive(
                        max(eta2, 1e-4),
                        logpdf=lambda e: self._eta2_logpost(e, z, alpha_obs, delta[1], f0, mask2),
                        rng=self.rng,
                        w=0.2,
                    )

            lambdas = np.array([1.0, np.exp(eta2)])

            # TVTP beta update / regime updates
            if not phaseA:
                Xw = np.column_stack([np.ones(W.shape[0]), W])
                prior_prec = np.eye(Xw.shape[1]) / (self.tvtp_prior_sd**2)
                for j in range(2):
                    yj = (s[1:] == 1).astype(float)
                    Xj = Xw[1:]
                    beta[j], _ = logistic_pg_gaussian_update(
                        Xj,
                        yj,
                        beta_prior_mean=np.zeros(Xw.shape[1]),
                        beta_prior_prec=prior_prec,
                        rng=self.rng,
                        omega_trunc=self.pg_trunc,
                    )

                mu = np.zeros((N, 2), dtype=float)
                for k in range(2):
                    mu[:, k] = alpha[ci] + delta[k] + lambdas[k] * f0
                log_em = aggregate_time_log_emissions(z, mu, ti)
                log_tr = tvtp_transition_log_probs(beta, W)
                s, _ = ffbs_sample_tvtp(log_em, log_tr, np.array([0.7, 0.3]), self.rng)

            if it >= retain_start and ((it - retain_start) % self.thin == 0):
                self.saved["alpha"].append(alpha.copy())
                self.saved["delta"].append(delta.copy())
                self.saved["eta2"].append(float(eta2))
                self.saved["lambda2"].append(float(np.exp(eta2)))
                self.saved["beta"].append(beta.copy())
                self.saved["s"].append(s.copy())
                self.saved["f_rules"].append(deepcopy(self.f0_model.ensemble.rules))

        self.f0_model.draw_rules = self.saved["f_rules"]
        self.runtime_sec_ = time.time() - t0
        return self

    def _predict_draw(self, Xv: np.ndarray, ci: np.ndarray, ti: np.ndarray, W_time: np.ndarray, d: int) -> np.ndarray:
        alpha = self.saved["alpha"][d]
        delta = self.saved["delta"][d]
        lam2 = self.saved["lambda2"][d]
        beta = self.saved["beta"][d]
        s = self.saved["s"][d]

        fprob = self.f0_model._predict_from_rules(Xv, self.saved["f_rules"][d])
        f0 = norm.ppf(np.clip(fprob, 1e-6, 1 - 1e-6))

        a = np.where(ci >= 0, alpha[ci], 0.0)
        st = np.zeros(len(ci), dtype=int)
        known = ti >= 0
        st[known] = s[ti[known]]

        if (~known).any():
            # fallback for unseen times: mixture by TVTP probabilities at mapped W_time rows
            Xw = np.column_stack([np.ones(W_time.shape[0]), W_time])
            p2 = self._sigmoid(Xw @ beta.T).mean(axis=1)
            t_unknown = np.where(~known)[0]
            st[t_unknown] = (p2[np.clip(np.zeros(len(t_unknown), dtype=int), 0, len(p2)-1)] > 0.5).astype(int)

        lam = np.where(st == 0, 1.0, lam2)
        latent = a + delta[st] + lam * f0
        return norm.cdf(latent)

    def predict_proba_samples(self, X, country_ids, time_ids, W_time=None):
        if not self.saved["alpha"]:
            return None
        Xv = self.f0_model._handle_missing(self.f0_model._to_array(X), fit=False)
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        W = self._prepare_W(W_time if W_time is not None else pd.DataFrame({"time_id": self.time_order_, **{f"w{i}": self.W_time_[:, i] for i in range(self.W_time_.shape[1])}}), fit=False)
        draws = [self._predict_draw(Xv, ci, ti, W, d) for d in range(len(self.saved["alpha"]))]
        return np.vstack(draws)

    def predict_proba(self, X, country_ids, time_ids, W_time=None):
        return self.predict_proba_samples(X, country_ids, time_ids, W_time=W_time).mean(axis=0)

    def posterior_summary(self) -> dict:
        eta = np.array(self.saved["eta2"]) if self.saved["eta2"] else np.array([np.nan])
        lam = np.array(self.saved["lambda2"]) if self.saved["lambda2"] else np.array([np.nan])
        return {
            "eta2_mean": float(np.nanmean(eta)),
            "eta2_q10": float(np.nanquantile(eta, 0.1)),
            "eta2_q90": float(np.nanquantile(eta, 0.9)),
            "lambda2_mean": float(np.nanmean(lam)),
            "lambda2_q10": float(np.nanquantile(lam, 0.1)),
            "lambda2_q90": float(np.nanquantile(lam, 0.9)),
        }

    def regime_posterior_summary(self) -> pd.DataFrame:
        sdraw = np.array(self.saved["s"])
        probs = np.zeros((sdraw.shape[1], 2))
        for t in range(sdraw.shape[1]):
            probs[t] = np.bincount(sdraw[:, t], minlength=2) / sdraw.shape[0]
        beta_mean = np.mean(np.array(self.saved["beta"]), axis=0)
        tr = np.exp(tvtp_transition_log_probs(beta_mean, self.W_time_))
        return pd.DataFrame(
            {
                "time_id": self.time_order_,
                "p_regime_1": probs[:, 0],
                "p_regime_2": probs[:, 1],
                "p_enter_stress": tr[:, 0, 1],
                "p_stay_stress": tr[:, 1, 1],
            }
        )

    def transition_risk_summary(self) -> pd.DataFrame:
        rp = self.regime_posterior_summary().copy()
        rp["p_stress_t"] = rp["p_regime_2"]
        return rp[["time_id", "p_stress_t", "p_enter_stress", "p_stay_stress"]]

    def component_summary_df(self, X, country_ids, time_ids, meta: pd.DataFrame | None = None) -> pd.DataFrame:
        Xv = self.f0_model._handle_missing(self.f0_model._to_array(X), fit=False)
        ci, ti = self._map_ids(country_ids, time_ids, fit=False)
        rows = []
        for d in range(len(self.saved["alpha"])):
            alpha = self.saved["alpha"][d]
            delta = self.saved["delta"][d]
            lam2 = self.saved["lambda2"][d]
            s = self.saved["s"][d]
            fprob = self.f0_model._predict_from_rules(Xv, self.saved["f_rules"][d])
            f0 = norm.ppf(np.clip(fprob, 1e-6, 1 - 1e-6))
            st = np.where(ti >= 0, s[ti], 0)
            a = np.where(ci >= 0, alpha[ci], 0.0)
            lam = np.where(st == 0, 1.0, lam2)
            lamf = lam * f0
            latent = a + delta[st] + lamf
            rows.append(np.column_stack([a, delta[st], f0, lam, lamf, latent, norm.cdf(latent), st]))
        arr = np.stack(rows)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        sdraw = np.array(self.saved["s"])
        rp = np.zeros((len(ti), 2))
        for i in range(len(ti)):
            if ti[i] >= 0:
                bc = np.bincount(sdraw[:, ti[i]], minlength=2) / sdraw.shape[0]
            else:
                bc = np.array([0.5, 0.5])
            rp[i] = bc

        out = pd.DataFrame(
            {
                "time_id": np.asarray(time_ids).astype(int),
                "alpha_mean": mean[:, 0],
                "delta_mean": mean[:, 1],
                "f0_mean": mean[:, 2],
                "lambda_active_mean": mean[:, 3],
                "lambda_f0_mean": mean[:, 4],
                "latent_total_mean": mean[:, 5],
                "latent_total_std": std[:, 5],
                "prob_mean": mean[:, 6],
                "prob_std": std[:, 6],
                "regime_prob_1": rp[:, 0],
                "regime_prob_2": rp[:, 1],
            }
        )
        if meta is not None:
            for c in ["country_id", "year", "month", "time_id"]:
                if c in meta.columns and c not in out.columns:
                    out[c] = meta[c].values
                elif c in meta.columns and c in out.columns:
                    out[c] = meta[c].values
        return out

    def diagnostics_summary(self) -> TVTPDiagnostics:
        sdraw = np.array(self.saved["s"])
        occ = (sdraw == 1).mean() if sdraw.size else np.nan
        empty = np.mean([(np.bincount(s, minlength=2) == 0).any() for s in sdraw]) if sdraw.size else np.nan
        eta = np.array(self.saved["eta2"]) if self.saved["eta2"] else np.array([np.nan])
        lam = np.array(self.saved["lambda2"]) if self.saved["lambda2"] else np.array([np.nan])
        return TVTPDiagnostics(
            runtime_sec=float(self.runtime_sec_),
            n_saved_draws=len(self.saved["s"]),
            avg_stress_occupancy=float(occ),
            empty_regime_frequency=float(empty),
            eta2_mean=float(np.nanmean(eta)),
            lambda2_mean=float(np.nanmean(lam)),
        )

    def plot_regime_diagnostics(self, output_path: str) -> None:
        rp = self.regime_posterior_summary()
        plt.figure(figsize=(9, 3))
        plt.plot(rp["time_id"], rp["p_regime_2"], label="P(stress)")
        plt.plot(rp["time_id"], rp["p_enter_stress"], label="P(enter stress)")
        plt.plot(rp["time_id"], rp["p_stay_stress"], label="P(stay stress)")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_country_component_timeseries(self, component_df: pd.DataFrame, output_path: str, country_id: str | None = None, col: str = "prob_mean") -> None:
        df = component_df.copy()
        if country_id is not None and "country_id" in df.columns:
            df = df[df["country_id"] == country_id]
        plt.figure(figsize=(9, 3))
        plt.plot(df["time_id"], df[col])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def feature_component_dependence_df(self, X: pd.DataFrame, feature: str, country_ids, time_ids, W_time=None, grid_points: int = 20) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be pandas DataFrame")
        grid = np.linspace(X[feature].quantile(0.05), X[feature].quantile(0.95), grid_points)
        rows = []
        Xb = X.copy()
        for v in grid:
            Xb[feature] = v
            p = self.predict_proba(Xb, country_ids=country_ids, time_ids=time_ids, W_time=W_time)
            rows.append({"feature": feature, "value": float(v), "prob_mean": float(np.mean(p))})
        return pd.DataFrame(rows)

    def plot_feature_component_dependence(self, dep_df: pd.DataFrame, output_path: str, y_col: str = "prob_mean") -> None:
        plt.figure(figsize=(6, 3))
        plt.plot(dep_df["value"], dep_df[y_col])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


TVTPAmplificationModel = TVTPAmplifiedRegimeSwitchingProbitMonotoneBART
