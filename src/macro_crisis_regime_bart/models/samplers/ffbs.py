"""Forward-filtering backward-sampling for latent discrete regimes."""

from __future__ import annotations

import numpy as np


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is None:
        return out.squeeze()
    return np.squeeze(out, axis=axis)


def ffbs_sample(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    init_probs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous-transition FFBS for backwards compatibility."""
    T, K = log_emissions.shape
    logA = np.log(np.clip(transition, 1e-12, 1.0))
    logpi = np.log(np.clip(init_probs, 1e-12, 1.0))

    log_alpha = np.zeros((T, K), dtype=float)
    filt = np.zeros((T, K), dtype=float)

    log_alpha[0] = logpi + log_emissions[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])
    filt[0] = np.exp(log_alpha[0])

    for t in range(1, T):
        pred = _logsumexp(log_alpha[t - 1][:, None] + logA, axis=0)
        log_alpha[t] = log_emissions[t] + pred
        log_alpha[t] -= _logsumexp(log_alpha[t])
        filt[t] = np.exp(log_alpha[t])

    s = np.zeros(T, dtype=int)
    s[T - 1] = rng.choice(K, p=filt[T - 1])

    for t in range(T - 2, -1, -1):
        logw = log_alpha[t] + logA[:, s[t + 1]]
        logw -= _logsumexp(logw)
        s[t] = rng.choice(K, p=np.exp(logw))

    return s, filt


def ffbs_sample_tvtp(
    log_emissions: np.ndarray,
    log_trans_t: np.ndarray,
    init_probs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """FFBS with time-varying transitions in log space.

    Parameters
    ----------
    log_emissions: (T,K)
    log_trans_t: (T,K,K), where log_trans_t[t,j,k]=log P(s_t=k|s_{t-1}=j)
        row t=0 is ignored.
    """
    T, K = log_emissions.shape
    logpi = np.log(np.clip(init_probs, 1e-12, 1.0))
    log_alpha = np.zeros((T, K), dtype=float)
    filt = np.zeros((T, K), dtype=float)

    log_alpha[0] = logpi + log_emissions[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])
    filt[0] = np.exp(log_alpha[0])

    for t in range(1, T):
        pred = _logsumexp(log_alpha[t - 1][:, None] + log_trans_t[t], axis=0)
        log_alpha[t] = log_emissions[t] + pred
        log_alpha[t] -= _logsumexp(log_alpha[t])
        filt[t] = np.exp(log_alpha[t])

    s = np.zeros(T, dtype=int)
    s[T - 1] = rng.choice(K, p=filt[T - 1])
    for t in range(T - 2, -1, -1):
        logw = log_alpha[t] + log_trans_t[t + 1][:, s[t + 1]]
        logw -= _logsumexp(logw)
        s[t] = rng.choice(K, p=np.exp(logw))
    return s, filt


def aggregate_time_log_emissions(
    z: np.ndarray,
    mu_by_regime: np.ndarray,
    time_index: np.ndarray,
    sigma2: float = 1.0,
) -> np.ndarray:
    """Aggregate observation-level Gaussian log-likelihoods into (T,K) emissions."""
    t_vals = np.unique(time_index)
    K = mu_by_regime.shape[1]
    out = np.zeros((len(t_vals), K), dtype=float)
    c = -0.5 * np.log(2 * np.pi * sigma2)
    for ti, t in enumerate(t_vals):
        m = time_index == t
        for k in range(K):
            r = z[m] - mu_by_regime[m, k]
            out[ti, k] = np.sum(c - 0.5 * (r * r) / sigma2)
    return out


def tvtp_transition_log_probs(beta: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Build (T,2,2) log transition tensor for K=2 logistic TVTP.

    beta shape: (2, p+1), one row for prev-state j in {0,1}.
    """
    W = np.asarray(W, dtype=float)
    X = np.column_stack([np.ones(W.shape[0]), W])
    logits = X @ beta.T  # (T,2)
    p2 = 1.0 / (1.0 + np.exp(-logits))
    out = np.zeros((W.shape[0], 2, 2), dtype=float)
    out[:, 0, 1] = np.log(np.clip(p2[:, 0], 1e-12, 1.0))
    out[:, 0, 0] = np.log(np.clip(1 - p2[:, 0], 1e-12, 1.0))
    out[:, 1, 1] = np.log(np.clip(p2[:, 1], 1e-12, 1.0))
    out[:, 1, 0] = np.log(np.clip(1 - p2[:, 1], 1e-12, 1.0))
    return out
