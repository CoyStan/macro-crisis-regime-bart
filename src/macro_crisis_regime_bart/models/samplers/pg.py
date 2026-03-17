"""Pólya-Gamma sampling utilities for binary logistic augmentation.

Implements a practical PG(1, c) sampler via truncated infinite-sum representation.
"""

from __future__ import annotations

import numpy as np


def sample_pg1_truncated(c: np.ndarray, rng: np.random.Generator, trunc: int = 200) -> np.ndarray:
    """Sample PG(1, c) random variables using truncated sum-of-gammas.

    PG(1,c) = 1/(2*pi^2) * sum_{n>=1} g_n / ((n-1/2)^2 + c^2/(4*pi^2)), g_n~Gamma(1,1)
    """
    c = np.asarray(c, dtype=float)
    shape = c.shape
    cflat = c.reshape(-1)
    n = np.arange(1, trunc + 1, dtype=float)
    base = (n - 0.5) ** 2
    out = np.zeros((cflat.shape[0],), dtype=float)
    for i, ci in enumerate(cflat):
        denom = base + (ci * ci) / (4.0 * np.pi * np.pi)
        g = rng.gamma(shape=1.0, scale=1.0, size=trunc)
        out[i] = np.sum(g / denom)
    out *= 1.0 / (2.0 * np.pi * np.pi)
    return out.reshape(shape)


def logistic_pg_gaussian_update(
    X: np.ndarray,
    y: np.ndarray,
    beta_prior_mean: np.ndarray,
    beta_prior_prec: np.ndarray,
    rng: np.random.Generator,
    omega_trunc: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """One Gibbs update for logistic regression coefficients via PG augmentation.

    y in {0,1}, p(y=1)=logistic(X beta).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    beta_prior_mean = np.asarray(beta_prior_mean, dtype=float)
    beta_prior_prec = np.asarray(beta_prior_prec, dtype=float)

    # initialize at prior mean for omega draw center
    c = X @ beta_prior_mean
    omega = sample_pg1_truncated(c, rng=rng, trunc=omega_trunc)
    kappa = y - 0.5

    XtOmega = X.T * omega
    post_prec = beta_prior_prec + XtOmega @ X
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (beta_prior_prec @ beta_prior_mean + X.T @ kappa)
    beta = rng.multivariate_normal(post_mean, post_cov)

    # refresh omega at sampled beta (returned for diagnostics)
    omega = sample_pg1_truncated(X @ beta, rng=rng, trunc=omega_trunc)
    return beta, omega
