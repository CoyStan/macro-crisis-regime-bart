"""Truncated normal sampling utilities for Albert-Chib latent-variable probit blocks."""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm


def sample_truncated_normal(
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized truncated normal draws with unit variance.

    Parameters
    ----------
    mean, lower, upper:
        Broadcastable arrays where each draw follows N(mean_i, 1) truncated
        to [lower_i, upper_i].
    rng:
        Numpy random generator for deterministic reproducibility.
    """
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    a = lower - mean
    b = upper - mean
    u = rng.random(mean.shape)
    return truncnorm.ppf(u, a, b, loc=mean, scale=1.0)


def sample_probit_latent_z(y: np.ndarray, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample latent probit z | y, eta for Albert-Chib data augmentation."""
    y = np.asarray(y).astype(int)
    eta = np.asarray(eta, dtype=float)
    lower = np.where(y == 1, 0.0, -np.inf)
    upper = np.where(y == 1, np.inf, 0.0)
    return sample_truncated_normal(eta, lower, upper, rng)
