import numpy as np
from macro_crisis_regime_bart.models.samplers.pg import sample_pg1_truncated, logistic_pg_gaussian_update


def test_pg_shapes_and_finiteness():
    rng = np.random.default_rng(0)
    c = np.array([0.0, 0.5, -1.2, 2.1])
    w = sample_pg1_truncated(c, rng=rng, trunc=80)
    assert w.shape == c.shape
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


def test_logistic_pg_update_runs():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(40), rng.normal(size=40)])
    y = (rng.random(40) > 0.6).astype(float)
    b, om = logistic_pg_gaussian_update(
        X, y,
        beta_prior_mean=np.zeros(2),
        beta_prior_prec=np.eye(2),
        rng=rng,
        omega_trunc=60,
    )
    assert b.shape == (2,)
    assert om.shape == (40,)
    assert np.all(np.isfinite(b))
