import numpy as np
from macro_crisis_regime_bart.models.samplers.ffbs import ffbs_sample_tvtp, tvtp_transition_log_probs


def test_ffbs_tvtp_logspace_stability():
    rng = np.random.default_rng(0)
    T = 30
    W = rng.normal(size=(T, 2))
    beta = np.array([[0.2, 0.5, -0.2], [-0.1, 0.3, 0.4]])
    log_tr = tvtp_transition_log_probs(beta, W)
    log_emissions = rng.normal(loc=-2.0, scale=0.5, size=(T, 2))
    s, filt = ffbs_sample_tvtp(log_emissions, log_tr, np.array([0.7, 0.3]), rng)
    assert s.shape == (T,)
    assert filt.shape == (T, 2)
    assert np.allclose(filt.sum(axis=1), 1.0)
    assert np.all(np.isfinite(filt))
