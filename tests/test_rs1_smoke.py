import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.rs1 import RegimeSwitchingProbitMonotoneBARTPhase1


def test_rs1_smoke():
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    country_ids = np.array([f"C{i%4}" for i in range(n)])
    time_ids = np.array([200001 + (i // 4) for i in range(n)])
    latent = 0.3 * X["x1"].to_numpy() - 0.2 * X["x2"].to_numpy()
    y = (latent + rng.normal(size=n) > 0).astype(int)

    m = RegimeSwitchingProbitMonotoneBARTPhase1(n_mcmc=20, burn_in=8, thin=2, n_trees=6, monotonicity=[1, -1])
    m.fit(X, y, country_ids=country_ids, time_ids=time_ids)
    p = m.predict_proba(X.iloc[:20], country_ids=country_ids[:20], time_ids=time_ids[:20])
    rp = m.regime_posterior_summary()
    assert p.shape == (20,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(rp.filter(like="p_regime_").sum(axis=1), 1.0)
