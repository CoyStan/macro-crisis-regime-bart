import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.rs2 import RegimeSwitchingProbitMonotoneBART


def test_rs2_smoke():
    rng = np.random.default_rng(2)
    n = 160
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    country_ids = np.array([f"C{i%5}" for i in range(n)])
    time_ids = np.array([200001 + (i // 5) for i in range(n)])
    regime = np.array([(i // 20) % 2 for i in range(n)])
    latent = 0.2 * X["x1"].to_numpy() + regime * 0.6 * np.sin(X["x2"].to_numpy())
    y = (latent + rng.normal(scale=0.8, size=n) > 0).astype(int)

    m = RegimeSwitchingProbitMonotoneBART(n_mcmc=24, burn_in=8, thin=2, n_trees_baseline=6, n_trees_deviation=4)
    m.fit(X, y, country_ids=country_ids, time_ids=time_ids)
    p = m.predict_proba(X.iloc[:25], country_ids=country_ids[:25], time_ids=time_ids[:25])
    comp = m.component_summary_df(X.iloc[:25], country_ids=country_ids[:25], time_ids=time_ids[:25])
    assert p.shape == (25,)
    assert not np.isnan(p).any()
    assert {"alpha", "delta", "f0", "g_active", "latent_total", "probability"}.issubset(comp.columns)
