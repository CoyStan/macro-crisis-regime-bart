import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.bart_static import StaticProbitMonotoneBART


def test_static_pmbart_smoke_and_missing_routing_and_save_load(tmp_path):
    rng = np.random.default_rng(4)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x1[::7] = np.nan
    X = pd.DataFrame({"x1": x1, "x2": x2})
    latent = 0.4 * np.nan_to_num(x1, nan=0.0) - 0.1 * x2
    y = (latent + rng.normal(size=n) > 0).astype(int)

    m = StaticProbitMonotoneBART(n_mcmc=30, burn_in=10, thin=2, n_trees=8, monotonicity=[1, -1], missing="native")
    m.fit(X, y)
    ps = m.predict_proba_samples(X.iloc[:15])
    p = m.predict_proba(X.iloc[:15])
    assert ps.shape[1] == 15
    assert p.shape == (15,)
    assert np.all((p >= 0) & (p <= 1))

    model_path = tmp_path / "model.joblib"
    m.save(model_path)
    loaded = StaticProbitMonotoneBART.load(model_path)
    p2 = loaded.predict_proba(X.iloc[:15])
    assert p2.shape == (15,)
