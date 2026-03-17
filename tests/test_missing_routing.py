import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.bart_static import StaticProbitMonotoneBART


def test_missing_native_routing_no_nan_predictions():
    rng = np.random.default_rng(12)
    n = 100
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    X.loc[::5, "x1"] = np.nan
    y = (np.nan_to_num(X["x1"].to_numpy(), nan=0.0) - 0.3 * X["x2"].to_numpy() + rng.normal(size=n) > 0).astype(int)
    m = StaticProbitMonotoneBART(n_mcmc=20, burn_in=8, thin=2, n_trees=6, missing="native")
    m.fit(X, y)
    p = m.predict_proba(X)
    assert not np.isnan(p).any()
