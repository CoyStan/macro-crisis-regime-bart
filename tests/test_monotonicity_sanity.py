import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.bart_static import StaticProbitMonotoneBART


def test_static_model_monotonicity_sanity():
    rng = np.random.default_rng(9)
    n = 160
    x = rng.uniform(-2, 2, size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = pd.DataFrame({"x": x})

    m = StaticProbitMonotoneBART(n_mcmc=40, burn_in=10, thin=2, n_trees=10, monotonicity=[1])
    m.fit(X, y)
    grid = pd.DataFrame({"x": np.linspace(-2, 2, 12)})
    p = m.predict_proba(grid)
    assert p[-1] >= p[0]
