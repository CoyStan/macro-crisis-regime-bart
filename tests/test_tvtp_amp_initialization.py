import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.tvtp_amp import TVTPAmplifiedRegimeSwitchingProbitMonotoneBART


def test_tvtp_staged_warmstart_and_init_source():
    rng = np.random.default_rng(4)
    T, C = 18, 3
    rows = []
    W = []
    for t in range(T):
        W.append({"time_id": 200001 + t, "w": float(t > 8)})
        for c in range(C):
            x = rng.normal()
            y = int(x + rng.normal() > 0)
            rows.append({"country_id": f"C{c}", "time_id": 200001 + t, "x": x, "y": y})
    df = pd.DataFrame(rows)
    Wdf = pd.DataFrame(W)
    m = TVTPAmplifiedRegimeSwitchingProbitMonotoneBART(n_trees=5, n0_warmup=3, n1_warmup=3, n_burn=4, n_draws=5, thin=1)
    m.fit(df[["x"]], df["y"].to_numpy(), df["country_id"].to_numpy(), df["time_id"].to_numpy(), W_time=Wdf)
    assert len(m.saved["s"]) == 5
    assert m.regime_init_source_ == "global_covariates_only"
