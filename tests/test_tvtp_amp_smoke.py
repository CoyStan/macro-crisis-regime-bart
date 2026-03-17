import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.tvtp_amp import TVTPAmplifiedRegimeSwitchingProbitMonotoneBART


def test_tvtp_amp_smoke():
    rng = np.random.default_rng(3)
    T = 24
    C = 4
    rows = []
    W_rows = []
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        p = 0.2 + 0.5 * (t > 10)
        s[t] = int(rng.random() < p)
    for t in range(T):
        W_rows.append({"time_id": 200001 + t, "w_stress": float(t > 10), "w_liq": rng.normal()})
        for c in range(C):
            x1 = rng.normal()
            x2 = rng.normal()
            lam = 1.0 if s[t] == 0 else 1.6
            latent = -0.2 + 0.3 * s[t] + lam * (0.5 * x1 - 0.2 * x2) + rng.normal(scale=0.7)
            y = int(latent > 0)
            rows.append({"country_id": f"C{c}", "time_id": 200001 + t, "year": 2000 + (t // 12), "month": 1 + (t % 12), "x1": x1, "x2": x2, "y": y})

    df = pd.DataFrame(rows)
    W = pd.DataFrame(W_rows)
    m = TVTPAmplifiedRegimeSwitchingProbitMonotoneBART(
        n_trees=8, n0_warmup=5, n1_warmup=5, n_burn=8, n_draws=8, thin=1, monotonicity=[1, -1]
    )
    m.fit(df[["x1", "x2"]], df["y"].to_numpy(), country_ids=df["country_id"].to_numpy(), time_ids=df["time_id"].to_numpy(), W_time=W)
    p = m.predict_proba(df[["x1", "x2"]].iloc[:20], country_ids=df["country_id"].to_numpy()[:20], time_ids=df["time_id"].to_numpy()[:20], W_time=W)
    rp = m.regime_posterior_summary()
    assert p.shape == (20,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose((rp["p_regime_1"] + rp["p_regime_2"]).to_numpy(), 1.0)
    assert np.nanmean(np.array(m.saved["eta2"])) > 0
