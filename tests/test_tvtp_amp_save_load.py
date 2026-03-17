import numpy as np
import pandas as pd
from macro_crisis_regime_bart.models.tvtp_amp import TVTPAmplifiedRegimeSwitchingProbitMonotoneBART


def test_tvtp_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    T, C = 14, 3
    rows = []
    W = []
    for t in range(T):
        W.append({"time_id": 200001 + t, "w": rng.normal()})
        for c in range(C):
            x = rng.normal()
            y = int(0.4 * x + rng.normal() > 0)
            rows.append({"country_id": f"C{c}", "time_id": 200001 + t, "x": x, "y": y})
    df = pd.DataFrame(rows)
    Wdf = pd.DataFrame(W)
    m = TVTPAmplifiedRegimeSwitchingProbitMonotoneBART(n_trees=5, n0_warmup=2, n1_warmup=2, n_burn=3, n_draws=4, thin=1)
    m.fit(df[["x"]], df["y"].to_numpy(), df["country_id"].to_numpy(), df["time_id"].to_numpy(), W_time=Wdf)
    p1 = m.predict_proba(df[["x"]].iloc[:10], df["country_id"].to_numpy()[:10], df["time_id"].to_numpy()[:10], W_time=Wdf)
    fp = tmp_path / "tvtp.joblib"
    m.save(fp)
    m2 = TVTPAmplifiedRegimeSwitchingProbitMonotoneBART.load(fp)
    p2 = m2.predict_proba(df[["x"]].iloc[:10], df["country_id"].to_numpy()[:10], df["time_id"].to_numpy()[:10], W_time=Wdf)
    assert p1.shape == p2.shape
