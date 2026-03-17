import pandas as pd
from macro_crisis_regime_bart.data.feature_builder import build_features


def test_feature_builder_outputs_metadata_and_monotonicity():
    df = pd.DataFrame({"country_id": ["A","A"], "year": [2000,2000], "month": [1,2], "x1": [1.0,2.0], "g1": [3.0,3.0]})
    registry = {"features": [
        {"name": "x1", "include": True, "monotone": 1, "transform": "none", "lags": {"forecast": 1}, "is_global": False},
        {"name": "g1", "include": True, "monotone": 0, "transform": "none", "lags": {"forecast": 0}, "is_global": True},
    ]}
    out = build_features(df, registry, mode="forecast")
    assert out.x_cols == ["x1_lag1"]
    assert out.w_cols == ["g1"]
    assert out.monotonicity == [1]
