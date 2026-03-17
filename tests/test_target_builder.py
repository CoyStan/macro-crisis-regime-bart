import pandas as pd
from macro_crisis_regime_bart.data.target_builder import build_target


def test_forecast_target_builds_forward_event():
    df = pd.DataFrame({"country_id": ["A"] * 4, "year": [2000] * 4, "month": [1,2,3,4], "crisis_any": [0,0,1,0]})
    out = build_target(df, "crisis_any", mode="forecast", forecast_horizon=2)
    assert out["y"].iloc[0] == 1


def test_nowcast_target():
    df = pd.DataFrame({"country_id": ["A"] * 3, "year": [2000]*3, "month": [1,2,3], "crisis_any": [0,1,0]})
    out = build_target(df, "crisis_any", mode="nowcast", nowcast_window=2)
    assert list(out["y"]) == [0,1,1]
