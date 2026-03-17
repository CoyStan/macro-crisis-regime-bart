from pathlib import Path
import pandas as pd
import yaml

from macro_crisis_regime_bart.benchmarks.run_benchmarks import run_benchmark_experiment


def test_benchmark_runner_smoke(tmp_path: Path):
    n = 120
    rows = []
    for y in range(2000, 2010):
        for m in range(1, 13):
            rows.append({"country_id": "A", "year": y, "month": m, "crisis_any": int((y+m) % 11 == 0)})
    crisis = pd.DataFrame(rows)
    feats = crisis[["country_id", "year", "month"]].copy()
    feats["credit_gap"] = range(len(feats))
    feats["fx_reserves"] = range(len(feats), 0, -1)
    feats["vix_global"] = 20

    crisis_path = tmp_path / "crisis.csv"
    feat_path = tmp_path / "features.csv"
    crisis.to_csv(crisis_path, index=False)
    feats.to_csv(feat_path, index=False)

    data_cfg = {"crisis_path": str(crisis_path), "feature_path": str(feat_path)}
    reg_cfg = {
        "features": [
            {"name": "credit_gap", "include": True, "monotone": 1, "transform": "none", "lags": {"forecast": 1}, "is_global": False},
            {"name": "fx_reserves", "include": True, "monotone": -1, "transform": "none", "lags": {"forecast": 1}, "is_global": False},
            {"name": "vix_global", "include": True, "monotone": 1, "transform": "none", "lags": {"forecast": 0}, "is_global": True},
        ]
    }
    model_cfg = {"model": "logistic", "params": {"max_iter": 200, "random_state": 42}}

    data_cfg_path = tmp_path / "data.yaml"
    reg_path = tmp_path / "reg.yaml"
    model_path = tmp_path / "logit.yaml"
    exp_path = tmp_path / "exp.yaml"
    for path, obj in [(data_cfg_path, data_cfg), (reg_path, reg_cfg), (model_path, model_cfg)]:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f)

    exp = {
        "run_name": f"pytest_smoke_run_{tmp_path.name}",
        "data_config": str(data_cfg_path),
        "feature_registry": str(reg_path),
        "target": {"column": "crisis_any", "mode": "forecast", "forecast_horizon": 3},
        "split": {"train_end_year": 2006, "val_years": 1},
        "models": ["logistic"],
        "model_configs": {"logistic": str(model_path)},
    }
    with open(exp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(exp, f)

    run_dir = run_benchmark_experiment(exp_path)
    assert (run_dir / "tables" / "metrics.csv").exists()
