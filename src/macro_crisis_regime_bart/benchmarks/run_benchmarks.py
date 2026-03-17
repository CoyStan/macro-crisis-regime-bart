from __future__ import annotations

import time
from pathlib import Path
import yaml
import pandas as pd
from sklearn.impute import SimpleImputer

from macro_crisis_regime_bart.data.io import read_table
from macro_crisis_regime_bart.data.merge import merge_crisis_features
from macro_crisis_regime_bart.data.target_builder import build_target
from macro_crisis_regime_bart.data.feature_builder import build_features
from macro_crisis_regime_bart.data.splitters import chronological_split
from macro_crisis_regime_bart.models.factory import create_model
from macro_crisis_regime_bart.registries.feature_registry import load_feature_registry
from macro_crisis_regime_bart.evaluation.metrics import compute_metrics, threshold_search
from macro_crisis_regime_bart.utils.paths import make_run_dir
from macro_crisis_regime_bart.utils.serialization import write_yaml


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_benchmark_experiment(config_path: str | Path) -> Path:
    cfg = _load_yaml(config_path)
    data_cfg = _load_yaml(cfg["data_config"])
    feature_registry = load_feature_registry(cfg["feature_registry"])
    run_dir = make_run_dir(run_name=cfg.get("run_name"))
    for d in ["tables", "predictions", "models", "diagnostics"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    crisis_df = read_table(data_cfg["crisis_path"])
    feature_df = read_table(data_cfg["feature_path"])
    merged = merge_crisis_features(crisis_df, feature_df)
    targeted = build_target(
        merged,
        target_col=cfg["target"]["column"],
        mode=cfg["target"]["mode"],
        forecast_horizon=int(cfg["target"].get("forecast_horizon", 12)),
        nowcast_window=int(cfg["target"].get("nowcast_window", 3)),
    )
    fb = build_features(targeted, feature_registry, mode=cfg["target"]["mode"])

    split = chronological_split(fb.df, train_end_year=int(cfg["split"]["train_end_year"]), val_years=int(cfg["split"].get("val_years", 2)))

    X = fb.df[fb.x_cols]
    y = fb.df["y"].astype(int)

    w_time_df = None
    if fb.w_cols:
        w_time_df = fb.df[["time_id", *fb.w_cols]].drop_duplicates("time_id").sort_values("time_id").reset_index(drop=True)

    X_train_raw, y_train = X.loc[split.train_idx], y.loc[split.train_idx]
    X_val_raw, y_val = X.loc[split.val_idx], y.loc[split.val_idx]
    X_test_raw, y_test = X.loc[split.test_idx], y.loc[split.test_idx]

    metrics_rows = []
    predictions = []
    for model_name in cfg["models"]:
        model_cfg = _load_yaml(cfg["model_configs"][model_name])
        params = model_cfg.get("params", {})
        start = time.time()
        model = create_model(model_name, params, monotonicity=fb.monotonicity)

        use_native_missing = model_name in {"static_pmBART", "rs1", "rs2", "tvtp_amp"} and params.get("missing", "native") == "native"
        if use_native_missing:
            X_train, X_val, X_test = X_train_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
        else:
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
            X_val = pd.DataFrame(imputer.transform(X_val_raw), columns=X_val_raw.columns, index=X_val_raw.index)
            X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

        if model_name in {"rs1", "rs2"}:
            model.fit(X_train, y_train, country_ids=fb.df.loc[split.train_idx, "country_id"], time_ids=fb.df.loc[split.train_idx, "time_id"])
            p_val = model.predict_proba(X_val, country_ids=fb.df.loc[split.val_idx, "country_id"], time_ids=fb.df.loc[split.val_idx, "time_id"])
            p_test = model.predict_proba(X_test, country_ids=fb.df.loc[split.test_idx, "country_id"], time_ids=fb.df.loc[split.test_idx, "time_id"])
        elif model_name == "tvtp_amp":
            if w_time_df is None:
                raise ValueError("tvtp_amp requires global W_t covariates in feature registry (is_global=true)")
            model.fit(
                X_train, y_train,
                country_ids=fb.df.loc[split.train_idx, "country_id"],
                time_ids=fb.df.loc[split.train_idx, "time_id"],
                W_time=w_time_df,
            )
            p_val = model.predict_proba(
                X_val,
                country_ids=fb.df.loc[split.val_idx, "country_id"],
                time_ids=fb.df.loc[split.val_idx, "time_id"],
                W_time=w_time_df,
            )
            p_test = model.predict_proba(
                X_test,
                country_ids=fb.df.loc[split.test_idx, "country_id"],
                time_ids=fb.df.loc[split.test_idx, "time_id"],
                W_time=w_time_df,
            )
        else:
            model.fit(X_train, y_train)
            p_val = model.predict_proba(X_val)
            p_test = model.predict_proba(X_test)

        threshold, _ = threshold_search(y_val.to_numpy(), p_val)
        val_metrics = compute_metrics(y_val.to_numpy(), p_val, threshold)
        test_metrics = compute_metrics(y_test.to_numpy(), p_test, threshold)
        elapsed = time.time() - start

        model_path = run_dir / "models" / f"{model_name}.joblib"
        model.save(model_path)

        metrics_rows.append({"model": model_name, "split": "val", "threshold": threshold, "runtime_sec": elapsed, **val_metrics})
        metrics_rows.append({"model": model_name, "split": "test", "threshold": threshold, "runtime_sec": elapsed, **test_metrics})
        pred_df = fb.df.loc[split.test_idx, ["country_id", "year", "month", "time_id", "y"]].copy()
        pred_df["p"] = p_test
        pred_df["model"] = model_name
        predictions.append(pred_df)

        if model_name in {"rs1", "rs2", "tvtp_amp"} and hasattr(model, "regime_posterior_summary"):
            model.regime_posterior_summary().to_csv(run_dir / "diagnostics" / f"{model_name}_regime_probs.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(predictions, ignore_index=True)

    metrics_df.to_csv(run_dir / "tables" / "metrics.csv", index=False)
    preds_df.to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
    fb.metadata.to_csv(run_dir / "tables" / "feature_metadata.csv", index=False)
    write_yaml(cfg, run_dir / "config_snapshot.yaml")
    return run_dir
