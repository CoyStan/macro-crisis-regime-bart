from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FeatureBuildResult:
    df: pd.DataFrame
    x_cols: list[str]
    w_cols: list[str]
    monotonicity: list[int]
    metadata: pd.DataFrame


def _apply_transform(series: pd.Series, name: str) -> pd.Series:
    if name == "none":
        return series
    if name == "log1p":
        return (series.clip(lower=0)).map(lambda x: None if pd.isna(x) else np.log1p(x))
    if name == "zscore":
        std = series.std(ddof=0)
        return (series - series.mean()) / (std if std else 1.0)
    raise ValueError(f"Unknown transform: {name}")


def build_features(df: pd.DataFrame, registry: dict, mode: str) -> FeatureBuildResult:
    out = df.copy()
    rows = []
    x_cols: list[str] = []
    w_cols: list[str] = []
    monotonicity: list[int] = []

    for spec in registry.get("features", []):
        if not spec.get("include", True):
            continue
        base = spec["name"]
        if base not in out.columns:
            continue
        transformed = _apply_transform(out[base], spec.get("transform", "none"))
        lag = int(spec.get("lags", {}).get(mode, 0))
        col = f"{base}_lag{lag}" if lag else base
        if lag:
            out[col] = transformed.groupby(out["country_id"]).shift(lag)
        else:
            out[col] = transformed
        rows.append({"name": col, "base_name": base, "category": spec.get("category", "unknown"), "monotone": int(spec.get("monotone", 0)), "is_global": bool(spec.get("is_global", False))})
        if spec.get("is_global", False):
            w_cols.append(col)
        else:
            x_cols.append(col)
            monotonicity.append(int(spec.get("monotone", 0)))

    metadata = pd.DataFrame(rows)
    return FeatureBuildResult(df=out, x_cols=x_cols, w_cols=w_cols, monotonicity=monotonicity, metadata=metadata)
