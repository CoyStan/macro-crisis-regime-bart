from __future__ import annotations

import pandas as pd


def coerce_key_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["country_id"] = out["country_id"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="raise").astype(int)
    return out


def validate_keys(df: pd.DataFrame, key_cols: list[str]) -> None:
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    if df[key_cols].isna().any().any():
        raise ValueError("Null values found in key columns")
    if df.duplicated(key_cols).any():
        raise ValueError("Duplicate country-year-month keys found")
    if not df["month"].between(1, 12).all():
        raise ValueError("month must be in 1..12")


def validate_sorted(df: pd.DataFrame, key_cols: list[str]) -> None:
    sorted_df = df.sort_values(key_cols).reset_index(drop=True)
    if not sorted_df[key_cols].equals(df[key_cols].reset_index(drop=True)):
        raise ValueError("Input dataframe must be sorted by key columns")
