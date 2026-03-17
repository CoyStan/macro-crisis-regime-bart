from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_event(series: pd.Series, horizon: int) -> pd.Series:
    vals = series.astype(float).to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        window = vals[i + 1 : i + 1 + horizon]
        if len(window) == 0:
            out[i] = np.nan
        elif np.isnan(window).all():
            out[i] = np.nan
        else:
            out[i] = float(np.nanmax(window) >= 1.0)
    return pd.Series(out, index=series.index)


def _backward_event(series: pd.Series, window: int) -> pd.Series:
    vals = series.astype(float).to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        hist = vals[max(0, i - window + 1) : i + 1]
        if np.isnan(hist).all():
            out[i] = np.nan
        else:
            out[i] = float(np.nanmax(hist) >= 1.0)
    return pd.Series(out, index=series.index)


def build_target(
    df: pd.DataFrame,
    target_col: str,
    mode: str,
    forecast_horizon: int = 12,
    nowcast_window: int = 3,
    group_col: str = "country_id",
    drop_missing_final_y: bool = True,
) -> pd.DataFrame:
    """Construct target variable y for forecast, nowcast, or scenario mode."""
    out = df.copy()
    out = out.sort_values([group_col, "year", "month"])
    if target_col not in out.columns:
        raise ValueError(f"Target column not found: {target_col}")

    if mode == "forecast":
        out["y"] = out.groupby(group_col, group_keys=False)[target_col].apply(lambda s: _forward_event(s, forecast_horizon))
    elif mode == "nowcast":
        out["y"] = out.groupby(group_col, group_keys=False)[target_col].apply(lambda s: _backward_event(s, nowcast_window))
    elif mode == "scenario":
        out["y"] = out[target_col].astype(float)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if drop_missing_final_y:
        out = out[out["y"].notna()].copy()
    return out
