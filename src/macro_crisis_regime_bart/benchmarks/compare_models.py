from __future__ import annotations

import pandas as pd


def rank_models(metrics_df: pd.DataFrame, metric: str = "aucpr") -> pd.DataFrame:
    out = metrics_df.sort_values(metric, ascending=False).copy()
    out["rank"] = range(1, len(out) + 1)
    return out
