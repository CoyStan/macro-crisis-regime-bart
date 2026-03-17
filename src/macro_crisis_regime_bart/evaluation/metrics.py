from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score


def threshold_search(y_true: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    tp = float(((pred == 1) & (y_true == 1)).sum())
    return {
        "auroc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "aucpr": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "type1_error": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "type2_error": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    out = df.groupby("bin", observed=False).agg(mean_pred=("p", "mean"), event_rate=("y", "mean"), n=("y", "size")).reset_index()
    return out


def panel_summary(df: pd.DataFrame, group_col: str = "country_id") -> pd.DataFrame:
    return df.groupby(group_col).agg(n=("y", "size"), event_rate=("y", "mean"), avg_pred=("p", "mean")).reset_index()
