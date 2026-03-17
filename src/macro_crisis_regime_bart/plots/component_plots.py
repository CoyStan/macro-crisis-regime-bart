from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str, top_n: int = 15) -> None:
    top = importance_df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(6, 4))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
