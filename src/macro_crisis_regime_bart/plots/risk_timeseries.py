from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_risk_timeseries(df: pd.DataFrame, output_path: str, country_id: str | None = None) -> None:
    xdf = df if country_id is None else df[df["country_id"] == country_id]
    xdf = xdf.sort_values(["year", "month"])
    plt.figure(figsize=(10, 3))
    plt.plot(xdf["time_id"], xdf["p"], label="Predicted risk")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
