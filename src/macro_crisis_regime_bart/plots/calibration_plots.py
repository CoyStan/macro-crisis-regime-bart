from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_calibration(calibration_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.plot(calibration_df["mean_pred"], calibration_df["event_rate"], marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
