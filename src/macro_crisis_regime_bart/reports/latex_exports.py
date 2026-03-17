from __future__ import annotations

import pandas as pd


def export_latex_table(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
