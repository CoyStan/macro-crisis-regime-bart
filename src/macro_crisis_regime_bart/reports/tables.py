from __future__ import annotations

import pandas as pd


def export_summary_table(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
