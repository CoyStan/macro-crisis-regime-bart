from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet table from disk."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported extension for {p}")
