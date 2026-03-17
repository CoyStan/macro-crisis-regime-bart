from __future__ import annotations

import pandas as pd

from .schema import KEY_COLS
from .validation import coerce_key_types, validate_keys


def merge_crisis_features(crisis_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and merge crisis and feature data."""
    c = coerce_key_types(crisis_df)
    f = coerce_key_types(feature_df)
    validate_keys(c, KEY_COLS)
    validate_keys(f, KEY_COLS)
    merged = c.merge(f, on=KEY_COLS, how="inner", validate="one_to_one")
    merged = merged.sort_values(KEY_COLS).reset_index(drop=True)
    merged["time_id"] = merged["year"] * 100 + merged["month"]
    return merged
