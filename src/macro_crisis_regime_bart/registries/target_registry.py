from __future__ import annotations


def validate_target_column(target_col: str, available_columns: list[str]) -> None:
    if target_col not in available_columns:
        raise ValueError(f"Configured target '{target_col}' not found in crisis dataset")
