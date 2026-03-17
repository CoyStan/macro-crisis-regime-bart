from dataclasses import dataclass, field


KEY_COLS = ["country_id", "year", "month"]


@dataclass
class DataSchema:
    key_cols: list[str] = field(default_factory=lambda: KEY_COLS.copy())
    crisis_target_col: str = "crisis_any"

