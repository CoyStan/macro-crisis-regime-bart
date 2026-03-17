from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class SplitIndices:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


def chronological_split(df: pd.DataFrame, train_end_year: int, val_years: int = 2) -> SplitIndices:
    train = df[df["year"] <= train_end_year].index
    val = df[(df["year"] > train_end_year) & (df["year"] <= train_end_year + val_years)].index
    test = df[df["year"] > train_end_year + val_years].index
    return SplitIndices(train, val, test)


def block_year_cv(df: pd.DataFrame, years_per_fold: int = 3) -> list[tuple[pd.Index, pd.Index]]:
    years = sorted(df["year"].unique())
    folds = []
    for i in range(0, len(years), years_per_fold):
        val_years = set(years[i : i + years_per_fold])
        val_idx = df[df["year"].isin(val_years)].index
        train_idx = df[~df["year"].isin(val_years)].index
        folds.append((train_idx, val_idx))
    return folds


def rolling_origin_splits(df: pd.DataFrame, initial_train_end_year: int, step_years: int = 1, val_years: int = 1) -> list[SplitIndices]:
    max_year = int(df["year"].max())
    cur = initial_train_end_year
    splits: list[SplitIndices] = []
    while cur + val_years < max_year:
        split = chronological_split(df, train_end_year=cur, val_years=val_years)
        if len(split.test_idx) == 0:
            break
        splits.append(split)
        cur += step_years
    return splits
