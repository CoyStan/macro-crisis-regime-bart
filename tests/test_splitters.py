import pandas as pd
from macro_crisis_regime_bart.data.splitters import chronological_split, block_year_cv, rolling_origin_splits


def _df():
    return pd.DataFrame({"year": [2000,2001,2002,2003,2004], "month": [1]*5})


def test_chronological_split():
    s = chronological_split(_df(), 2001, 1)
    assert len(s.train_idx) == 2
    assert len(s.val_idx) == 1
    assert len(s.test_idx) == 2


def test_block_cv():
    folds = block_year_cv(_df(), years_per_fold=2)
    assert len(folds) == 3


def test_rolling_origin():
    splits = rolling_origin_splits(_df(), 2001, step_years=1, val_years=1)
    assert len(splits) >= 1
