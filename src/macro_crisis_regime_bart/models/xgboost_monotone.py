from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from .base import BaseCrisisModel


class XGBoostModel(BaseCrisisModel):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class MonotoneXGBoostModel(XGBoostModel):
    def __init__(self, monotonicity: list[int], **kwargs):
        kwargs = dict(kwargs)
        kwargs["monotone_constraints"] = f"({','.join(str(m) for m in monotonicity)})"
        super().__init__(**kwargs)
