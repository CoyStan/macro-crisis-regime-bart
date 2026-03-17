from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
import statsmodels.api as sm

from .base import BaseCrisisModel


class LogisticModel(BaseCrisisModel):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class ElasticNetLogitModel(LogisticModel):
    pass


class RandomForestModel(BaseCrisisModel):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class ProbitModel(BaseCrisisModel):
    def __init__(self):
        self.result = None

    def fit(self, X, y):
        Xc = sm.add_constant(X, has_constant="add")
        self.result = sm.Probit(y, Xc).fit(disp=False)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xc = sm.add_constant(X, has_constant="add")
        lin = self.result.predict(Xc, linear=True)
        return norm.cdf(lin)
