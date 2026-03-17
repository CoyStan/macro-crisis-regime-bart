from __future__ import annotations

from .base import BaseCrisisModel
from .sklearn_wrappers import LogisticModel, ProbitModel, ElasticNetLogitModel, RandomForestModel
from .xgboost_monotone import XGBoostModel, MonotoneXGBoostModel
from .bart_static import StaticProbitMonotoneBART
from .rs1 import RegimeSwitchingProbitMonotoneBARTPhase1
from .rs2 import RegimeSwitchingProbitMonotoneBART
from .tvtp_amp import TVTPAmplifiedRegimeSwitchingProbitMonotoneBART


def create_model(model_name: str, params: dict, monotonicity: list[int] | None = None) -> BaseCrisisModel:
    if model_name == "logistic":
        return LogisticModel(**params)
    if model_name == "probit":
        return ProbitModel()
    if model_name == "elastic_net_logit":
        return ElasticNetLogitModel(**params)
    if model_name == "random_forest":
        return RandomForestModel(**params)
    if model_name == "xgboost":
        return XGBoostModel(**params)
    if model_name == "monotone_xgboost":
        if monotonicity is None:
            raise ValueError("Monotonicity constraints required for monotone_xgboost")
        return MonotoneXGBoostModel(monotonicity=monotonicity, **params)
    if model_name == "static_pmBART":
        p = dict(params)
        p.setdefault("monotonicity", monotonicity)
        return StaticProbitMonotoneBART(**p)
    if model_name == "rs1":
        p = dict(params)
        p.setdefault("monotonicity", monotonicity)
        return RegimeSwitchingProbitMonotoneBARTPhase1(**p)
    if model_name == "rs2":
        p = dict(params)
        p.setdefault("monotonicity_baseline", monotonicity)
        return RegimeSwitchingProbitMonotoneBART(**p)
    if model_name == "tvtp_amp":
        p = dict(params)
        p.setdefault("monotonicity", monotonicity)
        return TVTPAmplifiedRegimeSwitchingProbitMonotoneBART(**p)
    raise ValueError(f"Unknown model: {model_name}")
