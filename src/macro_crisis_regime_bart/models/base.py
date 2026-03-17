from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import joblib


class BaseCrisisModel(ABC):
    """Base interface for crisis prediction models."""

    @abstractmethod
    def fit(self, X, y) -> "BaseCrisisModel":
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba_samples(self, X) -> np.ndarray | None:
        return None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseCrisisModel":
        return joblib.load(path)
