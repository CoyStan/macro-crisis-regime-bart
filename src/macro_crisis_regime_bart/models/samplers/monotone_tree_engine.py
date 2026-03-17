"""Lightweight monotone tree ensemble engine with native NA routing.

This phase uses monotone stumps (depth-1 trees) with Bayesian backfitting updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StumpRule:
    feature_idx: int
    threshold: float
    na_go_left: bool
    left_value: float
    right_value: float


@dataclass
class MonotoneStumpEnsemble:
    n_trees: int
    monotonicity: list[int]
    tau_leaf: float = 0.2
    proposal_sd: float = 0.1
    rng: np.random.Generator | None = None
    rules: list[StumpRule] = field(default_factory=list)
    acceptance: dict[str, int] = field(default_factory=lambda: {"attempt": 0, "accept": 0})

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng(0)

    def _fit_stump(self, X: np.ndarray, r: np.ndarray) -> StumpRule:
        n, p = X.shape
        j = int(self.rng.integers(0, p))
        col = X[:, j]
        non_nan = col[~np.isnan(col)]
        thr = float(np.median(non_nan)) if len(non_nan) else 0.0

        best = None
        for na_left in (True, False):
            left = (col <= thr) | (np.isnan(col) & na_left)
            right = ~left
            lv = float(r[left].mean()) if left.any() else 0.0
            rv = float(r[right].mean()) if right.any() else 0.0

            m = int(self.monotonicity[j]) if j < len(self.monotonicity) else 0
            if m == 1 and lv > rv:
                lv, rv = min(lv, rv), max(lv, rv)
            elif m == -1 and lv < rv:
                lv, rv = max(lv, rv), min(lv, rv)

            pred = np.where(left, lv, rv)
            sse = float(np.sum((r - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, StumpRule(j, thr, na_left, lv, rv))
        assert best is not None
        return best[1]

    def predict_tree(self, X: np.ndarray, rule: StumpRule) -> np.ndarray:
        col = X[:, rule.feature_idx]
        left = (col <= rule.threshold) | (np.isnan(col) & rule.na_go_left)
        return np.where(left, rule.left_value, rule.right_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.rules:
            return np.zeros(X.shape[0], dtype=float)
        out = np.zeros(X.shape[0], dtype=float)
        for rule in self.rules:
            out += self.predict_tree(X, rule)
        return out

    def backfit_step(self, X: np.ndarray, target: np.ndarray) -> None:
        if not self.rules:
            self.rules = [self._fit_stump(X, target / max(self.n_trees, 1)) for _ in range(self.n_trees)]
            return

        for i, current in enumerate(self.rules):
            old_pred = self.predict_tree(X, current)
            resid = target - (self.predict(X) - old_pred)

            proposal = self._fit_stump(X, resid)
            old_sse = float(np.sum((resid - old_pred) ** 2))
            new_pred = self.predict_tree(X, proposal)
            new_sse = float(np.sum((resid - new_pred) ** 2))

            self.acceptance["attempt"] += 1
            log_acc = -0.5 * (new_sse - old_sse)
            if np.log(self.rng.random()) < min(0.0, log_acc):
                self.rules[i] = proposal
                self.acceptance["accept"] += 1

    def acceptance_rate(self) -> float:
        a = self.acceptance["attempt"]
        return 0.0 if a == 0 else self.acceptance["accept"] / a
