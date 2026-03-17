import numpy as np
from macro_crisis_regime_bart.evaluation.metrics import compute_metrics, threshold_search, calibration_table


def test_metrics_and_threshold():
    y = np.array([0,0,1,1])
    p = np.array([0.1,0.3,0.7,0.9])
    t, f1 = threshold_search(y, p)
    m = compute_metrics(y, p, t)
    assert 0 <= t <= 1
    assert f1 >= 0
    assert "auroc" in m


def test_calibration():
    y = np.array([0,1,0,1])
    p = np.array([0.1,0.2,0.8,0.9])
    c = calibration_table(y, p, bins=2)
    assert len(c) == 2
