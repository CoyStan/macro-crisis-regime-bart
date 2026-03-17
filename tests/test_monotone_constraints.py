from macro_crisis_regime_bart.models.xgboost_monotone import MonotoneXGBoostModel


def test_monotone_constraints_applied():
    m = MonotoneXGBoostModel(monotonicity=[1,-1], n_estimators=1, random_state=1, eval_metric='logloss')
    assert m.model.get_xgb_params()["monotone_constraints"] == "(1,-1)"
