"""Tests for blackjack/ml_advanced.py — EnsembleStacking & calibration."""

import pytest
import numpy as np

from blackjack.ml_advanced import (
    EnsembleStacking,
    ModelCalibration,
    BasicStrategyClassifier,
    ModelMetrics,
)


def _make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 10))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(0, 0.3, n)
    return X, y


# ---------------------------------------------------------------------------
# EnsembleStacking
# ---------------------------------------------------------------------------

def test_ensemble_fits_and_predicts():
    X, y = _make_dataset()
    model = EnsembleStacking(n_estimators=10, n_folds=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_ensemble_prediction_correlated_with_target():
    X, y = _make_dataset(n=500)
    model = EnsembleStacking(n_estimators=20, n_folds=3)
    model.fit(X, y)
    preds = model.predict(X)
    corr = float(np.corrcoef(preds, y)[0, 1])
    assert corr > 0.5  # should be positively correlated


def test_ensemble_not_fitted_raises():
    model = EnsembleStacking()
    X = np.zeros((5, 10))
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_ensemble_evaluate_returns_metrics():
    X, y = _make_dataset()
    model = EnsembleStacking(n_estimators=10, n_folds=3)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    assert isinstance(metrics, ModelMetrics)
    assert metrics.mae >= 0
    assert metrics.rmse >= metrics.mae  # RMSE ≥ MAE always


def test_ensemble_predict_with_uncertainty():
    X, y = _make_dataset()
    model = EnsembleStacking(n_estimators=10, n_folds=3)
    model.fit(X, y)
    results = model.predict_with_uncertainty(X[:5])
    assert len(results) == 5
    for r in results:
        assert 0.0 <= r.confidence <= 1.0
        assert r.uncertainty >= 0.0
        assert isinstance(r.recommended_action, str)
        assert len(r.individual_preds) == 3


def test_ensemble_feature_importances_sum_to_positive():
    X, y = _make_dataset()
    names = [f"f_{i}" for i in range(10)]
    model = EnsembleStacking(n_estimators=10, n_folds=3)
    model.fit(X, y, feature_names=names)
    fi = model.feature_importances(names)
    assert sum(fi.values()) > 0
    assert all(v >= 0 for v in fi.values())


def test_ensemble_permutation_importance():
    X, y = _make_dataset()
    model = EnsembleStacking(n_estimators=10, n_folds=3)
    model.fit(X, y)
    perm = model.permutation_importance(X, y, n_repeats=3)
    assert len(perm) == X.shape[1]


# ---------------------------------------------------------------------------
# ModelCalibration
# ---------------------------------------------------------------------------

def test_calibration_perfect_model():
    cal = ModelCalibration()
    pred = [0.1, 0.2, 0.3, 0.4, 0.5]
    real = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = cal.calibrate(pred, real)
    assert result["mae"] == pytest.approx(0.0, abs=1e-10)
    assert result["bias"] == pytest.approx(0.0, abs=1e-10)


def test_calibration_biased_model():
    cal = ModelCalibration()
    rng = np.random.default_rng(42)
    # Predictions near 0.3, actuals near -0.3 (biased by ~0.6)
    pred = list(rng.normal(0.3, 0.05, 50))
    real = list(rng.normal(-0.3, 0.05, 50))
    result = cal.calibrate(pred, real)
    assert result["bias"] > 0.5  # predicted much higher than actual
    assert result["is_calibrated"] is False


def test_calibration_correlation():
    cal = ModelCalibration()
    rng = np.random.default_rng(42)
    pred = rng.normal(0, 1, 100)
    real = pred + rng.normal(0, 0.1, 100)
    result = cal.calibrate(pred, real)
    assert result["spearman_rho"] > 0.9


# ---------------------------------------------------------------------------
# BasicStrategyClassifier
# ---------------------------------------------------------------------------

def test_basic_strategy_stand_on_17():
    clf = BasicStrategyClassifier()
    assert clf.predict_action(17, False, "T") == "stand"
    assert clf.predict_action(18, False, "T") == "stand"
    assert clf.predict_action(21, False, "T") == "stand"


def test_basic_strategy_hit_low_vs_ten():
    clf = BasicStrategyClassifier()
    assert clf.predict_action(12, False, "T") == "hit"
    assert clf.predict_action(13, False, "T") == "hit"


def test_basic_strategy_stand_vs_weak_dealer():
    clf = BasicStrategyClassifier()
    assert clf.predict_action(13, False, "6") == "stand"
    assert clf.predict_action(15, False, "4") == "stand"


def test_basic_strategy_double_11():
    clf = BasicStrategyClassifier()
    assert clf.predict_action(11, False, "6", can_double=True) == "double"


def test_basic_strategy_double_not_allowed_hit_11():
    clf = BasicStrategyClassifier()
    assert clf.predict_action(11, False, "6", can_double=False) == "hit"


def test_basic_strategy_soft_18_vs_weak():
    clf = BasicStrategyClassifier()
    # Soft 18 vs 6: double if allowed, else stand
    action = clf.predict_action(18, True, "6", can_double=True)
    assert action in {"double", "stand"}
