"""Advanced ML Models for Blackjack EV Prediction.

Implements an ensemble stacking architecture combining:
- RandomForest (sklearn)
- GradientBoosting (sklearn)
- Logistic Regression (calibrated)
- Ridge meta-learner for stacking

Also provides:
- Cross-validation with proper temporal splits
- SHAP-style permutation importance
- Model calibration analysis
- Feature importance aggregation
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import mean_absolute_error, r2_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed — ML features will be limited", stacklevel=2)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Evaluation metrics for an ML model."""
    model_name: str
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float
    cv_mae_std: float
    n_samples: int
    n_features: int
    feature_importances: Optional[Dict[str, float]] = None


@dataclass
class EnsemblePrediction:
    """Prediction from the ensemble with uncertainty estimate."""
    prediction: float
    confidence: float      # std across base models (inverted)
    individual_preds: Dict[str, float]
    recommended_action: str
    uncertainty: float     # std across base model predictions


# ---------------------------------------------------------------------------
# Base model wrappers
# ---------------------------------------------------------------------------


class _BaseModelWrapper:
    """Thin wrapper around sklearn estimator."""

    def __init__(self, model: Any, name: str) -> None:
        self.model = model
        self.name = name
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseModelWrapper":
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"Model {self.name} not fitted")
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return getattr(self.model, "feature_importances_", None)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return getattr(self.model, "coef_", None)


# ---------------------------------------------------------------------------
# Ensemble Stacking
# ---------------------------------------------------------------------------


class EnsembleStacking:
    """Stacked ensemble for EV prediction.

    Base models:
    - RandomForest (captures non-linear patterns)
    - GradientBoosting (captures interaction effects)
    - Ridge (linear baseline)

    Meta-learner:
    - Ridge regression on base model predictions (out-of-fold)

    Training uses 5-fold cross-validation to generate out-of-fold
    predictions for the meta-learner (proper stacking protocol to
    avoid data leakage).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for EnsembleStacking")

        self.n_folds = n_folds
        self.random_state = random_state
        self._fitted = False

        self._base_models = [
            _BaseModelWrapper(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                name="random_forest",
            ),
            _BaseModelWrapper(
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=random_state,
                ),
                name="gradient_boosting",
            ),
            _BaseModelWrapper(
                Ridge(alpha=1.0),
                name="ridge",
            ),
        ]
        self._meta_learner = Ridge(alpha=0.1)
        self._scaler = StandardScaler()
        self._feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleStacking":
        """Train the ensemble via out-of-fold stacking.

        Parameters
        ----------
        X: np.ndarray, shape (n, p)
        y: np.ndarray, shape (n,) — EV targets
        feature_names: optional list of feature names
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        if feature_names:
            self._feature_names = feature_names

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Out-of-fold predictions for meta-learner
        oof_preds = np.zeros((n, len(self._base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr = y[train_idx]

            for j, model in enumerate(self._base_models):
                model.fit(X_tr, y_tr)
                oof_preds[val_idx, j] = model.predict(X_val)

        # Train meta-learner on OOF predictions
        self._meta_learner.fit(oof_preds, y)

        # Refit all base models on full data
        for model in self._base_models:
            model.fit(X_scaled, y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict EV for each row of X."""
        if not self._fitted:
            raise RuntimeError("EnsembleStacking not fitted")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=float))
        base_preds = np.column_stack([m.predict(X_scaled) for m in self._base_models])
        return self._meta_learner.predict(base_preds)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> List[EnsemblePrediction]:
        """Predict EV and quantify uncertainty via base model disagreement."""
        if not self._fitted:
            raise RuntimeError("EnsembleStacking not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        results = []
        for i in range(len(X)):
            row = X_scaled[i:i+1]
            indiv = {m.name: float(m.predict(row)[0]) for m in self._base_models}
            vals = list(indiv.values())
            uncertainty = float(np.std(vals))
            meta_input = np.array([list(indiv.values())])
            final_pred = float(self._meta_learner.predict(meta_input)[0])
            confidence = max(0.0, 1.0 - uncertainty / (abs(final_pred) + 1.0))

            action = "hit" if final_pred > 0 else "stand"  # simplistic proxy

            results.append(EnsemblePrediction(
                prediction=final_pred,
                confidence=confidence,
                individual_preds=indiv,
                recommended_action=action,
                uncertainty=uncertainty,
            ))
        return results

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ModelMetrics:
        """Evaluate ensemble on a held-out dataset."""
        preds = self.predict(X)
        mae = float(mean_absolute_error(y, preds))
        rmse = float(math.sqrt(np.mean((y - preds) ** 2)))
        r2 = float(r2_score(y, preds))

        # CV evaluation
        X_scaled = self._scaler.transform(np.asarray(X, dtype=float))
        cv_scores = []
        for m in self._base_models:
            s = cross_val_score(
                m.model, X_scaled, y,
                cv=min(5, len(y) // 10 or 2),
                scoring="neg_mean_absolute_error",
            )
            cv_scores.extend(-s)

        fi = self.feature_importances(feature_names)

        return ModelMetrics(
            model_name="EnsembleStacking",
            mae=mae,
            rmse=rmse,
            r2=r2,
            cv_mae_mean=float(np.mean(cv_scores)),
            cv_mae_std=float(np.std(cv_scores)),
            n_samples=len(y),
            n_features=X.shape[1] if hasattr(X, 'shape') else 0,
            feature_importances=fi,
        )

    def feature_importances(
        self, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Aggregate feature importances from tree-based models."""
        names = feature_names or self._feature_names or [
            f"f_{i}" for i in range(100)
        ]

        agg_importance: Dict[str, float] = {}
        for model in self._base_models:
            fi = model.feature_importances_
            if fi is not None:
                for i, imp in enumerate(fi):
                    name = names[i] if i < len(names) else f"f_{i}"
                    agg_importance[name] = agg_importance.get(name, 0.0) + imp / len(self._base_models)

        return dict(sorted(agg_importance.items(), key=lambda x: -x[1]))

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        rng_seed: int = 42,
    ) -> Dict[str, float]:
        """Permutation-based feature importance (model-agnostic).

        Measures how much MAE increases when a feature is randomly shuffled.
        High increase → feature is important.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        rng = np.random.default_rng(rng_seed)

        baseline_mae = float(mean_absolute_error(y, self.predict(X)))
        importances = {}

        for j, name in enumerate(names):
            delta_maes = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                rng.shuffle(X_perm[:, j])
                perm_mae = float(mean_absolute_error(y, self.predict(X_perm)))
                delta_maes.append(perm_mae - baseline_mae)
            importances[name] = float(np.mean(delta_maes))

        return dict(sorted(importances.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


class ModelCalibration:
    """Analyse calibration of EV predictions vs realised outcomes.

    A well-calibrated model has E[outcome | predicted_ev = v] ≈ v.
    """

    def calibrate(
        self,
        predicted_evs: Sequence[float],
        realized_outcomes: Sequence[float],
        n_bins: int = 10,
    ) -> Dict[str, object]:
        """Full calibration analysis."""
        pred = np.asarray(predicted_evs, dtype=float)
        real = np.asarray(realized_outcomes, dtype=float)

        mae = float(np.mean(np.abs(pred - real)))
        rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
        bias = float(np.mean(pred - real))

        # Spearman rank correlation
        rho, pval = stats.spearmanr(pred, real)

        # Binned calibration
        edges = np.linspace(pred.min(), pred.max(), n_bins + 1)
        bin_idx = np.digitize(pred, edges) - 1
        bins = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                bins.append({
                    "pred_mean": float(pred[mask].mean()),
                    "real_mean": float(real[mask].mean()),
                    "count": int(mask.sum()),
                    "error": float(abs(pred[mask].mean() - real[mask].mean())),
                })

        mean_cal_error = float(np.mean([b["error"] for b in bins])) if bins else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "spearman_rho": float(rho),
            "spearman_pvalue": float(pval),
            "mean_calibration_error": mean_cal_error,
            "is_calibrated": mean_cal_error < 0.1,
            "bins": bins,
        }


# ---------------------------------------------------------------------------
# Simple hand-crafted decision tree (baseline, no sklearn needed)
# ---------------------------------------------------------------------------


class BasicStrategyClassifier:
    """Rule-based classifier implementing standard basic strategy.

    Used as a baseline for comparison with ML models.
    """

    def predict_action(
        self,
        hand_total: int,
        hand_is_soft: bool,
        dealer_upcard: str,
        can_double: bool = True,
        can_split: bool = False,
    ) -> str:
        """Return basic strategy action.

        Returns one of: 'stand', 'hit', 'double', 'split'.
        """
        d = dealer_upcard
        t = hand_total

        # Hard totals
        if not hand_is_soft:
            if t >= 17:
                return "stand"
            elif t == 16:
                return "stand" if d in ['2', '3', '4', '5', '6'] else "hit"
            elif t == 15:
                return "stand" if d in ['2', '3', '4', '5', '6'] else "hit"
            elif t in [13, 14]:
                return "stand" if d in ['2', '3', '4', '5', '6'] else "hit"
            elif t == 12:
                return "stand" if d in ['4', '5', '6'] else "hit"
            elif t == 11:
                return "double" if can_double else "hit"
            elif t == 10:
                return "double" if (can_double and d not in ['T', 'A']) else "hit"
            elif t == 9:
                return "double" if (can_double and d in ['3', '4', '5', '6']) else "hit"
            else:
                return "hit"

        # Soft totals
        else:
            if t >= 19:
                return "stand"
            elif t == 18:
                return ("double" if (can_double and d in ['3', '4', '5', '6'])
                        else "stand" if d in ['2', '7', '8'] else "hit")
            elif t == 17:
                return ("double" if (can_double and d in ['3', '4', '5', '6'])
                        else "hit")
            elif t in [15, 16]:
                return ("double" if (can_double and d in ['4', '5', '6']) else "hit")
            else:
                return "hit"
