"""Model Monitoring and Observability for the Blackjack Platform.

Tracks model performance over time, detects drift, triggers retraining
alerts, and provides A/B testing infrastructure.

Follows the principles of:
- MLOps monitoring best practices
- Statistical process control (SPC)
- Concept drift detection (DDM, ADWIN-inspired)
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Monitoring data structures
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """A single prediction record for monitoring."""
    record_id: str
    timestamp: float
    model_version: str
    predicted_ev: float
    predicted_action: str
    actual_outcome: Optional[float] = None
    hand_features_hash: Optional[int] = None
    latency_ms: float = 0.0


@dataclass
class DriftAlert:
    """An alert triggered by statistical drift detection."""
    alert_type: str     # 'accuracy_drop', 'distribution_shift', 'latency_spike'
    severity: str       # 'warning', 'critical'
    metric_name: str
    current_value: float
    baseline_value: float
    p_value: float
    timestamp: float
    description: str


@dataclass
class ABTestResult:
    """Result of an A/B test comparing two model versions."""
    model_a: str
    model_b: str
    n_samples_a: int
    n_samples_b: int
    mean_ev_a: float
    mean_ev_b: float
    t_statistic: float
    p_value: float
    winner: str
    is_significant: bool
    effect_size: float


# ---------------------------------------------------------------------------
# Main monitoring class
# ---------------------------------------------------------------------------


class ModelMonitoring:
    """Real-time model performance monitoring.

    Tracks:
    - Prediction accuracy (MAE between predicted and actual EV)
    - Calibration (whether predicted probabilities match realised rates)
    - Latency (P50, P95, P99)
    - Data drift (KS test on feature distributions)
    - A/B testing (compare model versions)

    Parameters
    ----------
    window_size:
        Rolling window for metric computation.
    accuracy_threshold:
        Alert if MAE exceeds this value.
    latency_p99_threshold_ms:
        Alert if P99 latency exceeds this.
    """

    def __init__(
        self,
        window_size: int = 1000,
        accuracy_threshold: float = 0.10,
        latency_p99_threshold_ms: float = 100.0,
        model_version: str = "1.0.0",
    ) -> None:
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold = latency_p99_threshold_ms
        self.model_version = model_version

        # Rolling windows
        self._predictions: Deque[PredictionRecord] = deque(maxlen=window_size)
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._maes: Deque[float] = deque(maxlen=window_size)
        self._alerts: List[DriftAlert] = []

        # A/B test storage
        self._ab_predictions: Dict[str, Deque[float]] = {}

        # Baseline metrics (set after warmup)
        self._baseline_mae: Optional[float] = None
        self._baseline_feature_dist: Optional[np.ndarray] = None

        # Audit trail
        self._audit_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Record predictions and outcomes
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        predicted_ev: float,
        predicted_action: str,
        latency_ms: float = 0.0,
        model_version: Optional[str] = None,
    ) -> str:
        """Log a prediction. Returns the record ID."""
        record_id = str(uuid.uuid4())
        record = PredictionRecord(
            record_id=record_id,
            timestamp=time.time(),
            model_version=model_version or self.model_version,
            predicted_ev=predicted_ev,
            predicted_action=predicted_action,
            latency_ms=latency_ms,
        )
        self._predictions.append(record)
        self._latencies.append(latency_ms)
        self._audit_log.append({
            "type": "prediction",
            "record_id": record_id,
            "timestamp": record.timestamp,
            "predicted_ev": predicted_ev,
        })
        return record_id

    def record_outcome(
        self,
        record_id: str,
        actual_outcome: float,
    ) -> None:
        """Update a prediction record with the actual outcome."""
        for rec in self._predictions:
            if rec.record_id == record_id:
                rec.actual_outcome = actual_outcome
                mae = abs(rec.predicted_ev - actual_outcome)
                self._maes.append(mae)
                self._audit_log.append({
                    "type": "outcome",
                    "record_id": record_id,
                    "actual": actual_outcome,
                    "mae": mae,
                })
                break

        self._check_accuracy_drift()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def current_mae(self) -> Optional[float]:
        """Current rolling MAE between predicted and actual EV."""
        resolved = [r for r in self._predictions if r.actual_outcome is not None]
        if not resolved:
            return None
        errors = [abs(r.predicted_ev - r.actual_outcome) for r in resolved]
        return float(np.mean(errors))

    def latency_percentiles(self) -> Dict[str, float]:
        """Latency percentiles from the rolling window."""
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.array(list(self._latencies))
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
        }

    def calibration_check(self) -> Dict[str, float]:
        """Check if predicted EVs are calibrated to actual outcomes."""
        resolved = [r for r in self._predictions if r.actual_outcome is not None]
        if len(resolved) < 10:
            return {"status": "insufficient_data", "n": len(resolved)}

        pred = np.array([r.predicted_ev for r in resolved])
        actual = np.array([r.actual_outcome for r in resolved])

        # Regression slope should be ≈1 for good calibration
        slope, intercept, r, p, se = stats.linregress(pred, actual)
        mae = float(np.mean(np.abs(pred - actual)))
        bias = float(np.mean(pred - actual))

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r ** 2),
            "mae": mae,
            "bias": bias,
            "is_calibrated": abs(slope - 1.0) < 0.2 and abs(bias) < 0.05,
            "n": len(resolved),
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def _check_accuracy_drift(self) -> None:
        """Internal: detect accuracy drift and emit alerts."""
        mae = self.current_mae()
        if mae is None:
            return

        # Set baseline after warmup
        if len(self._maes) == 50 and self._baseline_mae is None:
            self._baseline_mae = mae

        if self._baseline_mae is not None and mae > self._accuracy_threshold:
            if mae > self._baseline_mae * 1.5:
                self._alerts.append(DriftAlert(
                    alert_type="accuracy_drop",
                    severity="critical",
                    metric_name="mae",
                    current_value=mae,
                    baseline_value=self._baseline_mae,
                    p_value=0.0,  # threshold-based
                    timestamp=time.time(),
                    description=(
                        f"MAE {mae:.4f} is {mae/self._baseline_mae:.1f}× baseline. "
                        "Consider retraining."
                    ),
                ))

    @property
    def _accuracy_threshold(self) -> float:
        return self.accuracy_threshold

    def ks_feature_drift(
        self,
        reference_features: Sequence[float],
        current_features: Sequence[float],
        feature_name: str = "feature",
    ) -> DriftAlert | None:
        """KS test for distribution drift on a single feature."""
        ref = np.asarray(reference_features, dtype=float)
        cur = np.asarray(current_features, dtype=float)
        if len(ref) < 10 or len(cur) < 10:
            return None

        ks_stat, p_value = stats.ks_2samp(ref, cur)

        if p_value < 0.05:
            alert = DriftAlert(
                alert_type="distribution_shift",
                severity="warning" if p_value > 0.01 else "critical",
                metric_name=feature_name,
                current_value=float(np.mean(cur)),
                baseline_value=float(np.mean(ref)),
                p_value=float(p_value),
                timestamp=time.time(),
                description=(
                    f"KS drift detected on '{feature_name}': "
                    f"stat={ks_stat:.4f}, p={p_value:.4f}"
                ),
            )
            self._alerts.append(alert)
            return alert
        return None

    def check_latency_drift(self) -> Optional[DriftAlert]:
        """Check if P99 latency exceeds threshold."""
        pct = self.latency_percentiles()
        if pct["p99"] > self.latency_threshold:
            alert = DriftAlert(
                alert_type="latency_spike",
                severity="warning",
                metric_name="latency_p99_ms",
                current_value=pct["p99"],
                baseline_value=self.latency_threshold,
                p_value=0.0,
                timestamp=time.time(),
                description=f"P99 latency {pct['p99']:.1f}ms > threshold {self.latency_threshold:.0f}ms",
            )
            self._alerts.append(alert)
            return alert
        return None

    # ------------------------------------------------------------------
    # A/B testing
    # ------------------------------------------------------------------

    def ab_test_record(self, model_name: str, ev_outcome: float) -> None:
        """Record a realized EV for A/B testing."""
        if model_name not in self._ab_predictions:
            self._ab_predictions[model_name] = deque(maxlen=self.window_size)
        self._ab_predictions[model_name].append(ev_outcome)

    def ab_test_result(
        self, model_a: str, model_b: str, alpha: float = 0.05
    ) -> Optional[ABTestResult]:
        """Run Welch's t-test to compare two model versions."""
        data_a = list(self._ab_predictions.get(model_a, []))
        data_b = list(self._ab_predictions.get(model_b, []))
        if len(data_a) < 10 or len(data_b) < 10:
            return None

        arr_a = np.array(data_a)
        arr_b = np.array(data_b)
        t_stat, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False)

        # Cohen's d
        pooled_std = math.sqrt((arr_a.var(ddof=1) + arr_b.var(ddof=1)) / 2)
        effect_d = abs(float(np.mean(arr_a) - np.mean(arr_b))) / max(pooled_std, 1e-10)

        winner = model_a if np.mean(arr_a) > np.mean(arr_b) else model_b
        if p_value >= alpha:
            winner = "no_winner"

        return ABTestResult(
            model_a=model_a,
            model_b=model_b,
            n_samples_a=len(data_a),
            n_samples_b=len(data_b),
            mean_ev_a=float(np.mean(arr_a)),
            mean_ev_b=float(np.mean(arr_b)),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            winner=winner,
            is_significant=p_value < alpha,
            effect_size=effect_d,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def active_alerts(self) -> List[DriftAlert]:
        """Return recent unresolved alerts (last 10)."""
        return self._alerts[-10:]

    def health_report(self) -> Dict[str, object]:
        """Comprehensive health report."""
        mae = self.current_mae()
        lat = self.latency_percentiles()
        cal = self.calibration_check()
        return {
            "model_version": self.model_version,
            "total_predictions": len(self._predictions),
            "rolling_mae": mae,
            "is_accurate": mae is None or mae < self.accuracy_threshold,
            "latency": lat,
            "latency_ok": lat["p99"] < self.latency_threshold,
            "calibration": cal,
            "n_alerts": len(self._alerts),
            "critical_alerts": [a for a in self._alerts if a.severity == "critical"],
        }

    def audit_log(self, limit: int = 100) -> List[Dict]:
        """Return the most recent audit log entries."""
        return self._audit_log[-limit:]
