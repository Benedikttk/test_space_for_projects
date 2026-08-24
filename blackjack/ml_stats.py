"""ML statistics tracking for blackjack hands.

Tracks per-hand outcomes vs recommendations to build a learning model:

Per-hand features stored
-------------------------
- true_count, penetration, decks_remaining
- hand_type: 'soft' | 'hard' | 'pair'
- player_value, dealer_upcard
- num_aces_in_hand, num_tens_in_hand
- recommended_action, actual_action
- outcome: 'win' | 'loss' | 'push'
- ev (continuous)

Storage
-------
SQLite database with a rolling 10,000-hand window.

Model training
--------------
Uses scikit-learn RandomForestClassifier to predict hand outcome
probability.  Falls back gracefully if sklearn is unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hand_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    true_count      REAL    NOT NULL,
    penetration     REAL    NOT NULL,
    decks_remaining REAL    NOT NULL,
    hand_type       TEXT    NOT NULL,
    player_value    INTEGER NOT NULL,
    dealer_upcard   INTEGER NOT NULL,
    num_aces        INTEGER NOT NULL,
    num_tens        INTEGER NOT NULL,
    recommended     TEXT    NOT NULL,
    actual          TEXT    NOT NULL,
    outcome         TEXT    NOT NULL,
    ev              REAL    NOT NULL,
    bet_size        REAL    NOT NULL DEFAULT 0.0,
    net_result      REAL    NOT NULL DEFAULT 0.0
);
"""

_VALID_OUTCOMES = {"win", "loss", "push"}
_VALID_ACTIONS = {"hit", "stand", "double", "split", "surrender", "insurance"}


# ---------------------------------------------------------------------------
# Per-hand record
# ---------------------------------------------------------------------------

@dataclass
class HandRecord:
    """A single hand observation."""
    true_count: float
    penetration: float        # 0–1
    decks_remaining: float
    hand_type: str            # 'soft' | 'hard' | 'pair'
    player_value: int
    dealer_upcard: int        # 2–11 (11 = Ace)
    num_aces: int
    num_tens: int
    recommended: str
    actual: str
    outcome: str              # 'win' | 'loss' | 'push'
    ev: float
    bet_size: float = 0.0
    net_result: float = 0.0   # actual profit/loss in units


# ---------------------------------------------------------------------------
# Shoe-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class ShoeStats:
    """Aggregated statistics for a single shoe."""
    hands: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    total_ev: float = 0.0
    total_net: float = 0.0
    compliant_actions: int = 0  # actions matching recommendation
    total_bets: float = 0.0

    @property
    def win_rate(self) -> float:
        contested = self.wins + self.losses
        return self.wins / contested if contested > 0 else 0.0

    @property
    def ev_per_hand(self) -> float:
        return self.total_ev / self.hands if self.hands > 0 else 0.0

    @property
    def action_compliance(self) -> float:
        return self.compliant_actions / self.hands if self.hands > 0 else 0.0

    @property
    def roi(self) -> float:
        return self.total_net / self.total_bets if self.total_bets > 0 else 0.0


# ---------------------------------------------------------------------------
# ML Stats Tracker
# ---------------------------------------------------------------------------

class MLStatsTracker:
    """Track hand outcomes and train an ML model to predict results.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``data/ml_stats.db`` relative to the working directory.
    rolling_window:
        Number of most-recent hands to retain in the database.
    """

    def __init__(
        self,
        db_path: str = "data/ml_stats.db",
        rolling_window: int = 10_000,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rolling_window = rolling_window
        self._current_shoe = ShoeStats()
        self._model = None  # sklearn model loaded lazily
        self._con: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        con = self._get_con()
        con.execute(_SCHEMA)
        con.commit()

    def _get_con(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(
                str(self.db_path), check_same_thread=False
            )
        return self._con

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, record: HandRecord) -> None:
        """Persist one hand record and update shoe-level stats.

        Parameters
        ----------
        record:
            The completed hand observation.
        """
        if record.outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome {record.outcome!r}; "
                f"must be one of {_VALID_OUTCOMES}"
            )

        con = self._get_con()
        con.execute(
            """
            INSERT INTO hand_stats (
                timestamp, true_count, penetration, decks_remaining,
                hand_type, player_value, dealer_upcard,
                num_aces, num_tens,
                recommended, actual, outcome, ev,
                bet_size, net_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                record.true_count,
                record.penetration,
                record.decks_remaining,
                record.hand_type,
                record.player_value,
                record.dealer_upcard,
                record.num_aces,
                record.num_tens,
                record.recommended,
                record.actual,
                record.outcome,
                record.ev,
                record.bet_size,
                record.net_result,
            ),
        )
        con.execute(
            f"""
            DELETE FROM hand_stats
            WHERE id NOT IN (
                SELECT id FROM hand_stats ORDER BY id DESC LIMIT {self.rolling_window}
            )
            """
        )
        con.commit()

        # Update shoe aggregation
        self._current_shoe.hands += 1
        self._current_shoe.total_ev += record.ev
        self._current_shoe.total_bets += record.bet_size
        self._current_shoe.total_net += record.net_result
        if record.outcome == "win":
            self._current_shoe.wins += 1
        elif record.outcome == "loss":
            self._current_shoe.losses += 1
        else:
            self._current_shoe.pushes += 1
        if record.recommended == record.actual:
            self._current_shoe.compliant_actions += 1

    def reset_shoe(self) -> ShoeStats:
        """Finalise the current shoe and start a new one.

        Returns the completed shoe's stats.
        """
        completed = self._current_shoe
        self._current_shoe = ShoeStats()
        return completed

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def rolling_stats(self, last_n: int = 500) -> Dict:
        """Aggregate statistics over the last ``last_n`` hands.

        Returns a dict with keys: hands, win_rate, ev_per_hand,
        action_compliance, roi, total_net.
        """
        con = self._get_con()
        rows = con.execute(
            """
            SELECT recommended, actual, outcome, ev, bet_size, net_result
            FROM hand_stats
            ORDER BY id DESC
            LIMIT ?
            """,
            (last_n,),
        ).fetchall()

        if not rows:
            return {
                "hands": 0, "win_rate": 0.0, "ev_per_hand": 0.0,
                "action_compliance": 0.0, "roi": 0.0, "total_net": 0.0,
            }

        n = len(rows)
        wins = sum(1 for r in rows if r[2] == "win")
        losses = sum(1 for r in rows if r[2] == "loss")
        compliant = sum(1 for r in rows if r[0] == r[1])
        total_ev = sum(r[3] for r in rows)
        total_bets = sum(r[4] for r in rows)
        total_net = sum(r[5] for r in rows)
        contested = wins + losses

        return {
            "hands": n,
            "win_rate": wins / contested if contested > 0 else 0.0,
            "ev_per_hand": total_ev / n,
            "action_compliance": compliant / n,
            "roi": total_net / total_bets if total_bets > 0 else 0.0,
            "total_net": total_net,
        }

    @property
    def current_shoe_stats(self) -> ShoeStats:
        return self._current_shoe

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(record: HandRecord) -> List[float]:
        """Convert a HandRecord to a flat float feature vector.

        Feature order:
        [true_count, penetration, decks_remaining, player_value,
         dealer_upcard, num_aces, num_tens,
         is_soft (0/1), is_hard (0/1), is_pair (0/1)]
        """
        return [
            record.true_count,
            record.penetration,
            record.decks_remaining,
            float(record.player_value),
            float(record.dealer_upcard),
            float(record.num_aces),
            float(record.num_tens),
            1.0 if record.hand_type == "soft" else 0.0,
            1.0 if record.hand_type == "hard" else 0.0,
            1.0 if record.hand_type == "pair" else 0.0,
        ]

    # ------------------------------------------------------------------
    # ML model training
    # ------------------------------------------------------------------

    def train_model(self, min_samples: int = 100) -> Optional[Dict]:
        """Train a RandomForestClassifier on stored hands.

        Returns a dict with 'accuracy' and 'feature_importance' if
        training succeeds, or None if sklearn is unavailable or there
        are insufficient samples.

        Parameters
        ----------
        min_samples:
            Minimum number of hands required to attempt training.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            from sklearn.model_selection import cross_val_score  # type: ignore
            from sklearn.preprocessing import LabelEncoder  # type: ignore
        except ImportError:
            log.warning(
                "scikit-learn not installed — ML training unavailable. "
                "Install with: pip install scikit-learn"
            )
            return None

        con = self._get_con()
        rows = con.execute(
            """
            SELECT true_count, penetration, decks_remaining,
                   player_value, dealer_upcard, num_aces, num_tens,
                   hand_type, outcome
            FROM hand_stats
            ORDER BY id DESC
            LIMIT 10000
            """
        ).fetchall()

        if len(rows) < min_samples:
            log.info(
                "Insufficient data for ML training (%d < %d hands)",
                len(rows), min_samples,
            )
            return None

        features: List[List[float]] = []
        labels: List[str] = []
        for row in rows:
            tc, pen, dr, pv, du, na, nt, ht, outcome = row
            is_soft = 1.0 if ht == "soft" else 0.0
            is_hard = 1.0 if ht == "hard" else 0.0
            is_pair = 1.0 if ht == "pair" else 0.0
            features.append([tc, pen, dr, pv, du, na, nt, is_soft, is_hard, is_pair])
            labels.append(outcome)

        import numpy as np  # type: ignore

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        # Exclude pushes from win/loss classification
        mask = y != "push"
        X_clf = X[mask]
        y_clf = y[mask]

        if len(X_clf) < min_samples:
            return None

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

        scores = cross_val_score(clf, X_clf, y_clf, cv=3, scoring="accuracy")
        clf.fit(X_clf, y_clf)
        self._model = clf

        feature_names = [
            "true_count", "penetration", "decks_remaining",
            "player_value", "dealer_upcard", "num_aces", "num_tens",
            "is_soft", "is_hard", "is_pair",
        ]
        importance = {
            name: float(imp)
            for name, imp in zip(feature_names, clf.feature_importances_)
        }

        log.info(
            "ML model trained on %d hands, CV accuracy=%.3f",
            len(X_clf), scores.mean(),
        )
        return {
            "accuracy": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "n_samples": len(X_clf),
            "feature_importance": importance,
        }

    def predict_outcome(self, record: HandRecord) -> Optional[Dict[str, float]]:
        """Predict win/loss probability for a hand using the trained model.

        Returns a dict with keys 'win' and 'loss' (probabilities), or
        None if no model has been trained yet.
        """
        if self._model is None:
            return None
        try:
            import numpy as np  # type: ignore

            features = np.array(
                [self.extract_features(record)], dtype=np.float32
            )
            proba = self._model.predict_proba(features)[0]
            classes = list(self._model.classes_)
            return {cls: float(p) for cls, p in zip(classes, proba)}
        except Exception as exc:
            log.warning("Prediction failed: %s", exc)
            return None
