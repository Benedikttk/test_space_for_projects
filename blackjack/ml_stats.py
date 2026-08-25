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
- sequence_last_5_outcomes, sequence_last_5_true_counts
- shoe_rank_distribution (10-element vector)
- num_active_players
- shoe_id

Storage
-------
SQLite database with a rolling 10,000-hand window.

Model training
--------------
Uses GradientBoostingClassifier (falls back to RandomForestClassifier)
to predict hand outcome probability, plus several additional models.
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
    net_result      REAL    NOT NULL DEFAULT 0.0,
    sequence_last_5_outcomes    TEXT NOT NULL DEFAULT '[]',
    sequence_last_5_true_counts TEXT NOT NULL DEFAULT '[]',
    shoe_rank_distribution      TEXT NOT NULL DEFAULT '[]',
    num_active_players          INTEGER NOT NULL DEFAULT 0,
    shoe_id                     INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after initial schema; migrated lazily.
_MIGRATION_COLUMNS = [
    ("sequence_last_5_outcomes",    "TEXT NOT NULL DEFAULT '[]'"),
    ("sequence_last_5_true_counts", "TEXT NOT NULL DEFAULT '[]'"),
    ("shoe_rank_distribution",      "TEXT NOT NULL DEFAULT '[]'"),
    ("num_active_players",          "INTEGER NOT NULL DEFAULT 0"),
    ("shoe_id",                     "INTEGER NOT NULL DEFAULT 0"),
]

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
    sequence_last_5_outcomes: List[str] = field(default_factory=list)
    sequence_last_5_true_counts: List[float] = field(default_factory=list)
    shoe_rank_distribution: List[float] = field(default_factory=list)
    num_active_players: int = 0
    shoe_id: int = 0


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
        self._shoe_counter: int = 0
        self._model = None          # win-probability model
        self._bet_model = None      # bet-sizing regression model
        self._shoe_quality_model = None
        self._deviation_model = None
        self._sequence_model = None
        self._con: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        con = self._get_con()
        con.execute(_SCHEMA)
        # Migrate older databases that lack the new columns.
        existing = {
            row[1]
            for row in con.execute("PRAGMA table_info(hand_stats)").fetchall()
        }
        for col_name, col_def in _MIGRATION_COLUMNS:
            if col_name not in existing:
                con.execute(
                    f"ALTER TABLE hand_stats ADD COLUMN {col_name} {col_def}"
                )
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
                bet_size, net_result,
                sequence_last_5_outcomes, sequence_last_5_true_counts,
                shoe_rank_distribution, num_active_players, shoe_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(record.sequence_last_5_outcomes),
                json.dumps(record.sequence_last_5_true_counts),
                json.dumps(record.shoe_rank_distribution),
                record.num_active_players,
                record.shoe_id,
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
        self._shoe_counter += 1
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
        """Train a GradientBoostingClassifier (fallback: RandomForest) on stored hands.

        Returns a dict with 'accuracy', 'accuracy_std', 'n_samples',
        'feature_importance', and 'model_type' if training succeeds,
        or None if sklearn is unavailable or there are insufficient samples.

        Parameters
        ----------
        min_samples:
            Minimum number of hands required to attempt training.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
            from sklearn.model_selection import cross_val_score  # type: ignore
            _model_type = "GradientBoosting"
            def _make_clf():
                return GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42
                )
        except ImportError:
            try:
                from sklearn.ensemble import RandomForestClassifier  # type: ignore
                from sklearn.model_selection import cross_val_score  # type: ignore
                _model_type = "RandomForest"
                def _make_clf():
                    return RandomForestClassifier(
                        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                    )
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
                   hand_type, outcome,
                   shoe_rank_distribution, num_active_players
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

        import numpy as np  # type: ignore

        features: List[List[float]] = []
        labels: List[str] = []
        for row in rows:
            tc, pen, dr, pv, du, na, nt, ht, outcome, srd_json, nap = row
            is_soft = 1.0 if ht == "soft" else 0.0
            is_hard = 1.0 if ht == "hard" else 0.0
            is_pair = 1.0 if ht == "pair" else 0.0
            base = [tc, pen, dr, pv, du, na, nt, is_soft, is_hard, is_pair, float(nap or 0)]
            try:
                srd = json.loads(srd_json) if srd_json else []
            except (TypeError, json.JSONDecodeError):
                srd = []
            if len(srd) == 10:
                base.extend(float(v) for v in srd)
            else:
                base.extend([0.0] * 10)
            features.append(base)
            labels.append(outcome)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        mask = y != "push"
        X_clf = X[mask]
        y_clf = y[mask]

        if len(X_clf) < min_samples:
            return None

        clf = _make_clf()
        scores = cross_val_score(clf, X_clf, y_clf, cv=3, scoring="accuracy")
        clf.fit(X_clf, y_clf)
        self._model = clf

        base_names = [
            "true_count", "penetration", "decks_remaining",
            "player_value", "dealer_upcard", "num_aces", "num_tens",
            "is_soft", "is_hard", "is_pair", "num_active_players",
        ]
        srd_names = [f"shoe_rank_{i}" for i in range(10)]
        feature_names = base_names + srd_names

        importance = {
            name: float(imp)
            for name, imp in zip(feature_names, clf.feature_importances_)
        }

        log.info(
            "ML model (%s) trained on %d hands, CV accuracy=%.3f",
            _model_type, len(X_clf), scores.mean(),
        )
        return {
            "accuracy": float(scores.mean()),
            "accuracy_std": float(scores.std()),
            "n_samples": len(X_clf),
            "feature_importance": importance,
            "model_type": _model_type,
        }

    # ------------------------------------------------------------------
    # Bet sizing model
    # ------------------------------------------------------------------

    def train_bet_sizing_model(self, min_samples: int = 50) -> Optional[Dict]:
        """Train an XGBRegressor (fallback: GradientBoostingRegressor) to
        predict the optimal bet multiplier from shoe state + session history.

        Returns {"rmse": float, "n_samples": int, "model_type": str} or None.
        """
        try:
            from xgboost import XGBRegressor  # type: ignore
            _model_type = "XGBoostRegressor"
            def _make_reg():
                return XGBRegressor(
                    n_estimators=100, max_depth=4,
                    random_state=42, verbosity=0
                )
        except ImportError:
            try:
                from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
                _model_type = "GradientBoostingRegressor"
                def _make_reg():
                    return GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, random_state=42
                    )
            except ImportError:
                log.warning("Neither xgboost nor scikit-learn available.")
                return None

        import numpy as np  # type: ignore

        con = self._get_con()
        rows = con.execute(
            """
            SELECT true_count, penetration, decks_remaining,
                   player_value, dealer_upcard,
                   hand_type, num_active_players,
                   bet_size, net_result
            FROM hand_stats
            WHERE bet_size > 0
            ORDER BY id DESC
            LIMIT 10000
            """
        ).fetchall()

        if len(rows) < min_samples:
            return None

        features, targets = [], []
        for row in rows:
            tc, pen, dr, pv, du, ht, nap, bs, nr = row
            is_soft = 1.0 if ht == "soft" else 0.0
            is_hard = 1.0 if ht == "hard" else 0.0
            is_pair = 1.0 if ht == "pair" else 0.0
            features.append([tc, pen, dr, float(pv), float(du),
                              is_soft, is_hard, is_pair, float(nap or 0)])
            ratio = float(nr) / float(bs)
            targets.append(max(-3.0, min(3.0, ratio)))

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        reg = _make_reg()
        reg.fit(X, y)
        self._bet_model = reg

        preds = reg.predict(X)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        log.info("Bet sizing model (%s) trained; RMSE=%.4f", _model_type, rmse)
        return {"rmse": rmse, "n_samples": len(y), "model_type": _model_type}

    def predict_bet_multiplier(self, record: HandRecord) -> Optional[float]:
        """Predict bet multiplier for a hand. Returns None if not trained."""
        if self._bet_model is None:
            return None
        try:
            import numpy as np  # type: ignore
            ht = record.hand_type
            feat = np.array([[
                record.true_count, record.penetration, record.decks_remaining,
                float(record.player_value), float(record.dealer_upcard),
                1.0 if ht == "soft" else 0.0,
                1.0 if ht == "hard" else 0.0,
                1.0 if ht == "pair" else 0.0,
                float(record.num_active_players),
            ]], dtype=np.float32)
            return float(self._bet_model.predict(feat)[0])
        except Exception as exc:
            log.warning("predict_bet_multiplier failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Shoe quality model
    # ------------------------------------------------------------------

    def train_shoe_quality_model(self, min_shoes: int = 10) -> Optional[Dict]:
        """Train logistic regression on shoe rank distribution to predict
        whether a shoe will be net-positive.

        Returns {"auc": float, "n_shoes": int} or None.
        """
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.metrics import roc_auc_score  # type: ignore
        except ImportError:
            log.warning("scikit-learn not installed — shoe quality model unavailable.")
            return None

        import numpy as np  # type: ignore

        con = self._get_con()
        rows = con.execute(
            """
            SELECT shoe_id, shoe_rank_distribution, net_result
            FROM hand_stats
            ORDER BY shoe_id, id
            """
        ).fetchall()

        shoe_data: Dict[int, Dict] = {}
        for shoe_id, srd_json, nr in rows:
            if shoe_id not in shoe_data:
                shoe_data[shoe_id] = {"net": 0.0, "srd": None}
            shoe_data[shoe_id]["net"] += float(nr)
            if shoe_data[shoe_id]["srd"] is None:
                try:
                    srd = json.loads(srd_json) if srd_json else []
                    if len(srd) == 10:
                        shoe_data[shoe_id]["srd"] = srd
                except (TypeError, json.JSONDecodeError):
                    pass

        X, y = [], []
        for info in shoe_data.values():
            if info["srd"] is None:
                continue
            X.append([float(v) for v in info["srd"]])
            y.append(1 if info["net"] > 0 else 0)

        if len(X) < min_shoes:
            return None

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y)

        clf = LogisticRegression(random_state=42, max_iter=500)
        clf.fit(X_arr, y_arr)
        self._shoe_quality_model = clf

        if len(np.unique(y_arr)) < 2:
            auc = 0.5
        else:
            proba = clf.predict_proba(X_arr)[:, 1]
            auc = float(roc_auc_score(y_arr, proba))

        log.info("Shoe quality model trained; AUC=%.4f n_shoes=%d", auc, len(X))
        return {"auc": auc, "n_shoes": len(X)}

    def predict_shoe_quality(self, shoe_rank_dist: List[float]) -> Optional[float]:
        """Return P(positive shoe) for given rank distribution, or None."""
        if self._shoe_quality_model is None:
            return None
        try:
            import numpy as np  # type: ignore
            X = np.array([shoe_rank_dist], dtype=np.float32)
            return float(self._shoe_quality_model.predict_proba(X)[0, 1])
        except Exception as exc:
            log.warning("predict_shoe_quality failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Action deviation detector
    # ------------------------------------------------------------------

    def train_action_deviation_detector(self, min_samples: int = 50) -> Optional[Dict]:
        """Train a classifier to detect whether an action was a mistake.

        Returns {"accuracy": float, "n_samples": int} or None.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
            from sklearn.model_selection import cross_val_score  # type: ignore
            _model_type = "GradientBoosting"
            def _make_clf():
                return GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42
                )
        except ImportError:
            try:
                from sklearn.ensemble import RandomForestClassifier  # type: ignore
                from sklearn.model_selection import cross_val_score  # type: ignore
                _model_type = "RandomForest"
                def _make_clf():
                    return RandomForestClassifier(
                        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                    )
            except ImportError:
                log.warning("scikit-learn not installed.")
                return None

        import numpy as np  # type: ignore

        con = self._get_con()
        rows = con.execute(
            """
            SELECT true_count, penetration, decks_remaining,
                   player_value, dealer_upcard, num_aces, num_tens,
                   hand_type, recommended, actual
            FROM hand_stats
            ORDER BY id DESC
            LIMIT 10000
            """
        ).fetchall()

        if len(rows) < min_samples:
            return None

        features, labels = [], []
        for row in rows:
            tc, pen, dr, pv, du, na, nt, ht, rec, act = row
            is_soft = 1.0 if ht == "soft" else 0.0
            is_hard = 1.0 if ht == "hard" else 0.0
            is_pair = 1.0 if ht == "pair" else 0.0
            features.append([tc, pen, dr, float(pv), float(du),
                              float(na), float(nt), is_soft, is_hard, is_pair])
            labels.append(1 if rec != act else 0)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            return None

        clf = _make_clf()
        scores = cross_val_score(clf, X, y, cv=min(3, len(X) // 10 + 1), scoring="accuracy")
        clf.fit(X, y)
        self._deviation_model = clf

        log.info("Deviation model (%s) trained; accuracy=%.3f", _model_type, scores.mean())
        return {"accuracy": float(scores.mean()), "n_samples": len(y)}

    def predict_action_deviation(
        self, record: HandRecord, action: str
    ) -> Optional[float]:
        """Return P(action is a deviation/mistake) or None."""
        if self._deviation_model is None:
            return None
        try:
            import numpy as np  # type: ignore
            ht = record.hand_type
            feat = np.array([[
                record.true_count, record.penetration, record.decks_remaining,
                float(record.player_value), float(record.dealer_upcard),
                float(record.num_aces), float(record.num_tens),
                1.0 if ht == "soft" else 0.0,
                1.0 if ht == "hard" else 0.0,
                1.0 if ht == "pair" else 0.0,
            ]], dtype=np.float32)
            proba = self._deviation_model.predict_proba(feat)[0]
            classes = list(self._deviation_model.classes_)
            idx = classes.index(1) if 1 in classes else -1
            return float(proba[idx]) if idx >= 0 else None
        except Exception as exc:
            log.warning("predict_action_deviation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Sequence model
    # ------------------------------------------------------------------

    def train_sequence_model(self, min_samples: int = 50) -> Optional[Dict]:
        """Train a model using last-5 outcome/count sequences to predict win.

        Returns {"accuracy": float, "n_samples": int} or None.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
            from sklearn.model_selection import cross_val_score  # type: ignore
        except ImportError:
            log.warning("scikit-learn not installed — sequence model unavailable.")
            return None

        import numpy as np  # type: ignore

        con = self._get_con()
        rows = con.execute(
            """
            SELECT outcome, sequence_last_5_outcomes, sequence_last_5_true_counts
            FROM hand_stats
            ORDER BY id DESC
            LIMIT 10000
            """
        ).fetchall()

        features, labels = [], []
        for outcome, seq_out_json, seq_tc_json in rows:
            if outcome == "push":
                continue
            try:
                seq_out = json.loads(seq_out_json) if seq_out_json else []
                seq_tc = json.loads(seq_tc_json) if seq_tc_json else []
            except (TypeError, json.JSONDecodeError):
                seq_out, seq_tc = [], []
            if len(seq_out) < 5 or len(seq_tc) < 5:
                continue
            out_feats = [1.0 if o == "win" else 0.0 for o in seq_out[:5]]
            tc_feats = [float(v) for v in seq_tc[:5]]
            features.append(out_feats + tc_feats)
            labels.append(outcome)

        if len(features) < min_samples:
            return None

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        clf.fit(X, y)
        self._sequence_model = clf

        log.info("Sequence model trained; accuracy=%.3f", scores.mean())
        return {"accuracy": float(scores.mean()), "n_samples": len(y)}

    def predict_win_with_sequence(
        self, record: HandRecord
    ) -> Optional[Dict[str, float]]:
        """Return {"win": p, "loss": 1-p} using sequence features, or None."""
        if self._sequence_model is None:
            return None
        try:
            import numpy as np  # type: ignore
            seq_out = (record.sequence_last_5_outcomes or [])[:5]
            seq_tc = (record.sequence_last_5_true_counts or [])[:5]
            if len(seq_out) < 5 or len(seq_tc) < 5:
                return None
            out_feats = [1.0 if o == "win" else 0.0 for o in seq_out]
            tc_feats = [float(v) for v in seq_tc]
            feat = np.array([out_feats + tc_feats], dtype=np.float32)
            proba = self._sequence_model.predict_proba(feat)[0]
            classes = list(self._sequence_model.classes_)
            result = {cls: float(p) for cls, p in zip(classes, proba)}
            if "win" not in result:
                return None
            win_p = result["win"]
            return {"win": win_p, "loss": 1.0 - win_p}
        except Exception as exc:
            log.warning("predict_win_with_sequence failed: %s", exc)
            return None

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
