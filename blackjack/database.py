"""Enterprise Database Layer for Blackjack Platform.

Provides a SQLite-backed persistence layer with:
- Hand history storage with proper indexing
- Query interface for analytics
- Time-series partitioning support
- Batch insert for performance
- Data retention policies

For production, swap SQLite with PostgreSQL by changing the connection string.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    session_id      TEXT    NOT NULL,
    hand_number     INTEGER NOT NULL,
    hand_total      INTEGER,
    hand_is_soft    INTEGER,   -- boolean
    dealer_upcard   TEXT,
    true_count      REAL,
    running_count   INTEGER,
    deck_penetration REAL,
    observation_ratio REAL,
    recommended_action TEXT,
    predicted_ev    REAL,
    actual_outcome  REAL,
    bet_size        REAL,
    bankroll        REAL,
    features_json   TEXT,      -- serialised feature vector
    model_version   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT UNIQUE NOT NULL,
    start_time      REAL,
    end_time        REAL,
    n_decks         INTEGER,
    table_rules     TEXT,      -- JSON
    starting_bankroll REAL,
    ending_bankroll REAL,
    total_hands     INTEGER,
    win_rate        REAL,
    total_ev        REAL
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    alert_type      TEXT,
    severity        TEXT,
    message         TEXT,
    session_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_hands_true_count    ON hands(true_count);
CREATE INDEX IF NOT EXISTS idx_hands_penetration   ON hands(deck_penetration);
CREATE INDEX IF NOT EXISTS idx_hands_session       ON hands(session_id);
CREATE INDEX IF NOT EXISTS idx_hands_timestamp     ON hands(timestamp);
CREATE INDEX IF NOT EXISTS idx_hands_outcome       ON hands(actual_outcome);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class HandRecord:
    """One hand's data for storage."""
    session_id: str
    hand_number: int
    hand_total: int
    hand_is_soft: bool
    dealer_upcard: str
    true_count: float
    running_count: int
    deck_penetration: float
    observation_ratio: float
    recommended_action: str
    predicted_ev: float
    bet_size: float
    bankroll: float
    actual_outcome: Optional[float] = None
    model_version: str = "1.0.0"
    features_json: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionRecord:
    """Session-level summary."""
    session_id: str
    n_decks: int
    table_rules: Dict
    starting_bankroll: float
    start_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class EnterpriseDatabase:
    """SQLite-backed hand history and analytics database.

    Designed to scale to millions of hands with:
    - Proper indexing on (true_count, penetration, outcome)
    - Batch inserts (1000 records at a time)
    - Query helpers for analytics

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Use ':memory:' for in-memory testing.
    batch_size:
        Number of records to buffer before flushing to disk.
    retention_days:
        Delete hands older than this many days (0 = keep forever).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        batch_size: int = 1000,
        retention_days: int = 0,
    ) -> None:
        self.db_path = db_path
        self.batch_size = batch_size
        self.retention_days = retention_days
        self._pending: List[HandRecord] = []
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialise the database schema."""
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            yield self._conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, record: SessionRecord) -> None:
        """Create a new session record."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (session_id, start_time, n_decks, table_rules, starting_bankroll)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    record.session_id,
                    record.start_time,
                    record.n_decks,
                    json.dumps(record.table_rules),
                    record.starting_bankroll,
                ),
            )

    def end_session(
        self,
        session_id: str,
        ending_bankroll: float,
        total_hands: int,
        win_rate: float,
        total_ev: float,
    ) -> None:
        """Update session summary at end."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE sessions SET
                   end_time=?, ending_bankroll=?, total_hands=?, win_rate=?, total_ev=?
                   WHERE session_id=?""",
                (time.time(), ending_bankroll, total_hands, win_rate, total_ev, session_id),
            )

    # ------------------------------------------------------------------
    # Hand storage
    # ------------------------------------------------------------------

    def insert_hand(self, hand: HandRecord) -> None:
        """Buffer a hand record; flush when batch is full."""
        self._pending.append(hand)
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Write all pending records to the database."""
        if not self._pending:
            return
        rows = [
            (
                h.timestamp, h.session_id, h.hand_number,
                h.hand_total, int(h.hand_is_soft), h.dealer_upcard,
                h.true_count, h.running_count, h.deck_penetration,
                h.observation_ratio, h.recommended_action, h.predicted_ev,
                h.actual_outcome, h.bet_size, h.bankroll,
                h.features_json, h.model_version,
            )
            for h in self._pending
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO hands
                   (timestamp, session_id, hand_number, hand_total, hand_is_soft,
                    dealer_upcard, true_count, running_count, deck_penetration,
                    observation_ratio, recommended_action, predicted_ev,
                    actual_outcome, bet_size, bankroll, features_json, model_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        self._pending.clear()

    def update_outcome(
        self, session_id: str, hand_number: int, actual_outcome: float
    ) -> None:
        """Update the actual outcome of a previously stored hand."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE hands SET actual_outcome=? WHERE session_id=? AND hand_number=?",
                (actual_outcome, session_id, hand_number),
            )

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def ev_by_true_count(self, min_hands: int = 10) -> List[Dict]:
        """Mean EV grouped by true count bucket.

        Returns rows of {true_count_bucket, mean_ev, n_hands}.
        Only includes buckets with ≥ min_hands resolved hands.
        """
        self.flush()
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT
                       CAST(ROUND(true_count) AS INTEGER) AS tc_bucket,
                       AVG(actual_outcome - predicted_ev) AS mean_residual,
                       AVG(actual_outcome) AS mean_outcome,
                       AVG(predicted_ev) AS mean_predicted_ev,
                       COUNT(*) AS n_hands
                   FROM hands
                   WHERE actual_outcome IS NOT NULL
                   GROUP BY tc_bucket
                   HAVING n_hands >= ?
                   ORDER BY tc_bucket""",
                (min_hands,),
            ).fetchall()
        return [dict(r) for r in rows]

    def calibration_query(
        self, n_bins: int = 10, session_id: Optional[str] = None
    ) -> List[Dict]:
        """Binned calibration data (predicted vs actual EV)."""
        self.flush()
        filter_sql = "WHERE actual_outcome IS NOT NULL"
        params: list = []
        if session_id:
            filter_sql += " AND session_id=?"
            params.append(session_id)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT predicted_ev, actual_outcome
                    FROM hands {filter_sql}""",
                params,
            ).fetchall()

        if not rows:
            return []

        import numpy as np
        pred = [r[0] for r in rows]
        actual = [r[1] for r in rows]
        pred_arr = sorted(set(pred))
        step = max(1, len(pred_arr) // n_bins)
        bins = pred_arr[::step]

        result = []
        for i, low in enumerate(bins):
            high = bins[i + 1] if i + 1 < len(bins) else float('inf')
            bucket_pred = [p for p, a in zip(pred, actual) if low <= p < high]
            bucket_actual = [a for p, a in zip(pred, actual) if low <= p < high]
            if len(bucket_pred) > 0:
                result.append({
                    "bin_low": low,
                    "bin_high": high,
                    "mean_predicted": float(sum(bucket_pred) / len(bucket_pred)),
                    "mean_actual": float(sum(bucket_actual) / len(bucket_actual)),
                    "n": len(bucket_pred),
                })
        return result

    def win_rate_by_penetration(self, session_id: Optional[str] = None) -> List[Dict]:
        """Win rate grouped by deck penetration bucket."""
        self.flush()
        filter_sql = "WHERE actual_outcome IS NOT NULL"
        params: list = []
        if session_id:
            filter_sql += " AND session_id=?"
            params.append(session_id)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT
                       CAST(deck_penetration * 4 AS INTEGER) AS pen_bucket,
                       AVG(CASE WHEN actual_outcome > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                       COUNT(*) AS n_hands
                   FROM hands {filter_sql}
                   GROUP BY pen_bucket
                   ORDER BY pen_bucket""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def session_summary(self, session_id: str) -> Optional[Dict]:
        """Return summary for a specific session."""
        self.flush()
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_old_data(self) -> int:
        """Delete records older than retention_days. Returns rows deleted."""
        if self.retention_days <= 0:
            return 0
        cutoff = time.time() - self.retention_days * 86400
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM hands WHERE timestamp < ?", (cutoff,))
            return cur.rowcount

    def total_hands(self) -> int:
        """Total number of stored hand records."""
        self.flush()
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM hands").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Flush pending records and close the connection."""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
