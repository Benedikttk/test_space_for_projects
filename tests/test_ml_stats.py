"""Tests for blackjack/ml_stats.py."""
from __future__ import annotations

import pytest

from blackjack.ml_stats import HandRecord, MLStatsTracker, ShoeStats


def _make_record(**kwargs) -> HandRecord:
    defaults = dict(
        true_count=1.5,
        penetration=0.6,
        decks_remaining=3.2,
        hand_type="hard",
        player_value=15,
        dealer_upcard=10,
        num_aces=0,
        num_tens=1,
        recommended="stand",
        actual="stand",
        outcome="loss",
        ev=-0.12,
        bet_size=10.0,
        net_result=-10.0,
    )
    defaults.update(kwargs)
    return HandRecord(**defaults)


class TestHandRecord:
    def test_default_fields(self):
        rec = _make_record()
        assert rec.hand_type == "hard"
        assert rec.outcome == "loss"


class TestMLStatsTracker:
    def test_record_and_rolling_stats(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        tracker.record(_make_record(outcome="win", ev=0.1))
        tracker.record(_make_record(outcome="loss", ev=-0.05))
        stats = tracker.rolling_stats(last_n=100)
        assert stats["hands"] == 2
        # win_rate = 1/(1+1) = 0.5
        assert stats["win_rate"] == pytest.approx(0.5)
        tracker.close()

    def test_invalid_outcome_raises(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        with pytest.raises(ValueError, match="Invalid outcome"):
            tracker.record(_make_record(outcome="draw"))
        tracker.close()

    def test_reset_shoe(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        tracker.record(_make_record(outcome="win"))
        completed = tracker.reset_shoe()
        assert completed.hands == 1
        assert tracker.current_shoe_stats.hands == 0
        tracker.close()

    def test_shoe_stats_win_rate(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        for _ in range(3):
            tracker.record(_make_record(outcome="win"))
        for _ in range(1):
            tracker.record(_make_record(outcome="loss"))
        assert tracker.current_shoe_stats.win_rate == pytest.approx(0.75)
        tracker.close()

    def test_action_compliance(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        tracker.record(_make_record(recommended="stand", actual="stand"))
        tracker.record(_make_record(recommended="stand", actual="hit"))
        assert tracker.current_shoe_stats.action_compliance == pytest.approx(0.5)
        tracker.close()

    def test_extract_features_length(self):
        rec = _make_record()
        features = MLStatsTracker.extract_features(rec)
        assert len(features) == 10

    def test_extract_features_hand_type_flags(self):
        soft_rec = _make_record(hand_type="soft")
        features = MLStatsTracker.extract_features(soft_rec)
        # is_soft = index 7
        assert features[7] == 1.0
        assert features[8] == 0.0
        assert features[9] == 0.0

    def test_train_model_insufficient_data(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        # Only 5 records — below min_samples
        for _ in range(5):
            tracker.record(_make_record(outcome="win"))
        result = tracker.train_model(min_samples=100)
        assert result is None
        tracker.close()

    def test_rolling_stats_empty_db(self, tmp_path):
        tracker = MLStatsTracker(db_path=str(tmp_path / "test.db"))
        stats = tracker.rolling_stats(last_n=100)
        assert stats["hands"] == 0
        assert stats["win_rate"] == 0.0
        tracker.close()

    def test_rolling_window_trim(self, tmp_path):
        tracker = MLStatsTracker(
            db_path=str(tmp_path / "test.db"),
            rolling_window=5,
        )
        for i in range(8):
            tracker.record(_make_record(outcome="win" if i % 2 == 0 else "loss"))
        # DB should not have more than 5 rows after trim
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "test.db"))
        count = con.execute("SELECT COUNT(*) FROM hand_stats").fetchone()[0]
        con.close()
        assert count <= 5
        tracker.close()


class TestShoeStats:
    def test_empty_win_rate(self):
        s = ShoeStats()
        assert s.win_rate == 0.0

    def test_ev_per_hand(self):
        s = ShoeStats(hands=4, total_ev=0.8)
        assert s.ev_per_hand == pytest.approx(0.2)

    def test_roi_with_no_bets(self):
        s = ShoeStats(total_net=100.0, total_bets=0.0)
        assert s.roi == 0.0
