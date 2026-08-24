"""Tests for blackjack/cut_card.py."""
from __future__ import annotations

import numpy as np
import pytest

from blackjack.cut_card import (
    BayesianCardDistribution,
    CutCardDetector,
    PenetrationTracker,
    ShoeHistory,
)
from blackjack.shoe import ALL_RANKS


# ---------------------------------------------------------------------------
# BayesianCardDistribution
# ---------------------------------------------------------------------------

class TestBayesianCardDistribution:
    def test_initial_probs_are_uniform(self):
        dist = BayesianCardDistribution(decks=8)
        probs = dist.posterior_probs()
        assert set(probs.keys()) == set(ALL_RANKS)
        vals = list(probs.values())
        # All equal (uniform prior)
        assert max(vals) - min(vals) < 1e-9

    def test_probs_sum_to_one(self):
        dist = BayesianCardDistribution(decks=1)
        total = sum(dist.posterior_probs().values())
        assert abs(total - 1.0) < 1e-9

    def test_update_shifts_posterior(self):
        dist = BayesianCardDistribution(decks=2)
        # Observe many Aces
        dist.update_from_shoe({"A": 100})
        probs = dist.posterior_probs()
        # Ace should have highest probability
        assert probs["A"] == max(probs.values())

    def test_expected_counts_sum_to_region_size(self):
        dist = BayesianCardDistribution(decks=4)
        region_size = 50
        counts = dist.expected_counts_in_region(region_size)
        total = sum(counts.values())
        assert abs(total - region_size) < 1e-6

    def test_reset_prior(self):
        dist = BayesianCardDistribution(decks=2)
        dist.update_from_shoe({"A": 999})
        dist.reset_prior()
        probs = dist.posterior_probs()
        vals = list(probs.values())
        assert max(vals) - min(vals) < 1e-9


# ---------------------------------------------------------------------------
# PenetrationTracker
# ---------------------------------------------------------------------------

class TestPenetrationTracker:
    def test_default_state(self):
        t = PenetrationTracker(total_cards=416, alert_threshold=26)
        state = t.update(0)
        assert state.cards_dealt == 0
        assert state.penetration_pct == pytest.approx(0.0)
        assert not state.is_reshuffle_alert

    def test_set_cut_card_position(self):
        t = PenetrationTracker(total_cards=416)
        t.set_cut_card_position(312)
        state = t.update(0)
        assert state.cut_card_position == 312

    def test_penetration_increases(self):
        t = PenetrationTracker(total_cards=416)
        t.set_cut_card_position(312)
        s1 = t.update(100)
        s2 = t.update(200)
        assert s2.penetration_pct > s1.penetration_pct

    def test_reshuffle_alert_triggers(self):
        t = PenetrationTracker(total_cards=416, alert_threshold=26)
        t.set_cut_card_position(312)
        # 290 dealt → 22 remaining until cut (< 26 threshold)
        state = t.update(290)
        assert state.is_reshuffle_alert

    def test_no_alert_far_from_cut(self):
        t = PenetrationTracker(total_cards=416, alert_threshold=26)
        t.set_cut_card_position(312)
        state = t.update(50)
        assert not state.is_reshuffle_alert

    def test_reset_clears_dealt(self):
        t = PenetrationTracker(total_cards=416)
        t.update(200)
        t.reset()
        state = t.update(0)
        assert state.cards_dealt == 0

    def test_penetration_fraction_property(self):
        t = PenetrationTracker(total_cards=400)
        t.update(200)
        assert t.penetration_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CutCardDetector (without OpenCV — tests stub behaviour)
# ---------------------------------------------------------------------------

class TestCutCardDetectorNoCV:
    def test_returns_not_detected_on_empty_frame(self):
        det = CutCardDetector(total_cards=416)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = det.detect(empty)
        assert not result.detected

    def test_reset_smoothing(self):
        det = CutCardDetector(total_cards=416)
        # Manually set smoothed position
        det._smoothed_position = 0.5
        det.reset_smoothing()
        assert det._smoothed_position is None


# ---------------------------------------------------------------------------
# ShoeHistory
# ---------------------------------------------------------------------------

class TestShoeHistory:
    def test_empty_history(self):
        h = ShoeHistory()
        assert h.average_penetration() == 0.0
        assert h.all_records() == []

    def test_record_stored(self):
        h = ShoeHistory()
        rec = h.record(
            cut_card_position=312,
            total_cards_dealt=300,
            true_count_profile=[0.5, 1.0],
            penetration_profile=[10.0, 20.0],
            observed_counts={"A": 10, "T": 40},
        )
        assert rec.shoe_id == 1
        assert len(h.all_records()) == 1

    def test_max_shoes_eviction(self):
        h = ShoeHistory(max_shoes=3)
        for _ in range(5):
            h.record(312, 300, [], [], {})
        assert len(h.all_records()) == 3

    def test_average_penetration(self):
        h = ShoeHistory()
        h.record(400, 200, [], [], {})  # 50 %
        h.record(400, 300, [], [], {})  # 75 %
        avg = h.average_penetration()
        assert abs(avg - 62.5) < 0.1
