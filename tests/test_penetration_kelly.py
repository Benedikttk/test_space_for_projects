"""Tests for penetration-aware Kelly functions in blackjack/kelly.py."""
from __future__ import annotations

import pytest

from blackjack.kelly import (
    penetration_quality,
    penetration_adjusted_kelly,
    penetration_adjusted_bet,
)


class TestPenetrationQuality:
    def test_zero_penetration(self):
        assert penetration_quality(0.0) == pytest.approx(0.0)

    def test_below_threshold(self):
        assert penetration_quality(0.10) == pytest.approx(0.0)

    def test_at_threshold_boundary(self):
        # Just above 0.15
        pq = penetration_quality(0.16)
        assert 0.0 < pq < 1.0

    def test_full_penetration(self):
        assert penetration_quality(1.0) == pytest.approx(1.0)

    def test_monotone(self):
        vals = [penetration_quality(p) for p in [0.0, 0.2, 0.5, 0.75, 1.0]]
        assert vals == sorted(vals)


class TestPenetrationAdjustedKelly:
    def test_zero_ev_returns_zero(self):
        assert penetration_adjusted_kelly(0.0, penetration=0.8) == pytest.approx(0.0)

    def test_negative_ev_returns_zero(self):
        assert penetration_adjusted_kelly(-0.1, penetration=0.8) == pytest.approx(0.0)

    def test_low_penetration_conservative(self):
        # At near-zero penetration: adjusted ≈ 0.5 × base
        base_f = 0.5 * 0.1 / 1.15  # half-kelly with ev=0.1
        adjusted = penetration_adjusted_kelly(0.1, penetration=0.0)
        assert adjusted == pytest.approx(0.5 * base_f, rel=1e-3)

    def test_high_penetration_less_conservative(self):
        low = penetration_adjusted_kelly(0.1, penetration=0.1)
        high = penetration_adjusted_kelly(0.1, penetration=0.8)
        assert high > low

    def test_clamped_to_one(self):
        # Very high EV should still be clamped
        result = penetration_adjusted_kelly(100.0, penetration=1.0)
        assert result <= 1.0


class TestPenetrationAdjustedBet:
    def test_negative_ev_returns_min_bet(self):
        bet = penetration_adjusted_bet(-0.1, 0.8, bankroll=1000.0, min_bet=5.0)
        assert bet == pytest.approx(5.0)

    def test_bet_within_bounds(self):
        bet = penetration_adjusted_bet(0.05, 0.6, bankroll=1000.0, min_bet=5.0, max_bet=500.0)
        assert 5.0 <= bet <= 500.0

    def test_rounded_to_min_bet(self):
        bet = penetration_adjusted_bet(0.05, 0.6, bankroll=1000.0, min_bet=5.0, max_bet=500.0)
        assert bet % 5.0 == pytest.approx(0.0)
