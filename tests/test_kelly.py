import pytest
from blackjack.kelly import kelly_fraction, recommended_bet, kelly_summary


def test_kelly_fraction_zero_when_ev_nonpositive():
    assert kelly_fraction(0.0) == 0.0
    assert kelly_fraction(-0.1) == 0.0


def test_kelly_fraction_positive_when_ev_positive():
    assert kelly_fraction(0.01) > 0.0


def test_kelly_fraction_half_kelly_smaller_than_full():
    ev = 0.05
    full = kelly_fraction(ev, fraction=1.0)
    half = kelly_fraction(ev, fraction=0.5)
    assert half < full


def test_recommended_bet_min_when_ev_nonpositive():
    assert recommended_bet(0.0, bankroll=1000.0, min_bet=5.0) == 5.0
    assert recommended_bet(-0.1, bankroll=1000.0, min_bet=5.0) == 5.0


def test_recommended_bet_clamps_to_max():
    # Very high EV and bankroll should clamp to max_bet
    result = recommended_bet(1.0, bankroll=1_000_000.0, min_bet=5.0, max_bet=500.0)
    assert result == 500.0


def test_kelly_summary_keys():
    summary = kelly_summary(ev=0.02, bankroll=1000.0)
    assert set(summary.keys()) == {
        "kelly_fraction", "recommended_bet", "ev", "edge_percent", "is_positive_ev"
    }


def test_kelly_summary_positive_ev():
    summary = kelly_summary(ev=0.02, bankroll=1000.0)
    assert summary["is_positive_ev"] is True
    assert summary["edge_percent"] == pytest.approx(2.0)


def test_kelly_summary_negative_ev():
    summary = kelly_summary(ev=-0.005, bankroll=1000.0)
    assert summary["is_positive_ev"] is False
    assert summary["kelly_fraction"] == 0.0


def test_kelly_fraction_half_keyword_matches_half_kelly_formula():
    assert kelly_fraction(0.02, variance=1.0, half=True) == pytest.approx(0.01)


def test_recommended_bet_accepts_kelly_fraction_keyword():
    result = recommended_bet(ev=0.02, bankroll=1000.0, kelly_fraction=0.01)
    assert result == pytest.approx(10.0)


def test_kelly_summary_text_mode():
    summary = kelly_summary(ev=0.02, kelly_fraction=0.01)
    assert isinstance(summary, str)
