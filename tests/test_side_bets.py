import pytest
from blackjack.shoe import Shoe
from blackjack.side_bets import perfect_pairs_ev, twenty_one_plus_three_ev, side_bet_summary


def test_perfect_pairs_ev_negative_for_non_pair():
    shoe = Shoe(decks=8)
    ev = perfect_pairs_ev(shoe, 'T', '9')
    assert ev < 0.0


def test_perfect_pairs_ev_pair_beats_non_pair():
    shoe = Shoe(decks=8)
    ev_pair = perfect_pairs_ev(shoe, 'T', 'T')
    ev_non_pair = perfect_pairs_ev(shoe, 'T', '9')
    assert ev_pair > ev_non_pair


def test_twenty_one_plus_three_ev_returns_float():
    shoe = Shoe(decks=8)
    result = twenty_one_plus_three_ev(shoe, 'T', '5', '6')
    assert isinstance(result, float)


def test_twenty_one_plus_three_tok_returns_payout():
    shoe = Shoe(decks=8)
    ev = twenty_one_plus_three_ev(shoe, 'T', 'T', 'T')
    assert ev == pytest.approx(30.0)


def test_twenty_one_plus_three_straight_between_flush_and_plain():
    shoe = Shoe(decks=8)
    ev = twenty_one_plus_three_ev(shoe, '4', '5', '6')
    assert 10.0 < ev < 40.0


def test_perfect_pairs_uses_remaining_count():
    fresh = Shoe(decks=8)
    depleted = Shoe(decks=8)
    for _ in range(20):
        depleted.remove('T')
    ev_fresh = perfect_pairs_ev(fresh, 'T', 'T')
    ev_depleted = perfect_pairs_ev(depleted, 'T', 'T')
    assert ev_depleted < ev_fresh


def test_side_bet_summary_keys():
    shoe = Shoe(decks=8)
    summary = side_bet_summary(shoe, 'T', '5', '6')
    assert set(summary.keys()) == {'perfect_pairs', 'twenty_one_plus_three', 'insurance'}


def test_side_bet_summary_insurance_none_for_non_ace():
    shoe = Shoe(decks=8)
    summary = side_bet_summary(shoe, 'T', '5', '6')
    assert summary['insurance'] is None


def test_side_bet_summary_insurance_not_none_for_ace():
    shoe = Shoe(decks=8)
    summary = side_bet_summary(shoe, 'T', '5', 'A')
    assert summary['insurance'] is not None
    assert isinstance(summary['insurance'], float)
