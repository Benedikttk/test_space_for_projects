import pytest

from blackjack.ev import _stand_ev, action_evs, best_action, dealer_distribution
from blackjack.hand import Hand
from blackjack.rules import RuleSet
from blackjack.shoe import Shoe


def test_dealer_distribution_probabilities_sum_to_one():
    dist = dealer_distribution("6", Shoe(decks=8), RuleSet(), player_cards=["T", "9"])
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-10)


def test_dealer_distribution_differs_between_s17_and_h17_for_upcard_6():
    shoe = Shoe(decks=8)
    s17 = dealer_distribution("6", shoe, RuleSet(dealer_hits_soft17=False), player_cards=["T", "9"])
    h17 = dealer_distribution("6", shoe, RuleSet(dealer_hits_soft17=True), player_cards=["T", "9"])
    assert s17 != h17


def test_stand_ev_known_distribution_scalar():
    dist = {17: 0.2, 18: 0.3, 22: 0.5}
    assert _stand_ev(18, dist) == pytest.approx(0.7)


def test_stand_ev_signs_for_common_spots_in_fresh_8deck_shoe():
    shoe = Shoe(decks=8)
    rules = RuleSet()
    ev_20_v_6 = action_evs(Hand(["T", "T"]), "6", shoe, rules)["stand"]
    ev_12_v_t = action_evs(Hand(["T", "2"]), "T", shoe, rules)["stand"]
    assert ev_20_v_6 > 0
    assert ev_12_v_t < 0


def test_surrender_action_ev_is_always_minus_half():
    evs = action_evs(Hand(["9", "7"]), "A", Shoe(decks=8), RuleSet(surrender="late"))
    assert evs["surrender"] == -0.5


def test_blackjack_payout_natural_exceeds_non_natural_21():
    rules = RuleSet(blackjack_payout=1.5, natural_beats_dealer_21=True)
    dist = {21: 1.0}
    assert _stand_ev(21, dist, is_blackjack=True, rules=rules) > _stand_ev(
        21, dist, is_blackjack=False, rules=rules
    )


def test_best_action_returns_highest_ev_action():
    action, ev = best_action({"hit": -0.1, "stand": 0.2, "double": 0.1})
    assert (action, ev) == ("stand", 0.2)
