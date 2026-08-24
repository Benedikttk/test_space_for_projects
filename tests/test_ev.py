import pytest

from blackjack.ev import _stand_ev, action_evs, best_action, dealer_distribution, insurance_ev, dampened_ev, basic_strategy_ev
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


def test_dealer_peeks_does_not_remove_tens_from_subsequent_draws():
    shoe = Shoe(decks=1)
    shoe.counts = {r: 0 for r in shoe.counts}
    shoe.counts['A'] = 1
    shoe.counts['2'] = 1
    shoe.counts['T'] = 2

    dist = dealer_distribution('A', shoe, RuleSet(dealer_peeks=True))
    assert dist == {22: pytest.approx(1.0)}


def test_dealer_peeks_conditioning_sums_to_one():
    dist = dealer_distribution('A', Shoe(decks=8), RuleSet(dealer_peeks=True), player_cards=['9', '7'])
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-10)


def test_dealer_no_peek_vs_peek_differ_for_ace_upcard():
    shoe = Shoe(decks=8)
    no_peek = dealer_distribution('A', shoe, RuleSet(dealer_peeks=False), player_cards=['T', '6'])
    peek = dealer_distribution('A', shoe, RuleSet(dealer_peeks=True), player_cards=['T', '6'])
    assert no_peek != peek


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


# ---------------------------------------------------------------------------
# insurance_ev
# ---------------------------------------------------------------------------

def test_insurance_ev_negative_on_fresh_shoe():
    # On a neutral 8-deck shoe P(T) = 4/13 ≈ 0.308, EV ≈ -0.077
    shoe = Shoe(decks=8)
    ev = insurance_ev(shoe, 'A')
    assert ev < 0.0
    assert ev == pytest.approx(2 * (128 / (415)) - 1, abs=1e-6)


def test_insurance_ev_increases_with_more_tens():
    shoe_normal = Shoe(decks=8)
    shoe_rich = Shoe(decks=8)
    # Remove low cards to enrich the shoe with tens
    for _ in range(20):
        shoe_rich.remove('2')
    ev_normal = insurance_ev(shoe_normal, 'A')
    ev_rich = insurance_ev(shoe_rich, 'A')
    assert ev_rich > ev_normal


# ---------------------------------------------------------------------------
# basic_strategy_ev
# ---------------------------------------------------------------------------

def test_basic_strategy_ev_cached_returns_same_as_uncached():
    rules = RuleSet(insurance=False)  # disable insurance for cleaner comparison
    hand = Hand(['T', '6'])
    upcard = '7'
    fresh = Shoe(decks=8)
    bs_evs = basic_strategy_ev(hand, upcard, rules, decks=8)
    direct_evs = action_evs(hand, upcard, fresh, rules)
    for action in bs_evs:
        assert bs_evs[action] == pytest.approx(direct_evs[action], abs=1e-10)


def test_action_evs_accepts_player_cards_sequence():
    evs = action_evs(["T", "6"], "7", Shoe(decks=8), RuleSet())
    assert "stand" in evs


def test_basic_strategy_ev_scalar_compat_mode_returns_float():
    ev = basic_strategy_ev(
        player_total=16,
        dealer_upcard="T",
        is_soft=False,
        is_blackjack=False,
    )
    assert isinstance(ev, float)


# ---------------------------------------------------------------------------
# dampened_ev
# ---------------------------------------------------------------------------

def test_dampened_ev_returns_basic_when_ratio_zero():
    result = dampened_ev(raw_ev=0.1, basic_strategy_ev_val=-0.05,
                         observation_ratio=0.0)
    assert result == pytest.approx(-0.05)


def test_dampened_ev_returns_raw_when_ratio_one():
    result = dampened_ev(raw_ev=0.1, basic_strategy_ev_val=-0.05,
                         observation_ratio=1.0)
    assert result == pytest.approx(0.1)


def test_dampened_ev_sqrt_blend_midpoint():
    raw_ev = 1.0
    basic_ev = 0.0
    result = dampened_ev(raw_ev=raw_ev, basic_strategy_ev_val=basic_ev,
                         observation_ratio=0.575)
    assert result == pytest.approx(0.5 ** 0.5)
