import pytest

from blackjack.actions import get_legal_actions
from blackjack.ev import action_evs, split_ev
from blackjack.hand import Hand
from blackjack.rules import RuleSet
from blackjack.shoe import Shoe


def test_split_8s_vs_6_better_than_stand():
    evs = action_evs(Hand(["8", "8"]), "6", Shoe(decks=8), RuleSet())
    assert evs["split"] > evs["stand"]


def test_split_ev_raises_when_max_splits_reached():
    rules = RuleSet(max_splits=2)
    with pytest.raises(ValueError):
        split_ev(Hand(["8", "8"]), "6", Shoe(decks=8), rules, splits_used=2)


def test_post_split_ace_hands_locked_to_stand_only():
    actions = get_legal_actions(
        Hand(["A", "9"], splits_used=1, is_post_split_ace=True),
        RuleSet(split_aces_get_one_card=True),
        splits_used=1,
        is_post_split_ace=True,
    )
    assert actions.as_set() == frozenset({"stand"})


def test_das_off_blocks_double_on_split_hands():
    actions = get_legal_actions(Hand(["8", "3"]), RuleSet(double_after_split=False), splits_used=1)
    assert actions.double is False


def test_split_ev_uses_depleted_shoe_after_removing_split_pair(monkeypatch):
    captured_counts = []

    def fake_action_evs(hand, dealer_upcard, shoe, rules, splits_used=0, is_post_split_ace=False):
        captured_counts.append(shoe.snapshot())
        return {"stand": 0.0}

    monkeypatch.setattr("blackjack.ev.action_evs", fake_action_evs)

    shoe = Shoe(decks=1)
    split_ev(Hand(["8", "8"]), "6", shoe, RuleSet())

    original = 4
    assert captured_counts
    assert all(c["8"] <= original - 2 for c in captured_counts)
    assert any(c["8"] == original - 3 for c in captured_counts)


def test_rsa_false_allows_first_ace_split_but_blocks_second():
    rules = RuleSet(resplit_aces=False)
    first = get_legal_actions(Hand(["A", "A"]), rules, splits_used=0)
    second = get_legal_actions(Hand(["A", "A"]), rules, splits_used=1)
    assert first.split is True
    assert second.split is False
