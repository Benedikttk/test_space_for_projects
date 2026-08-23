from blackjack.actions import get_legal_actions
from blackjack.hand import Hand
from blackjack.rules import RuleSet


def test_first_decision_has_hit_stand_double():
    actions = get_legal_actions(Hand(["8", "3"]), RuleSet())
    assert actions.hit and actions.stand and actions.double


def test_post_split_ace_lock_only_stand_allowed():
    rules = RuleSet(split_aces_get_one_card=True)
    actions = get_legal_actions(Hand(["A", "9"]), rules, splits_used=1, is_post_split_ace=True)
    assert actions.as_set() == frozenset({"stand"})


def test_surrender_blocked_when_rule_none():
    actions = get_legal_actions(Hand(["9", "7"]), RuleSet(surrender="none"))
    assert actions.surrender is False


def test_surrender_blocked_on_split_hands():
    actions = get_legal_actions(Hand(["9", "7"]), RuleSet(), splits_used=1)
    assert actions.surrender is False


def test_double_blocked_when_das_off_after_split():
    actions = get_legal_actions(Hand(["9", "2"]), RuleSet(double_after_split=False), splits_used=1)
    assert actions.double is False


def test_split_blocked_when_max_splits_reached():
    rules = RuleSet(max_splits=1)
    actions = get_legal_actions(Hand(["8", "8"]), rules, splits_used=1)
    assert actions.split is False


def test_rsa_toggle_allows_or_blocks_ace_resplit():
    blocked = get_legal_actions(Hand(["A", "A"]), RuleSet(resplit_aces=False), splits_used=1)
    allowed = get_legal_actions(Hand(["A", "A"]), RuleSet(resplit_aces=True), splits_used=1)
    assert blocked.split is False
    assert allowed.split is True


def test_bust_hand_has_no_legal_actions():
    actions = get_legal_actions(Hand(["T", "9", "5"]), RuleSet())
    assert actions.as_set() == frozenset()


def test_three_plus_cards_no_double_or_split():
    actions = get_legal_actions(Hand(["5", "3", "2"]), RuleSet())
    assert actions.double is False
    assert actions.split is False
