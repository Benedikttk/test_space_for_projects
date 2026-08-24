import pytest

from blackjack.hand import Hand, can_split, hand_total, is_blackjack, is_soft


def test_hand_total_hard_and_soft_and_multi_ace_reduction():
    assert hand_total(["T", "7"]) == (17, False)
    assert hand_total(["A", "6"]) == (17, True)
    assert hand_total(["A", "A", "9"]) == (21, True)
    assert hand_total(["A", "A", "9", "9"]) == (20, False)


def test_is_blackjack_only_true_for_two_card_21():
    assert Hand(["A", "T"]).is_blackjack is True
    assert Hand(["A", "9", "A"]).is_blackjack is False
    assert Hand(["7", "7", "7"]).is_blackjack is False


def test_can_split_pairs_and_face_card_pairs():
    assert Hand(["8", "8"]).can_split is True
    assert Hand(["K", "Q"]).can_split is True
    assert Hand(["K", "9"]).can_split is False


def test_is_bust_variants():
    assert Hand(["T", "9", "5"]).is_bust is True
    assert Hand(["A", "9"]).is_bust is False


def test_is_soft_vs_hard_after_hit():
    assert Hand(["A", "6"]).is_soft is True
    assert Hand(["A", "6", "9"]).is_soft is False


def test_functional_hand_helpers_match_hand_properties():
    cards = ["A", "T"]
    assert is_soft(cards) is True
    assert is_blackjack(cards) is True
    assert can_split(["Q", "K"]) is True
