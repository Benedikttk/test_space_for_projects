import pytest

from blackjack.shoe import Shoe


def test_fresh_8deck_shoe_total_cards():
    assert Shoe(decks=8).total_remaining == 416


def test_remove_decrements_and_raises_on_over_removal():
    shoe = Shoe(decks=1)
    for _ in range(4):
        shoe.remove("2")
    assert shoe.counts["2"] == 0
    with pytest.raises(ValueError):
        shoe.remove("2")


def test_hilo_running_count_updates():
    shoe = Shoe(decks=1)
    shoe.remove("2")
    shoe.remove("T")
    shoe.remove("8")
    assert shoe.running_count == 0


def test_true_count_is_running_over_decks_remaining():
    shoe = Shoe(decks=1)
    shoe.remove("2")
    expected = 1 / (shoe.total_remaining / 52)
    assert shoe.true_count == pytest.approx(expected)


def test_rank_distribution_sums_to_one():
    shoe = Shoe(decks=1)
    assert sum(shoe.rank_distribution().values()) == pytest.approx(1.0)
