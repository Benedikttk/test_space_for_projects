"""Tests for blackjack.card_counter."""
from blackjack.card_counter import CardCounterUI, hi_lo_value


def test_hi_lo_value_low_cards():
    for rank in "23456":
        assert hi_lo_value(rank) == +1


def test_hi_lo_value_neutral_cards():
    for rank in "789":
        assert hi_lo_value(rank) == 0


def test_hi_lo_value_high_cards():
    for rank in "TJQKA":
        assert hi_lo_value(rank) == -1


def test_card_counter_hi_lo():
    """Test Hi-Lo count tracking."""
    counter = CardCounterUI()
    counter.add_card('2')  # +1
    counter.add_card('K')  # -1
    counter.add_card('5')  # +1
    assert counter.running_count == 1
    assert counter.observation_ratio == 3 / 416


def test_card_counter_display():
    """Test table formatting."""
    counter = CardCounterUI()
    counter.add_card('K')
    counter.add_card('K')
    counter.add_card('A')

    table = counter.get_table_display()
    assert len(table) == 2  # K and A

    k_row = next(r for r in table if r['Rank'] == 'K')
    assert k_row['Count'] == 2
    assert k_row['Card Value'] == 10
    assert k_row['Total Value'] == 20
    assert k_row['Hi-Lo'] == -2

    a_row = next(r for r in table if r['Rank'] == 'A')
    assert a_row['Count'] == 1
    assert a_row['Hi-Lo'] == -1


def test_card_counter_true_count_updates():
    counter = CardCounterUI(total_cards=416)
    for _ in range(52):
        counter.add_card('2')  # +1 each
    # 52 low cards seen → running count = 52
    # decks_remaining = (416 - 52) / 52 = 7.0
    assert counter.running_count == 52
    expected_true = 52 / 7.0
    assert abs(counter.true_count - expected_true) < 0.01


def test_card_counter_reset():
    counter = CardCounterUI()
    counter.add_card('A')
    counter.add_card('K')
    counter.reset()
    assert counter.running_count == 0
    assert counter.true_count == 0.0
    assert counter.observation_ratio == 0.0
    assert len(counter.observed_cards) == 0
    assert len(counter.order) == 0


def test_card_counter_get_summary():
    counter = CardCounterUI()
    counter.add_card('T')
    summary = counter.get_summary()
    assert summary['cards_observed'] == 1
    assert summary['running_count'] == -1
    assert summary['card_counts'] == {'T': 1}


def test_card_counter_table_rank_order():
    """Ranks in table should follow 23456789TJQKA order."""
    counter = CardCounterUI()
    for rank in 'AKQ2':
        counter.add_card(rank)
    table = counter.get_table_display()
    ranks = [r['Rank'] for r in table]
    assert ranks == sorted(ranks, key=lambda r: '23456789TJQKA'.index(r))
