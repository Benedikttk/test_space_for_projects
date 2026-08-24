"""Real-time card counting and inventory tracking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List

from blackjack.hand import RANK_VALUE

# Hi-Lo count value for each rank
_HI_LO: dict[str, int] = {
    '2': +1, '3': +1, '4': +1, '5': +1, '6': +1,
    '7':  0, '8':  0, '9':  0,
    'T': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1,
}

# Total cards in a standard 8-deck shoe
_TOTAL_CARDS = 416


def hi_lo_value(rank: str) -> int:
    """Return the Hi-Lo count contribution (+1, 0, or -1) for a rank."""
    return _HI_LO.get(rank, 0)


class CardCounterUI:
    """Tracks observed cards and computes running/true counts.

    This class is designed for use in the Streamlit session state so that
    card counts survive across reruns for the duration of a shoe.
    """

    def __init__(self, total_cards: int = _TOTAL_CARDS) -> None:
        self.total_cards = total_cards
        self.observed_cards: Counter[str] = Counter()
        self.order: list[str] = []
        self.running_count: int = 0
        self.true_count: float = 0.0
        self.observation_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_card(self, rank: str) -> None:
        """Record a newly detected card rank."""
        self.observed_cards[rank] += 1
        self.order.append(rank)

        # Update Hi-Lo running count
        self.running_count += hi_lo_value(rank)

        # Recompute true count (RC / decks remaining)
        cards_seen = sum(self.observed_cards.values())
        cards_remaining = self.total_cards - cards_seen
        decks_remaining = cards_remaining / 52
        if decks_remaining > 0:
            self.true_count = self.running_count / decks_remaining
        else:
            self.true_count = float(self.running_count)

        self.observation_ratio = cards_seen / self.total_cards

    def reset(self) -> None:
        """Reset counters for a new shoe."""
        self.observed_cards.clear()
        self.order.clear()
        self.running_count = 0
        self.true_count = 0.0
        self.observation_ratio = 0.0

    # ------------------------------------------------------------------
    # Read-only summaries
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a plain-dict summary suitable for metrics display."""
        return {
            'cards_observed': sum(self.observed_cards.values()),
            'running_count': self.running_count,
            'true_count': self.true_count,
            'observation_ratio': self.observation_ratio,
            'card_counts': dict(self.observed_cards),
            'card_values': {
                rank: RANK_VALUE[rank]
                for rank in self.observed_cards
                if rank in RANK_VALUE
            },
        }

    def get_table_display(self) -> list[dict]:
        """Return per-rank rows for a dataframe / table widget.

        Each row contains:
        ``{'Rank', 'Count', 'Card Value', 'Total Value', 'Hi-Lo'}``
        """
        rows = []
        for rank in '23456789TJQKA':
            if rank not in self.observed_cards:
                continue
            count = self.observed_cards[rank]
            card_value = RANK_VALUE.get(rank, 10)
            rows.append({
                'Rank': rank,
                'Count': count,
                'Card Value': card_value,
                'Total Value': count * card_value,
                'Hi-Lo': hi_lo_value(rank) * count,
            })
        return rows
