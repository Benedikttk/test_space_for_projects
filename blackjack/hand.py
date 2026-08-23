"""Hand representation and evaluation.

A Hand is an ordered list of card ranks (strings: '2'–'9', 'T', 'J', 'Q',
'K', 'A').  All face cards are stored as 'T' internally (value 10) except
'A' which is always stored as 'A'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# Rank → hard point value
RANK_VALUE: dict[str, int] = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11,
}


def _normalise(rank: str) -> str:
    """Normalise face cards to 'T'; keep everything else as-is."""
    return 'T' if rank in ('J', 'Q', 'K') else rank.upper()


def hand_total(ranks: Sequence[str]) -> Tuple[int, bool]:
    """Return (best_total, is_soft) for a sequence of ranks.

    *is_soft* is True when an Ace is being counted as 11 and the total is
    <= 21 (i.e., the hand is not busted using the soft count).
    """
    total = 0
    aces = 0
    for r in ranks:
        r = _normalise(r)
        if r == 'A':
            aces += 1
            total += 11
        else:
            total += RANK_VALUE[r]
    soft = False
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    # soft = at least one ace still counted as 11
    if aces > 0 and total <= 21:
        soft = True
    return total, soft


@dataclass
class Hand:
    """Mutable hand during a round."""

    cards: List[str] = field(default_factory=list)
    # Number of times this hand has been produced by splitting
    splits_used: int = 0
    # True when this hand was produced by splitting aces
    is_post_split_ace: bool = False
    # True when the hand was created by doubling (only one extra card allowed)
    doubled: bool = False

    def add(self, rank: str) -> None:
        self.cards.append(_normalise(rank))

    @property
    def total(self) -> int:
        return hand_total(self.cards)[0]

    @property
    def is_soft(self) -> bool:
        return hand_total(self.cards)[1]

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def is_blackjack(self) -> bool:
        """True only for a natural (exactly 2 cards totalling 21)."""
        return len(self.cards) == 2 and self.total == 21

    @property
    def can_split(self) -> bool:
        """True when both cards have equal value (pair)."""
        if len(self.cards) != 2:
            return False
        return RANK_VALUE[_normalise(self.cards[0])] == RANK_VALUE[_normalise(self.cards[1])]

    @property
    def split_rank(self) -> str | None:
        """The rank being split, or None if not a splittable pair."""
        if not self.can_split:
            return None
        return _normalise(self.cards[0])

    def __repr__(self) -> str:
        tot, soft = hand_total(self.cards)
        label = f"soft {tot}" if soft else str(tot)
        return f"Hand({self.cards}, total={label})"
