"""Shoe composition tracker.

Tracks the remaining card distribution in a shoe of N decks.  Provides
probability helpers used by the EV engine and the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Canonical ranks (face cards unified to 'T')
ALL_RANKS: List[str] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']

# Number of cards per rank in a single deck (T = T/J/Q/K = 4×4 = 16)
_PER_DECK: Dict[str, int] = {
    '2': 4, '3': 4, '4': 4, '5': 4, '6': 4,
    '7': 4, '8': 4, '9': 4, 'T': 16, 'A': 4,
}


@dataclass
class Shoe:
    """Mutable shoe composition."""

    decks: int = 8
    counts: Dict[str, int] = field(default_factory=dict)
    running_count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.reset()

    def reset(self) -> None:
        """Restore shoe to a freshly shuffled state."""
        self.counts = {r: _PER_DECK[r] * self.decks for r in ALL_RANKS}
        self.running_count = 0

    @property
    def total_remaining(self) -> int:
        return sum(self.counts.values())

    def prob(self, rank: str) -> float:
        """Probability that the next card drawn is *rank*."""
        total = self.total_remaining
        if total == 0:
            return 0.0
        return self.counts.get(rank, 0) / total

    def remove(self, rank: str) -> None:
        """Record that *rank* was dealt; update running count."""
        from blackjack.hand import _normalise
        rank = _normalise(rank)
        if self.counts.get(rank, 0) <= 0:
            raise ValueError(f"Rank {rank!r} not available in shoe")
        self.counts[rank] -= 1
        if rank in ('2', '3', '4', '5', '6'):
            self.running_count += 1
        elif rank in ('T', 'A'):
            self.running_count -= 1

    @property
    def true_count(self) -> float:
        """True count = running count / decks remaining."""
        decks_remaining = self.total_remaining / 52
        if decks_remaining < 0.1:
            return 0.0
        return self.running_count / decks_remaining

    def rank_distribution(self) -> Dict[str, float]:
        """Return probability distribution over all ranks."""
        total = self.total_remaining
        if total == 0:
            return {r: 0.0 for r in ALL_RANKS}
        return {r: self.counts[r] / total for r in ALL_RANKS}

    def snapshot(self) -> Dict[str, int]:
        """Return a copy of the current counts."""
        return dict(self.counts)
