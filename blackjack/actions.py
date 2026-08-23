"""Legal action gating.

Determines which actions are legal for a given hand state under a given
RuleSet.  This is the single source of truth for action availability.

Actions
-------
hit        Draw another card.
stand      End the player's turn.
double     Double the bet and draw exactly one more card.
split       Split a pair into two separate hands.
surrender  Forfeit half the bet (late or early depending on rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from blackjack.rules import RuleSet
from blackjack.hand import Hand


@dataclass(frozen=True)
class LegalActions:
    """Set of legal actions for a specific decision point."""

    hit: bool
    stand: bool
    double: bool
    split: bool
    surrender: bool

    def as_set(self) -> FrozenSet[str]:
        result = set()
        if self.hit:       result.add("hit")
        if self.stand:     result.add("stand")
        if self.double:    result.add("double")
        if self.split:     result.add("split")
        if self.surrender: result.add("surrender")
        return frozenset(result)

    def __repr__(self) -> str:
        return f"LegalActions({sorted(self.as_set())})"


def get_legal_actions(
    hand: Hand,
    rules: RuleSet,
    splits_used: int = 0,
    is_post_split_ace: bool = False,
    dealer_upcard: str | None = None,
) -> LegalActions:
    """Return the legal actions for *hand* under *rules*."""
    n_cards = len(hand.cards)

    if hand.is_bust:
        return LegalActions(hit=False, stand=False, double=False,
                            split=False, surrender=False)

    # Post-split-ace one-card restriction
    if is_post_split_ace and rules.split_aces_get_one_card and n_cards >= 2:
        return LegalActions(hit=False, stand=True, double=False,
                            split=False, surrender=False)

    first_decision = (n_cards == 2)

    can_hit = True
    can_stand = True

    can_double = (
        first_decision
        and not is_post_split_ace
        and (rules.double_after_split or splits_used == 0)
    )

    if hand.can_split and splits_used < rules.max_splits:
        is_ace_pair = (hand.split_rank == 'A')
        if is_ace_pair:
            can_split = rules.resplit_aces or splits_used == 0
        else:
            can_split = True
    else:
        can_split = False

    if rules.surrender == "none" or not first_decision or splits_used > 0:
        can_surrender = False
    else:
        can_surrender = True

    return LegalActions(
        hit=can_hit,
        stand=can_stand,
        double=can_double,
        split=can_split,
        surrender=can_surrender,
    )
