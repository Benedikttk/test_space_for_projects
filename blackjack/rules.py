"""RuleSet – complete blackjack rule configuration.

Every flag that changes which actions are legal or how payouts are
calculated is encoded here so that all downstream modules (EV engine,
action gating, strategy tables) derive their behaviour from a single
authoritative source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SurrenderMode = Literal["none", "late", "early"]


@dataclass(frozen=True)
class RuleSet:
    """Immutable collection of house rules for one blackjack variant.

    Parameters
    ----------
    dealer_hits_soft17:
        True  → dealer hits soft-17  (H17, more common in Vegas 6-deck games).
        False → dealer stands on all 17s (S17, player-favourable).
    double_after_split:
        Whether the player may double down on a hand created by splitting.
    resplit_aces:
        Whether aces that are split may themselves be re-split if another
        ace is dealt.
    max_splits:
        Maximum number of times a hand may be split.  Typical values: 1–4.
        A value of 1 means the original pair may be split once (producing
        two hands); 4 means up to four hands from the original pair.
    split_aces_get_one_card:
        The canonical casino restriction: after splitting aces each new hand
        receives exactly one additional card and may not hit further
        (independent of `resplit_aces`).
    blackjack_payout:
        Multiplier applied to the original bet on a natural blackjack win:
        1.5 → 3:2 (standard),  1.2 → 6:5 (common on single-deck),
        1.0 → 1:1 (even-money, very player-unfavourable).
    surrender:
        ``"none"``  – surrender is not offered.
        ``"late"``  – late surrender: allowed only after dealer checks for BJ.
        ``"early"`` – early surrender: allowed before dealer checks for BJ
                       (very rare, strongly player-favourable).
    natural_beats_dealer_21:
        True → a player natural (blackjack) beats a dealer total of 21 made
        with ≥3 cards (standard rule).  Set False only for exotic variants.
    """

    dealer_hits_soft17: bool = False          # S17 default (favourable)
    double_after_split: bool = True           # DAS allowed
    resplit_aces: bool = False                # RSA off by default
    max_splits: int = 3                       # up to 4 hands
    split_aces_get_one_card: bool = True      # standard casino rule
    blackjack_payout: float = 1.5            # 3:2
    surrender: SurrenderMode = "late"        # late surrender
    natural_beats_dealer_21: bool = True
    insurance: bool = True                   # insurance/even-money offered
    dealer_peeks: bool = True                # dealer peeks for BJ on A or T

    # ------------------------------------------------------------------
    # Derived helpers (no extra state, all computed from the flags above)
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.max_splits < 1:
            raise ValueError("max_splits must be >= 1")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack_payout must be positive")
        if self.surrender not in {"none", "late", "early"}:
            raise ValueError(f"Unknown surrender mode: {self.surrender!r}")

    # Convenience constructors for the two most common dealer-stand rules
    @classmethod
    def s17(cls, **kwargs) -> "RuleSet":
        """Return a RuleSet with S17 (dealer stands on soft-17)."""
        return cls(dealer_hits_soft17=False, **kwargs)

    @classmethod
    def h17(cls, **kwargs) -> "RuleSet":
        """Return a RuleSet with H17 (dealer hits soft-17)."""
        return cls(dealer_hits_soft17=True, **kwargs)


# ---------------------------------------------------------------------------
# Pre-built canonical rule sets referenced by tests and notebooks
# ---------------------------------------------------------------------------

#: Las Vegas Strip multi-deck (very common benchmark)
STRIP_S17 = RuleSet(
    dealer_hits_soft17=False,
    double_after_split=True,
    resplit_aces=False,
    max_splits=3,
    split_aces_get_one_card=True,
    blackjack_payout=1.5,
    surrender="late",
)

#: Downtown / Fremont Street multi-deck H17
DOWNTOWN_H17 = RuleSet(
    dealer_hits_soft17=True,
    double_after_split=True,
    resplit_aces=False,
    max_splits=3,
    split_aces_get_one_card=True,
    blackjack_payout=1.5,
    surrender="late",
)

#: Liberal rules benchmark (player-favourable)
LIBERAL = RuleSet(
    dealer_hits_soft17=False,
    double_after_split=True,
    resplit_aces=True,
    max_splits=4,
    split_aces_get_one_card=True,
    blackjack_payout=1.5,
    surrender="early",
)

#: Worst-case tourist trap rules (player-unfavourable)
TOURIST_TRAP = RuleSet(
    dealer_hits_soft17=True,
    double_after_split=False,
    resplit_aces=False,
    max_splits=1,
    split_aces_get_one_card=True,
    blackjack_payout=1.2,   # 6:5
    surrender="none",
)
