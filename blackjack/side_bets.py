"""Side bet EV calculators.

All EVs are per unit staked on the side bet (not the main bet).

Perfect Pairs
-------------
Pays when your first two cards are a pair.
Three tiers (typical payouts, configurable):
  - Mixed pair  (same rank, different colour):  5:1
  - Coloured pair (same rank, same colour):     10:1
  - Perfect pair (same rank, same suit):        30:1

Since the engine tracks rank but not suit, we model suit distribution
as uniform over remaining cards of that rank.

21+3
----
Pays when your two cards + dealer upcard form a poker hand:
  - Flush (same suit):           5:1
  - Straight (consecutive):      10:1
  - Three of a kind:             30:1
  - Straight flush:              40:1

Again modelled with uniform suit distribution over remaining ranks.
"""

from __future__ import annotations

from blackjack.shoe import Shoe
from blackjack.hand import _normalise

# Rank order for straight detection (A can be low or high)
_RANK_ORDER = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5,
               '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10}
# Suits per rank in a single deck (4 suits: clubs, diamonds, hearts, spades)
_SUITS_PER_RANK = 4
# Two red suits, two black suits
_RED_COUNT = 2   # hearts + diamonds
_BLACK_COUNT = 2  # clubs + spades


def perfect_pairs_ev(
    shoe: Shoe,
    card1: str,
    card2: str,
    payout_mixed: float = 5.0,
    payout_coloured: float = 10.0,
    payout_perfect: float = 30.0,
) -> float:
    """EV of Perfect Pairs given the player's two already-dealt cards.

    - If the two cards are not a pair, the side bet loses immediately.
    - If they are a pair, we estimate tier probabilities (perfect/coloured/mixed)
      from remaining same-rank cards using a uniform suit approximation.

    Returns net EV per unit staked (N:1 payout convention).
    """
    r1 = _normalise(card1)
    r2 = _normalise(card2)

    counts = shoe.snapshot()
    # Remove card1 from shoe before computing probabilities for card2
    c1_in_shoe = counts.get(r1, 0)
    total = sum(counts.values())

    if total <= 0:
        return -1.0

    if r1 != r2:
        # Not a pair — always loses
        return -1.0

    # It is a pair. Use remaining same-rank population to estimate suit tiers.
    n_rank_after = max(0, c1_in_shoe - 1)
    total_after = total - 1
    if total_after <= 0 or n_rank_after <= 0:
        return -1.0

    same_suit_remaining = n_rank_after // 4
    same_colour_remaining = n_rank_after // 2

    p_perfect_given_pair = same_suit_remaining / n_rank_after
    p_same_colour_given_pair = same_colour_remaining / n_rank_after
    p_coloured_given_pair = max(0.0, p_same_colour_given_pair - p_perfect_given_pair)
    p_mixed_given_pair = max(0.0, 1.0 - p_perfect_given_pair - p_coloured_given_pair)

    return (
        p_perfect_given_pair * payout_perfect
        + p_coloured_given_pair * payout_coloured
        + p_mixed_given_pair * payout_mixed
    )


def _is_straight(ranks: list) -> bool:
    """Check if three ranks form a straight (consecutive values)."""
    vals = sorted(_RANK_ORDER.get(r, 0) for r in ranks)
    # Normal straight
    if vals[2] - vals[1] == 1 and vals[1] - vals[0] == 1:
        return True
    # Ace-high wrap: A(1), not modelled since T=10 (no J/Q/K separation)
    return False


def twenty_one_plus_three_ev(
    shoe: Shoe,
    card1: str,
    card2: str,
    dealer_upcard: str,
    payout_flush: float = 5.0,
    payout_straight: float = 10.0,
    payout_three_of_a_kind: float = 30.0,
    payout_straight_flush: float = 40.0,
) -> float:
    """EV of the 21+3 side bet given three already-dealt ranks.

    Suits are not tracked, so we use a uniform suit prior:
    P(all three cards share suit) = 1/16.

    Rank configuration drives outcomes:
    - three of a kind: guaranteed TOK payout
    - straight: straight flush with 1/16, straight otherwise
    - otherwise: flush with 1/16, lose otherwise

    Returns net EV per unit staked (N:1 payout convention).
    """
    r1 = _normalise(card1)
    r2 = _normalise(card2)
    ru = _normalise(dealer_upcard)

    ranks = [r1, r2, ru]

    # Three of a kind: all same rank
    is_toak = (r1 == r2 == ru)

    # Straight: consecutive ranks
    is_straight_hand = _is_straight(ranks)

    p_flush_given_ranks = 1.0 / 16.0

    if is_toak:
        return payout_three_of_a_kind
    if is_straight_hand:
        p_sf = p_flush_given_ranks
        return p_sf * payout_straight_flush + (1.0 - p_sf) * payout_straight
    p_flush = p_flush_given_ranks
    return p_flush * payout_flush + (1.0 - p_flush) * (-1.0)


def side_bet_summary(
    shoe: Shoe,
    card1: str,
    card2: str,
    dealer_upcard: str,
) -> dict:
    """Return a dict with EVs for all side bets:
      'perfect_pairs': float
      'twenty_one_plus_three': float
      'insurance': float or None (None if dealer upcard is not A)
    """
    from blackjack.ev import insurance_ev

    r_upcard = _normalise(dealer_upcard)
    ins = insurance_ev(shoe, dealer_upcard) if r_upcard == 'A' else None

    return {
        "perfect_pairs": perfect_pairs_ev(shoe, card1, card2),
        "twenty_one_plus_three": twenty_one_plus_three_ev(
            shoe, card1, card2, dealer_upcard
        ),
        "insurance": ins,
    }
