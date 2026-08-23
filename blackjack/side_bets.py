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
    """EV of the Perfect Pairs side bet given the two cards dealt.

    Since we don't track suits, model:
    - P(perfect pair) = P(same rank, same suit) using uniform suit assumption
    - P(coloured pair) = P(same rank, same colour) - P(perfect pair)
    - P(mixed pair) = P(same rank, different colour)

    Returns EV per unit staked. Negative means house edge.
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

    # It is a pair. P(pair) = n(r) / total after removing card1
    n_rank_after = max(0, c1_in_shoe - 1)
    total_after = total - 1
    if total_after <= 0 or n_rank_after <= 0:
        return -1.0

    # P(any pair)
    p_pair = n_rank_after / total_after

    # Suit modelling (uniform assumption over remaining cards of same rank):
    # In a full shoe, each rank has 4 suits per deck, so suits_per_rank = 4*decks
    # After seeing card1, there are n_rank_after copies of the same rank left.
    # We model the suit of card1 as drawn uniformly from the 4 suit types.
    # Given card1's suit, among the n_rank_after remaining:
    #   - 1 copy shares the exact same suit per deck → but across decks uniformly
    # Use proportional model:
    #   P(perfect | pair) ≈ 1 / (4 * decks)   (one exact suit match per deck)
    #   P(coloured | pair) ≈ (red or black same) - perfect
    #   P(mixed | pair) = remaining

    suits_total = _SUITS_PER_RANK * shoe.decks  # total suits of this rank
    # Remove the first card's suit slot
    suits_remaining = suits_total - 1
    if suits_remaining <= 0:
        p_perfect_given_pair = 0.0
        p_coloured_given_pair = 0.0
    else:
        # Exact same suit: (decks - 1) remaining copies of same suit
        # (one per deck, minus the one already seen)
        same_suit_count = shoe.decks - 1
        # Same colour, different suit: there is 1 other suit of same colour per deck
        same_colour_diff_suit = shoe.decks * 1  # 1 other suit per deck in same colour
        p_perfect_given_pair = same_suit_count / suits_remaining
        p_same_colour_given_pair = (same_suit_count + same_colour_diff_suit) / suits_remaining
        p_coloured_given_pair = p_same_colour_given_pair - p_perfect_given_pair

    p_perfect = p_pair * p_perfect_given_pair
    p_coloured = p_pair * p_coloured_given_pair
    p_mixed = p_pair - p_perfect - p_coloured

    ev = (
        p_perfect * payout_perfect
        + p_coloured * payout_coloured
        + p_mixed * payout_mixed
        - (1 - p_pair) * 1.0  # lose the stake when no pair
    )
    # Subtract the stake for pair wins (payout is in addition to stake return)
    # Correct: EV = sum(p_outcome * net_return) where net_return on loss = -1
    # On win at N:1, net return = N (get back stake + N units)
    # The above formula is already correct since we use payout (net win)
    # and subtract losing probability
    return ev


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
    """EV of the 21+3 side bet (player card1, card2 + dealer upcard).

    Models poker hand probabilities using uniform suit distribution
    over remaining ranks in the shoe.

    Returns EV per unit staked.
    """
    r1 = _normalise(card1)
    r2 = _normalise(card2)
    ru = _normalise(dealer_upcard)

    ranks = [r1, r2, ru]

    # Three of a kind: all same rank
    is_toak = (r1 == r2 == ru)

    # Straight: consecutive ranks
    is_straight_hand = _is_straight(ranks)

    # Flush / straight flush: need all same suit (modelled uniformly)
    # P(all three share the same suit) using uniform suit assumption.
    # Each card independently has 4 suit options; P(all same suit) = 4/4^3 = 1/16
    # But we need to condition on the actual cards drawn from the shoe.
    # Use simplified uniform model: P(flush) ≈ 1/16 per combination
    # More precisely: given the ranks are fixed, P(all same suit) = 1/(4^2) = 1/16
    # since suit of first card is fixed, second must match (prob 1/4),
    # third must match (prob 1/4).
    p_flush_given_ranks = 1.0 / 16.0

    # Compute overall probabilities
    if is_toak:
        # Three of a kind: ranks match. Suit determines flush.
        # P(straight flush | TOK) = 0 (can't be straight if all same rank)
        p_straight_flush = 0.0
        p_toak = (1.0 - p_flush_given_ranks)  # TOK but not flush
        p_flush = p_flush_given_ranks  # flush TOK — paid as TOK in most rules
        # Actually TOK flush is a "suited three of a kind" - pays TOK payout
        # Per most 21+3 rules, three of a kind beats flush regardless of suit
        p_toak_total = 1.0
        ev = p_toak_total * payout_three_of_a_kind - 0.0
        # Wait — we should compute probability that this hand actually forms given shoe
        # For simplicity: the ranks are given (card1, card2, dealer_upcard are already dealt)
        # so probability = 1.0 for the rank configuration
        return payout_three_of_a_kind  # guaranteed win
    elif is_straight_hand:
        p_straight_flush = p_flush_given_ranks
        p_straight_only = 1.0 - p_flush_given_ranks
        ev = (p_straight_flush * payout_straight_flush
              + p_straight_only * payout_straight)
        return ev
    else:
        # Could still be a flush
        p_flush_only = p_flush_given_ranks
        p_nothing = 1.0 - p_flush_only
        ev = p_flush_only * payout_flush - p_nothing * 1.0
        return ev


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
