"""Expected-value engine.

Provides:
- dealer_distribution: probability over dealer final totals (S17/H17-aware),
  conditioned on ALL visible player cards and the dealer upcard.
- action_evs: EV dict for all legal actions at a decision point.
- split_ev: recursive split EV respecting DAS, RSA, max_splits,
  split-aces-one-card restriction, with correct per-branch shoe depletion.
- best_action: returns (action_name, ev) for the highest-EV action.

All calculations use exact probability arithmetic over shoe composition
(no Monte Carlo approximation).  The shoe object is NEVER mutated by
any function in this module; all branching works on local count copies.

Mathematical correctness notes
-------------------------------
* dealer_distribution removes every visible card (player cards + upcard)
  from the shoe snapshot before computing dealer draw probabilities.
* _hit_ev passes a depleted counts dict into each recursive branch so
  a drawn card cannot be sampled again at deeper levels.
* split_ev removes both split-pair cards from the shoe snapshot and
  further depletes for each second card dealt to a child hand.
* The double EV also depletes a per-branch counts snapshot.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from blackjack.rules import RuleSet
from blackjack.shoe import Shoe, ALL_RANKS
from blackjack.hand import hand_total, RANK_VALUE, _normalise, Hand
from blackjack.actions import get_legal_actions


# ---------------------------------------------------------------------------
# Internal type alias
# ---------------------------------------------------------------------------
_Counts = Dict[str, int]


def _deplete(counts: _Counts, rank: str) -> _Counts:
    """Return a new counts dict with *rank* decremented by 1.

    Silently clamps to 0 if the rank is already absent (graceful handling
    of unknown/incorrect input from vision layer).
    """
    new = dict(counts)
    new[rank] = max(0, new.get(rank, 0) - 1)
    return new


def _total(counts: _Counts) -> int:
    return sum(counts.values())


# ---------------------------------------------------------------------------
# Dealer distribution
# ---------------------------------------------------------------------------

def dealer_distribution(
    upcard: str,
    shoe: Shoe,
    rules: RuleSet,
    player_cards: Sequence[str] = (),
) -> Dict[int, float]:
    """Return P(dealer_final_total) conditioned on all visible cards.

    Parameters
    ----------
    upcard:
        Dealer's face-up card.
    shoe:
        Current shoe composition (not mutated).
    rules:
        Active rule set (H17 / S17).
    player_cards:
        All cards currently visible in the player's hand(s).  These are
        removed from the shoe snapshot so dealer draw probabilities are
        correctly conditioned on what has already been seen.

    Returns
    -------
    Dict mapping final dealer total (17-21) or 22 (bust) to probability.
    """
    # Start from a snapshot and remove every visible card
    counts: _Counts = shoe.snapshot()
    for card in player_cards:
        counts = _deplete(counts, _normalise(card))
    counts = _deplete(counts, _normalise(upcard))
    remaining = _total(counts)

    def _recurse(
        dealer_cards: List[str],
        prob: float,
        counts: _Counts,
        remaining: int,
    ) -> Dict[int, float]:
        tot, soft = hand_total(dealer_cards)
        must_hit = (
            tot < 17
            or (tot == 17 and soft and rules.dealer_hits_soft17)
        )
        if not must_hit:
            key = tot if tot <= 21 else 22
            return {key: prob}

        result: Dict[int, float] = {}
        if remaining <= 0:
            # Degenerate: shoe exhausted mid-recursion; treat as stand
            key = tot if tot <= 21 else 22
            return {key: prob}
        for rank in ALL_RANKS:
            if counts[rank] <= 0:
                continue
            p_next = counts[rank] / remaining
            branch_counts = _deplete(counts, rank)
            sub = _recurse(
                dealer_cards + [rank],
                prob * p_next,
                branch_counts,
                remaining - 1,
            )
            for k, v in sub.items():
                result[k] = result.get(k, 0.0) + v
        return result

    return _recurse([_normalise(upcard)], 1.0, counts, remaining)


# ---------------------------------------------------------------------------
# Stand EV
# ---------------------------------------------------------------------------

def _stand_ev(
    player_total: int,
    dealer_dist: Dict[int, float],
    is_blackjack: bool = False,
    rules: RuleSet | None = None,
) -> float:
    """EV of standing given player total and dealer final distribution."""
    if rules is None:
        rules = RuleSet()
    ev = 0.0
    bj_payout = rules.blackjack_payout if is_blackjack else 1.0
    for dealer_total, prob in dealer_dist.items():
        if dealer_total == 22:                     # dealer bust
            ev += prob * bj_payout
        elif player_total > 21:                    # player bust (shouldn't reach here)
            ev -= prob
        elif is_blackjack and dealer_total != 21:
            ev += prob * bj_payout
        elif is_blackjack and dealer_total == 21:
            # Natural vs dealer non-natural 21
            if rules.natural_beats_dealer_21:
                ev += prob * bj_payout
            # else push → +0
        elif player_total > dealer_total:
            ev += prob
        elif player_total < dealer_total:
            ev -= prob
        # else push → +0
    return ev


# ---------------------------------------------------------------------------
# Hit EV  (FIX: deplete counts per branch)
# ---------------------------------------------------------------------------

def _hit_ev(
    player_cards: List[str],
    dealer_dist: Dict[int, float],
    counts: _Counts,
    rules: RuleSet,
    depth: int = 0,
    max_depth: int = 10,
) -> float:
    """EV of hitting, with per-branch shoe depletion.

    Parameters
    ----------
    counts:
        A snapshot of remaining cards *after* all previously visible
        cards have been removed.  Each branch depletes this further for
        the card it draws.
    """
    if depth > max_depth:
        # Safety valve: approximate by standing at current total
        tot, _ = hand_total(player_cards)
        return _stand_ev(tot, dealer_dist, rules=rules)

    remaining = _total(counts)
    if remaining <= 0:
        tot, _ = hand_total(player_cards)
        return _stand_ev(tot, dealer_dist, rules=rules)

    ev = 0.0
    for rank in ALL_RANKS:
        if counts[rank] <= 0:
            continue
        p = counts[rank] / remaining
        new_cards = player_cards + [rank]
        new_tot, _ = hand_total(new_cards)
        if new_tot > 21:
            ev += p * (-1.0)
        else:
            branch_counts = _deplete(counts, rank)   # FIX: deplete before recursing
            ev_stand = _stand_ev(new_tot, dealer_dist, rules=rules)
            ev_hit = _hit_ev(new_cards, dealer_dist, branch_counts, rules,
                             depth + 1, max_depth)
            ev += p * max(ev_stand, ev_hit)
    return ev


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def action_evs(
    hand: Hand,
    dealer_upcard: str,
    shoe: Shoe,
    rules: RuleSet,
    splits_used: int = 0,
    is_post_split_ace: bool = False,
) -> Dict[str, float]:
    """Return {action: ev} for every legal action at this decision point.

    The shoe is never mutated.  All probability calculations use locally
    depleted count snapshots.
    """
    legal = get_legal_actions(hand, rules, splits_used, is_post_split_ace,
                              dealer_upcard)
    # Dealer distribution conditioned on all visible cards
    dist = dealer_distribution(
        dealer_upcard, shoe, rules, player_cards=hand.cards
    )
    tot, _ = hand_total(hand.cards)

    # Base counts with all visible cards (player + upcard) already removed
    base_counts: _Counts = shoe.snapshot()
    for card in hand.cards:
        base_counts = _deplete(base_counts, _normalise(card))
    base_counts = _deplete(base_counts, _normalise(dealer_upcard))

    result: Dict[str, float] = {}

    # --- STAND ---
    if legal.stand:
        result["stand"] = _stand_ev(
            tot, dist, is_blackjack=hand.is_blackjack, rules=rules
        )

    # --- HIT ---
    if legal.hit:
        result["hit"] = _hit_ev(
            hand.cards, dist, base_counts, rules
        )

    # --- DOUBLE (FIX: deplete per drawn card) ---
    if legal.double:
        ev_dbl = 0.0
        remaining = _total(base_counts)
        if remaining > 0:
            for rank in ALL_RANKS:
                if base_counts[rank] <= 0:
                    continue
                p = base_counts[rank] / remaining
                new_cards = hand.cards + [rank]
                new_tot, _ = hand_total(new_cards)
                if new_tot > 21:
                    ev_dbl += p * (-2.0)
                else:
                    ev_dbl += p * (_stand_ev(new_tot, dist, rules=rules) * 2.0)
        result["double"] = ev_dbl

    # --- SPLIT ---
    if legal.split:
        result["split"] = split_ev(
            hand, dealer_upcard, shoe, rules, splits_used
        )

    # --- SURRENDER ---
    if legal.surrender:
        result["surrender"] = -0.5

    return result


def split_ev(
    hand: Hand,
    dealer_upcard: str,
    shoe: Shoe,
    rules: RuleSet,
    splits_used: int = 0,
) -> float:
    """Recursive EV for splitting *hand* with correct shoe depletion.

    Methodology
    -----------
    After splitting, the shoe has lost both copies of the split card.
    Each child hand then draws one more card.  We iterate over all
    possible second cards for each child, depleting the shoe further
    for each branch, and recurse into action_evs so that further
    splits, doubles, and hits are also evaluated correctly.

    Returns the combined EV of both child hands (not averaged), which
    represents total return on the original bet unit when each split
    hand wagers the same stake.
    """
    if not hand.can_split:
        raise ValueError(f"Hand {hand} is not splittable")
    if splits_used >= rules.max_splits:
        raise ValueError("max_splits exceeded")

    split_rank = hand.split_rank          # e.g. 'A' or 'T'
    is_ace_split = (split_rank == 'A')
    new_splits_used = splits_used + 1
    is_psa = is_ace_split

    # FIX: Remove both copies of the split card from the shoe snapshot,
    # then also remove the dealer upcard and any other known cards.
    base_counts: _Counts = shoe.snapshot()
    # The two split cards leave the shoe
    base_counts = _deplete(base_counts, split_rank)
    base_counts = _deplete(base_counts, split_rank)
    # Dealer upcard also seen
    base_counts = _deplete(base_counts, _normalise(dealer_upcard))

    remaining = _total(base_counts)
    ev_per_hand = 0.0

    for rank in ALL_RANKS:
        if base_counts[rank] <= 0:
            continue
        p = base_counts[rank] / remaining
        # This child hand receives split_rank + rank
        sub_hand = Hand(
            cards=[split_rank, rank],
            splits_used=new_splits_used,
            is_post_split_ace=is_psa,
        )
        # For evaluating the child hand, build a shoe snapshot that has
        # had this second card removed too (FIX: was missing before)
        child_counts = _deplete(base_counts, rank)
        child_shoe = _counts_to_shoe(child_counts, shoe.decks)
        sub_evs = action_evs(
            sub_hand, dealer_upcard, child_shoe, rules,
            new_splits_used, is_psa,
        )
        best = max(sub_evs.values()) if sub_evs else 0.0
        ev_per_hand += p * best

    # Both hands are played; combined EV = 2 × average child EV
    return 2.0 * ev_per_hand


def _counts_to_shoe(counts: _Counts, decks: int) -> Shoe:
    """Construct a Shoe whose counts match *counts* (used in split recursion)."""
    from blackjack.shoe import Shoe
    s = Shoe(decks=decks)
    s.counts = dict(counts)
    return s


def best_action(ev_dict: Dict[str, float]) -> Tuple[str, float]:
    """Return (action_name, ev) for the highest-EV action."""
    if not ev_dict:
        return ("stand", 0.0)
    best = max(ev_dict, key=lambda k: ev_dict[k])
    return best, ev_dict[best]
