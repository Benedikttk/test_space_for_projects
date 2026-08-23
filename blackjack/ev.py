"""Expected-value engine.

Provides:
- dealer_distribution: probability over dealer final totals (S17/H17-aware).
- action_evs: EV dict for all legal actions at a decision point.
- split_ev: recursive split EV respecting DAS, RSA, max_splits,
  split-aces-one-card restriction.
- best_action: returns (action_name, ev) for the highest-EV action.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from blackjack.rules import RuleSet
from blackjack.shoe import Shoe, ALL_RANKS
from blackjack.hand import hand_total, RANK_VALUE, _normalise
from blackjack.actions import get_legal_actions
from blackjack.hand import Hand


# ---------------------------------------------------------------------------
# Dealer distribution
# ---------------------------------------------------------------------------

def dealer_distribution(
    upcard: str,
    shoe: Shoe,
    rules: RuleSet,
) -> Dict[int, float]:
    """Return P(dealer_final_total) given upcard and shoe composition.

    Bust is keyed as 22.  The shoe is NOT mutated.
    """
    counts = shoe.snapshot()
    total_in_shoe = sum(counts.values())
    upcard_n = _normalise(upcard)
    counts[upcard_n] = max(0, counts[upcard_n] - 1)
    remaining = total_in_shoe - 1

    def _recurse(
        dealer_cards: List[str],
        prob: float,
        counts: Dict[str, int],
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
        for rank in ALL_RANKS:
            if counts[rank] <= 0:
                continue
            p_next = counts[rank] / remaining
            new_counts = dict(counts)
            new_counts[rank] -= 1
            sub = _recurse(dealer_cards + [rank], prob * p_next,
                           new_counts, remaining - 1)
            for k, v in sub.items():
                result[k] = result.get(k, 0.0) + v
        return result

    return _recurse([upcard_n], 1.0, counts, remaining)


# ---------------------------------------------------------------------------
# EV helpers
# ---------------------------------------------------------------------------

def _stand_ev(
    player_total: int,
    dealer_dist: Dict[int, float],
    is_blackjack: bool = False,
    rules: RuleSet | None = None,
) -> float:
    if rules is None:
        rules = RuleSet()
    ev = 0.0
    bj_payout = rules.blackjack_payout if is_blackjack else 1.0
    for dealer_total, prob in dealer_dist.items():
        if dealer_total == 22:          # dealer bust
            ev += prob * bj_payout
        elif player_total > 21:         # player bust
            ev -= prob
        elif is_blackjack and dealer_total != 21:
            ev += prob * bj_payout
        elif is_blackjack and dealer_total == 21:
            if rules.natural_beats_dealer_21:
                ev += prob * bj_payout
            # else push → +0
        elif player_total > dealer_total:
            ev += prob
        elif player_total < dealer_total:
            ev -= prob
        # push → +0
    return ev


def _hit_ev(
    player_cards: List[str],
    dealer_dist: Dict[int, float],
    shoe: Shoe,
    rules: RuleSet,
    depth: int = 0,
    max_depth: int = 8,
) -> float:
    if depth > max_depth:
        tot, _ = hand_total(player_cards)
        return _stand_ev(tot, dealer_dist, rules=rules)

    ev = 0.0
    total_remaining = shoe.total_remaining
    for rank in ALL_RANKS:
        if shoe.counts[rank] <= 0:
            continue
        p = shoe.counts[rank] / total_remaining
        new_cards = player_cards + [rank]
        new_tot, _ = hand_total(new_cards)
        if new_tot > 21:
            ev += p * (-1.0)
        else:
            ev_stand = _stand_ev(new_tot, dealer_dist, rules=rules)
            ev_hit = _hit_ev(new_cards, dealer_dist, shoe, rules,
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
    """Return {action: ev} for every legal action."""
    legal = get_legal_actions(hand, rules, splits_used, is_post_split_ace,
                              dealer_upcard)
    dist = dealer_distribution(dealer_upcard, shoe, rules)
    tot, _ = hand_total(hand.cards)
    result: Dict[str, float] = {}

    if legal.stand:
        result["stand"] = _stand_ev(tot, dist,
                                     is_blackjack=hand.is_blackjack,
                                     rules=rules)
    if legal.hit:
        result["hit"] = _hit_ev(hand.cards, dist, shoe, rules)

    if legal.double:
        ev_dbl = 0.0
        total_remaining = shoe.total_remaining
        for rank in ALL_RANKS:
            if shoe.counts[rank] <= 0:
                continue
            p = shoe.counts[rank] / total_remaining
            new_cards = hand.cards + [rank]
            new_tot, _ = hand_total(new_cards)
            if new_tot > 21:
                ev_dbl += p * (-2.0)
            else:
                ev_dbl += p * (_stand_ev(new_tot, dist, rules=rules) * 2.0)
        result["double"] = ev_dbl

    if legal.split:
        result["split"] = split_ev(hand, dealer_upcard, shoe, rules,
                                    splits_used)

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
    """Recursive EV for splitting *hand*.

    Returns the combined EV across both resulting hands.
    """
    if not hand.can_split:
        raise ValueError(f"Hand {hand} is not splittable")
    if splits_used >= rules.max_splits:
        raise ValueError("max_splits exceeded")

    split_rank = hand.split_rank
    is_ace_split = (split_rank == 'A')
    new_splits_used = splits_used + 1
    is_psa = is_ace_split

    total_remaining = shoe.total_remaining
    ev_per_hand = 0.0

    for rank in ALL_RANKS:
        if shoe.counts[rank] <= 0:
            continue
        p = shoe.counts[rank] / total_remaining
        sub_hand = Hand(
            cards=[split_rank, rank],
            splits_used=new_splits_used,
            is_post_split_ace=is_psa,
        )
        sub_evs = action_evs(sub_hand, dealer_upcard, shoe, rules,
                              new_splits_used, is_psa)
        best = max(sub_evs.values()) if sub_evs else 0.0
        ev_per_hand += p * best

    return 2.0 * ev_per_hand


def best_action(ev_dict: Dict[str, float]) -> Tuple[str, float]:
    """Return (action_name, ev) for the highest-EV action."""
    if not ev_dict:
        return ("stand", 0.0)
    best = max(ev_dict, key=lambda k: ev_dict[k])
    return best, ev_dict[best]
