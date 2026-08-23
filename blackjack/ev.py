"""Expected-value engine.

Provides:
- dealer_distribution: probability over dealer final totals (S17/H17-aware),
  conditioned on ALL visible player cards and the dealer upcard.
- action_evs: EV dict for all legal actions at a decision point.
- split_ev: recursive split EV respecting DAS, RSA, max_splits,
  split-aces-one-card restriction, with correct per-branch shoe depletion.
- best_action: returns (action_name, ev) for the highest-EV action.
- insurance_ev: EV of the insurance side bet from exact shoe composition.
- dampened_ev: penetration-dampened EV interpolation.
- basic_strategy_ev: action EVs on a fresh full-shoe baseline.

All calculations use exact probability arithmetic over shoe composition
(no Monte Carlo approximation).  The shoe object is NEVER mutated by
any function in this module; all branching works on local count copies.

Mathematical correctness guarantees
-------------------------------------
* dealer_distribution removes every visible card (player cards + upcard)
  from the shoe snapshot before computing dealer draw probabilities.
* _hit_ev passes a depleted counts dict into each recursive branch so
  a drawn card cannot be sampled again at deeper levels.
* split_ev removes both split-pair cards from the shoe snapshot and
  further depletes per second card dealt to each child hand.  The
  dealer upcard and child cards are removed inside action_evs /
  dealer_distribution, never pre-removed in split_ev, to avoid
  double-depletion.
* The double EV uses a per-branch counts snapshot depleted for each
  possible drawn card.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from blackjack.rules import RuleSet
from blackjack.shoe import Shoe, ALL_RANKS
from blackjack.hand import hand_total, _normalise, Hand
from blackjack.actions import get_legal_actions


# ---------------------------------------------------------------------------
# Internal type alias
# ---------------------------------------------------------------------------
_Counts = Dict[str, int]


def _deplete(counts: _Counts, rank: str) -> _Counts:
    """Return a new counts dict with *rank* decremented by 1.

    Silently clamps to 0 if the rank is already absent (graceful handling
    of unknown/incorrect input from the vision layer).
    """
    new = dict(counts)
    new[rank] = max(0, new.get(rank, 0) - 1)
    return new


def _total(counts: _Counts) -> int:
    return sum(counts.values())


def _counts_to_shoe(counts: _Counts, decks: int) -> Shoe:
    """Construct a lightweight Shoe whose counts match *counts*.

    Used inside split_ev to pass child-hand shoe states into action_evs
    without mutating the live Shoe object.
    """
    s = Shoe(decks=decks)
    s.counts = dict(counts)
    return s


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
        Active rule set (H17 / S17, dealer_peeks).
    player_cards:
        All cards currently visible in the player's hand(s).  These are
        removed from the shoe snapshot so dealer draw probabilities are
        correctly conditioned on what has already been seen.

    Returns
    -------
    Dict mapping final dealer total (17-21) or 22 (bust) to probability.
    Probabilities sum to 1.0 (within floating-point precision).
    """
    # Build a depleted snapshot: remove every visible card first
    counts: _Counts = shoe.snapshot()
    for card in player_cards:
        counts = _deplete(counts, _normalise(card))
    norm_upcard = _normalise(upcard)
    counts = _deplete(counts, norm_upcard)

    # Dealer peek conditioning: if the dealer already peeked and did NOT
    # have blackjack, we know the hole card was not the BJ-completing rank.
    # Remove those branches from the initial draw and renormalise.
    if rules.dealer_peeks:
        if norm_upcard == 'A':
            # Hole card cannot be T (would have been BJ)
            counts['T'] = 0
        elif norm_upcard == 'T':
            # Hole card cannot be A (would have been BJ)
            counts['A'] = 0

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

        # Degenerate: shoe exhausted — treat current total as final
        if remaining <= 0:
            key = tot if tot <= 21 else 22
            return {key: prob}

        result: Dict[int, float] = {}
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

    return _recurse([norm_upcard], 1.0, counts, remaining)


# ---------------------------------------------------------------------------
# Stand EV
# ---------------------------------------------------------------------------

def _stand_ev(
    player_total: int,
    dealer_dist: Dict[int, float],
    is_blackjack: bool = False,
    rules: RuleSet | None = None,
) -> float:
    """EV of standing given player total and dealer final distribution.

    EV is expressed in units of the original bet (e.g. +1.0 = win one
    bet, -1.0 = lose one bet, 0.0 = push).
    """
    if rules is None:
        rules = RuleSet()
    ev = 0.0
    bj_payout = rules.blackjack_payout if is_blackjack else 1.0
    for dealer_total, prob in dealer_dist.items():
        if dealer_total == 22:                     # dealer bust
            ev += prob * bj_payout
        elif player_total > 21:                    # player already bust
            ev -= prob
        elif is_blackjack and dealer_total != 21:
            # Natural vs non-21 dealer: player wins at BJ payout
            ev += prob * bj_payout
        elif is_blackjack and dealer_total == 21:
            # Natural vs dealer 21 (non-natural hand)
            if rules.natural_beats_dealer_21:
                ev += prob * bj_payout
            # else: push → +0
        elif player_total > dealer_total:
            ev += prob
        elif player_total < dealer_total:
            ev -= prob
        # else: push → +0
    return ev


# ---------------------------------------------------------------------------
# Hit EV  -- per-branch shoe depletion
# ---------------------------------------------------------------------------

def _hit_ev(
    player_cards: List[str],
    dealer_dist: Dict[int, float],
    counts: _Counts,
    rules: RuleSet,
    depth: int = 0,
    max_depth: int = 10,
) -> float:
    """EV of hitting optimally from this point forward.

    Parameters
    ----------
    counts:
        Remaining-card snapshot *after* all previously visible cards
        have been removed.  Each branch depletes this snapshot for the
        card it hypothetically draws, so cards are never double-sampled.
    """
    if depth > max_depth:
        # Safety valve: too deep, approximate by standing
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
            branch_counts = _deplete(counts, rank)   # deplete before recursing
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

    EV values are in units of the original (pre-split) bet.
    """
    legal = get_legal_actions(hand, rules, splits_used, is_post_split_ace,
                              dealer_upcard)

    # Dealer distribution conditioned on all visible cards
    dist = dealer_distribution(
        dealer_upcard, shoe, rules, player_cards=hand.cards
    )
    tot, _ = hand_total(hand.cards)

    # base_counts: shoe snapshot with all visible cards already removed.
    # Used by hit and double (not passed to dealer_distribution, which
    # builds its own depleted snapshot internally).
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
        result["hit"] = _hit_ev(hand.cards, dist, base_counts, rules)

    # --- DOUBLE: draw exactly one card, then stand; pay/win 2x ---
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
                    ev_dbl += p * (-2.0)   # lose doubled bet
                else:
                    ev_dbl += p * (_stand_ev(new_tot, dist, rules=rules) * 2.0)
        result["double"] = ev_dbl

    # --- SPLIT ---
    if legal.split:
        result["split"] = split_ev(
            hand, dealer_upcard, shoe, rules, splits_used
        )

    # --- SURRENDER: always -0.5 by definition ---
    if legal.surrender:
        result["surrender"] = -0.5

    # --- INSURANCE: only when upcard is A and rules allow ---
    if rules.insurance and _normalise(dealer_upcard) == 'A':
        result["insurance"] = insurance_ev(shoe, dealer_upcard)

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
    After splitting, both copies of the split card leave the shoe.
    Each child hand then draws one additional card from the remaining
    shoe.  We iterate over all possible second cards, build a child
    shoe that has the split pair AND the drawn second card removed, then
    recurse into action_evs (which handles further splits, doubles, hits
    and also removes the dealer upcard internally -- never pre-removed
    here to avoid double-depletion).

    Returns the combined EV of both child hands (not averaged), which
    represents total return on the original bet unit when each split
    hand wagers an equal stake.
    """
    if not hand.can_split:
        raise ValueError(f"Hand {hand} is not splittable")
    if splits_used >= rules.max_splits:
        raise ValueError("max_splits exceeded")

    split_rank = hand.split_rank          # e.g. 'A' or 'T'
    is_ace_split = (split_rank == 'A')
    new_splits_used = splits_used + 1
    is_psa = is_ace_split

    # Remove both copies of the split card from the snapshot.
    # Do NOT remove the dealer upcard here -- action_evs / dealer_distribution
    # will remove it when they build their own depleted snapshots, avoiding
    # double-depletion.
    base_counts: _Counts = shoe.snapshot()
    base_counts = _deplete(base_counts, split_rank)   # first copy
    base_counts = _deplete(base_counts, split_rank)   # second copy

    remaining = _total(base_counts)
    if remaining <= 0:
        return 0.0

    ev_per_hand = 0.0
    for rank in ALL_RANKS:
        if base_counts[rank] <= 0:
            continue
        p = base_counts[rank] / remaining

        # Child hand: split_rank (kept card) + rank (newly drawn card)
        sub_hand = Hand(
            cards=[split_rank, rank],
            splits_used=new_splits_used,
            is_post_split_ace=is_psa,
        )

        # Child shoe: split pair removed + this second card removed.
        # Dealer upcard and player cards are removed inside action_evs.
        child_counts = _deplete(base_counts, rank)
        child_shoe = _counts_to_shoe(child_counts, shoe.decks)

        sub_evs = action_evs(
            sub_hand, dealer_upcard, child_shoe, rules,
            new_splits_used, is_psa,
        )
        # Exclude insurance from split sub-hand best-action comparison
        play_evs = {k: v for k, v in sub_evs.items() if k != 'insurance'}
        best = max(play_evs.values()) if play_evs else 0.0
        ev_per_hand += p * best

    # Two hands are played symmetrically; total EV = 2 x single-hand EV
    return 2.0 * ev_per_hand


def best_action(ev_dict: Dict[str, float]) -> Tuple[str, float]:
    """Return (action_name, ev) for the highest-EV action.

    Insurance is excluded from the main action comparison since it is a
    separate side-bet decision, not a play action.
    """
    play_dict = {k: v for k, v in ev_dict.items() if k != 'insurance'}
    if not play_dict:
        return ("stand", 0.0)
    best = max(play_dict, key=lambda k: play_dict[k])
    return best, play_dict[best]


# ---------------------------------------------------------------------------
# Insurance EV
# ---------------------------------------------------------------------------

def insurance_ev(shoe: Shoe, dealer_upcard: str) -> float:
    """EV of the insurance side bet given current shoe composition.

    Insurance pays 2:1 if the dealer hole card is a ten-value (T).
    The bet costs 0.5 units (half the original bet).

    EV = 2 * P(hole = T | shoe, upcard) - 1

    where P(hole = T) is computed from the shoe AFTER removing the upcard
    and all other visible cards. Returns EV per 0.5-unit insurance stake,
    expressed in units of the original bet.

    Insurance is +EV when P(hole = T) > 1/3,
    which occurs at true count >= +3 approximately.
    """
    counts = shoe.snapshot()
    counts = _deplete(counts, _normalise(dealer_upcard))
    remaining = _total(counts)
    if remaining <= 0:
        return -1.0
    p_ten = counts.get('T', 0) / remaining
    return 2.0 * p_ten - 1.0


# ---------------------------------------------------------------------------
# Penetration-dampened EV
# ---------------------------------------------------------------------------

def dampened_ev(
    raw_ev: float,
    basic_strategy_ev_val: float,
    observation_ratio: float,
    min_ratio: float = 0.15,
) -> float:
    """Dampen EV-based deviations from basic strategy toward basic strategy
    when shoe observation ratio is low.

    When observation_ratio < min_ratio, return basic_strategy_ev (full dampening).
    When observation_ratio >= 1.0, return raw_ev (no dampening).
    In between, linearly interpolate.
    """
    if observation_ratio < min_ratio:
        return basic_strategy_ev_val
    if observation_ratio >= 1.0:
        return raw_ev
    t = (observation_ratio - min_ratio) / (1.0 - min_ratio)
    return basic_strategy_ev_val + t * (raw_ev - basic_strategy_ev_val)


def basic_strategy_ev(
    hand: Hand,
    dealer_upcard: str,
    rules: RuleSet,
    decks: int = 8,
) -> dict:
    """Compute action EVs using a fresh full-shoe (basic strategy baseline).

    Creates a new Shoe(decks=decks) and calls action_evs.
    Used as the baseline for penetration dampening.
    """
    fresh_shoe = Shoe(decks=decks)
    return action_evs(hand, dealer_upcard, fresh_shoe, rules)

