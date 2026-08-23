"""Kelly Criterion bet sizing for blackjack.

Given the EV and variance of the best action, computes the optimal
fraction of bankroll to wager, and the recommended bet size.

Full Kelly:   f* = EV / variance
Half Kelly:   f* = EV / (2 * variance)   (recommended — lower risk)

Blackjack variance is approximately 1.15 for standard rules
(accounts for blackjack payouts, splits, doubles).

The Kelly fraction is only positive (i.e. bet > 0) when EV > 0.
When EV <= 0, Kelly says bet the table minimum.
"""

from __future__ import annotations

import math


def kelly_fraction(ev: float, variance: float = 1.15, fraction: float = 0.5) -> float:
    """Return the Kelly bet fraction (proportion of bankroll).

    Parameters
    ----------
    ev:
        Expected value of the best action (units of original bet).
    variance:
        Variance of outcomes. Default 1.15 (standard blackjack).
    fraction:
        Kelly multiplier. 1.0 = full Kelly, 0.5 = half Kelly (default).

    Returns
    -------
    Fraction of bankroll to bet, clamped to [0, 1].
    """
    if ev <= 0 or variance <= 0:
        return 0.0
    f = fraction * ev / variance
    return max(0.0, min(1.0, f))


def recommended_bet(
    ev: float,
    bankroll: float,
    min_bet: float = 5.0,
    max_bet: float = 500.0,
    variance: float = 1.15,
    fraction: float = 0.5,
) -> float:
    """Return the recommended bet size in currency units.

    Clamps to [min_bet, max_bet]. Returns min_bet when EV <= 0.
    Rounds to nearest min_bet increment.
    """
    if ev <= 0:
        return min_bet
    f = kelly_fraction(ev, variance, fraction)
    raw = f * bankroll
    # Round to nearest min_bet increment
    if min_bet > 0:
        raw = round(raw / min_bet) * min_bet
    raw = max(min_bet, min(max_bet, raw))
    return raw


def kelly_summary(
    ev: float,
    bankroll: float,
    min_bet: float = 5.0,
    max_bet: float = 500.0,
    variance: float = 1.15,
    fraction: float = 0.5,
) -> dict:
    """Return a dict with keys:
      'kelly_fraction': float
      'recommended_bet': float
      'ev': float
      'edge_percent': float   (ev * 100)
      'is_positive_ev': bool
    """
    return {
        "kelly_fraction": kelly_fraction(ev, variance, fraction),
        "recommended_bet": recommended_bet(ev, bankroll, min_bet, max_bet,
                                           variance, fraction),
        "ev": ev,
        "edge_percent": ev * 100.0,
        "is_positive_ev": ev > 0,
    }
