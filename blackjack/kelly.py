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


def kelly_fraction(
    ev: float,
    variance: float = 1.15,
    half: bool = True,
    fraction: float | None = None,
) -> float:
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
    if fraction is None:
        fraction = 0.5 if half else 1.0
    f = fraction * ev / variance
    return max(0.0, min(1.0, f))


def recommended_bet(
    ev: float,
    min_bet: float = 5.0,
    max_bet: float = 500.0,
    bankroll: float = 1000.0,
    kelly_fraction: float | None = None,
    kelly_fraction_value: float | None = None,
    variance: float = 1.15,
    half: bool = True,
    fraction: float | None = None,
) -> float:
    """Return the recommended bet size in currency units.

    Clamps to [min_bet, max_bet]. Returns min_bet when EV <= 0.
    Rounds to nearest min_bet increment.
    """
    if ev <= 0:
        return min_bet
    f = (
        (kelly_fraction if kelly_fraction is not None else kelly_fraction_value)
        if kelly_fraction_value is not None
        or kelly_fraction is not None
        else globals()["kelly_fraction"](ev, variance=variance, half=half, fraction=fraction)
    )
    raw = f * bankroll
    # Round to nearest min_bet increment
    if min_bet > 0:
        raw = round(raw / min_bet) * min_bet
    raw = max(min_bet, min(max_bet, raw))
    return raw


def kelly_summary(
    ev: float,
    bankroll: float | None = None,
    kelly_fraction: float | None = None,
    kelly_fraction_value: float | None = None,
    min_bet: float = 5.0,
    max_bet: float = 500.0,
    variance: float = 1.15,
    half: bool = True,
    fraction: float | None = None,
) -> dict | str:
    """Return a dict with keys:
      'kelly_fraction': float
      'recommended_bet': float
      'ev': float
      'edge_percent': float   (ev * 100)
      'is_positive_ev': bool
    """
    kf = (
        kelly_fraction
        if kelly_fraction is not None
        else (
            kelly_fraction_value
            if kelly_fraction_value is not None
            else globals()["kelly_fraction"](ev, variance=variance, half=half, fraction=fraction)
        )
    )
    if bankroll is None:
        return (
            "Positive edge — bet aggressively"
            if (kf > 0 and ev > 0)
            else "No positive edge — bet minimum"
        )

    return {
        "kelly_fraction": kf,
        "recommended_bet": recommended_bet(
            ev=ev,
            bankroll=bankroll,
            min_bet=min_bet,
            max_bet=max_bet,
            kelly_fraction_value=kf,
        ),
        "ev": ev,
        "edge_percent": ev * 100.0,
        "is_positive_ev": ev > 0,
    }
