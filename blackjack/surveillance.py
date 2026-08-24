"""Surveillance Countermeasure Analysis.

Models casino countermeasure detection and computes strategies for
minimising heat (casino suspicion) while preserving edge.

The heat index is a probabilistic model of how suspicious casino
surveillance considers a player's behaviour, updated with each bet,
play deviation, and session duration.

IMPORTANT: This module is for research and educational purposes.
All surveillance evasion techniques described here are purely academic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Heat model
# ---------------------------------------------------------------------------


@dataclass
class HeatEvent:
    """A single event that affects the heat index."""
    event_type: str     # 'large_bet', 'bet_spread', 'wong_in', 'session_length', etc.
    delta_heat: float   # positive = more suspicious
    hand_number: int
    description: str = ""


@dataclass
class ObfuscationStrategy:
    """Recommended obfuscation actions to reduce heat."""
    action: str             # 'reduce_bet_spread', 'wong_out', 'take_break', etc.
    expected_heat_reduction: float
    ev_cost: float          # EV loss from applying obfuscation
    priority: int           # 1 = most urgent
    description: str


# ---------------------------------------------------------------------------
# Main surveillance class
# ---------------------------------------------------------------------------


class SurveillanceAnalysis:
    """Model casino surveillance risk and recommend countermeasures.

    Heat model components:
    1. **Bet spread** — large min-to-max ratio is the #1 tell.
    2. **Deviation frequency** — deviating from basic strategy when justified
       by count is detectable via cross-referencing play.
    3. **Session duration** — longer sessions increase surveillance attention.
    4. **Win rate** — a win rate exceeding expectation raises flags.
    5. **Table behaviour** — wonging, mid-shoe entry, leaving after wins.

    Parameters
    ----------
    initial_heat:
        Starting heat level (0..1). New player = 0.
    heat_decay_per_hand:
        Natural heat decay per hand (forgetting).
    critical_heat:
        Heat threshold above which immediate action is recommended.
    """

    _EVENT_WEIGHTS: Dict[str, float] = {
        "large_bet_spread_10x": 0.15,
        "large_bet_spread_5x": 0.08,
        "wong_in": 0.05,
        "wong_out": 0.04,
        "deviation_from_basic": 0.03,
        "large_win_streak": 0.08,
        "session_duration_3hr": 0.05,
        "back_counting": 0.12,
        "team_play_signal": 0.20,
        "pit_boss_approach": 0.10,
    }

    def __init__(
        self,
        initial_heat: float = 0.0,
        heat_decay_per_hand: float = 0.001,
        critical_heat: float = 0.70,
    ) -> None:
        self.heat = max(0.0, min(1.0, initial_heat))
        self.decay = heat_decay_per_hand
        self.critical_heat = critical_heat
        self._history: List[HeatEvent] = []
        self._hand_number: int = 0
        self._bets: List[float] = []
        self._wins: List[int] = []   # +1 or -1

    # ------------------------------------------------------------------
    # Heat updates
    # ------------------------------------------------------------------

    def record_bet(
        self,
        bet: float,
        min_bet: float,
        max_bet: float,
        true_count: float = 0.0,
    ) -> None:
        """Update heat based on a new bet."""
        self._hand_number += 1
        self._bets.append(bet)

        # Bet spread contribution
        spread = max_bet / max(min_bet, 1.0)
        if spread >= 10:
            delta = self._EVENT_WEIGHTS["large_bet_spread_10x"]
        elif spread >= 5:
            delta = self._EVENT_WEIGHTS["large_bet_spread_5x"]
        else:
            delta = 0.0

        # Count-correlated betting is suspicious (higher bet = higher TC)
        if len(self._bets) >= 10:
            recent_bets = self._bets[-10:]
            tcs = [true_count]  # simplified: just current TC
            # In practice would need historical TCs
            if true_count > 2 and bet >= 3 * min_bet:
                delta += 0.01  # correlated betting flag

        # Natural decay
        self.heat = max(0.0, self.heat - self.decay) + delta
        self.heat = min(1.0, self.heat)

        if delta > 0:
            self._history.append(HeatEvent(
                event_type="bet_spread",
                delta_heat=delta,
                hand_number=self._hand_number,
                description=f"Bet {bet:.0f} with spread {spread:.1f}x",
            ))

    def record_outcome(self, outcome: int) -> None:
        """Record hand outcome (+1 win, 0 push, -1 loss)."""
        self._wins.append(outcome)

        # Check for suspicious win streak
        if len(self._wins) >= 10:
            recent = self._wins[-10:]
            win_rate = sum(1 for w in recent if w > 0) / 10
            if win_rate >= 0.70:  # >70% win rate in last 10 hands
                delta = self._EVENT_WEIGHTS["large_win_streak"]
                self.heat = min(1.0, self.heat + delta)
                self._history.append(HeatEvent(
                    event_type="large_win_streak",
                    delta_heat=delta,
                    hand_number=self._hand_number,
                    description=f"Win rate {win_rate:.0%} in last 10 hands",
                ))

    def record_event(self, event_type: str, description: str = "") -> None:
        """Record a named surveillance event."""
        delta = self._EVENT_WEIGHTS.get(event_type, 0.05)
        self.heat = min(1.0, self.heat + delta)
        self._history.append(HeatEvent(
            event_type=event_type,
            delta_heat=delta,
            hand_number=self._hand_number,
            description=description,
        ))

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend_obfuscation(
        self,
        current_ev: float,
        min_bet: float,
    ) -> List[ObfuscationStrategy]:
        """Recommend heat-reduction strategies ordered by priority."""
        strategies = []

        if self.heat < 0.3:
            return []  # no action needed

        if self.heat >= 0.7:
            strategies.append(ObfuscationStrategy(
                action="leave_table",
                expected_heat_reduction=self.heat * 0.9,
                ev_cost=0.0,  # no ongoing cost
                priority=1,
                description=f"Heat critical ({self.heat:.0%}). Leave immediately.",
            ))

        if self.heat >= 0.5:
            strategies.append(ObfuscationStrategy(
                action="flat_bet_2_sessions",
                expected_heat_reduction=0.20,
                ev_cost=0.005 * current_ev,
                priority=2,
                description="Flat bet for 20-30 hands to reduce suspicion.",
            ))

        if len(self._bets) >= 5:
            max_recent = max(self._bets[-5:])
            if max_recent >= 8 * min_bet:
                strategies.append(ObfuscationStrategy(
                    action="reduce_bet_spread_to_4x",
                    expected_heat_reduction=0.10,
                    ev_cost=0.002 * current_ev,
                    priority=3,
                    description="Cap bet spread at 4x to reduce betting tells.",
                ))

        strategies.append(ObfuscationStrategy(
            action="introduce_cover_plays",
            expected_heat_reduction=0.05,
            ev_cost=0.001 * current_ev,
            priority=4,
            description="Occasionally deviate from optimal play to appear as a recreational player.",
        ))

        return sorted(strategies, key=lambda x: x.priority)

    def optimal_vs_survival_tradeoff(
        self,
        ev_per_hand: float,
        bankroll: float,
        hands_before_barred: int = 500,
    ) -> Dict[str, float]:
        """Compute EV at different heat levels accounting for barring risk.

        Models:
        - If heat reaches 1.0, player is barred (EV = 0 thereafter)
        - Lower bet spread → lower heat accumulation → longer session
        - Higher bet spread → higher EV per hand → higher barring risk

        Returns dict with:
        - 'full_kelly_ev': EV without any heat constraint
        - 'conservative_ev': EV with 4x max spread (lower heat)
        - 'expected_total_hands': E[hands before being barred]
        - 'optimal_spread': bet spread that maximises total EV
        """
        # Model: heat grows proportionally to bet spread
        # barring_rate(spread) = spread / (200 / min_heat_increment)
        # expected_hands ~ heat_to_critical / heat_per_hand(spread)

        # Simplified model
        heat_per_hand_full = 0.002  # 500 hands to reach critical
        heat_per_hand_conservative = 0.001  # 1000 hands

        def expected_hands(heat_rate: float) -> float:
            return (self.critical_heat - self.heat) / max(heat_rate, 1e-6)

        h_full = expected_hands(heat_per_hand_full)
        h_conservative = expected_hands(heat_per_hand_conservative)

        total_ev_full = ev_per_hand * min(h_full, hands_before_barred)
        total_ev_conservative = ev_per_hand * 0.8 * min(h_conservative, hands_before_barred)

        return {
            "full_kelly_expected_hands": h_full,
            "conservative_expected_hands": h_conservative,
            "full_kelly_total_ev": total_ev_full,
            "conservative_total_ev": total_ev_conservative,
            "optimal_is_conservative": total_ev_conservative > total_ev_full,
            "current_heat": self.heat,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def heat_label(self) -> str:
        """Human-readable heat level."""
        h = self.heat
        if h < 0.2:
            return "Cool"
        elif h < 0.4:
            return "Warm"
        elif h < 0.6:
            return "Hot"
        elif h < 0.8:
            return "Very Hot"
        else:
            return "⚠ CRITICAL"

    @property
    def session_win_rate(self) -> float:
        """Win rate for the current session."""
        if not self._wins:
            return 0.0
        return sum(1 for w in self._wins if w > 0) / len(self._wins)

    def summary(self) -> Dict[str, object]:
        """Return a summary of current surveillance status."""
        return {
            "heat": self.heat,
            "heat_label": self.heat_label,
            "hands_played": self._hand_number,
            "session_win_rate": self.session_win_rate,
            "recent_events": [
                {"type": e.event_type, "delta": e.delta_heat, "hand": e.hand_number}
                for e in self._history[-5:]
            ],
            "at_critical": self.heat >= self.critical_heat,
        }
