"""UI state adapters for the Streamlit front-end.

This module owns all mutable session state for the app and provides
pure-function helpers that the UI widgets call.  Keeping UI logic here
makes it independently testable without importing Streamlit.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from blackjack.rules import RuleSet
from blackjack.shoe import Shoe
from blackjack.hand import Hand
from blackjack.ev import action_evs, best_action, dealer_distribution, insurance_ev, dampened_ev, basic_strategy_ev as _basic_strategy_ev
from blackjack.actions import get_legal_actions
from blackjack.kelly import kelly_summary
from blackjack.side_bets import side_bet_summary


@dataclass
class HandHistoryEntry:
    """Log entry for one completed hand."""
    hand_number: int
    player_cards: List[str]
    dealer_upcard: str
    recommended_action: str
    best_ev: float
    action_taken: Optional[str] = None
    outcome: Optional[str] = None   # 'win' | 'loss' | 'push' | 'surrender'
    confidence: float = 1.0


@dataclass
class AppState:
    """Central session state for the Streamlit app."""

    rules: RuleSet = field(default_factory=RuleSet)
    shoe: Shoe = field(default_factory=Shoe)

    # Current hand
    player_cards: List[str] = field(default_factory=list)
    dealer_upcard: str = ""
    splits_used: int = 0
    is_post_split_ace: bool = False

    # Bankroll / bet sizing
    bankroll: float = 1000.0
    min_bet: float = 5.0
    max_bet: float = 500.0

    # Detection health
    detection_confidence: float = 1.0
    camera_active: bool = False
    last_detection_warning: str = ""

    # History
    hand_history: List[HandHistoryEntry] = field(default_factory=list)
    hand_counter: int = 0

    # ------------------------------------------------------------------
    # Hand management
    # ------------------------------------------------------------------

    def set_player_cards(self, cards: List[str]) -> None:
        self.player_cards = [c.upper() for c in cards]

    def set_dealer_upcard(self, card: str) -> None:
        self.dealer_upcard = card.upper()

    def add_player_card(self, card: str) -> None:
        self.player_cards.append(card.upper())

    def clear_hand(self) -> None:
        self.player_cards = []
        self.dealer_upcard = ""
        self.splits_used = 0
        self.is_post_split_ace = False

    # ------------------------------------------------------------------
    # EV computation
    # ------------------------------------------------------------------

    def compute_evs(self) -> Dict[str, float]:
        """Compute EV dict for the current hand state.

        Returns an empty dict when the hand is not ready (missing cards
        or dealer upcard).
        """
        if len(self.player_cards) < 2 or not self.dealer_upcard:
            return {}
        hand = Hand(
            cards=list(self.player_cards),
            splits_used=self.splits_used,
            is_post_split_ace=self.is_post_split_ace,
        )
        try:
            return action_evs(
                hand, self.dealer_upcard, self.shoe, self.rules,
                self.splits_used, self.is_post_split_ace,
            )
        except Exception:
            return {}

    def get_recommendation(self) -> Tuple[str, float, Dict[str, float]]:
        """Return (best_action, best_ev, full_ev_dict)."""
        evs = self.compute_evs()
        action, ev = best_action(evs)
        return action, ev, evs

    def get_legal_actions_set(self) -> frozenset:
        if len(self.player_cards) < 2 or not self.dealer_upcard:
            return frozenset()
        hand = Hand(
            cards=list(self.player_cards),
            splits_used=self.splits_used,
            is_post_split_ace=self.is_post_split_ace,
        )
        return get_legal_actions(
            hand, self.rules, self.splits_used,
            self.is_post_split_ace, self.dealer_upcard,
        ).as_set()

    def get_kelly_recommendation(self) -> dict:
        """Return Kelly criterion bet sizing recommendation."""
        _, ev, _ = self.get_recommendation()
        return kelly_summary(
            ev=ev,
            bankroll=self.bankroll,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
        )

    def get_insurance_ev(self) -> Optional[float]:
        """Return insurance EV when dealer upcard is A and rules allow, else None."""
        from blackjack.hand import _normalise
        if not self.dealer_upcard:
            return None
        if not self.rules.insurance:
            return None
        if _normalise(self.dealer_upcard) != 'A':
            return None
        return insurance_ev(self.shoe, self.dealer_upcard)

    def get_side_bets(self) -> Optional[dict]:
        """Return side_bet_summary when player has exactly 2 cards and dealer upcard is set."""
        if len(self.player_cards) != 2 or not self.dealer_upcard:
            return None
        return side_bet_summary(
            self.shoe,
            self.player_cards[0],
            self.player_cards[1],
            self.dealer_upcard,
        )

    def get_dampened_recommendation(self) -> dict:
        """Return dampened EVs blended toward basic strategy by observation ratio."""
        evs = self.compute_evs()
        if not evs:
            return {}
        if len(self.player_cards) < 2 or not self.dealer_upcard:
            return {}
        hand = Hand(
            cards=list(self.player_cards),
            splits_used=self.splits_used,
            is_post_split_ace=self.is_post_split_ace,
        )
        bs_evs = _basic_strategy_ev(hand, self.dealer_upcard, self.rules,
                                    decks=self.shoe.decks)
        # Compute observation ratio from shoe
        starting = self.shoe.decks * 52
        seen = starting - self.shoe.total_remaining
        obs_ratio = seen / starting if starting > 0 else 1.0
        result = {}
        for action, raw in evs.items():
            bs_val = bs_evs.get(action, raw)
            result[action] = dampened_ev(raw, bs_val, obs_ratio)
        return result

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def log_hand(self, action_taken: str | None = None,
                 outcome: str | None = None) -> None:
        """Save the current hand to history."""
        if not self.player_cards or not self.dealer_upcard:
            return
        action, ev, _ = self.get_recommendation()
        self.hand_counter += 1
        self.hand_history.append(HandHistoryEntry(
            hand_number=self.hand_counter,
            player_cards=list(self.player_cards),
            dealer_upcard=self.dealer_upcard,
            recommended_action=action,
            best_ev=round(ev, 4),
            action_taken=action_taken,
            outcome=outcome,
            confidence=self.detection_confidence,
        ))

    def export_history_csv(self) -> str:
        """Return hand history as a CSV string."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "hand", "player_cards", "dealer_upcard",
            "recommended", "best_ev", "action_taken",
            "outcome", "confidence",
        ])
        for r in self.hand_history:
            writer.writerow([
                r.hand_number,
                " ".join(r.player_cards),
                r.dealer_upcard,
                r.recommended_action,
                r.best_ev,
                r.action_taken or "",
                r.outcome or "",
                r.confidence,
            ])
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Shoe helpers
    # ------------------------------------------------------------------

    def remove_seen_cards(self, cards: List[str]) -> List[str]:
        """Remove *cards* from shoe; return list of any that failed."""
        errors = []
        for c in cards:
            try:
                self.shoe.remove(c)
            except ValueError:
                errors.append(c)
        return errors

    def reset_shoe(self) -> None:
        self.shoe = Shoe(decks=self.shoe.decks)

    # ------------------------------------------------------------------
    # Rules helpers (used by sidebar widgets)
    # ------------------------------------------------------------------

    def update_rules(
        self,
        dealer_hits_soft17: bool | None = None,
        double_after_split: bool | None = None,
        resplit_aces: bool | None = None,
        max_splits: int | None = None,
        blackjack_payout: float | None = None,
        surrender: str | None = None,
    ) -> None:
        """Return a new RuleSet with updated fields (immutable dataclass)."""
        kw = {
            "dealer_hits_soft17": self.rules.dealer_hits_soft17,
            "double_after_split": self.rules.double_after_split,
            "resplit_aces": self.rules.resplit_aces,
            "max_splits": self.rules.max_splits,
            "split_aces_get_one_card": self.rules.split_aces_get_one_card,
            "blackjack_payout": self.rules.blackjack_payout,
            "surrender": self.rules.surrender,
            "natural_beats_dealer_21": self.rules.natural_beats_dealer_21,
        }
        if dealer_hits_soft17 is not None:
            kw["dealer_hits_soft17"] = dealer_hits_soft17
        if double_after_split is not None:
            kw["double_after_split"] = double_after_split
        if resplit_aces is not None:
            kw["resplit_aces"] = resplit_aces
        if max_splits is not None:
            kw["max_splits"] = max_splits
        if blackjack_payout is not None:
            kw["blackjack_payout"] = blackjack_payout
        if surrender is not None:
            kw["surrender"] = surrender
        self.rules = RuleSet(**kw)


# ---------------------------------------------------------------------------
# Pure helper: format EV table for display
# ---------------------------------------------------------------------------

def format_ev_table(ev_dict: Dict[str, float]) -> List[Dict[str, str]]:
    """Return a list of {action, ev, delta} rows sorted best-first.

    *delta* is the difference in EV vs. the best action (always <= 0).
    """
    if not ev_dict:
        return []
    best_ev = max(ev_dict.values())
    rows = []
    for action, ev in sorted(ev_dict.items(), key=lambda x: -x[1]):
        rows.append({
            "action": action.upper(),
            "ev": f"{ev:+.4f}",
            "delta": f"{ev - best_ev:+.4f}" if ev != best_ev else "best",
        })
    return rows


def health_status(confidence: float) -> Tuple[str, str]:
    """Return (label, colour) for a detection confidence value."""
    if confidence >= 0.85:
        return "Good", "#16a34a"
    elif confidence >= 0.75:
        return "Review", "#f59e0b"
    else:
        return "Low – verify manually", "#dc2626"


# Backwards-compatible alias for older imports/tests.
HandRecord = HandHistoryEntry
