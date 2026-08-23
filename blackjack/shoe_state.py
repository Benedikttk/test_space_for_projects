"""ShoeState — shoe tracker with mid-shoe-join support.

The core mathematical problem when joining a live game mid-shoe is:

    We do NOT know which cards were dealt before we arrived.
    The shoe composition is therefore uncertain.

This module models that uncertainty explicitly:

* Full-shoe mode (game started fresh or shoe was just shuffled):
  Shoe begins at N*52 cards, known exactly.

* Mid-shoe-join mode:
  We assume a full shoe MINUS every card we have directly observed
  since joining.  The prior for unseen cards is the full-deck
  distribution (i.e. we assume no systematic depletion bias).
  As more cards are observed the posterior converges to the true
  composition.

The uncertainty is quantified by `observation_ratio`: the fraction of
the shoe's starting cards that we have directly seen (or inferred).
At 0.0 we know nothing; at 1.0 we have seen every card.

For EV calculations this matters: with a fresh shoe, every rank
probability is exactly 1/13 (adjusted for 'T' grouping).  With a
partly-depleted shoe the probabilities shift, and those shifts are what
the EV engine uses to deviate from basic strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from blackjack.shoe import Shoe, ALL_RANKS, _PER_DECK
from blackjack.hand import _normalise
from blackjack.detector import DetectionResult

log = logging.getLogger(__name__)


@dataclass
class ObservedCard:
    """A single card observation from the detection pipeline."""
    rank: str          # engine-normalised rank ('T' for face cards)
    suit: str
    confidence: float
    source: str        # 'dealer' | 'seat_N' | 'other_player' | 'manual'
    hand_number: int = 0


@dataclass
class ShoeState:
    """Shoe tracker with mid-shoe-join awareness.

    Parameters
    ----------
    decks:
        Number of decks in the shoe.
    mid_shoe_join:
        Set True when we are joining a game already in progress and
        cannot know how many cards have been dealt before our arrival.
        The shoe is still initialised to full capacity; as we observe
        cards they are removed.  The `observation_ratio` property
        tells the UI how confident the shoe estimate is.
    accept_threshold:
        Minimum detection confidence to automatically remove a card
        from the shoe.  Below this the card is logged but not removed
        until manually confirmed.
    review_threshold:
        Detections between review_threshold and accept_threshold are
        flagged for manual review.
    """

    decks: int = 8
    mid_shoe_join: bool = False
    accept_threshold: float = 0.85
    review_threshold: float = 0.75

    shoe: Shoe = field(init=False)
    observations: List[ObservedCard] = field(default_factory=list)
    pending_review: List[DetectionResult] = field(default_factory=list)
    _cards_seen: int = field(default=0, init=False)
    _starting_cards: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.shoe = Shoe(decks=self.decks)
        self._starting_cards = self.shoe.total_remaining

    # ------------------------------------------------------------------
    # Core ingestion
    # ------------------------------------------------------------------

    def ingest(self, detection: DetectionResult, hand_number: int = 0) -> str:
        """Process one card detection from the vision pipeline.

        Returns
        -------
        'accepted'  – card removed from shoe, observation logged.
        'review'    – confidence too low, added to pending_review queue.
        'rejected'  – confidence below review_threshold, discarded.
        """
        rank = _normalise(detection.rank)
        conf = detection.confidence

        if conf >= self.accept_threshold:
            self._remove_card(rank, detection.suit, conf,
                              detection.source, hand_number)
            return 'accepted'
        elif conf >= self.review_threshold:
            self.pending_review.append(detection)
            log.info("Card %s pending review (conf=%.2f)", rank, conf)
            return 'review'
        else:
            log.debug("Card %s rejected (conf=%.2f < %.2f)",
                      rank, conf, self.review_threshold)
            return 'rejected'

    def ingest_manual(self, rank: str, suit: str = '',
                      source: str = 'manual', hand_number: int = 0) -> None:
        """Manually register a card (confidence=1.0)."""
        rank = _normalise(rank)
        self._remove_card(rank, suit, 1.0, source, hand_number)

    def confirm_review(self, detection: DetectionResult,
                       hand_number: int = 0) -> None:
        """Manually confirm a pending-review detection."""
        if detection in self.pending_review:
            self.pending_review.remove(detection)
        rank = _normalise(detection.rank)
        self._remove_card(rank, detection.suit, detection.confidence,
                          detection.source, hand_number)

    def discard_review(self, detection: DetectionResult) -> None:
        """Discard a pending-review detection without removing from shoe."""
        if detection in self.pending_review:
            self.pending_review.remove(detection)

    def _remove_card(self, rank: str, suit: str, confidence: float,
                     source: str, hand_number: int) -> None:
        try:
            self.shoe.remove(rank)
            self._cards_seen += 1
            self.observations.append(ObservedCard(
                rank=rank, suit=suit, confidence=confidence,
                source=source, hand_number=hand_number,
            ))
        except ValueError:
            log.warning(
                "Tried to remove %s from shoe but count is 0 — "
                "possible duplicate detection or mid-shoe uncertainty.", rank
            )

    # ------------------------------------------------------------------
    # Mid-shoe-join helpers
    # ------------------------------------------------------------------

    @property
    def observation_ratio(self) -> float:
        """Fraction of starting shoe cards that we have directly observed.

        0.0 = we just joined, know nothing beyond what's visible right now.
        1.0 = we have tracked every card from the start of the shoe.

        The EV calculations are always correct for the *observed* portion;
        the unobserved portion is modelled as still following the full-deck
        prior (uniform depletion assumption).
        """
        if self._starting_cards == 0:
            return 1.0
        return self._cards_seen / self._starting_cards

    @property
    def is_uncertain(self) -> bool:
        """True when we have seen less than 15% of the shoe.

        Below this threshold the shoe composition estimate is close to
        the uninformed prior and EV deltas from card counting are small
        and potentially misleading.
        """
        return self.mid_shoe_join and self.observation_ratio < 0.15

    @property
    def uncertainty_label(self) -> str:
        if not self.mid_shoe_join:
            return "Full tracking"
        r = self.observation_ratio
        if r < 0.10:
            return f"Mid-shoe join — very uncertain ({r:.0%} observed)"
        elif r < 0.25:
            return f"Mid-shoe join — uncertain ({r:.0%} observed)"
        elif r < 0.60:
            return f"Partial tracking ({r:.0%} observed)"
        else:
            return f"Good tracking ({r:.0%} observed)"

    def reset_for_new_shoe(self) -> None:
        """Reset when the dealer shuffles a new shoe."""
        self.shoe.reset()
        self.observations.clear()
        self.pending_review.clear()
        self._cards_seen = 0
        self._starting_cards = self.shoe.total_remaining
        self.mid_shoe_join = False
        log.info("Shoe reset for new %d-deck shoe.", self.decks)

    def join_mid_shoe(self) -> None:
        """Mark that we are joining a game in progress.

        The current shoe state is kept as-is (a full shoe minus any
        cards already observed).  Sets mid_shoe_join=True so the UI
        can warn the user that counts are uncertain.
        """
        self.mid_shoe_join = True
        log.info(
            "Mid-shoe join: shoe assumed full minus %d observed cards.",
            self._cards_seen,
        )

    # ------------------------------------------------------------------
    # Convenience pass-throughs to Shoe
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> int:
        return self.shoe.total_remaining

    @property
    def true_count(self) -> float:
        return self.shoe.true_count

    @property
    def running_count(self) -> int:
        return self.shoe.running_count

    def rank_distribution(self) -> Dict[str, float]:
        return self.shoe.rank_distribution()

    def snapshot(self) -> Dict[str, int]:
        return self.shoe.snapshot()
