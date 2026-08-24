"""Cut card detection, penetration tracking, and Bayesian card distribution.

Provides:
- CutCardDetector: visual detection of the cut card in the shoe image
- PenetrationTracker: real-time penetration % and cards-remaining calculation
- BayesianCardDistribution: Dirichlet-Multinomial posterior over unobserved cards
- ShoeHistory: per-shoe penetration + true-count profile storage
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from blackjack.shoe import ALL_RANKS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CARDS_PER_DECK = 52
_DEFAULT_DECKS = 8
_EMA_ALPHA = 0.3  # Exponential moving average smoothing


# ---------------------------------------------------------------------------
# Bayesian Card Distribution (Dirichlet-Multinomial)
# ---------------------------------------------------------------------------

class BayesianCardDistribution:
    """Dirichlet-Multinomial model for unobserved cards behind the cut card.

    The prior is uniform (all ranks equally likely) scaled by decks.
    After each shoe the posterior is updated with the observed card counts,
    giving an increasingly accurate model of the unobserved region.

    Attributes
    ----------
    decks:
        Number of decks in the shoe.
    alpha:
        Dirichlet concentration parameters per rank.  Starts at
        ``[4 * decks] * 13`` and is updated after each shoe.
    """

    def __init__(self, decks: int = _DEFAULT_DECKS) -> None:
        self.decks = decks
        per_rank = 4 * decks
        # alpha indexed by rank in ALL_RANKS order
        self.alpha: np.ndarray = np.full(len(ALL_RANKS), float(per_rank))
        self._rank_index: Dict[str, int] = {r: i for i, r in enumerate(ALL_RANKS)}

    # ------------------------------------------------------------------
    # Prior / posterior queries
    # ------------------------------------------------------------------

    def posterior_probs(self) -> Dict[str, float]:
        """Return the posterior probability for each rank.

        Returns P(rank | observed history) as a dict mapping
        normalised rank → probability.
        """
        total = float(self.alpha.sum())
        return {r: float(self.alpha[i]) / total for i, r in enumerate(ALL_RANKS)}

    def expected_counts_in_region(self, region_size: int) -> Dict[str, float]:
        """Expected count of each rank in an unobserved region of given size.

        Parameters
        ----------
        region_size:
            Number of cards in the unobserved cut-off region.
        """
        probs = self.posterior_probs()
        return {r: probs[r] * region_size for r in ALL_RANKS}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_from_shoe(self, observed_counts: Dict[str, int]) -> None:
        """Update the Dirichlet posterior with observed counts from a shoe.

        Parameters
        ----------
        observed_counts:
            Mapping rank → number of times that rank was observed during
            the shoe.  Only cards *observed* (not inferred) should be
            counted here.
        """
        for rank, count in observed_counts.items():
            idx = self._rank_index.get(rank)
            if idx is not None:
                self.alpha[idx] += count

    def reset_prior(self) -> None:
        """Reset alpha back to the uniform prior (start of tracking)."""
        per_rank = 4 * self.decks
        self.alpha = np.full(len(ALL_RANKS), float(per_rank))


# ---------------------------------------------------------------------------
# Cut Card Detector
# ---------------------------------------------------------------------------

@dataclass
class CutCardDetection:
    """Result of a single cut-card detection attempt."""
    detected: bool
    position_fraction: float  # 0.0 = front of shoe, 1.0 = end
    placement_cards: int      # estimated card index where cut card sits
    confidence: float         # 0–1 detection confidence


class CutCardDetector:
    """Detect the cut card in a shoe image using HSV colour thresholding.

    The algorithm:
    1. Convert BGR frame to HSV.
    2. Threshold for red and yellow hues (typical cut card colours).
    3. Find contours; select the largest matching expected aspect ratio.
    4. Compute x-position as fraction of shoe width.
    5. Apply exponential moving average (α=0.3) for temporal smoothing.
    6. Estimate placement_cards = position_fraction × total_cards.

    Parameters
    ----------
    total_cards:
        Total number of cards in the shoe (e.g. 416 for 8 decks).
    ema_alpha:
        Smoothing factor for the exponential moving average on position.
    min_area_fraction:
        Minimum contour area as a fraction of frame area.
    """

    def __init__(
        self,
        total_cards: int = _DEFAULT_DECKS * _CARDS_PER_DECK,
        ema_alpha: float = _EMA_ALPHA,
        min_area_fraction: float = 0.005,
    ) -> None:
        self.total_cards = total_cards
        self.ema_alpha = ema_alpha
        self.min_area_fraction = min_area_fraction
        self._smoothed_position: Optional[float] = None

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> CutCardDetection:
        """Run cut card detection on one BGR frame of the shoe region.

        Parameters
        ----------
        frame:
            BGR numpy array of the shoe region.

        Returns
        -------
        CutCardDetection with detected=False if opencv unavailable or no
        match found.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            log.debug("opencv not available — cut card detection disabled")
            return CutCardDetection(
                detected=False, position_fraction=0.0,
                placement_cards=0, confidence=0.0,
            )

        if frame is None or frame.size == 0:
            return CutCardDetection(
                detected=False, position_fraction=0.0,
                placement_cards=0, confidence=0.0,
            )

        h, w = frame.shape[:2]
        frame_area = h * w
        if frame_area == 0:
            return CutCardDetection(
                detected=False, position_fraction=0.0,
                placement_cards=0, confidence=0.0,
            )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._colour_mask(hsv)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return CutCardDetection(
                detected=False, position_fraction=0.0,
                placement_cards=0, confidence=0.0,
            )

        # Filter by minimum area and pick largest
        min_area = self.min_area_fraction * frame_area
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            return CutCardDetection(
                detected=False, position_fraction=0.0,
                placement_cards=0, confidence=0.0,
            )

        largest = max(valid, key=cv2.contourArea)
        x_rect, _y, w_rect, _h_rect = cv2.boundingRect(largest)
        # Centre x of the detected contour as fraction of shoe width
        cx = (x_rect + w_rect / 2.0) / w
        cx = float(np.clip(cx, 0.0, 1.0))

        # Confidence based on colour mask fill within bounding box
        roi_mask = mask[
            _y: _y + _h_rect,
            x_rect: x_rect + w_rect,
        ]
        fill_ratio = (
            float(roi_mask.sum() / 255) / (w_rect * _h_rect)
            if (w_rect * _h_rect) > 0
            else 0.0
        )
        confidence = float(np.clip(fill_ratio * 2, 0.0, 1.0))

        # Exponential moving average smoothing
        if self._smoothed_position is None:
            self._smoothed_position = cx
        else:
            self._smoothed_position = (
                self.ema_alpha * cx
                + (1.0 - self.ema_alpha) * self._smoothed_position
            )

        placement_cards = round(self._smoothed_position * self.total_cards)

        return CutCardDetection(
            detected=True,
            position_fraction=self._smoothed_position,
            placement_cards=placement_cards,
            confidence=confidence,
        )

    def reset_smoothing(self) -> None:
        """Reset the EMA state (call on shoe shuffle)."""
        self._smoothed_position = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _colour_mask(hsv: np.ndarray) -> np.ndarray:
        """Build a binary mask for red and yellow hues."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Red wraps around 0°/180° in HSV
        lower_red1 = np.array([0, 120, 80], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 120, 80], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        # Yellow
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
        combined = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_y)

        # Small morphological close to join nearby blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined


# ---------------------------------------------------------------------------
# Penetration Tracker
# ---------------------------------------------------------------------------

@dataclass
class PenetrationState:
    """Snapshot of shoe penetration at a given moment."""
    cards_dealt: int
    total_cards: int
    cut_card_position: int      # placement_cards from cut card detector
    penetration_pct: float      # 0–100 %
    cards_until_reshuffle: int  # cards remaining before cut card reached
    is_reshuffle_alert: bool    # True when cards_until_reshuffle <= alert_threshold


class PenetrationTracker:
    """Track shoe penetration in real-time.

    Parameters
    ----------
    total_cards:
        Total cards in shoe at the start.
    alert_threshold:
        Trigger reshuffle alert when this many cards remain before cut.
    """

    def __init__(
        self,
        total_cards: int = _DEFAULT_DECKS * _CARDS_PER_DECK,
        alert_threshold: int = 26,
    ) -> None:
        self.total_cards = total_cards
        self.alert_threshold = alert_threshold
        self._cards_dealt: int = 0
        self._cut_card_position: int = round(0.75 * total_cards)  # default 75 %

    # ------------------------------------------------------------------

    def set_cut_card_position(self, placement_cards: int) -> None:
        """Record where the cut card was placed (from CutCardDetector)."""
        self._cut_card_position = max(0, min(placement_cards, self.total_cards))
        log.info(
            "Cut card set at %d / %d cards (%.1f %%)",
            self._cut_card_position,
            self.total_cards,
            100.0 * self._cut_card_position / max(1, self.total_cards),
        )

    def update(self, cards_dealt: int) -> PenetrationState:
        """Update dealt count and return current penetration state.

        Parameters
        ----------
        cards_dealt:
            Total cards dealt so far in this shoe.
        """
        self._cards_dealt = max(0, cards_dealt)
        pct = 100.0 * self._cards_dealt / max(1, self.total_cards)
        remaining = max(0, self._cut_card_position - self._cards_dealt)
        alert = remaining <= self.alert_threshold
        return PenetrationState(
            cards_dealt=self._cards_dealt,
            total_cards=self.total_cards,
            cut_card_position=self._cut_card_position,
            penetration_pct=pct,
            cards_until_reshuffle=remaining,
            is_reshuffle_alert=alert,
        )

    def reset(self) -> None:
        """Reset for a new shoe (call after shuffle)."""
        self._cards_dealt = 0

    @property
    def penetration_fraction(self) -> float:
        """Penetration as a fraction in [0, 1]."""
        return self._cards_dealt / max(1, self.total_cards)


# ---------------------------------------------------------------------------
# Shoe History
# ---------------------------------------------------------------------------

@dataclass
class ShoeRecord:
    """Record of a single completed shoe."""
    shoe_id: int
    cut_card_position: int
    total_cards_dealt: int
    penetration_pct: float
    true_count_profile: List[float]    # true count sampled each hand
    penetration_profile: List[float]  # penetration % sampled each hand
    observed_counts: Dict[str, int]   # rank → count observed this shoe


class ShoeHistory:
    """Persist per-shoe penetration and true-count profiles.

    Stores the last ``max_shoes`` shoes in memory and provides summary
    statistics.
    """

    def __init__(self, max_shoes: int = 100) -> None:
        self.max_shoes = max_shoes
        self._records: List[ShoeRecord] = []
        self._shoe_counter = 0

    def record(
        self,
        cut_card_position: int,
        total_cards_dealt: int,
        true_count_profile: List[float],
        penetration_profile: List[float],
        observed_counts: Dict[str, int],
    ) -> ShoeRecord:
        """Add a completed shoe record."""
        self._shoe_counter += 1
        rec = ShoeRecord(
            shoe_id=self._shoe_counter,
            cut_card_position=cut_card_position,
            total_cards_dealt=total_cards_dealt,
            penetration_pct=100.0 * total_cards_dealt / max(1, cut_card_position),
            true_count_profile=list(true_count_profile),
            penetration_profile=list(penetration_profile),
            observed_counts=dict(observed_counts),
        )
        self._records.append(rec)
        if len(self._records) > self.max_shoes:
            self._records.pop(0)
        return rec

    def average_penetration(self) -> float:
        """Mean penetration % across stored shoes."""
        if not self._records:
            return 0.0
        return sum(r.penetration_pct for r in self._records) / len(self._records)

    def all_records(self) -> List[ShoeRecord]:
        return list(self._records)
