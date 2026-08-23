"""Screen-capture pipeline.

Orchestrates:
  1. Grab monitor regions via ``mss``.
  2. Crop each configured seat/dealer region.
  3. Run the card detector on each crop.
  4. Feed results into ShoeState for shoe tracking.
  5. Emit a callback whenever new cards are observed so the EV engine
     and UI can update.

Design principles
-----------------
* The capture loop runs in a background thread so the Streamlit UI
  stays responsive.
* All shoe mutations go through ShoeState.ingest() which applies
  confidence thresholds and mid-shoe-join awareness.
* The capturer never touches the EV engine directly; it fires an
  `on_cards_observed` callback with a list of DetectionResult objects.

Mid-shoe join
-------------
Call ``session.join_mid_shoe()`` before starting capture when joining a
game already in progress.  The shoe is kept at full capacity and cards
are only removed as they are observed on screen, so the EV calculations
are conservative but never wrong about what has actually been seen.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from blackjack.detector import CardDetector, DetectionResult, build_detector
from blackjack.shoe_state import ShoeState

log = logging.getLogger(__name__)

# Type alias for a region: (x, y, width, height) in screen pixels
Region = Tuple[int, int, int, int]


@dataclass
class CaptureConfig:
    """Configuration for one capture session.

    Attributes
    ----------
    fps:
        How many frames per second to capture.  2 is usually sufficient
        for a live blackjack table; higher rates increase CPU load.
    monitor_index:
        Which monitor to capture (1-based, matches mss convention).
    dealer_region:
        (x, y, w, h) pixel rectangle for the dealer's card area.
    player_region:
        (x, y, w, h) for the user's own cards.
    other_player_regions:
        Dict of seat label -> region for other players visible on screen.
        Cards from other players are removed from the shoe (they reduce
        the count) but do not influence the player's own EV calculation.
    backend:
        'template' or 'yolo'.
    template_dir:
        Path to template images (used by TemplateDetector).
    model_path:
        Path to YOLO weights (used by YOLODetector).
    accept_threshold:
        Confidence to auto-accept a detection.
    review_threshold:
        Confidence to flag for manual review.
    """
    fps: float = 2.0
    monitor_index: int = 1
    dealer_region: Optional[Region] = None
    player_region: Optional[Region] = None
    other_player_regions: Dict[str, Region] = field(default_factory=dict)
    backend: str = "template"
    template_dir: str = "data/templates"
    model_path: str = "data/models/cards.pt"
    accept_threshold: float = 0.85
    review_threshold: float = 0.75


class CaptureSession:
    """Manages one live screen-capture + detection session.

    Usage
    -----
    ::

        shoe_state = ShoeState(decks=8, mid_shoe_join=False)
        session = CaptureSession(config, shoe_state)
        session.on_cards_observed = my_callback   # optional
        session.start()
        # ... UI runs ...
        session.stop()

    Callback signature
    ------------------
    ``on_cards_observed(results: List[DetectionResult]) -> None``
    Called in the capture thread after each frame that produced at least
    one accepted or review-flagged detection.
    """

    def __init__(
        self,
        config: CaptureConfig,
        shoe_state: ShoeState,
        detector: Optional[CardDetector] = None,
    ) -> None:
        self.config = config
        self.shoe_state = shoe_state
        self.detector: CardDetector = detector or build_detector(
            backend=config.backend,
            template_dir=config.template_dir,
            model_path=config.model_path,
            confidence_threshold=config.review_threshold,
        )
        self.on_cards_observed: Optional[Callable[[List[DetectionResult]], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hand_number = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="capture", daemon=True
        )
        self._thread.start()
        log.info("Capture session started (%.1f FPS, monitor %d)",
                 self.config.fps, self.config.monitor_index)

    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("Capture session stopped.")

    def next_hand(self) -> None:
        """Increment hand counter (call when a new hand begins)."""
        self._hand_number += 1

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            import mss  # type: ignore
        except ImportError:
            log.error(
                "mss is not installed. "
                "Run: pip install mss  or  pip install -e .[dev]"
            )
            self._running = False
            return

        interval = 1.0 / max(self.config.fps, 0.1)
        with mss.mss() as sct:
            monitors = sct.monitors  # index 0 = all monitors combined
            monitor = monitors[self.config.monitor_index] \
                if self.config.monitor_index < len(monitors) else monitors[1]

            while self._running:
                t0 = time.monotonic()
                frame_results: List[DetectionResult] = []

                # --- grab and detect each configured region ---
                regions = self._build_regions()
                for source, region in regions.items():
                    crop = self._grab_region(sct, monitor, region)
                    if crop is None:
                        continue
                    detections = self.detector.detect(crop, source=source)
                    for det in detections:
                        status = self.shoe_state.ingest(
                            det, hand_number=self._hand_number
                        )
                        if status in ('accepted', 'review'):
                            frame_results.append(det)

                if frame_results and self.on_cards_observed:
                    try:
                        self.on_cards_observed(frame_results)
                    except Exception as exc:
                        log.warning("on_cards_observed callback raised: %s", exc)

                # --- sleep remainder of interval ---
                elapsed = time.monotonic() - t0
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

    def _build_regions(self) -> Dict[str, Region]:
        regions: Dict[str, Region] = {}
        if self.config.dealer_region:
            regions["dealer"] = self.config.dealer_region
        if self.config.player_region:
            regions["player"] = self.config.player_region
        for seat, region in self.config.other_player_regions.items():
            regions[seat] = region
        return regions

    @staticmethod
    def _grab_region(
        sct,
        monitor: Dict,
        region: Region,
    ) -> Optional[np.ndarray]:
        """Grab a sub-region of the monitor, return BGR numpy array."""
        try:
            import cv2  # type: ignore
            x, y, w, h = region
            mon_x = monitor["left"]
            mon_y = monitor["top"]
            grab_area = {
                "left": mon_x + x,
                "top":  mon_y + y,
                "width": w,
                "height": h,
            }
            raw = sct.grab(grab_area)
            # mss returns BGRA; convert to BGR for OpenCV
            frame = np.array(raw, dtype=np.uint8)[:, :, :3]
            return frame
        except Exception as exc:
            log.warning("Failed to grab region %s: %s", region, exc)
            return None


@dataclass
class FrameEvent:
    """Snapshot emitted after each processed frame."""
    timestamp: float
    hand_number: int
    detections: List[DetectionResult]
    shoe_remaining: int
    true_count: float
    observation_ratio: float
    uncertainty_label: str
