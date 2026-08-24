"""Detector with live visualization (green boxes, card values)."""

from __future__ import annotations

import numpy as np

from blackjack.detector import CardDetector, DetectionResult
from blackjack.hand import RANK_VALUE


def get_card_display_text(det: DetectionResult) -> str:
    """Return display text like 'K=10' or 'A=11/1'."""
    if det.rank == 'A':
        return "A=11/1"
    value = RANK_VALUE.get(det.rank, 10)
    return f"{det.rank}={value}"


class VisualizingDetector:
    """Wraps any CardDetector to add live visualization with green bounding boxes."""

    def __init__(self, detector: CardDetector) -> None:
        self.detector = detector

    def detect_and_draw(
        self,
        frame: np.ndarray,
        source: str = "",
    ) -> tuple[list[DetectionResult], np.ndarray]:
        """Detect cards and draw annotated boxes on a copy of the frame.

        Parameters
        ----------
        frame:
            BGR image as a numpy array.
        source:
            Region label passed to the underlying detector.

        Returns
        -------
        (detections, annotated_frame)
            *detections* – list of :class:`DetectionResult` objects.
            *annotated_frame* – copy of *frame* with green bounding boxes
            and card-value labels drawn on it.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return self.detector.detect(frame, source=source), frame.copy()

        detections = self.detector.detect(frame, source=source)
        annotated = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        for det in detections:
            if det.bbox is None:
                continue

            x, y, w, h = det.bbox

            # Green bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Card rank + value text (e.g. "K=10" or "A=11/1")
            text = get_card_display_text(det)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_x = x + 5
            text_y = y + 20

            # Black background rectangle for readability
            cv2.rectangle(
                annotated,
                (text_x - 2, text_y - text_size[1] - 2),
                (text_x + text_size[0] + 2, text_y + 2),
                (0, 0, 0),
                -1,
            )

            # White text
            cv2.putText(
                annotated,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

            # Confidence score at bottom-right of box
            conf_text = f"{det.confidence:.2f}"
            cv2.putText(
                annotated,
                conf_text,
                (x + w - 40, y + h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        return detections, annotated
