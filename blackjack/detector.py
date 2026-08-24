"""Card detection layer.

Abstract base + two concrete implementations:

* TemplateDetector  – pure OpenCV template matching, no GPU required.
* YOLODetector      – ultralytics YOLO, requires the `vision_dl` extra.

Both return a list of DetectionResult objects per image region.
The rest of the system only depends on DetectionResult, so the backend
can be swapped transparently.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Canonical rank strings (as used throughout the engine)
ALL_CARD_RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
ALL_SUITS      = ['S','H','D','C']


@dataclass
class DetectionResult:
    """One detected card in a frame region.

    Attributes
    ----------
    rank:
        Card rank string: '2'-'9', 'T', 'J', 'Q', 'K', 'A'.
        Face cards are NOT normalised to 'T' at this layer so that the
        UI can display the actual rank; normalisation happens in Shoe.remove.
    suit:
        'S' | 'H' | 'D' | 'C', or empty string if suit detection is not
        supported by the backend.
    confidence:
        Model confidence in [0, 1].
    bbox:
        Optional (x, y, w, h) bounding box in the source region.
    source:
        Which region this came from ('dealer', 'player', 'seat_N', etc.).
    """
    rank: str
    suit: str
    confidence: float
    bbox: Optional[Tuple[int,int,int,int]] = None
    source: str = ""

    @property
    def engine_rank(self) -> str:
        """Rank normalised to engine conventions (J/Q/K -> T)."""
        from blackjack.hand import _normalise
        return _normalise(self.rank)


class CardDetector(abc.ABC):
    """Abstract card detector."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray, source: str = "") -> List[DetectionResult]:
        """Detect cards in *image* (BGR numpy array).

        Parameters
        ----------
        image:
            A cropped BGR frame from a screen region.
        source:
            Label for the region ('dealer', 'seat_1', etc.).

        Returns
        -------
        List of DetectionResult, one per detected card.  May be empty.
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the detector backend is usable."""


class NullDetector(CardDetector):
    """No-op detector used when no backend is available."""

    def detect(self, image: np.ndarray, source: str = "") -> List[DetectionResult]:
        return []

    def is_available(self) -> bool:
        return False


class TemplateDetector(CardDetector):
    """OpenCV template-matching card detector.

    Loads a set of reference card images from *template_dir*.  Expected
    naming convention:  ``<rank>_<suit>.png``  e.g. ``A_S.png``, ``T_H.png``.

    This is the zero-dependency fallback when YOLO weights are not available.
    For a live 1080p screen at 2 FPS it is fast enough (~30 ms/region).
    """

    def __init__(
        self,
        template_dir: str | Path,
        confidence_threshold: float = 0.75,
        scale_factors: Tuple[float, ...] = (1.0, 0.85, 0.70),
    ) -> None:
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self.scale_factors = scale_factors
        self._templates: dict = {}
        self._load_templates()

    def _load_templates(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            log.warning("opencv-python not installed; TemplateDetector unavailable.")
            return
        import cv2  # type: ignore
        for rank in ALL_CARD_RANKS:
            for suit in ALL_SUITS:
                path = self.template_dir / f"{rank}_{suit}.png"
                if path.exists():
                    tmpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if tmpl is not None:
                        self._templates[(rank, suit)] = tmpl
        log.info("Loaded %d card templates from %s",
                 len(self._templates), self.template_dir)

    def is_available(self) -> bool:
        try:
            import cv2  # type: ignore
            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray, source: str = "") -> List[DetectionResult]:
        """Run multi-scale template matching over *image*."""
        if not self._templates or image is None or image.size == 0:
            return []
        try:
            import cv2  # type: ignore
        except ImportError:
            return []

        results: List[DetectionResult] = []
        best_conf = 0.0
        best_rank = ""
        best_suit = ""
        best_bbox = None

        h, w = image.shape[:2]
        for (rank, suit), tmpl in self._templates.items():
            th, tw = tmpl.shape[:2]
            for scale in self.scale_factors:
                new_w = int(tw * scale)
                new_h = int(th * scale)
                if new_w > w or new_h > h or new_w < 8 or new_h < 8:
                    continue
                resized = cv2.resize(tmpl, (new_w, new_h))
                result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_conf:
                    best_conf = max_val
                    best_rank = rank
                    best_suit = suit
                    best_bbox = (max_loc[0], max_loc[1], new_w, new_h)

        if best_conf >= self.confidence_threshold and best_rank:
            results.append(DetectionResult(
                rank=best_rank,
                suit=best_suit,
                confidence=float(best_conf),
                bbox=best_bbox,
                source=source,
            ))
        return results


class YOLODetector(CardDetector):
    """YOLO-based card detector using ultralytics.

    Requires:  pip install -e .[vision_dl]

    *model_path* should point to a YOLO weights file (.pt) trained on
    playing-card images.  The class names in the model must match the
    convention  '<rank>_<suit>'  (e.g. 'A_S', 'T_H').
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.75,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(self.model_path))
            self._model.to(self.device)
            log.info("YOLO model loaded from %s on %s", self.model_path, self.device)
        except Exception as exc:
            log.warning("Could not load YOLO model: %s", exc)
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, image: np.ndarray, source: str = "") -> List[DetectionResult]:
        if self._model is None or image is None or image.size == 0:
            return []
        try:
            results = self._model.predict(
                image,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            log.warning("YOLO inference failed: %s", exc)
            return []

        detections: List[DetectionResult] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = r.names[cls_id]          # e.g. 'A_S' or 'T_H'
                parts  = label.split('_', 1)
                rank   = parts[0] if parts else label
                suit   = parts[1] if len(parts) > 1 else ''
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                detections.append(DetectionResult(
                    rank=rank, suit=suit, confidence=conf,
                    bbox=bbox, source=source,
                ))
        return detections


def build_detector(
    backend: str = "template",
    template_dir: str | Path = "data/templates",
    model_path: str | Path = "data/models/cards.pt",
    confidence_threshold: float = 0.75,
) -> CardDetector:
    """Factory: return the best available detector.

    If *backend* == 'yolo' but the model file or ultralytics are absent,
    falls back to TemplateDetector automatically.
    """
    backend = (backend or "template").lower()
    if backend in {"yolo", "auto"}:
        d = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        if d.is_available():
            return d
        log.warning("YOLO detector unavailable, falling back to template matching.")
    template_detector = TemplateDetector(
        template_dir=template_dir,
        confidence_threshold=confidence_threshold,
    )
    if template_detector.is_available():
        return template_detector
    log.warning("No available detection backend; using NullDetector.")
    return NullDetector()


def get_detector(
    backend: str = "template",
    template_dir: str | Path = "data/templates",
    model_path: str | Path = "data/models/cards.pt",
    confidence_threshold: float = 0.75,
) -> CardDetector:
    """Compatibility alias for detector factory."""
    return build_detector(
        backend=backend,
        template_dir=template_dir,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
