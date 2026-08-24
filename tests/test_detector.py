import numpy as np

from blackjack.detector import CardDetector, NullDetector, build_detector, get_detector


class _UnavailableDetector(CardDetector):
    def detect(self, image: np.ndarray, source: str = ""):
        return []

    def is_available(self) -> bool:
        return False


class _AvailableDetector(CardDetector):
    def detect(self, image: np.ndarray, source: str = ""):
        return []

    def is_available(self) -> bool:
        return True


def test_build_detector_auto_falls_back_to_template(monkeypatch):
    monkeypatch.setattr("blackjack.detector.YOLODetector", lambda **kwargs: _UnavailableDetector())
    monkeypatch.setattr(
        "blackjack.detector.TemplateDetector",
        lambda **kwargs: _AvailableDetector(),
    )
    detector = build_detector(backend="auto")
    assert isinstance(detector, _AvailableDetector)


def test_build_detector_uses_null_when_no_backend_available(monkeypatch):
    monkeypatch.setattr("blackjack.detector.YOLODetector", lambda **kwargs: _UnavailableDetector())
    monkeypatch.setattr(
        "blackjack.detector.TemplateDetector",
        lambda **kwargs: _UnavailableDetector(),
    )
    detector = build_detector(backend="auto")
    assert isinstance(detector, NullDetector)
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_get_detector_alias_calls_factory(monkeypatch):
    marker = _AvailableDetector()
    monkeypatch.setattr("blackjack.detector.build_detector", lambda **kwargs: marker)
    assert get_detector(backend="template") is marker
