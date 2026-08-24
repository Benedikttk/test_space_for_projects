import time

from blackjack.capture import CaptureConfig, CaptureSession
from blackjack.detector import CardDetector
from blackjack.shoe_state import ShoeState


class _NoopDetector(CardDetector):
    def detect(self, image, source: str = ""):
        return []

    def is_available(self) -> bool:
        return True


def test_build_regions_includes_all_configured_regions():
    config = CaptureConfig(
        dealer_region=(1, 2, 3, 4),
        player_region=(5, 6, 7, 8),
        other_player_regions={"seat_1": (9, 10, 11, 12)},
    )
    session = CaptureSession(config, ShoeState(), detector=_NoopDetector())
    assert session._build_regions() == {
        "dealer": (1, 2, 3, 4),
        "player": (5, 6, 7, 8),
        "seat_1": (9, 10, 11, 12),
    }


def test_start_stop_are_idempotent_and_thread_safe():
    session = CaptureSession(CaptureConfig(fps=200.0), ShoeState(), detector=_NoopDetector())

    def fake_loop():
        while session._running.is_set():
            time.sleep(0.001)

    session._loop = fake_loop  # type: ignore[method-assign]

    session.start()
    session.start()
    assert session._running.is_set()
    session.stop()
    session.stop()
    assert not session._running.is_set()
