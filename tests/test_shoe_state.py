from blackjack.detector import DetectionResult
from blackjack.shoe_state import ShoeState


def test_observation_ratio_starts_zero_and_increases_with_ingest():
    state = ShoeState(decks=1)
    assert state.observation_ratio == 0
    state.ingest_manual("A")
    assert state.observation_ratio > 0


def test_mid_shoe_join_uncertain_below_threshold():
    state = ShoeState(decks=1, mid_shoe_join=True)
    assert state.is_uncertain is True


def test_ingest_returns_accepted_review_rejected_by_confidence():
    state = ShoeState(decks=1, accept_threshold=0.85, review_threshold=0.75)
    accepted = state.ingest(DetectionResult(rank="A", suit="S", confidence=0.90))
    review = state.ingest(DetectionResult(rank="K", suit="H", confidence=0.80))
    rejected = state.ingest(DetectionResult(rank="Q", suit="D", confidence=0.50))
    assert accepted == "accepted"
    assert review == "review"
    assert rejected == "rejected"


def test_manual_ingest_is_always_accepted_with_confidence_one():
    state = ShoeState(decks=1)
    state.ingest_manual("9")
    assert state.observations[-1].confidence == 1.0


def test_reset_for_new_shoe_restores_full_count_and_clears_observations():
    state = ShoeState(decks=1, mid_shoe_join=True)
    state.ingest_manual("A")
    assert state.remaining == 51
    state.reset_for_new_shoe()
    assert state.remaining == 52
    assert state.observations == []
    assert state.pending_review == []
    assert state.observation_ratio == 0
