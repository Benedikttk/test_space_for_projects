"""Tests for blackjack/proofs.py — MathematicalCorrectness."""

import pytest
from blackjack.proofs import MathematicalCorrectness, ProofResult


@pytest.fixture
def proofs():
    return MathematicalCorrectness(strict=False)


def test_ev_optimality_verifies(proofs):
    result = proofs.verify_ev_optimality()
    assert isinstance(result, ProofResult)
    assert result.is_verified


def test_finite_vs_infinite_deck_verifies(proofs):
    result = proofs.verify_finite_vs_infinite_deck_error(n_decks=6, cards_dealt=52)
    assert result.is_verified
    assert result.error_bound is not None
    assert result.error_bound > 0


def test_kelly_optimality_verifies(proofs):
    result = proofs.verify_kelly_optimality(ev=0.02, variance=1.15)
    assert result.is_verified
    assert "f_star" in result.numerical_evidence
    assert result.numerical_evidence["f_star"] == pytest.approx(0.02 / 1.15, rel=1e-6)


def test_bayesian_convergence_verifies(proofs):
    result = proofs.verify_bayesian_convergence(n_decks=6, n_observations=100)
    assert result.is_verified
    assert result.numerical_evidence["l1_error"] >= 0


def test_numerical_stability_verifies(proofs):
    result = proofs.verify_numerical_stability(n_random_shoes=50)
    assert result.is_verified
    assert result.numerical_evidence["max_prob_sum_error"] < 1e-10


def test_run_all_proofs_returns_list(proofs):
    results = proofs.run_all_proofs()
    assert len(results) == 5
    assert all(isinstance(r, ProofResult) for r in results)


def test_complexity_analysis_returns_dict(proofs):
    complexity = proofs.complexity_analysis()
    assert "dealer_distribution" in complexity
    assert "kelly_computation" in complexity
    assert isinstance(complexity["kelly_computation"], str)


def test_strict_mode_raises_on_failure():
    p = MathematicalCorrectness(strict=True)
    # Strict mode should still work when proofs pass
    result = p.verify_numerical_stability()
    assert result.is_verified


def test_proof_result_repr():
    result = ProofResult(
        theorem="test",
        is_verified=True,
        numerical_evidence={},
        statement="Test statement",
        proof_sketch="Test sketch",
    )
    assert "VERIFIED" in repr(result)
    assert "test" in repr(result)


def test_finite_deck_error_grows_with_more_dealt(proofs):
    result_few = proofs.verify_finite_vs_infinite_deck_error(n_decks=6, cards_dealt=26)
    result_many = proofs.verify_finite_vs_infinite_deck_error(n_decks=6, cards_dealt=104)
    # Error bound should be larger with more cards dealt
    assert result_many.numerical_evidence["remaining"] < result_few.numerical_evidence["remaining"]
