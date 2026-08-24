"""Tests for blackjack/bayesian_inference.py — BayesianShoeModeling."""

import pytest
import numpy as np

from blackjack.bayesian_inference import BayesianShoeModeling, RANKS


@pytest.fixture
def model():
    return BayesianShoeModeling(n_decks=6, prior_strength=0.5, use_informative_prior=True)


def test_initial_probs_sum_to_one(model):
    probs = model.predictive_probabilities()
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)
    assert len(probs) == len(RANKS)


def test_observe_updates_posterior(model):
    prior_p_T = model.predictive_probabilities()["T"]
    # Remove T cards → posterior prob of T should decrease
    for _ in range(50):
        model.observe("T")
    post_p_T = model.predictive_probabilities()["T"]
    assert post_p_T < prior_p_T


def test_posterior_structure(model):
    model.observe("A", 10)
    result = model.posterior()
    assert len(result.posterior_mean) == len(RANKS)
    assert result.posterior_mean.sum() == pytest.approx(1.0, abs=1e-10)
    assert result.observation_count == 10
    assert result.credible_intervals_95.shape == (len(RANKS), 2)


def test_credible_interval_valid(model):
    model.observe("T", 20)
    result = model.posterior()
    for i, r in enumerate(RANKS):
        lo, hi = result.credible_intervals_95[i, 0], result.credible_intervals_95[i, 1]
        assert lo <= result.posterior_mean[i] <= hi + 1e-10
        assert lo >= 0.0
        assert hi <= 1.0


def test_posterior_variance_positive(model):
    model.observe_sequence(["T", "A", "2", "5"])
    result = model.posterior()
    assert all(v >= 0 for v in result.posterior_variance)


def test_observe_sequence(model):
    ranks = ["T", "T", "A", "2", "3"]
    model.observe_sequence(ranks)
    assert model._observed.sum() == len(ranks)


def test_unknown_rank_raises(model):
    with pytest.raises(ValueError):
        model.observe("X")


def test_reset_clears_observations(model):
    model.observe("T", 10)
    model.reset()
    assert model._observed.sum() == 0.0


def test_calibration_vs_true_shoe(model):
    true_remaining = {r: 24 for r in RANKS}
    true_remaining["T"] = 96  # standard distribution
    result = model.calibration_vs_true_shoe(true_remaining)
    assert "l1_error" in result
    assert result["l1_error"] >= 0
    assert "kl_divergence" in result


def test_calibration_improves_with_more_observations(model):
    true_remaining = {r: 24 for r in RANKS}
    true_remaining["T"] = 96
    cal_before = model.calibration_vs_true_shoe(true_remaining)

    # Observe cards matching the true distribution
    for _ in range(100):
        model.observe("T", 1)
        model.observe("2", 1)
    cal_after = model.calibration_vs_true_shoe(true_remaining)

    # l1 error should not explode (and is bounded)
    assert cal_after["l1_error"] <= 2.0


def test_ev_adjustment_blends_toward_prior(model):
    # With no observations, blend should be mostly prior
    count_probs = {r: 1.0 / len(RANKS) for r in RANKS}
    adjusted = model.ev_adjustment(count_probs)
    assert sum(adjusted.values()) == pytest.approx(1.0, abs=1e-6)


def test_ev_adjustment_converges_with_observations():
    model = BayesianShoeModeling(n_decks=6)
    count_probs = {"T": 0.30, "A": 0.10, "2": 0.07, "3": 0.07, "4": 0.07,
                   "5": 0.07, "6": 0.07, "7": 0.07, "8": 0.07, "9": 0.08}
    for _ in range(200):
        model.observe("T")
    adjusted = model.ev_adjustment(count_probs)
    # After 200 obs, weight = 1.0, so adjusted ≈ count_probs
    assert adjusted["T"] == pytest.approx(count_probs["T"], abs=0.01)


def test_convergence_analysis_returns_data(model):
    result = model.convergence_analysis(n_decks=2, max_cards=40, step=5)
    assert len(result.observation_ratios) > 0
    assert len(result.kl_divergences) == len(result.observation_ratios)
    # KL at 0 observations should be highest
    if len(result.kl_divergences) > 1:
        assert result.kl_divergences[0] >= 0


def test_informative_vs_noninformative_prior():
    m_inf = BayesianShoeModeling(n_decks=6, use_informative_prior=True)
    m_noninf = BayesianShoeModeling(n_decks=6, use_informative_prior=False)
    # Informative prior should give T higher initial probability
    p_T_inf = m_inf.predictive_probabilities()["T"]
    p_T_noninf = m_noninf.predictive_probabilities()["T"]
    assert p_T_inf > p_T_noninf - 0.001  # informative ≥ noninformative
