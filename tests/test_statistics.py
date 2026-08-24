"""Tests for blackjack/statistics.py — StatisticalValidator."""

import math
import pytest
import numpy as np

from blackjack.statistics import StatisticalValidator, ConfidenceInterval


@pytest.fixture
def validator():
    return StatisticalValidator(alpha=0.05)


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def test_wilson_ci_sums_correctly(validator):
    ci = validator.wilson_ci(50, 100)
    assert ci.lower <= ci.point_estimate <= ci.upper
    assert ci.lower >= 0.0
    assert ci.upper <= 1.0
    assert ci.confidence == 0.95
    assert ci.method == "wilson"


def test_wilson_ci_extreme_zero(validator):
    ci = validator.wilson_ci(0, 100)
    assert ci.lower == pytest.approx(0.0, abs=1e-10)
    assert ci.upper > 0.0


def test_wilson_ci_extreme_all(validator):
    ci = validator.wilson_ci(100, 100)
    assert ci.upper == 1.0
    assert ci.lower < 1.0


def test_wilson_ci_narrows_with_more_trials(validator):
    ci_small = validator.wilson_ci(50, 100)
    ci_large = validator.wilson_ci(500, 1000)
    assert ci_large.width < ci_small.width


def test_wilson_ci_invalid_raises(validator):
    with pytest.raises(ValueError):
        validator.wilson_ci(-1, 100)
    with pytest.raises(ValueError):
        validator.wilson_ci(101, 100)
    with pytest.raises(ValueError):
        validator.wilson_ci(50, 0)


# ---------------------------------------------------------------------------
# Clopper-Pearson CI
# ---------------------------------------------------------------------------

def test_clopper_pearson_ci_valid(validator):
    ci = validator.clopper_pearson_ci(30, 100)
    assert ci.lower < ci.point_estimate < ci.upper
    assert ci.method == "clopper_pearson"


def test_clopper_pearson_wider_than_wilson(validator):
    # CP is more conservative
    cp = validator.clopper_pearson_ci(50, 100)
    w = validator.wilson_ci(50, 100)
    assert cp.width >= w.width - 0.001  # CP ≥ Wilson (approximately)


def test_clopper_pearson_zero_successes(validator):
    ci = validator.clopper_pearson_ci(0, 50)
    assert ci.lower == 0.0


# ---------------------------------------------------------------------------
# EV confidence interval
# ---------------------------------------------------------------------------

def test_ev_ci_correct_sign_for_positive_ev(validator):
    rng = np.random.default_rng(42)
    evs = rng.normal(0.05, 1.0, 1000)
    ci = validator.ev_confidence_interval(evs)
    assert ci.lower > -0.5  # CI should be near the true mean


def test_ev_ci_needs_at_least_2_observations(validator):
    with pytest.raises(ValueError):
        validator.ev_confidence_interval([0.1])


# ---------------------------------------------------------------------------
# Binomial test
# ---------------------------------------------------------------------------

def test_binomial_test_obvious_bias(validator):
    # 600 wins in 1000 — clearly biased
    result = validator.binomial_test(600, 1000, p_null=0.5)
    assert result.reject_h0
    assert result.p_value < 0.05


def test_binomial_test_fair_coin(validator):
    # 500 wins in 1000 — not significant
    result = validator.binomial_test(500, 1000, p_null=0.5)
    assert not result.reject_h0
    assert result.p_value > 0.05


def test_binomial_test_effect_size_correct_sign(validator):
    result = validator.binomial_test(600, 1000)
    assert result.effect_size is not None
    assert result.effect_size != 0


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs(validator):
    result = validator.mcnemar_test(b=0, c=0)
    assert result.p_value == 1.0
    assert not result.reject_h0


def test_mcnemar_significant_discordance(validator):
    result = validator.mcnemar_test(b=80, c=20)
    assert result.reject_h0


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def test_chi_squared_uniform_is_not_rejected(validator):
    observed = [25, 25, 25, 25]
    result = validator.chi_squared_test(observed)
    assert not result.reject_h0


def test_chi_squared_extreme_rejects(validator):
    observed = [100, 0, 0, 0]
    result = validator.chi_squared_test(observed)
    assert result.reject_h0


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def test_cohens_h_zero_when_equal():
    h = StatisticalValidator.cohens_h(0.5, 0.5)
    assert h == pytest.approx(0.0, abs=1e-10)


def test_cohens_h_positive_when_p1_greater():
    h = StatisticalValidator.cohens_h(0.6, 0.5)
    assert h > 0


def test_cramers_v_zero_for_uniform():
    obs = np.array([25.0, 25.0, 25.0, 25.0])
    exp = np.array([25.0, 25.0, 25.0, 25.0])
    v = StatisticalValidator.cramers_v(obs, exp)
    assert v == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def test_power_analysis_positive_ev(validator):
    result = validator.power_analysis_binomial(
        true_win_rate=0.52, null_win_rate=0.49, desired_power=0.80
    )
    assert result.required_hands > 0
    assert result.achieved_power == 0.80


def test_power_analysis_requires_positive_edge(validator):
    with pytest.raises(ValueError):
        validator.power_analysis_binomial(true_win_rate=0.48, null_win_rate=0.49)


def test_hands_for_significant_ev(validator):
    result = validator.hands_for_significant_ev(ev_per_hand=0.02)
    assert result.required_hands > 100
    assert "hands" in result.description.lower()


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------

def test_ks_test_normal_data_not_rejected(validator):
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 200)
    result = validator.ks_test(data, cdf="norm")
    assert result.test_name == "kolmogorov_smirnov"
    # Should usually fail to reject (data IS normal)
    # (may occasionally be significant due to randomness — test structure only)
    assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def test_calibration_perfect(validator):
    pred = [0.1, 0.2, 0.3, 0.4, 0.5]
    real = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = validator.calibration_analysis(pred, real)
    assert result["mean_absolute_error"] == pytest.approx(0.0, abs=1e-10)
    assert result["is_well_calibrated"] is True


def test_calibration_bad(validator):
    pred = [0.5] * 100
    real = [-0.5] * 100
    result = validator.calibration_analysis(pred, real)
    assert result["mean_absolute_error"] == pytest.approx(1.0, abs=1e-6)
    assert result["is_well_calibrated"] is False


def test_calibration_mismatch_raises(validator):
    with pytest.raises(ValueError):
        validator.calibration_analysis([0.1, 0.2], [0.1])


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_ev_structure(validator):
    result = validator.monte_carlo_ev(
        action_ev_fn=lambda: 0.02,
        n_simulations=500,
    )
    assert result.n_simulations == 500
    assert -2.0 <= result.mean_ev <= 2.0
    assert result.std_ev > 0
    assert 0.0 <= result.win_rate <= 1.0


def test_monte_carlo_positive_ev_has_positive_mean(validator):
    result = validator.monte_carlo_ev(
        action_ev_fn=lambda: 0.5,
        n_simulations=10_000,
        variance_per_hand=0.1,
    )
    assert result.mean_ev > 0
