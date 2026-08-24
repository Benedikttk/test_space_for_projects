"""Tests for blackjack/feature_engineering.py — FeatureEngineer."""

import pytest
import numpy as np

from blackjack.feature_engineering import FeatureEngineer, FEATURE_NAMES, N_FEATURES


@pytest.fixture
def fe():
    return FeatureEngineer()


def _default_shoe():
    return {r: 24 for r in ['2', '3', '4', '5', '6', '7', '8', '9', 'A']} | {"T": 96}


def test_feature_vector_length(fe):
    fv = fe.build_features(
        hand_total=16,
        hand_is_soft=False,
        hand_is_pair=False,
        dealer_upcard="T",
        shoe_counts=_default_shoe(),
    )
    assert len(fv.features) == N_FEATURES
    assert len(fv.feature_names) == N_FEATURES


def test_feature_names_match_constant(fe):
    fv = fe.build_features(16, False, False, "T", _default_shoe())
    assert fv.feature_names == FEATURE_NAMES


def test_soft_hand_sets_is_soft_flag(fe):
    fv = fe.build_features(17, True, False, "6", _default_shoe())
    idx = FEATURE_NAMES.index("hand_is_soft")
    assert fv.features[idx] == 1.0


def test_hard_hand_is_soft_false(fe):
    fv = fe.build_features(16, False, False, "T", _default_shoe())
    idx = FEATURE_NAMES.index("hand_is_soft")
    assert fv.features[idx] == 0.0


def test_pair_sets_is_pair_flag(fe):
    fv = fe.build_features(16, False, True, "8", _default_shoe())
    idx = FEATURE_NAMES.index("hand_is_pair")
    assert fv.features[idx] == 1.0


def test_true_count_affects_feature(fe):
    fv_neutral = fe.build_features(16, False, False, "T", _default_shoe(), running_count=0, decks_remaining=6)
    fv_positive = fe.build_features(16, False, False, "T", _default_shoe(), running_count=12, decks_remaining=6)
    tc_idx = FEATURE_NAMES.index("true_count")
    assert fv_positive.features[tc_idx] > fv_neutral.features[tc_idx]


def test_dealer_ace_sets_flag(fe):
    fv = fe.build_features(16, False, False, "A", _default_shoe())
    idx = FEATURE_NAMES.index("dealer_is_ace")
    assert fv.features[idx] == 1.0


def test_dealer_ten_sets_flag(fe):
    fv = fe.build_features(16, False, False, "T", _default_shoe())
    idx = FEATURE_NAMES.index("dealer_is_ten")
    assert fv.features[idx] == 1.0


def test_features_finite_and_bounded(fe):
    fv = fe.build_features(18, True, False, "6", _default_shoe(),
                           running_count=5, decks_remaining=3)
    assert np.all(np.isfinite(fv.features)), "Features must be finite"
    # Most features should be in reasonable range
    assert np.all(fv.features >= -10), "Features must be bounded below"
    assert np.all(fv.features <= 20), "Features must be bounded above"


def test_interaction_feature_tc_x_penetration(fe):
    fv = fe.build_features(16, False, False, "T", _default_shoe(),
                           running_count=6, decks_remaining=3,
                           observation_ratio=0.5)
    idx = FEATURE_NAMES.index("tc_x_penetration")
    tc_idx = FEATURE_NAMES.index("true_count")
    pen_idx = FEATURE_NAMES.index("deck_penetration")
    # Should be non-zero when true_count and penetration are both non-zero
    assert fv.features[idx] != 0 or fv.features[tc_idx] == 0 or fv.features[pen_idx] == 0


def test_bust_prob_high_for_near_bust_hand(fe):
    # Hard 20: almost any card busts
    fv = fe.build_features(20, False, False, "T", _default_shoe())
    idx = FEATURE_NAMES.index("hand_bust_prob")
    assert fv.features[idx] > 0.5  # most cards bust 20


def test_bust_prob_low_for_low_total(fe):
    fv = fe.build_features(5, False, False, "T", _default_shoe())
    idx = FEATURE_NAMES.index("hand_bust_prob")
    assert fv.features[idx] == 0.0  # impossible to bust with 5


def test_to_dataframe(fe):
    fvs = [
        fe.build_features(16, False, False, "T", _default_shoe()),
        fe.build_features(18, True, False, "6", _default_shoe()),
    ]
    df = fe.to_dataframe(fvs)
    assert df.shape == (2, N_FEATURES)
    assert list(df.columns) == FEATURE_NAMES


def test_feature_importance_from_ev(fe):
    rng = np.random.default_rng(42)
    fvs = [
        fe.build_features(
            hand_total=int(rng.integers(5, 21)),
            hand_is_soft=bool(rng.integers(0, 2)),
            hand_is_pair=False,
            dealer_upcard=rng.choice(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']),
            shoe_counts=_default_shoe(),
        )
        for _ in range(50)
    ]
    evs = [float(rng.normal(0, 0.1)) for _ in range(50)]
    importances = fe.feature_importance_from_ev(fvs, evs)
    assert len(importances) == N_FEATURES
    assert all(0 <= v <= 1 for v in importances.values())
