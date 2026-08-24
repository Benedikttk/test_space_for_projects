"""Tests for benchmarks, medium_comparison, and API endpoints."""

import math
import pytest
import numpy as np


# ===========================================================================
# Benchmarks
# ===========================================================================

from blackjack.benchmarks import BenchmarkSuite, LatencyResult, StrategyComparisonResult


@pytest.fixture
def suite():
    return BenchmarkSuite(n_latency_trials=20, n_strategy_hands=50)


def test_benchmark_latency_result_structure(suite):
    result = suite.benchmark_latency(lambda: None, "no-op")
    assert isinstance(result, LatencyResult)
    assert result.p50_ms >= 0
    assert result.p95_ms >= result.p50_ms
    assert result.p99_ms >= result.p95_ms
    assert result.n_trials == suite.n_latency_trials


def test_benchmark_ev_engine_meets_sla(suite):
    result = suite.benchmark_ev_engine()
    assert result.operation == "EV engine (action_evs)"
    # Should complete in well under 100ms P99
    assert result.meets_sla  # P99 < 100ms


def test_benchmark_shoe_update_fast(suite):
    result = suite.benchmark_shoe_update()
    assert result.p50_ms < 1.0  # very fast O(1) operation


def test_benchmark_kelly_fast(suite):
    result = suite.benchmark_kelly()
    assert result.p50_ms < 5.0  # closed-form, must be < 5ms


def test_strategy_comparison_returns_three_results(suite):
    results = suite.compare_strategies()
    assert len(results) == 3
    names = [r.strategy_name for r in results]
    assert "Basic Strategy" in names
    assert "Full EV Engine" in names


def test_full_ev_engine_ev_not_worse_than_basic(suite):
    results = suite.compare_strategies()
    by_name = {r.strategy_name: r for r in results}
    full_ev = by_name["Full EV Engine"].mean_ev
    basic = by_name["Basic Strategy"].mean_ev
    # Full EV engine should be at least as good as basic strategy
    assert full_ev >= basic - 0.05  # allow some statistical variation


def test_benchmark_report_to_markdown():
    # Use a minimal suite for report generation
    mini_suite = BenchmarkSuite(n_latency_trials=5, n_strategy_hands=20)
    report = mini_suite.run_all()
    md = report.to_markdown()
    assert "# Blackjack" in md
    assert "Latency" in md
    assert "Strategy Comparison" in md


# ===========================================================================
# Medium Comparison
# ===========================================================================

from blackjack.medium_comparison import MediumArticleAnalysis, ComparisonSummary


@pytest.fixture
def medium():
    return MediumArticleAnalysis()


def test_medium_article_features_shape(medium):
    features = medium.build_medium_article_features(
        hand_total=16, dealer_upcard="T", true_count=1.0,
        deck_penetration=0.5, hand_is_soft=False,
    )
    assert features.shape == (8,)
    assert np.all(np.isfinite(features))


def test_medium_win_prob_heuristic_bounded(medium):
    p = medium.predict_ml_win_prob(16, "T", 0.0)
    assert 0.0 <= p <= 1.0


def test_medium_win_prob_higher_for_strong_hand(medium):
    p_strong = medium.predict_ml_win_prob(20, "6", 0.0)
    p_weak = medium.predict_ml_win_prob(12, "T", 0.0)
    assert p_strong > p_weak


def test_medium_comparison_runs(medium):
    summary = medium.compare_on_hands(n_random=100)
    assert isinstance(summary, ComparisonSummary)
    assert summary.n_hands == 100
    assert 0.0 <= summary.agreement_rate <= 1.0
    assert summary.ev_wins_over_ml + summary.ml_wins_over_ev <= 100


def test_medium_comparison_paper(medium):
    summary = medium.compare_on_hands(n_random=50)
    paper = medium.generate_comparison_paper(summary)
    assert "Abstract" in paper
    assert "EV Engine" in paper
    assert "50" in paper


def test_medium_article_model_trains(medium):
    rng = np.random.default_rng(42)
    training_data = [
        {
            "hand_total": int(rng.integers(5, 21)),
            "dealer_upcard": str(rng.choice(["T", "A", "6", "3"])),
            "true_count": float(rng.normal(0, 2)),
            "deck_penetration": float(rng.uniform(0.2, 0.8)),
            "hand_is_soft": bool(rng.integers(0, 2)),
            "outcome": int(rng.integers(0, 2)),
        }
        for _ in range(200)
    ]
    medium.train_medium_article_model(training_data)
    assert medium._fitted is True

    # After training, ML predict should work
    p = medium.predict_ml_win_prob(16, "T", 0.0)
    assert 0.0 <= p <= 1.0


# ===========================================================================
# API Tests (if FastAPI available)
# ===========================================================================

try:
    from fastapi.testclient import TestClient
    from blackjack.api import app, create_api_token
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="FastAPI not installed")
class TestAPI:
    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(app)
        self.token = create_api_token("test_user", role="admin")
        self.auth_headers = {"Authorization": "Bearer " + self.token}

    def test_health_endpoint_no_auth(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "is_healthy" in data

    def test_recommend_endpoint(self):
        resp = self.client.post(
            "/recommend",
            headers=self.auth_headers,
            json={
                "hand": ["T", "6"],
                "dealer_upcard": "T",
                "n_decks": 6,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "ev" in data
        assert -1.0 <= data["ev"] <= 1.0
        assert data["latency_ms"] >= 0

    def test_recommend_endpoint_no_auth(self):
        resp = self.client.post(
            "/recommend",
            json={"hand": ["T", "6"], "dealer_upcard": "T"},
        )
        assert resp.status_code == 401

    def test_recommend_invalid_card_raises(self):
        resp = self.client.post(
            "/recommend",
            headers=self.auth_headers,
            json={"hand": ["XYZ", "6"], "dealer_upcard": "T"},
        )
        assert resp.status_code in (422, 400)

    def test_stats_endpoint(self):
        resp = self.client.get("/stats", headers=self.auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "n_hands" in data
        assert "win_rate" in data

    def test_predict_endpoint(self):
        resp = self.client.post(
            "/predict",
            headers=self.auth_headers,
            json={
                "hand_total": 16,
                "dealer_upcard": "T",
                "true_count": 1.0,
                "deck_penetration": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_ev" in data
        assert "predicted_action" in data

    def test_log_hand_endpoint(self):
        resp = self.client.post(
            "/log_hand",
            headers=self.auth_headers,
            json={
                "session_id": "test_session",
                "hand_number": 1,
                "hand_total": 16,
                "dealer_upcard": "T",
                "true_count": 0.0,
                "recommended_action": "hit",
                "predicted_ev": -0.05,
                "actual_outcome": -1.0,
                "bet_size": 10.0,
                "bankroll": 990.0,
            },
        )
        assert resp.status_code == 200

    def test_blackjack_gives_positive_ev(self):
        # Natural blackjack (21 total) vs dealer 6 should be +EV
        resp = self.client.post(
            "/recommend",
            headers=self.auth_headers,
            json={
                "hand": ["T", "A"],
                "dealer_upcard": "6",
                "n_decks": 6,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ev"] > 0

    def test_token_endpoint(self):
        resp = self.client.post("/token", params={"user_id": "test", "role": "player"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
