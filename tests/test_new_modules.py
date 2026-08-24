"""Tests for rl_bet_sizing, dealer_tells, surveillance, monitoring, database, security."""

import math
import time
import pytest
import numpy as np


# ===========================================================================
# RL Bet Sizing
# ===========================================================================

from blackjack.rl_bet_sizing import (
    RLBetOptimizer,
    _discretize_true_count,
    _discretize_penetration,
    _discretize_bankroll,
)


def test_discretize_true_count_bounds():
    assert _discretize_true_count(-10) == 0  # clamped to -6
    assert _discretize_true_count(10) == 12   # clamped to +6
    assert _discretize_true_count(0) == 6


def test_discretize_penetration():
    assert _discretize_penetration(0.0) == 0
    assert _discretize_penetration(0.5) == 2
    assert _discretize_penetration(1.0) == 3


def test_discretize_bankroll():
    assert _discretize_bankroll(0.3) == 0
    assert _discretize_bankroll(0.75) == 1
    assert _discretize_bankroll(1.5) == 2
    assert _discretize_bankroll(10.0) == 4


def test_rl_optimizer_choose_bet_range():
    rl = RLBetOptimizer(min_bet=5.0, max_bet=500.0)
    bet = rl.choose_bet(true_count=2.0, penetration=0.5)
    assert 5.0 <= bet <= 500.0


def test_rl_optimizer_greedy_uses_qtable():
    rl = RLBetOptimizer(min_bet=5.0, max_bet=200.0)
    # State: tc=0 → bucket 6, pen=0.45 → bucket 1, bk=0.75 → bucket 1
    # Set Q-table so action index 2 (multiplier=4) is best
    rl._Q[6, 1, 1, 2] = 10.0
    bet = rl.choose_bet(true_count=0.0, penetration=0.45, bankroll_ratio=0.75, explore=False)
    assert bet == pytest.approx(4 * 5.0, abs=0.01)


def test_rl_optimizer_qtable_stats():
    rl = RLBetOptimizer()
    stats = rl.q_table_stats()
    assert stats["q_mean"] == 0.0  # untrained
    assert stats["n_states_visited"] == 0


def test_rl_optimizer_train_basic():
    rl = RLBetOptimizer(min_bet=5.0, max_bet=100.0, epsilon_decay=0.99)
    rng = np.random.default_rng(42)

    def simulator(bet):
        profit = rng.choice([-bet, bet], p=[0.48, 0.52])
        tc_next = float(rng.normal(0, 2))
        pen_next = float(min(1.0, rng.uniform(0.3, 0.8)))
        return profit, tc_next, pen_next

    result = rl.train(simulator, n_episodes=20, hands_per_episode=20)
    assert result.n_episodes == 20
    assert 0 <= result.win_rate <= 1.0
    assert result.final_epsilon < 1.0


def test_rl_compare_with_kelly():
    rl = RLBetOptimizer(min_bet=5.0, max_bet=500.0)
    comparison = rl.compare_with_kelly(
        true_count=2.0, ev_per_hand=0.02, bankroll=1000.0
    )
    assert "kelly_bet" in comparison
    assert "rl_bet" in comparison
    assert comparison["kelly_bet"] >= 5.0
    assert comparison["rl_bet"] >= 5.0


# ===========================================================================
# Dealer Tells
# ===========================================================================

from blackjack.dealer_tells import DealerTellDetector, TellObservation


@pytest.fixture
def detector():
    return DealerTellDetector(min_observations=10)


def test_runs_test_insufficient_data(detector):
    result = detector.runs_test(sequence=[1, 2, 3])
    assert "error" in result


def test_runs_test_random_sequence(detector):
    rng = np.random.default_rng(42)
    seq = rng.normal(0, 1, 100)
    result = detector.runs_test(sequence=seq)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_autocorrelation_test_uncorrelated(detector):
    rng = np.random.default_rng(42)
    seq = rng.normal(0, 1, 100)
    result = detector.autocorrelation_test(lag=1, sequence=seq)
    assert "rho" in result
    assert abs(result["rho"]) < 1.0


def test_shuffle_bias_insufficient_data(detector):
    result = detector.shuffle_bias_test(["T", "A", "2"])
    assert not result.is_biased  # not enough data to conclude


def test_shuffle_bias_uniform_not_biased(detector):
    # Build a sequence with correct proportions
    per_deck = {'2': 4, '3': 4, '4': 4, '5': 4, '6': 4,
                '7': 4, '8': 4, '9': 4, 'T': 16, 'A': 4}
    deck = []
    for r, c in per_deck.items():
        deck.extend([r] * c * 3)
    result = detector.shuffle_bias_test(deck)
    assert not result.is_biased


def test_tell_outcome_correlation_insufficient(detector):
    sig = detector.tell_outcome_correlation("nonexistent")
    assert sig is None


def test_tell_outcome_correlation_structure(detector):
    for i in range(50):
        detector.record_tell_outcome("speed", float(i % 5), 1 if i % 3 else -1)
    sig = detector.tell_outcome_correlation("speed")
    if sig is not None:
        assert 0.0 <= sig.p_value <= 1.0


def test_anomaly_scores(detector):
    for i in range(10):
        obs = TellObservation("shuffle_speed", float(i), hand_number=i)
        detector.observe_tell(obs)
    scores = detector.anomaly_scores()
    assert "shuffle_speed" in scores
    assert scores["shuffle_speed"] >= 0


# ===========================================================================
# Surveillance
# ===========================================================================

from blackjack.surveillance import SurveillanceAnalysis


def test_initial_heat_zero():
    s = SurveillanceAnalysis()
    assert s.heat == 0.0


def test_heat_increases_with_large_bet_spread():
    s = SurveillanceAnalysis()
    s.record_bet(bet=200.0, min_bet=5.0, max_bet=500.0)
    assert s.heat > 0


def test_heat_stays_bounded():
    s = SurveillanceAnalysis()
    for i in range(100):
        s.record_bet(500.0, 5.0, 500.0)
    assert s.heat <= 1.0


def test_heat_label(detector):
    s = SurveillanceAnalysis(initial_heat=0.8)
    assert "hot" in s.heat_label.lower() or s.heat_label == "⚠ CRITICAL"


def test_recommend_empty_when_cool():
    s = SurveillanceAnalysis(initial_heat=0.1)
    recs = s.recommend_obfuscation(current_ev=0.02, min_bet=5.0)
    assert recs == []


def test_recommend_when_hot():
    s = SurveillanceAnalysis(initial_heat=0.75)
    recs = s.recommend_obfuscation(current_ev=0.02, min_bet=5.0)
    assert len(recs) > 0
    assert recs[0].priority == 1  # most urgent first


def test_record_win_streak_increases_heat():
    s = SurveillanceAnalysis()
    for _ in range(10):
        s.record_outcome(+1)  # all wins
    assert s.heat > 0


def test_summary_structure():
    s = SurveillanceAnalysis()
    summary = s.summary()
    assert "heat" in summary
    assert "heat_label" in summary
    assert "at_critical" in summary


def test_optimal_vs_survival_tradeoff():
    s = SurveillanceAnalysis(initial_heat=0.2)
    result = s.optimal_vs_survival_tradeoff(ev_per_hand=0.02, bankroll=1000.0)
    assert "full_kelly_total_ev" in result
    assert result["full_kelly_expected_hands"] > 0


# ===========================================================================
# Monitoring
# ===========================================================================

from blackjack.monitoring import ModelMonitoring


@pytest.fixture
def monitor():
    return ModelMonitoring(window_size=100)


def test_record_prediction_returns_id(monitor):
    rec_id = monitor.record_prediction(0.05, "hit", latency_ms=2.0)
    assert isinstance(rec_id, str)
    assert len(rec_id) > 0


def test_current_mae_none_when_no_outcomes(monitor):
    monitor.record_prediction(0.05, "stand")
    assert monitor.current_mae() is None


def test_current_mae_correct(monitor):
    rid = monitor.record_prediction(0.05, "stand")
    monitor.record_outcome(rid, 0.03)
    mae = monitor.current_mae()
    assert mae is not None
    assert mae == pytest.approx(0.02, abs=1e-6)


def test_latency_percentiles(monitor):
    for i in range(20):
        monitor.record_prediction(0.0, "hit", latency_ms=float(i))
    lat = monitor.latency_percentiles()
    assert "p50" in lat
    assert lat["p50"] >= 0


def test_ab_test_result_no_data(monitor):
    result = monitor.ab_test_result("A", "B")
    assert result is None


def test_ab_test_result_with_data(monitor):
    for _ in range(20):
        monitor.ab_test_record("model_v1", 0.05)
        monitor.ab_test_record("model_v2", 0.10)
    result = monitor.ab_test_result("model_v1", "model_v2")
    assert result is not None
    assert result.winner in {"model_v1", "model_v2", "no_winner"}


def test_health_report_structure(monitor):
    report = monitor.health_report()
    assert "model_version" in report
    assert "is_accurate" in report
    assert "total_predictions" in report


def test_audit_log(monitor):
    monitor.record_prediction(0.02, "stand")
    log = monitor.audit_log()
    assert len(log) > 0
    assert log[0]["type"] == "prediction"


def test_ks_feature_drift_detects_shift(monitor):
    ref = list(np.random.default_rng(42).normal(0, 1, 100))
    cur = list(np.random.default_rng(42).normal(5, 1, 100))  # shifted
    alert = monitor.ks_feature_drift(ref, cur, feature_name="true_count")
    assert alert is not None  # significant shift should be detected


# ===========================================================================
# Database
# ===========================================================================

from blackjack.database import EnterpriseDatabase, HandRecord, SessionRecord


@pytest.fixture
def db():
    d = EnterpriseDatabase(db_path=":memory:")
    yield d
    d.close()


def test_db_inserts_and_counts(db):
    hand = HandRecord(
        session_id="s1", hand_number=1, hand_total=16, hand_is_soft=False,
        dealer_upcard="T", true_count=-1.0, running_count=-2, deck_penetration=0.5,
        observation_ratio=0.8, recommended_action="hit", predicted_ev=-0.05,
        bet_size=10.0, bankroll=1000.0,
    )
    db.insert_hand(hand)
    db.flush()
    assert db.total_hands() == 1


def test_db_batch_insert(db):
    for i in range(50):
        db.insert_hand(HandRecord(
            session_id="s1", hand_number=i, hand_total=16 + (i % 5),
            hand_is_soft=False, dealer_upcard="T",
            true_count=float(i % 6 - 3), running_count=i % 10,
            deck_penetration=0.5, observation_ratio=0.9,
            recommended_action="stand", predicted_ev=0.02,
            bet_size=10.0, bankroll=1000.0 - i * 5,
        ))
    db.flush()
    assert db.total_hands() == 50


def test_db_update_outcome(db):
    db.insert_hand(HandRecord(
        session_id="s1", hand_number=1, hand_total=20, hand_is_soft=False,
        dealer_upcard="6", true_count=2.0, running_count=4, deck_penetration=0.6,
        observation_ratio=1.0, recommended_action="stand", predicted_ev=0.20,
        bet_size=20.0, bankroll=1000.0,
    ))
    db.flush()
    db.update_outcome("s1", 1, actual_outcome=1.0)


def test_db_ev_by_true_count(db):
    for i in range(20):
        db.insert_hand(HandRecord(
            session_id="s1", hand_number=i, hand_total=16, hand_is_soft=False,
            dealer_upcard="T", true_count=float(i % 6 - 3), running_count=0,
            deck_penetration=0.5, observation_ratio=0.9,
            recommended_action="hit", predicted_ev=0.02,
            bet_size=5.0, bankroll=1000.0, actual_outcome=1.0 if i % 2 else -1.0,
        ))
    db.flush()
    result = db.ev_by_true_count(min_hands=1)
    assert isinstance(result, list)


def test_db_session_management(db):
    session = SessionRecord(
        session_id="test_session",
        n_decks=6,
        table_rules={"s17": True},
        starting_bankroll=1000.0,
    )
    db.start_session(session)
    db.end_session("test_session", ending_bankroll=1050.0, total_hands=50,
                   win_rate=0.52, total_ev=50.0)
    summary = db.session_summary("test_session")
    assert summary is not None
    assert summary["ending_bankroll"] == 1050.0


# ===========================================================================
# Security
# ===========================================================================

from blackjack.security import (
    AccessControl, Role, AuditLog, PIIAnonymiser,
    sanitise_rank, sanitise_float,
    encrypt_data, decrypt_data,
    verify_password, _pbkdf2_hash,
)


def test_password_hash_and_verify():
    pw_data = _pbkdf2_hash("correct_password")
    assert verify_password("correct_password", pw_data["hash"], pw_data["salt"])
    assert not verify_password("wrong_password", pw_data["hash"], pw_data["salt"])


def test_encrypt_decrypt_roundtrip():
    import secrets as sec
    key = sec.token_bytes(32)
    plaintext = "sensitive blackjack data: EV=0.025"
    encrypted = encrypt_data(plaintext, key)
    decrypted = decrypt_data(encrypted, key)
    assert decrypted == plaintext


def test_decrypt_tampered_returns_none():
    import secrets as sec
    key = sec.token_bytes(32)
    encrypted = encrypt_data("secret", key)
    # Tamper: flip one character
    chars = list(encrypted)
    chars[-5] = "x" if chars[-5] != "x" else "y"
    tampered = "".join(chars)
    result = decrypt_data(tampered, key)
    assert result is None


def test_audit_log_chain_integrity():
    log = AuditLog()
    log.record("user1", "login", "session", "success")
    log.record("user1", "get_recommendation", "ev_engine", "success")
    log.record("user2", "login", "session", "denied")
    assert log.verify_chain() is True


def test_audit_log_tamper_detected():
    log = AuditLog()
    log.record("user1", "login", "session", "success")
    log.record("user1", "action", "resource", "success")
    # Tamper with first entry
    log._entries[0].actor = "evil_user"
    assert log.verify_chain() is False


def test_access_control_create_authenticate():
    ac = AccessControl()
    user = ac.create_user("alice", "securepassword123", Role.PLAYER)
    assert user.username == "alice"
    assert user.role == Role.PLAYER

    auth_user = ac.authenticate("alice", "securepassword123")
    assert auth_user is not None

    bad_auth = ac.authenticate("alice", "wrongpassword")
    assert bad_auth is None


def test_access_control_permissions():
    ac = AccessControl()
    user = ac.create_user("bob", "password", Role.VIEWER)
    assert ac.check_permission(user, "read_analytics")
    assert not ac.check_permission(user, "retrain_model")


def test_access_control_require_permission_raises():
    ac = AccessControl()
    user = ac.create_user("charlie", "password", Role.VIEWER)
    with pytest.raises(PermissionError):
        ac.require_permission(user, "retrain_model")


def test_pii_anonymiser_consistent():
    anon = PIIAnonymiser(secret_key=b"test_key_32_bytes_long_test_key!")
    assert anon.pseudonymise("alice@example.com") == anon.pseudonymise("alice@example.com")
    assert anon.pseudonymise("alice@example.com") != anon.pseudonymise("bob@example.com")


def test_pii_anonymiser_record():
    anon = PIIAnonymiser()
    record = {"name": "Alice", "ev": 0.02, "hand_total": 16}
    anon_record = anon.anonymise_record(record)
    assert anon_record["ev"] == 0.02
    assert anon_record["name"] != "Alice"


def test_sanitise_rank_valid():
    assert sanitise_rank("T") == "T"
    assert sanitise_rank("A") == "A"
    assert sanitise_rank("j") == "T"   # J → T
    assert sanitise_rank("Q") == "T"   # Q → T
    assert sanitise_rank("k") == "T"   # K → T
    assert sanitise_rank("2") == "2"


def test_sanitise_rank_invalid():
    with pytest.raises(ValueError):
        sanitise_rank("XYZ")
    with pytest.raises(ValueError):
        sanitise_rank("")


def test_sanitise_float_valid():
    assert sanitise_float(0.5, 0.0, 1.0) == 0.5
    assert sanitise_float("0.3", 0.0, 1.0) == pytest.approx(0.3)


def test_sanitise_float_out_of_range():
    with pytest.raises(ValueError):
        sanitise_float(2.0, 0.0, 1.0, "test")


def test_sanitise_float_non_numeric():
    with pytest.raises(ValueError):
        sanitise_float("abc", 0.0, 1.0, "test")
