from blackjack.ui_state import AppState, HandHistoryEntry, format_ev_table, health_status


def test_compute_evs_empty_when_player_has_fewer_than_two_cards():
    state = AppState()
    state.set_player_cards(["A"])
    state.set_dealer_upcard("6")
    assert state.compute_evs() == {}


def test_compute_evs_non_empty_for_valid_hand():
    state = AppState()
    state.set_player_cards(["8", "8"])
    state.set_dealer_upcard("6")
    evs = state.compute_evs()
    assert evs
    assert "stand" in evs


def test_update_rules_updates_selected_flags():
    state = AppState()
    state.update_rules(dealer_hits_soft17=True, double_after_split=False, max_splits=2, surrender="none")
    assert state.rules.dealer_hits_soft17 is True
    assert state.rules.double_after_split is False
    assert state.rules.max_splits == 2
    assert state.rules.surrender == "none"


def test_export_history_csv_has_expected_headers():
    state = AppState()
    state.set_player_cards(["A", "T"])
    state.set_dealer_upcard("6")
    state.log_hand(action_taken="stand", outcome="win")
    csv_text = state.export_history_csv()
    assert csv_text.splitlines()[0] == "hand,player_cards,dealer_upcard,recommended,best_ev,action_taken,outcome,confidence"


def test_format_ev_table_sorts_best_first_and_marks_best_delta():
    rows = format_ev_table({"hit": -0.1, "stand": 0.2, "double": 0.15})
    assert rows[0]["action"] == "STAND"
    assert rows[0]["delta"] == "best"


def test_health_status_confidence_bands():
    assert health_status(0.90) == ("Good", "#16a34a")
    assert health_status(0.80) == ("Review", "#f59e0b")
    assert health_status(0.70) == ("Low – verify manually", "#dc2626")


def test_log_hand_uses_hand_history_entry_dataclass():
    state = AppState()
    state.set_player_cards(["A", "T"])
    state.set_dealer_upcard("6")
    state.log_hand(action_taken="stand", outcome="win")
    assert isinstance(state.hand_history[0], HandHistoryEntry)
