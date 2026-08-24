from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from blackjack.capture import CaptureConfig, CaptureSession
from blackjack.shoe_state import ShoeState
from blackjack.ui_state import AppState, format_ev_table, health_status
from blackjack.ml_stats import MLStatsTracker, HandRecord
from blackjack.card_counter import CardCounterUI

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
OUTCOMES = ["win", "loss", "push", "surrender"]
ACTIONS = ["hit", "stand", "double", "split", "surrender"]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_capture_config(repo_root: Path) -> CaptureConfig:
    app_cfg = _load_yaml(repo_root / "configs" / "app.yaml")
    vision_cfg = _load_yaml(repo_root / "configs" / "vision.yaml")

    vision = app_cfg.get("vision", {})
    layout = vision_cfg.get("table_layout", {})
    player_regions = layout.get("player_regions", {})

    player_region = None
    other_regions: dict[str, tuple[int, int, int, int]] = {}
    if player_regions:
        if "seat_2" in player_regions:
            player_region = tuple(player_regions["seat_2"])
            other_regions = {
                seat: tuple(region)
                for seat, region in player_regions.items()
                if seat != "seat_2"
            }
        else:
            first_key = next(iter(player_regions))
            player_region = tuple(player_regions[first_key])
            other_regions = {
                seat: tuple(region)
                for seat, region in player_regions.items()
                if seat != first_key
            }

    return CaptureConfig(
        fps=float(vision.get("fps", 2.0)),
        monitor_index=int(vision.get("monitor_index", 1)),
        dealer_region=tuple(layout.get("dealer_region", [])) or None,
        player_region=player_region,
        other_player_regions=other_regions,
        backend=str(vision_cfg.get("detection", {}).get("card_model", "template")),
        template_dir=str(repo_root / "data" / "templates"),
        accept_threshold=float(vision.get("confidence_threshold", 0.85)),
        review_threshold=float(vision.get("review_threshold", 0.75)),
    )


def _cards_to_text(cards: list[str]) -> str:
    return " ".join(cards) if cards else "(none)"


def _rebuild_shoe_state(decks: int) -> None:
    old_capture = st.session_state.capture_session
    old_capture.stop()

    shoe_state = ShoeState(
        decks=decks,
        mid_shoe_join=False,
        accept_threshold=old_capture.config.accept_threshold,
        review_threshold=old_capture.config.review_threshold,
    )
    st.session_state.shoe_state = shoe_state
    st.session_state.app_state.shoe = shoe_state.shoe

    new_capture = CaptureSession(st.session_state.capture_config, shoe_state)
    new_capture.on_cards_observed = _on_cards_observed
    st.session_state.capture_session = new_capture
    st.session_state.app_state.camera_active = False


def _on_cards_observed(results) -> None:
    if not results:
        return
    conf = sum(r.confidence for r in results) / len(results)
    st.session_state.app_state.detection_confidence = conf


def _ensure_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()

    if "capture_config" not in st.session_state:
        st.session_state.capture_config = _build_capture_config(Path(__file__).resolve().parent)

    if "shoe_state" not in st.session_state:
        st.session_state.shoe_state = ShoeState(
            decks=st.session_state.app_state.shoe.decks,
            mid_shoe_join=False,
            accept_threshold=st.session_state.capture_config.accept_threshold,
            review_threshold=st.session_state.capture_config.review_threshold,
        )

    st.session_state.app_state.shoe = st.session_state.shoe_state.shoe

    if "capture_session" not in st.session_state:
        session = CaptureSession(st.session_state.capture_config, st.session_state.shoe_state)
        session.on_cards_observed = _on_cards_observed
        st.session_state.capture_session = session

    if "ml_stats" not in st.session_state:
        st.session_state.ml_stats = MLStatsTracker()

    if "card_counter" not in st.session_state:
        st.session_state.card_counter = CardCounterUI()


def main() -> None:
    st.set_page_config(page_title="Blackjack EV Engine", layout="wide")
    _ensure_state()

    app_state: AppState = st.session_state.app_state
    shoe_state: ShoeState = st.session_state.shoe_state
    capture: CaptureSession = st.session_state.capture_session

    st.title("Blackjack EV Engine")

    with st.sidebar:
        st.subheader("Live Table View")
        st.write(f"Camera status: {'🟢 active' if app_state.camera_active else '⚪ inactive'}")

        if app_state.camera_active:
            if st.button("Stop Capture", use_container_width=True):
                capture.stop()
                app_state.camera_active = False
        else:
            if st.button("Start Capture", use_container_width=True):
                capture.start()
                app_state.camera_active = True

        health_label, health_colour = health_status(app_state.detection_confidence)
        st.markdown(
            f"Detection confidence: "
            f"<span style='color:{health_colour};font-weight:700'>{health_label}</span> "
            f"({app_state.detection_confidence:.2f})",
            unsafe_allow_html=True,
        )

        placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
        if app_state.camera_active:
            st.image(placeholder, caption="Camera feed placeholder (capture active)")
        else:
            st.image(placeholder, caption="No camera feed")

        st.subheader("Shoe & Rules")
        st.bar_chart(pd.DataFrame([shoe_state.rank_distribution()]).T.rename(columns={0: "prob"}))
        st.caption(
            f"Cards remaining: {shoe_state.remaining}  |  "
            f"Decks remaining: {shoe_state.remaining / 52:.2f}"
        )
        st.caption(
            f"Running count: {shoe_state.running_count}  |  True count: {shoe_state.true_count:+.2f}"
        )

        if shoe_state.mid_shoe_join:
            st.warning(shoe_state.uncertainty_label)
        else:
            st.info(shoe_state.uncertainty_label)

        st.markdown(
            f"**Active rules**: {'H17' if app_state.rules.dealer_hits_soft17 else 'S17'}, "
            f"DAS {'on' if app_state.rules.double_after_split else 'off'}, "
            f"RSA {'on' if app_state.rules.resplit_aces else 'off'}, "
            f"max_splits={app_state.rules.max_splits}, "
            f"surrender={app_state.rules.surrender}, "
            f"BJ payout={app_state.rules.blackjack_payout:.2f}"
        )

        h17 = st.checkbox("Dealer hits soft 17 (H17)", value=app_state.rules.dealer_hits_soft17)
        das = st.checkbox("Double after split (DAS)", value=app_state.rules.double_after_split)
        rsa = st.checkbox("Resplit aces (RSA)", value=app_state.rules.resplit_aces)
        max_splits = st.selectbox("Max splits", options=[1, 2, 3, 4], index=app_state.rules.max_splits - 1)
        surrender_mode = st.selectbox(
            "Surrender",
            options=["none", "late", "early"],
            index=["none", "late", "early"].index(app_state.rules.surrender),
        )
        bj_payout = st.selectbox(
            "Blackjack payout",
            options=[1.0, 1.2, 1.5],
            index=[1.0, 1.2, 1.5].index(app_state.rules.blackjack_payout)
            if app_state.rules.blackjack_payout in {1.0, 1.2, 1.5}
            else 2,
        )
        app_state.update_rules(
            dealer_hits_soft17=h17,
            double_after_split=das,
            resplit_aces=rsa,
            max_splits=max_splits,
            surrender=surrender_mode,
            blackjack_payout=float(bj_payout),
        )

        deck_count = st.selectbox("Deck count", options=[1, 2, 4, 6, 8], index=[1, 2, 4, 6, 8].index(shoe_state.decks))
        if st.button("Reset Shoe", use_container_width=True):
            _rebuild_shoe_state(deck_count)
            st.rerun()

        if st.button("Join Mid-Shoe", use_container_width=True):
            shoe_state.mid_shoe_join = True

    left, center = st.columns([1, 2])

    with left:
        st.subheader("Manual Card Entry")

        player_rank = st.selectbox("Player card rank", options=RANKS)
        if st.button("Add Card", use_container_width=True):
            app_state.add_player_card(player_rank)

        dealer_rank = st.selectbox(
            "Dealer upcard",
            options=[""] + RANKS,
            index=([""] + RANKS).index(app_state.dealer_upcard)
            if app_state.dealer_upcard in RANKS
            else 0,
        )
        app_state.set_dealer_upcard(dealer_rank) if dealer_rank else None

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remove last card", use_container_width=True):
                app_state.set_player_cards(app_state.player_cards[:-1])
        with c2:
            if st.button("Clear Hand", use_container_width=True):
                app_state.clear_hand()

        app_state.is_post_split_ace = st.toggle("Post-split ace", value=app_state.is_post_split_ace)
        app_state.splits_used = int(
            st.number_input("splits_used", min_value=0, max_value=10, value=int(app_state.splits_used), step=1)
        )

        st.caption(f"Player cards: {_cards_to_text(app_state.player_cards)}")
        st.caption(f"Dealer upcard: {app_state.dealer_upcard or '(none)'}")

    with center:
        st.subheader("Decision Panel")
        hand_ready = len(app_state.player_cards) >= 2 and bool(app_state.dealer_upcard)
        if hand_ready:
            action, best_ev, evs = app_state.get_recommendation()
            color_map = {
                "hit": "#dc2626",
                "stand": "#16a34a",
                "double": "#2563eb",
                "split": "#9333ea",
                "surrender": "#ea580c",
            }
            badge_color = color_map.get(action, "#374151")
            st.markdown(
                f"<div style='font-size:2.2rem;font-weight:700;padding:0.5rem 0.9rem;"
                f"border-radius:0.6rem;display:inline-block;background:{badge_color};color:white'>"
                f"{action.upper()} ({best_ev:+.4f})</div>",
                unsafe_allow_html=True,
            )

            table_rows = format_ev_table(evs)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Enter at least 2 player cards and a dealer upcard to compute EVs.")

    st.subheader("Hand History")
    history_cols = st.columns([1, 1, 1, 3])
    with history_cols[0]:
        action_taken = st.selectbox("Action taken", options=[""] + ACTIONS)
    with history_cols[1]:
        outcome = st.selectbox("Outcome", options=OUTCOMES)
    with history_cols[2]:
        if st.button("Log Hand", use_container_width=True):
            app_state.log_hand(action_taken=action_taken or None, outcome=outcome)
    with history_cols[3]:
        csv_data = app_state.export_history_csv()
        st.download_button(
            "Export CSV",
            data=csv_data,
            file_name="blackjack_hand_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if app_state.hand_history:
        history_df = pd.DataFrame(
            [
                {
                    "hand #": r.hand_number,
                    "player cards": " ".join(r.player_cards),
                    "dealer upcard": r.dealer_upcard,
                    "recommended action": r.recommended_action,
                    "EV": r.best_ev,
                    "action taken": r.action_taken or "",
                    "outcome": r.outcome or "",
                }
                for r in app_state.hand_history
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No hands logged yet.")

    if st.button("Clear History"):
        app_state.hand_history.clear()
        app_state.hand_counter = 0
        st.rerun()

    # ------------------------------------------------------------------
    # Cut Card Panel
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("✂️ Cut Card & Penetration")

    pen_state = capture.penetration_state
    if pen_state is not None:
        col_pen1, col_pen2, col_pen3 = st.columns(3)
        with col_pen1:
            st.metric("Penetration", f"{pen_state.penetration_pct:.1f} %")
        with col_pen2:
            st.metric("Cards until reshuffle", pen_state.cards_until_reshuffle)
        with col_pen3:
            st.metric("Cards dealt", pen_state.cards_dealt)

        progress = min(1.0, pen_state.cards_dealt / max(1, pen_state.cut_card_position))
        st.progress(progress, text=f"Penetration {pen_state.penetration_pct:.1f} %")

        if pen_state.is_reshuffle_alert:
            st.warning(
                f"⚠️ Reshuffle approaching — only {pen_state.cards_until_reshuffle} cards "
                "until cut card!"
            )
    else:
        st.info(
            "No cut card data yet. "
            "Configure a `cut_card_region` in `CaptureConfig` to enable tracking."
        )

    # ------------------------------------------------------------------
    # Live Card Detection Feed
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🎥 Live Card Detection Feed")

    counter: CardCounterUI = st.session_state.card_counter

    # Running statistics
    live_cols = st.columns(4)
    with live_cols[0]:
        st.metric("Cards Observed", sum(counter.observed_cards.values()))
    with live_cols[1]:
        st.metric("Running Count", counter.running_count)
    with live_cols[2]:
        st.metric("True Count", f"{counter.true_count:+.2f}")
    with live_cols[3]:
        st.metric("Penetration", f"{counter.observation_ratio:.1%}")

    # Update card counter whenever new detections arrive
    def _on_cards_detected_for_viz(results) -> None:
        for det in results:
            st.session_state.card_counter.add_card(det.rank)
        _on_cards_observed(results)

    capture.on_cards_observed = _on_cards_detected_for_viz

    # Reset counter button
    if st.button("Reset Card Counter", use_container_width=False):
        st.session_state.card_counter.reset()
        st.rerun()

    # Live annotated video frame
    annotated_frame = capture.latest_annotated_frame
    if annotated_frame is not None:
        st.image(
            annotated_frame,
            channels="BGR",
            caption="Live detection — green boxes = detected cards",
            use_container_width=True,
        )
    else:
        placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
        st.image(
            placeholder,
            caption="No live frame yet — start capture to see detections.",
            use_container_width=False,
        )

    # Card inventory table
    st.write("**Card Inventory (observed so far)**")
    table_data = counter.get_table_display()
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info("No cards detected yet. Start capture to begin.")

    # ------------------------------------------------------------------
    # Stats Dashboard
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Statistics Dashboard")

    ml_stats: MLStatsTracker = st.session_state.ml_stats
    rolling = ml_stats.rolling_stats(last_n=500)
    shoe_stats = ml_stats.current_shoe_stats

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Hands tracked", rolling["hands"])
    with stat_cols[1]:
        st.metric("Win rate (last 500)", f"{rolling['win_rate']:.1%}")
    with stat_cols[2]:
        st.metric("EV / hand", f"{rolling['ev_per_hand']:+.4f}")
    with stat_cols[3]:
        st.metric("Action compliance", f"{rolling['action_compliance']:.1%}")

    shoe_cols = st.columns(4)
    with shoe_cols[0]:
        st.metric("This shoe – hands", shoe_stats.hands)
    with shoe_cols[1]:
        st.metric("This shoe – win rate", f"{shoe_stats.win_rate:.1%}")
    with shoe_cols[2]:
        st.metric("This shoe – EV/hand", f"{shoe_stats.ev_per_hand:+.4f}")
    with shoe_cols[3]:
        st.metric("This shoe – ROI", f"{shoe_stats.roi:.1%}")

    # Log hand to ML stats when user logs a hand
    with st.expander("Log hand to ML stats"):
        ml_hand_type = st.selectbox(
            "Hand type", options=["hard", "soft", "pair"], key="ml_hand_type"
        )
        ml_outcome = st.selectbox(
            "Outcome", options=["win", "loss", "push"], key="ml_outcome"
        )
        ml_bet = st.number_input("Bet size", min_value=0.0, value=5.0, key="ml_bet")
        ml_net = st.number_input("Net result", value=0.0, key="ml_net")
        if st.button("Record to ML stats", use_container_width=True):
            if len(app_state.player_cards) >= 2 and app_state.dealer_upcard:
                _, best_ev, _ = app_state.get_recommendation()
                rec_action = app_state.get_recommendation()[0]
                from blackjack.hand import hand_total, _normalise
                num_aces = sum(1 for c in app_state.player_cards if _normalise(c) == "A")
                num_tens = sum(1 for c in app_state.player_cards if _normalise(c) == "T")
                try:
                    du = {"A": 11, "T": 10}.get(
                        _normalise(app_state.dealer_upcard),
                        int(app_state.dealer_upcard),
                    )
                except (ValueError, TypeError):
                    du = 0
                rec = HandRecord(
                    true_count=float(shoe_state.true_count),
                    penetration=float(1.0 - shoe_state.remaining / max(1, shoe_state.decks * 52)),
                    decks_remaining=float(shoe_state.remaining / 52),
                    hand_type=ml_hand_type,
                    player_value=int(hand_total(app_state.player_cards)[0]),
                    dealer_upcard=du,
                    num_aces=num_aces,
                    num_tens=num_tens,
                    recommended=rec_action,
                    actual=rec_action,
                    outcome=ml_outcome,
                    ev=float(best_ev),
                    bet_size=float(ml_bet),
                    net_result=float(ml_net),
                )
                ml_stats.record(rec)
                st.success("Recorded.")
            else:
                st.warning("Enter player cards and dealer upcard first.")

    # Train ML model button
    if st.button("Train ML model"):
        result = ml_stats.train_model(min_samples=50)
        if result is None:
            st.warning("Not enough data to train (need at least 50 hands with win/loss outcomes).")
        else:
            st.success(
                f"Model trained on {result['n_samples']} hands. "
                f"CV accuracy: {result['accuracy']:.1%} ± {result['accuracy_std']:.1%}"
            )
            st.json(result["feature_importance"])


if __name__ == "__main__":
    main()
