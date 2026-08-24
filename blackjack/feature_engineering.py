"""Advanced Feature Engineering for Blackjack ML Models.

Provides a comprehensive feature engineering pipeline that transforms raw
hand state into a rich numerical feature vector suitable for ML training.

Features are grouped into:
- Hand features: hardness, bust probability, card counts
- Dealer features: upcard risk, bust probability, peak conditioning
- Count features: running count, true count, Z-score
- Penetration features: deck penetration, depletion rate
- Interaction features: true_count × penetration, etc.
- Temporal features: recent win rate, table heat
- Statistical features: percentile counts, SHAP-ready design matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature set definition
# ---------------------------------------------------------------------------


FEATURE_NAMES = [
    # Hand features
    "hand_total",
    "hand_is_soft",
    "hand_is_pair",
    "hand_is_blackjack",
    "hand_hardness",         # hard_total / 21
    "hand_bust_prob",        # P(bust if hit)
    "hand_cards_count",
    # Dealer features
    "dealer_upcard_value",
    "dealer_bust_prob",      # P(dealer busts)
    "dealer_is_ace",
    "dealer_is_ten",
    # Count features
    "running_count",
    "true_count",
    "true_count_zscore",     # (tc - 0) / std_tc
    "true_count_clipped",    # clip to [-6, 6]
    "count_per_deck",
    # Shoe/penetration features
    "deck_penetration",      # (starting - remaining) / starting
    "decks_remaining",
    "observation_ratio",
    # Interaction features
    "tc_x_penetration",      # true_count * deck_penetration
    "hand_x_dealer",         # hand_total * dealer_upcard_value
    "tc_x_hand",             # true_count * hand_total
    "tc_x_dealer_bust",      # true_count * dealer_bust_prob
    # Ten-rich features
    "ten_rich",              # fraction of T in remaining shoe
    "ten_rich_zscore",       # (ten_fraction - 4/13) / expected_std
    "low_card_fraction",     # fraction of 2-6 remaining
    "high_card_fraction",    # fraction of 9-A remaining
    # Temporal / table heat
    "recent_winrate",        # win rate last 20 hands
    "table_heat",            # 0-1 suspicion proxy
    "hand_number",           # hand in current shoe
    # Derived probabilities
    "p_dealer_17",
    "p_dealer_18",
    "p_dealer_19",
    "p_dealer_20",
    "p_dealer_21",
    "p_dealer_bust",
    "p_player_win",          # approximate P(player wins) given stand
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureVector:
    """A single feature vector and its label."""
    features: np.ndarray        # shape (N_FEATURES,)
    feature_names: List[str]
    hand_ev: Optional[float] = None      # target EV
    hand_outcome: Optional[int] = None   # +1 win, 0 push, -1 lose

    def to_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.feature_names, self.features)}


# ---------------------------------------------------------------------------
# Feature engineering pipeline
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Transforms raw blackjack hand state into a rich feature matrix.

    Usage
    -----
    ::
        fe = FeatureEngineer()
        fv = fe.build_features(
            hand_total=16, hand_is_soft=False, hand_is_pair=False,
            dealer_upcard='T', shoe_counts={'T': 100, ...},
            running_count=-2, decks_remaining=4.0,
            observation_ratio=0.6, recent_winrate=0.48,
        )
    """

    # Dealer bust probabilities by upcard (approximate empirical values)
    _DEALER_BUST_PROB: Dict[str, float] = {
        '2': 0.354, '3': 0.374, '4': 0.401, '5': 0.423, '6': 0.423,
        '7': 0.262, '8': 0.238, '9': 0.230, 'T': 0.212, 'A': 0.117,
    }

    # Dealer final-total distribution priors (approximate)
    _DEALER_FINAL_PROBS: Dict[str, Dict[int, float]] = {
        '6': {17: 0.165, 18: 0.106, 19: 0.106, 20: 0.106, 21: 0.094, 22: 0.423},
        'T': {17: 0.128, 18: 0.079, 19: 0.079, 20: 0.348, 21: 0.079, 22: 0.212},
        'A': {17: 0.130, 18: 0.130, 19: 0.130, 20: 0.130, 21: 0.363, 22: 0.117},
    }

    def __init__(self, true_count_std: float = 2.0) -> None:
        """
        Parameters
        ----------
        true_count_std:
            Expected standard deviation of true count for Z-score normalisation.
        """
        self.true_count_std = true_count_std

    def build_features(
        self,
        hand_total: int,
        hand_is_soft: bool,
        hand_is_pair: bool,
        dealer_upcard: str,
        shoe_counts: Dict[str, int],
        running_count: int = 0,
        decks_remaining: float = 6.0,
        observation_ratio: float = 1.0,
        recent_winrate: float = 0.48,
        table_heat: float = 0.0,
        hand_number: int = 0,
        hand_is_blackjack: bool = False,
        dealer_distribution: Optional[Dict[int, float]] = None,
    ) -> FeatureVector:
        """Build a full feature vector for one hand decision point.

        Parameters
        ----------
        hand_total: int
            Current player hand total (hard or soft).
        hand_is_soft: bool
            Whether the hand total is a soft count.
        hand_is_pair: bool
            Whether the initial two cards form a pair.
        dealer_upcard: str
            Dealer upcard rank (normalised, e.g. 'T' for 10/J/Q/K).
        shoe_counts: Dict[str, int]
            Current remaining shoe composition.
        running_count: int
            Hi-Lo running count.
        decks_remaining: float
            Estimated decks remaining in the shoe.
        observation_ratio: float
            Fraction of shoe we have tracked (0..1).
        recent_winrate: float
            Win rate over the last ~20 hands.
        table_heat: float
            Suspicion index 0..1.
        hand_number: int
            Current hand number in this shoe.
        hand_is_blackjack: bool
            Whether the player has a natural blackjack.
        dealer_distribution: Optional[Dict[int, float]]
            Precomputed dealer final-total distribution. If None, uses priors.
        """
        dealer_upcard = dealer_upcard.upper()
        total_remaining = max(1, sum(shoe_counts.values()))

        # -- Hand features --
        hand_total_norm = hand_total / 21.0
        hand_hardness = (hand_total if not hand_is_soft else hand_total - 10) / 21.0
        hand_bust_prob = self._bust_prob_if_hit(hand_total, hand_is_soft, shoe_counts, total_remaining)

        # -- Dealer features --
        dealer_val = self._card_value(dealer_upcard)
        dealer_bust_prob = self._DEALER_BUST_PROB.get(dealer_upcard, 0.23)
        dealer_is_ace = 1.0 if dealer_upcard == 'A' else 0.0
        dealer_is_ten = 1.0 if dealer_upcard == 'T' else 0.0

        # -- Count features --
        true_count = running_count / max(decks_remaining, 0.1)
        tc_zscore = true_count / max(self.true_count_std, 0.1)
        tc_clipped = max(-6.0, min(6.0, true_count))
        count_per_deck = running_count / max(decks_remaining, 0.1)

        # -- Shoe/penetration features --
        starting_cards = total_remaining / max(1.0 - observation_ratio * 0.1, 0.5)
        deck_penetration = max(0.0, 1.0 - total_remaining / max(starting_cards, 1))
        ten_count = shoe_counts.get('T', 0)
        ace_count = shoe_counts.get('A', 0)
        ten_fraction = ten_count / total_remaining
        ten_rich_zscore = (ten_fraction - 4.0 / 13.0) / 0.03
        low_card_fraction = sum(shoe_counts.get(r, 0) for r in ['2', '3', '4', '5', '6']) / total_remaining
        high_card_fraction = sum(shoe_counts.get(r, 0) for r in ['9', 'T', 'A']) / total_remaining

        # -- Interaction features --
        tc_x_pen = true_count * deck_penetration
        hand_x_dealer = (hand_total / 21.0) * (dealer_val / 11.0)
        tc_x_hand = true_count * (hand_total / 21.0)
        tc_x_dealer_bust = true_count * dealer_bust_prob

        # -- Dealer final-total distribution --
        if dealer_distribution is not None:
            dd = dealer_distribution
        else:
            dd = self._DEALER_FINAL_PROBS.get(dealer_upcard, {17: 0.15, 18: 0.10, 19: 0.10, 20: 0.15, 21: 0.08, 22: 0.23})

        p_17 = dd.get(17, 0.0)
        p_18 = dd.get(18, 0.0)
        p_19 = dd.get(19, 0.0)
        p_20 = dd.get(20, 0.0)
        p_21 = dd.get(21, 0.0)
        p_bust = dd.get(22, dealer_bust_prob)

        # P(player wins) if standing
        p_player_win = sum(
            prob for total, prob in dd.items()
            if total > 21 or total < hand_total
        )

        features = np.array([
            hand_total_norm,
            float(hand_is_soft),
            float(hand_is_pair),
            float(hand_is_blackjack),
            hand_hardness,
            hand_bust_prob,
            float(min(10, total_remaining)),  # normalised card count proxy
            dealer_val / 11.0,
            dealer_bust_prob,
            dealer_is_ace,
            dealer_is_ten,
            float(running_count) / 20.0,   # normalise
            true_count / 6.0,              # normalise
            tc_zscore,
            tc_clipped / 6.0,
            count_per_deck / 6.0,
            deck_penetration,
            decks_remaining / 8.0,
            observation_ratio,
            tc_x_pen,
            hand_x_dealer,
            tc_x_hand,
            tc_x_dealer_bust,
            ten_fraction,
            ten_rich_zscore / 5.0,         # normalise
            low_card_fraction,
            high_card_fraction,
            recent_winrate,
            table_heat,
            float(hand_number) / 300.0,    # normalise by ~shoe length
            p_17,
            p_18,
            p_19,
            p_20,
            p_21,
            p_bust,
            p_player_win,
        ], dtype=float)

        assert len(features) == N_FEATURES, f"Feature count mismatch: {len(features)} vs {N_FEATURES}"

        return FeatureVector(
            features=features,
            feature_names=list(FEATURE_NAMES),
        )

    # ------------------------------------------------------------------
    # Feature analysis
    # ------------------------------------------------------------------

    def feature_importance_from_ev(
        self,
        feature_vectors: List[FeatureVector],
        ev_labels: Sequence[float],
    ) -> Dict[str, float]:
        """Compute feature importance via Spearman correlation with EV.

        Returns dict of feature_name → |correlation| (absolute value).
        """
        X = np.stack([fv.features for fv in feature_vectors])
        y = np.asarray(ev_labels, dtype=float)

        from scipy.stats import spearmanr
        importances = {}
        for i, name in enumerate(FEATURE_NAMES):
            col = X[:, i]
            if col.std() < 1e-10:
                importances[name] = 0.0
            else:
                rho, _ = spearmanr(col, y)
                importances[name] = abs(float(rho))

        return dict(sorted(importances.items(), key=lambda x: -x[1]))

    def vif_analysis(
        self,
        feature_vectors: List[FeatureVector],
    ) -> Dict[str, float]:
        """Compute Variance Inflation Factors for multicollinearity detection.

        VIF_j = 1 / (1 - R²_j), where R²_j is the R² from regressing
        feature j on all other features.  VIF > 10 indicates strong
        multicollinearity.
        """
        X = np.stack([fv.features for fv in feature_vectors])
        n, p = X.shape
        if n < p + 2:
            return {name: float('nan') for name in FEATURE_NAMES}

        vifs = {}
        for j, name in enumerate(FEATURE_NAMES):
            y_j = X[:, j]
            X_others = np.delete(X, j, axis=1)
            # Add intercept
            X_int = np.column_stack([np.ones(n), X_others])
            try:
                # OLS R²
                beta, _, _, _ = np.linalg.lstsq(X_int, y_j, rcond=None)
                y_pred = X_int @ beta
                ss_res = np.sum((y_j - y_pred) ** 2)
                ss_tot = np.sum((y_j - y_j.mean()) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
                vifs[name] = 1.0 / max(1.0 - r2, 1e-10)
            except np.linalg.LinAlgError:
                vifs[name] = float('inf')

        return vifs

    def mutual_information_selection(
        self,
        feature_vectors: List[FeatureVector],
        labels: Sequence[float],
        n_select: int = 20,
    ) -> List[str]:
        """Select top features by mutual information with labels.

        Parameters
        ----------
        n_select:
            Number of top features to return.

        Returns
        -------
        List of feature names ordered by MI (highest first).
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
            X = np.stack([fv.features for fv in feature_vectors])
            y = np.asarray(labels, dtype=float)
            mi = mutual_info_regression(X, y, random_state=42)
            order = np.argsort(-mi)
            return [FEATURE_NAMES[i] for i in order[:n_select]]
        except ImportError:
            # Fallback: rank by Spearman correlation
            return list(self.feature_importance_from_ev(feature_vectors, labels).keys())[:n_select]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _card_value(rank: str) -> float:
        """Numeric value of a rank (A=11 for dealer upcard purposes)."""
        mapping = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10, 'A': 11,
        }
        return float(mapping.get(rank, 10))

    @staticmethod
    def _bust_prob_if_hit(
        hand_total: int,
        is_soft: bool,
        shoe_counts: Dict[str, int],
        total_remaining: int,
    ) -> float:
        """Probability that hitting results in a bust."""
        if total_remaining == 0:
            return 0.5  # fallback
        bust_count = 0
        for rank, count in shoe_counts.items():
            card_val = FeatureEngineer._card_value(rank)
            if card_val == 11:  # Ace
                new_total = hand_total + 1  # always use 1 to avoid bust
            else:
                new_total = hand_total + int(card_val)
            if is_soft and new_total > 21:
                new_total -= 10  # soft becomes hard
            if new_total > 21:
                bust_count += count
        return bust_count / total_remaining

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self, feature_vectors: List[FeatureVector]) -> pd.DataFrame:
        """Convert a list of feature vectors to a DataFrame."""
        data = np.stack([fv.features for fv in feature_vectors])
        return pd.DataFrame(data, columns=FEATURE_NAMES)
