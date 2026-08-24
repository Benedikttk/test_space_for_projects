"""Dealer Tell Detection via Statistical Analysis.

Analyses temporal patterns in dealing behaviour to detect non-random
shuffles, card handling patterns, and biased dealing sequences.

Statistical methods:
- Serial autocorrelation (Durbin-Watson test)
- Runs test (Wald-Wolfowitz) for sequence randomness
- Binomial test for individual tell significance
- Chi-squared test for shuffle uniformity
- Anomaly detection via z-scores

IMPORTANT: This module is for educational/research purposes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TellObservation:
    """A single dealer tell observation."""
    tell_type: str          # e.g. 'shuffle_speed', 'card_position', 'deal_delay'
    value: float            # numerical measurement
    hand_number: int
    timestamp: float = 0.0


@dataclass
class TellSignificance:
    """Statistical significance of a detected tell."""
    tell_type: str
    n_observations: int
    effect_size: float
    p_value: float
    is_significant: bool
    direction: str      # 'positive' | 'negative' | 'none'
    description: str


@dataclass
class ShuffleBiasResult:
    """Result of shuffle bias detection."""
    is_biased: bool
    p_value: float
    chi_squared_stat: float
    observed_frequencies: Dict[str, float]
    expected_frequencies: Dict[str, float]
    bias_description: str


# ---------------------------------------------------------------------------
# Dealer tell detector
# ---------------------------------------------------------------------------


class DealerTellDetector:
    """Detect non-random patterns in dealer behaviour.

    Tracks multiple tell types and tests each for statistical significance
    using appropriate tests (binomial, runs, correlation).

    Parameters
    ----------
    window_size:
        Number of recent observations to keep per tell type.
    min_observations:
        Minimum observations before statistical testing.
    significance_level:
        α for hypothesis tests.
    """

    def __init__(
        self,
        window_size: int = 200,
        min_observations: int = 30,
        significance_level: float = 0.05,
    ) -> None:
        self.window_size = window_size
        self.min_obs = min_observations
        self.alpha = significance_level

        # Circular buffers for each tell type
        self._observations: Dict[str, Deque[TellObservation]] = {}
        # Card sequence (ranks as numeric values)
        self._card_sequence: Deque[int] = deque(maxlen=window_size)
        # Outcome sequence (+1 win, -1 lose, 0 push)
        self._outcome_sequence: Deque[int] = deque(maxlen=window_size)
        # Tell → outcome pairs
        self._tell_outcomes: Dict[str, List[Tuple[float, int]]] = {}

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def observe_tell(self, obs: TellObservation) -> None:
        """Record a dealer tell observation."""
        if obs.tell_type not in self._observations:
            self._observations[obs.tell_type] = deque(maxlen=self.window_size)
        self._observations[obs.tell_type].append(obs)

    def observe_card(self, rank: str) -> None:
        """Record a dealt card rank (for sequence analysis)."""
        _rank_to_val = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, 'T': 10, 'A': 11,
        }
        self._card_sequence.append(_rank_to_val.get(rank, 10))

    def observe_outcome(self, outcome: int) -> None:
        """Record a hand outcome (+1, 0, -1)."""
        self._outcome_sequence.append(outcome)

    def record_tell_outcome(
        self, tell_type: str, tell_value: float, outcome: int
    ) -> None:
        """Record a (tell, outcome) pair for correlation analysis."""
        if tell_type not in self._tell_outcomes:
            self._tell_outcomes[tell_type] = []
        self._tell_outcomes[tell_type].append((tell_value, outcome))

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def runs_test(self, sequence: Optional[Sequence[float]] = None) -> Dict[str, object]:
        """Wald-Wolfowitz runs test for sequence randomness.

        Tests whether a binary sequence (above/below median) is random.
        H0: sequence is random.

        Parameters
        ----------
        sequence:
            If None, uses internal card sequence.
        """
        if sequence is None:
            seq = list(self._card_sequence)
        else:
            seq = list(sequence)

        if len(seq) < self.min_obs:
            return {"error": f"Need ≥ {self.min_obs} observations", "n": len(seq)}

        arr = np.array(seq, dtype=float)
        median = float(np.median(arr))
        binary = (arr >= median).astype(int)

        # Count runs
        n1 = int(binary.sum())
        n2 = len(binary) - n1
        n = n1 + n2

        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                runs += 1

        # Expected runs and variance under H0
        mu_r = (2 * n1 * n2) / n + 1
        sigma_r_sq = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
        if sigma_r_sq <= 0:
            return {"error": "Zero variance in runs test", "n": n}

        z = (runs - mu_r) / math.sqrt(sigma_r_sq)
        p_value = float(2 * stats.norm.sf(abs(z)))  # two-sided

        return {
            "test": "wald_wolfowitz_runs",
            "n_runs": runs,
            "expected_runs": mu_r,
            "z_statistic": z,
            "p_value": p_value,
            "is_random": p_value >= self.alpha,
            "n_observations": n,
            "interpretation": (
                "Sequence appears random (H0 not rejected)"
                if p_value >= self.alpha
                else f"Non-random sequence detected (p={p_value:.4f})"
            ),
        }

    def autocorrelation_test(
        self, lag: int = 1, sequence: Optional[Sequence[float]] = None
    ) -> Dict[str, object]:
        """Test for autocorrelation in the card or tell sequence.

        Uses the Durbin-Watson inspired test: checks whether lag-1
        autocorrelation is significantly different from 0.

        Parameters
        ----------
        lag:
            Autocorrelation lag to test.
        """
        if sequence is None:
            seq = list(self._card_sequence)
        else:
            seq = list(sequence)

        if len(seq) < self.min_obs:
            return {"error": f"Need ≥ {self.min_obs} observations"}

        arr = np.array(seq, dtype=float)
        n = len(arr)

        # Pearson correlation with lag
        arr_mean = arr.mean()
        numerator = float(np.sum((arr[lag:] - arr_mean) * (arr[:-lag] - arr_mean)))
        denominator = float(np.sum((arr - arr_mean) ** 2))

        if denominator == 0:
            return {"error": "Zero variance"}

        rho = numerator / denominator

        # Approximate z-test for correlation: z = rho * sqrt(n - lag)
        z = rho * math.sqrt(n - lag)
        p_value = float(2 * stats.norm.sf(abs(z)))

        return {
            "test": "autocorrelation",
            "lag": lag,
            "rho": rho,
            "z_statistic": z,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "interpretation": (
                f"Lag-{lag} autocorrelation = {rho:.4f} "
                + ("(SIGNIFICANT)" if p_value < self.alpha else "(not significant)")
            ),
        }

    def tell_outcome_correlation(
        self, tell_type: str
    ) -> Optional[TellSignificance]:
        """Test whether a tell type predicts hand outcome.

        Uses point-biserial correlation and a binomial test.
        """
        pairs = self._tell_outcomes.get(tell_type, [])
        if len(pairs) < self.min_obs:
            return None

        tells = np.array([p[0] for p in pairs])
        outcomes = np.array([p[1] for p in pairs])

        # Point-biserial correlation (tell value vs outcome)
        rho, p_value = stats.pearsonr(tells, outcomes)

        # Effect size
        effect = abs(float(rho))

        # Wins when tell is "positive" (above median)
        median_tell = float(np.median(tells))
        high_tell_mask = tells >= median_tell
        low_tell_mask = ~high_tell_mask

        wins_high = int(np.sum(outcomes[high_tell_mask] > 0))
        n_high = int(high_tell_mask.sum())
        wins_low = int(np.sum(outcomes[low_tell_mask] > 0))
        n_low = int(low_tell_mask.sum())

        # Binomial test: does high-tell give higher win rate?
        p_win_high = wins_high / max(n_high, 1)
        bin_result = stats.binomtest(wins_high, n_high, p=0.48)

        direction = "none"
        if bin_result.pvalue < self.alpha:
            direction = "positive" if p_win_high > 0.48 else "negative"

        return TellSignificance(
            tell_type=tell_type,
            n_observations=len(pairs),
            effect_size=effect,
            p_value=float(p_value),
            is_significant=p_value < self.alpha,
            direction=direction,
            description=(
                f"Tell '{tell_type}': r={rho:.3f}, p={p_value:.4f}. "
                f"Win rate when tell active: {p_win_high:.1%}."
            ),
        )

    def shuffle_bias_test(
        self, card_sequence: Optional[Sequence[str]] = None
    ) -> ShuffleBiasResult:
        """Chi-squared test for shuffle uniformity (biased shuffle detection).

        A perfect shuffle should produce a uniform distribution of ranks.
        A biased shuffle (e.g. clumped) will show rank clusters.
        """
        if card_sequence is not None:
            seq = list(card_sequence)
        else:
            seq = [str(v) for v in self._card_sequence]

        # Expected frequencies: uniform over 10 ranks (T = 4x others)
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']
        expected_prop = {
            '2': 4/52, '3': 4/52, '4': 4/52, '5': 4/52, '6': 4/52,
            '7': 4/52, '8': 4/52, '9': 4/52, 'T': 16/52, 'A': 4/52,
        }

        n = len(seq)
        if n < self.min_obs:
            return ShuffleBiasResult(
                is_biased=False,
                p_value=1.0,
                chi_squared_stat=0.0,
                observed_frequencies={},
                expected_frequencies=expected_prop,
                bias_description=f"Insufficient data (need ≥ {self.min_obs})",
            )

        obs_counts = {r: seq.count(r) for r in ranks}
        exp_counts = {r: expected_prop[r] * n for r in ranks}

        obs_arr = np.array([obs_counts[r] for r in ranks])
        exp_arr = np.array([exp_counts[r] for r in ranks])

        chi2, p_value = stats.chisquare(obs_arr, f_exp=exp_arr)

        obs_freq = {r: obs_counts[r] / n for r in ranks}

        return ShuffleBiasResult(
            is_biased=p_value < self.alpha,
            p_value=float(p_value),
            chi_squared_stat=float(chi2),
            observed_frequencies=obs_freq,
            expected_frequencies=expected_prop,
            bias_description=(
                f"Shuffle appears biased (χ²={chi2:.2f}, p={p_value:.4f})"
                if p_value < self.alpha
                else "Shuffle appears uniform (no significant bias detected)"
            ),
        )

    def all_significant_tells(self) -> List[TellSignificance]:
        """Return all tell types with significant outcome correlation."""
        results = []
        for tell_type in self._tell_outcomes:
            sig = self.tell_outcome_correlation(tell_type)
            if sig is not None and sig.is_significant:
                results.append(sig)
        return sorted(results, key=lambda x: x.p_value)

    def anomaly_scores(self) -> Dict[str, float]:
        """Z-score anomaly detection for each tell type.

        Returns dict of tell_type → z_score (how unusual the latest value is).
        """
        scores = {}
        for tell_type, obs_deque in self._observations.items():
            obs = list(obs_deque)
            if len(obs) < 5:
                continue
            values = [o.value for o in obs]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if std == 0:
                continue
            latest = values[-1]
            scores[tell_type] = abs((latest - mean) / std)
        return scores
