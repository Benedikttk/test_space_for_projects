"""Bayesian Shoe Composition Modelling.

Implements Dirichlet-Multinomial conjugate Bayesian inference for the
unobserved portion of the shoe.  As cards are dealt and observed, the
posterior distribution over remaining card probabilities is updated in
closed-form (no MCMC required).

Background
----------
Prior:    θ ~ Dirichlet(α)
Likelihood: x | θ ~ Multinomial(n, θ)
Posterior:  θ | x ~ Dirichlet(α + x)

This conjugacy means every observation is a single vector addition —
O(K) per card, extremely fast.

The posterior predictive probability for the *next* card being rank k is:
    P(x_new = k | data) = (α_k + n_k) / (Σ_j α_j + N)

which is just the posterior mean — exactly what the EV engine should use
instead of the raw count-based estimate when operating in mid-shoe-join mode
or with low observation ratios.

References
----------
- Gelman et al. (2013): "Bayesian Data Analysis", 3rd ed., Ch.3
- Jaynes (2003): "Probability Theory: The Logic of Science", Ch.18
- Ferguson (1973): "A Bayesian Analysis of Some Nonparametric Problems"
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, digamma


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS: List[str] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']
K = len(RANKS)  # 10 distinct ranks

# Cards per rank per deck (T = T/J/Q/K = 16 per deck)
_PER_DECK: Dict[str, int] = {
    '2': 4, '3': 4, '4': 4, '5': 4, '6': 4,
    '7': 4, '8': 4, '9': 4, 'T': 16, 'A': 4,
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PosteriorResult:
    """Dirichlet posterior over shoe card probabilities."""
    alpha_posterior: np.ndarray    # shape (K,): posterior Dirichlet params
    posterior_mean: np.ndarray     # shape (K,): E[θ|data]
    posterior_variance: np.ndarray  # shape (K,): Var[θ_k|data]
    credible_intervals_95: np.ndarray  # shape (K, 2): lower/upper 95% CI
    observation_count: int
    rank_labels: List[str]

    def prob(self, rank: str) -> float:
        """Posterior predictive probability for rank."""
        idx = self.rank_labels.index(rank)
        return float(self.posterior_mean[idx])

    def credible_interval(self, rank: str) -> Tuple[float, float]:
        """95% credible interval for P(rank)."""
        idx = self.rank_labels.index(rank)
        return (float(self.credible_intervals_95[idx, 0]),
                float(self.credible_intervals_95[idx, 1]))

    def uncertainty(self, rank: str) -> float:
        """Standard deviation of posterior for rank."""
        idx = self.rank_labels.index(rank)
        return float(math.sqrt(self.posterior_variance[idx]))

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return summary dict for all ranks."""
        return {
            r: {
                "mean": float(self.posterior_mean[i]),
                "std": float(math.sqrt(self.posterior_variance[i])),
                "ci_lower": float(self.credible_intervals_95[i, 0]),
                "ci_upper": float(self.credible_intervals_95[i, 1]),
            }
            for i, r in enumerate(self.rank_labels)
        }


@dataclass
class ConvergenceAnalysis:
    """Posterior convergence analysis results."""
    observation_ratios: List[float]
    kl_divergences: List[float]  # KL(posterior || prior)
    l1_errors_to_truth: List[float]  # |posterior_mean - true_probs|
    converged_at_ratio: Optional[float]  # ratio where KL < 0.01
    convergence_rate: Optional[float]   # fitted exponent b from KL ~ a/N^b


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BayesianShoeModeling:
    """Dirichlet-Multinomial Bayesian inference for shoe card composition.

    Tracks:
    - Prior: Dirichlet(α) — by default, the Jeffreys prior α_k = 1/2
      which is maximum-entropy for the constrained simplex.
    - Posterior: Dirichlet(α + observed_counts) — updated in O(K) per card.
    - Posterior predictive: E[θ_k|data] used in place of count_k/total.

    Parameters
    ----------
    n_decks:
        Number of decks in the shoe.
    prior_strength:
        Concentration parameter for the prior.  Default 0.5 (Jeffreys).
        Use 1.0 for Laplace (uniform Dirichlet), larger for stronger prior.
    use_informative_prior:
        If True, initialise the prior proportional to the known deck
        composition (α_k ∝ cards_per_rank_per_deck * n_decks).
        This encodes knowledge of the standard shoe composition as a
        "virtual observation count".
    """

    def __init__(
        self,
        n_decks: int = 6,
        prior_strength: float = 0.5,
        use_informative_prior: bool = True,
    ) -> None:
        if n_decks < 1:
            raise ValueError("n_decks must be ≥ 1")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be > 0")

        self.n_decks = n_decks
        self.prior_strength = prior_strength

        # Build prior α
        if use_informative_prior:
            total_per_deck = sum(_PER_DECK.values())  # 52
            self._alpha_prior = np.array(
                [_PER_DECK[r] * prior_strength / total_per_deck for r in RANKS],
                dtype=float,
            )
        else:
            self._alpha_prior = np.full(K, prior_strength, dtype=float)

        # Observed counts (starts at zero)
        self._observed: np.ndarray = np.zeros(K, dtype=float)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def observe(self, rank: str, count: int = 1) -> None:
        """Record observation of `count` cards of `rank`."""
        if rank not in RANKS:
            raise ValueError(f"Unknown rank {rank!r}. Must be one of {RANKS}")
        if count < 0:
            raise ValueError("count must be non-negative")
        idx = RANKS.index(rank)
        self._observed[idx] += count

    def observe_sequence(self, ranks: Sequence[str]) -> None:
        """Observe a sequence of cards."""
        for r in ranks:
            self.observe(r)

    def reset(self) -> None:
        """Reset observations (new shoe)."""
        self._observed[:] = 0.0

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------

    @property
    def _alpha_posterior(self) -> np.ndarray:
        """Dirichlet parameters for the REMAINING shoe composition.

        As cards are dealt (observed), the expected remaining counts decrease.
        We model: α_posterior_k = max(floor, initial_k - observed_k + prior_k)

        This gives the desired blackjack property:
        - Observing many T's → fewer T's remain → lower T posterior probability
        - Uncertainty (via prior) prevents the posterior from collapsing to 0
        """
        initial = np.array([_PER_DECK[r] * self.n_decks for r in RANKS], dtype=float)
        remaining = initial - self._observed
        # Add prior as a Bayesian smoothing term (prevents zero counts)
        smoothed = remaining + self._alpha_prior
        # Clamp to minimum of the prior to avoid negative values
        return np.maximum(self._alpha_prior * 0.5, smoothed)

    def posterior(self) -> PosteriorResult:
        """Compute the full Dirichlet posterior.

        Returns
        -------
        PosteriorResult with:
        - alpha_posterior: Dirichlet concentration parameters
        - posterior_mean: E[θ_k|data] = α_k / Σα_j
        - posterior_variance: Var[θ_k|data]
        - 95% credible intervals (via marginal Beta distribution)
        """
        alpha = self._alpha_posterior
        alpha_0 = alpha.sum()

        mean = alpha / alpha_0
        variance = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1))

        # Marginal Beta credible intervals
        ci = np.zeros((K, 2))
        from scipy.stats import beta as beta_dist
        for i in range(K):
            a_i = float(alpha[i])
            b_i = float(alpha_0 - alpha[i])
            ci[i, 0] = float(beta_dist.ppf(0.025, a_i, b_i))
            ci[i, 1] = float(beta_dist.ppf(0.975, a_i, b_i))

        return PosteriorResult(
            alpha_posterior=alpha.copy(),
            posterior_mean=mean,
            posterior_variance=variance,
            credible_intervals_95=ci,
            observation_count=int(self._observed.sum()),
            rank_labels=list(RANKS),
        )

    def predictive_probabilities(self) -> Dict[str, float]:
        """Return posterior predictive P(next card = k | observations).

        This is the mean of the Dirichlet posterior and is the optimal
        Bayesian estimate for the EV engine.
        """
        alpha = self._alpha_posterior
        alpha_0 = alpha.sum()
        return {RANKS[i]: float(alpha[i] / alpha_0) for i in range(K)}

    # ------------------------------------------------------------------
    # Comparison with observed shoe (calibration)
    # ------------------------------------------------------------------

    def calibration_vs_true_shoe(
        self,
        true_remaining: Dict[str, int],
    ) -> Dict[str, float]:
        """Compare posterior predictive to the true remaining shoe composition.

        Parameters
        ----------
        true_remaining:
            Dict of rank → count in the true shoe (ground truth).

        Returns
        -------
        Dict with:
        - 'l1_error': L1 distance between posterioror mean and true probs
        - 'kl_divergence': KL(true || posterior)
        - 'max_rank_error': max |posterior_k - true_k| over ranks
        - 'well_calibrated': True if l1_error < 0.05
        """
        total = sum(true_remaining.values())
        if total == 0:
            return {"error": "Empty true shoe"}

        true_probs = np.array(
            [true_remaining.get(r, 0) / total for r in RANKS], dtype=float
        )
        pred_probs = np.array(
            [self.predictive_probabilities()[r] for r in RANKS], dtype=float
        )

        l1 = float(np.sum(np.abs(pred_probs - true_probs)))
        # KL(true || pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            kl = float(np.sum(
                np.where(true_probs > 0, true_probs * np.log(true_probs / np.clip(pred_probs, 1e-10, None)), 0.0)
            ))
        max_err = float(np.max(np.abs(pred_probs - true_probs)))

        return {
            "l1_error": l1,
            "kl_divergence": kl,
            "max_rank_error": max_err,
            "well_calibrated": l1 < 0.05,
        }

    # ------------------------------------------------------------------
    # Convergence rate analysis
    # ------------------------------------------------------------------

    def convergence_analysis(
        self,
        n_decks: int = 6,
        max_cards: int = 200,
        step: int = 10,
        rng_seed: int = 42,
    ) -> ConvergenceAnalysis:
        """Simulate posterior convergence vs. observation count.

        Generates a shoe, observes cards sequentially, tracks KL divergence
        and L1 error to the true shoe at each step.

        Returns
        -------
        ConvergenceAnalysis with lists of (ratio, KL, L1) triples.
        """
        rng = np.random.default_rng(rng_seed)

        # True shoe composition
        true_counts = np.array([_PER_DECK[r] * n_decks for r in RANKS], dtype=float)
        total_cards = int(true_counts.sum())

        # Draw cards from shoe (without replacement)
        deck = []
        for i, r in enumerate(RANKS):
            deck.extend([r] * int(true_counts[i]))
        rng.shuffle(deck)

        saved = (self._alpha_prior.copy(), self._observed.copy())
        self._observed = np.zeros(K, dtype=float)

        observation_ratios = []
        kl_divergences = []
        l1_errors = []

        for n in range(0, min(max_cards, total_cards) + 1, step):
            if n > 0:
                for i in range(step):
                    idx = n - step + i
                    if idx < len(deck):
                        self.observe(deck[idx])

            # True remaining shoe
            remaining = {RANKS[i]: float(true_counts[i]) for i in range(len(RANKS))}
            for r in deck[:n]:
                remaining[r] = max(0.0, remaining.get(r, 0) - 1)
            true_total = sum(remaining.values())
            if true_total == 0:
                break
            true_probs = np.array([remaining.get(r, 0) / true_total for r in RANKS])
            pred = np.array([self.predictive_probabilities()[r] for r in RANKS])

            kl = float(np.sum(
                np.where(true_probs > 0,
                         true_probs * np.log(np.clip(true_probs, 1e-10, None) / np.clip(pred, 1e-10, None)),
                         0.0)
            ))
            l1 = float(np.sum(np.abs(pred - true_probs)))

            observation_ratios.append(n / total_cards)
            kl_divergences.append(kl)
            l1_errors.append(l1)

        # Restore state
        self._alpha_prior, self._observed = saved

        # Find convergence point (KL < 0.01)
        converged_at = None
        for r, kl in zip(observation_ratios, kl_divergences):
            if kl < 0.01 and r > 0:
                converged_at = r
                break

        # Fit convergence rate: KL ~ a / N^b
        convergence_rate = None
        if len(kl_divergences) > 3:
            ns = np.array([max(r, 1e-6) for r in observation_ratios[1:]])
            kls = np.array(kl_divergences[1:])
            valid = kls > 0
            if valid.sum() > 2:
                from scipy import stats as sp
                slope, _, _, _, _ = sp.linregress(np.log(ns[valid]), np.log(kls[valid]))
                convergence_rate = float(slope)  # should be ~ -1.0

        return ConvergenceAnalysis(
            observation_ratios=observation_ratios,
            kl_divergences=kl_divergences,
            l1_errors_to_truth=l1_errors,
            converged_at_ratio=converged_at,
            convergence_rate=convergence_rate,
        )

    # ------------------------------------------------------------------
    # Comparison with EV engine
    # ------------------------------------------------------------------

    def ev_adjustment(
        self,
        count_based_probs: Dict[str, float],
    ) -> Dict[str, float]:
        """Return posterior-adjusted probabilities (blended with prior).

        When observation_ratio is low, blends towards the prior.
        When observation_ratio is high, returns count-based probs.
        Used to smooth EV calculations early in tracking.

        Parameters
        ----------
        count_based_probs:
            Current count-based probabilities from the Shoe.

        Returns
        -------
        Blended probabilities: (1-w) * prior_mean + w * count_based
        where w = min(1, observation_count / 100).
        """
        n = int(self._observed.sum())
        w = min(1.0, n / 100.0)

        prior_mean = self._alpha_prior / self._alpha_prior.sum()
        result = {}
        for i, r in enumerate(RANKS):
            prior_p = float(prior_mean[i])
            count_p = count_based_probs.get(r, prior_p)
            result[r] = (1.0 - w) * prior_p + w * count_p
        return result
