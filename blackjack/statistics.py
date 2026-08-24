"""Statistical Validation Framework for Blackjack Advantage Play.

Provides publication-grade statistical methods for validating edge,
quantifying uncertainty, and proving statistical significance.

References
----------
- Wilson (1927): "Probable inference, the law of succession, and statistical inference"
- Clopper & Pearson (1934): "The use of confidence or fiducial limits"
- Cohen (1988): "Statistical Power Analysis for the Behavioral Sciences"
- Kolmogorov (1933), Smirnov (1948): goodness-of-fit tests
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.special import betainc, betaln


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInterval:
    """A confidence interval result."""
    lower: float
    upper: float
    point_estimate: float
    confidence: float
    method: str

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:
        return (
            f"CI({self.method}, {self.confidence:.0%}): "
            f"[{self.lower:.6f}, {self.upper:.6f}] "
            f"point={self.point_estimate:.6f}"
        )


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    reject_h0: bool
    effect_size: Optional[float] = None
    power: Optional[float] = None
    notes: str = ""

    def __repr__(self) -> str:
        decision = "REJECT H0" if self.reject_h0 else "fail to reject H0"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, {decision} (α={self.alpha})"
        )


@dataclass
class PowerAnalysisResult:
    """Result of a statistical power analysis."""
    required_hands: int
    achieved_power: float
    effect_size: float
    alpha: float
    description: str


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo simulation."""
    n_simulations: int
    mean_ev: float
    std_ev: float
    ci_95: Tuple[float, float]
    win_rate: float
    expected_win_rate: float
    convergence_hands: int  # hands until 95% CI width < 0.01
    raw_evs: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main statistical validator
# ---------------------------------------------------------------------------


class StatisticalValidator:
    """Publication-grade statistical validation for blackjack advantage play.

    Implements:
    - Confidence intervals: Wilson score, Clopper-Pearson exact binomial
    - Hypothesis tests: binomial, McNemar's, chi-squared goodness-of-fit
    - Monte Carlo simulation framework (configurable up to 1 M hands)
    - Power analysis: hands until edge is statistically significant
    - Effect size calculations: Cohen's h, Cramér's V
    - Goodness-of-fit: KS test, Anderson-Darling
    - Calibration analysis: predicted vs. realized outcomes
    - Convergence analysis: observation_ratio → accuracy mapping
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha:
            Default significance level for all hypothesis tests.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def wilson_ci(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Wilson score confidence interval for a proportion.

        Outperforms normal approximation for extreme proportions and small n.
        See Wilson (1927).

        Parameters
        ----------
        successes:
            Number of wins (or successes).
        trials:
            Total number of trials.
        confidence:
            Desired confidence level in (0, 1).

        Returns
        -------
        ConfidenceInterval with method='wilson'
        """
        if trials <= 0:
            raise ValueError("trials must be > 0")
        if not 0 <= successes <= trials:
            raise ValueError("successes must be in [0, trials]")

        alpha = 1.0 - confidence
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        p_hat = successes / trials
        n = trials

        centre = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
        margin = (z / (1 + z * z / n)) * math.sqrt(
            p_hat * (1 - p_hat) / n + z * z / (4 * n * n)
        )

        return ConfidenceInterval(
            lower=max(0.0, centre - margin),
            upper=min(1.0, centre + margin),
            point_estimate=p_hat,
            confidence=confidence,
            method="wilson",
        )

    def clopper_pearson_ci(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Clopper-Pearson exact binomial confidence interval.

        The "exact" interval; conservative but guaranteed to contain the true
        proportion with probability ≥ confidence.  See Clopper & Pearson (1934).

        Parameters
        ----------
        successes, trials, confidence:
            Same as `wilson_ci`.
        """
        if trials <= 0:
            raise ValueError("trials must be > 0")
        if not 0 <= successes <= trials:
            raise ValueError("successes must be in [0, trials]")

        alpha = 1.0 - confidence
        k, n = successes, trials
        p_hat = k / n

        # Lower bound: Beta(α/2; k, n-k+1)
        lower = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
        # Upper bound: Beta(1 - α/2; k+1, n-k)
        upper = 1.0 if k == n else stats.beta.ppf(1.0 - alpha / 2, k + 1, n - k)

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            point_estimate=p_hat,
            confidence=confidence,
            method="clopper_pearson",
        )

    def ev_confidence_interval(
        self,
        evs: Sequence[float],
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Bootstrap-t confidence interval for mean EV.

        Parameters
        ----------
        evs:
            Array of per-hand EV outcomes.
        confidence:
            Desired confidence level.
        """
        arr = np.asarray(evs, dtype=float)
        n = len(arr)
        if n < 2:
            raise ValueError("Need at least 2 observations")

        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(n))
        alpha = 1.0 - confidence
        t_crit = stats.t.ppf(1.0 - alpha / 2, df=n - 1)
        margin = t_crit * se

        return ConfidenceInterval(
            lower=mean - margin,
            upper=mean + margin,
            point_estimate=mean,
            confidence=confidence,
            method="t_interval",
        )

    # ------------------------------------------------------------------
    # Hypothesis tests
    # ------------------------------------------------------------------

    def binomial_test(
        self,
        successes: int,
        trials: int,
        p_null: float = 0.5,
        alternative: str = "two-sided",
    ) -> HypothesisTestResult:
        """Exact binomial test: H0: p = p_null.

        Parameters
        ----------
        successes:
            Observed wins.
        trials:
            Total hands.
        p_null:
            Null hypothesis win probability.
        alternative:
            'two-sided', 'greater', or 'less'.
        """
        result = stats.binomtest(successes, trials, p=p_null, alternative=alternative)
        p_hat = successes / max(trials, 1)

        # Effect size: Cohen's h
        h = self.cohens_h(p_hat, p_null)

        return HypothesisTestResult(
            test_name="binomial_exact",
            statistic=float(successes),
            p_value=float(result.pvalue),
            alpha=self.alpha,
            reject_h0=result.pvalue < self.alpha,
            effect_size=h,
            notes=f"H0: p={p_null}, observed p={p_hat:.4f}",
        )

    def mcnemar_test(
        self,
        b: int,
        c: int,
    ) -> HypothesisTestResult:
        """McNemar's test for paired binary outcomes.

        Used to compare two strategies on the same hands:
        b = strategy A correct, B incorrect
        c = strategy A incorrect, B correct

        H0: p(B correct | A incorrect) = p(A correct | B incorrect)
        """
        if b + c == 0:
            return HypothesisTestResult(
                test_name="mcnemar",
                statistic=0.0,
                p_value=1.0,
                alpha=self.alpha,
                reject_h0=False,
                notes="No discordant pairs",
            )
        statistic = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
        p_value = float(stats.chi2.sf(statistic, df=1))

        return HypothesisTestResult(
            test_name="mcnemar",
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha,
            reject_h0=p_value < self.alpha,
            notes=f"discordant pairs b={b}, c={c}",
        )

    def chi_squared_test(
        self,
        observed: Sequence[float],
        expected: Optional[Sequence[float]] = None,
    ) -> HypothesisTestResult:
        """Chi-squared goodness-of-fit test.

        Parameters
        ----------
        observed:
            Observed frequencies.
        expected:
            Expected frequencies. If None, uniform distribution assumed.
        """
        obs = np.asarray(observed, dtype=float)
        if expected is None:
            exp = np.full_like(obs, obs.sum() / len(obs))
        else:
            exp = np.asarray(expected, dtype=float)

        if len(obs) != len(exp):
            raise ValueError("observed and expected must have same length")

        statistic, p_value = stats.chisquare(obs, f_exp=exp)
        cramers_v = self.cramers_v(obs, exp)

        return HypothesisTestResult(
            test_name="chi_squared_goodness_of_fit",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_h0=p_value < self.alpha,
            effect_size=cramers_v,
            notes=f"df={len(obs)-1}",
        )

    # ------------------------------------------------------------------
    # Effect sizes
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_h(p1: float, p2: float) -> float:
        """Cohen's h effect size for the difference between two proportions.

        h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        |h| ≈ 0.2 small, 0.5 medium, 0.8 large.
        """
        return 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p1)))) - \
               2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))

    @staticmethod
    def cramers_v(observed: np.ndarray, expected: np.ndarray) -> float:
        """Cramér's V effect size for chi-squared test.

        V = sqrt(chi2 / (n * (k - 1)))
        where k = number of categories, n = total count.
        """
        obs = np.asarray(observed, dtype=float)
        exp = np.asarray(expected, dtype=float)
        n = obs.sum()
        if n == 0:
            return 0.0
        k = len(obs)
        chi2 = float(np.sum((obs - exp) ** 2 / np.where(exp > 0, exp, 1.0)))
        return math.sqrt(chi2 / (n * max(k - 1, 1)))

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def power_analysis_binomial(
        self,
        true_win_rate: float,
        null_win_rate: float = 0.49,
        desired_power: float = 0.80,
        alpha: Optional[float] = None,
    ) -> PowerAnalysisResult:
        """How many hands until the edge is statistically significant?

        Uses the normal approximation to the binomial.

        Parameters
        ----------
        true_win_rate:
            The actual win rate under the alternative hypothesis.
        null_win_rate:
            The win rate under H0 (usually ≤ 0.5 — the break-even point).
        desired_power:
            Target statistical power (1 - β).
        alpha:
            Significance level. Uses instance default if None.
        """
        alpha = alpha or self.alpha
        if not 0 < true_win_rate < 1:
            raise ValueError("true_win_rate must be in (0, 1)")
        if true_win_rate <= null_win_rate:
            raise ValueError("true_win_rate must exceed null_win_rate for one-sided test")

        z_alpha = stats.norm.ppf(1.0 - alpha)
        z_beta = stats.norm.ppf(desired_power)

        p1 = true_win_rate
        p0 = null_win_rate

        # Formula: n = ((z_α√p0(1-p0) + z_β√p1(1-p1)) / (p1 - p0))²
        numerator = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
        n = math.ceil((numerator / (p1 - p0)) ** 2)

        h = self.cohens_h(p1, p0)

        return PowerAnalysisResult(
            required_hands=n,
            achieved_power=desired_power,
            effect_size=abs(h),
            alpha=alpha,
            description=(
                f"Need {n:,} hands to detect p={p1:.4f} vs p0={p0:.4f} "
                f"with {desired_power:.0%} power (α={alpha})"
            ),
        )

    def hands_for_significant_ev(
        self,
        ev_per_hand: float,
        std_per_hand: float = 1.1,
        desired_power: float = 0.80,
        alpha: Optional[float] = None,
    ) -> PowerAnalysisResult:
        """How many hands to prove EV > 0 with given power?

        Uses one-sample t-test power formula.

        Parameters
        ----------
        ev_per_hand:
            Expected value per hand (must be > 0 for positive edge).
        std_per_hand:
            Standard deviation of per-hand outcome (~1.1 for BJ).
        desired_power:
            Target power.
        """
        alpha = alpha or self.alpha
        if ev_per_hand <= 0:
            raise ValueError("ev_per_hand must be > 0")

        z_alpha = stats.norm.ppf(1.0 - alpha)
        z_beta = stats.norm.ppf(desired_power)
        cohen_d = ev_per_hand / std_per_hand

        n = math.ceil(((z_alpha + z_beta) / cohen_d) ** 2)

        return PowerAnalysisResult(
            required_hands=n,
            achieved_power=desired_power,
            effect_size=cohen_d,
            alpha=alpha,
            description=(
                f"Need {n:,} hands to prove EV={ev_per_hand:.4f} > 0 "
                f"with {desired_power:.0%} power (d={cohen_d:.3f})"
            ),
        )

    # ------------------------------------------------------------------
    # Goodness-of-fit tests
    # ------------------------------------------------------------------

    def ks_test(
        self,
        sample: Sequence[float],
        cdf: str = "norm",
        cdf_args: tuple = (),
    ) -> HypothesisTestResult:
        """Kolmogorov-Smirnov one-sample goodness-of-fit test.

        Parameters
        ----------
        sample:
            Observed data.
        cdf:
            Name of the scipy distribution to test against.
        cdf_args:
            Arguments for the distribution (loc, scale, ...).
        """
        arr = np.asarray(sample, dtype=float)
        if cdf == "norm":
            loc, scale = float(np.mean(arr)), float(np.std(arr, ddof=1))
            dist = stats.norm(loc=loc, scale=scale)
            result = stats.kstest(arr, dist.cdf)
        else:
            result = stats.kstest(arr, cdf, args=cdf_args)

        return HypothesisTestResult(
            test_name="kolmogorov_smirnov",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            alpha=self.alpha,
            reject_h0=result.pvalue < self.alpha,
            notes=f"testing against {cdf}, n={len(arr)}",
        )

    def anderson_darling_test(
        self,
        sample: Sequence[float],
        distribution: str = "norm",
    ) -> HypothesisTestResult:
        """Anderson-Darling test (more powerful than KS in the tails).

        Parameters
        ----------
        sample:
            Observed data.
        distribution:
            Distribution to test ('norm', 'expon', 'logistic', ...).
        """
        arr = np.asarray(sample, dtype=float)
        result = stats.anderson(arr, dist=distribution)

        # Use 5% significance level
        sig_levels = list(result.significance_level)
        critical_values = list(result.critical_values)

        # Find the p-value index closest to our alpha
        alpha_pct = self.alpha * 100
        idx = min(range(len(sig_levels)), key=lambda i: abs(sig_levels[i] - alpha_pct))
        critical = critical_values[idx]
        reject = result.statistic > critical

        # Anderson-Darling doesn't give an exact p-value; approximate
        p_approx = float(sig_levels[idx]) / 100.0

        return HypothesisTestResult(
            test_name="anderson_darling",
            statistic=float(result.statistic),
            p_value=p_approx if reject else 1.0 - p_approx,
            alpha=self.alpha,
            reject_h0=reject,
            notes=f"critical_value={critical:.4f} at {sig_levels[idx]}%",
        )

    # ------------------------------------------------------------------
    # Calibration analysis
    # ------------------------------------------------------------------

    def calibration_analysis(
        self,
        predicted_evs: Sequence[float],
        realized_outcomes: Sequence[float],
        n_bins: int = 10,
    ) -> Dict[str, object]:
        """Analyse how well predicted EVs calibrate to realised outcomes.

        Returns a dict with:
        - 'mean_absolute_error': float
        - 'root_mean_squared_error': float
        - 'correlation': float
        - 'calibration_error': float  (mean |bin_predicted - bin_realized|)
        - 'bin_data': list of (pred_mean, real_mean, count) per bin
        - 'is_well_calibrated': bool  (calibration_error < 0.05)
        """
        pred = np.asarray(predicted_evs, dtype=float)
        real = np.asarray(realized_outcomes, dtype=float)
        if len(pred) != len(real):
            raise ValueError("predicted_evs and realized_outcomes must have the same length")
        if len(pred) == 0:
            raise ValueError("Empty sequences")

        mae = float(np.mean(np.abs(pred - real)))
        rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
        corr = float(np.corrcoef(pred, real)[0, 1]) if len(pred) > 1 else 0.0

        # Bin by predicted EV
        pred_min, pred_max = pred.min(), pred.max()
        if pred_max == pred_min:
            bins = [pred_min]
        else:
            bins = np.linspace(pred_min, pred_max, n_bins + 1)

        bin_indices = np.digitize(pred, bins) - 1
        bin_data = []
        calibration_errors = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            bin_pred_mean = float(pred[mask].mean())
            bin_real_mean = float(real[mask].mean())
            bin_data.append((bin_pred_mean, bin_real_mean, int(mask.sum())))
            calibration_errors.append(abs(bin_pred_mean - bin_real_mean))

        calibration_error = float(np.mean(calibration_errors)) if calibration_errors else 0.0

        return {
            "mean_absolute_error": mae,
            "root_mean_squared_error": rmse,
            "correlation": corr,
            "calibration_error": calibration_error,
            "is_well_calibrated": calibration_error < 0.05,
            "bin_data": bin_data,
            "n_samples": len(pred),
        }

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_ev(
        self,
        action_ev_fn,
        n_simulations: int = 100_000,
        rng_seed: Optional[int] = 42,
        variance_per_hand: float = 1.15,
    ) -> MonteCarloResult:
        """Monte Carlo simulation of expected value over many hands.

        Parameters
        ----------
        action_ev_fn:
            Callable() → float, returns the EV of one hand.
        n_simulations:
            Number of hands to simulate.
        rng_seed:
            Random seed for reproducibility.
        variance_per_hand:
            Per-hand variance for outcome sampling around EV.

        Notes
        -----
        Each simulated hand outcome is drawn as:
            outcome_i = EV_i + ε_i,  ε_i ~ N(0, σ²)
        This provides a realistic sampling distribution.
        """
        rng = np.random.default_rng(rng_seed)
        evs = []
        for _ in range(n_simulations):
            ev = action_ev_fn()
            # Realise outcome with noise
            noise = rng.normal(0.0, math.sqrt(variance_per_hand))
            realised = float(ev) + noise
            evs.append(realised)

        arr = np.array(evs)
        mean_ev = float(np.mean(arr))
        std_ev = float(np.std(arr, ddof=1))
        ci = (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
        win_rate = float(np.mean(arr > 0))

        # Convergence: find n where running CI width < 0.01
        convergence_n = n_simulations  # default if never converges
        cumsum = np.cumsum(arr)
        for n in range(10, n_simulations, 100):
            running_mean = cumsum[n - 1] / n
            running_std = float(np.std(arr[:n], ddof=1))
            ci_width = 2 * 1.96 * running_std / math.sqrt(n)
            if ci_width < 0.01:
                convergence_n = n
                break

        # Expected win rate (from mean EV using normal approx)
        z = mean_ev / (math.sqrt(variance_per_hand) / math.sqrt(n_simulations))
        expected_win_rate = float(stats.norm.cdf(z))

        return MonteCarloResult(
            n_simulations=n_simulations,
            mean_ev=mean_ev,
            std_ev=std_ev,
            ci_95=ci,
            win_rate=win_rate,
            expected_win_rate=expected_win_rate,
            convergence_hands=convergence_n,
        )

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    def convergence_analysis(
        self,
        observation_ratios: Sequence[float],
        actual_accuracies: Sequence[float],
    ) -> Dict[str, object]:
        """Analyse convergence of accuracy as observation_ratio increases.

        Fits a power-law model: accuracy(r) = 1 - a * r^(-b)

        Returns dict with:
        - 'fitted_a': float
        - 'fitted_b': float (convergence exponent)
        - 'r_squared': float (fit quality)
        - 'predicted_accuracy_at_full': float
        - 'hands_for_95pct_accuracy': int
        """
        ratios = np.asarray(observation_ratios, dtype=float)
        accs = np.asarray(actual_accuracies, dtype=float)

        if len(ratios) < 3:
            return {"error": "Need at least 3 data points"}

        valid = (ratios > 0) & (accs < 1) & (accs > 0)
        if valid.sum() < 3:
            return {"error": "Insufficient valid data points"}

        r = ratios[valid]
        a = accs[valid]

        # Log-linear fit: log(1 - a) = log(c) + b * log(r)
        try:
            log_r = np.log(r)
            log_residual = np.log(np.clip(1.0 - a, 1e-10, None))
            slope, intercept, r_value, p_value, se = stats.linregress(log_r, log_residual)
            b = slope
            c = math.exp(intercept)
        except Exception:
            return {"error": "Convergence fit failed"}

        r_squared = float(r_value ** 2)
        pred_at_full = 1.0 - c * 1.0 ** b  # at ratio=1.0

        # Find ratio where accuracy crosses 0.95
        if b < 0 and c > 0:
            target_residual = 1.0 - 0.95
            r_95 = (target_residual / c) ** (1.0 / b) if c > 0 else None
        else:
            r_95 = None

        return {
            "fitted_c": c,
            "fitted_b": b,
            "r_squared": r_squared,
            "p_value": float(p_value),
            "predicted_accuracy_at_full": pred_at_full,
            "observation_ratio_for_95pct": r_95,
        }
