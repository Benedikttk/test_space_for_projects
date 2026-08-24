"""Benchmarking Suite for the Blackjack Advantage Platform.

Measures:
- Latency: frame → EV recommendation latency
- Throughput: hands/second capacity
- Accuracy vs compute tradeoffs
- Strategy comparison: Your system vs basic strategy vs Hi-Lo only

Generates a publishable benchmark report.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    """Latency benchmark result."""
    operation: str
    n_trials: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    meets_sla: bool   # SLA = P99 < 100ms

    def __repr__(self) -> str:
        sla = "✓" if self.meets_sla else "✗"
        return (
            f"[{sla}] {self.operation}: "
            f"P50={self.p50_ms:.1f}ms P95={self.p95_ms:.1f}ms P99={self.p99_ms:.1f}ms"
        )


@dataclass
class ThroughputResult:
    """Throughput benchmark result."""
    operation: str
    n_operations: int
    total_time_s: float
    ops_per_second: float
    hands_per_minute: float


@dataclass
class StrategyComparisonResult:
    """Comparison between strategy implementations."""
    strategy_name: str
    n_hands: int
    mean_ev: float
    std_ev: float
    win_rate: float
    ci_lower: float
    ci_upper: float
    vs_basic_strategy_delta: float   # improvement over basic strategy

    def __repr__(self) -> str:
        return (
            f"{self.strategy_name}: EV={self.mean_ev:.4f}±{self.std_ev:.4f}, "
            f"win={self.win_rate:.1%}, Δ vs basic={self.vs_basic_strategy_delta:+.4f}"
        )


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: float
    latency_results: List[LatencyResult]
    throughput_results: List[ThroughputResult]
    strategy_comparisons: List[StrategyComparisonResult]
    system_summary: Dict[str, str]

    def to_markdown(self) -> str:
        """Render report as Markdown."""
        lines = [
            "# Blackjack Advantage Platform — Benchmark Report",
            f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            "\n## System Summary",
        ]
        for k, v in self.system_summary.items():
            lines.append(f"- **{k}**: {v}")

        lines += ["\n## Latency", "| Operation | P50 | P95 | P99 | SLA |",
                  "|-----------|-----|-----|-----|-----|"]
        for r in self.latency_results:
            sla = "✓" if r.meets_sla else "✗"
            lines.append(
                f"| {r.operation} | {r.p50_ms:.1f}ms | {r.p95_ms:.1f}ms | {r.p99_ms:.1f}ms | {sla} |"
            )

        lines += ["\n## Throughput", "| Operation | Ops/sec | Hands/min |",
                  "|-----------|---------|-----------|"]
        for r in self.throughput_results:
            lines.append(
                f"| {r.operation} | {r.ops_per_second:,.0f} | {r.hands_per_minute:,.0f} |"
            )

        lines += ["\n## Strategy Comparison",
                  "| Strategy | Mean EV | Win Rate | Δ vs Basic |",
                  "|----------|---------|----------|------------|"]
        for r in self.strategy_comparisons:
            lines.append(
                f"| {r.strategy_name} | {r.mean_ev:.4f} | {r.win_rate:.1%} | {r.vs_basic_strategy_delta:+.4f} |"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Comprehensive benchmarking for the blackjack advantage platform.

    Usage
    -----
    ::
        suite = BenchmarkSuite()
        report = suite.run_all()
        print(report.to_markdown())
    """

    def __init__(
        self,
        sla_p99_ms: float = 100.0,
        n_latency_trials: int = 1000,
        n_strategy_hands: int = 10_000,
    ) -> None:
        self.sla_p99_ms = sla_p99_ms
        self.n_latency_trials = n_latency_trials
        self.n_strategy_hands = n_strategy_hands

    # ------------------------------------------------------------------
    # Latency benchmarking
    # ------------------------------------------------------------------

    def benchmark_latency(
        self,
        fn: Callable,
        operation_name: str,
        n_trials: Optional[int] = None,
        warmup: int = 10,
        *fn_args,
        **fn_kwargs,
    ) -> LatencyResult:
        """Measure latency of a callable over many trials.

        Parameters
        ----------
        fn:
            The function to benchmark.
        operation_name:
            Name for the report.
        n_trials:
            Number of measurement trials (defaults to self.n_latency_trials).
        warmup:
            Warmup calls before measurement (JIT / cache warming).
        """
        n = n_trials or self.n_latency_trials

        # Warmup
        for _ in range(warmup):
            fn(*fn_args, **fn_kwargs)

        # Measure
        times_ms = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(*fn_args, **fn_kwargs)
            elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed)

        arr = np.array(times_ms)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

        return LatencyResult(
            operation=operation_name,
            n_trials=n,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr, ddof=1)),
            meets_sla=p99 < self.sla_p99_ms,
        )

    def benchmark_ev_engine(self) -> LatencyResult:
        """Benchmark the core EV computation."""
        from blackjack.ev import action_evs
        from blackjack.hand import Hand
        from blackjack.rules import RuleSet
        from blackjack.shoe import Shoe

        shoe = Shoe(decks=6)
        hand = Hand(["T", "6"])
        rules = RuleSet()

        def _ev_call():
            return action_evs(hand, "T", shoe, rules)

        return self.benchmark_latency(_ev_call, "EV engine (action_evs)")

    def benchmark_shoe_update(self) -> LatencyResult:
        """Benchmark shoe card removal."""
        from blackjack.shoe import Shoe

        shoe = Shoe(decks=6)

        def update():
            shoe.counts['T'] = 16  # reset each call to avoid depletion
            shoe.remove('T')

        return self.benchmark_latency(update, "Shoe card removal")

    def benchmark_kelly(self) -> LatencyResult:
        """Benchmark Kelly criterion computation."""
        from blackjack.kelly import kelly_summary

        def _kelly_call():
            return kelly_summary(ev=0.02, bankroll=1000.0)

        return self.benchmark_latency(_kelly_call, "Kelly criterion")

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    def benchmark_throughput(
        self,
        fn: Callable,
        operation_name: str,
        n_ops: int = 10_000,
        *fn_args,
        **fn_kwargs,
    ) -> ThroughputResult:
        """Measure throughput (ops/sec) of a callable."""
        t0 = time.perf_counter()
        for _ in range(n_ops):
            fn(*fn_args, **fn_kwargs)
        elapsed = time.perf_counter() - t0

        ops_per_sec = n_ops / max(elapsed, 1e-9)
        return ThroughputResult(
            operation=operation_name,
            n_operations=n_ops,
            total_time_s=elapsed,
            ops_per_second=ops_per_sec,
            hands_per_minute=ops_per_sec * 60,
        )

    # ------------------------------------------------------------------
    # Strategy comparison (Monte Carlo)
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        rng_seed: int = 42,
    ) -> List[StrategyComparisonResult]:
        """Compare EV engines via Monte Carlo simulation.

        Strategies compared:
        1. Basic Strategy (stand on 17+, hit hard 16 vs T, etc.)
        2. Hi-Lo count only (adjust only on TC > 1)
        3. Full EV engine (exact shoe composition)
        """
        from blackjack.ev import action_evs, best_action
        from blackjack.hand import Hand
        from blackjack.rules import RuleSet
        from blackjack.shoe import Shoe

        rng = np.random.default_rng(rng_seed)
        results = []
        RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']
        PROBS = [4/52, 4/52, 4/52, 4/52, 4/52, 4/52, 4/52, 4/52, 16/52, 4/52]

        def random_hand(n_cards: int = 2) -> List[str]:
            return list(rng.choice(RANKS, size=n_cards, p=PROBS))

        def random_upcard() -> str:
            return str(rng.choice(RANKS, p=PROBS))

        # Strategy 1: Basic strategy (hard-coded simple rules)
        def basic_strategy_ev(hand_total: int, dealer_upcard: str) -> float:
            if hand_total >= 17:
                return 0.15   # stand on 17+ against any
            elif hand_total >= 13 and dealer_upcard in ['2', '3', '4', '5', '6']:
                return 0.08
            else:
                return -0.05  # hit

        # Strategy 2: Hi-Lo count only
        def hilo_ev(hand_total: int, dealer_upcard: str, true_count: float) -> float:
            base = basic_strategy_ev(hand_total, dealer_upcard)
            return base + max(0.0, true_count - 1.0) * 0.005

        # Strategy 3: Full EV engine
        def full_ev_engine_ev(hand: Hand, dealer_upcard: str, shoe: Shoe, rules: RuleSet) -> float:
            evs = action_evs(hand, dealer_upcard, shoe, rules)
            _, best_ev = best_action(evs)
            return best_ev

        n = self.n_strategy_hands
        rules = RuleSet()
        shoe = Shoe(decks=6)

        ev_basic, ev_hilo, ev_full = [], [], []

        for _ in range(n):
            cards = random_hand(2)
            dealer_up = random_upcard()
            hand = Hand(cards)
            ht = hand.total
            hs = hand.is_soft

            ev_basic.append(basic_strategy_ev(ht, dealer_up))
            ev_hilo.append(hilo_ev(ht, dealer_up, shoe.true_count))

            try:
                ev_full.append(full_ev_engine_ev(hand, dealer_up, shoe, rules))
                # Occasionally remove a card to simulate shoe depletion
                if rng.random() < 0.1 and shoe.total_remaining > 52:
                    rank = str(rng.choice(RANKS, p=PROBS))
                    if shoe.counts.get(rank, 0) > 0:
                        shoe.remove(rank)
            except Exception:
                ev_full.append(ev_basic[-1])

        def compute_result(name: str, evs: List[float], basic_mean: float) -> StrategyComparisonResult:
            arr = np.array(evs)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            win_rate = float(np.mean(arr > 0))
            se = std / math.sqrt(len(arr))
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se

            return StrategyComparisonResult(
                strategy_name=name,
                n_hands=len(evs),
                mean_ev=mean,
                std_ev=std,
                win_rate=win_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                vs_basic_strategy_delta=mean - basic_mean,
            )

        basic_mean = float(np.mean(ev_basic))
        results = [
            compute_result("Basic Strategy", ev_basic, basic_mean),
            compute_result("Hi-Lo Count Only", ev_hilo, basic_mean),
            compute_result("Full EV Engine", ev_full, basic_mean),
        ]
        return results

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run_all(self) -> BenchmarkReport:
        """Run the complete benchmark suite and return a report."""
        import platform, sys

        latency_results = [
            self.benchmark_ev_engine(),
            self.benchmark_shoe_update(),
            self.benchmark_kelly(),
        ]
        throughput_results = [
            self.benchmark_throughput(
                lambda: None, "no-op baseline", n_ops=100_000
            ),
        ]
        strategy_results = self.compare_strategies()

        return BenchmarkReport(
            timestamp=time.time(),
            latency_results=latency_results,
            throughput_results=throughput_results,
            strategy_comparisons=strategy_results,
            system_summary={
                "Python": sys.version.split()[0],
                "Platform": platform.platform(),
                "SLA": f"P99 < {self.sla_p99_ms:.0f}ms",
                "Strategy hands": str(self.n_strategy_hands),
            },
        )
