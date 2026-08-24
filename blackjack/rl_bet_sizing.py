"""Reinforcement Learning Bet Sizing Optimizer.

Implements a Q-learning agent for bet sizing under uncertainty.
The agent learns to map (bankroll, true_count, penetration) → bet_size.

State space:
    s = (bankroll_bucket, true_count_bucket, penetration_bucket)

Action space:
    a ∈ {1, 2, 4, 8, 16} × min_bet   (spread of 1-16 units)

Reward:
    r = profit on hand / initial_bet   (normalised)

Algorithm: tabular Q-learning with ε-greedy exploration.
For continuous state, a neural net approximation is sketched.

Comparison vs Kelly:
    The RL agent learns from experience; Kelly is analytically optimal
    under the model assumptions.  RL may discover nuanced patterns when
    the Kelly model mismatches reality (e.g. correlated hands, shuffle bias).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State / action spaces
# ---------------------------------------------------------------------------

# Discretize true count into 13 buckets: [-6, -5, ..., 0, ..., +6]
TRUE_COUNT_BUCKETS = list(range(-6, 7))  # 13 states

# Penetration buckets: 0–25%, 25–50%, 50–75%, 75–100%
PENETRATION_BUCKETS = [0, 1, 2, 3]

# Bankroll buckets: fraction of starting bankroll
BANKROLL_BUCKETS = [0, 1, 2, 3, 4]  # < 0.5x, 0.5–1x, 1–2x, 2–4x, 4x+

# Bet actions as multipliers of min_bet
BET_MULTIPLIERS = [1, 2, 4, 8, 16]
N_ACTIONS = len(BET_MULTIPLIERS)


def _discretize_true_count(tc: float) -> int:
    """Map true count to bucket index 0..12."""
    clipped = max(-6, min(6, round(tc)))
    return clipped + 6  # shift to [0, 12]


def _discretize_penetration(pen: float) -> int:
    """Map penetration (0..1) to bucket 0..3."""
    return min(3, int(pen * 4))


def _discretize_bankroll(ratio: float) -> int:
    """Map bankroll ratio (current/starting) to bucket 0..4."""
    if ratio < 0.5:
        return 0
    elif ratio < 1.0:
        return 1
    elif ratio < 2.0:
        return 2
    elif ratio < 4.0:
        return 3
    else:
        return 4


State = Tuple[int, int, int]  # (tc_bucket, pen_bucket, bk_bucket)


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------


@dataclass
class RLTrainingResult:
    """Training history for the RL agent."""
    episode_rewards: List[float]
    mean_reward_per_episode: float
    final_epsilon: float
    n_episodes: int
    win_rate: float
    total_profit: float


class RLBetOptimizer:
    """Tabular Q-learning agent for bet sizing.

    State:  (true_count_bucket, penetration_bucket, bankroll_bucket)
    Action: bet multiplier ∈ {1, 2, 4, 8, 16}
    Reward: (hand_profit) / min_bet  (normalised)

    Q-learning update:
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

    Parameters
    ----------
    min_bet: float
        Table minimum bet.
    max_bet: float
        Table maximum bet (limits high multipliers).
    learning_rate: float
        Q-learning α ∈ (0, 1].
    discount: float
        γ ∈ [0, 1] — for single-hand games, γ=0.99.
    epsilon_start: float
        Initial ε for ε-greedy exploration.
    epsilon_end: float
        Final ε after annealing.
    epsilon_decay: float
        Multiplicative decay per episode.
    """

    def __init__(
        self,
        min_bet: float = 5.0,
        max_bet: float = 500.0,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        random_state: int = 42,
    ) -> None:
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(random_state)
        random.seed(random_state)

        # Q-table: shape (13, 4, 5, N_ACTIONS)
        n_tc = len(TRUE_COUNT_BUCKETS)
        n_pen = len(PENETRATION_BUCKETS)
        n_bk = len(BANKROLL_BUCKETS)
        self._Q = np.zeros((n_tc, n_pen, n_bk, N_ACTIONS))
        self._visit_counts = np.zeros_like(self._Q)

    def _state(self, true_count: float, penetration: float, bankroll_ratio: float) -> State:
        return (
            _discretize_true_count(true_count),
            _discretize_penetration(penetration),
            _discretize_bankroll(bankroll_ratio),
        )

    def choose_bet(
        self,
        true_count: float,
        penetration: float,
        bankroll_ratio: float = 1.0,
        explore: bool = False,
    ) -> float:
        """Choose a bet size using ε-greedy policy.

        Parameters
        ----------
        true_count:
            Current Hi-Lo true count.
        penetration:
            Deck penetration (0..1, fraction of shoe dealt).
        bankroll_ratio:
            Current bankroll / starting bankroll.
        explore:
            If True, apply ε-greedy. If False, use greedy policy.
        """
        s = self._state(true_count, penetration, bankroll_ratio)

        if explore and self._rng.random() < self.epsilon:
            action_idx = int(self._rng.integers(0, N_ACTIONS))
        else:
            action_idx = int(np.argmax(self._Q[s]))

        multiplier = BET_MULTIPLIERS[action_idx]
        bet = multiplier * self.min_bet
        return max(self.min_bet, min(self.max_bet, bet))

    def update(
        self,
        state: State,
        action_idx: int,
        reward: float,
        next_state: State,
    ) -> None:
        """Q-learning update step."""
        current_q = self._Q[state][action_idx]
        max_next_q = float(np.max(self._Q[next_state]))
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        # Adaptive learning rate via visit count
        self._visit_counts[state][action_idx] += 1
        adaptive_lr = self.lr / math.sqrt(float(self._visit_counts[state][action_idx]))

        self._Q[state][action_idx] += adaptive_lr * td_error

    def train(
        self,
        hand_simulator,
        n_episodes: int = 10_000,
        starting_bankroll: float = 1000.0,
        hands_per_episode: int = 50,
    ) -> RLTrainingResult:
        """Train the agent by simulating hands.

        Parameters
        ----------
        hand_simulator:
            Callable(bet) → (profit, true_count_next, penetration_next)
        n_episodes:
            Number of training episodes.
        starting_bankroll:
            Initial bankroll for ratio calculation.
        hands_per_episode:
            Hands per episode (episode = one shoe).
        """
        episode_rewards = []
        total_wins = 0
        total_hands = 0
        total_profit = 0.0

        for ep in range(n_episodes):
            bankroll = starting_bankroll
            ep_reward = 0.0
            true_count = 0.0
            penetration = 0.0

            for _ in range(hands_per_episode):
                bk_ratio = bankroll / starting_bankroll
                s = self._state(true_count, penetration, bk_ratio)

                # ε-greedy action
                if self._rng.random() < self.epsilon:
                    a_idx = int(self._rng.integers(0, N_ACTIONS))
                else:
                    a_idx = int(np.argmax(self._Q[s]))

                bet = min(self.max_bet, BET_MULTIPLIERS[a_idx] * self.min_bet)
                bet = min(bet, bankroll)  # can't bet more than we have

                # Simulate hand
                profit, tc_next, pen_next = hand_simulator(bet)
                reward = profit / max(self.min_bet, 1.0)

                bankroll += profit
                ep_reward += reward
                total_profit += profit
                total_hands += 1
                if profit > 0:
                    total_wins += 1

                # Next state
                bk_ratio_next = bankroll / starting_bankroll
                s_next = self._state(tc_next, pen_next, bk_ratio_next)

                # Q update
                self.update(s, a_idx, reward, s_next)

                true_count = tc_next
                penetration = pen_next

                if bankroll <= 0:
                    break

            episode_rewards.append(ep_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return RLTrainingResult(
            episode_rewards=episode_rewards,
            mean_reward_per_episode=float(np.mean(episode_rewards)),
            final_epsilon=self.epsilon,
            n_episodes=n_episodes,
            win_rate=total_wins / max(total_hands, 1),
            total_profit=total_profit,
        )

    def compare_with_kelly(
        self,
        true_count: float,
        ev_per_hand: float,
        variance: float = 1.15,
        bankroll: float = 1000.0,
        starting_bankroll: float = 1000.0,
        penetration: float = 0.5,
    ) -> Dict[str, float]:
        """Compare RL bet size with Kelly criterion bet size."""
        from blackjack.kelly import kelly_fraction, recommended_bet

        kf = kelly_fraction(ev_per_hand, variance=variance, half=True)
        kelly_bet = recommended_bet(
            ev=ev_per_hand,
            bankroll=bankroll,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
        )

        rl_bet = self.choose_bet(
            true_count=true_count,
            penetration=penetration,
            bankroll_ratio=bankroll / starting_bankroll,
            explore=False,
        )

        return {
            "kelly_fraction": kf,
            "kelly_bet": kelly_bet,
            "rl_bet": rl_bet,
            "difference": abs(rl_bet - kelly_bet),
            "kelly_units": kelly_bet / self.min_bet,
            "rl_units": rl_bet / self.min_bet,
        }

    def q_table_stats(self) -> Dict[str, float]:
        """Summary statistics of the Q-table."""
        return {
            "q_mean": float(np.mean(self._Q)),
            "q_std": float(np.std(self._Q)),
            "q_max": float(np.max(self._Q)),
            "q_min": float(np.min(self._Q)),
            "n_states_visited": int(np.sum(self._visit_counts > 0)),
            "total_states": int(np.prod(self._Q.shape[:-1])),
        }
