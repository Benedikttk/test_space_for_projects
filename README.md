# Blackjack Probability, Shoe Tracking & EV Engine

A research-grade Python system that watches a live blackjack table via screen
capture, classifies cards using computer vision, tracks the shoe composition in
real time, and computes the mathematically optimal action and its exact expected
value at every decision point.

> **Note:** GitHub renders LaTeX math in Markdown using `$...$` for inline and `$$...$$` for display equations.
> All formulas below are written in standard LaTeX notation.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Data is Captured](#how-data-is-captured)
3. [Mid-Shoe Join — Joining a Live Game](#mid-shoe-join)
4. [The Mathematics](#the-mathematics)
   - [1. Shoe Composition & Card Probabilities](#1-shoe-composition--card-probabilities)
   - [2. Hand Evaluation — Best Total](#2-hand-evaluation--best-total)
   - [3. Dealer Outcome Distribution](#3-dealer-outcome-distribution)
   - [4. Expected Value of Standing](#4-expected-value-of-standing)
   - [5. Expected Value of Hitting](#5-expected-value-of-hitting)
   - [6. Expected Value of Doubling](#6-expected-value-of-doubling)
   - [7. Expected Value of Splitting (Recursive)](#7-expected-value-of-splitting-recursive)
   - [8. Surrender](#8-surrender)
   - [9. Optimal Action Selection](#9-optimal-action-selection)
   - [10. Card Counting — Hi-Lo True Count](#10-card-counting--hi-lo-true-count)
   - [11. Mid-Shoe Join — Maximum-Entropy Prior](#11-mid-shoe-join--maximum-entropy-prior)
5. [Rule Variants & EV Impact](#rule-variants--ev-impact)
6. [Module Overview](#module-overview)
7. [Project Layout](#project-layout)
8. [Compliance Note](#compliance-note)

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]            # core + pytest + ruff + mypy
pip install -e .[vision_dl]      # optional: YOLO detector
pytest
streamlit run app.py             # launch the UI
```

---

## How Data is Captured

The system has two input modes that can be mixed freely.

### Mode 1 — Screen-capture + ML detection (automated)

```
Monitor pixels  (mss, 2 FPS)
      │
      ▼  crop configured regions from vision.yaml
 ┌──────────────────────────────────────────────┐
 │  dealer_region   player_region   seat_1…N    │
 └──────────────────────────────────────────────┘
      │
      ▼  CardDetector
 DetectionResult(rank, suit, confidence)
      │
      ├── confidence ≥ 0.85  →  accepted  →  Shoe.remove(rank)
      ├── 0.75 – 0.84        →  review queue in UI
      └── < 0.75             →  rejected
      │
      ▼
 action_evs(hand, upcard, shoe, rules)  →  recommendation + EV table
```

**TemplateDetector** (zero GPU, default): loads `<rank>_<suit>.png` reference
images from `data/templates/`, runs multi-scale OpenCV `matchTemplate`, returns
best match with normalised correlation score as confidence.

**YOLODetector** (GPU optional): loads `.pt` YOLO weights, maps predicted class
label (`A_S`, `T_H`, …) to `DetectionResult`. Install: `pip install -e .[vision_dl]`.

Cards from **all seats** (other players + dealer) are removed from the shoe —
they reduce the remaining composition and are the information that drives
count-adjusted EV. Only **your cards and the dealer upcard** enter the EV formula.

### Mode 2 — Manual entry

Type each rank into the UI. Calls `ShoeState.ingest_manual(rank)` at confidence 1.0.

---

## Mid-Shoe Join

When you sit down at a table already in progress, call:

```python
shoe_state = ShoeState(decks=8, mid_shoe_join=True)
# or at runtime: shoe_state.join_mid_shoe()
```

The shoe starts at full capacity. Cards are only removed as they are directly
observed. See [Section 11](#11-mid-shoe-join--maximum-entropy-prior) for the
mathematics.

---

## The Mathematics

All expected values are in units of the **original bet**:
$+1$ = win one bet, $-1$ = lose one bet, $0$ = push.

---

### 1. Shoe Composition & Card Probabilities

A shoe of $N$ decks contains $52N$ cards. Face cards J, Q, K are grouped with
T (all worth 10), giving 10 canonical ranks:

| Rank | Count per deck | Hard value |
|------|---------------|------------|
| 2 – 9 | 4 each | face value |
| T (T/J/Q/K) | 16 | 10 |
| A | 4 | 1 or 11 |

Let $n_r$ denote the count of rank $r$ remaining in the shoe, and $M = \sum_r n_r$
the total cards remaining. The probability that the next card drawn is rank $r$ is:

$$P(\text{next} = r \mid \mathcal{S}) = \frac{n_r}{M}$$

This is **sampling without replacement** — the exact model for a finite shoe.
It is strictly more accurate than the infinite-deck approximation (uniform
sampling with replacement) because it correctly captures how probabilities
shift as cards are removed.

**Code:** `Shoe.prob(rank)` → `shoe.counts[rank] / shoe.total_remaining`

---

### 2. Hand Evaluation — Best Total

A hand $H = (c_1, c_2, \ldots, c_k)$ has a **best total** computed by the
greedy ace-reduction algorithm. Let $a$ be the number of aces and $s$ the sum
treating each ace as 11:

$$T(H) = s - 10 \cdot \max\!\left(0,\; \left\lceil \frac{s - 21}{10} \right\rceil\right) \text{ (clamped to reduce } a \text{ aces)}$$

More precisely in code: start with all aces at 11, then while $T > 21$ and
there are aces counted as 11, subtract 10. The hand is **soft** if at least
one ace remains at value 11 after reduction and $T \leq 21$:

$$\text{soft}(H) = \mathbb{1}\!\left[a_{\text{remaining}} \geq 1 \;\wedge\; T(H) \leq 21\right]$$

**Examples:**

| Hand | Total | Soft? |
|------|-------|-------|
| $[\text{A}, 6]$ | 17 | ✓ (A=11) |
| $[\text{A}, 6, 9]$ | 16 | ✗ (A reduced to 1) |
| $[\text{A}, \text{A}]$ | 12 | ✓ (one A=11, one A=1) |
| $[\text{A}, \text{A}, 9]$ | 21 | ✓ |

A **natural blackjack** satisfies $|H| = 2$ and $T(H) = 21$.

**Code:** `hand_total(cards)` → `(total: int, is_soft: bool)`

---

### 3. Dealer Outcome Distribution

The dealer follows a deterministic rule: hit while $T < 17$, or additionally
while $T = 17$ and $\text{soft}(H) = \text{true}$ under H17 rules. The dealer's
final total distribution is computed by **recursive probability weighting** over
all draw sequences.

Let $\mathcal{S}^{-} = \mathcal{S} \setminus \{u\} \setminus H_{\text{player}}$
denote the shoe snapshot after removing the dealer upcard $u$ and all visible
player cards $H_{\text{player}}$. Then:

$$P(D = d \mid u, \mathcal{S}) = \sum_{\sigma \in \Sigma(u,d)} \prod_{i=1}^{|\sigma|} \frac{n_{\sigma_i}(\mathcal{S}^{-} \setminus \sigma_{1:i-1})}{M(\mathcal{S}^{-} \setminus \sigma_{1:i-1})}$$

where $\Sigma(u, d)$ is the set of all card sequences starting from upcard $u$
that cause the dealer to stop at final total $d$.

In practice this is computed by the recursive function:

$$\text{DealerDist}(H_D, p, \mathcal{C}) = \begin{cases}
\{T(H_D) \mapsto p\} & \text{if dealer must stand} \\
\displaystyle\sum_{r:\, n_r > 0} \text{DealerDist}\!\left(H_D \cup \{r\},\; p \cdot \frac{n_r}{M},\; \mathcal{C} \setminus \{r\}\right) & \text{otherwise}
\end{cases}$$

Bust is encoded as $d = 22$. The shoe $\mathcal{C}$ is a **local copy** at each
branch — the live shoe is never mutated.

$$\sum_{d \in \{17,18,19,20,21,22\}} P(D = d) = 1$$

**Code:** `dealer_distribution(upcard, shoe, rules, player_cards)`

---

### 4. Expected Value of Standing

Given player total $P$ and dealer final distribution $\{P(D=d)\}$:

$$\text{EV}_{\text{stand}}(P, D) = \sum_{d} P(D = d) \cdot \omega(P, d)$$

where the outcome function $\omega$ is:

$$\omega(P, d) = \begin{cases}
+\lambda & d = 22 \quad \text{(dealer bust)} \\
+\lambda & P = \text{BJ},\; d \neq 21 \\
+\lambda & P = \text{BJ},\; d = 21,\; \texttt{natural\_beats\_dealer\_21} = \text{true} \\
\phantom{+}0 & P = \text{BJ},\; d = 21,\; \texttt{natural\_beats\_dealer\_21} = \text{false} \quad \text{(push)} \\
+1 & P > d \quad \text{(player wins)} \\
-1 & P < d \quad \text{(player loses)} \\
\phantom{+}0 & P = d \quad \text{(push)}
\end{cases}$$

Here $\lambda$ is the **payout multiplier**: $\lambda = \texttt{blackjack\_payout}$
(e.g. $1.5$ for 3:2) for naturals, and $\lambda = 1$ for all other wins.

**Code:** `_stand_ev(player_total, dealer_dist, is_blackjack, rules)`

---

### 5. Expected Value of Hitting

Hitting is a **decision under uncertainty with future choices** — a stochastic
dynamic programme. At each step the player draws a card and then plays optimally.
This is solved by **backward induction**:

$$\text{EV}_{\text{hit}}(H, \mathcal{C}) = \sum_{r:\, n_r > 0} \frac{n_r}{M(\mathcal{C})} \cdot \begin{cases}
-1 & T(H \cup \{r\}) > 21 \quad \text{(bust)} \\[4pt]
\max\!\Bigl(\text{EV}_{\text{stand}}\bigl(T(H \cup \{r\})\bigr),\; \text{EV}_{\text{hit}}\bigl(H \cup \{r\},\; \mathcal{C} \setminus \{r\}\bigr)\Bigr) & \text{otherwise}
\end{cases}$$

Key correctness properties:

- **Per-branch shoe depletion**: the shoe $\mathcal{C} \setminus \{r\}$ passed into
  the recursive call has rank $r$ removed, so a drawn card can never be
  sampled again at deeper levels.
- **Optimal continuation**: the $\max(\text{stand}, \text{hit})$ at each node
  means the player always chooses the better option — this is exactly backward
  induction over the probability tree.
- **Termination**: recursion stops when the player busts ($T > 21$) or a depth
  limit is reached (safety valve at depth 10, unreachable for real shoes).

**Code:** `_hit_ev(player_cards, dealer_dist, counts, rules)`

---

### 6. Expected Value of Doubling

Double down: place an additional equal bet, draw **exactly one card**, then
stand. The total stake is $2 \times$ the original bet:

$$\text{EV}_{\text{double}}(H, \mathcal{C}) = \sum_{r:\, n_r > 0} \frac{n_r}{M(\mathcal{C})} \cdot \begin{cases}
-2 & T(H \cup \{r\}) > 21 \quad \text{(bust)} \\[4pt]
2 \cdot \text{EV}_{\text{stand}}\bigl(T(H \cup \{r\})\bigr) & \text{otherwise}
\end{cases}$$

The factor of 2 reflects that both the original and the doubled bet are at risk.
No further hits are allowed after the double card — this is enforced by
`get_legal_actions` which sets `can_hit = False` when `hand.doubled = True`.

**Code:** inside `action_evs`, double branch

---

### 7. Expected Value of Splitting (Recursive)

Splitting is the most complex action because each child hand can itself be
split again (up to `max_splits` times), doubled, or hit optimally.

**Setup:** after splitting a pair of rank $r$, both copies of $r$ leave the shoe:

$$\mathcal{C}^{(r)} = \mathcal{C} \setminus \{r\} \setminus \{r\}$$

Each child hand then draws one second card $c$ from $\mathcal{C}^{(r)}$,
producing hand $[r, c]$ with a further-depleted shoe $\mathcal{C}^{(r)} \setminus \{c\}$.

$$\text{EV}_{\text{split}}(r, \mathcal{C}) = 2 \sum_{c:\, n_c^{(r)} > 0} \frac{n_c^{(r)}}{M(\mathcal{C}^{(r)})} \cdot \max_{a \in \mathcal{A}([r,c])} \text{EV}_a\!\left([r, c],\; \mathcal{C}^{(r)} \setminus \{c\}\right)$$

where $\mathcal{A}([r, c])$ is the set of legal actions for the child hand
(determined by `get_legal_actions`), and the $\max$ recurses into
`action_evs` — which can include further splits, doubles, hits, and stands.

The factor of **2** reflects that both child hands are played for the same
original stake.

**Ace-split restriction:** when `split_aces_get_one_card = True`, each
post-split-ace hand is locked to exactly one drawn card (only stand allowed):

$$\mathcal{A}([A, c]) = \{\text{stand}\} \quad \text{when \texttt{is\_post\_split\_ace} = true}$$

**Single-responsibility depletion:** the dealer upcard and player cards are
removed **exclusively** inside `dealer_distribution` and `action_evs`.
`split_ev` only removes the two split-pair cards. This prevents double-depletion
across the recursion boundary — a subtle correctness invariant.

**Code:** `split_ev(hand, dealer_upcard, shoe, rules, splits_used)`

---

### 8. Surrender

Late surrender: the player forfeits half the bet before playing. The EV is
constant regardless of hand or dealer upcard:

$$\text{EV}_{\text{surrender}} = -\tfrac{1}{2}$$

Surrender is optimal when:

$$-\tfrac{1}{2} > \max\!\left(\text{EV}_{\text{stand}},\; \text{EV}_{\text{hit}},\; \text{EV}_{\text{double}},\; \text{EV}_{\text{split}}\right)$$

The most common case where this holds is hard 16 vs. dealer T under standard rules,
where $\text{EV}_{\text{stand}} \approx -0.54$ and $\text{EV}_{\text{hit}} \approx -0.51$.

**Early surrender** (very rare): can be taken before the dealer checks for blackjack.
In this engine both modes return $-0.5$; the difference is encoded in *when*
the action is gated (early surrender is available even when the dealer upcard
is A or T).

**Code:** `result["surrender"] = -0.5` inside `action_evs`

---

### 9. Optimal Action Selection

Given the full EV dictionary for all legal actions at a decision point:

$$a^* = \arg\max_{a \in \mathcal{A}(H)} \text{EV}_a(H, \mathcal{C})$$

$$\text{EV}^* = \text{EV}_{a^*}(H, \mathcal{C})$$

The **delta** for each suboptimal action $a \neq a^*$ is:

$$\Delta_a = \text{EV}_a - \text{EV}^* \leq 0$$

This delta is what is shown in the UI's EV table — it tells you how much EV
you lose by choosing that action instead of the optimal one.

**Code:** `best_action(ev_dict)` → `(action_name, ev)`

---

### 10. Card Counting — Hi-Lo True Count

The **running count** $\text{RC}$ is updated on every observed card $r$:

$$\Delta \text{RC}(r) = \begin{cases}
+1 & r \in \{2, 3, 4, 5, 6\} \quad \text{(low cards help dealer)} \\
\phantom{+}0 & r \in \{7, 8, 9\} \quad \text{(neutral)} \\
-1 & r \in \{T, A\} \quad \text{(high cards help player)}
\end{cases}$$

The **true count** $\text{TC}$ normalises for decks remaining:

$$\text{TC} = \frac{\text{RC}}{D_{\text{remaining}}} = \frac{\text{RC}}{\,M / 52\,}$$

A true count of $+1$ corresponds to approximately $+0.5\%$ player EV relative
to the neutral shoe.

**Important:** the EV engine does **not** use $\text{TC}$ directly. It uses the
**exact shoe composition** $\{n_r\}$, which strictly subsumes all count
information. The true count is displayed in the UI as a human-readable summary
only.

**Code:** `Shoe.true_count` → `running_count / (total_remaining / 52)`

---

### 11. Mid-Shoe Join — Maximum-Entropy Prior

When joining a live game in progress, an unknown number of cards $k$ have
been dealt before arrival. Let $\mathcal{S}_0$ be the full fresh shoe
($52N$ cards) and $\mathcal{O}$ be the set of cards directly observed since
joining. The model maintains:

$$\hat{\mathcal{S}} = \mathcal{S}_0 \setminus \mathcal{O}$$

Cards dealt before arrival are **not** removed — they remain as if still in
the shoe. This is the **maximum-entropy prior**: in the absence of information
about which ranks were dealt early, a uniform residual distribution is the
least-biased estimate. Formally, for any rank $r$:

$$\hat{P}(\text{next} = r) = \frac{n_r^{(0)} - n_r^{(\mathcal{O})}}{52N - |\mathcal{O}|}$$

where $n_r^{(0)}$ is the full-shoe count and $n_r^{(\mathcal{O})}$ is the
number of rank $r$ observed. As $|\mathcal{O}| \to 52N$, this converges to
the true posterior.

The **observation ratio** tracks information quality:

$$\rho = \frac{|\mathcal{O}|}{52N} \in [0, 1]$$

| $\rho$ | EV quality |
|--------|-----------|
| $< 0.10$ | Near basic strategy; count signal unreliable |
| $0.10 - 0.25$ | Mild count signal; treat with caution |
| $0.25 - 0.60$ | Count-adjusted EVs meaningful |
| $> 0.60$ | High-confidence shoe-aware EVs |

**Code:** `ShoeState.observation_ratio`, `ShoeState.is_uncertain`, `ShoeState.uncertainty_label`

---

## Rule Variants & EV Impact

All rule variants are encoded in `RuleSet` and propagate through every calculation automatically:

| Rule flag | Effect on house edge |
|-----------|---------------------|
| `dealer_hits_soft17` (H17 vs S17) | +0.22% to house |
| `double_after_split` (DAS) | −0.14% to house |
| `resplit_aces` (RSA) | −0.08% to house |
| `max_splits` 1 → 4 | −0.05% per extra split allowed |
| `blackjack_payout` 3:2 → 6:5 | +1.37% to house (large) |
| `surrender = late` | −0.07% to house |
| `surrender = early` | −0.24% to house |

Pre-built canonical rule sets: `STRIP_S17`, `DOWNTOWN_H17`, `LIBERAL`, `TOURIST_TRAP`

---

## Module Overview

```
blackjack/
├── rules.py        RuleSet — single source of truth for all rule flags
├── hand.py         hand_total, is_soft, is_blackjack, can_split
├── shoe.py         Shoe — counts, remove, running_count, true_count
├── shoe_state.py   ShoeState — mid-shoe join, confidence thresholds, observation_ratio
├── actions.py      get_legal_actions — pure gate function
├── ev.py           dealer_distribution, action_evs, split_ev, best_action
├── detector.py     TemplateDetector (OpenCV) + YOLODetector (YOLO)
├── capture.py      CaptureSession — background grab → detect → ingest loop
└── ui_state.py     AppState, format_ev_table, health_status
```

---

## Project Layout

```
test_space_for_projects/
├── app.py              Streamlit 5-panel UI
├── blackjack/          Python engine package
├── tests/              pytest test suite (7 files)
├── configs/
│   ├── app.yaml        FPS, monitor, confidence thresholds
│   └── vision.yaml     Screen region coordinates per seat
├── data/
│   ├── templates/      Card PNG templates for TemplateDetector
│   ├── models/         YOLO weights (cards.pt)
│   └── observations.csv
├── notebooks/
├── experiments/
├── reports/
└── pyproject.toml
```

---

## Compliance Note

Screen capture of a live platform may violate that platform's Terms of Service.
This codebase is intended for **research and educational purposes only**.
Always verify legal constraints before use in any live environment.
