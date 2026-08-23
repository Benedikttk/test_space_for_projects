# Blackjack Probability, Shoe Tracking & EV Engine

A research-grade Python system that watches a live blackjack table via screen
capture, classifies cards using computer vision, tracks the shoe composition in
real time, and computes the mathematically optimal action and its exact expected
value at every decision point.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Data is Captured](#how-data-is-captured)
3. [Mid-Shoe Join](#mid-shoe-join)
4. [The Mathematics](#the-mathematics)

   * [1. Shoe Composition and Card Probabilities](#1-shoe-composition-and-card-probabilities)
   * [2. Hand Evaluation](#2-hand-evaluation)
   * [3. Dealer Outcome Distribution](#3-dealer-outcome-distribution)
   * [4. Expected Value of Standing](#4-expected-value-of-standing)
   * [5. Expected Value of Hitting](#5-expected-value-of-hitting)
   * [6. Expected Value of Doubling](#6-expected-value-of-doubling)
   * [7. Expected Value of Splitting](#7-expected-value-of-splitting)
   * [8. Surrender](#8-surrender)
   * [9. Insurance](#9-insurance)
   * [10. Optimal Action Selection](#10-optimal-action-selection)
   * [11. Card Counting — Hi-Lo True Count](#11-card-counting--hi-lo-true-count)
   * [12. Penetration-Dampened EV](#12-penetration-dampened-ev)
   * [13. Kelly Criterion Bet Sizing](#13-kelly-criterion-bet-sizing)
   * [14. Mid-Shoe Join — Maximum-Entropy Prior](#14-mid-shoe-join--maximum-entropy-prior)
5. [Rule Variants and EV Impact](#rule-variants-and-ev-impact)
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

### Mode 1 — Screen-capture + ML detection

```text
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

**TemplateDetector** (zero GPU): multi-scale OpenCV `matchTemplate` against
`data/templates/<rank>_<suit>.png`. **YOLODetector** (GPU optional): `.pt`
weights via `pip install -e .[vision_dl]`.

Cards from **all seats** are removed from the shoe. Only **your cards and the
dealer upcard** enter the EV computation.

### Mode 2 — Manual entry

`ShoeState.ingest_manual(rank)` at confidence 1.0.

---

## Mid-Shoe Join

```python
shoe_state = ShoeState(decks=8, mid_shoe_join=True)
```

See [Section 14](#14-mid-shoe-join--maximum-entropy-prior) for the mathematics.

---

## The Mathematics

All EV values are in units of the **original bet**: $+1$ = win, $-1$ = lose, $0$ = push.

---

### 1. Shoe Composition and Card Probabilities

A shoe of $N$ decks contains $52N$ cards. Face cards J, Q, K are grouped with T
(all worth 10), giving 10 canonical ranks:

| Rank        | Count per deck | Value   |
| ----------- | -------------- | ------- |
| 2 to 9      | 4 each         | face    |
| T (T/J/Q/K) | 16             | 10      |
| A           | 4              | 1 or 11 |

Let $n_r$ be the count of rank $r$ remaining and $M = \sum_r n_r$ the total
remaining. The probability of drawing rank $r$ next is:

$$
P(\text{next} = r \mid \mathcal{S}) = \frac{n_r}{M}
$$

This is **sampling without replacement** — strictly more accurate than the
infinite-deck approximation because it tracks how each removal shifts the
distribution.

**Code:** `Shoe.prob(rank)` in `shoe.py`

---

### 2. Hand Evaluation

A hand $H = (c_1, \ldots, c_k)$ has a **best total** via greedy ace reduction.
Let $s$ be the sum with every ace counted as 11, and let $a$ be the number of
aces:

$$
T(H)
====

s - 10k,
\qquad
k =
\min!\left(
a,,
\max!\left(
0,,
\left\lceil\frac{s-21}{10}\right\rceil
\right)
\right).
$$

The hand is **soft** when at least one ace remains at 11 and $T(H) \leq 21$:

$$
\operatorname{soft}(H)
======================

\mathbf{1}!\left[
a-k \geq 1
;\wedge;
T(H) \leq 21
\right].
$$

| Hand    | Total | Soft? |
| ------- | ----- | ----- |
| A, 6    | 17    | yes   |
| A, 6, 9 | 16    | no    |
| A, A    | 12    | yes   |
| A, A, 9 | 21    | yes   |

A **natural blackjack** has $|H|=2$ and $T(H)=21$.

**Code:** `hand_total(cards)` in `hand.py`

---

### 3. Dealer Outcome Distribution

The dealer hits while $T < 17$, or while $T = 17$ and the hand is soft under
H17 rules.

Let $\mathcal{S}^{-}$ be the shoe after removing the upcard $u$ and all visible
player cards. The probability of each dealer final total is:

$$
P(D=d \mid u,\mathcal{S})
=========================

\sum_{\sigma \in \Sigma(u,d)}
\prod_{i=1}^{|\sigma|}
\frac{
n_{\sigma_i}
!\left(
\mathcal{S}^{-}\setminus\sigma_{1:i-1}
\right)
}{
M
!\left(
\mathcal{S}^{-}\setminus\sigma_{1:i-1}
\right)
}
$$

where $\Sigma(u,d)$ is the set of all draw sequences from upcard $u$ that end
at total $d$.

**Dealer peek conditioning** (when `dealer_peeks = True`): if the dealer checked
for blackjack and did not have it, the hole card cannot be the BJ-completing rank.
Let $f$ be the forbidden hole rank ($f=T$ when $u=A$; $f=A$ when $u=T$).
The distribution is conditioned on this by summing only over hole cards $h \neq f$
and renormalising:

$$
P(D=d \mid u,\mathcal{S},\neg\mathrm{BJ})
=========================================

\frac{
\displaystyle
\sum_{h\neq f}
\frac{n_h}{M}
,
P!\left(
D=d
\mid
[u,h],
\mathcal{S}^{-}\setminus{h}
\right)
}{
\displaystyle
\sum_{h\neq f}
\frac{n_h}{M}
}.
$$

The conditioning applies **only to the first draw** (the hole card). All
subsequent draws use the full depleted shoe. Bust is encoded as $d=22$.

$$
\sum_{d\in{17,18,19,20,21,22}} P(D=d)=1
$$

**Code:** `dealer_distribution(upcard, shoe, rules, player_cards)` in `ev.py`

---

### 4. Expected Value of Standing

$$
\operatorname{EV}_{\text{stand}}(P,D)
=====================================

\sum_d P(D=d),\omega(P,d)
$$

$$
\omega(P,d)
===========

\begin{cases}
+\lambda,
& d=22
\quad\text{(dealer bust)},\
+\lambda,
& P=\mathrm{BJ},; d\neq21,\
+\lambda,
& P=\mathrm{BJ},; d=21,;
\text{natural beats dealer 21},\
0,
& P=\mathrm{BJ},; d=21,;
\text{natural does not beat dealer 21},\
+1,
& P>d,\
-1,
& P<d,\
0,
& P=d.
\end{cases}
$$

$\lambda$ = `blackjack_payout` (e.g. $1.5$ for 3:2) for naturals;
$\lambda=1$ otherwise.

**Code:** `_stand_ev(player_total, dealer_dist, is_blackjack, rules)` in `ev.py`

---

### 5. Expected Value of Hitting

Hitting is solved by **backward induction** over the probability tree:

$$
\operatorname{EV}_{\text{hit}}(H,\mathcal{C})
=============================================

\sum_{r:,n_r>0}
\frac{n_r}{M(\mathcal{C})}
\begin{cases}
-1,
&
T(H\cup{r})>21,
[4pt]
\displaystyle
\max!\Bigl(
\operatorname{EV}*{\text{stand}}
\bigl(T(H\cup{r})\bigr),
\operatorname{EV}*{\text{hit}}
\bigl(H\cup{r},\mathcal{C}\setminus{r}\bigr)
\Bigr),
&
\text{otherwise}.
\end{cases}
$$

Each branch depletes the shoe by $r$ before recursing, so no card is
sampled twice. The $\max(\text{stand},\text{hit})$ at every node is exact
backward induction — the player always takes the better option.

**Code:** `_hit_ev(player_cards, dealer_dist, counts, rules)` in `ev.py`

---

### 6. Expected Value of Doubling

Draw exactly one card, then stand. Both bets are at risk, so the stake is
$2\times$:

$$
\operatorname{EV}_{\text{double}}(H,\mathcal{C})
================================================

\sum_{r:,n_r>0}
\frac{n_r}{M(\mathcal{C})}
\begin{cases}
-2,
&
T(H\cup{r})>21,
[4pt]
2\cdot
\operatorname{EV}_{\text{stand}}
\bigl(T(H\cup{r})\bigr),
&
\text{otherwise}.
\end{cases}
$$

After the double card is drawn, `hand.doubled = True` and the action gater
locks all further actions to **stand only**.

**Code:** double branch inside `action_evs` in `ev.py`

---

### 7. Expected Value of Splitting

After splitting a pair of rank $r$, both copies leave the shoe:

$$
\mathcal{C}^{(r)}
=================

\mathcal{C}\setminus{r,r}
$$

Each child hand draws one second card $c$, producing hand $[r,c]$ with shoe
$\mathcal{C}^{(r)}\setminus{c}$. The combined EV of both child hands is:

$$
\operatorname{EV}_{\text{split}}(r,\mathcal{C})
===============================================

2
\sum_{c:,n_c^{(r)}>0}
\frac{n_c^{(r)}}{M(\mathcal{C}^{(r)})}
\max_{a\in\mathcal{A}([r,c])}
\operatorname{EV}_a
!\left(
[r,c],
\mathcal{C}^{(r)}\setminus{c}
\right)
$$

The factor 2 reflects both hands playing for the same stake. The $\max$ recurses
into `action_evs` — allowing further splits, doubles, hits, and stands.
Each child is treated as an independent draw from $\mathcal{C}^{(r)}$ (standard
industry approximation; error is negligible for multi-deck shoes).

**Ace-split restriction:** when `split_aces_get_one_card = True`:

$$
\mathcal{A}([A,c])={\text{stand}}
$$

**Code:** `split_ev(hand, dealer_upcard, shoe, rules, splits_used)` in `ev.py`

---

### 8. Surrender

$$
\operatorname{EV}_{\text{surrender}}=-\frac{1}{2}
$$

Surrender is optimal when:

$$
-\frac{1}{2}

>

\max!\left(
\operatorname{EV}*{\text{stand}},
\operatorname{EV}*{\text{hit}},
\operatorname{EV}*{\text{double}},
\operatorname{EV}*{\text{split}}
\right)
$$

Classic example: hard 16 vs. dealer T where
$\operatorname{EV}_{\text{stand}}\approx-0.54$.

**Code:** `result["surrender"] = -0.5` in `ev.py`

---

### 9. Insurance

When the dealer shows an A, insurance pays 2:1 if the hole card is T.
The EV is computed from the **exact remaining shoe** after removing the upcard:

$$
\operatorname{EV}_{\text{insurance}}
====================================

2\cdot
P(\text{hole}=T\mid\mathcal{S}\setminus{A})
-1
==

\frac{2n_T}{M-1}-1
$$

Insurance is $+\text{EV}$ when
$P(\text{hole}=T)>\frac{1}{3}$, which corresponds to a true count of
approximately $+3$.

On a fresh 8-deck shoe:
$P(T)=128/415\approx0.308$, giving
$\operatorname{EV}\approx-0.077$ (house edge).

**Code:** `insurance_ev(shoe, dealer_upcard)` in `ev.py`

---

### 10. Optimal Action Selection

$$
a^{*}
=====

\arg\max_{a\in\mathcal{A}(H)}
\operatorname{EV}_a(H,\mathcal{C})
$$

The **delta** for each suboptimal action $a\neq a^{*}$:

$$
\Delta_a
========

## \operatorname{EV}_a

\operatorname{EV}^{*}
\leq0
$$

Insurance is excluded from this comparison — it is a separate side-bet decision.

**Code:** `best_action(ev_dict)` in `ev.py`

---

### 11. Card Counting — Hi-Lo True Count

$$
\Delta\mathrm{RC}(r)
====================

\begin{cases}
+1,
& r\in{2,3,4,5,6},\
0,
& r\in{7,8,9},\
-1,
& r\in{T,A}.
\end{cases}
$$

$$
\mathrm{TC}
===========

\frac{\mathrm{RC}}{M/52}
$$

A true count of $+1\approx+0.5%$ player EV. The EV engine uses the **exact
shoe composition** ${n_r}$, which strictly subsumes all count information.
The true count is displayed as a human-readable summary only.

**Code:** `Shoe.true_count` in `shoe.py`

---

### 12. Penetration-Dampened EV

When few cards have been observed, count-based EV deviations from basic strategy
are unreliable. Let $\rho=|\mathcal{O}|/(52N)$ be the observation ratio. The
dampened EV blends the count-adjusted value toward the basic-strategy baseline
using a **square-root schedule** (more conservative than linear at low
penetration):

$$
\operatorname{EV}_{\text{damp}}
===============================

\begin{cases}
\operatorname{EV}*{\text{basic}},
&
\rho<\rho*{\min},
[6pt]
\displaystyle
\operatorname{EV}*{\text{basic}}
+
\sqrt{
\frac{\rho-\rho*{\min}}{1-\rho_{\min}}
}
\left(
\operatorname{EV}_{\text{raw}}
------------------------------

\operatorname{EV}*{\text{basic}}
\right),
&
\rho*{\min}\leq\rho<1,
[6pt]
\operatorname{EV}_{\text{raw}},
&
\rho\geq1.
\end{cases}
$$

Default $\rho_{\min}=0.15$. The square-root schedule is better calibrated than
linear because count information grows faster than linearly with penetration.

| $\rho$           | EV quality                                   |
| ---------------- | -------------------------------------------- |
| $<0.10$          | Near basic strategy; count signal unreliable |
| $0.10$ to $0.25$ | Mild count signal; treat with caution        |
| $0.25$ to $0.60$ | Count-adjusted EVs meaningful                |
| $>0.60$          | High-confidence shoe-aware EVs               |

**Code:** `dampened_ev(raw_ev, basic_strategy_ev_val, observation_ratio)` in `ev.py`

---

### 13. Kelly Criterion Bet Sizing

Given the EV of the best action and the variance of blackjack outcomes
($\sigma^2\approx1.15$ for standard rules), the **Kelly fraction** of bankroll
to wager is:

$$
f^{*}
=====

\frac{\operatorname{EV}}{\sigma^2}
$$

The engine uses **half-Kelly** by default
($f=0.5\cdot f^{*}$) to reduce variance while retaining approximately 75% of
the long-run growth rate:

$$
f_{\text{half}}
===============

\frac{\operatorname{EV}}{2\sigma^2}
$$

The recommended bet is clamped to
$[\text{min_bet},,\text{max_bet}]$ and rounded to the nearest min-bet
increment. When $\operatorname{EV}\leq0$, bet the table minimum.

**Code:** `kelly_fraction`, `recommended_bet`, `kelly_summary` in `kelly.py`

---

### 14. Mid-Shoe Join — Maximum-Entropy Prior

Let $\mathcal{S}_0$ be the full fresh shoe and $\mathcal{O}$ the cards observed
since joining:

$$
\hat{\mathcal{S}}
=================

\mathcal{S}_0\setminus\mathcal{O}
$$

$$
\hat{P}(\text{next}=r)
======================

\frac{
n_r^{(0)}-n_r^{(\mathcal{O})}
}{
52N-|\mathcal{O}|
}
$$

Cards dealt before arrival are **not** removed — they remain in the model. This
is the **maximum-entropy prior**: a uniform residual distribution is the
least-biased estimate in the absence of prior information. As
$|\mathcal{O}|\to52N$, the estimate converges to the true posterior.

The observation ratio:

$$
\rho
====

\frac{|\mathcal{O}|}{52N}
\in[0,1]
$$

**Code:** `ShoeState.observation_ratio` in `shoe_state.py`

---

## Rule Variants and EV Impact

| Rule flag                       | Effect on house edge   |
| ------------------------------- | ---------------------- |
| `dealer_hits_soft17` H17 vs S17 | +0.22%                 |
| `double_after_split` DAS        | −0.14%                 |
| `resplit_aces` RSA              | −0.08%                 |
| `max_splits` 1 to 4             | −0.05% per extra split |
| `blackjack_payout` 3:2 to 6:5   | +1.37% (large)         |
| `surrender = late`              | −0.07%                 |
| `surrender = early`             | −0.24%                 |
| `dealer_peeks = False`          | +0.03% to house        |
| `insurance = False`             | removes side bet       |

Pre-built rule sets: `STRIP_S17`, `DOWNTOWN_H17`, `LIBERAL`, `TOURIST_TRAP`

---

## Module Overview

```text
blackjack/
├── rules.py       RuleSet — all rule flags
├── hand.py        hand_total, is_soft, is_blackjack, can_split
├── shoe.py        Shoe — counts, remove, running_count, true_count
├── shoe_state.py  ShoeState — mid-shoe join, observation_ratio
├── actions.py     get_legal_actions — doubled-hand lock, post-split-ace lock
├── ev.py          dealer_distribution (peek-conditioned), action_evs,
│                  split_ev, insurance_ev, dampened_ev, basic_strategy_ev
├── kelly.py       kelly_fraction, recommended_bet, kelly_summary
├── side_bets.py   perfect_pairs_ev, twenty_one_plus_three_ev
├── detector.py    TemplateDetector (OpenCV) + YOLODetector (YOLO)
├── capture.py     CaptureSession — screen grab → detect → ingest loop
└── ui_state.py    AppState — kelly, insurance, side bets, dampened EVs
```

---

## Project Layout

```text
test_space_for_projects/
├── app.py
├── blackjack/
├── tests/
├── configs/
│   ├── app.yaml
│   └── vision.yaml
├── data/
│   ├── templates/
│   ├── models/
│   └── observations.csv
├── notebooks/
├── experiments/
├── reports/
└── pyproject.toml
```

---

## Compliance Note

Screen capture of a live platform may violate its Terms of Service.
This codebase is for **research and educational purposes only**.
