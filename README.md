# Blackjack Probability, Shoe Tracking & EV Engine

A research-grade Python system that watches a live blackjack table via screen
capture, classifies cards using computer vision, tracks the shoe composition in
real time, and computes the mathematically optimal action and its exact expected
value at every decision point.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How Data is Captured](#how-data-is-captured)
3. [Mid-Shoe Join — Joining a Live Game](#mid-shoe-join)
4. [The Mathematics](#the-mathematics)
   - [Shoe & Card Probabilities](#shoe--card-probabilities)
   - [Hand Evaluation](#hand-evaluation)
   - [Dealer Outcome Distribution](#dealer-outcome-distribution)
   - [Expected Value of Standing](#expected-value-of-standing)
   - [Expected Value of Hitting](#expected-value-of-hitting)
   - [Expected Value of Doubling](#expected-value-of-doubling)
   - [Expected Value of Splitting (Recursive)](#expected-value-of-splitting-recursive)
   - [Surrender](#surrender)
   - [Card Counting — Hi-Lo True Count](#card-counting--hi-lo-true-count)
5. [Rule Variants](#rule-variants)
6. [Module Overview](#module-overview)
7. [Project Layout](#project-layout)
8. [Compliance Note](#compliance-note)

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]            # core + pytest + ruff + mypy
# optional GPU/YOLO detector:
pip install -e .[vision_dl]
pytest
```

Run the Streamlit UI (manual card entry, no camera required):

```bash
streamlit run app.py
```

Start a live screen-capture session:

```bash
python -m blackjack.capture      # uses configs/app.yaml + vision.yaml
```

---

## How Data is Captured

The system has two input modes that can be mixed freely:

### Mode 1 — Screen-capture + ML detection (automated)

```
Monitor pixels  (mss, 2 FPS default)
      │
      ▼  crop configured screen regions
 ┌─────────────────────────────────────────────────────┐
 │  dealer_region   player_region   seat_1 … seat_N    │  ← vision.yaml
 └─────────────────────────────────────────────────────┘
      │
      ▼  CardDetector (TemplateDetector or YOLODetector)
 DetectionResult(rank, suit, confidence)
      │
      ├── confidence ≥ 0.85  →  auto-accepted, removed from shoe
      ├── 0.75–0.84           →  flagged for manual review in UI
      └── < 0.75              →  rejected, not used
      │
      ▼  ShoeState.ingest()
 Shoe.remove(rank)   →   rank counts updated   →   running_count updated
      │
      ▼
 action_evs(hand, upcard, shoe, rules)   →   recommendation + EV table
```

**TemplateDetector** (zero GPU, default): loads `<rank>_<suit>.png` reference
images from `data/templates/`, runs multi-scale OpenCV `matchTemplate` at each
configured screen region, and returns the best-matching card with its
normalised correlation score as the confidence.

**YOLODetector** (GPU optional): loads a `.pt` YOLO model from
`data/models/cards.pt`, runs inference on each cropped region, and maps the
predicted class label (format `RANK_SUIT`, e.g. `A_S`, `T_H`) to a
DetectionResult.  Install with `pip install -e .[vision_dl]`.

Cards from **every seat** (other players + dealer) are removed from the shoe
because they reduce the remaining composition — this is the information that
drives count-adjusted EV calculations.  Only **your own cards and the dealer
upcard** are used for the EV computation itself.

### Mode 2 — Manual card entry (always available)

Type each card rank/suit into the UI.  The system calls
`ShoeState.ingest_manual(rank)` which removes the card from the shoe with
`confidence=1.0`.  Useful when the camera angle is poor or for verifying
decisions at a physical table.

---

## Mid-Shoe Join

When you sit down at a table already in progress, cards have been dealt that
you never saw.  The system handles this explicitly:

```python
shoe_state = ShoeState(decks=8, mid_shoe_join=True)
# or at runtime:
session.shoe_state.join_mid_shoe()
```

**What this means mathematically:**

- The shoe is initialised at full capacity (8 × 52 = 416 cards).
- Every card you observe is removed normally.
- Cards dealt *before you arrived* are **not** removed — they remain in the
  model as if still in the shoe.  This is the **maximum-entropy prior**: in the
  absence of information about which ranks were dealt early, assuming a uniform
  residual distribution is the least-biased estimate.
- The `observation_ratio` property tracks `cards_seen / starting_cards`.
  At 0 % you are flying blind; at 100 % you have full information.

**Practical consequence for EV:**

The EV calculation is always *exact for the information you actually have*.
What changes is how much that information deviates from a fresh shoe:

| observation_ratio | EV quality |
|---|---|
| < 10 % | Near-basic-strategy; count adjustments are unreliable |
| 10–25 % | Mild count signal; treat recommendations with caution |
| 25–60 % | Count-adjusted EVs are meaningful |
| > 60 % | High-confidence shoe-aware EVs |

The UI always displays the `uncertainty_label` so you know which regime you
are in.

---

## The Mathematics

### Shoe & Card Probabilities

A shoe of **N decks** contains N × 52 cards.  Face cards (J, Q, K) are grouped
with Tens because they all have value 10, giving 10 canonical ranks:

| Rank | Count per deck | Value |
|------|---------------|-------|
| 2–9  | 4 each        | face  |
| T (T/J/Q/K) | 16      | 10    |
| A    | 4             | 1 or 11 |

After observing a set of cards **C** from the shoe, the probability that the
next card drawn is rank **r** is:

```
P(next = r | C) = count(r, remaining) / total_remaining
```

This is **sampling without replacement** — the canonical model for a finite
shoe.  It is strictly more accurate than the infinite-deck approximation
(sampling with replacement) because it correctly captures how probabilities
shift as the shoe depletes.

---

### Hand Evaluation

A hand's **best total** is computed by the greedy ace-reduction algorithm:

1. Sum all cards, counting each Ace as 11.
2. While `total > 21` and there are aces counted as 11, subtract 10
   (re-count one Ace as 1).
3. The hand is **soft** if at least one Ace remains counted as 11 and
   `total ≤ 21`.

Examples:
- `[A, 6]` → 17, soft (Ace = 11)
- `[A, 6, 9]` → 16, hard (Ace reduced to 1)
- `[A, A]` → 12, soft (one Ace = 11, one = 1)
- `[A, A, 9]` → 21, soft (one Ace = 11, one = 1, 9)

A **natural blackjack** is exactly 2 cards totalling 21.

---

### Dealer Outcome Distribution

The dealer follows a fixed rule: **hit until total ≥ 17** (or ≥ 18 under H17
when the hand is soft-17).  Given a visible upcard **u** and current shoe
counts **S**, the function `dealer_distribution(u, S, rules)` computes the
exact probability of each dealer final total by **recursive probability
weighting**:

```
P(dealer_final = d | u, S) = Σ over all draw sequences
     P(draw sequence) × 1[sequence leads to final total d]
```

Concretely, at each step where the dealer must hit, we branch over all
remaining ranks weighted by their shoe probability, deplete the count for that
branch, and recurse.  The shoe is **never mutated**; each branch operates on
its own local count copy.

All visible cards — the player's cards **and** the dealer upcard — are removed
from the snapshot before the recursion begins.  This correctly conditions the
dealer probabilities on everything that has already been observed.

Bust is encoded as total = 22 for uniform comparison arithmetic.

---

### Expected Value of Standing

EV is measured in units of the **original bet** (+1 = win one bet, −1 = lose
one bet, 0 = push).

```
EV(stand | player_total P, dealer_dist D) =
    Σ_d  P(dealer = d) × outcome(P, d)

where outcome(P, d) =
    +payout   if d = 22 (dealer bust)
    +payout   if P = blackjack and d ≠ 21
    +payout   if P = blackjack and d = 21 and natural_beats_dealer_21
     0        if P = blackjack and d = 21 and not natural_beats_dealer_21  (push)
    +1        if P > d  (player wins)
    -1        if P < d  (dealer wins)
     0        if P = d  (push)
```

`payout` = 1.0 for non-blackjack wins, `blackjack_payout` (e.g. 1.5 for 3:2)
for naturals.

---

### Expected Value of Hitting

Hitting is a **decision under uncertainty with future choices**.  The player
draws a card and then plays optimally from the new state.  This is solved by
**backward induction** (dynamic programming over the probability tree):

```
EV(hit | cards H, shoe S) =
    Σ_r  P(next = r | S)
       × [ bust(H+r) ? −1
                      : max( EV(stand | H+r), EV(hit | H+r, S−r) ) ]
```

Key properties:
- The shoe is depleted by rank **r** before recursing (`S−r`), so each branch
  conditions on all cards drawn so far.
- `max(stand, hit)` at each depth means the player always chooses the
  better option after the next card.
- Recursion terminates when the player busts or reaches a depth limit (safety
  valve at depth 10, effectively unreachable for non-degenerate shoes).

---

### Expected Value of Doubling

Double down: place an additional equal bet, draw **exactly one more card**,
then stand.  All wins/losses are at 2× the original stake:

```
EV(double | cards H, shoe S) =
    Σ_r  P(next = r | S)
       × [ bust(H+r) ? −2.0
                      : 2.0 × EV(stand | H+r) ]
```

The factor of 2 appears because both the original bet and the double bet are at
risk.  The shoe is depleted per branch for the drawn card.

---

### Expected Value of Splitting (Recursive)

Splitting is the most complex action because it creates new hands that can
themselves be split again (up to `max_splits` times).

**Setup:**
After splitting a pair of rank **r**, both copies of **r** leave the shoe.
Each child hand starts with one **r** and draws one additional card.

```
EV(split | pair r, shoe S, rules) =
    2 × Σ_c  P(next = c | S − 2r)
           × max_action EV(child hand [r, c], S − 2r − c, rules)
```

Where `S − 2r` means the shoe with both split cards removed, and `S − 2r − c`
means the shoe with both split cards **and** the drawn second card **c** removed
before evaluating the child hand.  The factor of 2 reflects that both hands
are played for the same stake.

The `max_action EV` recurses into `action_evs` for each child hand, which
can itself include further splits (if `resplit_aces` is on and `splits_used <
max_splits`), doubles (if `double_after_split` is on), hits and stands.

**Ace-split restriction:** when `split_aces_get_one_card = True`, each
post-split-ace hand receives exactly one additional card and may not hit,
double, or re-split.  This is implemented in the action gater
(`get_legal_actions`) which locks the hand after 2 cards when
`is_post_split_ace = True`.

**Single responsibility for shoe depletion:** the dealer upcard and player
cards are removed from the shoe snapshot exclusively inside
`dealer_distribution` and `action_evs`.  `split_ev` only removes the two
split-pair cards.  This prevents double-depletion across the recursion boundary.

---

### Surrender

Late surrender: the player forfeits **half the bet** before playing.

```
EV(surrender) = −0.5   (always, by definition)
```

Surrender is optimal whenever `max(stand, hit, double, split) < −0.5`.
The most common case is hard 16 vs. dealer 10 under standard rules.

Early surrender (rare): permitted before the dealer checks for blackjack,
which gives it a slightly higher EV in some spots.  In this engine both
modes return −0.5; the difference is captured only in *when* the option
is offered (early surrender can be taken even if the dealer has blackjack,
which `action_evs` gates via the `surrender` mode flag).

---

### Card Counting — Hi-Lo True Count

The **running count** is updated every time a card is observed:

```
+1  for ranks 2, 3, 4, 5, 6   (low cards favour the dealer)
 0  for ranks 7, 8, 9          (neutral)
−1  for ranks T, A             (high cards favour the player)
```

The **true count** normalises for the number of decks remaining:

```
true_count = running_count / decks_remaining
           = running_count / (cards_remaining / 52)
```

A true count of +1 corresponds to roughly **+0.5% player EV** relative to
the neutral shoe.  The EV engine does not use the true count directly —
it uses the **exact shoe composition** which subsumes all count information
and is strictly more precise.  The true count is displayed in the UI as a
human-readable summary.

---

## Rule Variants

All rule variants are encoded in the `RuleSet` dataclass and affect every
calculation automatically:

| Flag | Effect on EV |
|------|-------------|
| `dealer_hits_soft17` (H17) | +0.22% house edge vs S17 |
| `double_after_split` (DAS) | +0.14% player |
| `resplit_aces` (RSA) | +0.08% player |
| `max_splits` 1→4 | +0.05% player per additional split allowed |
| `blackjack_payout` 3:2→6:5 | −1.37% player (large) |
| `surrender = late` | +0.07% player |
| `surrender = early` | +0.24% player |

Pre-built canonical rule sets: `STRIP_S17`, `DOWNTOWN_H17`, `LIBERAL`,
`TOURIST_TRAP`.

---

## Module Overview

```
blackjack/
├── rules.py        RuleSet dataclass — single source of truth for all rule flags
├── hand.py         Hand evaluation: hand_total, is_soft, is_blackjack, can_split
├── shoe.py         Shoe: card counts, removal, running count, true count
├── shoe_state.py   ShoeState: wraps Shoe with mid-shoe-join uncertainty tracking
├── actions.py      LegalActions gater — pure function of (hand, splits_used, rules)
├── ev.py           EV engine: dealer_distribution, action_evs, split_ev, best_action
├── detector.py     CardDetector: TemplateDetector (OpenCV) + YOLODetector (YOLO)
├── capture.py      CaptureSession: screen grab → detect → ingest → callback loop
└── ui_state.py     AppState: session state + EV/history adapters for Streamlit
```

---

## Project Layout

```
test_space_for_projects/
├── blackjack/          Python package (see above)
├── configs/
│   ├── app.yaml        FPS, monitor, confidence thresholds, simulation defaults
│   └── vision.yaml     Screen region coordinates per seat + detector settings
├── data/
│   ├── templates/      Card template images for TemplateDetector (PNG files)
│   ├── models/         YOLO weights (cards.pt) for YOLODetector
│   ├── raw_frames/     Saved screen grabs (optional, app.save_parquet)
│   └── observations.csv  Event log of all accepted card detections
├── tests/              pytest test suite
├── notebooks/          Jupyter exploration notebooks
├── experiments/        Scratch experiments
├── reports/            Generated analysis reports
└── pyproject.toml      Dependencies and tool configuration
```

---

## Compliance Note

Screen capture of a live platform may violate that platform's Terms of Service.
This codebase is intended for **research and educational purposes only**.
Always verify legal constraints before use in any live environment.
