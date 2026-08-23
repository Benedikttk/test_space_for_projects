# Blackjack Probability, Shoe Tracking, and Statistical Validation Engine

This repository contains a research-grade Python system for:

- event-sourced blackjack observation logging
- exact shoe composition tracking
- no-look-ahead decision snapshots
- analytical EV calculations (modular foundation)
- Monte Carlo validation
- red-card/cut-position empirical modeling
- strategy evaluation with statistical inference
- optional vision ingestion (screen capture + card detection pipeline scaffolding)

## Scientific and methodological guardrails

- **No look-ahead bias:** hidden dealer hole card and future draws are excluded from decision features.
- **Dependence-aware inference:** report both naive and shoe-cluster/block-bootstrap uncertainty where possible.
- **Paired strategy comparison:** shared scenarios support lower-variance A/B estimates.
- **Empirical cut-card model:** no forced normality assumptions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
pytest
```

Run a short screen-capture session:

```bash
bj capture --seconds 5
```

Run a small blackjack simulation:

```bash
bj simulate --hands 1000 --seed 42
```

## Project layout

```text
blackjack_model/
├── configs/
├── data/
├── blackjack/
├── tests/
├── notebooks/
├── reports/
└── experiments/
```

## Current status

- Phase 0: Vision ingestion scaffolding ✅
- Phase 1: Cards + shoe representation ✅
- Phase 2: Hand evaluation ✅
- Phase 3-8: Dealer/probability/EV/MC validation baseline ✅ (extensible)
- Phase 9+: Logging/red-card/strategy comparison/statistics baseline ✅ (extensible)

## Important compliance note

Before using live screen capture on a real platform, verify platform ToS and legal constraints.
This code is intended for **research/testing**.
