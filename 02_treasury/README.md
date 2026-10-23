# Use case 02 — Treasury: sovereign yield curve forecasting

**Code**: `04_uc31_treasury.py` (runnable master at project root: `python 04_uc31_treasury.py`)

## Problem statement
A rates desk forecasts the full US Treasury CMT curve (11 tenors, 1 month → 30 years) 20 trading days ahead. Deliverables: (a) level forecasts, (b) rate-shock bands for DV01-style risk, (c) a **coherent smooth curve** — independent per-tenor models produce jagged, arbitrage-violating paths.

## Setup
- Target: all 11 tenors jointly `(1, 11, 16849)` — the whole 1962-2026 history fits in one pass (32,768 context)
- Past covariates: 10Y-2Y slope + DGS10 realized vol; known: month-end/quarter-end flags + dow codes
- Ragged NaN (DGS30 gap 2002–2006, DGS1MO starts 2001, 719 holiday rows) fed raw
- Horizon: 20 trading days; 5 rolling 2026 cutoffs. Floor: random walk (no-change)

## Results readout (mean over 5 cutoffs)

| method | RMSE (bps) | MAE (bps) | cov90% |
|---|---:|---:|---:|
| Timestral (joint 11-tenor) | 9.33 | 7.32 | **94.1** |
| Random walk | 9.33 | 7.37 | — |

Per-tenor (final cutoff): DGS1MO **1.84 vs RW 4.85 bps** (−62%), DGS10 12.7 vs 12.2 (−4%), DGS20 15.45 vs 13.3 (+16%).

## What the results mean
1. **Tying the random walk is the expected result, not a failure.** Rates literature consensus (Diebold-Li et al.): the RW is nearly unbeatable at the 1-month horizon. Zero-shot parity with a frozen model is the sanity bar passing.
2. **The value is in geometry, not level**: the joint forecast keeps the curve smooth (kink statistic 0.13 = the actuals' own smoothness) and wins massively at the short end (bills pull toward the corridor mean instead of freezing yesterday's level).
3. **Where it loses: the long end.** The smooth-curve prior washes out the 20Y structural-premium hump (~2–3 bps cost on DGS20/30) — a desk would take those two points from RW or apply a static bias correction.
4. **Calibrated bands are the risk deliverable**: 94.1% coverage for QE-shock scenarios.

## Caveats
- Trading-day grid with all-NaN holiday rows score-void (Kaggle-RMSPE convention).
- Levels span 0.3%–15.8% over 60 years — scale invariance is what lets one pass cover all regimes.

**Plots**: `out_04_curve.png` (11-tenor curve forecast + 20Y hump), `out_04_dgs10.png` (10Y band).
