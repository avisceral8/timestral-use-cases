# Use case 08 — Beijing air quality: 12-station PM2.5 day-ahead nowcast & alerts

**Code**: `10_uc_air.py` (runnable master at project root: `python 10_uc_air.py`)

## Problem statement
Beijing's 12 monitoring stations hourly PM2.5 (2013-03 → 2017-02, 35,064 h). Two jobs: (a) day-ahead (24 h) city-wide nowcast with the urban/suburban coupling, (b) a **decision deliverable**: a day-ahead air-alert rule with SEVIRI-class threshold 150 µg/m³ — alert if the predicted daily peak exceeds the threshold. Winter heating season (Nov-15 → Mar-15) is the smog driver.

## Setup
- Targets: PM2.5 across 12 stations jointly `(12, 35064)` (T > 32,768 ⇒ sliding 1-year context)
- Past covariates: PM10, TEMP, WSPM (raw NaN); known: heating-season flag + hour/month codes
- H=24; 12 rolling daily cutoffs (Feb-2017 tail) + the 2015-12-19 red-alert smog vignette
- Floor: SeasonalNaive(lag-24h)

## Results readout (12 daily cutoffs)

| metric | mean |
|---|---:|
| MAE | **26.6** µg/m³ (naive 57.2, −54%) |
| MASE | **0.46** (naive 1.0 by construction) |
| cov90% | 90.0 |

Alert confusion: cell-q90 rule TP 3 / FP 9 / FN 0; **corrected median-peak rule TP 3 / FP 4 / FN 0**. Smog vignette: MAE 40.1 vs naive 75.3, alert fired (q90 peak 403 vs actual 251).

## What the results mean
1. **Decisive win over the same-day-repeat rule** — the joint station tensor + air-mass covariates + heating calendar carry the ventilation-driven rhythm.
2. **Union-bound foot-gun (the case's key lesson)**: max over 288 station-hour cells of per-cell q90 is NOT the quantile of a max — it fires in ~every window. Day-ahead alerting needs maximum-distribution semantics; the corrected median-peak rule keeps recall 100% while cutting false alarms 2.3×.
3. **The smog-episode vignette under-calls the level but gets the severity class right** — the decision-relevant outcome for alerting.
4. **Regime-break signature (again)**: the worst window is a wind-clearing collapse with no covariate; recovers with fresh context.

## Caveats
- Sliding 1-year context means each cutoff sees exactly one winter (no multi-year amplitude learning).
- No standardized leaderboard for this dataset; the local seasonal-naive floor is the anchor.

**Plots**: `out_10_air.png` (4-station day-ahead nowcast), `out_10_alert.png` (predicted vs actual peak alert map).
