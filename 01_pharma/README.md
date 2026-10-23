# Use case 01 — Pharma: cross-class drug cannibalization & stock targets

**Code**: `03_uc11_pharma.py` (runnable master stays at project root: `python 03_uc11_pharma.py` from `timestral-poc/`)

## Problem statement
A single-pharmacy daily sales dataset (8 ATC drug classes: analgesics, antipyretics, psycholeptics, asthma and allergy meds). The planner needs, four weeks ahead: (a) per-class demand forecasts where classes can substitute for each other / trade off across seasons (cannibalization), and (b) order-up-to stock levels at a 90% service level.

## Setup
- Target: 8 classes jointly `(1, 8, 2106)` daily (explicit 3-D = joint multivariate)
- Known covariates: day-of-week, month (categorical codes), is-weekend
- Horizon: 28 days; 5 rolling cutoffs (May–Sep 2019)
- Floor: SeasonalNaive(lag-7), computed locally

## Results readout (mean over 5 cutoffs)

| method | MASE | sMAPE% | cov90% |
|---|---:|---:|---:|
| Timestral-209M (joint + cal covs) | **0.672** | 66.6 | 92.2 |
| SeasonalNaive(7) | 1.009 | 74.2 | — |

Per-class MASE range 0.55–0.82; the deliverable table is the q90 28-day stock target (e.g. N02BE: median 827 vs q90 ~1,350 units).

## What the results mean
1. **Zero-shot beats the seasonal floor on all 8 classes** (−33% MASE worst-case → best are well-observed classes). The weakest class (N05C, near-zero burst counts, sMAPE 177%) is a small-denominator artifact, not a failure.
2. **Quantile bands arrive calibrated for free** (92.2% coverage vs 90% nominal) — usable directly as service-level safety stock, which is the planner's actual money question.
3. **Cannibalization structure is seasonal, not transactional**: antipyretic demand +0.34 with asthma medication (viral waves climb together) and −0.21 with antihistamines (viral vs allergy seasons trade off). Only a joint forecast makes that portfolio effect visible; substitution within NSAID pairs washes out at daily grain.
4. **Scope limit**: single-day spikes (asthma class bursts of 25–31 units) sit far outside the band — the median forecast can't see event-day bursts; use q90 for those classes.

## Caveats
- Single pharmacy, small daily counts, high variance → MASE (scale-free) is the primary metric.
- The PLAN's "12–18% safety-stock cut" claim is only directionally supported; the honest deliverable is the per-class q90 order-up-to table.

**Plots**: `out_03_forecast.png` (8-class forecast, final window), `out_03_comovement.png` (cross-class correlation structure).
