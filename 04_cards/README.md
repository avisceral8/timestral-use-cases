# Use case 04 — Credit cards: sector spend nowcast (COVID stress test)

**Code**: `06_uc32_cards.py` (runnable master at project root: `python 06_uc32_cards.py`)

## Problem statement
A card issuer / merchant-payments team needs a **nowcast of consumer spend by 14 MCC sectors** (grocery, gas, travel, e-commerce, …) 30 days ahead — for liquidity forecasting and sector exposure. The entire evaluation window lands inside the COVID first wave, the largest structural break in the data — which makes this the regime-break stress test.

## Setup
- Target: 14 sector daily totals jointly `(1, 14, 538)`; past covariates: mean ticket size + active cards; known: payday (1st/15th) + Q4 flags + dow/month codes
- H=30 days; cutoffs 2020-01-24 → 2020-05-23 (all COVID). Floor: SeasonalNaive(lag-7)

## Results readout (mean over 5 cutoffs)

| method | WAPE | sMAPE% | dir% | cov90% |
|---|---:|---:|---:|---:|
| Timestral (joint 14-sector) | **16.5%** | 18.5 | 70.4 | 91.0 |
| SeasonalNaive(7) | 22.4% | 26.0 | 67.8 | — |

Per-sector: gas/grocery ~10% WAPE, typical sectors 13–22%, **travel 44%** (the only event-driven sector; 40k spike day in the final window).

## What the results mean
1. **First clear consumer-rhythm win**: −26% WAPE vs the weekly floor + better directional accuracy — the joint tensor's shared weekly rhythm + shared wallet level is what pays.
2. **The regime-break lesson (repeated in later cases)**: the worst window is the COVID-onset cutoff (WAPE 26.4%, direction 53%); every later window recovers to 12–16% as **fresh rolling context carries the new regime — with no COVID covariate at all**. Roll context forward rather than trusting a stale distribution.
3. **Skill is sector-dependent — ship the table, not an aggregate**: rhythm sectors are production-ready (~10% WAPE); travel-class sectors need the q90 band, not the median.
4. **Bands widen correctly under stress**: 91% coverage with visibly fatter travel bands.

## Caveats
- Shipped file is smaller than the plan's verified card (1.30M vs 1.85M rows, ends 2020-06-21): schema drift moved the backtest into COVID instead of the planned holiday peak.
- No Kaggle leaderboard exists for this aggregated view; seasonal-naive floor is the anchor.

**Plots**: `out_06_corr.png` (daily cross-sector correlation), `out_06_sectors.png` (per-sector nowcasts through the break).
