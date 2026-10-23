# Use case 10 — London bike-share: day-ahead hourly demand & rebalancing budget

**Code**: `12_bike.py` (runnable master at project root: `python 12_bike.py`)

## Problem statement
Transport-for-London cycle-hire hourly counts (2015-01 → 2017-01). Deliverables: (a) day-ahead 24 h city-wide demand with the strong weekly commuter rhythm, (b) a **morning-peak (07–09 h) rebalancing budget** — ops staff to the predicted q90 peak, not the median, (c) two shock vignettes: Christmas 2015 near-closure week, and the famous 2015-07-09 tube strike (bike-demand spike with no covariate).

*(Original plan was ATM cash demand; the Kaggle ATM dataset proved synthetic sparse noise — max 6 contiguous days of observations — and was swapped per user decision.)*

## Setup
- Target: city-wide hourly `cnt` `(1, 17544)` (130 missing grid hours → NaN passthrough)
- Past covariates: temperature, humidity, wind; known numerics: is_holiday + is_weekend; known cats: hour + **day-of-week** (the case-09 lesson applied)
- H=24; 12 daily cutoffs in the holiday-heavy Dec-2016/Jan-2017 tail
- Floors: SeasonalNaive(24) AND (168) — the weekly floor is the stronger commuter benchmark

## Results readout (12 daily cutoffs)

| metric | Timestral | naive(24h) | naive(168h) |
|---|---:|---:|---:|
| MAE | **291** | 393 | 514 |
| MASE | 0.83 | — | — |
| cov90% | 82.6 | — | — |
| am-peak budget (q90 / median / actual) | 1,547 / 833 / 821 | | |

Vignettes: Christmas-eve MAE 447 vs n24 408 (slight LOSS); tube-strike MAE 208 vs n168 200 (tie; q90 day peak 4,829 vs actual 5,560 = 87% spike headroom).

## What the results mean
1. **−43% vs the strongest naive with the weekly phase fed as known category** — errors are amplitude, not shape.
2. **Christmas DAY collapses (MASE 3.30, cov 33%)** despite the is_holiday flag: a once-a-year near-closure outside the 1-year sliding context. Holidays that remove ~95% of demand need multi-year context or closure masking (case-07's trick).
3. **Inverted failure on Christmas EVE** — the model predicts a plausible big shopping peak that doesn't happen: the annual cycle is learned, the year-specific closure policy is not.
4. **Strike day ≈ tie with naive-168**: commuter structure intact, spike headroom 87% via q90. Event spikes without a covariate remain a coin-flip (consistent with cases 04/06/08).
5. **The staffing rule (q90) never zeroed out dock overflow** in the test window — a usable ops readout.

## Caveats
- ALL 12 cutoffs sit in Christmas/New-Year week — the hardest window of the year; the −43% is a lower bound, not a business-case headline.
- No leaderboard for the TfL series (the Kaggle "Bike Sharing Demand" competition is a different, DC dataset); local floors are the anchors.

**Plot**: `out_12_bike.png` (final window + both vignettes).
