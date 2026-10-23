# Use case 09 — German power system: day-ahead triad (solar / wind / load)

**Code**: `11_solar.py` (runnable master at project root: `python 11_solar.py`)

## Problem statement
Day-ahead (24 h) forecasting of Germany's power-system triad: solar generation, onshore wind, and nation load (ENTSO-E Transparency data 2015–2020, hourly). The business lens is a **balance-market KPI**: commit day-ahead energy (horizon-sum of the median) and measure the resulting imbalance — vs committing yesterday's profile.

## Setup
- Targets: **(solar, wind_onshore, load) jointly as a 3-variate tensor** `(3, 50401)` (sliding 1-year context)
- **First case exercising known-future NUMERICS**: DE day-ahead load forecast + DE-LU day-ahead price — both genuinely knowable at issuance — fed as broadcast numerics; known cats: hour (UTC), month
- H=24; 12 daily cutoffs (Sep-2020 tail) + Easter-Monday 2020-04-13 COVID vignette
- Floor: SeasonalNaive(lag-24h)

## Results readout (12 daily cutoffs, pooled)

| metric | Timestral | naive |
|---|---:|---:|
| MAE (pooled) | **1,147 MW** | 3,249 |
| MASE | **0.48** | 0.93 |
| cov90% | 93.2 | — |
| Commit imbalance (mean) | **66.6 GW·h** | 214.7 GW·h (11/12 windows better) |

Vignette: Easter Monday at 83% of prior Mondays (holiday × lockdown) — MAE 1,856 vs naive 5,603, cov 100%.

## What the results mean
1. **Known numerics work and the model leans on them** — the load panel visibly rides the issued ENTSO-E forecast. This is the template for "we have an official issued forecast, what does the foundation model add?"
2. **Point-forecast skill converts directly into a balance-market KPI** with no optimizer — median-sum commit cuts imbalance to ~31% of the naive commit.
3. **Cross-variate attention pays where regimes differ per variate**: the biggest naive blowups are load-ramp/week-transition days (MAE up to 5.3 GW); the triad tensor absorbs them.
4. **Weak spot**: weekend windows (the only MASE > 1.0) — month+hour codes don't carry the weekly phase; feeding day-of-week codes (as done in cases 10/12) is the standing upgrade.

## Caveats
- Hour-of-day is UTC (solar noon ~10 UTC) — not a bug, but it makes seasonal-naive an unusually strong solar floor since the daylight phase repeats exactly.
- MAE pools MW across variates of different magnitude → MASE is primary.
- OPSD Transparency edition ends 2020-09.

**Plot**: `out_11_solar.png` (solar / wind / load final window + known-load overlay + Easter vignette).
