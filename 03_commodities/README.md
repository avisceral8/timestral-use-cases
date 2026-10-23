# Use case 03 — Commodities: purchase price locking (COGS hedging policy)

**Code**: `05_uc22_commodities.py` (runnable master at project root: `python 05_uc22_commodities.py`)

## Problem statement
A procurement desk buys 6 commodities (Aluminum, Copper, Nickel, Rubber TSR20, Wheat HRW, Zinc). Deliverables: 6-month-ahead price forecasts with uncertainty, and a **lock-vs-float purchasing policy**: lock a price today or float on the realized mean — and when, mid-cycle, is locking worth it? The plan claimed "3–8% COGS savings vs rolling-average benchmarks"; this case tests that claim.

## Setup
- Target: 6 commodities jointly, monthly World Bank Pink Sheet 2004–2021 (ragged: rubber NaN until 1999); past covariates Brent / NatGas-US / Coal-AUS (NaN heads) fed raw; month-of-year categorical codes
- Horizon: 6 months; **69 rolling quarterly cutoffs**; no models trained for the policy (pure forecast-consumption simulation)
- Floor: random walk; policy sim: lock@spot vs float / RollAvg-12M / q25-lock / median-lock

## Results readout

| metric | value |
|---|---:|
| Timestral MAPE (6m, 6 commodities × 69 cutoffs) | **12.21%** |
| Random-walk MAPE | 12.25% |
| 10–90% band coverage | 89.9% (nominal 90) |

Policy savings vs float: PLAN's q25-lock rule ≈ 0% (fires ~1/69 windows); median-lock variant **+0.2%…+2.2%** (best Zinc +2.17%); AlwaysLock 0.8–2.1% but only because 2004–2021 is a bull-biased sample.

## What the results mean
1. **Point forecast ties the random walk** (martingale pattern) — zero-shot gives no point-forecast alpha for commodity *prices*, consistent with finance literature.
2. **The quantile bands are the hedging deliverable**: coverage stayed 89.9% and only dipped to 78% through the 2008–2012 crisis — a stress-test pass that straight-faced naive rules can't provide.
3. **PLAN's "3–8% COGS savings" does NOT reproduce** under the stated assumptions: honest informed-rule savings ≈ 0.5–2.2%. The genuinely useful output is *mid-cycle trigger timing* (TimestralMedian savings peak post-2008 and 2015–16) rather than a blanket savings claim.
4. **Schema drift is real**: the PLAN's verified column names (Rubber RSS3) don't match the shipped file (TSR20; trailing spaces) — always re-inspect at load time.

## Caveats
- Pink Sheet stops 2021-08 — no current data in this file version.
- Lock-vs-float ignores credit-cost differences (documented simplification).
- No ablation of what the energy covariates bought (ladder declined per project rules).

**Plots**: `out_05_history.png` (6-commodity history + forecast bands), `out_05_policy.png` (policy P&L by regime).
