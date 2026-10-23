# Use case 07 — Rossmann: closures, promotions & refurbishment cold-start

**Code**: `09_uc12_rossmann.py` (runnable master at project root: `python 09_uc12_rossmann.py`)

## Problem statement
Rossmann drugstores (the classic Kaggle competition, 1,115 stores). Deliverables: (a) daily sales 42 days ahead with the official RMSPE metric scored on **open days only**, (b) the closure/refurbishment problem: stores with **184-day gaps** (renovations) must be forecast through the gap and track the reopening ramp.

## Setup
- Targets: ALL 1,115 stores, batched univariate chunks of 100; daily 2013-01 → 2015-07-31
- The core trick: Sales/Customers masked to NaN wherever Open==0 or row missing (33k store-days); predictions post-zeroed on known-future closed days
- Known: Promo, SchoolHoliday, StateHoliday, open-mask, dow/month codes; past: Customers
- H=42; official T-42 + T-84 windows. Floor: SeasonalNaive

## Results readout

| window | RMSPE | RMSPE naive | cov90% |
|---|---:|---:|---:|
| official T-42 | **0.1265** | 0.3500 (−64%) | 87.5 |
| T-84 | **0.1268** | 0.3047 (−58%) | 93.2 |

Cold-start vignettes (forecast issued at the start of a 184-day renovation gap): stores 1092/1094/13 RMSPE **0.103–0.130 vs naive 0.19–0.48**, reopening ramp tracked properly.

## What the results mean
1. **Strongest deployment-style case of the batch**: −64%/−58% vs naive in both windows, and the days-of-week closure calendar (Sundays) reproduced exactly through the known covariates.
2. **The closure-NaN trick works as advertised**: forecast through the renovation gap, rebuilt the pre-gap weekly rhythm with zero retraining — PLAN's reopening/stockout scenario is what the plots show.
3. **Honest positioning**: our scored windows are in-sample (the official test period lies beyond the file), so RMSPE 0.1265 sits in the "solid supervised model" tier (0.13–0.17) of the competition ladder, not the winning 0.10–0.11 tier — the robust claim is the relative-vs-naive win, every window.
4. **Vignette design lesson**: an evaluation horizon shorter than the gap scores all-NaN (nothing scorable); use gap_length + H when evaluating reopening.

## Caveats
- Kaggle rules not accepted → mirror dataset used.
- Promo2 (continuing promo campaign) calendar not modeled — supervised LB models use it heavily; their out-of-sample advantage stands.
- GPU lesson: chunk=200 OOM'd decode (7.6 GB attention on 8 GB); chunk=100 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the safe default.

**Plot**: `out_09_rossmann.png` (top: closure calendar reproduction; bottom: reopening ramps).
