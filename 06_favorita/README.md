# Use case 06 — Favorita: retail replenishment + newsvendor ordering

**Code**: `08_uc23_favorita.py` (runnable master at project root: `python 08_uc23_favorita.py`)

## Problem statement
Ecuadorian grocery chain (Corporación Favorita, Kaggle's most famous retail competition). 300 sampled (store, family) daily series; deliverables: (a) 28-day RMSLE forecasts vs literature anchors, (b) an **earthquake-regime shock test** (M7.8 quake 2016-04-16, no covariate), (c) **newsvendor ordering policy** — perishables vs staples with different waste/stockout cost ratios.

## Setup
- Targets: 300 series, batched univariate chunks of 100; daily 2013-01→2017-08
- Past covariates: oil price (raw NaN) + store transactions; known: onpromotion (idealized future promo calendar), paydays, national holidays, dow/month codes
- H=28; windows: official T-28, T-56, T-84 cutoffs + quake window. Floor: SeasonalNaive

## Results readout

| window | RMSLE (per-series) | RMSLE naive | cov90% |
|---|---:|---:|---:|
| official T-28 | 0.3343 | 0.4545 | 91.7 |
| T-56 | 0.3223 | 0.4447 | 92.8 |
| T-84 | 0.3434 | 0.4477 | 90.8 |
| quake 2016-04-16 | **0.4348** | 0.4724 | 94.8 |

Mean per-series RMSLE **0.3587 vs naive 0.4548 (−21%)**. Literature anchors (28d/300-series protocol): LightGBM 0.1672, Chronos-2 zero-shot 0.204, AutoARIMA 0.2105, SeasonalNaive 0.2145.

## What the results mean
1. **Decisive every-window point win** — the first case where the model beats the naive floor on the official metric in all windows.
2. **Still beats naive through the quake shock** (issued with no quake covariate; the demand surge is absorbed through fresh context within the horizon; day-1 spike magnitude under-called — honest box).
3. **Newsvendor money**: q0.65 (perishables) / q0.90 (staples) ordering costs **74%** of plain-median ordering and 37% of the "last-week average" heuristic (which effectively double-orders at 201.6% cost). Quantile choice per shelf-life is where the profit lives.
4. **Anchor placement**: our −21% vs naive is weaker than the Chronos-2 anchor's draw, but seed/window mismatches mean only relative deltas are comparable (see caveats).

## Caveats
- Pointwise-aggregate RMSLE (0.4335) vs per-series-mean RMSSE (0.3587) differ ~0.075 — metric semantics trap; literature uses per-series on their own draw.
- Future onpromotion treated as known (idealized; standard, symmetric across methods).
- Study data quirks: 4 zero-row Christmas days break naive pivots; 53 fully-empty series; transactions unstack is (T,S) not (S,T).

**Plots**: `out_08_favorita.png` (quake-window surges), `out_08_newsvendor.png` (ordering policy cost curve).
