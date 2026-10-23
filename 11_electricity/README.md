# Use case 11 — UCI electricity loads: 120-client batch-scale day-ahead

**Code**: `13_electricity.py` (runnable master at project root: `python 13_electricity.py`)

## Problem statement
UCI ElectricityLoadDiagrams20112014: 370 Portuguese clients' hourly loads 2011–2014. The batch-scale showcase: sample 120 active clients as ONE tensor `(120, 35065)` and run chunked forward passes — answering "does the deployment stack scale?". Decision deliverable: **system-level aggregation** — sum 120 client medians for a group commit and compare imbalance vs committing naive (variance pooling).

## Setup
- Source is 15-min data → aggregated to hourly MEANS, cached float32 npz (369 active clients; sample 120, seed 0)
- Known covariates: categorical only (hour + day-of-week codes); no past covariates — pure cross-variate attention
- H=24; 12 rolling daily cutoffs (Dec-2014 tail); sliding half-year context; chunk=60

## Results readout (12 daily cutoffs, pooled 120 clients)

| metric | Timestral | SeasonalNaive(24) |
|---|---:|---:|
| MAE (mean kW) | **~57** | ~93 |
| MASE | 1.18 | 1.83 (−35%) |
| cov90% | 88.0% | — |
| group-sum commit | **~1/3 naive imbalance** on normal days | |

Christmas week: naive MASE 4.84/4.45 for both methods (model closes some gap: 2.16/3.95).

## What the results mean
1. **The batching showcase works end-to-end**: ~14 s per cutoff-pair of chunk-60 at 4,392-h context, zero OOM with `expandable_segments:True` (~2.9 min for the 12-cutoff sweep).
2. **Both methods sit above MASE 1.0 — a metric-context lesson**: the lag-24 scale computed inside a flat-holiday week is tiny, inflating absolute MASE. **Relative-vs-naive (−35%)** is the honest headline — the model improves on naive everywhere except Christmas Day itself.
3. **System-level aggregation is a free win**: summing 120 client medians gives group imbalance ~1/3 of the naive commit — cross-variate attention + median-sums monotonically improve at aggregate level.
4. **No past covariates still lands**: hour+dow codes carry the phase.

## Caveats
- All cutoffs sit in the Dec-2014 holiday tail (again a lower bound, not representative).
- Client sample is seed-0 arbitrary; per-client spread is wide (`results_per_client.csv` in this folder has the full table).
- Direction-only anchors: GluonTS electricity protocol WQL (SeasonalNaive 0.333 → Chronos-Bolt ~0.122) uses different horizons/splits than this case.

**Plot**: `out_13_electricity.png` (4 sampled clients, final window).
