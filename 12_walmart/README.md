# Use case 12 — M5 (Walmart): 300-series, 28-day retail demand from real grocery data

**Code**: `14_m5.py` (runnable master at project root: `python 14_m5.py`)

## Problem statement
**Dataset**: M5 Forecasting – Accuracy — Walmart's real open retail dataset (10 stores in CA/TX/WI × ~3,050 FOODS/HOBBIES/HOUSEHOLD items = 30,490 store-item series, daily unit sales 2011-01-29 → 2016-05-22).
**Prediction target**: the next **28 days of daily unit sales** for each store-item combination, evaluated per-series with the competition's RMSSE metric. The first sampled cutoff (d_1913) is the competition's exact public-leaderboard evaluation window.

## Setup
- Sample: 300 series stratified 30/store among active items (≥1 sale in last 56 d, seed 0) of 30,490 — daily counts are sparse/intermittent (multiple 0/1/2-unit days)
- Known numerics: rescaled unit price (known at issue; ffilled) + SNAP food-stamp flag (per state); known cats: day-of-week + event-presence codes
- H=28; cutoffs d_1913 (public-LB window) + d_1885; chunk-100 forward passes; full-history context (no cap)

## Results readout

| cutoff | RMSSE | RMSSE SeasonalNaive | WAPE | cov90% |
|---|---:|---:|---:|---:|
| d_1913 (public-LB window) | **0.9507** | 1.2515 | 92.7% | 94% |
| d_1885 | 0.9678 | 1.2520 | 92.8% | 92% |
| Verified competition winner anchor (WRMSSE) | 0.515 | | | |

Per category (first cutoff): FOODS 0.92 / HOBBIES 0.93 / HOUSEHOLD 1.01 vs naive 1.18–1.30 — improvement is uniform across categories.

## What the results mean
1. **Zero-shot beats SeasonalNaive(28) on the exact public-LB window (−24% RMSSE, −38% pooled WAPE)** — straight out of the box, no training, no tuning.
2. **The 92.7% WAPE headline is a denominator artifact of intermittent demand**: item-store daily sales are mostly 0/1/2 units, so a good median still earns ~100% WAPE. The competition weights by **dollar sales**, shifting emphasis to high-volume FOODS_3 items — my unweighted mean keeps every sparse series at equal weight. RMSSE is the primary metric here, and per-series is the competition-consistent aggregate.
3. **Honest gap to the winner**: 0.95 sampled RMSSE vs winner WRMSSE 0.515 — the winners are purpose-trained LightGBM/DeepAR stacks with per-level hierarchy. Foundation model = useful zero-shot floor (−24% vs naive), not a competition substitute.
4. **Dead/all-zero series are a free win** (RMSSE 0.00 vs naive 0.51 on FOODS_3_662) — automatic, no postprocessing.
5. **Metric trap caught**: all-zero 28-day truth windows make a naive `|truth|`-denominator WAPE explode to e8-scale garbage; fixed by skipping zero-denominator series (competition-consistent).

## Caveats
- Sample is active-item biased (≥1 sale in last 56 d) and unweighted — NOT the competition's WRMSSE; only the model-vs-naive ratio is comparable.
- Local RMSSE scale computed in-sample per cutoff; official uses d_1..d_1913 only.
- Kaggle rules not accepted → Google-Drive mirror (Temporian's verified copy) used.

**Plots**: `out_14_m5.png` (4 sampled series incl. dead-item and price overlay), `out_14_m5_cats.png` (per-category RMSSE bars).
Full per-series/per-category CSVs are in this folder; the generated `results_m5.md` stays in `bench/cases/14_m5/`.
