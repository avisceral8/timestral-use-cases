# Use case 05 — Warehouses: multi-warehouse component replenishment

**Code**: `07_uc21_warehouses.py` (runnable master at project root: `python 07_uc21_warehouses.py`)

## Problem statement
A component planner replenishes 4 warehouses × 10 product categories. Key design: each category forecasts its **4 warehouse streams jointly** (cross-warehouse attention), categories batched independently — the batching showcase. Deliverables: 12-week demand forecasts + **q95-based reorder policies** per (category, warehouse).

## Setup
- Target: `(B=10, V=4, T=315)` weekly (2011-01 → 2017-01, 31% zero cells); known: quarter-month flag, week-of-year + month codes
- H=12 weeks; 5 non-overlapping cutoffs. Floor: SeasonalNaive(lag-52w)
- Data quirks: "(…)" return rows floored at 0 (gross-demand); unparseable dates dropped (1.1%)

## Results readout (mean over cutoffs)

| metric | Timestral | SeasonalNaive(52w) |
|---|---:|---:|
| WAPE | **29.3%** | 40.8% |
| MASE | **0.61** | 0.89 |
| pinball @ q95 | **21.4k** | 63.0k (3.0× worse) |
| cov90% | 90.1 | — |

Reorder policy example: Category_019/Whse_J weekly q95 ≈ +32% buffer over median; smaller/lumpier warehouses need +171% (Whse_S) — vs the naive q95 proxy that would over-order all of them.

## What the results mean
1. **Batching semantics validated**: one call forecasts 40 series (10 batches × 4 joint variates) in 150–400 ms; batch = "same model, many planning objects."
2. **The quantile head is where replenishment value lives**: pinball@q95 is 3× better than the naive's flat proxy — safety stock is a *quantile* question, not a median question.
3. **Christmas surge anticipated with zero holiday features** (learned from annual rhythm) — but surge amplitude varies year-to-year: spike *timing* skill ≠ spike *amplitude* precision; layer q95 on top.
4. **Lumpy categories still improve** (Cat_032: 92% vs naive 210%); the production-ready set is the top ~5 categories (WAPE 24–53%).
5. **Scale robustness**: weekly means from ~1e4 to ~1.3e7 in the same tensor, all sensible — internal per-batch scaling handles it.

## Caveats
- Span drift confirmed (2011–2017, not the plan's 2012–2016).
- What the 4-way joint attention itself bought is not isolated (ablation declined per project rules).
- No leaderboard exists for this aggregated weekly view; local SeasonalNaive floor is the anchor.

**Plot**: `out_07_whsflow.png` (final-window flow forecasts incl. December ramp).
