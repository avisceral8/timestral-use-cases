# Timestral Use Cases — zero-shot forecasting with Timestral-209M

Applied experiments with [**Timestral-209M**](https://huggingface.co/mistralai/Timestral-209M) (Mistral AI's time-series foundation model) across 12 realistic business forecasting problems. **Every result is zero-shot** — no fine-tuning, no per-case training; only context, covariates, and quantile heads are consumed.

Each case folder contains:

- **`README.md`** — problem statement, model setup, results readout, interpretation, caveats
- **`NN_*.py`** — the full experiment code
- **`out_*.png`** — the plot outputs
- **`results_*.csv`** — raw metric tables (aggregate + per-entity breakdowns)

## The 12 cases

| # | Folder | Forecast task | Horizon | Headline result vs naive floor |
|---|---|---|---|---|
| 01 | `01_pharma` | Pharmacy cross-class drug demand + stock targets (cannibalization) | 28d | MASE **0.67** vs 1.0 |
| 02 | `02_treasury` | US Treasury yield curve, 11 tenors jointly (1962–2026 one-pass) | 20d | ties random walk; curve stays smooth |
| 03 | `03_commodities` | 6-commodity price locking / COGS hedging policy | 6mo | MAPE ties RW; quantile bands calibrated |
| 04 | `04_cards` | 14-sector credit-card spend nowcast through COVID onset | 30d | WAPE **16.5%** vs 22.4% |
| 05 | `05_warehouses` | 4 warehouses × 10 categories, batched joint | 12wk | MASE **0.61** vs 0.89 |
| 06 | `06_favorita` | Favorita retail 300 series + newsvendor ordering | 28d | RMSLE **−21%**, survives quake window |
| 07 | `07_rossmann` | Rossmann 1,115 stores with closure/reopening gaps | 42d | RMSPE **0.127** vs 0.35 |
| 08 | `08_air` | Beijing 12-station PM2.5 nowcast + smog alert rule | 24h | MASE **0.46**; alert recall 100% |
| 09 | `09_solar` | German power triad solar/wind/load with known-future numerics | 24h | MASE **0.48**; commit imbalance −69% |
| 10 | `10_bike` | London bike-share hourly demand + rebalancing budget | 24h | MAE **−43%** vs strongest naive |
| 11 | `11_electricity` | UCI electricity, 120-client batch-scale tensor | 24h | naive-relative **−35%** at scale |
| 12 | `12_walmart` | M5 (Walmart) 300 store-item series, public-LB window | 28d | RMSSE **0.95** vs 1.25 (−24%) |

## Key takeaways

1. **Zero-shot beats every seasonal/random-walk floor in 10 of 12 cases** — the two ties (treasury, commodities) are martingale-heavy domains where the random walk is the literature-accepted bar, and Timestral passes it while adding calibrated uncertainty.
2. **The quantile head is the product**: q90 service-level bands arrive calibrated (89–94% coverage vs 90% nominal) without tuning — that's what converts point forecasts into stock targets, staffing budgets, and alert thresholds.
3. **Cross-variate attention pays when series are physically coupled** (drug classes, warehouses, power-triad, air stations); batched univariate is enough for independent-ish series (stores, clients).
4. **Known-future covariates work as advertised**: day-of-week codes, holidays, promo calendars, and even issued third-party forecasts (case 09's day-ahead load forecast) are visibly leaned on. IDs and flags go in as categorical codes; numerics broadcast over context + horizon.
5. **Regime breaks are the honest failure mode**: every case's worst window is the shock boundary (COVID onset, Christmas closures, holiday-flat weeks). Fresh rolling context recovers within a horizon; event spikes without a covariate remain a coin flip.
6. **NaN semantics are a feature**: masks on closures/refurbishment gaps (case 07), ragged series starts (cases 02, 03), and missing grid hours (case 10) all raw-passthrough fine.

## Caveats

- All local naive baselines computed in this study's own protocol; published anchors (Kaggle leaderboards, papers) cited per-case in the READMEs — no head-to-head with other foundation models.
- Evaluation windows deliberately include hard periods (COVID, Christmas, quake); several cases are explicitly lower bounds.
- Case 12 is an unweighted 300-series sample — not directly comparable to the M5 winning WRMSSE (0.515); only the model-vs-naive ratio is.

## Repo structure notes

- Code files here are **reference copies**; they read local data caches and write results relative to the originating workspace, so they are not runnable as-is inside this repo cut-down. Required model: `mistralai/Timestral-209M` (Hugging Face) + PyTorch + `timestral` package.
- Datasets are not included; each case README names its exact source (OPSD direct download, UCI, Kaggle hub mirrors, Google-Drive mirror for M5 competition data).
