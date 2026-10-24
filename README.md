# Inventory Demand Forecasting: Kalman Filtering and Global DeepAR Models

**Phase 1: AML baselines and local state estimation.**
We forecast daily unit sales for 99 product-store series from the M5 (Walmart) data and compare:

| Model | Type |
|---|---|
| Seasonal naive | repeats last week |
| Kalman filter | one local-level filter per series, with weekday effects and stockout handling |
| GBDT | one global gradient boosted tree model on lag, calendar, price and item features |

Every model is tested on the same 6 rolling forecast origins with a 14-day horizon and scored with
MAE, RMSE, WAPE, pinball loss (q10 / q50 / q90) and 80% interval coverage.
Phase 2 will add a global DeepAR / LSTM model to the same backtest.

The results are in [`reports/phase1_report.md`](reports/phase1_report.md).

## How to run

1. Install the packages (Python 3.8 or newer):

   ```bash
   pip install -r requirements.txt
   ```

2. Download the M5 data from Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
   (you need to accept the competition rules first). Copy these three files into `data/raw/`:

   ```
   data/raw/sales_train_evaluation.csv
   data/raw/calendar.csv
   data/raw/sell_prices.csv
   ```

3. Run everything (about one minute):

   ```bash
   python3 run_phase1.py --rebuild
   ```

4. Run the checks:

   ```bash
   python3 -m pytest tests/
   ```

5. Optional walkthrough: open `notebooks/phase1_walkthrough.ipynb`.

## Project layout

```
config.yaml          all settings (sample size, horizon, origins, model settings)
data_loader.py       read M5, pick series, complete date grid, launch dates, stockout flags
features.py          lag / rolling / calendar / price features (past data only)
metrics.py           MAE, RMSE, WAPE, pinball loss, coverage
baselines.py         seasonal naive and GBDT
kalman_filter.py     per-series Kalman filter
backtest.py          rolling-origin backtest (same rules for every model)
report.py            metric tables, charts and the Markdown report
run_phase1.py        runs the three steps above in order
tests/               small checks for each part
notebooks/           walkthrough notebook
reports/             results (report, metrics CSVs, charts, all predictions)
```

## Main design choices

- **Stockouts are marked, never imputed.** A run of zero-sales days is flagged as a stockout when it
  is too long to happen by chance for that item. Flagged days are skipped by the Kalman filter, treated as
  missing in the GBDT features, and not scored.
- **No look-ahead.** Each model only sees data up to the origin; stockout flags for training are
  re-computed from that data only. Calendar and price data are used for future days because they are
  known in advance.
- **Cold start.** 24 of the series launched near the end of the data. Series with less than 90 days of
  history at the origin are reported separately.
- **Ready for Phase 2.** `reports/predictions.csv` has one row per model, series and forecast day.
  `reports/backtest_setup.json` stores the exact series and origins, so DeepAR can be scored on the same split.
