# Inventory Demand Forecasting: Kalman Filtering and Global DeepAR Models

**Phase 1 & Phase 2: Complete Implementation**  
We forecast daily unit sales for 99 product-store series from the M5 (Walmart) competition data and compare four paradigms:

| Model | Type | Paradigm |
|---|---|---|
| Seasonal naive | repeats last week | Simple baseline |
| Kalman filter | one local-level filter per series, with weekday effects and stockout handling | Local state-space estimation |
| GBDT | global gradient boosted tree model on lag, calendar, price and item features | Tabular AML |
| DeepAR (LSTM) | global autoregressive encoder-decoder LSTM with parametric Gaussian likelihood | Global Deep Learning sequence model |

Every model is evaluated strictly on the exact same 6 rolling forecast origins with a 14-day horizon and scored with:
- Point metrics: **MAE, RMSE, WAPE**
- Probabilistic metrics: **pinball loss ($q_{10}, q_{50}, q_{90}$)** and **80% interval coverage**.

Phase 1 findings are in [`reports/phase1_report.md`](reports/phase1_report.md).  
Phase 2 findings and comparative analysis are in [`reports/phase2_report.md`](reports/phase2_report.md).

## How to run

1. Install all dependencies (Python 3.8+):

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure M5 dataset files are in `data/raw/`:
   - `sales_train_evaluation.csv`
   - `calendar.csv`
   - `sell_prices.csv`

3. **Phase 1 Execution (Baselines & Kalman Filter):**

   ```bash
   python3 run_phase1.py --rebuild
   ```

4. **Phase 2 Execution (DeepAR / LSTM on identical split):**

   ```bash
   python3 run_phase2.py
   ```

5. **Run test suite (Phase 1 + Phase 2 checks):**

   ```bash
   python3 -m pytest tests/
   ```

6. Interactive Walkthroughs:
   - Phase 1: `notebooks/phase1_walkthrough.ipynb`
   - Phase 2: `notebooks/phase2_walkthrough.ipynb`

## Project layout

```
config.yaml              all settings (sample size, horizon, origins, GBDT, Kalman, and LSTM parameters)
data_loader.py           read M5, pick series, complete date grid, launch dates, stockout flags
features.py              lag / rolling / calendar / price features (past data only)
metrics.py               MAE, RMSE, WAPE, pinball loss, coverage
baselines.py             seasonal naive and GBDT
kalman_filter.py         per-series Kalman filter
backtest.py              rolling-origin backtest framework
lstm_dataset.py          PyTorch Dataset, Scaler, and DataLoader generation for LSTM (Phase 2)
lstm_model.py            DeepAR LSTM architecture, Gaussian NLL loss, training loop (Phase 2)
lstm_forecast.py         Inference pipeline and quantile formatting for DeepAR (Phase 2)
report.py                Phase 1 metric tables, charts and markdown report
report_phase2.py         Phase 2 4-model metric tables, charts and markdown report (Phase 2)
run_phase1.py            Phase 1 pipeline orchestrator
run_phase2.py            Phase 2 pipeline orchestrator (Phase 2)
tests/                   comprehensive test suite (Phase 1 unit tests + test_lstm.py)
notebooks/               walkthrough notebooks for Phase 1 and Phase 2
reports/                 output reports, CSVs, prediction records, and charts
```

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
