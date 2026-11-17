# Contributions

## Phase 1

| Member | Part | Files |
|---|---|---|
| Ankita | Data preparation: M5 loading, series sampling, complete date grid, launch dates, stockout marking | `data_loader.py`, `tests/test_data_loader.py` |
| Aradhya | Features and evaluation metrics | `features.py`, `metrics.py`, `tests/test_features_metrics.py` |
| Anuj | AML baselines: seasonal naive and GBDT with quantile models | `baselines.py`, `tests/test_baselines.py` |
| Yashvi | Per-series Kalman filter (local state estimation) | `kalman_filter.py`, `tests/test_kalman_filter.py` |
| Mouli | Rolling-origin backtest, report, charts, run script and notebook | `backtest.py`, `report.py`, `run_phase1.py`, `tests/test_backtest.py`, `notebooks/`, `reports/` |

Every member reviewed at least one other member's pull request and ran the full pipeline before submission.

---

## Phase 2

| Member | Part | Files |
|---|---|---|
| Ankita | PyTorch Dataset/DataLoader for LSTM: sliding-window builder, per-series Scaler, encoder/decoder feature tensors (12 encoder features, 7 decoder features) | `lstm_dataset.py` |
| Aradhya | DeepAR LSTM architecture, Gaussian NLL loss, train/eval loop, `fit_lstm()` with LR decay | `lstm_model.py` |
| Anuj | `LSTMForecaster` class and `deepar_forecast()` functional interface: inference, inverse-scaling, output in Phase 1 table format | `lstm_forecast.py` |
| Yashvi | Full pytest suite for LSTM: Scaler roundtrip, Dataset shapes, model forward pass, quantile ordering, end-to-end smoke test | `tests/test_lstm.py` |
| Mouli | Phase 2 runner script, extended report comparing all 4 models (SeasonalNaive, KalmanFilter, GBDT, DeepAR), walkthrough notebook | `run_phase2.py`, `report_phase2.py`, `notebooks/phase2_walkthrough.ipynb` |

Shared infrastructure updated by the team: `config.yaml` (lstm section added), `requirements.txt` (torch>=2.0 added).

Every member reviewed at least one other member's pull request and ran the full pipeline before submission.
