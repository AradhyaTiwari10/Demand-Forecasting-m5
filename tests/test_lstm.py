"""
tests/test_lstm.py — Phase 2 LSTM component tests  (Yashvi's contribution)
===========================================================================
Run with:
    python3 -m pytest tests/

All tests are skipped automatically when PyTorch is not installed.
"""

import sys
from pathlib import Path

# Make project root importable (helpers.py also does this, but we set it here
# so the file is usable standalone with `python3 -m pytest tests/`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------
# Skip the whole module if PyTorch is not available
# ------------------------------------------------------------------
torch = pytest.importorskip("torch")

from helpers import make_series

# ---------------------------------------------------------------------------
# Lazily import Phase-2 modules so a missing optional dep gives a clean skip
# ---------------------------------------------------------------------------

def _import_scaler():
    from lstm_dataset import Scaler
    return Scaler


def _import_dataset():
    from lstm_dataset import M5WindowDataset, Scaler
    return M5WindowDataset, Scaler


def _import_model():
    from lstm_model import DeepARLSTM
    return DeepARLSTM


def _import_loss():
    from lstm_model import gaussian_nll_loss
    return gaussian_nll_loss


def _import_forecaster():
    from lstm_forecast import LSTMForecaster
    return LSTMForecaster


# ---------------------------------------------------------------------------
# Shared helper: extend make_series with calendar columns required by the
# M5WindowDataset (day_of_week, day_of_month, month, days_since_launch).
# ---------------------------------------------------------------------------

def _make_full_series(demand, series_id="A", start="2020-01-01"):
    """Return a fake series DataFrame with all columns the LSTM dataset needs."""
    df = make_series(demand, series_id=series_id, start=start)
    df["day_of_week"]     = df["date"].dt.dayofweek
    df["day_of_month"]    = df["date"].dt.day
    df["month"]           = df["date"].dt.month
    df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days
    return df


# ===========================================================================
# Test 1 — Scaler round-trip
# ===========================================================================

def test_scaler_roundtrip():
    """
    Fit a Scaler on a fake series, transform demand, inverse-transform, and
    verify that we recover the original values within floating-point tolerance.
    """
    Scaler = _import_scaler()

    demand = ([3, 4, 5, 3, 4, 5] * 20)   # 120 days
    df = _make_full_series(demand, series_id="S1")

    scaler = Scaler(df, series_ids=["S1"])

    original = np.array(demand, dtype=np.float32)
    scaled   = scaler.transform("S1", original)
    recovered = scaler.inverse_transform("S1", scaled)

    np.testing.assert_allclose(
        recovered, original, atol=1e-5,
        err_msg="Scaler round-trip did not recover original demand within 1e-5",
    )


# ===========================================================================
# Test 2 — M5WindowDataset length
# ===========================================================================

def test_m5window_dataset_length():
    """
    Two fake series of 200 days each; the sliding-window dataset must yield
    more than zero samples.
    """
    M5WindowDataset, Scaler = _import_dataset()

    demand = list(range(1, 201))   # 1 … 200, all positive
    df = pd.concat([
        _make_full_series(demand, series_id="A"),
        _make_full_series(demand, series_id="B"),
    ], ignore_index=True)

    series_ids = ["A", "B"]
    scaler     = Scaler(df, series_ids)
    dataset    = M5WindowDataset(
        df,
        series_ids,
        context_length=90,
        prediction_length=14,
        scaler=scaler,
        min_days=30,
    )

    assert len(dataset) > 0, (
        "M5WindowDataset is empty — expected multiple windows for 2×200-day series"
    )


# ===========================================================================
# Test 3 — M5WindowDataset item shapes
# ===========================================================================

def test_m5window_dataset_shapes():
    """
    One fake series of 200 days; a single dataset item must have the expected
    encoder (90×12), decoder (14×7) and target (14,) shapes.
    """
    M5WindowDataset, Scaler = _import_dataset()
    from lstm_dataset import ENCODER_FEATURES, DECODER_FEATURES

    demand     = list(range(1, 201))
    df         = _make_full_series(demand, series_id="X")
    series_ids = ["X"]
    scaler     = Scaler(df, series_ids)
    dataset    = M5WindowDataset(
        df,
        series_ids,
        context_length=90,
        prediction_length=14,
        scaler=scaler,
        min_days=30,
    )

    assert len(dataset) > 0, "Dataset is unexpectedly empty"
    item = dataset[0]

    enc = item["encoder_input"]
    dec = item["decoder_input"]
    tgt = item["target"]

    assert enc.shape == (90, ENCODER_FEATURES), (
        f"encoder_input shape {enc.shape} != (90, {ENCODER_FEATURES})"
    )
    assert dec.shape == (14, DECODER_FEATURES), (
        f"decoder_input shape {dec.shape} != (14, {DECODER_FEATURES})"
    )
    assert tgt.shape == (14,), (
        f"target shape {tgt.shape} != (14,)"
    )


# ===========================================================================
# Test 4 — DeepARLSTM forward-pass output shape
# ===========================================================================

def test_deepar_lstm_forward_shape():
    """
    Forward pass through a tiny DeepARLSTM must return shape (batch, pred, 2).
    """
    DeepARLSTM = _import_model()

    model = DeepARLSTM(
        encoder_features=12,
        decoder_features=7,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()

    enc_input = torch.randn(2, 90, 12)   # (batch=2, context=90, enc_feat=12)
    dec_input = torch.randn(2, 14, 7)    # (batch=2, horizon=14, dec_feat=7)

    with torch.no_grad():
        out = model(enc_input, dec_input)

    assert out.shape == (2, 14, 2), (
        f"Expected forward output shape (2, 14, 2) but got {tuple(out.shape)}"
    )


# ===========================================================================
# Test 5 — predict_quantiles ordering and non-negativity
# ===========================================================================

def test_deepar_lstm_predict_quantiles_values():
    """
    predict_quantiles must return q10 ≤ q50 ≤ q90 (checked on mean across the
    batch/horizon) and all values must be ≥ 0 (demand cannot be negative).
    """
    DeepARLSTM = _import_model()

    model = DeepARLSTM(
        encoder_features=12,
        decoder_features=7,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    enc_input  = torch.randn(2, 90, 12)
    enc_mask   = torch.ones(2, 90, dtype=torch.bool)
    dec_input  = torch.randn(2, 14, 7)
    quantiles  = [0.1, 0.5, 0.9]

    preds = model.predict_quantiles(enc_input, enc_mask, dec_input, quantiles)

    q10 = preds[0.1]
    q50 = preds[0.5]
    q90 = preds[0.9]

    # Ordering: mean q10 ≤ mean q50 ≤ mean q90
    assert q10.mean().item() <= q90.mean().item(), (
        "Mean q10 should be ≤ mean q90"
    )
    assert q10.mean().item() <= q50.mean().item() + 1e-6, (
        "Mean q10 should be ≤ mean q50"
    )
    assert q50.mean().item() <= q90.mean().item() + 1e-6, (
        "Mean q50 should be ≤ mean q90"
    )

    # Non-negativity (demand is clipped at 0)
    assert (q10 >= 0).all(), "q10 contains negative values"
    assert (q50 >= 0).all(), "q50 contains negative values"
    assert (q90 >= 0).all(), "q90 contains negative values"


# ===========================================================================
# Test 6 — gaussian_nll_loss is finite
# ===========================================================================

def test_gaussian_nll_loss_finite():
    """
    gaussian_nll_loss on random inputs with an all-True mask must return a
    finite (non-NaN, non-Inf) scalar.
    """
    gaussian_nll_loss = _import_loss()

    batch, horizon = 4, 14
    mu         = torch.randn(batch, horizon)
    log_sigma  = torch.randn(batch, horizon)
    target     = torch.abs(torch.randn(batch, horizon))   # non-negative demand
    mask       = torch.ones(batch, horizon, dtype=torch.bool)

    loss = gaussian_nll_loss(mu, log_sigma, target, mask)

    assert torch.isfinite(loss), (
        f"gaussian_nll_loss returned non-finite value: {loss.item()}"
    )


# ===========================================================================
# Test 7 — LSTMForecaster smoke test (end-to-end)
# ===========================================================================

def test_lstm_forecaster_smoke_test():
    """
    End-to-end smoke test: fit LSTMForecaster for 1 epoch on 5 fake series,
    then call forecast() and check the result schema and size.
    """
    LSTMForecaster = _import_forecaster()

    # ---- Minimal config matching the lstm section of config.yaml ----
    lstm_config = {
        "epochs":            1,
        "hidden_size":       16,
        "num_layers":        1,
        "batch_size":        8,
        "context_length":    60,
        "prediction_length": 14,
        "dropout":           0.0,
        "learning_rate":     0.01,
        "num_workers":       0,
        "device":            "cpu",
    }

    # ---- Build 5 fake series of 200 days ----
    horizon    = 14
    n_series   = 5
    n_days     = 200
    train_days = n_days - horizon          # 186 training days

    demand = ([2, 3, 4, 2, 3, 4] * 34)[:n_days]   # 200 days, cyclic demand

    all_dfs = [
        _make_full_series(demand, series_id=f"S{i}")
        for i in range(n_series)
    ]
    full_df = pd.concat(all_dfs, ignore_index=True)

    # ---- Split into train / future ----
    pivot_date = full_df["date"].min() + pd.Timedelta(days=train_days - 1)
    train = full_df[full_df["date"] <= pivot_date].copy()
    future = full_df[full_df["date"] > pivot_date].copy()

    # future_df must NOT contain demand or is_stockout (those are unknowns)
    future = future.drop(columns=["demand", "is_stockout"])

    # ---- Fit & forecast ----
    forecaster = LSTMForecaster(lstm_config)
    forecaster.fit(train)
    result = forecaster.forecast(train, future)

    # ---- Structural checks ----
    required_cols = {"series_id", "date", "step", "point", "q10", "q50", "q90"}
    assert required_cols.issubset(result.columns), (
        f"Result is missing columns: {required_cols - set(result.columns)}"
    )

    expected_rows = n_series * horizon    # 5 * 14 = 70
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows but got {len(result)}"
    )

    assert (result["q10"] <= result["q90"]).all(), (
        "Some q10 values exceed q90 — quantile ordering violated"
    )
