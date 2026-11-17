"""
run_phase2.py — Phase 2 runner  (Mouli's contribution)
=======================================================
Runs the DeepAR (global LSTM) model through the same rolling-origin backtest
used in Phase 1, appends its predictions to the Phase 1 CSV, and writes a
combined Phase 2 report.

Usage
-----
    python3 run_phase2.py                    # use existing prepared data
    python3 run_phase2.py --rebuild          # rebuild data from raw M5 files
    python3 run_phase2.py --config cfg.yaml  # custom config file

Outputs (written to config.output.reports_dir, default: reports/)
------
    predictions_phase2.csv   — Phase 1 rows + DeepAR rows combined
    phase2_report.md         — markdown comparison report
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# PyTorch is optional — give a clear message if it's missing
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401  (just checking availability)
except ImportError:
    print(
        "\n[run_phase2.py] ERROR: PyTorch is not installed.\n"
        "Install it with:  pip install torch\n"
        "Then re-run this script.\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports  (these modules are required at runtime)
# ---------------------------------------------------------------------------
from backtest import make_origins, series_ready_to_forecast
from data_loader import prepare_dataset, mark_stockouts
from lstm_forecast import deepar_forecast, LSTMForecaster  # noqa: F401
from report_phase2 import build_report_phase2


# ===========================================================================
# Helpers
# ===========================================================================

def _load_config(config_path: str) -> dict:
    """Parse config.yaml and return as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_backtest_setup(setup_path: Path) -> dict:
    """Load the Phase 1 backtest_setup.json written by run_phase1.py."""
    if not setup_path.exists():
        raise FileNotFoundError(
            f"Phase 1 backtest setup not found at {setup_path}.\n"
            "Run `python3 run_phase1.py` first to generate it."
        )
    with open(setup_path) as f:
        return json.load(f)


def _load_phase1_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load the Phase 1 predictions.csv written by run_phase1.py."""
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Phase 1 predictions not found at {predictions_path}.\n"
            "Run `python3 run_phase1.py` first to generate them."
        )
    return pd.read_csv(predictions_path, parse_dates=["date", "origin"])


def _build_lstm_config(config: dict) -> dict:
    """
    Extract the ``lstm`` section from the config, applying sensible defaults
    for any missing keys so the script works even if config.yaml predates
    Phase 2.
    """
    defaults = {
        "epochs":            30,
        "hidden_size":       64,
        "num_layers":        2,
        "batch_size":        64,
        "context_length":    90,
        "prediction_length": 14,
        "dropout":           0.1,
        "learning_rate":     1e-3,
        "num_workers":       0,
        "device":            "cpu",
    }
    lstm_section = config.get("lstm", {})
    return {**defaults, **lstm_section}


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # -----------------------------------------------------------------------
    # 1. CLI arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Phase 2: DeepAR (global LSTM) backtest on M5 data"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild data/processed from the raw M5 CSV files",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Load config
    # -----------------------------------------------------------------------
    config     = _load_config(args.config)
    lstm_cfg   = _build_lstm_config(config)
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 3. Prepare dataset  (loads from disk unless --rebuild is set)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 2: DeepAR (global LSTM) demand forecasting")
    print("=" * 60)
    print("\nStep 1/4: preparing data")
    df = prepare_dataset(config, rebuild=args.rebuild)
    print(
        f"  {df['series_id'].nunique()} series, "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    # -----------------------------------------------------------------------
    # 4. Load Phase 1 artifacts (predictions + backtest setup)
    # -----------------------------------------------------------------------
    print("\nStep 2/4: loading Phase 1 backtest setup")
    setup_path       = reports_dir / "backtest_setup.json"
    predictions_path = reports_dir / "predictions.csv"

    setup              = _load_backtest_setup(setup_path)
    phase1_predictions = _load_phase1_predictions(predictions_path)

    # Reconstruct the same origins as Phase 1 (as Timestamps, oldest first)
    origins = [pd.Timestamp(o) for o in setup["origins"]]
    series_list_all = set(setup["series"])   # all series used in Phase 1
    horizon         = int(setup["horizon"])
    quantiles       = [float(q) for q in setup["quantiles"]]
    min_days        = int(setup["min_days_to_forecast"])
    cold_days       = int(setup["cold_start_days"])

    print(
        f"  {len(origins)} folds | horizon={horizon} days | "
        f"{len(series_list_all)} series"
    )

    # -----------------------------------------------------------------------
    # 5. Rolling-origin DeepAR backtest
    # -----------------------------------------------------------------------
    print("\nStep 3/4: rolling-origin DeepAR backtest")
    all_deepar_rows = []

    for fold, origin in enumerate(origins, start=1):
        print(f"\n  Fold {fold}/{len(origins)}: origin {origin.date()}")

        # --- (a) Train / test split ---
        train = df[df["date"] <= origin].drop(columns=["is_stockout"])
        # Re-detect stockouts from training data only (no future leakage)
        train = mark_stockouts(
            train,
            config["stockouts"]["min_zero_run"],
            config["stockouts"]["max_chance"],
        )

        test_end = origin + pd.Timedelta(days=horizon)
        test = df[(df["date"] > origin) & (df["date"] <= test_end)].copy()

        # --- (b) Series that have enough history ---
        series_ready, days_live = series_ready_to_forecast(train, origin, min_days)
        # Intersect with the Phase 1 series list (same series, same fold)
        series_list = sorted(series_list_all & series_ready)
        test = test[test["series_id"].isin(series_list)]
        print(f"    forecasting {len(series_list)} series")

        if not series_list:
            print("    [skip] no series ready for this fold")
            continue

        # --- (c) Build the future feature DataFrame (no demand / stockout) ---
        future = test.drop(
            columns=[c for c in ["demand", "is_stockout"] if c in test.columns]
        )

        # --- (d) DeepAR forecast ---
        forecast_df = deepar_forecast(
            train=train,
            future=future,
            series_list=series_list,
            horizon=horizon,
            quantiles=quantiles,
            config={"lstm": lstm_cfg},
        )
        # forecast_df expected columns: series_id, date, step, point, q10, q50, q90

        # --- (e) Join with ground truth ---
        truth = test[["series_id", "date", "demand", "is_stockout", "is_active"]].rename(
            columns={"demand": "actual"}
        )
        forecast_df = forecast_df.merge(truth, on=["series_id", "date"], how="left")

        # --- (f) Add metadata columns (match Phase 1 format) ---
        forecast_df["model"]  = "DeepAR"
        forecast_df["fold"]   = fold
        forecast_df["origin"] = origin

        forecast_df["days_since_launch_at_origin"] = (
            forecast_df["series_id"].map(days_live)
        )
        forecast_df["is_cold_start"] = (
            forecast_df["days_since_launch_at_origin"] < cold_days
        ).astype(int)

        # Scoring rule: active, not a stockout, actual demand known
        forecast_df["is_scored"] = (
            (forecast_df["is_active"] == 1)
            & (forecast_df["is_stockout"] == 0)
            & forecast_df["actual"].notna()
        ).astype(int)

        all_deepar_rows.append(forecast_df)

    # -----------------------------------------------------------------------
    # 6. Combine Phase 1 + DeepAR predictions and save
    # -----------------------------------------------------------------------
    if not all_deepar_rows:
        print("\n[WARNING] No DeepAR predictions were produced. Exiting.")
        sys.exit(0)

    deepar_predictions = pd.concat(all_deepar_rows, ignore_index=True)

    # Align columns with Phase 1 output
    p1_cols = [
        "model", "fold", "origin", "series_id", "date", "step",
        "actual", "point", "q10", "q50", "q90",
        "is_stockout", "is_active", "is_cold_start",
        "days_since_launch_at_origin", "is_scored",
    ]
    # Keep only columns that exist in deepar_predictions (graceful if some missing)
    deepar_cols = [c for c in p1_cols if c in deepar_predictions.columns]
    deepar_predictions = deepar_predictions[deepar_cols]

    # Append DeepAR rows to Phase 1 predictions
    all_predictions = pd.concat(
        [phase1_predictions, deepar_predictions], ignore_index=True
    )

    out_path = reports_dir / "predictions_phase2.csv"
    all_predictions.to_csv(out_path, index=False)
    print(f"\n  Saved combined predictions → {out_path}")
    print(
        f"  Phase 1 rows : {len(phase1_predictions):,}  |  "
        f"DeepAR rows : {len(deepar_predictions):,}"
    )

    # -----------------------------------------------------------------------
    # 7. Generate Phase 2 report
    # -----------------------------------------------------------------------
    print("\nStep 4/4: generating Phase 2 report")
    build_report_phase2(df, all_predictions, origins, config, reports_dir)

    # -----------------------------------------------------------------------
    # 8. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary (scored days only)")
    print("=" * 60)
    scored = all_predictions[all_predictions["is_scored"] == 1].copy()
    if scored.empty:
        print("  (no scored rows found)")
    else:
        # Quick MAE per model
        scored["abs_error"] = (scored["actual"] - scored["point"]).abs()
        summary = (
            scored.groupby("model")["abs_error"]
            .mean()
            .rename("MAE")
            .reset_index()
            .sort_values("MAE")
        )
        print(summary.to_string(index=False))

    print(f"\nDone. Full report → {reports_dir / 'phase2_report.md'}")


if __name__ == "__main__":
    main()
