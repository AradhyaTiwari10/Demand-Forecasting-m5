"""
Rolling-origin backtest for Phase 1.

Runs all models (Seasonal Naive, Kalman Filter, GBDT) across rolling forecast origins
with identical data slices, zero look-ahead, and fair stockout / cold-start evaluation.

Every model produces predictions adhering to the schema:
    [series_id, date, step, point, q10, q50, q90, model, origin, ...]
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from baselines import GBDTForecaster, seasonal_naive_forecast
from data_loader import mark_stockouts, prepare_dataset
from kalman_filter import kalman_forecast
from metrics import score


def compute_origins(df, config):
    """Compute the list of rolling forecast origins (cut-off dates).

    The latest origin is horizon days before the end of the dataset, so that
    a full forecast window of actuals is available for scoring.
    Earlier origins step back by step_days.
    """
    max_date = pd.to_datetime(df["date"].max())
    horizon = int(config["backtest"]["horizon"])
    num_origins = int(config["backtest"]["num_origins"])
    step_days = int(config["backtest"]["step_days"])

    latest_origin = max_date - pd.Timedelta(days=horizon)
    origins = [latest_origin - pd.Timedelta(days=i * step_days) for i in range(num_origins)]
    return sorted(origins)


def classify_series_at_origin(df, origin, config):
    """Determine series eligibility and cold-start status at a given origin.

    Returns a DataFrame with columns: series_id, days_since_launch, is_cold_start.
    Series with fewer than min_days_to_forecast are excluded.
    """
    min_days = config["cold_start"]["min_days_to_forecast"]
    max_days = config["cold_start"]["max_days_since_launch"]

    # One row per series to inspect launch_date
    series_info = df.groupby("series_id", as_index=False)["launch_date"].first()
    series_info["days_since_launch"] = (origin - pd.to_datetime(series_info["launch_date"])).dt.days

    # Keep only series that launched at least min_days before the origin
    eligible = series_info[series_info["days_since_launch"] >= min_days].copy()
    eligible["is_cold_start"] = eligible["days_since_launch"] < max_days
    return eligible


def run_backtest_origin(df, origin, config, models=None):
    """Run all models for a single forecast origin.

    Ensures zero look-ahead:
    - Training data is strictly date <= origin.
    - Training stockout flags are computed only from data up to origin.
    - Future rows provide known calendar/price features, but demand is hidden.
    """
    if models is None:
        models = ["seasonal_naive", "kalman", "gbdt"]

    horizon = int(config["backtest"]["horizon"])
    quantiles = config["backtest"]["quantiles"]
    min_zero_run = config["stockouts"]["min_zero_run"]
    max_chance = config["stockouts"]["max_chance"]

    series_status = classify_series_at_origin(df, origin, config)
    eligible_series = set(series_status["series_id"])

    if len(eligible_series) == 0:
        return pd.DataFrame()

    # Strictly past data for training
    train = df[(df["date"] <= origin) & (df["series_id"].isin(eligible_series))].copy()
    if "is_stockout" in train.columns:
        train = train.drop(columns=["is_stockout"])
    train = mark_stockouts(train, min_zero_run=min_zero_run, max_chance=max_chance)

    # Future window for evaluation
    future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = df[(df["date"].isin(future_dates)) & (df["series_id"].isin(eligible_series))].copy()

    model_preds = []

    if "seasonal_naive" in models:
        sn_preds = seasonal_naive_forecast(train, eligible_series, horizon, quantiles)
        sn_preds["model"] = "seasonal_naive"
        model_preds.append(sn_preds)

    if "kalman" in models:
        kf_preds = kalman_forecast(train, eligible_series, horizon, quantiles, config)
        kf_preds["model"] = "kalman"
        model_preds.append(kf_preds)

    if "gbdt" in models:
        gbdt = GBDTForecaster(config)
        gbdt.fit(train)
        gbdt_preds = gbdt.forecast(train, future)
        gbdt_preds["model"] = "gbdt"
        model_preds.append(gbdt_preds)

    all_preds = pd.concat(model_preds, ignore_index=True)
    all_preds["origin"] = origin

    # Merge with future ground truth and metadata
    meta_cols = [
        c for c in ["series_id", "date", "demand", "is_stockout", "is_active",
                    "cat_id", "dept_id", "store_id", "state_id"]
        if c in future.columns
    ]
    ground_truth = future[meta_cols].rename(columns={"demand": "actual"})

    merged = all_preds.merge(ground_truth, on=["series_id", "date"], how="left")
    merged = merged.merge(
        series_status[["series_id", "is_cold_start", "days_since_launch"]],
        on="series_id",
        how="left"
    )

    return merged


def run_backtest(df, config, models=None, verbose=True):
    """Run the rolling-origin backtest across all origins.

    Returns a single concatenated DataFrame of all predictions and actuals.
    """
    origins = compute_origins(df, config)
    all_results = []

    for i, origin in enumerate(origins, 1):
        if verbose:
            origin_str = pd.to_datetime(origin).strftime("%Y-%m-%d")
            print(f"Running backtest origin {i}/{len(origins)}: {origin_str}...")
        origin_preds = run_backtest_origin(df, origin, config, models=models)
        all_results.append(origin_preds)

    return pd.concat(all_results, ignore_index=True)


def evaluate_predictions(predictions, quantiles):
    """Compute benchmark metrics across models, segments, origins, and steps.

    Scoring rule:
    Only evaluate days when the item is active, not during a stockout, and actual demand is recorded.
    """
    usable = (predictions["is_active"] == 1) & (predictions["is_stockout"] == 0) & predictions["actual"].notna()
    scorable = predictions[usable].copy()

    # 1. Overall metrics per model
    overall_rows = []
    for model_name, grp in scorable.groupby("model", sort=False):
        res = score(grp, quantiles)
        res["model"] = model_name
        overall_rows.append(res)
    overall_df = pd.DataFrame(overall_rows)
    if not overall_df.empty:
        cols = ["model"] + [c for c in overall_df.columns if c != "model"]
        overall_df = overall_df[cols]

    # 2. Segment metrics (Established vs Cold Start)
    segment_rows = []
    for (model_name, is_cold), grp in scorable.groupby(["model", "is_cold_start"], sort=False):
        res = score(grp, quantiles)
        res["model"] = model_name
        res["segment"] = "cold_start" if is_cold else "established"
        segment_rows.append(res)
    segment_df = pd.DataFrame(segment_rows)
    if not segment_df.empty:
        cols = ["model", "segment"] + [c for c in segment_df.columns if c not in ["model", "segment"]]
        segment_df = segment_df[cols]

    # 3. Metrics by origin
    origin_rows = []
    for (model_name, origin), grp in scorable.groupby(["model", "origin"], sort=False):
        res = score(grp, quantiles)
        res["model"] = model_name
        res["origin"] = pd.to_datetime(origin).strftime("%Y-%m-%d")
        origin_rows.append(res)
    origin_df = pd.DataFrame(origin_rows)
    if not origin_df.empty:
        cols = ["model", "origin"] + [c for c in origin_df.columns if c not in ["model", "origin"]]
        origin_df = origin_df[cols]

    # 4. Metrics by forecast step (1 to horizon)
    step_rows = []
    for (model_name, step), grp in scorable.groupby(["model", "step"], sort=False):
        res = score(grp, quantiles)
        res["model"] = model_name
        res["step"] = int(step)
        step_rows.append(res)
    step_df = pd.DataFrame(step_rows)
    if not step_df.empty:
        cols = ["model", "step"] + [c for c in step_df.columns if c not in ["model", "step"]]
        step_df = step_df[cols]

    # 5. Metrics by category (if cat_id is present)
    cat_df = pd.DataFrame()
    if "cat_id" in scorable.columns and scorable["cat_id"].notna().any():
        cat_rows = []
        for (model_name, cat), grp in scorable.groupby(["model", "cat_id"], sort=False):
            res = score(grp, quantiles)
            res["model"] = model_name
            res["cat_id"] = cat
            cat_rows.append(res)
        cat_df = pd.DataFrame(cat_rows)
        if not cat_df.empty:
            cols = ["model", "cat_id"] + [c for c in cat_df.columns if c not in ["model", "cat_id"]]
            cat_df = cat_df[cols]

    return {
        "overall": overall_df,
        "by_segment": segment_df,
        "by_origin": origin_df,
        "by_step": step_df,
        "by_category": cat_df,
    }


def save_backtest_results(predictions, metrics, config, output_dir=None):
    """Save predictions, evaluation metric CSVs, and backtest_setup.json."""
    if output_dir is None:
        output_dir = Path(config["output"]["reports_dir"])
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions.csv (README: one row per model, series and forecast day)
    pred_file = output_dir / "predictions.csv"
    predictions.to_csv(pred_file, index=False)
    print("Saved predictions to " + str(pred_file))

    # Save backtest_setup.json (README: exact series and origins, so DeepAR can be scored on the same split)
    origins = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in sorted(predictions["origin"].unique())]
    series_ids = sorted(predictions["series_id"].unique().tolist())
    models = sorted(predictions["model"].unique().tolist())

    setup = {
        "num_origins": len(origins),
        "origins": origins,
        "horizon": int(config["backtest"]["horizon"]),
        "step_days": int(config["backtest"]["step_days"]),
        "quantiles": [float(q) for q in config["backtest"]["quantiles"]],
        "cold_start_max_days": int(config["cold_start"]["max_days_since_launch"]),
        "min_days_to_forecast": int(config["cold_start"]["min_days_to_forecast"]),
        "num_series": len(series_ids),
        "series_ids": series_ids,
        "models": models,
    }
    setup_file = output_dir / "backtest_setup.json"
    with open(setup_file, "w") as f:
        json.dump(setup, f, indent=2)
    print("Saved backtest setup to " + str(setup_file))

    # Save metric CSVs
    if "overall" in metrics and not metrics["overall"].empty:
        metrics["overall"].to_csv(output_dir / "metrics_overall.csv", index=False)
    if "by_segment" in metrics and not metrics["by_segment"].empty:
        metrics["by_segment"].to_csv(output_dir / "metrics_by_segment.csv", index=False)
    if "by_origin" in metrics and not metrics["by_origin"].empty:
        metrics["by_origin"].to_csv(output_dir / "metrics_by_origin.csv", index=False)
    if "by_step" in metrics and not metrics["by_step"].empty:
        metrics["by_step"].to_csv(output_dir / "metrics_by_step.csv", index=False)
    if "by_category" in metrics and not metrics["by_category"].empty:
        metrics["by_category"].to_csv(output_dir / "metrics_by_category.csv", index=False)

    print("Saved all metric summaries to " + str(output_dir))


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    df = prepare_dataset(config)
    predictions = run_backtest(df, config)
    metrics = evaluate_predictions(predictions, config["backtest"]["quantiles"])
    save_backtest_results(predictions, metrics, config)
    print("\nOverall backtest results:")
    print(metrics["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
