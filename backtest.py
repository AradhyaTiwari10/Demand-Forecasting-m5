"""
Rolling-origin backtest.

For every origin (cut-off date):
  1. train data = everything up to and including the origin
  2. stockouts are re-detected using the train data only, so no model sees the future
  3. each model forecasts the next `horizon` days for the same list of series
  4. forecasts are joined with what really happened

Fairness rules (the same for every model, and for DeepAR in Phase 2):
  - same series, same origins, same horizon
  - a series is forecast only if it has at least `min_days_to_forecast` days of history
  - days before launch and stockout days are not scored (the recorded 0 is not real demand)
"""

import json
from pathlib import Path

import pandas as pd

from baselines import GBDTForecaster, seasonal_naive_forecast
from data_loader import mark_stockouts
from kalman_filter import kalman_forecast


def make_origins(df, horizon, num_origins, step_days):
    """Origins counted back from the last date, oldest first."""
    last_date = df["date"].max()
    origins = []
    for i in range(num_origins):
        origin = last_date - pd.Timedelta(days=horizon + i * step_days)
        origins.append(origin)
    origins.reverse()
    return origins


def series_ready_to_forecast(train, origin, min_days):
    """Series that have at least `min_days` days since launch at the origin."""
    launch = train.groupby("series_id")["launch_date"].first()
    days_live = (origin - launch).dt.days + 1
    return set(days_live[days_live >= min_days].index), days_live


def run_backtest(df, config):
    horizon = config["backtest"]["horizon"]
    quantiles = config["backtest"]["quantiles"]
    min_days = config["cold_start"]["min_days_to_forecast"]
    cold_days = config["cold_start"]["max_days_since_launch"]

    origins = make_origins(df, horizon, config["backtest"]["num_origins"], config["backtest"]["step_days"])
    all_predictions = []

    for fold, origin in enumerate(origins, start=1):
        print("\nFold " + str(fold) + "/" + str(len(origins)) + ": origin " + str(origin.date()))

        # 1. train data, with stockouts detected from the past only
        train = df[df["date"] <= origin].drop(columns=["is_stockout"])
        train = mark_stockouts(train, config["stockouts"]["min_zero_run"], config["stockouts"]["max_chance"])

        # 2. test data (the future)
        test_end = origin + pd.Timedelta(days=horizon)
        test = df[(df["date"] > origin) & (df["date"] <= test_end)]

        series_list, days_live = series_ready_to_forecast(train, origin, min_days)
        test = test[test["series_id"].isin(series_list)]
        print("  forecasting " + str(len(series_list)) + " series")

        # 3. forecasts from every model
        predictions = {}
        print("  seasonal naive...")
        predictions["SeasonalNaive"] = seasonal_naive_forecast(train, series_list, horizon, quantiles)
        print("  kalman filter...")
        predictions["KalmanFilter"] = kalman_forecast(train, series_list, horizon, quantiles, config)
        print("  gbdt...")
        future = test.drop(columns=["demand", "is_stockout"])
        predictions["GBDT"] = GBDTForecaster(config).fit(train).forecast(train, future)

        # 4. join with what really happened
        truth = test[["series_id", "date", "demand", "is_stockout", "is_active"]]
        truth = truth.rename(columns={"demand": "actual"})
        for model_name, forecast in predictions.items():
            forecast = forecast.merge(truth, on=["series_id", "date"], how="left")
            forecast["model"] = model_name
            forecast["fold"] = fold
            forecast["origin"] = origin
            forecast["days_since_launch_at_origin"] = forecast["series_id"].map(days_live)
            forecast["is_cold_start"] = (forecast["days_since_launch_at_origin"] < cold_days).astype(int)
            all_predictions.append(forecast)

    predictions = pd.concat(all_predictions, ignore_index=True)

    # rows we score: the item was launched and the recorded sales are real demand
    predictions["is_scored"] = (
        (predictions["is_active"] == 1) & (predictions["is_stockout"] == 0) & predictions["actual"].notna()
    ).astype(int)

    columns = ["model", "fold", "origin", "series_id", "date", "step", "actual", "point",
               "q10", "q50", "q90", "is_stockout", "is_active", "is_cold_start",
               "days_since_launch_at_origin", "is_scored"]
    return predictions[columns], origins


def save_backtest_setup(df, origins, config, path):
    """Write the exact series and origins to a file, so Phase 2 can reuse the same split."""
    setup = {
        "series": sorted(df["series_id"].unique().tolist()),
        "origins": [str(o.date()) for o in origins],
        "horizon": config["backtest"]["horizon"],
        "quantiles": config["backtest"]["quantiles"],
        "min_days_to_forecast": config["cold_start"]["min_days_to_forecast"],
        "cold_start_days": config["cold_start"]["max_days_since_launch"],
    }
    Path(path).write_text(json.dumps(setup, indent=2))
