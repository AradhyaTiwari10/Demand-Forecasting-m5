"""Checks for baselines.py (run with: python3 -m pytest tests/)"""

import numpy as np
import pandas as pd

from helpers import make_series
from baselines import GBDTForecaster, seasonal_naive_forecast


def test_seasonal_naive_repeats_last_week():
    df = make_series(list(range(1, 8)) * 4)
    forecast = seasonal_naive_forecast(df, {"A"}, horizon=14, quantiles=[0.1, 0.5, 0.9])
    assert forecast["point"].tolist() == list(range(1, 8)) * 2


def test_gbdt_quantiles_are_ordered_and_median_is_not_zero():
    rng = np.random.RandomState(0)
    train = pd.concat([
        make_series(rng.poisson(5, 150), series_id="A"),
        make_series(rng.poisson(8, 150), series_id="B"),
    ], ignore_index=True)
    config = {
        "features": {"lags": [1, 7], "rolling_windows": [7]},
        "backtest": {"quantiles": [0.1, 0.5, 0.9]},
        "gbdt": {"max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 20},
    }
    future = pd.concat([make_series([0] * 7, series_id=s, start="2020-05-30") for s in ["A", "B"]])
    future["launch_date"] = pd.Timestamp("2020-01-01")
    forecast = GBDTForecaster(config).fit(train).forecast(train, future.drop(columns=["demand"]))
    assert len(forecast) == 14
    assert (forecast["q10"] <= forecast["q50"]).all() and (forecast["q50"] <= forecast["q90"]).all()
    assert forecast["q50"].mean() > 2
