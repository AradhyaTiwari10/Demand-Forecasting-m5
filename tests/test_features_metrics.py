"""Checks for features.py and metrics.py (run with: python3 -m pytest tests/)"""

import numpy as np

from helpers import make_series
from features import build_features
from metrics import mae, pinball, rmse, wape


def test_lag_and_rolling_features_only_use_the_past():
    df = make_series(list(range(1, 41)))      # demand on day i is i + 1
    table = build_features(df, lags=[1, 7], windows=[7])
    row = table[table["date"] == df["date"].iloc[20]].iloc[0]   # demand that day is 21
    assert row["lag_1"] == 20
    assert row["lag_7"] == 14
    assert row["mean_7"] == np.mean(range(14, 21))


def test_stockout_days_are_missing_in_lag_features():
    df = make_series([3] * 20)
    df.loc[10, "is_stockout"] = 1
    table = build_features(df, lags=[1], windows=[7])
    assert np.isnan(table["lag_1"].iloc[11])


def test_metrics_on_simple_numbers():
    actual = np.array([0.0, 2.0, 4.0])
    forecast = np.array([1.0, 2.0, 2.0])
    assert mae(actual, forecast) == 1.0
    assert abs(rmse(actual, forecast) - np.sqrt(5 / 3)) < 1e-9
    assert wape(actual, forecast) == 0.5
    assert abs(pinball(np.array([4.0]), np.array([2.0]), 0.9) - 1.8) < 1e-9
    assert np.isnan(wape(np.zeros(3), forecast))
