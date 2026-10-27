"""Checks for backtest.py (run with: python3 -m pytest tests/)"""

import pandas as pd

from helpers import make_series
from backtest import make_origins, series_ready_to_forecast


def test_origins_are_evenly_spaced_and_leave_room_for_the_horizon():
    df = make_series([1] * 100)
    origins = make_origins(df, horizon=14, num_origins=3, step_days=14)
    assert origins[-1] == df["date"].max() - pd.Timedelta(days=14)
    assert (origins[1] - origins[0]).days == 14


def test_items_with_too_little_history_are_not_forecast():
    old = make_series([1] * 60, series_id="OLD")
    new = make_series([1] * 60, series_id="NEW")
    new["launch_date"] = pd.Timestamp("2020-02-27")      # 3 days before the origin below
    train = pd.concat([old, new], ignore_index=True)
    ready, _ = series_ready_to_forecast(train, pd.Timestamp("2020-02-29"), min_days=7)
    assert ready == {"OLD"}
