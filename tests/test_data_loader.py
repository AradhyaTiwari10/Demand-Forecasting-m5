"""Checks for data_loader.py (run with: python3 -m pytest tests/)"""

import numpy as np

from helpers import make_series
from data_loader import make_complete_grid, mark_stockouts


def test_long_zero_run_is_marked_as_stockout():
    demand = [5] * 30 + [0] * 5 + [5] * 30
    df = mark_stockouts(make_series(demand).drop(columns=["is_stockout"]))
    assert df["is_stockout"].iloc[30:35].sum() == 5
    assert df["is_stockout"].sum() == 5
    assert df["demand"].tolist() == [float(x) for x in demand]   # sales are not changed


def test_short_zero_runs_of_a_slow_item_are_not_stockouts():
    demand = [1, 0, 0] * 20     # sells 0.33 a day, two zero days in a row are normal
    df = mark_stockouts(make_series(demand).drop(columns=["is_stockout"]))
    assert df["is_stockout"].sum() == 0


def test_complete_grid_adds_missing_days_as_nan():
    df = make_series([1, 2, 3, 4, 5]).drop(index=2)
    full = make_complete_grid(df)
    assert len(full) == 5
    assert np.isnan(full["demand"].iloc[2])
