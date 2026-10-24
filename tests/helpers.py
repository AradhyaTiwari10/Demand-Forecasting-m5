"""Shared helper for the tests: builds a tiny fake series with the same columns as the real data."""

import sys
from pathlib import Path

# make the project files importable from the tests folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def make_series(demand, series_id="A", start="2020-01-01"):
    dates = pd.date_range(start, periods=len(demand), freq="D")
    return pd.DataFrame({
        "series_id": series_id,
        "item_id": series_id,
        "dept_id": "D",
        "cat_id": "C",
        "store_id": "S",
        "state_id": "CA",
        "date": dates,
        "demand": np.array(demand, dtype=float),
        "launch_date": pd.Timestamp(start),
        "is_active": 1,
        "is_stockout": 0,
        "cat_code": 0,
        "dept_code": 0,
        "store_code": 0,
        "has_event": 0,
        "snap": 0,
        "sell_price": 1.0,
    })
