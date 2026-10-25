"""Checks for kalman_filter.py (run with: python3 -m pytest tests/)"""

import numpy as np
import pandas as pd

from helpers import make_series
from kalman_filter import fit_series, forecast_series, run_filter


def test_kalman_holds_level_and_grows_uncertainty_on_stockout_days():
    values = [5.0] * 10 + [0.0] * 5
    usable = [True] * 10 + [False] * 5
    run = run_filter(values, usable, q=0.1, r=1.0, start_level=5.0, start_P=1.0)
    assert run["levels"][14] == run["levels"][9]                  # zeros did not pull the level down
    assert np.all(np.diff(run["level_variances"][9:]) > 0)        # uncertainty kept growing


def test_kalman_forecast_follows_level_and_interval_widens():
    rng = np.random.RandomState(1)
    history = make_series(rng.poisson(4, 120))
    fit = fit_series(history, ratios=[0.01, 0.1], min_weeks=4)
    rows = forecast_series(fit, history["date"].max(), horizon=14, quantiles=[0.1, 0.5, 0.9])
    forecast = pd.DataFrame(rows)
    assert abs(forecast["point"].mean() - 4) < 1.5
    width = forecast["q90"] - forecast["q10"]
    assert width.iloc[-1] >= width.iloc[0]
