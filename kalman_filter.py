"""
Local demand tracking with a Kalman filter: one small filter per series.

The model (a "local level" model):
    demand level today   = demand level yesterday + small random change   (variance q)
    sales today          = demand level x weekday effect + noise          (variance r)

Each day the filter does two steps:
    predict:  keep the level, but become a bit less sure about it (P grows by q)
    update:   compare today's sales with the level and move part of the way towards them;
              the Kalman gain K (between 0 and 1) decides how far

Stockout days and days before launch are skipped in the update step. The filter keeps
its level and only becomes less sure, so an empty shelf does not pull demand down to zero.

Per series we estimate:
    - the weekday effects (e.g. Saturdays sell 1.3x an average day)
    - q / r, i.e. how quickly the level is allowed to move (chosen from a short list)
    - r, the noise size, so that the filter's uncertainty matches its real errors
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from metrics import quantile_column


def weekday_factors(days_of_week, values, min_weeks):
    """How much busier each weekday is than an average day (1.0 = average).

    Returns seven 1.0 values when there is not enough history (for example for a new item).
    """
    factors = np.ones(7)
    if len(values) < min_weeks * 7 or np.mean(values) <= 0:
        return factors

    overall = np.mean(values)
    for day in range(7):
        on_this_day = values[days_of_week == day]
        if len(on_this_day) > 0:
            factors[day] = np.mean(on_this_day) / overall

    # keep factors in a sensible range so we never divide by (almost) zero
    return np.clip(factors, 0.5, 2.0)


def run_filter(values, usable, q, r, start_level, start_P):
    """Run the Kalman filter once over the history.

    values: weekday-adjusted sales, one per day
    usable: True on days we can learn from (in stock, after launch, not missing)
    """
    level = start_level
    P = start_P
    errors = []           # one-day-ahead errors on usable days
    error_variances = []  # how big the filter expected those errors to be
    levels = []
    level_variances = []

    for t in range(len(values)):
        # 1) predict: same level as yesterday, a bit less certain
        P = P + q

        # 2) update with today's sales (skipped on stockout / missing days)
        if usable[t]:
            S = P + r                  # expected variance of today's sales
            K = P / S                  # Kalman gain
            error = values[t] - level  # surprise
            level = level + K * error
            P = (1 - K) * P
            errors.append(error)
            error_variances.append(S)

        levels.append(level)
        level_variances.append(P)

    return {
        "level": level,
        "P": P,
        "errors": np.array(errors),
        "error_variances": np.array(error_variances),
        "levels": np.array(levels),
        "level_variances": np.array(level_variances),
    }


def fit_series(history, ratios, min_weeks):
    """Fit the filter to one series and return its final state."""
    history = history.sort_values("date")
    history = history[history["is_active"] == 1]       # start on the launch day

    demand = history["demand"].values
    days_of_week = history["date"].dt.dayofweek.values
    usable = (history["is_stockout"].values == 0) & ~np.isnan(demand)

    factors = weekday_factors(days_of_week[usable], demand[usable], min_weeks)
    adjusted = demand / factors[days_of_week]           # remove the weekday effect

    first_week = adjusted[usable][:7]
    start_level = float(np.mean(first_week)) if len(first_week) > 0 else 0.0

    # Plain Python lists make the day-by-day loop faster
    values = adjusted.tolist()
    usable_list = usable.tolist()

    # Try each q/r ratio with r = 1 and keep the one with the smallest one-day-ahead error
    best_ratio = ratios[0]
    best_mse = None
    best_run = None
    for ratio in ratios:
        run = run_filter(values, usable_list, q=ratio, r=1.0, start_level=start_level, start_P=1.0)
        errors = run["errors"][7:]                      # ignore the first week while it settles
        if len(errors) == 0:
            errors = run["errors"]
        mse = float(np.mean(errors ** 2)) if len(errors) > 0 else 0.0
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_ratio = ratio
            best_run = run

    # Pick r so that, on average, the real squared errors match the filter's expected ones
    if len(best_run["errors"]) > 1:
        r = float(np.mean(best_run["errors"] ** 2 / best_run["error_variances"]))
    else:
        r = 1.0
    r = max(r, 1e-3)
    q = best_ratio * r

    final = run_filter(values, usable_list, q=q, r=r, start_level=start_level, start_P=r)
    final["q"] = q
    final["r"] = r
    final["factors"] = factors
    final["dates"] = history["date"].values
    final["demand"] = demand
    final["usable"] = usable
    return final


def forecast_series(fit, origin, horizon, quantiles):
    """Forecast `horizon` days ahead from the filter's final state.

    The level stays flat, the weekday effect is put back, and the uncertainty
    grows with every step ahead (P + step * q).
    """
    rows = []
    for step in range(1, horizon + 1):
        day = origin + pd.Timedelta(days=step)
        factor = fit["factors"][day.dayofweek]

        mean = fit["level"] * factor
        variance = fit["P"] + step * fit["q"] + fit["r"]
        std = math.sqrt(variance) * factor

        row = {"series_id": None, "date": day, "step": step, "point": max(0.0, mean)}
        for q in quantiles:
            row[quantile_column(q)] = max(0.0, mean + norm.ppf(q) * std)
        rows.append(row)
    return rows


def kalman_forecast(train, series_list, horizon, quantiles, config):
    """Fit one filter per series and forecast."""
    ratios = config["kalman"]["noise_ratios"]
    min_weeks = config["kalman"]["min_weeks_for_weekly_pattern"]
    origin = train["date"].max()

    all_rows = []
    for series_id, history in train.groupby("series_id"):
        if series_id not in series_list:
            continue
        fit = fit_series(history, ratios, min_weeks)
        rows = forecast_series(fit, origin, horizon, quantiles)
        for row in rows:
            row["series_id"] = series_id
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)
