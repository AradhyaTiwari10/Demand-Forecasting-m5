"""
Forecast accuracy metrics.

Point forecast:  MAE, RMSE, WAPE
Quantiles:       pinball loss for each quantile, and how often the actual value falls
                 inside the interval between the lowest and highest quantile (coverage)
"""

import numpy as np


def quantile_column(q):
    """0.1 -> 'q10', 0.5 -> 'q50', 0.9 -> 'q90'"""
    return "q" + str(int(round(q * 100)))


def mae(actual, forecast):
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def wape(actual, forecast):
    """Total absolute error divided by total sales. Undefined (NaN) when nothing was sold."""
    total_sales = np.sum(np.abs(actual))
    if total_sales == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - forecast)) / total_sales)


def pinball(actual, quantile_forecast, q):
    """Pinball (quantile) loss.

    For q = 0.9, forecasting too low costs 0.9 per unit and forecasting too high costs 0.1 per unit,
    so the best forecast is the value that is exceeded only 10% of the time.
    """
    diff = actual - quantile_forecast
    loss = np.maximum(q * diff, (q - 1) * diff)
    return float(np.mean(loss))


def coverage(actual, lower, upper):
    """Share of actual values that fall inside [lower, upper]."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def score(rows, quantiles):
    """All metrics for one set of forecast rows.

    `rows` needs the columns: actual, point, and one column per quantile (q10, q50, q90).
    """
    actual = rows["actual"].values
    point = rows["point"].values

    result = {
        "rows": len(rows),
        "MAE": mae(actual, point),
        "RMSE": rmse(actual, point),
        "WAPE": wape(actual, point),
    }

    losses = []
    for q in quantiles:
        name = quantile_column(q)
        loss = pinball(actual, rows[name].values, q)
        result["pinball_" + name] = loss
        losses.append(loss)
    result["mean_pinball"] = float(np.mean(losses))

    low = quantile_column(min(quantiles))
    high = quantile_column(max(quantiles))
    result["coverage"] = coverage(actual, rows[low].values, rows[high].values)
    return result
