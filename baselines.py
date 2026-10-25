"""
The two AML baselines.

1. Seasonal naive: "next Monday will look like last Monday".
2. GBDT: gradient boosted trees on lag, rolling, calendar, price and item features.

Every model in this project returns the same table:
    series_id, date, step, point, q10, q50, q90
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_features, feature_columns
from metrics import quantile_column


# ---------------------------------------------------------------------------
# 1. Seasonal naive
# ---------------------------------------------------------------------------

def seasonal_naive_forecast(train, series_list, horizon, quantiles):
    """Repeat the last observed week.

    For the uncertainty band we look at the errors this method would have made over
    the last 26 weeks (this week's sales minus last week's sales) and add their
    10% / 50% / 90% quantiles to the forecast.
    """
    origin = train["date"].max()
    rows = []

    for series_id, history in train.groupby("series_id"):
        if series_id not in series_list:
            continue

        history = history.sort_values("date")
        history = history[history["is_active"] == 1]
        demand = history["demand"].values
        last_week = demand[-7:]

        recent = demand[-182:]
        errors = recent[7:] - recent[:-7]
        errors = errors[~np.isnan(errors)]

        for step in range(1, horizon + 1):
            point = last_week[(step - 1) % 7]   # same weekday, last week
            if np.isnan(point):
                point = np.nanmean(demand)

            row = {
                "series_id": series_id,
                "date": origin + pd.Timedelta(days=step),
                "step": step,
                "point": point,
            }
            for q in quantiles:
                if len(errors) >= 10:
                    offset = np.quantile(errors, q)
                else:
                    offset = norm.ppf(q) * np.nanstd(demand)
                row[quantile_column(q)] = max(0.0, point + offset)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Gradient boosted trees
# ---------------------------------------------------------------------------

class GBDTForecaster:
    """One model for the point forecast plus one model per quantile."""

    def __init__(self, config):
        self.lags = config["features"]["lags"]
        self.windows = config["features"]["rolling_windows"]
        self.quantiles = config["backtest"]["quantiles"]
        self.settings = config["gbdt"]
        self.columns = feature_columns(self.lags, self.windows)
        self.point_model = None
        self.quantile_models = {}

    def make_model(self, loss, quantile=None):
        return HistGradientBoostingRegressor(
            loss=loss,
            quantile=quantile,
            max_iter=self.settings["max_iter"],
            learning_rate=self.settings["learning_rate"],
            max_leaf_nodes=self.settings["max_leaf_nodes"],
            min_samples_leaf=self.settings["min_samples_leaf"],
            categorical_features=[0, 1, 2],   # cat_code, dept_code, store_code
            # Early stopping is switched off on purpose: with many zero sales it stopped the
            # quantile models after a few rounds and the median forecast stayed at 0.
            early_stopping=False,
            random_state=42,
        )

    def fit(self, train):
        table = build_features(train, self.lags, self.windows)

        # learn only from real demand: after launch, not during a stockout, not missing
        keep = (table["is_active"] == 1) & (table["is_stockout"] == 0) & table["demand"].notna()
        table = table[keep]
        X = table[self.columns]
        y = table["demand"]

        # Poisson loss suits daily counts and never predicts negative sales
        self.point_model = self.make_model("poisson")
        self.point_model.fit(X, y)

        for q in self.quantiles:
            model = self.make_model("quantile", quantile=q)
            model.fit(X, y)
            self.quantile_models[q] = model
        return self

    def forecast(self, train, future):
        """Forecast day by day (recursive forecasting).

        `future` holds the rows we want to forecast. Its calendar and price columns are
        real (they are known in advance) but its demand is hidden. After each day is
        predicted, the prediction is written in as that day's demand, so the next day's
        lag_1 can use it.
        """
        origin = train["date"].max()
        horizon = (future["date"].max() - origin).days
        series_list = future["series_id"].unique()

        # 60 days of history is enough for the longest lag (28) plus the longest window (28)
        recent = train[train["series_id"].isin(series_list) & (train["date"] > origin - pd.Timedelta(days=60))]
        future = future.copy()
        future["demand"] = np.nan
        future["is_stockout"] = 0   # we assume the item will be in stock
        frame = pd.concat([recent, future], ignore_index=True)

        results = []
        for step in range(1, horizon + 1):
            day = origin + pd.Timedelta(days=step)
            table = build_features(frame, self.lags, self.windows)
            today = table[table["date"] == day]
            X = today[self.columns]

            out = pd.DataFrame({
                "series_id": today["series_id"].values,
                "date": day,
                "step": step,
                "point": self.point_model.predict(X),
            })

            # predict every quantile, then sort so that q10 <= q50 <= q90 always holds
            quantile_preds = []
            for q in self.quantiles:
                quantile_preds.append(self.quantile_models[q].predict(X))
            quantile_preds = np.sort(np.column_stack(quantile_preds), axis=1)
            for i, q in enumerate(self.quantiles):
                out[quantile_column(q)] = np.maximum(quantile_preds[:, i], 0.0)
            results.append(out)

            # feed today's point forecast back in as "demand" for tomorrow's features
            predicted = dict(zip(out["series_id"], out["point"]))
            is_today = frame["date"] == day
            frame.loc[is_today, "demand"] = frame.loc[is_today, "series_id"].map(predicted)

        return pd.concat(results, ignore_index=True)
