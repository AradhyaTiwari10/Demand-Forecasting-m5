"""
Feature engineering for the GBDT baseline (the same columns can be reused by DeepAR in Phase 2).

The one rule: a feature for day t may only use sales from day t-1 or earlier.
Calendar, price and launch information are allowed because they are known in advance.
"""

STATIC_FEATURES = ["cat_code", "dept_code", "store_code"]
KNOWN_IN_ADVANCE = ["day_of_week", "day_of_month", "month", "has_event", "snap", "sell_price", "days_since_launch"]


def add_calendar_features(df):
    """Add simple calendar columns."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days
    return df


def add_demand_features(df, lags, windows):
    """Add lag and rolling-mean columns, built only from past days.

    Stockout days and days before launch are set to NaN first ("clean demand"),
    so an empty-shelf zero is never used as if it were real demand.
    The tree model can handle NaN values directly.
    """
    df = df.sort_values(["series_id", "date"]).copy()
    usable = (df["is_active"] == 1) & (df["is_stockout"] == 0)
    df["clean_demand"] = df["demand"].where(usable)

    by_series = df.groupby("series_id")["clean_demand"]

    # lag_k = clean demand k days ago
    for lag in lags:
        df["lag_" + str(lag)] = by_series.shift(lag)

    # rolling means end yesterday (shift(1)), so today's sales are never included
    yesterday = by_series.shift(1)
    for window in windows:
        rolling_mean = yesterday.groupby(df["series_id"]).rolling(window, min_periods=1).mean()
        df["mean_" + str(window)] = rolling_mean.reset_index(level=0, drop=True)

    return df


def build_features(df, lags, windows):
    """Calendar features + demand features."""
    df = add_calendar_features(df)
    df = add_demand_features(df, lags, windows)
    return df


def feature_columns(lags, windows):
    """The list of columns the model is trained on (static columns first)."""
    columns = STATIC_FEATURES + KNOWN_IN_ADVANCE
    for lag in lags:
        columns.append("lag_" + str(lag))
    for window in windows:
        columns.append("mean_" + str(window))
    return columns
