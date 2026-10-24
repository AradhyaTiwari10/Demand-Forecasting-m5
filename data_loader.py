"""
Data preparation for Phase 1.

Reads the raw M5 files and builds one table with one row per series per day:

    series_id, item_id, dept_id, cat_id, store_id, state_id, date, demand,
    sell_price, has_event, snap, launch_date, is_active,
    cat_code, dept_code, store_code, is_stockout

Two important choices:
- We never change the sales numbers. Stockout days are only marked (is_stockout = 1).
- An item is "active" from the day of its first ever sale (its launch date).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]


def check_raw_files(raw_dir):
    """Stop with a clear message if an M5 file is missing."""
    missing = []
    for name in REQUIRED_FILES:
        if not (Path(raw_dir) / name).exists():
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "These M5 files are missing from '" + str(raw_dir) + "': " + ", ".join(missing) + "\n"
            "Download them from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
            "and copy them into that folder."
        )


def pick_series(sales, config):
    """Choose which series we use.

    - established items: launched before our history window, a fixed number per category
    - new items: launched close to the end of the data, so they are cold start in the backtest
    """
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    total_days = len(day_columns)
    has_sale = sales[day_columns].values > 0
    first_sale_day = has_sale.argmax(axis=1)   # position of the first day with a sale

    window = config["data"]["days_of_history"]
    rng = np.random.RandomState(config["data"]["random_seed"])
    chosen_rows = []

    # Established items
    is_established = first_sale_day < total_days - window
    for category in CATEGORIES:
        candidates = np.where(is_established & (sales["cat_id"].values == category))[0]
        picked = rng.choice(candidates, size=config["data"]["established_per_category"], replace=False)
        chosen_rows.extend(picked.tolist())

    # New items
    earliest, latest = config["data"]["new_item_launch_window"]
    is_new = (first_sale_day >= total_days - earliest) & (first_sale_day <= total_days - latest)
    candidates = np.where(is_new)[0]
    how_many = min(config["data"]["new_items"], len(candidates))
    picked = rng.choice(candidates, size=how_many, replace=False)
    chosen_rows.extend(picked.tolist())

    chosen_rows = sorted(chosen_rows)
    return chosen_rows, first_sale_day[chosen_rows]


def make_complete_grid(df):
    """Make sure every series has exactly one row for every date.

    Missing days are added with demand = NaN (unknown). We do not fill them with 0,
    because "no record" is not the same as "sold nothing".
    """
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    all_series = df["series_id"].unique()
    grid = pd.MultiIndex.from_product([all_series, all_dates], names=["series_id", "date"]).to_frame(index=False)

    full = grid.merge(df, on=["series_id", "date"], how="left")
    added = full["demand"].isna().sum()
    print("Complete date grid: " + str(len(full)) + " rows, " + str(added) + " missing days added as NaN")

    # copy the static columns into any added rows
    static_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "launch_date"]
    full[static_columns] = full.groupby("series_id")[static_columns].transform("first")
    return full


def mark_stockouts(df, min_zero_run=3, max_chance=0.01):
    """Flag days that look like "the shelf was empty" rather than "nobody wanted it".

    If an item normally sells `rate` units a day, the chance of seeing k zero days in a row
    just by bad luck is about exp(-rate * k). When a run of zeros is so long that this
    chance is below `max_chance`, we call the whole run a stockout.

    Only the flag is added. The demand numbers stay exactly as they were.
    """
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    flags = np.zeros(len(df), dtype=int)

    start_row = 0
    for series_id, group in df.groupby("series_id", sort=True):
        demand = group["demand"].values
        active = group["is_active"].values == 1
        n = len(demand)

        rate = np.nanmean(demand[active]) if active.any() else 0.0
        if rate > 0:
            # shortest run of zeros that is "too unlikely to be bad luck"
            needed_run = max(min_zero_run, math.ceil(math.log(1 / max_chance) / rate))

            run_start = None
            for i in range(n + 1):
                is_zero_day = i < n and active[i] and demand[i] == 0
                if is_zero_day and run_start is None:
                    run_start = i
                if not is_zero_day and run_start is not None:
                    if i - run_start >= needed_run:
                        flags[start_row + run_start: start_row + i] = 1
                    run_start = None

        start_row += n

    df["is_stockout"] = flags
    return df


def build_dataset(config):
    """Read the raw M5 files and return the prepared table."""
    raw_dir = Path(config["data"]["raw_dir"])
    check_raw_files(raw_dir)

    print("Reading M5 sales file...")
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])

    rows, first_sale_day = pick_series(sales, config)
    picked = sales.iloc[rows].copy()
    picked["series_id"] = picked["id"].str.replace("_evaluation", "")
    # calendar row i is day d_(i+1), so this turns "position of first sale" into a date
    picked["launch_date"] = calendar["date"].values[first_sale_day]
    print("Picked " + str(len(picked)) + " series")

    # wide -> long: one row per series per day, only the most recent days
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    keep_days = day_columns[-config["data"]["days_of_history"]:]
    id_columns = ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "launch_date"]
    df = picked.melt(id_vars=id_columns, value_vars=keep_days, var_name="d", value_name="demand")
    df["demand"] = df["demand"].astype(float)

    # calendar information (known in advance)
    calendar["has_event"] = calendar["event_name_1"].notna().astype(int)
    calendar_columns = ["d", "date", "wm_yr_wk", "has_event", "snap_CA", "snap_TX", "snap_WI"]
    df = df.merge(calendar[calendar_columns], on="d", how="left")

    # SNAP (food stamp) days are different in each state
    df["snap"] = 0
    for state in ["CA", "TX", "WI"]:
        in_state = df["state_id"] == state
        df.loc[in_state, "snap"] = df.loc[in_state, "snap_" + state]

    # weekly prices (also known in advance in M5)
    print("Reading M5 price file...")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices["item_id"].isin(picked["item_id"])]
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df = df.drop(columns=["d", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"])
    df = make_complete_grid(df)

    df["is_active"] = (df["date"] >= df["launch_date"]).astype(int)

    # small integer codes for the static columns (used by the GBDT, and by DeepAR in Phase 2)
    df["cat_code"] = df["cat_id"].astype("category").cat.codes
    df["dept_code"] = df["dept_id"].astype("category").cat.codes
    df["store_code"] = df["store_id"].astype("category").cat.codes

    df = mark_stockouts(df, config["stockouts"]["min_zero_run"], config["stockouts"]["max_chance"])
    return df


def prepare_dataset(config, rebuild=False):
    """Load the prepared table from disk, or build it the first time."""
    processed_file = Path(config["data"]["processed_file"])

    if processed_file.exists() and not rebuild:
        print("Loading prepared data from " + str(processed_file))
        return pd.read_csv(processed_file, parse_dates=["date", "launch_date"])

    df = build_dataset(config)
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False)
    print("Saved prepared data to " + str(processed_file))
    return df
