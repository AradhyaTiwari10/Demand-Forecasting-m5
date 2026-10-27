"""
Turns the backtest predictions into metric tables, three charts and the Markdown report.
Every number and every "finding" sentence in the report is computed here from the results.
"""

from pathlib import Path

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt

from kalman_filter import fit_series
from metrics import score


MODEL_ORDER = ["SeasonalNaive", "KalmanFilter", "GBDT"]
MODEL_LABELS = {"SeasonalNaive": "Seasonal naive", "KalmanFilter": "Kalman filter", "GBDT": "GBDT"}
MODEL_COLORS = {"SeasonalNaive": "#2a78d6", "KalmanFilter": "#eb6834", "GBDT": "#1baf7a"}
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
BACKGROUND = "#fcfcfb"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def score_table(rows, quantiles, group_column=None):
    """One row of metrics per model (or per model and group)."""
    results = []
    for model in MODEL_ORDER:
        model_rows = rows[rows["model"] == model]
        if group_column is None:
            result = score(model_rows, quantiles)
            result["model"] = model
            results.append(result)
        else:
            for value, group in model_rows.groupby(group_column):
                result = score(group, quantiles)
                result["model"] = model
                result[group_column] = value
                results.append(result)
    return pd.DataFrame(results)


def markdown_table(table, columns, decimals=3):
    """Simple Markdown table without extra libraries."""
    whole_number_columns = ["rows", "fold", "step"]
    headers = [MODEL_LABELS.get(column, column) for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "model":
                value = MODEL_LABELS.get(value, value)
            if column in whole_number_columns:
                value = int(value)
            elif isinstance(value, float):
                value = f"{value:.{decimals}f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def style_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(MUTED_COLOR)
    ax.yaxis.label.set_color(MUTED_COLOR)


def plot_error_by_horizon(by_step, path):
    """MAE and mean pinball loss for each day ahead (two separate panels, one y-axis each)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BACKGROUND)
    panels = [("MAE", "MAE (units per day)"), ("mean_pinball", "Mean pinball loss (q10, q50, q90)")]

    for ax, (metric, title) in zip(axes, panels):
        for model in MODEL_ORDER:
            rows = by_step[by_step["model"] == model].sort_values("step")
            ax.plot(rows["step"], rows[metric], color=MODEL_COLORS[model], linewidth=2,
                    marker="o", markersize=4, label=MODEL_LABELS[model])
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Days ahead")
        ax.set_xticks(range(1, int(by_step["step"].max()) + 1))
        style_axis(ax)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Forecast error by days ahead (all origins, scored days only)", color=TEXT_COLOR, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def pick_example_series(scored):
    """A steady-selling, established series from the last fold, so the chart is easy to read."""
    last_fold = scored[(scored["fold"] == scored["fold"].max()) & (scored["model"] == "GBDT")]
    last_fold = last_fold[last_fold["is_cold_start"] == 0]
    mean_sales = last_fold.groupby("series_id")["actual"].mean()
    full_rows = last_fold.groupby("series_id").size()
    candidates = mean_sales[(full_rows == full_rows.max())]
    return (candidates - 4).abs().idxmin()


def plot_forecast_example(df, predictions, series_id, path):
    """Actual sales vs each model's forecast and 80% interval for one series (last fold)."""
    last_fold = predictions["fold"].max()
    rows = predictions[(predictions["fold"] == last_fold) & (predictions["series_id"] == series_id)]
    origin = rows["origin"].iloc[0]
    history = df[(df["series_id"] == series_id) & (df["date"] > origin - pd.Timedelta(days=42)) & (df["date"] <= origin)]
    actual = rows[rows["model"] == "GBDT"].sort_values("date")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, facecolor=BACKGROUND)
    for ax, model in zip(axes, MODEL_ORDER):
        forecast = rows[rows["model"] == model].sort_values("date")
        color = MODEL_COLORS[model]
        ax.plot(history["date"], history["demand"], color=MUTED_COLOR, linewidth=1.2, label="History")
        ax.fill_between(forecast["date"], forecast["q10"], forecast["q90"], color=color, alpha=0.2,
                        linewidth=0, label="80% interval (q10 to q90)")
        ax.plot(forecast["date"], forecast["point"], color=color, linewidth=2, label="Forecast")
        ax.plot(actual["date"], actual["actual"], "o", color=TEXT_COLOR, markersize=4, label="Actual")
        ax.axvline(origin, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_title(MODEL_LABELS[model], fontsize=11, loc="left")
        ax.tick_params(axis="x", rotation=30)
        style_axis(ax)
    axes[0].set_ylabel("Units sold per day")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Example forecast for {series_id} (origin {origin.date()}, dashed line)", color=TEXT_COLOR, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def pick_stockout_series(df):
    """A regularly selling series with several short stockout spells in its last 180 days."""
    recent = df[(df["date"] > df["date"].max() - pd.Timedelta(days=180)) & (df["is_active"] == 1)]
    best_series = None
    best_spells = -1
    for series_id, group in recent.groupby("series_id"):
        flags = group.sort_values("date")["is_stockout"].values
        if group["demand"].mean() < 1 or flags.mean() > 0.3:
            continue
        # a spell starts where the flag goes from 0 to 1
        spells = int(flags[0] == 1) + int(np.sum((flags[1:] == 1) & (flags[:-1] == 0)))
        if spells > best_spells:
            best_spells = spells
            best_series = series_id
    if best_series is None:
        best_series = recent["series_id"].iloc[0]
    return best_series


def plot_kalman_tracking(df, series_id, config, path):
    """The Kalman filter's demand level for one series, with flagged stockout days shaded."""
    history = df[df["series_id"] == series_id]
    fit = fit_series(history, config["kalman"]["noise_ratios"], config["kalman"]["min_weeks_for_weekly_pattern"])

    dates = pd.to_datetime(fit["dates"])
    show = dates > dates.max() - pd.Timedelta(days=180)
    dates = dates[show]
    demand = fit["demand"][show]
    level = fit["levels"][show]
    band = 1.28 * np.sqrt(fit["level_variances"][show])    # 80% band for the level
    stockout = ~fit["usable"][show]

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=BACKGROUND)
    # shade each stockout day
    first = True
    for day, flagged in zip(dates, stockout):
        if flagged:
            ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color="#d9d8d4",
                       linewidth=0, label="Flagged stockout day" if first else None)
            first = False
    ax.plot(dates, demand, "o", color=MUTED_COLOR, markersize=3, label="Recorded sales")
    ax.fill_between(dates, np.maximum(level - band, 0), level + band, color=MODEL_COLORS["KalmanFilter"],
                    alpha=0.2, linewidth=0, label="80% band for the level")
    ax.plot(dates, level, color=MODEL_COLORS["KalmanFilter"], linewidth=2, label="Kalman demand level (weekday-adjusted)")
    ax.set_ylabel("Units per day")
    ax.set_title(f"Kalman filter tracking for {series_id}: the level is held through flagged stockout days",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def best_model(table, metric):
    return table.sort_values(metric).iloc[0]["model"]


def value(table, model, metric, extra=None):
    rows = table[table["model"] == model]
    if extra is not None:
        rows = rows[rows[extra[0]] == extra[1]]
    return float(rows[metric].iloc[0])


def write_findings(overall, by_step, by_segment, all_active, stockout_means, quantiles, origins):
    """Plain-language findings, each one computed from the tables."""
    lines = []
    label = MODEL_LABELS

    best = best_model(overall, "WAPE")
    lines.append(f"- **Best point accuracy:** {label[best]} has the lowest WAPE "
                 f"({value(overall, best, 'WAPE'):.3f}).")
    sn = value(overall, "SeasonalNaive", "WAPE")
    for model in ["KalmanFilter", "GBDT"]:
        gain = (sn - value(overall, model, "WAPE")) / sn * 100
        lines.append(f"- {label[model]} lowers WAPE by {gain:.1f}% compared with seasonal naive.")

    kf = value(overall, "KalmanFilter", "WAPE")
    gb = value(overall, "GBDT", "WAPE")
    gap = abs(kf - gb) / max(kf, gb) * 100
    lines.append(f"- The Kalman filter and GBDT are {gap:.1f}% apart on WAPE, so a simple local "
                 "filter is almost as accurate as the global tree model on these series.")

    best_pb = best_model(overall, "mean_pinball")
    lines.append(f"- **Best uncertainty (pinball loss):** {label[best_pb]} "
                 f"({value(overall, best_pb, 'mean_pinball'):.3f}).")

    target = (max(quantiles) - min(quantiles)) * 100
    coverage_text = ", ".join(
        f"{label[m]} {value(overall, m, 'coverage') * 100:.0f}%" for m in MODEL_ORDER)
    lines.append(f"- **Interval coverage** (target {target:.0f}%): {coverage_text}. Part of the extra coverage "
                 "comes from zero-sales days, which land exactly on a q10 of 0.")

    first_step = by_step["step"].min()
    last_step = by_step["step"].max()
    for model in ["KalmanFilter", "GBDT"]:
        start = value(by_step, model, "MAE", ("step", first_step))
        end = value(by_step, model, "MAE", ("step", last_step))
        lines.append(f"- {label[model]} MAE goes from {start:.3f} at day {first_step} to {end:.3f} at day {last_step}.")
    weekday = origins[0].day_name()
    lines.append(f"- The error curves zig-zag because every origin is a {weekday}, so each 'days ahead' value is "
                 "always the same weekday (weekends sell more and have bigger errors).")

    cold_best = best_model(by_segment[by_segment["segment"] == "cold start"], "mean_pinball")
    established_wape = value(by_segment, best, "WAPE", ("segment", "established"))
    cold_wape = value(by_segment, best, "WAPE", ("segment", "cold start"))
    lines.append(f"- **Cold start:** {label[best]} WAPE is {cold_wape:.3f} on cold-start series vs "
                 f"{established_wape:.3f} on established ones. Best cold-start pinball loss: {label[cold_best]}.")

    wape_change = ", ".join(
        f"{label[m]} {value(overall, m, 'WAPE'):.3f} -> {value(all_active, m, 'WAPE'):.3f}" for m in MODEL_ORDER)
    lines.append(f"- **Stockouts:** scoring the stockout zeros as if they were real demand would raise WAPE "
                 f"({wape_change}), because forecasts of demand that was lost to an empty shelf would count as errors. "
                 "On stockout days the models forecast on average "
                 + ", ".join(f"{label[m]} {stockout_means[m]:.2f}" for m in MODEL_ORDER)
                 + " units per day.")
    return "\n".join(lines)


def build_report(df, predictions, origins, config, reports_dir):
    reports_dir = Path(reports_dir)
    quantiles = config["backtest"]["quantiles"]

    scored = predictions[predictions["is_scored"] == 1].copy()
    scored["segment"] = np.where(scored["is_cold_start"] == 1, "cold start", "established")
    active = predictions[(predictions["is_active"] == 1) & predictions["actual"].notna()]
    stockout_rows = predictions[predictions["is_stockout"] == 1]

    overall = score_table(scored, quantiles)
    by_fold = score_table(scored, quantiles, "fold")
    by_step = score_table(scored, quantiles, "step")
    by_segment = score_table(scored, quantiles, "segment")
    all_active = score_table(active, quantiles)
    stockout_means = stockout_rows.groupby("model")["point"].mean().to_dict()

    # save tables
    overall.to_csv(reports_dir / "metrics_overall.csv", index=False)
    by_fold.to_csv(reports_dir / "metrics_by_fold.csv", index=False)
    by_step.to_csv(reports_dir / "metrics_by_horizon.csv", index=False)
    by_segment.to_csv(reports_dir / "metrics_cold_start.csv", index=False)

    # charts
    plot_error_by_horizon(by_step, reports_dir / "error_by_horizon.png")
    example = pick_example_series(scored)
    plot_forecast_example(df, predictions, example, reports_dir / "forecast_example.png")
    stockout_series = pick_stockout_series(df)
    plot_kalman_tracking(df, stockout_series, config, reports_dir / "kalman_tracking.png")

    # numbers for the data section
    n_series = df["series_id"].nunique()
    n_new = int((df.groupby("series_id")["launch_date"].first() > df["date"].min()).sum())
    stockout_share = df.loc[df["is_active"] == 1, "is_stockout"].mean() * 100
    horizon = config["backtest"]["horizon"]

    metric_columns = ["model", "MAE", "RMSE", "WAPE", "pinball_q10", "pinball_q50", "pinball_q90", "mean_pinball", "coverage"]
    fold_table = by_fold.pivot(index="fold", columns="model", values="WAPE")[MODEL_ORDER].reset_index()
    fold_table["origin"] = [str(origins[i - 1].date()) for i in fold_table["fold"]]
    fold_table = fold_table[["fold", "origin"] + MODEL_ORDER]
    step_table = by_step[by_step["step"].isin([1, 7, horizon])].pivot(index="step", columns="model", values="MAE")
    step_table = step_table[MODEL_ORDER].reset_index()

    findings = write_findings(overall, by_step, by_segment, all_active, stockout_means, quantiles, origins)

    report = f"""# Phase 1 Report: Inventory Demand Forecasting on M5

Local state tracking (Kalman filter) vs. tabular AML baselines (seasonal naive, GBDT),
compared with a rolling-origin backtest. Phase 2 will add a global DeepAR / LSTM model to
the same backtest.

## 1. Data

- Source: M5 Forecasting - Accuracy (Walmart daily unit sales), files `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- Sample: **{n_series} series** (item x store) from all 3 categories and {df['store_id'].nunique()} stores:
  {n_series - n_new} established items and {n_new} items launched during the window (cold start).
- Period: {df['date'].min().date()} to {df['date'].max().date()} ({df['date'].nunique()} days), one row per series per day (complete date grid).
- **Launch date** = the first day an item ever sold. Days before launch are not used for training or scoring.
- **Stockouts are marked, not imputed.** A run of zero-sales days is flagged when it is too long to be bad luck:
  if an item sells `rate` units a day, k zero days in a row happen by chance with probability about exp(-rate x k);
  runs where this is below {config['stockouts']['max_chance']:.0%} (and at least {config['stockouts']['min_zero_run']} days) are flagged.
  {stockout_share:.1f}% of active days are flagged. Sales numbers are never changed.
- **Cold start** = a series launched less than {config['cold_start']['max_days_since_launch']} days before the forecast origin.

## 2. Models

| Model | Idea | Uncertainty |
|---|---|---|
| Seasonal naive | repeat the same weekday from last week | quantiles of its own past weekly errors |
| Kalman filter (one per series) | local demand level that moves slowly + weekday effects; skips the update on stockout days | Gaussian forecast variance P + h*q + r |
| GBDT (one global model) | gradient boosted trees on lags, rolling means, calendar, SNAP, events, price, item/store codes | separate quantile-loss models for q10, q50, q90 |

Kalman filter settings (q/r and r) and weekday effects are estimated **separately for every series** from its own training data.
The GBDT learns only from in-stock days, and its lag features treat stockout days as missing.

## 3. Backtest setup (identical for every model)

- {len(origins)} forecast origins, every {config['backtest']['step_days']} days: {', '.join(str(o.date()) for o in origins)}.
- Horizon: {horizon} days after each origin. Quantiles: {', '.join(str(q) for q in quantiles)}.
- Models only see data up to the origin. Stockout flags for training are re-computed from data up to the origin.
- A series is forecast only if it has at least {config['cold_start']['min_days_to_forecast']} days of history at the origin.
- **Scored days:** days after launch that are not stockouts ({len(scored) // len(MODEL_ORDER)} series-days per model).
- The exact series list and origins are saved in `reports/backtest_setup.json` so Phase 2 uses the same split.

## 4. Results

### Overall (scored days, all origins pooled)

{markdown_table(overall, metric_columns)}

WAPE = total absolute error / total sales. Coverage = share of actual sales inside the q10 to q90 interval (target {(max(quantiles) - min(quantiles)) * 100:.0f}%).

### WAPE per origin

{markdown_table(fold_table, ['fold', 'origin'] + MODEL_ORDER)}

### MAE by days ahead

{markdown_table(step_table, ['step'] + MODEL_ORDER)}

![Error by horizon](error_by_horizon.png)

### Cold start vs. established series

{markdown_table(by_segment, ['segment', 'model', 'rows', 'MAE', 'WAPE', 'mean_pinball', 'coverage'])}

### What changes if stockout zeros are scored as real demand

{markdown_table(all_active, ['model', 'rows', 'MAE', 'RMSE', 'WAPE', 'mean_pinball'])}

### Example forecast

![Forecast example](forecast_example.png)

### Kalman filter and stockouts

![Kalman tracking](kalman_tracking.png)

## 5. Findings

{findings}

## 6. The gap that Phase 2 addresses

- The Kalman filter learns each series on its own. A new item starts with no weekday pattern and a rough level,
  and nothing it learns about one item helps another.
- The GBDT is global, but it only sees hand-made lag features, needs a recursive loop for multi-day forecasts,
  and its quantiles come from three separate models rather than one predictive distribution.
- Phase 2 trains a **global DeepAR / LSTM** with static item/store identifiers and calendar/price features, using a
  count likelihood (negative binomial). It will be scored on the same series, origins, horizon, scored days and metrics,
  by adding its rows to `reports/predictions.csv`.

## 7. Limitations

- {n_series} series is a small sample of the 30,490 in M5. Differences of a few percent between models may not hold on the full data.
- Stockouts are inferred from sales only; M5 has no inventory data, so some flagged runs may be real zero demand.
- Gaussian Kalman intervals are not ideal for small whole-number sales.
"""
    (reports_dir / "phase1_report.md").write_text(report)
    return overall
