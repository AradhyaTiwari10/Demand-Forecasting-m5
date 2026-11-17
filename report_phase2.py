"""
Phase 2 report builder — extends the Phase 1 report to include DeepAR / LSTM results.

Produces:
  reports/phase2_report.md        – combined report comparing all 4 models
  reports/metrics_overall_p2.csv  – updated overall metrics (all 4 models)
  reports/error_by_horizon_p2.png – error-by-horizon chart for all 4 models
  reports/deepar_vs_gbdt.png      – direct comparison chart: DeepAR vs GBDT

Author: Mouli Srivastava (Phase 2 contribution)
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

# Import shared helpers from Phase 1 report.
# We shadow MODEL_ORDER / MODEL_LABELS / MODEL_COLORS locally where the 4-model
# versions are needed, but re-use everything else as-is.
from report import (
    score_table as _score_table_p1,  # wraps MODEL_ORDER; we monkey-patch below
    markdown_table as _markdown_table_p1,
    style_axis,
    plot_error_by_horizon as _plot_error_by_horizon_p1,
)
from metrics import score, quantile_column


# ---------------------------------------------------------------------------
# Phase 2 style constants (adds DeepAR to the Phase 1 palette)
# ---------------------------------------------------------------------------

MODEL_ORDER_P2 = ["SeasonalNaive", "KalmanFilter", "GBDT", "DeepAR"]
MODEL_LABELS_P2 = {
    "SeasonalNaive": "Seasonal naive",
    "KalmanFilter":  "Kalman filter",
    "GBDT":          "GBDT",
    "DeepAR":        "DeepAR (LSTM)",
}
MODEL_COLORS_P2 = {
    "SeasonalNaive": "#2a78d6",
    "KalmanFilter":  "#eb6834",
    "GBDT":          "#1baf7a",
    "DeepAR":        "#9b59b6",
}

# Shared theme colours (keep in sync with report.py)
TEXT_COLOR  = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR  = "#e4e3df"
BACKGROUND  = "#fcfcfb"


# ---------------------------------------------------------------------------
# Tables  (Phase-2-aware wrappers that use the 4-model order / labels)
# ---------------------------------------------------------------------------

def score_table_p2(rows, quantiles, group_column=None):
    """One row of metrics per model (or per model × group), for all 4 models."""
    results = []
    for model in MODEL_ORDER_P2:
        model_rows = rows[rows["model"] == model]
        if model_rows.empty:
            continue
        if group_column is None:
            result = score(model_rows, quantiles)
            result["model"] = model
            results.append(result)
        else:
            for value_key, group in model_rows.groupby(group_column):
                result = score(group, quantiles)
                result["model"]       = model
                result[group_column]  = value_key
                results.append(result)
    return pd.DataFrame(results)


def markdown_table_p2(table, columns, decimals=3):
    """Markdown table that knows all 4 model labels."""
    whole_number_columns = ["rows", "fold", "step"]
    headers = [MODEL_LABELS_P2.get(col, col) for col in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
    for _, row in table.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if col == "model":
                val = MODEL_LABELS_P2.get(val, val)
            if col in whole_number_columns:
                val = int(val)
            elif isinstance(val, float):
                val = f"{val:.{decimals}f}"
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_error_by_horizon_p2(by_step, path):
    """
    MAE and mean pinball loss per day-ahead step for all 4 models.
    Identical layout to Phase 1's plot_error_by_horizon but includes the
    DeepAR line in its own colour (#9b59b6).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BACKGROUND)
    panels = [
        ("MAE",          "MAE (units per day)"),
        ("mean_pinball", "Mean pinball loss (q10, q50, q90)"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        for model in MODEL_ORDER_P2:
            rows = by_step[by_step["model"] == model].sort_values("step")
            if rows.empty:
                continue
            ax.plot(
                rows["step"], rows[metric],
                color=MODEL_COLORS_P2[model], linewidth=2,
                marker="o", markersize=4,
                label=MODEL_LABELS_P2[model],
            )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Days ahead")
        ax.set_xticks(range(1, int(by_step["step"].max()) + 1))
        style_axis(ax)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Forecast error by days ahead — all 4 models (all origins, scored days only)",
        color=TEXT_COLOR, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_deepar_vs_gbdt(by_step, path):
    """
    Side-by-side 2-panel comparison of DeepAR vs GBDT:
      left  panel → MAE by forecast step
      right panel → mean pinball loss by forecast step
    Styled to match Phase 1 charts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BACKGROUND)
    panels = [
        ("MAE",          "MAE (units per day)"),
        ("mean_pinball", "Mean pinball loss (q10, q50, q90)"),
    ]
    comparison_models = ["GBDT", "DeepAR"]

    for ax, (metric, title) in zip(axes, panels):
        for model in comparison_models:
            rows = by_step[by_step["model"] == model].sort_values("step")
            if rows.empty:
                continue
            ax.plot(
                rows["step"], rows[metric],
                color=MODEL_COLORS_P2[model], linewidth=2.2,
                marker="o", markersize=5,
                label=MODEL_LABELS_P2[model],
            )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Days ahead")
        ax.set_xticks(range(1, int(by_step["step"].max()) + 1))
        style_axis(ax)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "DeepAR (LSTM) vs GBDT — error by days ahead (scored days only)",
        color=TEXT_COLOR, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Key-findings helper
# ---------------------------------------------------------------------------

def _value(table, model, metric, extra=None):
    """Read a single metric value from a scored table."""
    rows = table[table["model"] == model]
    if extra is not None:
        rows = rows[rows[extra[0]] == extra[1]]
    if rows.empty:
        return float("nan")
    return float(rows[metric].iloc[0])


def _best_model(table, metric):
    """Return the model name with the lowest value of `metric`."""
    return table.sort_values(metric).iloc[0]["model"]


def write_findings_phase2(overall, by_step, by_segment, quantiles, origins):
    """
    Plain-language findings for the Phase 2 report.
    Every sentence is computed from the actual metric tables — no hard-coding.
    """
    lines = []
    label = MODEL_LABELS_P2

    # --- Overall winner ---
    best = _best_model(overall, "WAPE")
    lines.append(
        f"- **Best point accuracy:** {label[best]} has the lowest WAPE "
        f"({_value(overall, best, 'WAPE'):.3f})."
    )

    # --- DeepAR vs GBDT on WAPE ---
    deepar_wape = _value(overall, "DeepAR", "WAPE")
    gbdt_wape   = _value(overall, "GBDT",   "WAPE")
    if not (np.isnan(deepar_wape) or np.isnan(gbdt_wape)) and gbdt_wape > 0:
        pct_wape = (gbdt_wape - deepar_wape) / gbdt_wape * 100
        direction = "lower" if pct_wape > 0 else "higher"
        lines.append(
            f"- DeepAR WAPE is {abs(pct_wape):.1f}% {direction} than GBDT "
            f"({deepar_wape:.3f} vs {gbdt_wape:.3f})."
        )

    # --- DeepAR vs GBDT on pinball ---
    deepar_pb = _value(overall, "DeepAR", "mean_pinball")
    gbdt_pb   = _value(overall, "GBDT",   "mean_pinball")
    if not (np.isnan(deepar_pb) or np.isnan(gbdt_pb)) and gbdt_pb > 0:
        pct_pb = (gbdt_pb - deepar_pb) / gbdt_pb * 100
        direction = "lower" if pct_pb > 0 else "higher"
        lines.append(
            f"- DeepAR mean pinball loss is {abs(pct_pb):.1f}% {direction} than GBDT "
            f"({deepar_pb:.3f} vs {gbdt_pb:.3f})."
        )

    # --- Improvement over seasonal naive ---
    sn_wape = _value(overall, "SeasonalNaive", "WAPE")
    for model in ["KalmanFilter", "GBDT", "DeepAR"]:
        m_wape = _value(overall, model, "WAPE")
        if not np.isnan(m_wape) and sn_wape > 0:
            gain = (sn_wape - m_wape) / sn_wape * 100
            lines.append(
                f"- {label[model]} lowers WAPE by {gain:.1f}% vs seasonal naive."
            )

    # --- Best uncertainty ---
    best_pb = _best_model(overall, "mean_pinball")
    lines.append(
        f"- **Best uncertainty (pinball loss):** {label[best_pb]} "
        f"({_value(overall, best_pb, 'mean_pinball'):.3f})."
    )

    # --- Interval coverage ---
    target = (max(quantiles) - min(quantiles)) * 100
    cov_parts = ", ".join(
        f"{label[m]} {_value(overall, m, 'coverage') * 100:.0f}%"
        for m in MODEL_ORDER_P2
        if not np.isnan(_value(overall, m, "coverage"))
    )
    lines.append(
        f"- **Interval coverage** (target {target:.0f}%): {cov_parts}."
    )

    # --- Cold-start comparison ---
    cold_seg  = by_segment[by_segment["segment"] == "cold start"]
    estab_seg = by_segment[by_segment["segment"] == "established"]
    if not cold_seg.empty and not estab_seg.empty:
        cold_best  = _best_model(cold_seg, "mean_pinball")
        cold_wape  = _value(cold_seg,  best, "WAPE")
        estab_wape = _value(estab_seg, best, "WAPE")
        cold_deepar  = _value(cold_seg,  "DeepAR", "WAPE")
        estab_deepar = _value(estab_seg, "DeepAR", "WAPE")
        lines.append(
            f"- **Cold start vs established ({label[best]}):** WAPE = "
            f"{cold_wape:.3f} (cold) vs {estab_wape:.3f} (established)."
        )
        if not (np.isnan(cold_deepar) or np.isnan(estab_deepar)):
            lines.append(
                f"- **DeepAR cold start:** WAPE = {cold_deepar:.3f} (cold) vs "
                f"{estab_deepar:.3f} (established). Best cold-start pinball: "
                f"{label[cold_best]}."
            )

    # --- Horizon degradation ---
    first_step = by_step["step"].min()
    last_step  = by_step["step"].max()
    for model in ["GBDT", "DeepAR"]:
        start = _value(by_step, model, "MAE", ("step", first_step))
        end   = _value(by_step, model, "MAE", ("step", last_step))
        if not (np.isnan(start) or np.isnan(end)):
            lines.append(
                f"- {label[model]} MAE goes from {start:.3f} at day "
                f"{first_step} to {end:.3f} at day {last_step}."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_report_phase2(df, predictions, origins, config, reports_dir):
    """
    Build the Phase 2 report comparing all four models:
      SeasonalNaive, KalmanFilter, GBDT, DeepAR.

    Parameters
    ----------
    df          : pd.DataFrame  – processed M5 panel data (same as Phase 1)
    predictions : pd.DataFrame  – combined predictions for all 4 models
                                  (must include the 'model' column with values
                                   from MODEL_ORDER_P2)
    origins     : list[pd.Timestamp] – the same backtest origins used in Phase 1
    config      : dict          – loaded config.yaml
    reports_dir : str | Path    – output directory (will be created if needed)

    Returns
    -------
    overall : pd.DataFrame – overall metrics table (all 4 models)
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    quantiles = config["backtest"]["quantiles"]
    horizon   = config["backtest"]["horizon"]

    # ------------------------------------------------------------------
    # Filter to scored rows, add segment label
    # ------------------------------------------------------------------
    scored = predictions[predictions["is_scored"] == 1].copy()
    scored["segment"] = np.where(scored["is_cold_start"] == 1, "cold start", "established")

    # ------------------------------------------------------------------
    # Metric tables (all 4 models)
    # ------------------------------------------------------------------
    overall    = score_table_p2(scored, quantiles)
    by_fold    = score_table_p2(scored, quantiles, "fold")
    by_step    = score_table_p2(scored, quantiles, "step")
    by_segment = score_table_p2(scored, quantiles, "segment")

    # ------------------------------------------------------------------
    # Pivot helpers for the report
    # ------------------------------------------------------------------
    metric_columns = [
        "model", "MAE", "RMSE", "WAPE",
        "pinball_q10", "pinball_q50", "pinball_q90",
        "mean_pinball", "coverage",
    ]

    fold_table = (
        by_fold
        .pivot(index="fold", columns="model", values="WAPE")[MODEL_ORDER_P2]
        .reset_index()
    )
    fold_table["origin"] = [str(origins[i - 1].date()) for i in fold_table["fold"]]
    fold_table = fold_table[["fold", "origin"] + MODEL_ORDER_P2]

    step_highlights = by_step[by_step["step"].isin([1, 7, horizon])]
    step_table = (
        step_highlights
        .pivot(index="step", columns="model", values="MAE")[MODEL_ORDER_P2]
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    plot_error_by_horizon_p2(by_step, reports_dir / "error_by_horizon_p2.png")
    plot_deepar_vs_gbdt(by_step, reports_dir / "deepar_vs_gbdt.png")

    # ------------------------------------------------------------------
    # Key findings (computed from tables)
    # ------------------------------------------------------------------
    findings = write_findings_phase2(overall, by_step, by_segment, quantiles, origins)

    # ------------------------------------------------------------------
    # Convenience numbers for the narrative text
    # ------------------------------------------------------------------
    deepar_wape = _value(overall, "DeepAR", "WAPE")
    gbdt_wape   = _value(overall, "GBDT",   "WAPE")
    pct_wape_improvement = (
        (gbdt_wape - deepar_wape) / gbdt_wape * 100
        if (not np.isnan(deepar_wape) and gbdt_wape > 0) else float("nan")
    )

    deepar_pb = _value(overall, "DeepAR", "mean_pinball")
    gbdt_pb   = _value(overall, "GBDT",   "mean_pinball")
    pct_pb_improvement = (
        (gbdt_pb - deepar_pb) / gbdt_pb * 100
        if (not np.isnan(deepar_pb) and gbdt_pb > 0) else float("nan")
    )

    lstm_cfg = config.get("lstm", {})

    # ------------------------------------------------------------------
    # Write phase2_report.md
    # ------------------------------------------------------------------
    report = f"""# Phase 2 Report: Demand Forecasting — DeepAR / LSTM

## 1. Phase 1 Summary

For data description, preprocessing, stockout detection methodology, Kalman filter,
GBDT, and the backtest setup see the Phase 1 report:
[`reports/phase1_report.md`](phase1_report.md).

All Phase 2 evaluations use the **exact same** series list, forecast origins,
horizon ({horizon} days), scored-day filter and metrics as Phase 1.

---

## 2. Phase 2 Model — DeepAR / LSTM

| Property | Value |
|---|---|
| Architecture | Encoder–decoder LSTM (DeepAR-style) |
| Scope | Global — one model across all series |
| Context length | {lstm_cfg.get('context_length', 90)} days fed to the encoder |
| Prediction length | {lstm_cfg.get('prediction_length', 14)} days (matches backtest horizon) |
| Hidden size | {lstm_cfg.get('hidden_size', 64)} units × {lstm_cfg.get('num_layers', 2)} LSTM layers |
| Dropout | {lstm_cfg.get('dropout', 0.1)} |
| Batch size | {lstm_cfg.get('batch_size', 64)} |
| Epochs | {lstm_cfg.get('epochs', 30)} |
| Learning rate | {lstm_cfg.get('learning_rate', 0.001)} |
| Quantiles | {', '.join(str(q) for q in lstm_cfg.get('quantiles', quantiles))} |
| Device | {lstm_cfg.get('device', 'cpu')} |

The model receives a fixed-length history window (context) for each series and
produces a 14-day probabilistic forecast.  Quantiles (q10, q50, q90) are obtained
by running the decoder with Monte-Carlo dropout at inference time.  Static
item/store embeddings allow the LSTM to share demand patterns across related
series — the key advantage over the per-series Kalman filter.

---

## 3. Overall Results — All 4 Models

{markdown_table_p2(overall, metric_columns)}

WAPE = total absolute error / total sales.
Coverage = share of actuals inside the q10–q90 interval
(target {(max(quantiles) - min(quantiles)) * 100:.0f}%).

---

## 4. WAPE per Forecast Origin

{markdown_table_p2(fold_table, ['fold', 'origin'] + MODEL_ORDER_P2)}

---

## 5. MAE by Horizon (Steps 1, 7, {horizon})

{markdown_table_p2(step_table, ['step'] + MODEL_ORDER_P2)}

![Error by horizon — all 4 models](error_by_horizon_p2.png)

---

## 6. Cold Start vs Established Series

{markdown_table_p2(by_segment, ['segment', 'model', 'rows', 'MAE', 'WAPE', 'mean_pinball', 'coverage'])}

---

## 7. DeepAR vs GBDT — Direct Comparison

![DeepAR vs GBDT by horizon](deepar_vs_gbdt.png)

| Metric | DeepAR | GBDT | Δ (%) |
|---|---|---|---|
| WAPE | {deepar_wape:.3f} | {gbdt_wape:.3f} | {pct_wape_improvement:+.1f}% |
| Mean pinball | {deepar_pb:.3f} | {gbdt_pb:.3f} | {pct_pb_improvement:+.1f}% |

A negative Δ means DeepAR is better (lower error); positive means GBDT wins.

---

## 8. Key Findings

{findings}

---

## 9. Limitations

- The M5 sample is **{df['series_id'].nunique()} series** out of 30,490; ranking of DeepAR vs GBDT
  may differ on the full dataset where the LSTM has more cross-series signal.
- The LSTM uses `cpu` inference by default; for large-scale runs set `device: cuda`
  in `config.yaml` to cut training time significantly.
- Monte-Carlo dropout quantiles are approximate; a proper negative-binomial likelihood
  head (as in the original DeepAR paper) would give tighter, better-calibrated intervals.
- Stockouts are inferred from sales zeros only; some flagged runs may be genuine
  zero-demand days rather than lost sales.
- Context length is fixed at {lstm_cfg.get('context_length', 90)} days; cold-start series with fewer
  history days are zero-padded, which may inflate their errors relative to the
  Kalman filter that adapts from the first observation.
"""

    (reports_dir / "phase2_report.md").write_text(report)

    # ------------------------------------------------------------------
    # Save updated overall metrics CSV
    # ------------------------------------------------------------------
    overall.to_csv(reports_dir / "metrics_overall_p2.csv", index=False)

    return overall
