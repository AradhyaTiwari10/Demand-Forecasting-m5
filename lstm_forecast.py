"""
lstm_forecast.py — Phase 2: Global LSTM (DeepAR-style) Forecaster
==================================================================
Anuj's contribution to the demand forecasting pipeline.

This module provides:
  - LSTMForecaster: sklearn-style class (.fit / .forecast) that trains a
    DeepARLSTM model and produces probabilistic forecasts.
  - deepar_forecast(): functional interface matching the Phase 1 baseline
    calling convention used in backtest.py.

Output format matches Phase 1 baselines exactly:
    columns: series_id, date, step, point, q10, q50, q90
"""

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PyTorch imports — fail gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as _torch_err:
    raise ImportError(
        "PyTorch is required for the LSTM forecaster but is not installed. "
        "Install it with:  pip install torch\n"
        f"Original error: {_torch_err}"
    ) from _torch_err

try:
    from lstm_dataset import Scaler, make_loaders, make_forecast_dataset
    from lstm_model import fit_lstm
    from features import add_calendar_features
except ImportError as _imp_err:
    raise ImportError(
        "Could not import one or more project modules (lstm_dataset, "
        "lstm_model, features). Make sure they are on the Python path.\n"
        f"Original error: {_imp_err}"
    ) from _imp_err


# ---------------------------------------------------------------------------
# LSTMForecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """
    Sklearn-style wrapper around DeepARLSTM for multi-series demand forecasting.

    Usage
    -----
    forecaster = LSTMForecaster(config)
    forecaster.fit(train_df)
    preds = forecaster.forecast(train_df, future_df)
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Project-wide configuration dict.  The LSTM-specific sub-dict is
            expected at config["lstm"] and may contain:
              - device       : "cpu" | "cuda" | "mps"  (default "cpu")
              - val_frac     : fraction of time-steps held out for validation
                               (default 0.20)
              - quantiles    : list of quantile floats (default [0.1, 0.5, 0.9])
        """
        self.config = config
        self._lstm_cfg = config.get("lstm", {})
        self.device = self._lstm_cfg.get("device", "cpu")
        self.val_frac = self._lstm_cfg.get("val_frac", 0.20)
        self.quantiles = self._lstm_cfg.get("quantiles", [0.1, 0.5, 0.9])

        # Set after fit()
        self.model = None
        self.scaler = None
        self._train_df = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "LSTMForecaster":
        """
        Train the LSTM model on historical demand data.

        Steps
        -----
        1. Add calendar features to train_df.
        2. Build a Scaler from training data.
        3. Time-based train / validation split (last val_frac of dates).
        4. Create DataLoaders via make_loaders().
        5. Train model via fit_lstm().

        Parameters
        ----------
        train_df : pd.DataFrame
            Historical data with at least columns: series_id, date, demand
            (plus any raw calendar / price columns already present).

        Returns
        -------
        self
        """
        # --- 1. Calendar features -----------------------------------------
        train_df = train_df.copy()
        train_df = add_calendar_features(train_df)

        # --- 2. Scaler -------------------------------------------------------
        self.scaler = Scaler()
        self.scaler.fit(train_df)

        # --- 3. Time-based train / val split ---------------------------------
        series_ids = train_df["series_id"].unique().tolist()
        all_dates = sorted(train_df["date"].unique())
        n_dates = len(all_dates)
        cutoff_idx = int(n_dates * (1.0 - self.val_frac))
        # Ensure at least one date in each split
        cutoff_idx = max(1, min(cutoff_idx, n_dates - 1))
        train_cutoff = all_dates[cutoff_idx - 1]   # last date in training split
        val_cutoff = all_dates[-1]                  # last date overall

        # --- 4. DataLoaders --------------------------------------------------
        train_loader, val_loader = make_loaders(
            df=train_df,
            series_ids=series_ids,
            config=self.config,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            scaler=self.scaler,
        )

        # --- 5. Train --------------------------------------------------------
        # fit_lstm reads hyper-parameters (epochs, hidden_size, ...) directly
        # from the dict it receives, so pass the `lstm:` section, not the
        # whole project config.
        self.model = fit_lstm(
            train_loader=train_loader,
            val_loader=val_loader,
            config=self._lstm_cfg,
            device=torch.device(self.device),
        )
        self.model.eval()

        # Persist the feature-enriched training data for use in forecast()
        self._train_df = train_df

        return self

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def forecast(self, train_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce probabilistic forecasts for each series over the forecast horizon.

        Parameters
        ----------
        train_df : pd.DataFrame
            Historical demand data (same schema as passed to fit()).  Calendar
            features are added internally if not already present.
        future_df : pd.DataFrame
            Future dates with calendar / price features but **no demand column**.
            Rows × columns must match what make_forecast_dataset expects.

        Returns
        -------
        pd.DataFrame
            Columns: series_id, date, step, point, q10, q50, q90
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() before forecast().")

        # --- Add calendar features to both frames ---------------------------
        train_df = train_df.copy()
        train_df = add_calendar_features(train_df)

        future_df = future_df.copy()
        future_df = add_calendar_features(future_df)

        series_ids = train_df["series_id"].unique().tolist()

        # --- Build forecast DataLoader --------------------------------------
        forecast_loader = make_forecast_dataset(
            train_df=train_df,
            future_df=future_df,
            series_ids=series_ids,
            config=self.config,
            scaler=self.scaler,   # same scaling as in training
        )

        # q10 / q50 / q90 are always needed for the output schema
        quantiles = sorted(set(self.quantiles) | {0.1, 0.5, 0.9})

        # --- Inference loop -------------------------------------------------
        device = torch.device(self.device)
        self.model.to(device)
        self.model.eval()

        # Containers for assembling the output table
        rows = []

        with torch.no_grad():
            for batch in forecast_loader:
                # Expected batch keys from make_forecast_dataset:
                #   encoder_input  : (B, enc_len, enc_feat)
                #   encoder_mask   : (B, enc_len)
                #   decoder_input  : (B, dec_len, dec_feat)
                #   series_ids_b   : list[str] of length B
                #   dates_b        : list of arrays, each shape (dec_len,)
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                batch_series_ids = batch["series_ids"]   # list of str
                batch_dates = batch["dates"]              # list of date arrays

                # predict_quantiles returns dict: quantile_float -> Tensor(B, H)
                # and also "mean" -> Tensor(B, H)
                q_preds = self.model.predict_quantiles(
                    encoder_input=encoder_input,
                    encoder_mask=encoder_mask,
                    decoder_input=decoder_input,
                    quantiles=quantiles,
                )

                # Move tensors to CPU numpy
                mean_pred = q_preds["mean"].cpu().numpy()          # (B, H)
                q10_pred = q_preds[0.1].cpu().numpy()              # (B, H)
                q50_pred = q_preds[0.5].cpu().numpy()              # (B, H)
                q90_pred = q_preds[0.9].cpu().numpy()              # (B, H)

                batch_size, horizon = mean_pred.shape

                for b_idx in range(batch_size):
                    sid = batch_series_ids[b_idx]
                    dates = batch_dates[b_idx]  # array-like of length horizon

                    # Inverse-transform each prediction back to demand scale
                    mu_inv = self.scaler.inverse_transform(
                        mean_pred[b_idx], series_id=sid
                    )
                    q10_inv = self.scaler.inverse_transform(
                        q10_pred[b_idx], series_id=sid
                    )
                    q50_inv = self.scaler.inverse_transform(
                        q50_pred[b_idx], series_id=sid
                    )
                    q90_inv = self.scaler.inverse_transform(
                        q90_pred[b_idx], series_id=sid
                    )

                    # Clip at zero — demand cannot be negative
                    mu_inv = np.maximum(mu_inv, 0.0)
                    q10_inv = np.maximum(q10_inv, 0.0)
                    q50_inv = np.maximum(q50_inv, 0.0)
                    q90_inv = np.maximum(q90_inv, 0.0)

                    for step_idx in range(horizon):
                        rows.append(
                            {
                                "series_id": sid,
                                "date": dates[step_idx],
                                "step": step_idx + 1,          # 1-indexed
                                "point": float(mu_inv[step_idx]),
                                "q10": float(q10_inv[step_idx]),
                                "q50": float(q50_inv[step_idx]),
                                "q90": float(q90_inv[step_idx]),
                            }
                        )

        result_df = pd.DataFrame(
            rows,
            columns=["series_id", "date", "step", "point", "q10", "q50", "q90"],
        )
        return result_df


# ---------------------------------------------------------------------------
# Functional interface (matches Phase 1 / backtest.py calling convention)
# ---------------------------------------------------------------------------

def deepar_forecast(
    train: pd.DataFrame,
    future_df: pd.DataFrame,
    series_list: list,
    horizon: int,
    quantiles: list,
    config: dict,
) -> pd.DataFrame:
    """
    Functional interface for the LSTM / DeepAR forecaster.

    Mirrors the Phase 1 baseline calling convention used in backtest.py:

        predictions["DeepAR"] = deepar_forecast(
            train, future, series_list, horizon, quantiles, config
        )

    Parameters
    ----------
    train : pd.DataFrame
        Historical demand data (series_id, date, demand, …).
    future_df : pd.DataFrame
        Future dates with calendar / price features but no demand.
        Typically produced in backtest.py as:
            future = test.drop(columns=["demand", "is_stockout"])
    series_list : list[str]
        Series identifiers to forecast (currently passed through for
        documentation purposes; LSTMForecaster uses all series in train).
    horizon : int
        Forecast horizon in days (informational; actual horizon is derived
        from future_df row count per series).
    quantiles : list[float]
        Requested quantile levels, e.g. [0.1, 0.5, 0.9].
    config : dict
        Project configuration dict (lstm sub-dict controls model hyper-params).

    Returns
    -------
    pd.DataFrame
        Columns: series_id, date, step, point, q10, q50, q90
    """
    # Inject requested quantiles into config so LSTMForecaster picks them up
    cfg = {**config}
    lstm_cfg = {**cfg.get("lstm", {})}
    lstm_cfg["quantiles"] = quantiles
    cfg["lstm"] = lstm_cfg

    forecaster = LSTMForecaster(cfg)
    forecaster.fit(train)
    return forecaster.forecast(train, future_df)
