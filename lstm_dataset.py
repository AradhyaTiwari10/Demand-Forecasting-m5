"""
lstm_dataset.py: data pipeline for the Phase 2 global LSTM (DeepAR-style) model.

Turns the prepared M5 table (one row per series per day, see data_loader.py)
into fixed-size windows the LSTM can train on.

    Scaler                 per-series demand scale and price z-score, fitted on training data only
    build_windows          list every usable (series_id, cutoff_date) training window
    M5WindowDataset        PyTorch Dataset: one sliding window per item
    make_loaders           train / validation DataLoaders with a time-based split
    make_forecast_dataset  inference DataLoader: last `context_length` days of history
                           plus the known future calendar / price features

One window = `context_length` past days (encoder) + `prediction_length` future days (decoder).
`cutoff_date` is the LAST day of the prediction part of the window.

Encoder features (ENCODER_FEATURES = 12), one row per past day:
    0  demand_norm        demand / series mean demand (0 where the day is not usable)
    1  demand_valid       1 if the day is active, not a stockout and demand is known
    2  day_of_week        0..6   -> /6
    3  day_of_month       1..31  -> (d-1)/30
    4  month              1..12  -> (m-1)/11
    5  has_event          0/1
    6  snap               0/1
    7  sell_price_norm    per-series z-score (0 when the price is missing)
    8  days_since_launch  /600
    9  cat_code           /max(cat_code)
    10 dept_code          /max(dept_code)
    11 store_code         /max(store_code)

Decoder features (DECODER_FEATURES = 7), one row per future day.
Only information known in advance (calendar and price), never demand:
    0 day_of_week  1 day_of_month  2 month  3 has_event  4 snap
    5 sell_price_norm  6 days_since_launch

No look-ahead:
- The Scaler only sees the DataFrame it is fitted on. Callers pass data up to
  the forecast origin only.
- Training windows end on or before `train_cutoff`. Validation windows end in
  (train_cutoff, val_cutoff].
- Stockout days, days before launch and missing days are masked
  (`encoder_mask` / `target_mask` = False). They are never used as real zeros
  and never count in the loss.

Batches
-------
The DataLoaders return a `WindowBatch`. It unpacks like a tuple:

    enc_inp, enc_mask, dec_inp, target, tgt_mask = batch

and also supports key access: `batch["encoder_input"]`, `batch["series_ids"]`,
`batch["dates"]` (the forecast dates, inference only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODER_FEATURES: int = 12
DECODER_FEATURES: int = 7

_DOW_DIV = 6.0
_DOM_DIV = 30.0
_MONTH_DIV = 11.0
_LAUNCH_DIV = 600.0

_CALENDAR_COLUMNS = ["day_of_week", "day_of_month", "month", "days_since_launch"]
_STATIC_CODES = ["cat_code", "dept_code", "store_code"]

# Default for "the item must have been live this many days before we forecast it".
# Same rule as config.yaml -> cold_start.min_days_to_forecast.
DEFAULT_MIN_DAYS = 7


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _lstm_section(config: dict) -> dict:
    """Accept either the whole project config or just its `lstm:` section."""
    if isinstance(config, dict) and isinstance(config.get("lstm"), dict):
        return config["lstm"]
    return config


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by series and date, with every column the LSTM needs.

    Calendar columns are added if they are missing. Optional columns get safe
    defaults (no event, no SNAP, item active, no stockout, static code 0).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "launch_date" in df.columns:
        df["launch_date"] = pd.to_datetime(df["launch_date"])
    else:
        df["launch_date"] = df.groupby("series_id")["date"].transform("min")

    if any(c not in df.columns for c in _CALENDAR_COLUMNS):
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days

    defaults = {
        "has_event": 0, "snap": 0, "is_active": 1, "is_stockout": 0,
        "cat_code": 0, "dept_code": 0, "store_code": 0, "sell_price": np.nan,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value
    if "demand" not in df.columns:
        df["demand"] = np.nan

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def _usable_demand(df: pd.DataFrame) -> np.ndarray:
    """True where demand can be trusted: item live, not a stockout, value known."""
    active = df["is_active"].fillna(0).to_numpy() == 1
    stockout = df["is_stockout"].fillna(1).to_numpy() == 1
    known = df["demand"].notna().to_numpy()
    return active & ~stockout & known


def _code_max(df: pd.DataFrame) -> Dict[str, float]:
    """Largest static code, used to squeeze the codes into [0, 1]."""
    return {c: max(float(df[c].max()), 1.0) if df[c].notna().any() else 1.0 for c in _STATIC_CODES}


def _split_series_args(a, b, series_id) -> Tuple[str, np.ndarray]:
    """Allow both transform(series_id, values) and transform(values, series_id=...)."""
    if series_id is not None:
        return series_id, a
    if isinstance(a, str):
        return a, b
    return b, a


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class Scaler:
    """Per-series scaling statistics, computed from training data only.

    Demand: divided by the series' mean demand over usable days. Mean scaling
    (not z-scoring) keeps zero at zero and keeps sales non-negative, which is
    important for intermittent retail series.

    Price: z-scored per series.

    Usage
    -----
        scaler = Scaler(train_df, series_ids)      # or: Scaler().fit(train_df)
        y_norm = scaler.transform("S1", y)
        y      = scaler.inverse_transform("S1", y_norm)
        # keyword form also works:  scaler.inverse_transform(y_norm, series_id="S1")
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, series_ids: Optional[Sequence[str]] = None) -> None:
        self.demand_scale_: Dict[str, float] = {}
        self.price_mean_: Dict[str, float] = {}
        self.price_std_: Dict[str, float] = {}
        if df is not None:
            self.fit(df, series_ids)

    def fit(self, df: pd.DataFrame, series_ids: Optional[Sequence[str]] = None) -> "Scaler":
        if series_ids is None:
            series_ids = df["series_id"].unique().tolist()
        df = df[df["series_id"].isin(series_ids)]

        usable = df["demand"].where(_usable_mask_aligned(df))
        mean_demand = usable.groupby(df["series_id"]).mean()
        price = df["sell_price"] if "sell_price" in df.columns else pd.Series(np.nan, index=df.index)
        price_mean = price.groupby(df["series_id"]).mean()
        price_std = price.groupby(df["series_id"]).std()

        for sid in series_ids:
            m = mean_demand.get(sid, np.nan)
            self.demand_scale_[sid] = float(m) if np.isfinite(m) and m > 0 else 1.0

            pm = price_mean.get(sid, np.nan)
            ps = price_std.get(sid, np.nan)
            self.price_mean_[sid] = float(pm) if np.isfinite(pm) else 0.0
            self.price_std_[sid] = float(ps) if np.isfinite(ps) and ps > 0 else 1.0
        return self

    def scale(self, series_id: str) -> float:
        return self.demand_scale_.get(series_id, 1.0)

    def transform(self, a, b=None, series_id: Optional[str] = None) -> np.ndarray:
        """Demand -> normalised demand. NaN becomes 0 (the mask marks those days)."""
        sid, values = _split_series_args(a, b, series_id)
        values = np.asarray(values, dtype=np.float64)
        out = np.where(np.isnan(values), 0.0, values / self.scale(sid))
        return out.astype(np.float32)

    def inverse_transform(self, a, b=None, series_id: Optional[str] = None) -> np.ndarray:
        """Normalised model output -> demand in units."""
        sid, values = _split_series_args(a, b, series_id)
        return (np.asarray(values, dtype=np.float64) * self.scale(sid)).astype(np.float32)

    def transform_price(self, a, b=None, series_id: Optional[str] = None) -> np.ndarray:
        """Price -> z-score. A missing price becomes 0 (the series' average price)."""
        sid, values = _split_series_args(a, b, series_id)
        values = np.asarray(values, dtype=np.float64)
        z = (values - self.price_mean_.get(sid, 0.0)) / self.price_std_.get(sid, 1.0)
        return np.where(np.isnan(z), 0.0, z).astype(np.float32)


def _usable_mask_aligned(df: pd.DataFrame) -> pd.Series:
    """Usable-demand mask with the same index as `df` (missing columns get safe defaults)."""
    no = pd.Series(False, index=df.index)
    active = (df["is_active"].fillna(0) == 1) if "is_active" in df.columns else ~no
    stockout = (df["is_stockout"].fillna(1) == 1) if "is_stockout" in df.columns else no
    return active & ~stockout & df["demand"].notna()


# ---------------------------------------------------------------------------
# Per-series feature arrays (computed once, then windows are cheap slices)
# ---------------------------------------------------------------------------

class _SeriesArrays:
    """All features of one series as numpy arrays, one row per day."""

    __slots__ = ("dates", "encoder", "decoder", "target", "valid", "days_live")

    def __init__(self, rows: pd.DataFrame, sid: str, scaler: Scaler, code_max: Dict[str, float]) -> None:
        n = len(rows)
        valid = _usable_demand(rows)
        demand_norm = scaler.transform(sid, rows["demand"].to_numpy(dtype=np.float64))
        demand_norm = np.where(valid, demand_norm, 0.0).astype(np.float32)

        decoder = self.known_in_advance(rows, sid, scaler)
        codes = np.stack(
            [rows[c].fillna(0).to_numpy(dtype=np.float32) / code_max[c] for c in _STATIC_CODES], axis=1
        )

        # encoder = demand + valid flag + the 7 known features + 3 static codes
        self.encoder = np.concatenate(
            [demand_norm[:, None], valid.astype(np.float32)[:, None], decoder, codes], axis=1
        ).astype(np.float32)
        self.decoder = decoder
        self.target = np.nan_to_num(rows["demand"].to_numpy(dtype=np.float32), nan=0.0)
        self.valid = valid
        self.dates = rows["date"].to_numpy()
        # number of days the item has been live up to and including each day
        self.days_live = np.cumsum(rows["is_active"].fillna(0).to_numpy() == 1)
        assert self.encoder.shape == (n, ENCODER_FEATURES)

    @staticmethod
    def known_in_advance(rows: pd.DataFrame, sid: str, scaler: Scaler) -> np.ndarray:
        """The 7 decoder features (calendar + price), shape (n, 7)."""
        price = scaler.transform_price(sid, rows["sell_price"].to_numpy(dtype=np.float64))
        return np.stack([
            rows["day_of_week"].fillna(0).to_numpy(dtype=np.float32) / _DOW_DIV,
            (rows["day_of_month"].fillna(1).to_numpy(dtype=np.float32) - 1) / _DOM_DIV,
            (rows["month"].fillna(1).to_numpy(dtype=np.float32) - 1) / _MONTH_DIV,
            rows["has_event"].fillna(0).to_numpy(dtype=np.float32),
            rows["snap"].fillna(0).to_numpy(dtype=np.float32),
            price,
            rows["days_since_launch"].fillna(0).to_numpy(dtype=np.float32) / _LAUNCH_DIV,
        ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    series_ids: Sequence[str],
    context_length: int,
    prediction_length: int,
    min_days: int = DEFAULT_MIN_DAYS,
    stride: int = 1,
) -> List[Tuple[str, pd.Timestamp]]:
    """List all usable (series_id, cutoff_date) windows.

    A window is kept when
    - the series has `context_length + prediction_length` days up to the cutoff,
    - the item had been live for at least `min_days` days when the prediction part starts
      (same rule as the backtest, so the model also sees cold-start-like windows),
    - at least one prediction day has usable demand (otherwise the loss would be empty).

    `stride` > 1 keeps every stride-th window per series (faster training).
    """
    df = _prepare_frame(df[df["series_id"].isin(series_ids)])
    total = context_length + prediction_length
    windows: List[Tuple[str, pd.Timestamp]] = []

    for sid, rows in df.groupby("series_id", sort=False):
        if len(rows) < total:
            continue
        valid = _usable_demand(rows).astype(int)
        valid_cum = np.concatenate([[0], np.cumsum(valid)])
        days_live = np.cumsum(rows["is_active"].fillna(0).to_numpy() == 1)
        dates = rows["date"].to_numpy()

        for end in range(len(rows) - 1, total - 2, -stride):   # end = cutoff row (inclusive)
            dec_start = end - prediction_length + 1
            if days_live[dec_start - 1] < min_days:
                continue
            if valid_cum[end + 1] - valid_cum[dec_start] == 0:
                continue
            windows.append((sid, pd.Timestamp(dates[end])))

    windows.sort(key=lambda w: (w[0], w[1]))
    return windows


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class M5WindowDataset(Dataset):
    """Sliding encoder/decoder windows over many series (one global model).

    Each item is a dict:
        encoder_input  (context_length, 12)  float32
        encoder_mask   (context_length,)     bool
        decoder_input  (prediction_length, 7) float32
        target         (prediction_length,)  float32, scaled demand (demand / series mean)
        target_mask    (prediction_length,)  bool
        series_id      str
        scale          float, the series mean (demand = target * scale)
        dates          list of the prediction dates

    The target is in the same normalised space as the encoder demand, so the
    model's mu / sigma are normalised too and lstm_forecast.py turns them back
    into units with scaler.inverse_transform.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_ids: Sequence[str],
        context_length: int,
        prediction_length: int,
        scaler: Scaler,
        min_days: int = DEFAULT_MIN_DAYS,
        windows: Optional[List[Tuple[str, pd.Timestamp]]] = None,
        cat_max: Optional[Dict[str, float]] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        df = _prepare_frame(df[df["series_id"].isin(series_ids)])
        self.series_ids = list(series_ids)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.scaler = scaler
        self.cat_max = cat_max if cat_max is not None else _code_max(df)

        if windows is None:
            windows = build_windows(df, self.series_ids, context_length, prediction_length, min_days, stride)
        self.windows = list(windows)

        # Feature arrays per series, built once
        self._arrays: Dict[str, _SeriesArrays] = {}
        self._date_index: Dict[str, Dict[np.datetime64, int]] = {}
        for sid, rows in df.groupby("series_id", sort=False):
            arrays = _SeriesArrays(rows, sid, scaler, self.cat_max)
            self._arrays[sid] = arrays
            self._date_index[sid] = {d: i for i, d in enumerate(arrays.dates)}

        # (series, cutoff row) for every window, so __getitem__ is just slicing
        total = self.context_length + self.prediction_length
        self._index: List[Tuple[str, int]] = []
        for sid, cutoff in self.windows:
            if sid not in self._arrays:
                raise KeyError("Series " + str(sid) + " is not in the DataFrame.")
            end = self._date_index[sid].get(np.datetime64(pd.Timestamp(cutoff), "ns"))
            if end is None:
                raise IndexError("Cutoff date " + str(cutoff) + " not found for series " + str(sid) + ".")
            if end - total + 1 < 0:
                raise IndexError("Not enough history for series " + str(sid) + " at cutoff " + str(cutoff) + ".")
            self._index.append((sid, end))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sid, end = self._index[idx]
        a = self._arrays[sid]
        dec_start = end - self.prediction_length + 1
        enc_start = dec_start - self.context_length

        return {
            "encoder_input": torch.from_numpy(a.encoder[enc_start:dec_start]),
            "encoder_mask": torch.from_numpy(a.valid[enc_start:dec_start].copy()),
            "decoder_input": torch.from_numpy(a.decoder[dec_start:end + 1]),
            "target": torch.from_numpy(a.target[dec_start:end + 1] / np.float32(self.scaler.scale(sid))),
            "target_mask": torch.from_numpy(a.valid[dec_start:end + 1].copy()),
            "series_id": sid,
            "scale": float(self.scaler.scale(sid)),
            "dates": [pd.Timestamp(d) for d in a.dates[dec_start:end + 1]],
        }


# ---------------------------------------------------------------------------
# Inference dataset
# ---------------------------------------------------------------------------

class _ForecastDataset(Dataset):
    """One item per series: the last `context_length` days of history plus the
    known features of the next `prediction_length` days. Series with less
    history are left-padded with zeros (mask = False), which handles cold start."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        series_ids: Sequence[str],
        context_length: int,
        prediction_length: int,
        scaler: Scaler,
        cat_max: Dict[str, float],
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.scaler = scaler

        train = _prepare_frame(train_df[train_df["series_id"].isin(series_ids)])
        future = _prepare_frame(future_df[future_df["series_id"].isin(series_ids)])
        train_groups = dict(tuple(train.groupby("series_id", sort=False)))
        future_groups = dict(tuple(future.groupby("series_id", sort=False)))

        self.items: List[Dict[str, object]] = []
        self.skipped: List[str] = []
        for sid in series_ids:
            history = train_groups.get(sid)
            ahead = future_groups.get(sid)
            if history is None or ahead is None or len(ahead) < self.prediction_length:
                self.skipped.append(sid)
                continue
            self.items.append(self._make_item(sid, history, ahead.iloc[: self.prediction_length], cat_max))

        if self.skipped:
            print("  [lstm_dataset] skipped " + str(len(self.skipped))
                  + " series without history or a full future horizon")

    def _make_item(self, sid, history, ahead, cat_max) -> Dict[str, object]:
        arrays = _SeriesArrays(history.iloc[-self.context_length:], sid, self.scaler, cat_max)
        encoder = arrays.encoder
        mask = arrays.valid
        pad = self.context_length - len(encoder)
        if pad > 0:
            encoder = np.vstack([np.zeros((pad, ENCODER_FEATURES), dtype=np.float32), encoder])
            mask = np.concatenate([np.zeros(pad, dtype=bool), mask])

        decoder = _SeriesArrays.known_in_advance(ahead, sid, self.scaler)
        return {
            "encoder_input": torch.from_numpy(np.ascontiguousarray(encoder)),
            "encoder_mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "decoder_input": torch.from_numpy(decoder),
            "target": torch.zeros(self.prediction_length, dtype=torch.float32),   # unknown
            "target_mask": torch.zeros(self.prediction_length, dtype=torch.bool),
            "series_id": sid,
            "scale": float(self.scaler.scale(sid)),
            "dates": [pd.Timestamp(d) for d in ahead["date"].to_numpy()],
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Batch type and collate
# ---------------------------------------------------------------------------

class WindowBatch(tuple):
    """A batch that unpacks like a tuple of 5 tensors:

        enc_inp, enc_mask, dec_inp, target, tgt_mask = batch

    and also answers key lookups: batch["encoder_input"], batch["series_ids"],
    batch["dates"], batch["scale"].
    """

    TENSOR_KEYS = ("encoder_input", "encoder_mask", "decoder_input", "target", "target_mask")

    def __new__(cls, tensors, meta=None):
        batch = super().__new__(cls, tensors)
        batch.meta = dict(meta or {})
        return batch

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.TENSOR_KEYS:
                return tuple.__getitem__(self, self.TENSOR_KEYS.index(key))
            if key == "series_id":
                key = "series_ids"
            return self.meta[key]
        return tuple.__getitem__(self, key)

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return key in self.TENSOR_KEYS or key in self.meta or key == "series_id"
        return any(item is key for item in self)

    def keys(self):
        return list(self.TENSOR_KEYS) + list(self.meta.keys())

    def to(self, device) -> "WindowBatch":
        return WindowBatch(tuple(t.to(device) for t in self), self.meta)


def _collate(samples: List[Dict[str, object]]) -> WindowBatch:
    tensors = tuple(torch.stack([s[k] for s in samples], dim=0) for k in WindowBatch.TENSOR_KEYS)
    meta = {
        "series_ids": [s["series_id"] for s in samples],
        "dates": [s["dates"] for s in samples],
        "scale": torch.tensor([s["scale"] for s in samples], dtype=torch.float32),
    }
    return WindowBatch(tensors, meta)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def _windows_in_range(
    windows: List[Tuple[str, pd.Timestamp]],
    lo: Optional[pd.Timestamp],
    hi: Optional[pd.Timestamp],
) -> List[Tuple[str, pd.Timestamp]]:
    """Windows whose cutoff date is in (lo, hi]."""
    lo = pd.Timestamp(lo) if lo is not None else None
    hi = pd.Timestamp(hi) if hi is not None else None
    return [
        (sid, cutoff) for sid, cutoff in windows
        if (hi is None or cutoff <= hi) and (lo is None or cutoff > lo)
    ]


def make_loaders(
    df: pd.DataFrame,
    series_ids: Sequence[str],
    config: dict,
    train_cutoff: pd.Timestamp,
    val_cutoff: Optional[pd.Timestamp] = None,
    scaler: Optional[Scaler] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Train (and validation) DataLoaders with a time-based split.

        train: windows whose prediction part ends on or before train_cutoff
        val:   windows whose prediction part ends in (train_cutoff, val_cutoff]

    `df` must only contain data up to the forecast origin. `config` may be the
    whole project config or its `lstm:` section. If no scaler is given, one is
    fitted on `df`. Both loaders share it (`loader.dataset.scaler`).
    """
    cfg = _lstm_section(config)
    ctx = int(cfg.get("context_length", 90))
    pred = int(cfg.get("prediction_length", 14))
    batch_size = int(cfg.get("batch_size", 64))
    workers = int(cfg.get("num_workers", 0))
    min_days = int(cfg.get("min_days", DEFAULT_MIN_DAYS))
    stride = int(cfg.get("window_stride", 1))
    seed = int(cfg.get("seed", 42))

    df = _prepare_frame(df[df["series_id"].isin(series_ids)])
    if scaler is None:
        scaler = Scaler(df, series_ids)
    code_max = _code_max(df)

    all_windows = build_windows(df, series_ids, ctx, pred, min_days, stride)

    def loader(windows, shuffle):
        dataset = M5WindowDataset(df, series_ids, ctx, pred, scaler, min_days, windows, code_max)
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                          collate_fn=_collate, generator=generator, drop_last=False)

    train_loader = loader(_windows_in_range(all_windows, None, train_cutoff), shuffle=True)
    val_loader = None
    if val_cutoff is not None:
        val_windows = _windows_in_range(all_windows, train_cutoff, val_cutoff)
        if val_windows:
            val_loader = loader(val_windows, shuffle=False)

    print("  [lstm_dataset] " + str(len(train_loader.dataset)) + " training windows, "
          + (str(len(val_loader.dataset)) if val_loader is not None else "0") + " validation windows")
    return train_loader, val_loader


def make_forecast_dataset(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    series_ids: Sequence[str],
    config: dict,
    scaler: Optional[Scaler] = None,
) -> DataLoader:
    """Inference DataLoader, one item per series (no shuffling).

    Encoder: the last `context_length` days of `train_df`.
    Decoder: the first `prediction_length` days of `future_df` (calendar and
    price only; a demand column, if present, is ignored).

    Pass the scaler that was used in training. If none is given, one is fitted
    on `train_df` (same data, so the statistics are identical).
    """
    cfg = _lstm_section(config)
    ctx = int(cfg.get("context_length", 90))
    pred = int(cfg.get("prediction_length", 14))
    batch_size = int(cfg.get("batch_size", 64))
    workers = int(cfg.get("num_workers", 0))

    train = _prepare_frame(train_df[train_df["series_id"].isin(series_ids)])
    if scaler is None:
        scaler = Scaler(train, series_ids)

    dataset = _ForecastDataset(train, future_df, series_ids, ctx, pred, scaler, _code_max(train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      collate_fn=_collate, drop_last=False)
