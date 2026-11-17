"""
lstm_model.py — Phase 2: Global LSTM (DeepAR-style) for M5 Demand Forecasting
===============================================================================
Author : Aradhya (Phase 2 contribution)
Purpose: Defines the DeepARLSTM architecture and the training/evaluation loop
         used by the global LSTM forecasting pipeline.

Architecture summary
--------------------
* Encoder LSTM  — reads the full context window of historical features
* Decoder LSTM  — steps through the forecast horizon using known-future features,
                  initialised with the encoder's final hidden/cell state
* Output head   — linear layer mapping decoder hidden state → (mu, log_sigma),
                  parametrising a Gaussian predictive distribution

Loss
----
Gaussian negative log-likelihood, masked to ignore padded / missing time-steps.

Quantile forecasts
------------------
Derived analytically from the Gaussian CDF:
    q-quantile = mu + sigma * Φ⁻¹(q)
where Φ⁻¹ is the standard-normal percent-point function (scipy.stats.norm.ppf).
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Feature-dimension constants — kept in sync with lstm_dataset.py
# ---------------------------------------------------------------------------
try:
    from lstm_dataset import ENCODER_FEATURES, DECODER_FEATURES
except ImportError:  # graceful fallback so the file can be imported standalone
    ENCODER_FEATURES = 12
    DECODER_FEATURES = 7


# Smallest allowed standard deviation (in scaled-demand units). Without a floor the model
# shrinks sigma towards 0 on the many zero-sales days and the loss explodes during training.
MIN_LOG_SIGMA = math.log(0.1)


# ===========================================================================
# 1.  Model
# ===========================================================================

class DeepARLSTM(nn.Module):
    """Global LSTM model with an encoder–decoder structure.

    Parameters
    ----------
    encoder_features : int
        Number of features in the encoder (context) input.
    decoder_features : int
        Number of known-in-advance features fed to the decoder at each
        future time step.
    hidden_size : int
        Number of units in each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers (ignored when
        num_layers == 1).
    """

    def __init__(
        self,
        encoder_features: int = ENCODER_FEATURES,
        decoder_features: int = DECODER_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Encoder ----------------------------------------------------------
        # Reads the context window: (batch, context_length, encoder_features)
        self.encoder = nn.LSTM(
            input_size=encoder_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Decoder ----------------------------------------------------------
        # Steps through the forecast horizon one day at a time, conditioned on
        # known-future features: input shape (batch, 1, decoder_features)
        self.decoder = nn.LSTM(
            input_size=decoder_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Output head ------------------------------------------------------
        # Maps decoder hidden state → (mu, log_sigma) per time step
        self.output_head = nn.Linear(hidden_size, 2)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform init for LSTM weights; zero bias for output head."""
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.zeros_(self.output_head.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder then decoder to produce distributional forecasts.

        Parameters
        ----------
        encoder_input : Tensor, shape (batch, context_length, encoder_features)
        decoder_input : Tensor, shape (batch, prediction_length, decoder_features)

        Returns
        -------
        Tensor, shape (batch, prediction_length, 2)
            [..., 0] = mu        (raw, can be negative before clipping)
            [..., 1] = log_sigma (unbounded; exponentiate to get std-dev)
        """
        # 1. Encode the context window
        _, (h_enc, c_enc) = self.encoder(encoder_input)
        # h_enc / c_enc : (num_layers, batch, hidden_size)

        # 2. Decode step-by-step across the forecast horizon
        prediction_length = decoder_input.size(1)
        outputs = []

        h, c = h_enc, c_enc
        for t in range(prediction_length):
            # (batch, 1, decoder_features)
            dec_in_t = decoder_input[:, t : t + 1, :]
            dec_out_t, (h, c) = self.decoder(dec_in_t, (h, c))
            # dec_out_t : (batch, 1, hidden_size)
            params_t = self.output_head(dec_out_t)  # (batch, 1, 2)
            outputs.append(params_t)

        # Stack along time axis → (batch, prediction_length, 2)
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_quantiles(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,          # kept for API compatibility
        decoder_input: torch.Tensor,
        quantiles: list,
    ) -> dict:
        """Return quantile forecasts derived from the Gaussian distribution.

        Parameters
        ----------
        encoder_input : Tensor, shape (batch, context_length, encoder_features)
        encoder_mask  : Tensor, shape (batch, context_length)  — unused here but
                        kept for a consistent API across model types.
        decoder_input : Tensor, shape (batch, prediction_length, decoder_features)
        quantiles     : list of floats, e.g. [0.1, 0.5, 0.9]

        Returns
        -------
        dict  {quantile_float → Tensor of shape (batch, prediction_length)}
            Values are clipped to >= 0 (demand cannot be negative).
        """
        self.eval()
        raw = self.forward(encoder_input, decoder_input)
        # raw : (batch, prediction_length, 2)

        mu = raw[..., 0]                        # (batch, prediction_length)
        sigma = torch.exp(raw[..., 1].clamp(min=MIN_LOG_SIGMA))

        result = {}
        for q in quantiles:
            # Gaussian quantile: mu + sigma * Φ⁻¹(q)
            z = _normal_ppf(q)                  # scalar float
            q_forecast = mu + sigma * z
            # Demand is non-negative
            result[q] = q_forecast.clamp(min=0.0)

        result["mean"] = mu.clamp(min=0.0)   # point forecast, used by lstm_forecast.py
        return result


# ===========================================================================
# 2.  Loss
# ===========================================================================

def gaussian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked Gaussian negative log-likelihood loss.

    Parameters
    ----------
    mu         : Tensor (batch, prediction_length)
    log_sigma  : Tensor (batch, prediction_length)
    target     : Tensor (batch, prediction_length)  — observed demand
    mask       : BoolTensor (batch, prediction_length)  — True = valid position

    Returns
    -------
    Scalar tensor — mean NLL over all valid positions.
    """
    log_sigma = log_sigma.clamp(min=MIN_LOG_SIGMA)
    sigma = torch.exp(log_sigma)

    # Element-wise NLL:  0.5 * log(2π) + log(σ) + (y − μ)² / (2σ²)
    nll = (
        0.5 * math.log(2 * math.pi)
        + log_sigma
        + 0.5 * ((target - mu) / sigma) ** 2
    )

    # Zero-out invalid positions and average over valid ones
    nll = nll * mask.float()
    n_valid = mask.float().sum().clamp(min=1.0)
    return nll.sum() / n_valid


# ===========================================================================
# 3.  Training helpers
# ===========================================================================

def train_epoch(
    model: DeepARLSTM,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full training epoch.

    Parameters
    ----------
    model     : DeepARLSTM
    loader    : DataLoader yielding (encoder_input, encoder_mask,
                                     decoder_input, target, target_mask)
    optimizer : torch.optim.Optimizer
    device    : torch.device

    Returns
    -------
    float — mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        enc_inp, enc_mask, dec_inp, target, tgt_mask = _unpack_batch(batch, device)

        optimizer.zero_grad()

        raw = model(enc_inp, dec_inp)           # (batch, pred_len, 2)
        mu       = raw[..., 0]
        log_sigma = raw[..., 1]

        loss = gaussian_nll_loss(mu, log_sigma, target, tgt_mask)
        loss.backward()

        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: DeepARLSTM,
    loader,
    device: torch.device,
) -> float:
    """Compute mean validation loss (no gradient updates).

    Parameters
    ----------
    model  : DeepARLSTM
    loader : DataLoader (same schema as train loader)
    device : torch.device

    Returns
    -------
    float — mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            enc_inp, enc_mask, dec_inp, target, tgt_mask = _unpack_batch(batch, device)

            raw = model(enc_inp, dec_inp)
            mu        = raw[..., 0]
            log_sigma = raw[..., 1]

            loss = gaussian_nll_loss(mu, log_sigma, target, tgt_mask)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ===========================================================================
# 4.  Top-level training entry point
# ===========================================================================

def fit_lstm(
    train_loader,
    val_loader,
    config: dict,
    device=None,
) -> DeepARLSTM:
    """Instantiate, train, and return a DeepARLSTM model.

    Parameters
    ----------
    train_loader : DataLoader
    val_loader   : DataLoader or None
    config       : dict — the ``lstm`` section of config.yaml, e.g.::

        {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 30,
            "learning_rate": 0.001,
        }

    device       : torch.device or None
                   If None, resolved from config["device"] (default "cpu").

    Returns
    -------
    DeepARLSTM — the trained model (in eval mode).
    """
    # --- Resolve device -------------------------------------------------------
    if device is None:
        device_str = config.get("device", "cpu")
        device = torch.device(device_str)

    # Same random start every run, so results can be reproduced exactly
    torch.manual_seed(int(config.get("seed", 42)))

    # --- Build model ----------------------------------------------------------
    model = DeepARLSTM(
        encoder_features=ENCODER_FEATURES,
        decoder_features=DECODER_FEATURES,
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    # --- Optimizer ------------------------------------------------------------
    lr = float(config.get("learning_rate", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop --------------------------------------------------------
    epochs = int(config.get("epochs", 30))
    patience = 5          # epochs without improvement before halving lr
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

            # Simple LR decay: halve lr after `patience` stagnant epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())   # remember the best epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.5
                    new_lr = optimizer.param_groups[0]["lr"]
                    print(f"  → LR reduced to {new_lr:.2e} (no improvement for {patience} epochs)")
                    epochs_no_improve = 0   # reset counter after reduction
        else:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f}")

    # Use the epoch with the lowest validation loss, not simply the last one
    model.load_state_dict(best_state)
    model.eval()
    return model


# ===========================================================================
# 5.  Private utilities
# ===========================================================================

def _unpack_batch(batch, device: torch.device):
    """Move a batch tuple/list to `device` and return its five tensors."""
    enc_inp, enc_mask, dec_inp, target, tgt_mask = batch
    enc_inp  = enc_inp.to(device)
    enc_mask = enc_mask.to(device)
    dec_inp  = dec_inp.to(device)
    target   = target.to(device)
    tgt_mask = tgt_mask.to(device)
    return enc_inp, enc_mask, dec_inp, target, tgt_mask


def _normal_ppf(q: float) -> float:
    """Percent-point function (inverse CDF) of the standard normal.

    Uses the rational approximation from Abramowitz & Stegun §26.2.17 so we
    avoid a scipy dependency at inference time.
    """
    # For common quantiles just use the exact values
    _lut = {0.1: -1.2816, 0.5: 0.0, 0.9: 1.2816}
    if q in _lut:
        return _lut[q]

    # Beasley-Springer-Moro approximation (accurate to ~1e-9 for 0 < q < 1)
    p = q if q <= 0.5 else 1.0 - q
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t ** 2) / (1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return -z if q <= 0.5 else z
