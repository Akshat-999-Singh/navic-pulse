"""Frozen autoencoder inference: scale, forward pass, reconstruction error.

Arrays are loaded once at import (cold start), not per request. The forward
pass is the one export_weights.py verified against scores.csv.
"""

import numpy as np

from .meta import CHANNELS, LAYER_SIZES, META, WEIGHTS_PATH

_ACTIVATIONS = {
    "relu": lambda h: np.maximum(h, 0.0),
    "identity": lambda h: h,
}
_HIDDEN = _ACTIVATIONS[META["hidden_activation"]]
_OUTPUT = _ACTIVATIONS[META["output_activation"]]

with np.load(WEIGHTS_PATH) as _npz:
    WEIGHTS = [_npz[name] for name in META["arrays"]["weights"]]
    BIASES = [_npz[name] for name in META["arrays"]["biases"]]
    SCALER_MEAN = _npz[META["arrays"]["scaler"]["mean"]]
    SCALER_SCALE = _npz[META["arrays"]["scaler"]["scale"]]

for _i, (_w, _b) in enumerate(zip(WEIGHTS, BIASES)):
    if _w.shape != (LAYER_SIZES[_i], LAYER_SIZES[_i + 1]) or _b.shape != (LAYER_SIZES[_i + 1],):
        raise ValueError(f"layer {_i}: w{_i} {_w.shape}, b{_i} {_b.shape} "
                         f"do not match layer_sizes {LAYER_SIZES}")
if SCALER_MEAN.shape != (len(CHANNELS),) or SCALER_SCALE.shape != (len(CHANNELS),):
    raise ValueError("scaler arrays do not have one entry per channel")


def scale(raw):
    """(n_days, n_channels) raw -> scaled, per channel in meta order."""
    return (raw - SCALER_MEAN) / SCALER_SCALE


def forward(x):
    h = x
    last = len(WEIGHTS) - 1
    for i, (w, b) in enumerate(zip(WEIGHTS, BIASES)):
        h = h @ w + b
        h = _OUTPUT(h) if i == last else _HIDDEN(h)
    return h


def residuals(features):
    """Squared residual per feature, (n_windows, n_features)."""
    return (features - forward(features)) ** 2


def scores(squared):
    """Mean squared reconstruction error per window."""
    return squared.mean(axis=1)
