"""30-day sliding windows per satellite, flattened in the model's feature order.

Layout (preprocess.py, recorded in model_meta.json): day-major, oldest day
first, every channel in meta order within a day -- [d0c0..d0c4, d1c0..d1c4, ...].
A wrong order still produces 150 numbers and a plausible score, so the layout is
checked against the meta file's feature names rather than trusted.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .meta import CHANNELS, FEATURE_NAMES, LAYER_SIZES, WINDOW


class LayoutError(Exception):
    """The runtime's feature layout disagrees with model_meta.json."""


def expected_names():
    return [f"{channel}_t{offset - WINDOW + 1:+d}"
            for offset in range(WINDOW) for channel in CHANNELS]


def _check_meta():
    n = WINDOW * len(CHANNELS)
    if expected_names() != FEATURE_NAMES:
        raise LayoutError("feature names in model_meta.json do not match the day-major "
                          f"layout of {CHANNELS} over {WINDOW} days")
    if n != LAYER_SIZES[0] or n != LAYER_SIZES[-1]:
        raise LayoutError(f"{WINDOW} days x {len(CHANNELS)} channels = {n} features, "
                          f"but the model's layers are {LAYER_SIZES}")


_check_meta()


def raw_matrix(record):
    """(n_days, n_channels) raw values, columns looked up by name in meta order."""
    return np.column_stack([np.asarray(record[name], dtype=float) for name in CHANNELS])


def build(scaled):
    """(n_days, n_channels) scaled values -> (n_windows, WINDOW * n_channels)."""
    n_days, n_channels = scaled.shape
    if n_channels != len(CHANNELS):
        raise LayoutError(f"got {n_channels} channels, model expects {len(CHANNELS)}")
    # sliding_window_view gives (n_windows, n_channels, WINDOW); time axis first
    # makes the flattened row day-major.
    windows = np.transpose(sliding_window_view(scaled, WINDOW, axis=0), (0, 2, 1))
    features = windows.reshape(len(windows), -1)

    if features.shape != (n_days - WINDOW + 1, len(FEATURE_NAMES)):
        raise LayoutError(f"built {features.shape} features, expected "
                          f"({n_days - WINDOW + 1}, {len(FEATURE_NAMES)})")
    # Spot-check the corners named in meta: feature 0 is channel 0 on the oldest
    # day, the last feature is the last channel on the newest day.
    if not (features[0, 0] == scaled[0, 0]
            and features[-1, -1] == scaled[-1, -1]
            and features[0, len(CHANNELS)] == scaled[1, 0]):
        raise LayoutError("flattened windows are not day-major")
    return features
