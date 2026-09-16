"""Per-channel share of each window's reconstruction error.

Squared residuals reshaped to (windows, WINDOW, channels) -- the day-major
layout -- summed over the window's 30 days, then normalised so each window's
shares add to 1.0. Same computation as export_demo_run.channel_contributions.
"""

import numpy as np

from .meta import CHANNELS, WINDOW


def shares(squared):
    per_channel = squared.reshape(len(squared), WINDOW, len(CHANNELS)).sum(axis=1)
    total = per_channel.sum(axis=1, keepdims=True)
    # A perfect reconstruction has no error to apportion; report zeros, not NaN.
    return np.divide(per_channel, total, out=np.zeros_like(per_channel), where=total > 0)
