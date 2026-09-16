"""Fixed-limit out-of-limits (OOL) baseline on raw channels -- ported from baseline_ool.py.

Limits: per channel mean +/- K_SIGMA_OOL * population sd. They come from
model_meta.json when it carries them; otherwise from the satellites in this
batch the model never trips (no window over threshold, no confirmed detection).
With neither, there is nothing honest to calibrate from and the baseline
reports null for every satellite.

Matched to detect.py: a day is out of limits if any channel is outside its band,
window i is in alarm when its end day is, and the same persistence rule applies.
"""

import numpy as np

from .detect import first_index, sustained
from .meta import BASELINE_OOL, CHANNELS, PERSISTENCE, WINDOW

K_SIGMA_OOL = 3.0


def limits_from_meta():
    if not BASELINE_OOL or "limits" not in BASELINE_OOL:
        return None
    return {c: {"low": float(BASELINE_OOL["limits"][c]["low"]),
                "high": float(BASELINE_OOL["limits"][c]["high"])} for c in CHANNELS}


def limits_from_batch(raw_by_id, healthy_ids):
    if not healthy_ids:
        return None
    pooled = np.concatenate([raw_by_id[i] for i in healthy_ids], axis=0)
    limits = {}
    for j, channel in enumerate(CHANNELS):
        mean, sd = float(pooled[:, j].mean()), float(pooled[:, j].std())
        limits[channel] = {"low": mean - K_SIGMA_OOL * sd, "high": mean + K_SIGMA_OOL * sd}
    return limits


def evaluate(raw, limits):
    """raw (n_days, n_channels) -> sustained index and the channels that confirmed it."""
    out = {c: (raw[:, j] < limits[c]["low"]) | (raw[:, j] > limits[c]["high"])
           for j, c in enumerate(CHANNELS)}
    day_flag = np.logical_or.reduce([out[c] for c in CHANNELS])
    # Window i ends on day i + WINDOW - 1; the first WINDOW - 1 days close no window.
    alarm = day_flag[WINDOW - 1:]
    confirmed = first_index(sustained(alarm))
    if confirmed is None:
        triggered = []
    else:
        run = slice(confirmed - PERSISTENCE + 1, confirmed + 1)
        triggered = [c for c in CHANNELS if out[c][WINDOW - 1:][run].any()]
    return confirmed, triggered


def run(raw_by_id, detections_by_id):
    """{id: (index or None, channels)}, plus how the limits were obtained."""
    limits = limits_from_meta()
    source = {"kind": "meta", "satellites": []}
    if limits is None:
        healthy = [i for i, d in detections_by_id.items()
                   if d["first_flagged_index"] is None
                   and d["persistence_confirmed_index"] is None]
        limits = limits_from_batch(raw_by_id, healthy)
        source = {"kind": "batch", "satellites": healthy}
    if limits is None:
        return {i: (None, []) for i in raw_by_id}, None
    return {i: evaluate(raw, limits) for i, raw in raw_by_id.items()}, source
