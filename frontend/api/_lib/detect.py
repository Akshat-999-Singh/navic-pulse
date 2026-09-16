"""Static threshold, severity, CUSUM and persistence -- ported from the root detect.py.

Parameters are the calibrated values in model_meta.json (reproduced
detections.csv in export_weights.py verification C); nothing is re-fitted on an
uploaded batch.
"""

import numpy as np

from .meta import CUSUM_H, CUSUM_K, CUSUM_MU, PERSISTENCE, SEVERITY_CRITICAL, THRESHOLD


def cusum(scores, mu=CUSUM_MU, k=CUSUM_K):
    """One-sided CUSUM: S = max(0, S_prev + (score - mu) - k), starting at 0."""
    out = np.empty(len(scores))
    s = 0.0
    for i, score in enumerate(scores):
        s = max(0.0, s + (score - mu) - k)
        out[i] = s
    return out


def sustained(flags, m=PERSISTENCE):
    """True where the trailing `m` windows are all in alarm."""
    f = np.asarray(flags, dtype=int)
    if len(f) < m:
        return np.zeros(len(f), dtype=bool)
    run = np.convolve(f, np.ones(m, dtype=int), mode="valid") == m
    out = np.zeros(len(f), dtype=bool)
    out[m - 1:] = run
    return out


def first_index(flags):
    hit = np.flatnonzero(flags)
    return int(hit[0]) if len(hit) else None


def severity(scores):
    labels = np.full(len(scores), "normal", dtype=object)
    labels[scores > THRESHOLD] = "warning"
    labels[scores > SEVERITY_CRITICAL] = "critical"
    return labels


def run(scores):
    """Per-satellite detection over window scores in day order."""
    threshold_alarm = scores > THRESHOLD
    s = cusum(scores)
    cusum_alarm = s > CUSUM_H
    cusum_sustained = sustained(cusum_alarm)
    return {
        "threshold_alarm": threshold_alarm,
        "severity": severity(scores),
        "cusum_s": s,
        "cusum_alarm": cusum_alarm,
        "cusum_sustained": cusum_sustained,
        "threshold_sustained": sustained(threshold_alarm),
        "first_flagged_index": first_index(threshold_alarm),
        "persistence_confirmed_index": first_index(cusum_sustained),
    }
