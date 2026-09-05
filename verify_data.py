"""
NavIC Pulse -- dashboard data layer: build and verify docs/data.js.

docs/data.js is the single source the dashboard screens read. It is generated
from the pipeline outputs, never hand-edited:

    python verify_data.py --write    regenerate docs/data.js
    python verify_data.py            verify the committed file still matches

Verification is byte-exact. The generated payload carries no timestamp and no
run-dependent value, so a clean tree round-trips identically; any difference
means either the pipeline outputs moved or the file was edited by hand. The
exit status is 0 on a clean verify and 1 on a stale, missing or invariant-
failing one, so this runs unattended.

The CSVs it reads are gitignored -- they regenerate deterministically from
RANDOM_SEED in config.py -- so docs/data.js is the only committed record of
the pipeline outputs. On a fresh clone, run the pipeline before verifying.

On top of the byte check, `check_invariants` asserts the facts the dashboard is
allowed to claim. Those assertions exist because an earlier design draft
carried figures this backend does not support -- a 28 V bus, a 42 C clock
package, a 42-day advance warning, a thermal-intervention narrative. The
invariants and the BANNED_TERMS list below are what stop any of that coming
back in through the data layer.

Derived quantities are imported from the modules that own them
(detect.WARNING_SHARE, rul.ols_crossing) or read back from the files those
modules wrote, rather than reimplemented here, so this script cannot drift
from the pipeline it is checking.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from config import (CHANNELS, CUSUM_PERSISTENCE, K_SIGMA, N_DAYS, SATELLITES,
                    SPEC_LIMIT_DRIFT, WINDOW)
from detect import WARNING_SHARE
from rul import SIGNAL_FAILURE_LIMIT, TREND_WINDOW_DAYS, ols_crossing

INPUT_TELEMETRY = "telemetry.csv"
INPUT_SCORES = "scores.csv"
INPUT_DETECTIONS = "detections.csv"
INPUT_SUMMARY = "summary.json"
INPUT_RUL = "rul.json"
OUTPUT_JS = os.path.join("docs", "data.js")

JS_GLOBAL = "NAVIC_PULSE"
SCHEMA_VERSION = 1

# -- sampling -------------------------------------------------------------
# Full resolution is 2,871 windows x 10 satellites. A uniform stride plus a
# dense band around each fault keeps the payload small without blurring the
# only region of the record where the score actually moves.
SPARSE_STRIDE = 12          # score / drift series, away from a fault
DENSE_STRIDE = 2            # score / drift series, inside the fault band
DENSE_BEFORE = 60           # days of dense sampling before onset
DENSE_AFTER = 120           # days of dense sampling after onset
SPARKLINE_STRIDE = 20       # screen 1 signal-level sparkline
PROGNOSIS_PAD_BEFORE = 130  # days of context before the fit window (screen 3)
PROGNOSIS_PAD_AFTER = 90    # days of observed data past the projected crossing
PROGNOSIS_STRIDE = 2        # screen 3 observed series

HIST_BINS = 60              # log-spaced bins for the score distribution
CI_Z = 1.96                 # 95% normal interval on the RUL standard error
PLATEAU_WINDOW = 90         # days averaged to state where the decline settled

# Vocabulary the dashboard must not carry, because the backend cannot support
# it. Checked against the rendered file, case-insensitively.
BANNED_TERMS = (
    r"\bGEO\b", r"\bGSO\b", r"\bSpectratime\b", r"\bPRN\b",
    r"\bunit A\b", r"\bunit B\b", r"\bRAFS MAIN\b", r"\bterminated\b",
    r"advance warning", r"thermal trim", r"thermal intervention",
    r"\bSpace Applications Centre\b", r"\bISRO archive\b",
    r"\bdossier\b", r"sign telemetry", r"ground truth",
    r"\bepoch 20\d\d", r"\bphotocell\b", r"\bC-field\b",
)

# Channel envelopes taken from telemetry.csv, not from any design mock. A
# generated value outside these is a scale error, which is exactly the class of
# bug that produced a 28 V bus voltage in the first design draft.
CHANNEL_BOUNDS = {
    "freq_offset_y": (-1e-10, 1e-10),
    "trb_core_temp_c": (0.0, 20.0),
    "rafs_signal_level": (0.80, 1.10),
    "bus_voltage_v": (45.0, 55.0),
    "steering_correction_ns": (-1e4, 1e4),
}

CHANNEL_META = {
    "freq_offset_y": {
        "label": "Fractional frequency offset",
        "unit": "s/s",
        "format": "sci",
        "note": "Ground-segment steered residual, not a free-running offset.",
    },
    "trb_core_temp_c": {
        "label": "TRB core package temperature",
        "unit": "°C",
        "format": "fixed2",
        "note": "Spans the iRAFS specification band, -5 to +15 °C.",
    },
    "rafs_signal_level": {
        "label": "RAFS signal level",
        "unit": "",
        "format": "fixed4",
        "note": "Dimensionless ratio against the commissioned level. "
                "Failure limit 0.85.",
    },
    "bus_voltage_v": {
        "label": "Bus voltage",
        "unit": "V",
        "format": "fixed2",
        "note": "Nominal 50 V rail.",
    },
    "steering_correction_ns": {
        "label": "Steering correction",
        "unit": "ns",
        "format": "fixed1",
        "note": "Clock offset removed per correction upload, not a daily rate.",
    },
}

PROFILE_LABEL = {
    "healthy": "Healthy",
    "slow_drift": "Slow drift",
    "sudden_jump": "Sudden jump",
    "noise_increase": "Noise increase",
    "degraded": "Long-running degradation",
}

CLOCK_TYPE_LABEL = {"imported": "Imported", "indigenous": "Indigenous"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def r(value, places):
    """Round for serialisation, mapping NaN and None to null."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, places)


def sig(value, digits=4):
    """Round to `digits` significant figures; keeps 1e-13 readable in JSON."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(value))))
    return round(value, -(exponent - digits + 1))


def sample_mask(days, onset):
    """Sparse stride overall, a finer one inside the band around a fault onset.

    Away from a fault the score is flat and a 12-day stride loses nothing. The
    band is the only stretch where the shape carries information.
    """
    days = np.asarray(days)
    keep = np.zeros(len(days), dtype=bool)
    keep[::SPARSE_STRIDE] = True
    if onset is not None:
        band = (days >= onset - DENSE_BEFORE) & (days <= onset + DENSE_AFTER)
        keep |= band & ((days - days[0]) % DENSE_STRIDE == 0)
    keep[-1] = True
    return keep


def longest_run(flags):
    """Longest consecutive run of True. The evidence behind the 60-window rule."""
    flags = np.asarray(flags, dtype=bool)
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        if current > best:
            best = current
    return int(best)


def first_day(days, flags):
    hit = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(np.asarray(days)[hit[0]]) if len(hit) else None


def calibration(scores):
    """Reproduce detect.py's threshold and severity split from scores.csv.

    detect.py computes both at run time and only prints them, so neither exists
    anywhere on disk. The dashboard needs them to draw its threshold lines.
    """
    healthy = scores.loc[scores["profile"] == "healthy",
                         "anomaly_score"].to_numpy()
    mu = float(healthy.mean())
    sd = float(healthy.std())
    threshold = mu + K_SIGMA * sd
    flagged = scores.loc[scores["anomaly_score"] > threshold, "anomaly_score"]
    critical = float(np.quantile(flagged, WARNING_SHARE))
    return {"mu": mu, "sd": sd, "threshold": threshold, "critical": critical,
            "flagged_windows": int(len(flagged))}


# ---------------------------------------------------------------------------
# screen builders
# ---------------------------------------------------------------------------
def build_meta(summary, rul, calib):
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "verify_data.py",
        "provenance": (
            "Synthetic telemetry from generate_telemetry.py (seed 42), modelled "
            "on published iRAFS specification figures and four measured on-orbit "
            "RAFS failure magnitudes. Not flight data."
        ),
        "n_days": int(N_DAYS),
        "years": r(N_DAYS / 365.25, 2),
        "cadence": "one sample per day",
        "window_days": int(WINDOW),
        "channels": [dict(key=name, **CHANNEL_META[name]) for name in CHANNELS],
        "counts": {
            "satellites": int(summary["dataset"]["satellites"]),
            "windows": int(summary["dataset"]["windows"]),
            "healthy_windows": int(summary["dataset"]["healthy_windows"]),
            "anomalous_windows": int(summary["dataset"]["anomalous_windows"]),
            "faulted_satellites": sum(
                1 for s in SATELLITES if s["profile"] != "healthy"),
            "healthy_satellites": sum(
                1 for s in SATELLITES if s["profile"] == "healthy"),
            "imported": sum(
                1 for s in SATELLITES if s["clock_type"] == "imported"),
            "indigenous": sum(
                1 for s in SATELLITES if s["clock_type"] == "indigenous"),
        },
        "calibration": {
            "healthy_mean": r(calib["mu"], 5),
            "healthy_sd": r(calib["sd"], 5),
            "k_sigma": K_SIGMA,
            "threshold": r(calib["threshold"], 5),
            "critical": r(calib["critical"], 5),
            "flagged_windows": calib["flagged_windows"],
            "cusum_persistence_windows": int(
                summary["cusum_persistence_windows"]),
            # The multiplier is read from K_SIGMA, never written out here, so
            # retuning the threshold cannot leave this sentence contradicting
            # the number the dashboard computes from k_sigma alongside it.
            "note": (
                f"Threshold is mean + {K_SIGMA:g} sd of the healthy training "
                "windows, calibrated once on the healthy population rather "
                "than on a per-satellite rolling baseline, which would adapt "
                "to a drift and stop seeing it."
            ),
        },
        "limits": {
            "signal_failure_limit": rul["signal_failure_limit"],
            "steering_limit_ns": r(rul["secondary_limit_ns"], 1),
            "steering_healthy_p95_ns": r(rul["secondary_healthy_p95_ns"], 1),
            "spec_limit_drift_per_day": SPEC_LIMIT_DRIFT,
            "trend_window_days": int(rul["trend_window_days"]),
            "prognosis_lag_days": int(rul["prognosis_lag_days"]),
        },
    }


def build_constellation(telemetry, detections, rul_by_id):
    """Screen 1: roster, current channel values, status, remaining life."""
    rows = []
    for sat in SATELLITES:
        sat_id = sat["id"]
        frame = telemetry[telemetry["satellite_id"] == sat_id].sort_values("day")
        det = detections[detections["satellite_id"] == sat_id].sort_values("end_day")
        record = rul_by_id[sat_id]

        reference_day = int(record["reference_day"])
        at_reference = frame[frame["day"] == reference_day].iloc[0]
        latest = frame.iloc[-1]

        # Two severities, because they answer different questions and a card
        # showing one number cannot answer both. The reference-day reading is
        # the instant the remaining-life fit is anchored to, so status and
        # remaining life describe one moment; the last-day reading is how
        # abnormal the clock looks at the end of the record. For a fault they
        # differ -- the reference day sits only 30 days past detection, before
        # the score has climbed into the Critical band.
        def window_at(day, column):
            row = det[det["end_day"] == day]
            return row.iloc[0][column] if len(row) else None

        last_day = int(latest["day"])

        spark = frame.iloc[::SPARKLINE_STRIDE]
        rul_days = record["remaining_useful_life_days"]
        se_days = record["remaining_useful_life_se_days"]

        rows.append({
            "id": sat_id,
            "clock_type": sat["clock_type"],
            "clock_type_label": CLOCK_TYPE_LABEL[sat["clock_type"]],
            "profile": sat["profile"],
            "profile_label": PROFILE_LABEL[sat["profile"]],
            "is_faulted": sat["profile"] != "healthy",
            "severity_at_reference": str(window_at(reference_day, "severity")),
            "severity_at_last_day": str(window_at(last_day, "severity")),
            "score_at_reference": sig(window_at(reference_day, "anomaly_score")),
            "score_at_last_day": sig(window_at(last_day, "anomaly_score")),
            "anomaly_start_day": (None if sat["anomaly_start_day"] is None
                                  else int(sat["anomaly_start_day"])),
            "detection_day": (None if record["detection_day"] is None
                              else int(record["detection_day"])),
            "detection_latency_days": (
                None if record["detection_day"] is None
                or sat["anomaly_start_day"] is None
                else int(record["detection_day"] - sat["anomaly_start_day"])),
            "reference_day": reference_day,
            "rul_status": record["status"],
            "rul_days": None if rul_days is None else int(rul_days),
            "rul_se_days": se_days,
            "rul_ci95": (None if rul_days is None or se_days is None else
                         [int(round(rul_days - CI_Z * se_days)),
                          int(round(rul_days + CI_Z * se_days))]),
            "signal_slope_per_day": sig(record["signal_slope_per_day"]),
            "signal_level_at_reference": r(record["signal_level_at_reference"], 5),
            "channels_at_reference": {
                name: (sig(at_reference[name]) if name == "freq_offset_y"
                       else r(at_reference[name], 4)) for name in CHANNELS},
            "channels_at_last_day": {
                name: (sig(latest[name]) if name == "freq_offset_y"
                       else r(latest[name], 4)) for name in CHANNELS},
            "last_day": last_day,
            "signal_sparkline": {
                "day": [int(d) for d in spark["day"]],
                "value": [r(v, 4) for v in spark["rafs_signal_level"]],
            },
        })
    return rows


def build_evidence(telemetry, detections, calib):
    """Screen 2: score trajectories, drift residuals, score distribution."""
    series = {}
    for sat in SATELLITES:
        sat_id = sat["id"]
        det = detections[detections["satellite_id"] == sat_id].sort_values("end_day")
        frame = telemetry[telemetry["satellite_id"] == sat_id].sort_values("day")
        onset = sat["anomaly_start_day"]

        score_days = det["end_day"].to_numpy()
        score_keep = sample_mask(score_days, onset)
        tel_days = frame["day"].to_numpy()
        tel_keep = sample_mask(tel_days, onset)

        # The CUSUM statistic is not carried as a series: screen 4 tells the
        # CUSUM-vs-threshold story with detection days and alarm-run lengths,
        # which is the comparison that matters, and the score trajectory here
        # already shows the shape the statistic accumulates over.
        series[sat_id] = {
            "score": {
                "day": [int(d) for d in score_days[score_keep]],
                "value": [sig(v) for v in
                          det["anomaly_score"].to_numpy()[score_keep]],
            },
            "drift": {
                "day": [int(d) for d in tel_days[tel_keep]],
                "value": [sig(v) for v in
                          frame["freq_offset_y"].to_numpy()[tel_keep]],
            },
        }

    # Split on is_true_anomaly, so "nominal" here means every window not inside
    # a fault -- which includes the pre-onset history of the five faulted
    # units. That is a wider population than the healthy-satellite windows
    # detect.py calibrates the threshold on, and the two means differ
    # (0.163 against 0.152). Kept distinct, and named distinctly, because a
    # dashboard that showed them as one number would be wrong.
    scores = detections["anomaly_score"].to_numpy()
    is_anom = detections["is_true_anomaly"].to_numpy(dtype=bool)
    edges = np.logspace(np.log10(scores.min()), np.log10(scores.max()),
                        HIST_BINS + 1)
    nominal_counts, _ = np.histogram(scores[~is_anom], bins=edges)
    anom_counts, _ = np.histogram(scores[is_anom], bins=edges)

    return {
        "series": series,
        "distribution": {
            "scale": "log10",
            "bin_edges": [sig(e, 5) for e in edges],
            "nominal_counts": [int(c) for c in nominal_counts],
            "anomalous_counts": [int(c) for c in anom_counts],
            "nominal_mean": r(float(scores[~is_anom].mean()), 4),
            "anomalous_mean": r(float(scores[is_anom].mean()), 4),
            "nominal_max": r(float(scores[~is_anom].max()), 4),
            "anomalous_min": r(float(scores[is_anom].min()), 4),
            "nominal_n": int((~is_anom).sum()),
            "anomalous_n": int(is_anom.sum()),
            "nominal_population": (
                "every window not inside a fault, including the pre-onset "
                "history of the faulted units"
            ),
            "threshold": r(calib["threshold"], 5),
            "critical": r(calib["critical"], 5),
            "note": (
                "The two populations overlap: the lowest anomalous window "
                "scores below the highest nominal one. Satellite-level "
                "separation comes from persistence, not from a clean cut in "
                "this histogram."
            ),
        },
    }


def build_prognosis(telemetry, rul_by_id, rul):
    """Screen 3: fitted decline, extrapolation to the limit, observed plateau."""
    rows = []
    for sat in SATELLITES:
        sat_id = sat["id"]
        record = rul_by_id[sat_id]
        frame = telemetry[telemetry["satellite_id"] == sat_id].sort_values("day")
        reference_day = int(record["reference_day"])
        lo = reference_day - TREND_WINDOW_DAYS + 1

        entry = {
            "id": sat_id,
            "profile": sat["profile"],
            "profile_label": PROFILE_LABEL[sat["profile"]],
            "clock_type": sat["clock_type"],
            "status": record["status"],
            "anomaly_start_day": (None if sat["anomaly_start_day"] is None
                                  else int(sat["anomaly_start_day"])),
            "detection_day": (None if record["detection_day"] is None
                              else int(record["detection_day"])),
            "reference_day": reference_day,
            "fit_window": [int(lo), reference_day],
            "rul_days": record["remaining_useful_life_days"],
            "rul_se_days": record["remaining_useful_life_se_days"],
            "slope_per_day": sig(record["signal_slope_per_day"]),
            "level_at_reference": r(record["signal_level_at_reference"], 5),
        }

        if record["remaining_useful_life_days"] is None:
            entry.update({"has_projection": False, "fit_line": None,
                          "projection": None, "observed": None, "outcome": None})
            rows.append(entry)
            continue

        window = frame[(frame["day"] >= lo) & (frame["day"] <= reference_day)]
        fit = ols_crossing(window["day"].to_numpy(dtype=float),
                           window["rafs_signal_level"].to_numpy(dtype=float),
                           SIGNAL_FAILURE_LIMIT, declining=True)
        crossing, se = fit["crossing"], fit["se"]

        def line(day):
            return fit["intercept"] + fit["slope"] * float(day)

        # Observed record either side of the projection. Everything past the
        # reference day was unavailable at prognosis time and is shown only to
        # say what the projection got right and where it stopped holding.
        hi = min(int(round(crossing)) + PROGNOSIS_PAD_AFTER,
                 int(frame["day"].max()))
        observed = frame[(frame["day"] >= lo - PROGNOSIS_PAD_BEFORE)
                         & (frame["day"] <= hi)].iloc[::PROGNOSIS_STRIDE]
        after = frame[(frame["day"] > reference_day) & (frame["day"] <= hi)]
        plateau = frame[(frame["day"] >= crossing)
                        & (frame["day"] <= crossing + PLATEAU_WINDOW)]

        entry.update({
            "has_projection": True,
            "rul_ci95": [
                int(round(record["remaining_useful_life_days"] - CI_Z * se)),
                int(round(record["remaining_useful_life_days"] + CI_Z * se)),
            ],
            "crossing_day": r(crossing, 1),
            "crossing_se_days": r(se, 1),
            "failure_limit": SIGNAL_FAILURE_LIMIT,
            "fit_line": {
                "day": [int(lo), reference_day],
                "value": [r(line(lo), 5), r(line(reference_day), 5)],
            },
            "projection": {
                "day": [reference_day, r(crossing, 1)],
                "value": [r(line(reference_day), 5), SIGNAL_FAILURE_LIMIT],
            },
            "observed": {
                "day": [int(d) for d in observed["day"]],
                "value": [r(v, 5) for v in observed["rafs_signal_level"]],
                "known_through_day": reference_day,
            },
            # The honest part. The fit is a straight line through a decline
            # that is not straight for ever; past the projected crossing the
            # observed level flattens well above the limit.
            "outcome": {
                "crossed_limit": bool(
                    (after["rafs_signal_level"] <= SIGNAL_FAILURE_LIMIT).any()),
                "observed_min_after_reference": (
                    r(float(after["rafs_signal_level"].min()), 5)
                    if len(after) else None),
                "observed_plateau_level": (
                    r(float(plateau["rafs_signal_level"].mean()), 5)
                    if len(plateau) else None),
                "plateau_window_days": PLATEAU_WINDOW,
                "note": (
                    "The fitted decline is extrapolated at a constant rate. "
                    "Past the projected crossing the observed level flattens "
                    "above the limit instead of continuing down, so the "
                    "projection bounds the decline rate rather than predicting "
                    "a failure date."
                ),
            },
        })
        rows.append(entry)

    return {
        "indicator": rul["primary_indicator"],
        "failure_limit": rul["signal_failure_limit"],
        "trend_window_days": int(rul["trend_window_days"]),
        "prognosis_lag_days": int(rul["prognosis_lag_days"]),
        "method": (
            "Ordinary least squares on the raw rafs_signal_level channel over "
            "the 90-day window ending on the reference day, extrapolated at a "
            "constant rate to the 0.85 failure limit. The crossing standard "
            "error is propagated by the delta method. The autoencoder residual "
            "drives detection, not this projection."
        ),
        "satellites": rows,
    }


def build_performance(summary, detections):
    """Screen 4: detector metrics, CUSUM vs threshold, the persistence rule."""
    detectors = {}
    for name in ("cusum", "threshold"):
        block = summary["detectors"][name]
        window, satellite = block["window_level"], block["satellite_level"]
        detectors[name] = {
            "label": "CUSUM" if name == "cusum" else "Plain threshold",
            "window_level": {
                "precision": r(window["precision"], 4),
                "recall": r(window["recall"], 4),
                "f1": r(window["f1"], 4),
                "true_positives": int(window["true_positives"]),
                "false_positives": int(window["false_positives"]),
                "true_negatives": int(window["true_negatives"]),
                "false_negatives": int(window["false_negatives"]),
            },
            "satellite_level": {
                "faulted_caught": int(satellite["faulted_caught"]),
                "faulted_total": int(satellite["faulted_total"]),
                "healthy_false_alarms": int(satellite["healthy_false_alarms"]),
                "healthy_total": int(satellite["healthy_total"]),
            },
        }

    # Head-to-head, and the alarm-run evidence behind CUSUM_PERSISTENCE.
    comparison = []
    runs = {"healthy": [], "faulted": []}
    for sat in SATELLITES:
        sat_id = sat["id"]
        group = detections[detections["satellite_id"] == sat_id].sort_values("end_day")
        days = group["end_day"].to_numpy()
        onset = sat["anomaly_start_day"]
        cusum_day = first_day(days, group["cusum_sustained"].to_numpy())
        thresh_day = first_day(days, group["threshold_sustained"].to_numpy())

        cusum_run = longest_run(group["cusum_alarm"].to_numpy())
        thresh_run = longest_run(group["threshold_alarm"].to_numpy())
        runs["healthy" if onset is None else "faulted"].append(
            {"id": sat_id, "cusum": cusum_run, "threshold": thresh_run})

        comparison.append({
            "id": sat_id,
            "profile": sat["profile"],
            "profile_label": PROFILE_LABEL[sat["profile"]],
            "onset_day": None if onset is None else int(onset),
            "cusum_detection_day": cusum_day,
            "threshold_detection_day": thresh_day,
            # Positive latency means the fault was declared after it began.
            # Every fault here is declared late; nothing detects in advance.
            "cusum_latency_days": (None if cusum_day is None or onset is None
                                   else int(cusum_day - onset)),
            "threshold_latency_days": (None if thresh_day is None or onset is None
                                       else int(thresh_day - onset)),
            "cusum_gain_days": (None if cusum_day is None or thresh_day is None
                                else int(thresh_day - cusum_day)),
            "cusum_longest_alarm_run": cusum_run,
            "threshold_longest_alarm_run": thresh_run,
        })

    healthy_mask = ~detections["is_true_anomaly"].to_numpy(dtype=bool)
    latencies = [row["cusum_latency_days"] for row in comparison
                 if row["cusum_latency_days"] is not None]
    gains = [row["cusum_gain_days"] for row in comparison
             if row["cusum_gain_days"] is not None]

    # Which way the head-to-head actually fell, counted rather than asserted.
    # Retuning the detector flips these sentences instead of stranding them.
    gain_by_id = {row["id"]: row["cusum_gain_days"] for row in comparison
                  if row["cusum_gain_days"] is not None}
    first = sum(1 for g in gains if g > 0)
    later = sum(1 for g in gains if g < 0)
    level = sum(1 for g in gains if g == 0)
    if first == len(gains):
        gain_outcome = f"CUSUM declared first on all {len(gains)} faults."
    elif later == len(gains):
        gain_outcome = f"CUSUM declared later on all {len(gains)} faults."
    else:
        parts = [f"first on {first} of {len(gains)} faults", f"later on {later}"]
        if level:
            parts.append(f"level on {level}")
        gain_outcome = "CUSUM declared " + ", ".join(parts) + "."
    best = max(gain_by_id, key=gain_by_id.get)
    worst = min(gain_by_id, key=gain_by_id.get)
    gain_spread = (f"The spread runs from {worst} at {gain_by_id[worst]:+d} "
                   f"days to {best} at {gain_by_id[best]:+d} days.")

    return {
        "roc_auc": r(summary["anomaly_score_roc_auc"], 4),
        "roc_auc_note": (
            "Window-level ranking quality of the raw reconstruction error, "
            "before any threshold or persistence rule is applied."
        ),
        "detectors": detectors,
        "comparison": comparison,
        "latency": {
            "min_days": int(min(latencies)),
            "max_days": int(max(latencies)),
            "mean_days": r(float(np.mean(latencies)), 1),
            "note": (
                "Every fault is declared after it begins, never before. The "
                "persistence rule that holds healthy false alarms at zero is "
                "what costs this latency."
            ),
        },
        "cusum_gain": {
            "min_days": int(min(gains)),
            "max_days": int(max(gains)),
            "mean_days": r(float(np.mean(gains)), 1),
            "note": (
                "Days by which CUSUM declares a fault ahead of the plain "
                "threshold, with the same persistence rule applied to both so "
                "the comparison is matched. Positive means CUSUM declared "
                f"first. {gain_outcome} {gain_spread}"
            ),
        },
        "persistence": {
            "windows_required": int(CUSUM_PERSISTENCE),
            "healthy_single_window_alarm_rate": {
                "cusum": r(float(detections["cusum_alarm"]
                                 .to_numpy()[healthy_mask].mean()), 5),
                "threshold": r(float(detections["threshold_alarm"]
                                     .to_numpy()[healthy_mask].mean()), 5),
            },
            "longest_alarm_run": runs,
            "healthy_max_run": {
                "cusum": max(row["cusum"] for row in runs["healthy"]),
                "threshold": max(row["threshold"] for row in runs["healthy"]),
            },
            "faulted_min_run": {
                "cusum": min(row["cusum"] for row in runs["faulted"]),
                "threshold": min(row["threshold"] for row in runs["faulted"]),
            },
            "note": (
                "A single flagged window is not a detection. Healthy clocks "
                "alarm in short episodes while faults alarm continuously, so a "
                "run-length bar separates them where a per-window bar cannot. "
                "60 sits above every healthy run and far below every fault run."
            ),
        },
    }


# ---------------------------------------------------------------------------
# build / render / verify
# ---------------------------------------------------------------------------
def build_payload():
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    scores = pd.read_csv(INPUT_SCORES)
    detections = pd.read_csv(INPUT_DETECTIONS)
    with open(INPUT_SUMMARY, encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(INPUT_RUL, encoding="utf-8") as handle:
        rul = json.load(handle)

    rul_by_id = {row["satellite_id"]: row for row in rul["satellites"]}
    calib = calibration(scores)

    return {
        "meta": build_meta(summary, rul, calib),
        "constellation": build_constellation(telemetry, detections, rul_by_id),
        "evidence": build_evidence(telemetry, detections, calib),
        "prognosis": build_prognosis(telemetry, rul_by_id, rul),
        "performance": build_performance(summary, detections),
    }


def render(payload):
    """Serialise the payload, keeping numeric arrays on one line each.

    json.dumps(indent=2) puts every array element on its own line, which turns
    a 9,000-point series payload into a 700 KB file no one can read a diff of.
    Numeric arrays are stashed behind a token, rendered inline, and substituted
    back, so the object structure stays indented and the series stay compact.
    """
    inline = {}

    def stash(node):
        if isinstance(node, list):
            if node and all(isinstance(v, (int, float)) or v is None
                            for v in node):
                token = f"@@inline:{len(inline)}@@"
                inline[token] = "[" + ", ".join(
                    json.dumps(v, allow_nan=False) for v in node) + "]"
                return token
            return [stash(v) for v in node]
        if isinstance(node, dict):
            return {key: stash(value) for key, value in node.items()}
        return node

    body = json.dumps(stash(payload), indent=2, sort_keys=True, allow_nan=False)
    for token, rendered_list in inline.items():
        body = body.replace(json.dumps(token), rendered_list)
    return (
        "// NavIC Pulse -- dashboard data layer.\n"
        "// GENERATED FILE. Do not edit by hand.\n"
        "//   regenerate:  python verify_data.py --write\n"
        "//   check:       python verify_data.py\n"
        f"window.{JS_GLOBAL} = {body};\n"
    )


def check_invariants(payload, rendered):
    """Assert what the dashboard is allowed to claim; return a list of failures."""
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    meta = payload["meta"]
    counts = meta["counts"]
    check(counts["satellites"] == 10, "satellite count is not 10")
    check(counts["faulted_satellites"] == 5 and counts["healthy_satellites"] == 5,
          "faulted/healthy split is not 5/5")
    check(counts["imported"] == 8 and counts["indigenous"] == 2,
          "imported/indigenous split is not 8/2")
    check(meta["n_days"] == 2900, "record length is not 2900 days")
    check(len(meta["channels"]) == 5, "channel list is not the 5 model inputs")

    # Channel scales. The first design draft carried a 28 V bus and a 42 C
    # clock package; these bounds are what make shipping that impossible.
    for row in payload["constellation"]:
        for field in ("channels_at_reference", "channels_at_last_day"):
            for name, value in row[field].items():
                low, high = CHANNEL_BOUNDS[name]
                check(value is not None and low <= value <= high,
                      f"{row['id']}.{field}.{name} = {value}, "
                      f"outside {low} .. {high}")

    # No advance warning: every fault is declared after onset.
    for row in payload["performance"]["comparison"]:
        for detector in ("cusum", "threshold"):
            latency = row[f"{detector}_latency_days"]
            if latency is not None:
                check(latency > 0,
                      f"{row['id']} {detector} latency {latency} is not "
                      f"positive -- that would be an advance-warning claim")
    check(payload["performance"]["latency"]["min_days"] > 0,
          "minimum detection latency is not positive")

    # Zero healthy false alarms, on both detectors.
    for name, block in payload["performance"]["detectors"].items():
        check(block["satellite_level"]["healthy_false_alarms"] == 0,
              f"{name}: healthy false alarms is not zero")
        check(block["satellite_level"]["faulted_caught"] == 5,
              f"{name}: faulted caught is not 5")
        check(block["window_level"]["false_positives"] == 0,
              f"{name}: window-level false positives is not zero")

    # The persistence bar has to actually separate the two populations.
    persistence = payload["performance"]["persistence"]
    check(persistence["healthy_max_run"]["cusum"] < persistence["windows_required"],
          "a healthy unit sustains an alarm run at or past the persistence bar")
    check(persistence["faulted_min_run"]["cusum"] > persistence["windows_required"],
          "a fault fails to sustain an alarm run past the persistence bar")

    # No satellite reaches the failure limit, so no projection may claim it did.
    limit = payload["prognosis"]["failure_limit"]
    for row in payload["prognosis"]["satellites"]:
        if not row.get("has_projection"):
            continue
        check(row["outcome"]["crossed_limit"] is False,
              f"{row['id']} claims an observed crossing of the failure limit")
        check(row["outcome"]["observed_min_after_reference"] > limit,
              f"{row['id']} observed level falls below the failure limit")

    # Healthy units carry no remaining-life number.
    for row in payload["constellation"]:
        if not row["is_faulted"]:
            check(row["rul_days"] is None,
                  f"{row['id']} is healthy but carries a remaining-life figure")

    # Vocabulary the backend cannot support.
    for pattern in BANNED_TERMS:
        match = re.search(pattern, rendered, re.IGNORECASE)
        check(match is None,
              f"banned term: /{pattern}/ matched "
              f"{match.group(0)!r}" if match else "")

    return failures


def report(payload, rendered, failures, writing):
    width = 78
    print()
    print("docs/data.js  --  " + ("build" if writing else "verify"))
    print("=" * width)

    meta = payload["meta"]
    counts = meta["counts"]
    print(f"  {counts['satellites']} satellites, {counts['windows']:,} windows, "
          f"{meta['n_days']} days ({meta['years']} years), "
          f"{len(meta['channels'])} channels")
    print(f"  threshold {meta['calibration']['threshold']}   "
          f"critical {meta['calibration']['critical']}   "
          f"persistence {meta['calibration']['cusum_persistence_windows']} windows")

    print()
    print(f"{'SCREEN':<16}{'ROWS':>6}  CONTENT")
    print("-" * width)
    series = payload["evidence"]["series"]
    points = sum(len(block[name]["day"]) for block in series.values()
                 for name in ("score", "drift"))
    projected = sum(1 for row in payload["prognosis"]["satellites"]
                    if row["has_projection"])
    print(f"{'constellation':<16}{len(payload['constellation']):>6}  "
          f"status, channel values, remaining life")
    print(f"{'evidence':<16}{len(series):>6}  "
          f"{points:,} sampled series points, "
          f"{len(payload['evidence']['distribution']['bin_edges']) - 1} "
          f"histogram bins")
    print(f"{'prognosis':<16}{projected:>6}  "
          f"fitted decline, extrapolation, observed plateau")
    print(f"{'performance':<16}"
          f"{len(payload['performance']['comparison']):>6}  "
          f"ROC AUC {payload['performance']['roc_auc']}, "
          f"2 detectors, persistence evidence")
    print("-" * width)
    print(f"{'total':<16}{len(rendered) / 1024:>6.1f}  KB rendered")

    print()
    print("INVARIANTS")
    print("-" * width)
    if failures:
        for message in failures:
            print(f"  FAIL  {message}")
    else:
        latency = payload["performance"]["latency"]
        prognosis = payload["prognosis"]["satellites"]
        plateau = [row["outcome"]["observed_plateau_level"] for row in prognosis
                   if row["has_projection"]]
        print(f"  ok  roster 10 = 5 faulted + 5 healthy = 8 imported + 2 indigenous")
        print(f"  ok  all channel values inside the telemetry.csv envelopes")
        print(f"  ok  every fault declared after onset "
              f"({latency['min_days']}-{latency['max_days']} d latency, "
              f"no advance warning)")
        print(f"  ok  zero healthy false alarms on both detectors")
        print(f"  ok  persistence bar separates healthy runs from fault runs")
        print(f"  ok  no observed crossing of the {payload['prognosis']['failure_limit']} "
              f"limit (plateau {min(plateau)}-{max(plateau)})")
        print(f"  ok  healthy units carry no remaining-life figure")
        print(f"  ok  none of the {len(BANNED_TERMS)} unsupported terms present")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Build or verify docs/data.js against the pipeline outputs.")
    parser.add_argument("--write", action="store_true",
                        help="regenerate docs/data.js instead of verifying it")
    args = parser.parse_args()

    payload = build_payload()
    rendered = render(payload)
    failures = check_invariants(payload, rendered)
    report(payload, rendered, failures, args.write)

    if args.write:
        os.makedirs(os.path.dirname(OUTPUT_JS), exist_ok=True)
        with open(OUTPUT_JS, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        print(f"wrote {OUTPUT_JS}")
        print()
        return 1 if failures else 0

    if not os.path.exists(OUTPUT_JS):
        print(f"{OUTPUT_JS} does not exist. Run: python verify_data.py --write")
        print()
        return 1

    with open(OUTPUT_JS, encoding="utf-8", newline="") as handle:
        on_disk = handle.read().replace("\r\n", "\n")

    if on_disk == rendered:
        print(f"{OUTPUT_JS} matches the pipeline outputs")
    else:
        print(f"{OUTPUT_JS} is STALE -- it no longer matches the pipeline "
              f"outputs.\nRun: python verify_data.py --write")
        failures.append("data.js out of date")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
