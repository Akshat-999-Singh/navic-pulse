"""
NavIC Pulse -- export the demo fixture the React frontend reads.

The frontend has no backend: frontend/public/demo-run.json IS its data source.
This script is a formatter over existing pipeline outputs, not an analysis step.
It reads:

  detections.csv            per-window score, severity, alarms (detect.py)
  window_metadata.csv       row order of preprocessed.npz X_all (preprocess.py)
  preprocessed.npz          X_all, re-scored below for channel contributions
  model.joblib              the frozen autoencoder (train_model.py) -- NOT retrained
  rul.json                  remaining useful life (rul.py)
  navic_recalibration.json  the shipped static threshold (mu + K_SIGMA * sd)
  baseline_ool.json         raw-channel fixed-limit baseline (baseline_ool.py)
  summary.json              its mtime stamps generated_at, so reruns are identical
  telemetry.csv             source of the sample upload batch

The only values derived here are aggregates and presentation choices, each named
where it is computed. Nothing is fitted.

Outputs:
  frontend/public/demo-run.json
  frontend/public/sample-batch.csv
"""

import base64
import io
import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHANNELS, SATELLITES, WINDOW

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DETECTIONS = os.path.join(HERE, "detections.csv")
INPUT_META = os.path.join(HERE, "window_metadata.csv")
INPUT_NPZ = os.path.join(HERE, "preprocessed.npz")
INPUT_MODEL = os.path.join(HERE, "model.joblib")
INPUT_RUL = os.path.join(HERE, "rul.json")
INPUT_RECAL = os.path.join(HERE, "navic_recalibration.json")
INPUT_BASELINE = os.path.join(HERE, "baseline_ool.json")
INPUT_SUMMARY = os.path.join(HERE, "summary.json")
INPUT_TELEMETRY = os.path.join(HERE, "telemetry.csv")

PUBLIC_DIR = os.path.join(HERE, "frontend", "public")
OUTPUT_JSON = os.path.join(PUBLIC_DIR, "demo-run.json")
OUTPUT_BATCH = os.path.join(PUBLIC_DIR, "sample-batch.csv")

MAX_FIXTURE_BYTES = 8 * 1024 * 1024
PLOT_DPI_STEPS = (110, 80, 60, 45)

MAX_FLAGGED = 300         # flagged windows per satellite, evenly thinned
MAX_CONTEXT = 100         # unflagged windows sampled across the pre-fault span
CONTEXT_STRIDE = 10       # healthy satellites: every Nth window

SAMPLE_SAT = "IRNSS-1I"   # slow_drift; the slice straddles its day-1750 onset
SAMPLE_FIRST_DAY = 1650
SAMPLE_ROWS = 200

# ===========================================================================
# STATIC REFERENCE DATA -- NOT PIPELINE OUTPUT
# Public launch record for the roster in config.SATELLITES. Hand-maintained.
# clock_type here must agree with the pipeline's own clock_type (checked below).
# ===========================================================================
REFERENCE = {
    "IRNSS-1A": {"launch_date": "2013-07-01", "orbit": "IGSO", "clock_type": "imported"},
    "IRNSS-1B": {"launch_date": "2014-04-04", "orbit": "IGSO", "clock_type": "imported"},
    "IRNSS-1C": {"launch_date": "2014-10-16", "orbit": "GEO", "clock_type": "imported"},
    "IRNSS-1D": {"launch_date": "2015-03-28", "orbit": "IGSO", "clock_type": "imported"},
    "IRNSS-1E": {"launch_date": "2016-01-20", "orbit": "IGSO", "clock_type": "imported"},
    "IRNSS-1F": {"launch_date": "2016-03-10", "orbit": "GEO", "clock_type": "imported"},
    "IRNSS-1G": {"launch_date": "2016-04-28", "orbit": "GEO", "clock_type": "imported"},
    "IRNSS-1I": {"launch_date": "2018-04-11", "orbit": "IGSO", "clock_type": "imported"},
    "NVS-01":   {"launch_date": "2023-05-29", "orbit": "GEO", "clock_type": "irafs"},   # 129.55 E
    # Never reached its slot; the orbit is the intended one, not an achieved one.
    "NVS-02":   {"launch_date": "2025-01-29", "orbit": "IGSO (intended)", "clock_type": "irafs"},
}
# ===========================================================================

CLOCK_TYPE = {"imported": "imported", "indigenous": "irafs"}
SEVERITY_RANK = {"normal": 0, "warning": 1, "critical": 2}

# Frontend :root tokens (frontend/src/styles.css), so the plots sit on the page.
GROUND = "#0d0d0f"
INK = (236 / 255, 230 / 255, 218 / 255)
ACCENT = "#5fa8f0"
WARNING = "#f0ac5f"
CRITICAL = "#f05f5f"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load():
    detections = pd.read_csv(INPUT_DETECTIONS)
    meta = pd.read_csv(INPUT_META)
    with open(INPUT_RUL, encoding="utf-8") as handle:
        rul = json.load(handle)
    with open(INPUT_RECAL, encoding="utf-8") as handle:
        threshold = float(json.load(handle)["calibration"]["static_threshold"])

    # The threshold lives in a different file from the alarms it produced; a
    # stale navic_recalibration.json would silently mislabel every window.
    recomputed = detections["anomaly_score"] > threshold
    if not (recomputed == detections["threshold_alarm"]).all():
        raise SystemExit("static_threshold does not reproduce threshold_alarm -- "
                         "re-run detect.py / navic_recalibrate.py")

    if not os.path.exists(INPUT_BASELINE):
        raise SystemExit("baseline_ool.json is missing -- run baseline_ool.py first")
    with open(INPUT_BASELINE, encoding="utf-8") as handle:
        baseline_meta = json.load(handle)
    baseline = {b["satellite_id"]: b for b in baseline_meta["satellites"]}
    return detections, meta, rul, threshold, baseline, baseline_meta


def channel_contributions(meta, detections):
    """Per-window share of reconstruction error by channel, from the frozen model.

    Re-inference only: model.joblib is loaded and applied to X_all as saved.
    X_all rows are day-major, [d0c0..d0c4, d1c0..d1c4, ...] (preprocess.py), so
    reshaping to (windows, WINDOW, channels) and summing the time axis gives the
    per-channel squared residual across the window's 30 positions.
    """
    x_all = np.load(INPUT_NPZ, allow_pickle=True)["X_all"]
    if len(x_all) != len(meta):
        raise SystemExit("preprocessed.npz and window_metadata.csv are out of step")
    model = joblib.load(INPUT_MODEL)
    residual = (x_all - model.predict(x_all)) ** 2

    # The per-feature residuals must average back to the published score, or
    # the model/arrays on disk are not the ones detections.csv came from.
    keyed = meta[["satellite_id", "end_day"]].assign(score=residual.mean(axis=1))
    check = keyed.merge(detections[["satellite_id", "end_day", "anomaly_score"]],
                        on=["satellite_id", "end_day"], validate="one_to_one")
    if not np.allclose(check["score"], check["anomaly_score"], rtol=1e-6, atol=1e-9):
        raise SystemExit("re-inferred scores do not match detections.csv -- "
                         "model.joblib or preprocessed.npz has changed since scoring")

    per_channel = residual.reshape(len(x_all), WINDOW, len(CHANNELS)).sum(axis=1)
    shares = per_channel / per_channel.sum(axis=1, keepdims=True)
    frame = meta[["satellite_id", "end_day"]].copy()
    frame[CHANNELS] = shares
    return frame.set_index(["satellite_id", "end_day"])


# ---------------------------------------------------------------------------
# Per-satellite fields
# ---------------------------------------------------------------------------
def first_index(flags):
    hit = np.flatnonzero(flags)
    return int(hit[0]) if len(hit) else None


def select_windows(flagged, keep):
    """Indices to ship for one satellite.

    Faulted: up to MAX_FLAGGED flagged windows, evenly thinned, plus up to
    MAX_CONTEXT unflagged windows evenly spaced over the pre-fault span (every
    window before the first flag). Healthy: every CONTEXT_STRIDE-th window.
    `keep` (first flag, persistence confirmation, latest window) always survives
    and counts against the budget of the tier it belongs to.
    """
    n = len(flagged)
    idx = np.arange(n)
    first = first_index(flagged)
    if first is None:
        extra = [i for i in keep if i is not None]
        return np.union1d(idx[idx % CONTEXT_STRIDE == 0], extra).astype(int), False

    required = np.array(sorted({i for i in keep if i is not None}), dtype=int)

    def thin(pool, count):
        pool = np.setdiff1d(pool, required)
        if count <= 0 or len(pool) == 0:
            return np.array([], dtype=int)
        if len(pool) <= count:
            return pool
        return pool[np.linspace(0, len(pool) - 1, count).round().astype(int)]

    all_flagged = idx[flagged]
    pre_fault = idx[(idx < first) & ~flagged]
    kept_flagged = thin(all_flagged, MAX_FLAGGED - int(flagged[required].sum()))
    kept_context = thin(pre_fault, MAX_CONTEXT - int((~flagged[required]).sum()))
    chosen = np.union1d(np.union1d(kept_flagged, kept_context), required).astype(int)
    thinned = len(np.setdiff1d(all_flagged, chosen)) > 0
    return chosen, thinned


def plot_png(days, scores, flagged, threshold, confirmed, dpi):
    """Reconstruction error over time, threshold, flagged spans, confirmed onset."""
    muted = (*INK, 0.6)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)

    edges = np.diff(np.concatenate([[0], flagged.astype(int), [0]]))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(days[start], days[stop - 1], color=WARNING, alpha=0.16, linewidth=0)

    ax.plot(days, scores, color=ACCENT, linewidth=0.9)
    ax.axhline(threshold, color=muted, linestyle=":", linewidth=1.2)
    if confirmed is not None:
        ax.axvline(days[confirmed], color=CRITICAL, linestyle="--", linewidth=1.3)

    ax.set_yscale("log")
    ax.set_xlim(days[0], days[-1])
    ax.set_xlabel("Window start day", color=muted, fontsize=9)
    ax.set_ylabel("Reconstruction error", color=muted, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color((*INK, 0.14))
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=GROUND)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_satellite(sat, detections, contributions, rul_by_id, threshold, baseline_by_id):
    sat_id = sat["id"]
    ref = REFERENCE[sat_id]
    clock_type = CLOCK_TYPE[sat["clock_type"]]
    if ref["clock_type"] != clock_type:
        raise SystemExit(f"{sat_id}: REFERENCE clock_type {ref['clock_type']!r} "
                         f"disagrees with the pipeline's {clock_type!r}")

    g = detections[detections["satellite_id"] == sat_id].sort_values("end_day")
    g = g.reset_index(drop=True)
    end_days = g["end_day"].to_numpy()
    scores = g["anomaly_score"].to_numpy()
    flagged = g["threshold_alarm"].to_numpy(dtype=bool)
    severity = g["severity"].str.lower().to_numpy()

    first_flagged = first_index(flagged)
    confirmed = first_index(g["cusum_sustained"].to_numpy(dtype=bool))
    # Sustained raw-channel OOL detection, same persistence rule as CUSUM.
    # lead > 0: CUSUM confirmed that many days before the OOL baseline.
    baseline = baseline_by_id[sat_id]["baseline_sustained_index"]
    lead = (None if baseline is None or confirmed is None
            else int(baseline - confirmed))

    r = rul_by_id[sat_id]
    if r["remaining_useful_life_days"] is None:
        remaining = None
    else:
        est = int(r["remaining_useful_life_days"])
        se = r["remaining_useful_life_se_days"] or 0.0
        remaining = {"estimate": est, "low": int(round(est - se)),
                     "high": int(round(est + se))}

    last = len(g) - 1
    chosen, thinned = select_windows(flagged, keep=(first_flagged, confirmed, last))
    shares = contributions.loc[sat_id].reindex(end_days[chosen])
    windows = []
    for i, row in zip(chosen, shares.itertuples(index=False)):
        end = int(end_days[i])
        windows.append({
            "index": int(i),
            "start": end - WINDOW + 1,
            "end": end,
            "error": round(float(scores[i]), 6),
            "flagged": bool(flagged[i]),
            "channels": [{"name": name, "contribution": round(float(v), 5)}
                         for name, v in zip(CHANNELS, row)],
        })

    record = {
        "id": sat_id,
        "name": sat_id,
        "clock_type": clock_type,
        "launch_date": ref["launch_date"],
        "orbit": ref["orbit"],
        "severity": severity[last],
        "peak_severity": max(severity, key=SEVERITY_RANK.__getitem__),
        "anomaly_score": round(float(scores[last]), 6),
        "threshold": round(threshold, 6),
        "first_flagged_index": first_flagged,
        "persistence_confirmed_index": confirmed,
        "baseline_flagged_index": baseline,
        "lead_time_days": lead,
        "remaining_life_days": remaining,
        "plot": None,
        "windows": windows,
    }
    series = (end_days - WINDOW + 1, scores, flagged, confirmed)
    return record, series, thinned


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def clock_comparison(detections):
    """Healthy imported vs indigenous mean error -- reported, never ranked.

    The two healthy populations are a statistical tie on this synthetic data, so
    the block carries both raw means and an UNSIGNED difference, and no score a
    frontend could order. The all-satellite comparison is left out entirely: every
    faulted unit is imported, so it measures fault assignment, not clock quality.
    """
    healthy = detections[detections["profile"] == "healthy"]
    means = healthy.groupby("clock_type")["anomaly_score"].mean()
    imported, irafs = float(means["imported"]), float(means["indigenous"])
    # Relative to the midpoint, so neither population is the reference.
    pct = abs(imported - irafs) / ((imported + irafs) / 2) * 100
    return {
        "healthy_imported_mean_error": round(imported, 6),
        "healthy_irafs_mean_error": round(irafs, 6),
        "difference_pct": round(pct, 2),
        "conclusive": False,
        "note": (f"Healthy imported and indigenous clock populations are "
                 f"statistically indistinguishable in this synthetic telemetry "
                 f"({pct:.1f}% apart). The comparison becomes meaningful with real "
                 f"per-clock data. The all-satellite figure is confounded: every "
                 f"faulted satellite carries an imported clock."),
    }


def lead_time_detail(satellites, baseline_by_id, baseline_meta):
    """Per-fault spread of lead time, with the note's claims checked, not assumed."""
    onsets = {s["id"]: s["anomaly_start_day"] for s in SATELLITES}
    faults = [s for s in satellites if onsets[s["id"]] is not None]
    timed = [s for s in faults if s["lead_time_days"] is not None]
    leads = [s["lead_time_days"] for s in timed]
    if not leads:
        return None

    # Onset is a day; the first window ending on it has index onset - (WINDOW - 1).
    for s in timed:
        onset_index = onsets[s["id"]] - (WINDOW - 1)
        if min(s["persistence_confirmed_index"], s["baseline_flagged_index"]) < onset_index:
            raise SystemExit(f"{s['id']}: a detector now confirms before onset -- "
                             "the lead_time_detail note no longer holds, revise it")

    channels = sorted({c for s in faults
                       for c in baseline_by_id[s["id"]]["channels_triggered"]})
    beaten = [s for s in timed if s["lead_time_days"] < 0]
    if beaten:
        beaten_txt = " ".join(
            f"On {s['id']} the baseline confirms {-s['lead_time_days']} "
            f"day{'s' if s['lead_time_days'] != -1 else ''} earlier." for s in beaten)
    else:
        beaten_txt = "CUSUM confirms first on every fault."

    return {
        "mean_days": round(float(np.mean(leads)), 2),
        "min_days": int(min(leads)),
        "max_days": int(max(leads)),
        "faults_where_cusum_earlier": sum(1 for v in leads if v > 0),
        "faults_total": len(faults),
        "baseline_channels_triggered": channels,
        "note": (f"Lead time is measured against a fixed-limit baseline on raw "
                 f"telemetry channels, limits set at mean +/- {baseline_meta['k_sigma']:g} "
                 f"sigma from healthy satellites, same "
                 f"{baseline_meta['persistence_windows']}-window persistence rule. "
                 f"{beaten_txt} Both detectors confirm after fault onset, never before. "
                 f"Positive lead_time_days means CUSUM confirmed first."),
    }


def summarise(satellites, detections, baseline_by_id, baseline_meta):
    leads = [s["lead_time_days"] for s in satellites if s["lead_time_days"] is not None]
    return {
        "satellites": len(satellites),
        # A satellite counts as flagged once the pipeline's detection -- the
        # 60-window persistence rule -- confirms, not on a single-window alarm.
        "flagged": sum(1 for s in satellites if s["persistence_confirmed_index"] is not None),
        "mean_lead_time_days": round(float(np.mean(leads)), 2) if leads else None,
        "lead_time_detail": lead_time_detail(satellites, baseline_by_id, baseline_meta),
        "clock_comparison": clock_comparison(detections),
    }


# ---------------------------------------------------------------------------
def main():
    detections, meta, rul, threshold, baseline_by_id, baseline_meta = load()
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    contributions = channel_contributions(meta, detections)
    rul_by_id = {r["satellite_id"]: r for r in rul["satellites"]}

    pipeline_ids = set(detections["satellite_id"])
    roster_ids = {s["id"] for s in SATELLITES}
    if pipeline_ids != roster_ids or roster_ids != set(REFERENCE):
        raise SystemExit(f"roster mismatch: detections {sorted(pipeline_ids)}, "
                         f"config {sorted(roster_ids)}, REFERENCE {sorted(REFERENCE)}")

    satellites, series, thinned = [], [], []
    for sat in SATELLITES:
        record, s, was_thinned = build_satellite(
            sat, detections, contributions, rul_by_id, threshold, baseline_by_id)
        satellites.append(record)
        series.append(s)
        if was_thinned:
            thinned.append(sat["id"])

    payload = {
        # The pipeline run's time, not the export's: reruns stay byte-identical.
        "generated_at": datetime.fromtimestamp(os.path.getmtime(INPUT_SUMMARY),
                                               timezone.utc).isoformat(timespec="seconds"),
        "cadence": "daily",
        "synthetic": True,
        "source": "demo",
        "satellites": satellites,
        "summary": summarise(satellites, detections, baseline_by_id, baseline_meta),
    }

    os.makedirs(PUBLIC_DIR, exist_ok=True)
    sizes = []
    for dpi in PLOT_DPI_STEPS:
        for record, (days, scores, flagged, confirmed) in zip(satellites, series):
            record["plot"] = plot_png(days, scores, flagged, threshold, confirmed, dpi)
        text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        size = len(text.encode("utf-8"))
        sizes.append((dpi, size))
        if size <= MAX_FIXTURE_BYTES:
            break
    with open(OUTPUT_JSON, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    batch = telemetry[(telemetry["satellite_id"] == SAMPLE_SAT)
                      & (telemetry["day"] >= SAMPLE_FIRST_DAY)].sort_values("day")
    batch.head(SAMPLE_ROWS).to_csv(OUTPUT_BATCH, index=False)

    report(payload, sizes, thinned, len(batch.head(SAMPLE_ROWS)))


def report(payload, sizes, thinned, batch_rows):
    width = 96
    print()
    print("Demo fixture")
    print("=" * width)
    print(f"  {'SATELLITE':<11}{'CLOCK':<10}{'SEVERITY':<10}{'PEAK':<10}{'SCORE':>10}"
          f"{'CONFIRMED':>11}{'OOL':>7}{'LEAD':>7}{'RUL':>16}{'WINDOWS':>9}")
    print("-" * width)
    dash = lambda v: "--" if v is None else str(v)
    for s in payload["satellites"]:
        r = s["remaining_life_days"]
        rul = "--" if r is None else f"{r['estimate']} [{r['low']}-{r['high']}]"
        print(f"  {s['id']:<11}{s['clock_type']:<10}{s['severity']:<10}{s['peak_severity']:<10}"
              f"{s['anomaly_score']:>10.4f}{dash(s['persistence_confirmed_index']):>11}"
              f"{dash(s['baseline_flagged_index']):>7}{dash(s['lead_time_days']):>7}"
              f"{rul:>16}{len(s['windows']):>9}")
    print("-" * width)
    summary = payload["summary"]
    lead = summary["mean_lead_time_days"]
    print(f"  satellites          {summary['satellites']}")
    print(f"  flagged             {summary['flagged']}")
    print(f"  mean lead time      {'null' if lead is None else f'{lead:+} d'}  "
          f"(OOL sustained - CUSUM confirmed; + means CUSUM earlier)")
    detail = summary["lead_time_detail"]
    if detail:
        print(f"  lead time range     {detail['min_days']:+d} to {detail['max_days']:+d} d, "
              f"CUSUM earlier on {detail['faults_where_cusum_earlier']}/{detail['faults_total']}")
    print("  clock_comparison")
    for line in json.dumps(summary["clock_comparison"], indent=2).splitlines():
        print(f"    {line}")
    generated = payload["generated_at"]
    print(f"  generated_at        {generated}  (summary.json mtime)")
    if thinned:
        print(f"  flagged thinned to {MAX_FLAGGED}: {', '.join(thinned)}")
    for dpi, size in sizes:
        print(f"  size at {dpi:>3} dpi      {size / 1024 / 1024:.2f} MB")
    print(f"  wrote {os.path.relpath(OUTPUT_JSON, HERE)}")
    print(f"  wrote {os.path.relpath(OUTPUT_BATCH, HERE)}  ({batch_rows} rows, {SAMPLE_SAT})")
    print()


if __name__ == "__main__":
    main()
