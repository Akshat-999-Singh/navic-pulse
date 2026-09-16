"""One request's run: ingest -> window -> infer -> detect -> baseline -> explain -> prognosis.

Produces the demo-run.json shape (export_demo_run.py) with "source": "computed",
no "plot", and "stages" holding each module's measured wall time.

Differences from the fixture, all forced by an upload carrying no ground truth:
  * window "error" and "anomaly_score" are full precision, so they can be
    compared exactly against the offline pipeline;
  * lead_time_detail counts satellites the model confirmed rather than known
    faults, and makes no claim about onset, which an upload does not record;
  * clock_comparison's "healthy" populations are the satellites the model never
    flagged; either one missing makes the block null.
"""

import time
from datetime import datetime, timezone

import numpy as np

from . import baseline, detect, explain, infer, ingest, prognosis, window
from .meta import CHANNELS, PERSISTENCE, THRESHOLD, WINDOW

MAX_FLAGGED = 300         # flagged windows per satellite, evenly thinned
MAX_CONTEXT = 100         # unflagged windows sampled across the pre-fault span
CONTEXT_STRIDE = 10       # never-flagged satellites: every Nth window

CLOCK_TYPE = {"imported": "imported", "indigenous": "irafs", "irafs": "irafs"}
SEVERITY_RANK = {"normal": 0, "warning": 1, "critical": 2}
METADATA_FIELDS = ("name", "launch_date", "clock_type", "orbit")


class Stages:
    def __init__(self):
        self.timings = []

    def __call__(self, name, fn, *args):
        start = time.perf_counter()
        result = fn(*args)
        self.timings.append({"name": name,
                             "ms": round((time.perf_counter() - start) * 1000, 2)})
        return result


# ---------------------------------------------------------------------------
# Window selection -- export_demo_run.select_windows
# ---------------------------------------------------------------------------
def select_windows(flagged, keep):
    n = len(flagged)
    idx = np.arange(n)
    first = detect.first_index(flagged)
    if first is None:
        extra = [i for i in keep if i is not None]
        return np.union1d(idx[idx % CONTEXT_STRIDE == 0], extra).astype(int)

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
    return np.union1d(np.union1d(kept_flagged, kept_context), required).astype(int)


# ---------------------------------------------------------------------------
# Stage bodies, one per module
# ---------------------------------------------------------------------------
def _window(parsed):
    raw = {sid: window.raw_matrix(rec) for sid, rec in parsed.items()}
    features = {sid: window.build(infer.scale(r)) for sid, r in raw.items()}
    return raw, features


def _infer(features):
    squared = {sid: infer.residuals(f) for sid, f in features.items()}
    return squared, {sid: infer.scores(sq) for sid, sq in squared.items()}


def _detect(scores):
    return {sid: detect.run(s) for sid, s in scores.items()}


def _explain(squared):
    return {sid: explain.shares(sq) for sid, sq in squared.items()}


def _prognosis(parsed, detections):
    out = {}
    for sid, rec in parsed.items():
        confirmed = detections[sid]["persistence_confirmed_index"]
        detection_day = None if confirmed is None else rec["days"][confirmed + WINDOW - 1]
        out[sid] = prognosis.remaining_life(rec["days"], rec[prognosis.INDICATOR], detection_day)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _never_flagged(d):
    return d["first_flagged_index"] is None and d["persistence_confirmed_index"] is None


def lead_time_detail(satellites, triggered, baseline_source):
    confirmed = [s for s in satellites if s["persistence_confirmed_index"] is not None]
    timed = [s for s in confirmed if s["lead_time_days"] is not None]
    leads = [s["lead_time_days"] for s in timed]
    if not leads:
        return None

    beaten = [s for s in timed if s["lead_time_days"] < 0]
    if beaten:
        beaten_txt = " ".join(
            f"On {s['id']} the baseline confirms {-s['lead_time_days']} "
            f"day{'s' if s['lead_time_days'] != -1 else ''} earlier." for s in beaten)
    else:
        beaten_txt = "CUSUM confirms first on every detection."
    if baseline_source["kind"] == "meta":
        calibrated = "from the model's healthy training fleet"
    else:
        ids = baseline_source["satellites"]
        calibrated = (f"from the {len(ids)} satellite{'s' if len(ids) != 1 else ''} in this "
                      f"batch the model never flagged ({', '.join(ids)})")

    return {
        "mean_days": round(float(np.mean(leads)), 2),
        "min_days": int(min(leads)),
        "max_days": int(max(leads)),
        "faults_where_cusum_earlier": sum(1 for v in leads if v > 0),
        "faults_total": len(confirmed),
        "baseline_channels_triggered": sorted({c for s in confirmed for c in triggered[s["id"]]}),
        "note": (f"Lead time is measured against a fixed-limit baseline on raw telemetry "
                 f"channels, limits set at mean +/- {baseline.K_SIGMA_OOL:g} sigma "
                 f"{calibrated}, same {PERSISTENCE}-window persistence rule. {beaten_txt} "
                 f"Both detectors start from the first uploaded day; an upload records no "
                 f"fault onset, so detections are not checked against one. "
                 f"Positive lead_time_days means CUSUM confirmed first."),
    }


def clock_comparison(satellites, scores, detections):
    groups = {"imported": [], "irafs": []}
    for s in satellites:
        if _never_flagged(detections[s["id"]]) and s["clock_type"] in groups:
            groups[s["clock_type"]].append(s["id"])
    if not groups["imported"] or not groups["irafs"]:
        return None

    def mean(ids):
        return float(np.concatenate([scores[i] for i in ids]).mean())

    imported, irafs = mean(groups["imported"]), mean(groups["irafs"])
    pct = abs(imported - irafs) / ((imported + irafs) / 2) * 100
    return {
        "healthy_imported_mean_error": round(imported, 6),
        "healthy_irafs_mean_error": round(irafs, 6),
        "difference_pct": round(pct, 2),
        "conclusive": False,
        "note": (f"Mean reconstruction error over the satellites in this batch the model "
                 f"never flagged: {len(groups['imported'])} imported, "
                 f"{len(groups['irafs'])} indigenous ({pct:.1f}% apart). Reported, not "
                 f"ranked: one uploaded batch cannot separate clock quality."),
    }


# ---------------------------------------------------------------------------
def build_satellite(sid, rec, score, det, shares, base, remaining, metadata):
    days = rec["days"]
    n = len(score)
    flagged = det["threshold_alarm"]
    severity = det["severity"]
    confirmed = det["persistence_confirmed_index"]
    base_index = base[0]
    lead = None if base_index is None or confirmed is None else int(base_index - confirmed)

    last = n - 1
    chosen = select_windows(flagged, keep=(det["first_flagged_index"], confirmed, last))
    windows = [{
        "index": int(i),
        "start": int(days[i]),
        "end": int(days[i + WINDOW - 1]),
        "error": float(score[i]),
        "flagged": bool(flagged[i]),
        "channels": [{"name": name, "contribution": round(float(v), 5)}
                     for name, v in zip(CHANNELS, shares[i])],
    } for i in chosen]

    csv_clock = rec["clock_type"].strip().lower()
    applies = metadata is not None
    return {
        "id": sid,
        "name": metadata.get("name") if applies else None,
        "clock_type": (metadata.get("clock_type") if applies and metadata.get("clock_type")
                       else CLOCK_TYPE.get(csv_clock, csv_clock)),
        "launch_date": metadata.get("launch_date") if applies else None,
        "orbit": metadata.get("orbit") if applies else None,
        "severity": severity[last],
        "peak_severity": max(severity, key=SEVERITY_RANK.__getitem__),
        "anomaly_score": float(score[last]),
        "threshold": round(THRESHOLD, 6),
        "first_flagged_index": det["first_flagged_index"],
        "persistence_confirmed_index": confirmed,
        "baseline_flagged_index": base_index,
        "lead_time_days": lead,
        "remaining_life_days": remaining,
        "windows": windows,
    }


def metadata_for(satellite_ids, metadata):
    """Uploaded metadata describes one source satellite.

    A single-satellite batch takes it directly; in a larger batch it applies to
    the satellite whose id equals the supplied name, and to no other.
    """
    supplied = {k: v for k, v in (metadata or {}).items() if k in METADATA_FIELDS and v}
    if not supplied:
        return {}
    if len(satellite_ids) == 1:
        return {satellite_ids[0]: supplied}
    name = supplied.get("name")
    return {name: supplied} if name in satellite_ids else {}


def run(data, cadence, metadata=None, synthetic=None):
    """bytes -> response payload. Raises ingest.IngestError on invalid input."""
    stage = Stages()
    parsed = stage("ingest", ingest.parse, data, cadence)
    raw, features = stage("window", _window, parsed)
    squared, scores = stage("infer", _infer, features)
    detections = stage("detect", _detect, scores)
    base, base_source = stage("baseline", baseline.run, raw, detections)
    shares = stage("explain", _explain, squared)
    remaining = stage("prognosis", _prognosis, parsed, detections)

    ids = list(parsed)
    meta_by_id = metadata_for(ids, metadata)
    satellites = [build_satellite(sid, parsed[sid], scores[sid], detections[sid],
                                  shares[sid], base[sid], remaining[sid],
                                  meta_by_id.get(sid))
                  for sid in ids]
    leads = [s["lead_time_days"] for s in satellites if s["lead_time_days"] is not None]
    triggered = {sid: base[sid][1] for sid in ids}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cadence": cadence,
        "synthetic": synthetic,
        "source": "computed",
        "satellites": satellites,
        "summary": {
            "satellites": len(satellites),
            "flagged": sum(1 for s in satellites if s["persistence_confirmed_index"] is not None),
            "mean_lead_time_days": round(float(np.mean(leads)), 2) if leads else None,
            "lead_time_detail": (None if base_source is None
                                 else lead_time_detail(satellites, triggered, base_source)),
            "clock_comparison": clock_comparison(satellites, scores, detections),
        },
        "stages": stage.timings,
    }
