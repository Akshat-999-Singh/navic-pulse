"""
NavIC Pulse -- fixed-limit Out-Of-Limits (OOL) baseline on raw telemetry.

Operator ground segments watch raw housekeeping channels against fixed limits,
not a model's score. This is that monitor, so the autoencoder + CUSUM detector
has something to be "earlier" than.

Limits are derived from observed nominal behaviour: per channel, mean +/-
K_SIGMA_OOL * sd over the HEALTHY satellites' full records. SPEC_LIMIT_DRIFT is
deliberately not used -- freq_offset_y is a steered ground-segment residual and
healthy units already sit above the 5e-13 free-running figure (see rul.py).

Matched comparison with detect.py:
  * a day is out of limits if ANY channel is outside its band
  * window index = end_day - (WINDOW - 1), and a window is in alarm when its end
    day is out of limits -- the state an operator sees on that day
  * sustained detection uses detect.sustained, the same CUSUM_PERSISTENCE rule
    that produces cusum_sustained

Lead time = baseline sustained index - CUSUM sustained index. Positive means
CUSUM confirmed first; negative means the raw-channel baseline beat it.

Outputs:
  baseline_ool.json
"""

import json

import numpy as np
import pandas as pd

from config import CHANNELS, CUSUM_PERSISTENCE, SATELLITES, WINDOW
from detect import sustained

INPUT_TELEMETRY = "telemetry.csv"
INPUT_DETECTIONS = "detections.csv"
OUTPUT_JSON = "baseline_ool.json"

K_SIGMA_OOL = 3.0


def first_index(flags):
    hit = np.flatnonzero(flags)
    return int(hit[0]) if len(hit) else None


def ool_limits(telemetry):
    """Per-channel (mean, sd) over the healthy fleet's full span."""
    healthy_ids = [s["id"] for s in SATELLITES if s["profile"] == "healthy"]
    healthy = telemetry[telemetry["satellite_id"].isin(healthy_ids)]
    limits = {}
    for channel in CHANNELS:
        values = healthy[channel].to_numpy(dtype=float)
        # Population sd, matching the healthy-score calibration in detect.py.
        mean, sd = float(values.mean()), float(values.std())
        limits[channel] = {"mean": mean, "sd": sd,
                           "low": mean - K_SIGMA_OOL * sd,
                           "high": mean + K_SIGMA_OOL * sd}
    return healthy_ids, limits


def evaluate(frame, limits):
    """Window-indexed alarm, sustained flags and per-channel out-of-limit days."""
    out = {c: ((frame[c] < limits[c]["low"]) | (frame[c] > limits[c]["high"])).to_numpy()
           for c in CHANNELS}
    day_flag = np.logical_or.reduce([out[c] for c in CHANNELS])
    # Window i ends on day i + WINDOW - 1; the first WINDOW - 1 days close no window.
    alarm = day_flag[WINDOW - 1:]
    return alarm, sustained(alarm), {c: out[c][WINDOW - 1:] for c in CHANNELS}


def main():
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    detections = pd.read_csv(INPUT_DETECTIONS)
    healthy_ids, limits = ool_limits(telemetry)

    results, rows = [], []
    for sat in SATELLITES:
        frame = telemetry[telemetry["satellite_id"] == sat["id"]].sort_values("day")
        if not (frame["day"].to_numpy() == np.arange(len(frame))).all():
            raise SystemExit(f"{sat['id']}: telemetry days are not contiguous from 0")
        alarm, held, out = evaluate(frame, limits)

        flagged = first_index(alarm)
        confirmed = first_index(held)
        # Channels out of limits on any day of the run that confirmed the
        # detection -- what the operator would have been looking at.
        if confirmed is None:
            triggered = []
        else:
            run = slice(confirmed - CUSUM_PERSISTENCE + 1, confirmed + 1)
            triggered = [c for c in CHANNELS if out[c][run].any()]

        g = detections[detections["satellite_id"] == sat["id"]].sort_values("end_day")
        if not (g["end_day"].to_numpy() - (WINDOW - 1) == np.arange(len(alarm))).all():
            raise SystemExit(f"{sat['id']}: detections.csv windows do not align with telemetry")
        cusum = first_index(g["cusum_sustained"].to_numpy(dtype=bool))

        results.append({
            "satellite_id": sat["id"],
            "baseline_flagged_index": flagged,
            "baseline_sustained_index": confirmed,
            "channels_triggered": triggered,
        })
        rows.append((sat, confirmed, cusum, triggered,
                     {c: int(out[c].sum()) for c in CHANNELS}))

    payload = {
        "method": "raw-channel fixed limits, healthy-fleet mean +/- k*sd, any channel",
        "k_sigma": K_SIGMA_OOL,
        "persistence_windows": CUSUM_PERSISTENCE,
        "healthy_satellites": healthy_ids,
        "limits": limits,
        "satellites": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    report(limits, rows)


def report(limits, rows):
    width = 104
    print()
    print(f"OOL limits   (healthy fleet, mean +/- {K_SIGMA_OOL:g} sd)")
    print("=" * width)
    print(f"  {'CHANNEL':<24}{'MEAN':>14}{'SD':>14}{'LOW':>14}{'HIGH':>14}")
    print("-" * width)
    for c, lim in limits.items():
        print(f"  {c:<24}{lim['mean']:>14.4g}{lim['sd']:>14.4g}{lim['low']:>14.4g}{lim['high']:>14.4g}")
    print()

    print(f"Sustained detection, window index ({CUSUM_PERSISTENCE} consecutive alarm "
          f"windows; lead = OOL - CUSUM, + means CUSUM first)")
    print("=" * width)
    print(f"  {'SATELLITE':<11}{'PROFILE':<16}{'ONSET':>6}{'OOL':>7}{'CUSUM':>7}"
          f"{'LEAD':>9}   CHANNELS (confirming run)")
    print("-" * width)
    leads = []
    for sat, ool, cusum, triggered, _ in rows:
        onset = sat["anomaly_start_day"]
        # Onset is a day; express it as the index of the first window ending on it.
        onset_txt = "--" if onset is None else str(onset - (WINDOW - 1))
        if ool is not None and cusum is not None:
            lead = ool - cusum
            leads.append(lead)
            lead_txt = f"{lead:+d} d"
        elif cusum is not None:
            lead_txt = "OOL never"
        elif ool is not None:
            lead_txt = "OOL only"
        else:
            lead_txt = "--"
        dash = lambda v: "never" if v is None else str(v)
        print(f"  {sat['id']:<11}{sat['profile']:<16}{onset_txt:>6}{dash(ool):>7}"
              f"{dash(cusum):>7}{lead_txt:>9}   {', '.join(triggered) or '--'}")
    print("-" * width)
    if leads:
        print(f"  mean lead over {len(leads)} satellites both detectors confirm: "
              f"{np.mean(leads):+.1f} d")
    false_alarms = [s["id"] for s, ool, _, _, _ in rows
                    if s["profile"] == "healthy" and ool is not None]
    print(f"  healthy satellites with a sustained OOL detection: "
          f"{len(false_alarms)}/5 {false_alarms if false_alarms else ''}")
    print()
    print("Out-of-limit days per channel (whole record)")
    print("-" * width)
    print(f"  {'SATELLITE':<11}" + "".join(f"{c[:20]:>21}" for c in CHANNELS))
    for sat, _, _, _, counts in rows:
        print(f"  {sat['id']:<11}" + "".join(f"{counts[c]:>21}" for c in CHANNELS))
    print("-" * width)
    print(f"  wrote {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
