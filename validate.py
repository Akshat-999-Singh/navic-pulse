"""
NavIC Pulse -- validation.

Scores the detection layer against the known labels and collects everything the
project produces into one summary.json:

  * window-level precision / recall / F1 on SUSTAINED detections, for both the
    plain threshold and CUSUM
  * satellite-level detection (did each unit get caught at all, and was any
    healthy unit ever flagged)
  * per-satellite lead time: anomaly_start_day - first sustained detection day
  * the RUL projection from rul.json

Window-level recall is structurally capped: a sustained detection needs
CUSUM_PERSISTENCE consecutive alarm windows, so the windows spanning that
build-up are counted as misses even when the fault is eventually caught. That
is detection latency showing up as recall, not a scoring error.

Outputs:
  summary.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from config import CUSUM_PERSISTENCE, SATELLITES

INPUT_DETECTIONS = "detections.csv"
INPUT_RUL = "rul.json"
OUTPUT_JSON = "summary.json"

DETECTORS = {"cusum": "cusum_sustained", "threshold": "threshold_sustained"}


def window_metrics(truth, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def satellite_metrics(detections, column):
    """Did each satellite get caught, and were any healthy units flagged?"""
    caught, false_alarms, leads = [], [], {}
    for sat in SATELLITES:
        g = detections[detections["satellite_id"] == sat["id"]]
        hits = g.loc[g[column], "end_day"]
        first = int(hits.min()) if len(hits) else None
        if sat["profile"] == "healthy":
            if first is not None:
                false_alarms.append(sat["id"])
        else:
            if first is not None:
                caught.append(sat["id"])
        leads[sat["id"]] = {
            "profile": sat["profile"],
            "anomaly_start_day": sat["anomaly_start_day"],
            "first_sustained_detection_day": first,
            "lead_days": (None if first is None or sat["anomaly_start_day"] is None
                          else int(sat["anomaly_start_day"] - first)),
        }
    faulted = [s["id"] for s in SATELLITES if s["profile"] != "healthy"]
    healthy = [s["id"] for s in SATELLITES if s["profile"] == "healthy"]
    return {
        "faulted_caught": len(caught),
        "faulted_total": len(faulted),
        "healthy_false_alarms": len(false_alarms),
        "healthy_total": len(healthy),
        "false_alarm_satellites": false_alarms,
        "per_satellite": leads,
    }


def main():
    detections = pd.read_csv(INPUT_DETECTIONS)
    with open(INPUT_RUL, encoding="utf-8") as handle:
        rul = json.load(handle)

    truth = detections["is_true_anomaly"].to_numpy()
    results = {}
    for name, column in DETECTORS.items():
        results[name] = {
            "window_level": window_metrics(truth, detections[column].to_numpy()),
            "satellite_level": satellite_metrics(detections, column),
        }

    # The raw score is a ranking, independent of any threshold choice.
    auc = float(roc_auc_score(truth, detections["anomaly_score"].to_numpy()))

    summary = {
        "dataset": {
            "windows": int(len(detections)),
            "anomalous_windows": int(truth.sum()),
            "healthy_windows": int((~truth).sum()),
            "satellites": len(SATELLITES),
        },
        "anomaly_score_roc_auc": auc,
        "cusum_persistence_windows": CUSUM_PERSISTENCE,
        "detectors": results,
        "remaining_useful_life": rul,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report(summary)


def report(summary):
    width = 96
    print()
    print("Validation")
    print("=" * width)
    d = summary["dataset"]
    print(f"  {d['windows']:,} windows over {d['satellites']} satellites"
          f"   ({d['anomalous_windows']:,} anomalous / {d['healthy_windows']:,} healthy)")
    print(f"  anomaly score ROC AUC (threshold-free): {summary['anomaly_score_roc_auc']:.4f}")
    print(f"  sustained detection requires {summary['cusum_persistence_windows']} "
          f"consecutive alarm windows")
    print()

    print("Window-level metrics on sustained detections")
    print("-" * width)
    print(f"  {'DETECTOR':<12}{'PRECISION':>11}{'RECALL':>9}{'F1':>9}"
          f"{'TP':>9}{'FP':>8}{'FN':>9}{'TN':>10}")
    print("-" * width)
    for name in DETECTORS:
        m = summary["detectors"][name]["window_level"]
        print(f"  {name:<12}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}"
              f"{m['true_positives']:>9,}{m['false_positives']:>8,}"
              f"{m['false_negatives']:>9,}{m['true_negatives']:>10,}")
    print("-" * width)
    print()

    print("Satellite-level detection")
    print("-" * width)
    for name in DETECTORS:
        s = summary["detectors"][name]["satellite_level"]
        print(f"  {name:<12}faults caught {s['faulted_caught']}/{s['faulted_total']}"
              f"   healthy false alarms {s['healthy_false_alarms']}/{s['healthy_total']}")
    print()

    print("Lead time (CUSUM sustained; negative = detected after onset)")
    print("-" * width)
    print(f"  {'SATELLITE':<11}{'PROFILE':<16}{'ONSET':>7}{'DETECTED':>11}{'LEAD':>9}{'RUL':>22}")
    print("-" * width)
    rul_by_id = {r["satellite_id"]: r for r in summary["remaining_useful_life"]["satellites"]}
    per_sat = summary["detectors"]["cusum"]["satellite_level"]["per_satellite"]
    for sat in SATELLITES:
        info = per_sat[sat["id"]]
        onset = "--" if info["anomaly_start_day"] is None else str(info["anomaly_start_day"])
        first = info["first_sustained_detection_day"]
        first_txt = "never" if first is None else f"day {first}"
        lead = info["lead_days"]
        lead_txt = "--" if lead is None else f"{lead:+d} d"
        r = rul_by_id[sat["id"]]
        if r["remaining_useful_life_days"] is not None:
            se = r["remaining_useful_life_se_days"]
            rul_txt = f"{r['remaining_useful_life_days']:,} +/- {se:,.0f} d"
        else:
            rul_txt = r["status"]
        print(f"  {sat['id']:<11}{sat['profile']:<16}{onset:>7}{first_txt:>11}"
              f"{lead_txt:>9}{rul_txt:>22}")
    print("-" * width)
    print(f"  wrote {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
