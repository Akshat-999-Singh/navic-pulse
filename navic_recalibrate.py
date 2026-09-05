"""Do the two ESA-ADB calibration lessons change anything on NavIC?

Lesson (a) -- if a rolling statistic is used anywhere, the trailing window must
exclude windows the detector itself flagged. On ESA-ADB the contaminated variant
let the bar climb 5056x inside a 14-day event and recall fell 1.000 -> 0.000.
detect.py already refuses per-satellite rolling baselines and says why, so the
question here is whether that refusal is still the right call, and what it costs
or saves. Both variants are built and measured rather than argued.

Lesson (b) -- on ESA-ADB the winning rule replaced the 3*sigma spread term with a
fixed absolute offset, because sigma estimated on a heavy-tailed score is the
fragile half of mean + 3*sigma. NavIC's bar is static, so there mu + 3*sd IS a
fixed offset and the lesson cannot apply as stated. What can be asked is whether
sigma is the right scale at all: a swept multiplier and a healthy-quantile bar
are compared against the shipped 3*sigma, and the ESA winner's actual form
(rolling mean + fixed offset) is run alongside.

Calibration protocol matches detect.py: mu and sd come from every window of the
five healthy-profile satellites, and everything is evaluated on all ten.
"""

import json

import numpy as np
import pandas as pd

from config import CUSUM_PERSISTENCE, K_SIGMA, SATELLITES

INPUT_SCORES = "scores.csv"
OUTPUT_JSON = "navic_recalibration.json"

TRAIL_DAYS = (90, 180)
LATE_ONSET = ("slow_drift", "sudden_jump", "noise_increase")
MIN_TRAIL = 30


def fbeta(p, r, b):
    return (1 + b * b) * p * r / (b * b * p + r) if (p and r) else 0.0


def sustained(flags, m=CUSUM_PERSISTENCE):
    f = np.asarray(flags, dtype=int)
    if f.size < m:
        return np.zeros(f.size, dtype=bool)
    out = np.zeros(f.size, dtype=bool)
    out[m - 1:] = np.convolve(f, np.ones(m, dtype=int), mode="valid") == m
    return out


def evaluate(df, pred):
    """Window-level scores plus the satellite-level verdict detect.py reports."""
    y = df["is_true_anomaly"].to_numpy().astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    onsets = {s["id"]: s["anomaly_start_day"] for s in SATELLITES}
    profiles = {s["id"]: s["profile"] for s in SATELLITES}
    caught, false_alarm, per_sat, late_rec = 0, 0, {}, []
    n_faulted = sum(1 for s in SATELLITES if s["profile"] != "healthy")
    for sat_id, g in df.assign(_p=pred).groupby("satellite_id", sort=False):
        g = g.sort_values("end_day")
        sus = sustained(g["_p"].to_numpy())
        hit = np.flatnonzero(sus)
        day = int(g["end_day"].to_numpy()[hit[0]]) if hit.size else None
        prof = profiles[sat_id]
        gy = g["is_true_anomaly"].to_numpy().astype(bool)
        r = float(g["_p"].to_numpy()[gy].mean()) if gy.any() else None
        per_sat[sat_id] = {"profile": prof, "sustained_day": day,
                           "lead": None if day is None or prof == "healthy"
                           else onsets[sat_id] - day, "window_recall": r}
        if prof == "healthy":
            false_alarm += day is not None
        else:
            caught += day is not None
            if prof in LATE_ONSET:
                late_rec.append(r)
    return {"precision": prec, "recall": rec, "f1": fbeta(prec, rec, 1.0),
            "f05": fbeta(prec, rec, 0.5), "TP": tp, "FP": fp, "FN": fn,
            "faulted_caught": caught, "faulted_total": n_faulted,
            "healthy_false_alarms": false_alarm,
            "late_onset_recall": float(np.mean(late_rec)) if late_rec else None,
            "per_satellite": per_sat}


def rolling_pred(df, window, mode, offset=None, k=K_SIGMA, fallback=None):
    """Per-satellite causal trailing bar over the previous `window` days.

    mode 'all' uses every trailing score; 'unflag' uses only the trailing scores
    this detector did not flag, which is the ESA-ADB fix. offset=None gives
    mean + k*sd; a numeric offset gives mean + offset.
    """
    pred = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("satellite_id", sort=False):
        idx = g.sort_values("end_day").index.to_numpy()
        v = df.loc[idx, "anomaly_score"].to_numpy()
        d = df.loc[idx, "end_day"].to_numpy()
        lo = np.searchsorted(d, d - window, side="left")   # causal: [lo, i)
        n = v.size
        cn = np.zeros(n + 1); cs = np.zeros(n + 1); cq = np.zeros(n + 1)
        p = np.zeros(n, dtype=int)
        for i in range(n):
            if i > 0:                       # prefix sums cover v[0..i-1] only
                w = 1.0 if mode == "all" else (0.0 if p[i - 1] else 1.0)
                cn[i] = cn[i - 1] + w
                cs[i] = cs[i - 1] + w * v[i - 1]
                cq[i] = cq[i - 1] + w * v[i - 1] * v[i - 1]
            a, b = lo[i], i
            cnt = cn[b] - cn[a] if b > a else 0.0
            if cnt >= MIN_TRAIL:
                mu = (cs[b] - cs[a]) / cnt
                sd = np.sqrt(max((cq[b] - cq[a]) / cnt - mu * mu, 0.0))
                thr = mu + (k * sd if offset is None else offset)
            else:
                thr = fallback
            p[i] = v[i] > thr
        pred[idx] = p
    return pred


def main():
    df = pd.read_csv(INPUT_SCORES).sort_values(
        ["satellite_id", "end_day"]).reset_index(drop=True)
    healthy = df.loc[df["profile"] == "healthy", "anomaly_score"].to_numpy()
    mu, sd = float(healthy.mean()), float(healthy.std())
    static_thr = mu + K_SIGMA * sd
    v = df["anomaly_score"].to_numpy()
    rows, res = [], {}

    def add(name, pred, note=""):
        m = evaluate(df, pred)
        rows.append({"name": name, "note": note, **m})
        return m

    print("=" * 118)
    print("NavIC -- applying the two ESA-ADB calibration lessons")
    print("=" * 118)
    print(f"  healthy calibration: mu {mu:.5f}  sd {sd:.5f}  "
          f"mu+{K_SIGMA:g}sd = {static_thr:.5f}")
    print(f"  healthy score quantiles: p95 {np.quantile(healthy,.95):.4f}  "
          f"p99 {np.quantile(healthy,.99):.4f}  max {healthy.max():.4f}")
    print(f"  windows {len(df):,}   positive {int(df['is_true_anomaly'].sum()):,}")

    add(f"static  mu + {K_SIGMA:g}sd   (shipped)", (v > static_thr).astype(int))

    # ---- lesson (b): is sigma the right scale? ------------------------------
    ksweep = []
    for kk in (1, 2, 3, 4, 5, 8, 12, 20):
        m = evaluate(df, (v > mu + kk * sd).astype(int))
        ksweep.append({"k": kk, "thr": mu + kk * sd, **{q: m[q] for q in
                       ("precision", "recall", "f05", "healthy_false_alarms",
                        "faulted_caught", "late_onset_recall")}})
    for q in (0.99, 0.999, 1.0):
        thr = float(np.quantile(healthy, q))
        add(f"static  healthy p{100*q:g} bar = {thr:.4f}", (v > thr).astype(int))

    # ---- lesson (a): rolling, contaminated vs self-flag-excluded ------------
    for W in TRAIL_DAYS:
        for mode in ("all", "unflag"):
            # k is pinned at 3 here, not K_SIGMA: these rows exist to reproduce
            # the historical rolling-baseline failure, so they must not drift
            # when the shipped multiplier changes.
            add(f"rolling {W:>3d}d mean+3sd [{mode}]",
                rolling_pred(df, W, mode, k=3.0, fallback=static_thr))
    # ESA winner's form: rolling mean + fixed absolute offset
    best_off, best_f = None, -1
    offs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    offsweep = []
    for c in offs:
        m = evaluate(df, rolling_pred(df, 90, "unflag", offset=c, fallback=static_thr))
        offsweep.append({"offset": c, **{q: m[q] for q in
                         ("precision", "recall", "f05", "healthy_false_alarms",
                          "late_onset_recall")}})
        if m["f05"] > best_f:
            best_f, best_off = m["f05"], c
    add(f"rolling  90d mean + {best_off:g} offset [unflag]",
        rolling_pred(df, 90, "unflag", offset=best_off, fallback=static_thr))

    # ---------------------------------------------------------------- report --
    W = 118
    print("\n" + "=" * W)
    print("DETECTOR COMPARISON  (window level; satellite level uses persistence "
          f"M={CUSUM_PERSISTENCE})")
    print("=" * W)
    print(f"  {'detector':<42}{'prec':>7}{'rec':>7}{'F1':>7}{'F0.5':>8}"
          f"{'late-onset rec':>16}{'caught':>8}{'false alarm sats':>18}")
    print("  " + "-" * (W - 2))
    for r in rows:
        lr = f"{r['late_onset_recall']:.3f}" if r["late_onset_recall"] is not None else "-"
        print(f"  {r['name']:<42}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r['f05']:>8.3f}{lr:>16}{r['faulted_caught']:>5}/{r['faulted_total']:<2}"
              f"{r['healthy_false_alarms']:>13}/5")

    print("\n" + "=" * W)
    print("LESSON (b)  sigma multiplier sweep -- is 3*sd the right scale on NavIC?")
    print("=" * W)
    print(f"  {'k':>4}{'threshold':>12}{'prec':>9}{'rec':>8}{'F0.5':>8}"
          f"{'late-onset rec':>16}{'caught':>9}{'false alarm sats':>18}")
    for r in ksweep:
        lr = f"{r['late_onset_recall']:.3f}" if r["late_onset_recall"] is not None else "-"
        mark = "  <-- shipped" if r["k"] == K_SIGMA else ""
        print(f"  {r['k']:>4}{r['thr']:>12.5f}{r['precision']:>9.3f}{r['recall']:>8.3f}"
              f"{r['f05']:>8.3f}{lr:>16}{r['faulted_caught']:>6}/5{r['healthy_false_alarms']:>14}/5{mark}")

    print("\n" + "=" * W)
    print("LESSON (b)  rolling mean + FIXED OFFSET (the ESA-ADB winner's form), 90 d [unflag]")
    print("=" * W)
    print(f"  {'offset':>8}{'prec':>9}{'rec':>8}{'F0.5':>8}{'late-onset rec':>16}"
          f"{'false alarm sats':>18}")
    for r in offsweep:
        lr = f"{r['late_onset_recall']:.3f}" if r["late_onset_recall"] is not None else "-"
        print(f"  {r['offset']:>8.2f}{r['precision']:>9.3f}{r['recall']:>8.3f}"
              f"{r['f05']:>8.3f}{lr:>16}{r['healthy_false_alarms']:>14}/5")

    print("\n" + "=" * W)
    print("LESSON (a)  per-satellite recall on the three late-onset faults")
    print("=" * W)
    keep = [r for r in rows if "rolling" in r["name"] or "shipped" in r["name"]]
    late = [s["id"] for s in SATELLITES if s["profile"] in LATE_ONSET]
    print(f"  {'detector':<42}" + "".join(f"{i:>16}" for i in late))
    print("  " + "-" * (W - 2))
    for r in keep:
        cells = []
        for i in late:
            ps = r["per_satellite"][i]
            cells.append(f"{ps['window_recall']:.3f}" if ps["window_recall"] is not None else "-")
        print(f"  {r['name']:<42}" + "".join(f"{c:>16}" for c in cells))
    print("\n  degraded units (IRNSS-1D, IRNSS-1G) score ~74 against a 0.28 bar and are")
    print("  caught by everything; the late-onset three are where a bar's quality shows.")
    print("=" * W)

    # ---- why recall never moves ---------------------------------------------
    # Every detector above lands on recall ~0.86. That ceiling is not the detector:
    # the two 'degraded' units carry is_true_anomaly=True from day 29 while their own
    # anomaly_start_day is 400 and 900, so ~1.2k windows are labelled anomalous over a
    # stretch whose scores are indistinguishable from the healthy fleet.
    print("\n" + "=" * W)
    print("RECALL CEILING -- the labelling, not the detector")
    print("=" * W)
    onsets = {s["id"]: s["anomaly_start_day"] for s in SATELLITES}
    profiles = {s["id"]: s["profile"] for s in SATELLITES}
    pre = np.array([profiles[s] == "degraded" and d < onsets[s]
                    for s, d in zip(df["satellite_id"], df["end_day"])])
    y = df["is_true_anomaly"].to_numpy().astype(bool)
    print(f"  windows labelled anomalous before their own onset day: {int(pre.sum()):,}")
    print(f"  their median score {np.median(v[pre]):.4f}  vs healthy fleet "
          f"{np.median(healthy):.4f}  (degraded post-onset: {np.median(v[y & ~pre]):.1f})")
    corr = []
    for name, thr in ((f"shipped mu+{K_SIGMA:g}sd", static_thr),
                      ("previous mu+3sd", mu + 3 * sd)):
        p = v > thr
        fn_pre = int((y & ~p & pre).sum())
        yc = y & ~pre                                  # pre-onset -> negative
        tp = int((p & yc).sum()); fp = int((p & ~yc).sum()); fn = int((~p & yc).sum())
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        corr.append({"bar": name, "threshold": float(thr),
                     "fn_that_are_pre_onset": fn_pre,
                     "corrected_precision": pr, "corrected_recall": rc,
                     "corrected_f05": fbeta(pr, rc, 0.5)})
        print(f"\n  {name}: of {int((y & ~p).sum()):,} false negatives, "
              f"{fn_pre:,} are pre-onset degraded windows "
              f"({100*fn_pre/max(int((y & ~p).sum()),1):.0f}%)")
        print(f"    relabelling those as negative: precision {pr:.3f}  recall {rc:.3f}  "
              f"F0.5 {fbeta(pr, rc, 0.5):.3f}")
    print("=" * W)

    res = {"calibration": {"mu": mu, "sd": sd, "static_threshold": static_thr},
           "rows": rows, "sigma_sweep": ksweep, "offset_sweep": offsweep,
           "recall_ceiling": {"n_pre_onset_labelled_positive": int(pre.sum()),
                              "corrected": corr}}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
