"""
NavIC Pulse -- thresholding, severity and CUSUM detection.

Two detectors run over the same anomaly scores:

  * A plain threshold, calibrated once on the HEALTHY TRAINING windows
    (mean + K_SIGMA * std, K_SIGMA = 8 -- see config.py for why not 3). Not
    per-satellite rolling statistics computed over every trailing window: such a
    baseline, run on a drifting satellite, adapts to the drift and stops seeing
    it. Measured in navic_recalibrate.py, a 90-day rolling mean + 3*sd catches
    0 of 5 faulted units and drops late-onset recall to 0.009. Excluding the
    windows the detector itself flagged from the trailing statistic repairs that
    completely (late-onset recall back to 1.000), so rolling is workable if it is
    ever wanted here -- it is simply not better than the static bar.
  * A CUSUM, which accumulates small persistent excesses instead of waiting for
    one window to clear the bar. This is what closes the gap on the three
    late-onset faults, whose per-window scores sit close to the healthy
    distribution.

CUSUM (k, h) are calibrated empirically, not set at nominal sigma multiples.
The healthy score distribution is right-skewed, so a nominal k = 0.5*sd,
h = 5*sd puts a clean clock in alarm state ~35% of the time. Instead k is
swept over a small grid and, for each k, h is set to the smallest value holding
the healthy alarm rate at or below MAX_HEALTHY_ALARM_RATE. The winning pair is
the one giving the largest mean detection lead on the late-onset faults.

Severity bands are likewise empirical: Critical sits at a percentile of the
flagged-window score distribution chosen to split flagged windows roughly
60/40 Warning/Critical, rather than at a fixed multiple of the threshold that
put 97% of flagged windows in one band.

Outputs:
  detections.csv   per-window score, severity, CUSUM statistic and alarm flag
"""

import numpy as np
import pandas as pd

from config import CUSUM_PERSISTENCE, K_SIGMA, SATELLITES

INPUT_SCORES = "scores.csv"
OUTPUT_CSV = "detections.csv"

MAX_HEALTHY_ALARM_RATE = 0.01     # <= 1% of healthy windows may be in alarm
K_GRID_SIGMA = (0.5, 1.0, 2.0)    # CUSUM slack candidates, in healthy sigmas
WARNING_SHARE = 0.60              # target share of flagged windows kept Warning

LATE_ONSET = ("slow_drift", "sudden_jump", "noise_increase")


def cusum(scores, mu, k):
    """One-sided CUSUM: S = max(0, S_prev + (score - mu) - k)."""
    out = np.empty(len(scores))
    s = 0.0
    for i, score in enumerate(scores):
        s = max(0.0, s + (score - mu) - k)
        out[i] = s
    return out


def first_day(days, flags):
    hit = np.flatnonzero(flags)
    return int(days[hit[0]]) if len(hit) else None


def sustained(flags, m=CUSUM_PERSISTENCE):
    """True where the trailing `m` windows are all in alarm.

    A single-window alarm is not a detection. At a 1% false-alarm rate an
    isolated alarm lands on a healthy clock early in the record by chance, so
    first-alarm-day rewards trigger-happiness and cannot separate detectors.
    Persistence is the structural difference: healthy units alarm in short
    episodes, faults alarm continuously for hundreds of windows.
    """
    f = np.asarray(flags, dtype=int)
    if len(f) < m:
        return np.zeros(len(f), dtype=bool)
    run = np.convolve(f, np.ones(m, dtype=int), mode="valid") == m
    out = np.zeros(len(f), dtype=bool)
    out[m - 1:] = run
    return out


def cusum_by_satellite(scores, mu, k):
    """{satellite_id: (days, S)} -- CUSUM never carries across satellites."""
    out = {}
    for sat_id, group in scores.groupby("satellite_id", sort=False):
        group = group.sort_values("end_day")
        out[sat_id] = (group["end_day"].to_numpy(),
                       cusum(group["anomaly_score"].to_numpy(), mu, k))
    return out


def calibrate_cusum(scores, mu, sd):
    """Sweep (k, h); return the pair with the best late-onset lead at <=1% FPR.

    S depends only on (mu, k), so it is computed once per k and every h is then
    evaluated by comparison -- no re-integration per h.
    """
    healthy_ids = [s["id"] for s in SATELLITES if s["profile"] == "healthy"]
    onsets = {s["id"]: s["anomaly_start_day"] for s in SATELLITES}
    late_ids = [s["id"] for s in SATELLITES if s["profile"] in LATE_ONSET]

    sweep = []
    for k_sigma in K_GRID_SIGMA:
        k = k_sigma * sd
        stats = cusum_by_satellite(scores, mu, k)
        healthy_s = np.concatenate([stats[i][1] for i in healthy_ids])

        # Smallest h with healthy alarm rate <= the cap is exactly the
        # (1 - cap) quantile of the pooled healthy CUSUM statistic.
        h = float(np.quantile(healthy_s, 1.0 - MAX_HEALTHY_ALARM_RATE))
        healthy_rate = float((healthy_s > h).mean())

        healthy_clean = all(
            not sustained(stats[i][1] > h).any() for i in healthy_ids)

        leads, pre_onset_rates = [], []
        for sat_id in late_ids:
            days, s_values = stats[sat_id]
            day = first_day(days, sustained(s_values > h))
            onset = onsets[sat_id]
            leads.append(None if day is None else onset - day)
            # A pre-onset alarm on a late-onset fault is a false alarm too:
            # those windows are genuinely healthy-looking by construction.
            pre = s_values[days < onset]
            pre_onset_rates.append(float((pre > h).mean()))

        valid = all(l is not None for l in leads) and healthy_clean
        sweep.append({
            "k_sigma": k_sigma, "k": k, "h": h,
            "h_sigma": h / sd,
            "healthy_alarm_rate": healthy_rate,
            "healthy_clean": healthy_clean,
            "pre_onset_alarm_rate": float(np.mean(pre_onset_rates)),
            "leads": leads,
            "mean_lead": float(np.mean(leads)) if valid else float("-inf"),
            "stats": stats,
        })

    best = max(sweep, key=lambda r: r["mean_lead"])
    return best, sweep, late_ids


def main():
    scores = pd.read_csv(INPUT_SCORES)

    # Select the training rows by profile, not by a positional mask: this file
    # gets re-sorted before it is written out, so any index-aligned mask from
    # preprocessed.npz would silently attach to the wrong rows.
    train_scores = scores.loc[scores["profile"] == "healthy",
                              "anomaly_score"].to_numpy()
    mu = float(train_scores.mean())
    sd = float(train_scores.std())
    threshold = mu + K_SIGMA * sd

    scores = scores.sort_values(["satellite_id", "end_day"]).reset_index(drop=True)
    best, sweep, late_ids = calibrate_cusum(scores, mu, sd)

    # -- severity bands, set on the flagged-window distribution -------------
    # Caveat: the 60th-percentile split lands almost exactly on the
    # degraded/non-degraded boundary, because degraded windows score around 80
    # against a Warning bar of 0.28. So in practice these bands say "degraded
    # units are Critical, everything else flagged is Warning" -- a
    # satellite-level distinction wearing a window-level label. It is a real
    # split and a useful one, but it is not grading severity within a
    # satellite's own history.
    flagged = scores.loc[scores["anomaly_score"] > threshold, "anomaly_score"]
    critical = float(np.quantile(flagged, WARNING_SHARE))
    labels = np.full(len(scores), "Normal", dtype=object)
    values = scores["anomaly_score"].to_numpy()
    labels[values > threshold] = "Warning"
    labels[values > critical] = "Critical"
    scores["severity"] = labels

    scores["cusum_s"] = 0.0
    for sat_id, (_, s_values) in best["stats"].items():
        idx = scores.index[scores["satellite_id"] == sat_id]
        scores.loc[idx, "cusum_s"] = s_values
    scores["cusum_alarm"] = scores["cusum_s"] > best["h"]

    # Persistence is applied to BOTH detectors, so the comparison stays
    # matched. Without it the threshold column keeps the same contaminated
    # first-alarm leads that persistence exists to remove.
    scores["threshold_alarm"] = scores["anomaly_score"] > threshold
    scores["cusum_sustained"] = False
    scores["threshold_sustained"] = False
    for sat_id, group in scores.groupby("satellite_id", sort=False):
        idx = group.sort_values("end_day").index
        scores.loc[idx, "cusum_sustained"] = sustained(
            scores.loc[idx, "cusum_alarm"].to_numpy())
        scores.loc[idx, "threshold_sustained"] = sustained(
            scores.loc[idx, "threshold_alarm"].to_numpy())
    scores.to_csv(OUTPUT_CSV, index=False)

    report(scores, mu, sd, threshold, critical, flagged, best, sweep)


def report(scores, mu, sd, threshold, critical, flagged, best, sweep):
    width = 104
    print()
    print("Calibration  (healthy TRAINING windows only)")
    print("=" * width)
    print(f"  mu = {mu:.5f}   sd = {sd:.5f}   threshold = mu + {K_SIGMA:g}*sd = {threshold:.5f}")
    print()

    print("CUSUM (k, h) sweep -- h set to the smallest value holding healthy alarms <= "
          f"{100*MAX_HEALTHY_ALARM_RATE:g}%")
    print("-" * width)
    print(f"  {'k':>8}{'h':>12}{'h/sd':>9}{'healthy alarm':>16}"
          f"{'pre-onset alarm':>18}{'mean lead (late-onset)':>25}")
    print("-" * width)
    for row in sweep:
        mark = "  <-- chosen" if row is best else ""
        lead = "never fires" if row["mean_lead"] == float("-inf") else f"{row['mean_lead']:.0f} d"
        print(f"  {row['k_sigma']:>6.1f}sd{row['h']:>12.4f}{row['h_sigma']:>9.1f}"
              f"{100*row['healthy_alarm_rate']:>15.2f}%{100*row['pre_onset_alarm_rate']:>17.2f}%"
              f"{lead:>25}{mark}")
    print("-" * width)
    print(f"  chosen: k = {best['k_sigma']:g}*sd = {best['k']:.5f}, "
          f"h = {best['h']:.5f} ({best['h_sigma']:.1f}*sd)")
    print()

    print(f"First detection by satellite   ({CUSUM_PERSISTENCE} consecutive alarm "
          f"windows required; lead > 0 == caught before onset)")
    print("=" * width)
    print(f"  {'SATELLITE':<10}{'PROFILE':<15}{'ONSET':>7}"
          f"{'THRESHOLD':>11}{'lead':>7}{'CUSUM':>11}{'lead':>7}{'CUSUM GAIN':>13}")
    print("-" * width)
    for sat in SATELLITES:
        g = scores[scores["satellite_id"] == sat["id"]].sort_values("end_day")
        days = g["end_day"].to_numpy()
        onset = sat["anomaly_start_day"]
        t_day = first_day(days, g["threshold_sustained"].to_numpy())
        c_day = first_day(days, g["cusum_sustained"].to_numpy())

        def fmt(day):
            if day is None:
                return "never", "--"
            if onset is None:
                return f"day {day}", "FALSE"
            return f"day {day}", f"{onset - day:+d}"

        t_txt, t_lead = fmt(t_day)
        c_txt, c_lead = fmt(c_day)
        if t_day is not None and c_day is not None:
            gain = f"{t_day - c_day:+d} d"
        elif c_day is not None:
            gain = "only CUSUM"
        elif t_day is not None:
            gain = "only thresh"
        else:
            gain = "--"
        onset_txt = "--" if onset is None else str(onset)
        print(f"  {sat['id']:<10}{sat['profile']:<15}{onset_txt:>7}"
              f"{t_txt:>11}{t_lead:>7}{c_txt:>11}{c_lead:>7}{gain:>13}")
    print("-" * width)

    healthy = scores[scores["profile"] == "healthy"]
    n_thr = int(healthy["threshold_sustained"].sum())
    n_cus = int(healthy["cusum_sustained"].sum())
    print(f"  healthy units, single-window alarms:  threshold "
          f"{100*healthy['threshold_alarm'].mean():.2f}%   CUSUM {100*healthy['cusum_alarm'].mean():.2f}%")
    print(f"  healthy units, sustained detections:  threshold {n_thr}   CUSUM {n_cus}"
          f"   {'-- both clean' if n_thr == n_cus == 0 else '-- NOT clean'}")
    print()

    counts = scores["severity"].value_counts()
    n_flagged = len(flagged)
    warn, crit = int(counts.get("Warning", 0)), int(counts.get("Critical", 0))
    print("Severity bands")
    print("-" * width)
    print(f"  Warning  above {threshold:.5f}   {warn:>7,} windows   {100*warn/n_flagged:>5.1f}% of flagged")
    print(f"  Critical above {critical:.5f}   {crit:>7,} windows   {100*crit/n_flagged:>5.1f}% of flagged")
    print(f"  Normal                        {int(counts.get('Normal', 0)):>7,} windows")
    print("-" * width)
    print(f"  wrote {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
