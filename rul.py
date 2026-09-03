"""
NavIC Pulse -- prognosis layer (remaining useful life).

Degradation indicator: rafs_signal_level.

Why not the drift rate of freq_offset_y. Two things ruled it out, in order:

  * SPEC_LIMIT_DRIFT (5e-13/day) is a free-running iRAFS specification, while
    freq_offset_y here is a steered ground-segment residual -- the ground
    segment estimates and removes modelled drift, leaving a small unmodelled
    part. The two are not commensurate. Measured on this data every 90-day
    block sits at 0.1-1.2% of the spec figure, and healthy median
    |freq_offset_y| is 1.81e-12, already above the 5e-13 figure outright, so
    comparing either the value or its rate to that number is meaningless here.
  * Re-basing the limit on the healthy fleet fixed the target but not the
    indicator: 90-day block slopes are dominated by the random walk in
    freq_offset_y (healthy blocks scatter to 1.25e-14 against a largest
    fault-driven rate of ~8e-14), so no reliable upward trend can be fitted
    from the handful of blocks available before detection.

rafs_signal_level is the indicator this projection wants: it declines
monotonically, it leads the frequency fault by ~60 days, and it is a wear curve
rather than a noise-dominated rate.

Method, per satellite:
  1. Reference day = first sustained detection + PROGNOSIS_LAG_DAYS. The
     no-hindsight rule still holds -- nothing after that day is used. The 30-day
     lag is admissible because signal level leads frequency by ~60 days and is
     therefore already in decline at detection.
  2. Fit an OLS line to rafs_signal_level over the TREND_WINDOW_DAYS-day
     rolling window ending on the reference day.
  3. Extrapolate to SIGNAL_FAILURE_LIMIT and report the crossing day, with a
     standard error propagated through the ratio by the delta method.
  4. remaining_useful_life_days = crossing day - reference day, capped at
     HORIZON_DAYS.

Healthy units are never detected, so their reference day is the last day of the
record: their RUL is measured from the end of available history.

Outputs:
  rul.json   per-satellite days remaining, standard error and supporting figures
"""

import json

import numpy as np
import pandas as pd

from config import SATELLITES

INPUT_TELEMETRY = "telemetry.csv"
INPUT_DETECTIONS = "detections.csv"
OUTPUT_JSON = "rul.json"

TREND_WINDOW_DAYS = 90
PROGNOSIS_LAG_DAYS = 30
HORIZON_DAYS = 3650

# Set below the ~0.91 post-fault plateau, so crossing it represents further
# degradation rather than the state the satellite is already in.
SIGNAL_FAILURE_LIMIT = 0.85

# Secondary indicator. steering_correction_ns has no external limit, so its
# failure level is set from the healthy fleet the same way the drift limit was:
# a multiple of the healthy population's 95th-percentile rolling mean.
SECONDARY_CHANNEL = "steering_correction_ns"
SECONDARY_MULTIPLE = 10.0
SECONDARY_PERCENTILE = 95

STATUS_PAST_LIMIT = "already past limit"
STATUS_BEYOND_HORIZON = "beyond horizon"
STATUS_NO_TREND = "no declining trend"
STATUS_TOO_FEW = "too few points"


def ols_crossing(x, y, limit, declining=True):
    """OLS fit of y on x, plus the crossing of `limit` and its standard error.

    The crossing is a ratio of correlated estimates, so its variance comes from
    the delta method on x* = (limit - b0) / b1 rather than from the slope alone.
    """
    n = len(x)
    if n < 3:
        return None
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0:
        return None
    b1 = float(((x - x_mean) * (y - y_mean)).sum() / sxx)
    b0 = float(y_mean - b1 * x_mean)

    residuals = y - (b0 + b1 * x)
    s2 = float((residuals ** 2).sum() / (n - 2))
    var_b1 = s2 / sxx
    var_b0 = s2 * (1.0 / n + x_mean ** 2 / sxx)
    cov_b01 = -s2 * x_mean / sxx

    # A declining indicator needs a negative slope to reach a lower limit; a
    # rising one needs a positive slope.
    if (declining and b1 >= 0) or (not declining and b1 <= 0):
        return {"slope": b1, "intercept": b0, "crossing": None, "se": None}

    crossing = (limit - b0) / b1
    var_x = (var_b0 + crossing ** 2 * var_b1 + 2 * crossing * cov_b01) / (b1 ** 2)
    se = float(np.sqrt(var_x)) if var_x > 0 else None
    return {"slope": b1, "intercept": b0, "crossing": float(crossing), "se": se}


def first_detection_day(detections, sat_id):
    g = detections[(detections["satellite_id"] == sat_id)
                   & detections["cusum_sustained"]]
    return int(g["end_day"].min()) if len(g) else None


def secondary_limit(telemetry):
    """Failure level for the secondary indicator, set from the healthy fleet."""
    values = []
    for sat in SATELLITES:
        if sat["profile"] != "healthy":
            continue
        frame = telemetry[telemetry["satellite_id"] == sat["id"]].sort_values("day")
        rolling = (frame[SECONDARY_CHANNEL].abs()
                   .rolling(TREND_WINDOW_DAYS).mean().dropna())
        values.append(rolling.to_numpy())
    pooled = np.concatenate(values)
    baseline = float(np.percentile(pooled, SECONDARY_PERCENTILE))
    return SECONDARY_MULTIPLE * baseline, baseline


def evaluate(fit, reference_day):
    """Turn a fit into (status, rul_days, se_days)."""
    if fit is None:
        return STATUS_TOO_FEW, None, None
    if fit["crossing"] is None:
        return STATUS_NO_TREND, None, None
    days = fit["crossing"] - reference_day
    if days < 0:
        return STATUS_PAST_LIMIT, None, None
    if days > HORIZON_DAYS:
        return STATUS_BEYOND_HORIZON, None, None
    return "days remaining", int(round(days)), fit["se"]


def main():
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    detections = pd.read_csv(INPUT_DETECTIONS)
    last_day = int(telemetry["day"].max())
    sec_limit, sec_baseline = secondary_limit(telemetry)

    results = []
    for sat in SATELLITES:
        frame = telemetry[telemetry["satellite_id"] == sat["id"]].sort_values("day")
        detected = first_detection_day(detections, sat["id"])
        reference_day = (min(detected + PROGNOSIS_LAG_DAYS, last_day)
                         if detected is not None else last_day)
        lo = reference_day - TREND_WINDOW_DAYS + 1

        # -- primary: rafs_signal_level ---------------------------------
        window = frame[(frame["day"] >= lo) & (frame["day"] <= reference_day)]
        x = window["day"].to_numpy(dtype=float)
        y = window["rafs_signal_level"].to_numpy(dtype=float)
        level_now = float(y.mean()) if len(y) else float("nan")

        fit = None
        if level_now <= SIGNAL_FAILURE_LIMIT:
            status, rul, se = STATUS_PAST_LIMIT, None, None
        else:
            fit = ols_crossing(x, y, SIGNAL_FAILURE_LIMIT, declining=True)
            status, rul, se = evaluate(fit, reference_day)

        # -- secondary: steering_correction_ns --------------------------
        days_all = frame["day"].to_numpy(dtype=float)
        rolled = (frame[SECONDARY_CHANNEL].abs()
                  .rolling(TREND_WINDOW_DAYS).mean().to_numpy())
        keep = (days_all >= lo) & (days_all <= reference_day) & ~np.isnan(rolled)
        sec_now = float(rolled[keep].mean()) if keep.any() else float("nan")
        if keep.sum() < 3:
            sec_status, sec_rul, sec_se = STATUS_TOO_FEW, None, None
        elif sec_now >= sec_limit:
            sec_status, sec_rul, sec_se = STATUS_PAST_LIMIT, None, None
        else:
            sec_fit = ols_crossing(days_all[keep], rolled[keep], sec_limit,
                                   declining=False)
            sec_status, sec_rul, sec_se = evaluate(sec_fit, reference_day)

        results.append({
            "satellite_id": sat["id"],
            "clock_type": sat["clock_type"],
            "profile": sat["profile"],
            "anomaly_start_day": sat["anomaly_start_day"],
            "detection_day": detected,
            "reference_day": reference_day,
            "signal_level_at_reference": level_now,
            "signal_slope_per_day": None if fit is None else fit["slope"],
            "status": status,
            "remaining_useful_life_days": rul,
            "remaining_useful_life_se_days": None if se is None else round(float(se), 1),
            "secondary_value_at_reference": sec_now,
            "secondary_status": sec_status,
            "secondary_rul_days": sec_rul,
            "secondary_rul_se_days": None if sec_se is None else round(float(sec_se), 1),
        })

    payload = {
        "primary_indicator": "rafs_signal_level",
        "signal_failure_limit": SIGNAL_FAILURE_LIMIT,
        "trend_window_days": TREND_WINDOW_DAYS,
        "prognosis_lag_days": PROGNOSIS_LAG_DAYS,
        "horizon_days": HORIZON_DAYS,
        "secondary_indicator": SECONDARY_CHANNEL,
        "secondary_limit_ns": sec_limit,
        "secondary_healthy_p95_ns": sec_baseline,
        "satellites": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    report(results, sec_limit, sec_baseline)


def report(results, sec_limit, sec_baseline):
    width = 112
    print()
    print("Remaining useful life   (primary indicator: rafs_signal_level)")
    print("=" * width)
    print(f"  failure limit    {SIGNAL_FAILURE_LIMIT}   (below the ~0.91 post-fault plateau)")
    print(f"  trend window     {TREND_WINDOW_DAYS} days, ending on the reference day")
    print(f"  reference day    first sustained detection + {PROGNOSIS_LAG_DAYS} d "
          f"(healthy: last day of record)")
    print(f"  horizon          {HORIZON_DAYS} days")
    print("-" * width)
    print(f"  {'SATELLITE':<10}{'PROFILE':<15}{'REF DAY':>8}{'SIGNAL':>9}"
          f"{'SLOPE /day':>13}{'RUL (days)':>25}{'SECONDARY (days)':>26}")
    print("-" * width)
    for r in results:
        slope = r["signal_slope_per_day"]
        slope_txt = "--" if slope is None else f"{slope:+.2e}"
        if r["remaining_useful_life_days"] is not None:
            se = r["remaining_useful_life_se_days"]
            rul_txt = (f"{r['remaining_useful_life_days']:,} +/- {se:,.0f}"
                       if se is not None else f"{r['remaining_useful_life_days']:,}")
        else:
            rul_txt = r["status"]
        if r["secondary_rul_days"] is not None:
            sse = r["secondary_rul_se_days"]
            sec_txt = (f"{r['secondary_rul_days']:,} +/- {sse:,.0f}"
                       if sse is not None else f"{r['secondary_rul_days']:,}")
        else:
            sec_txt = r["secondary_status"]
        print(f"  {r['satellite_id']:<10}{r['profile']:<15}{r['reference_day']:>8}"
              f"{r['signal_level_at_reference']:>9.4f}{slope_txt:>13}"
              f"{rul_txt:>25}{sec_txt:>26}")
    print("-" * width)
    healthy = [r for r in results if r["profile"] == "healthy"]
    faulted = [r for r in results if r["profile"] != "healthy"]
    print(f"  healthy with no finite RUL: "
          f"{sum(1 for r in healthy if r['remaining_useful_life_days'] is None)}/{len(healthy)}")
    print(f"  faulted with a finite RUL:  "
          f"{sum(1 for r in faulted if r['remaining_useful_life_days'] is not None)}/{len(faulted)}")
    print(f"  secondary limit: {SECONDARY_MULTIPLE:g} x healthy p{SECONDARY_PERCENTILE} "
          f"({sec_baseline:.1f} ns) = {sec_limit:.1f} ns")
    print(f"  wrote {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
