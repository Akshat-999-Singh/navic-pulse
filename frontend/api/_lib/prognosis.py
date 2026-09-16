"""Remaining useful life from the rafs_signal_level decline -- ported from rul.py (primary indicator).

  1. Reference day = first sustained detection's end day + PROGNOSIS_LAG_DAYS,
     capped at the batch's last day; the last day when never detected.
  2. OLS line over the TREND_WINDOW_DAYS days ending on the reference day.
  3. Crossing of SIGNAL_FAILURE_LIMIT, standard error by the delta method.
  4. RUL = crossing - reference day, within HORIZON_DAYS.

Returned as {estimate, low, high} (estimate +/- the standard error rounded to
0.1 d, as export_demo_run.py does), or None: already past the limit, no declining
trend, too few points, beyond the horizon, or no standard error to make a band.
"""

import numpy as np

INDICATOR = "rafs_signal_level"
TREND_WINDOW_DAYS = 90
PROGNOSIS_LAG_DAYS = 30
HORIZON_DAYS = 3650
SIGNAL_FAILURE_LIMIT = 0.85


def ols_crossing(x, y, limit, declining=True):
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

    if (declining and b1 >= 0) or (not declining and b1 <= 0):
        return {"slope": b1, "intercept": b0, "crossing": None, "se": None}

    crossing = (limit - b0) / b1
    var_x = (var_b0 + crossing ** 2 * var_b1 + 2 * crossing * cov_b01) / (b1 ** 2)
    se = float(np.sqrt(var_x)) if var_x > 0 else None
    return {"slope": b1, "intercept": b0, "crossing": float(crossing), "se": se}


def remaining_life(days, signal, detection_day):
    days = np.asarray(days, dtype=float)
    signal = np.asarray(signal, dtype=float)
    last_day = int(days.max())
    reference_day = (min(detection_day + PROGNOSIS_LAG_DAYS, last_day)
                     if detection_day is not None else last_day)
    keep = (days >= reference_day - TREND_WINDOW_DAYS + 1) & (days <= reference_day)
    x, y = days[keep], signal[keep]

    if len(y) == 0 or float(y.mean()) <= SIGNAL_FAILURE_LIMIT:
        return None
    fit = ols_crossing(x, y, SIGNAL_FAILURE_LIMIT, declining=True)
    if fit is None or fit["crossing"] is None:
        return None
    remaining = fit["crossing"] - reference_day
    if remaining < 0 or remaining > HORIZON_DAYS or fit["se"] is None:
        return None

    estimate = int(round(remaining))
    se = round(float(fit["se"]), 1)
    return {"estimate": estimate, "low": int(round(estimate - se)),
            "high": int(round(estimate + se))}
