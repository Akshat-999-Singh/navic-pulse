"""
NavIC Pulse -- synthetic telemetry generator.

Builds N_DAYS of daily telemetry for every satellite in the roster and writes
telemetry.csv.

Construction order (each stage feeds the next):
  1. freq_offset_y base  -- mean-reverting random walk + residual nominal drift
  2. trb_core_temp_c     -- 10 C baseline + annual eclipse-season sinusoid
  3. temperature coupled back into frequency at TEMP_COEFF
  4. failure injection   -- ramped, never abrupt
  5. clock_bias_ns       -- the INTEGRAL of the finished frequency series
  6. rafs_signal_level, bus_voltage_v

Note on ordering: the spec asks for integration at step 4 and failures at
step 6, but clock_bias_ns must be the integral of the *faulted* frequency --
otherwise a frequency jump leaves no trace in the bias, which is both
physically wrong and would hide the fault from the detector. Integration is
therefore performed last, after failure injection. Everything else follows the
requested order.
"""

import numpy as np
import pandas as pd

from config import (
    CHANNELS,
    DRIFT_SLOW_LARGE,
    JUMP_SUDDEN,
    JUMP_SUDDEN_WEAK,
    N_DAYS,
    NOISE_INCREASE_FACTOR,
    NOMINAL_DRIFT_PER_DAY,
    RANDOM_SEED,
    SATELLITES,
    TEMP_COEFF,
    TEMP_RANGE_C,
)

# ---------------------------------------------------------------------------
# Generator constants
# ---------------------------------------------------------------------------
SECONDS_PER_DAY = 86400
NS_PER_SECOND = 1e9
YEAR_DAYS = 365.25

# -- temperature -----------------------------------------------------------
TEMP_BASELINE_C = 10.0     # mid-range of the -5..+15 C spec window
TEMP_ANNUAL_AMP_C = 4.5    # keeps the sinusoid inside TEMP_RANGE_C
TEMP_NOISE_SD_C = 0.15

# -- frequency -------------------------------------------------------------
# The iRAFS 5e-13/day figure is the drift *limit* on the free-running clock.
# Applied undamped for 2900 days it would accumulate to 1.4e-9, far outside the
# 1e-12..1e-11 band this telemetry is meant to occupy. The ground segment
# estimates and removes the modelled drift, so what survives in a
# ground-derived frequency offset is a small unmodelled residual -- that
# fraction is applied here, and the full spec figure remains the reference the
# faults are sized against.
DRIFT_RESIDUAL_FRACTION = 0.002        # -> ~1e-15/day, ~3e-12 over the mission
DEGRADED_DRIFT_MULT = 10.0             # long-degraded units drift ~10x faster

RW_STEP_SD = 5e-14                     # daily random-walk innovation
RW_REVERSION = 0.003                   # weak mean reversion (keeps y bounded)
FREQ_INIT_SD = 5e-13                   # post-calibration offset at day 0

# -- failure ramp lengths (days) -- nothing steps instantaneously ----------
RAMP_SLOW_DRIFT = 250
RAMP_SUDDEN_JUMP = 30
RAMP_NOISE_INCREASE = 150
RAMP_DEGRADED_KNEE = 200

# -- signal level ----------------------------------------------------------
SIGNAL_NOMINAL = 1.0
SIGNAL_NOISE_SD = 0.004
SIGNAL_LEAD_DAYS = 60          # signal degrades this far BEFORE the frequency
SIGNAL_RAMP_DAYS = 120         # half the decline lands inside the lead
                               # window, so the warning clears the noise
                               # floor well before the frequency faults
SIGNAL_DROP = {                # total decline by end of mission, by profile
    "slow_drift": 0.09,
    "sudden_jump": 0.06,
    "noise_increase": 0.07,
    "degraded": 0.12,
}

# -- power bus -------------------------------------------------------------
BUS_NOMINAL_V = 50.0
BUS_NOISE_SD = 0.02
BUS_ECLIPSE_DIP_V = 0.35
BUS_LOAD_SD = 0.05

# -- ground segment clock steering ----------------------------------------
# A real ground segment uploads clock corrections regularly, so accumulated
# bias is reset and never runs to kilometre scale. Once |bias| exceeds this
# threshold the accumulated offset is subtracted and bias returns to zero,
# making the series a sawtooth rather than a monotonic ramp. Reset frequency
# is itself a degradation signal: a healthy clock needs correcting rarely, a
# drifting one constantly.
STEERING_THRESHOLD_NS = 50.0

OUTPUT_CSV = "telemetry.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smooth_ramp(days, start, length):
    """Smoothstep from 0 to 1 beginning at `start`, saturating after `length`.

    Used for every fault onset so nothing enters as a discontinuity.
    """
    r = np.clip((days - start) / float(length), 0.0, 1.0)
    return r * r * (3.0 - 2.0 * r)


def linear_ramp(days, start, length):
    """Linear 0 -> 1 ramp beginning at `start`, saturating after `length`.

    Used for the signal-level decay rather than smooth_ramp: smoothstep has
    zero gradient at onset, so a signal decaying on a smoothstep stays inside
    its own noise floor for the whole lead window and the early warning is
    mathematically present but undetectable. A linear decay has slope from the
    first day, which is what makes the lead observable.
    """
    return np.clip((days - start) / float(length), 0.0, 1.0)


def mean_reverting_walk(rng, sigma_per_day):
    """Random walk with weak mean reversion; sigma may vary per day."""
    n = len(sigma_per_day)
    out = np.empty(n)
    value = 0.0
    innovations = rng.normal(0.0, 1.0, n) * sigma_per_day
    for t in range(n):
        value = value * (1.0 - RW_REVERSION) + innovations[t]
        out[t] = value
    return out


# ---------------------------------------------------------------------------
# Channel builders
# ---------------------------------------------------------------------------
def build_temperature(rng, days):
    """TRB core package temperature: annual eclipse-season cycle.

    Returns (temperature, eclipse_factor) where eclipse_factor is 0..1 and
    peaks during the cold half of the year.
    """
    phase = rng.uniform(0.0, 2.0 * np.pi)
    seasonal = np.sin(2.0 * np.pi * days / YEAR_DAYS + phase)
    temp = (
        TEMP_BASELINE_C
        + TEMP_ANNUAL_AMP_C * seasonal
        + rng.normal(0.0, TEMP_NOISE_SD_C, len(days))
    )
    temp = np.clip(temp, TEMP_RANGE_C[0], TEMP_RANGE_C[1])
    eclipse = np.clip(-seasonal, 0.0, 1.0)
    return temp, eclipse


def build_drift_rate(rng, days, profile, onset):
    """Per-day frequency drift rate, including any fault contribution."""
    n = len(days)
    sign = rng.choice([-1.0, 1.0])
    base = NOMINAL_DRIFT_PER_DAY * DRIFT_RESIDUAL_FRACTION * rng.uniform(0.6, 1.4)
    rate = np.full(n, sign * base)

    if profile == "degraded":
        # Elevated drift from day 0, worsening again at the anomaly knee.
        rate = rate * DEGRADED_DRIFT_MULT
        rate = rate * (1.0 + smooth_ramp(days, onset, RAMP_DEGRADED_KNEE))

    elif profile == "slow_drift":
        # Drift rate climbs, scaled so the accumulated excursion reaches
        # DRIFT_SLOW_LARGE by the end of the record.
        shape = smooth_ramp(days, onset, RAMP_SLOW_DRIFT)
        total = shape.sum()
        if total > 0:
            rate = rate + shape * (DRIFT_SLOW_LARGE / total)

    return rate


def build_frequency(rng, days, temp, profile, onset):
    """freq_offset_y: random walk + residual drift + temperature + faults."""
    n = len(days)

    # Random-walk innovation scale, inflated for the noise-increase fault.
    sigma = np.full(n, RW_STEP_SD)
    if profile == "noise_increase":
        growth = smooth_ramp(days, onset, RAMP_NOISE_INCREASE)
        sigma = sigma * (1.0 + (NOISE_INCREASE_FACTOR - 1.0) * growth)

    walk = mean_reverting_walk(rng, sigma)
    drift = np.cumsum(build_drift_rate(rng, days, profile, onset))

    # Temperature coupling -- the physical relationship the autoencoder must
    # learn. A frequency excursion accompanied by a matching temperature
    # excursion is normal; one without it is a fault.
    thermal = TEMP_COEFF * (temp - TEMP_BASELINE_C)

    y = rng.normal(0.0, FREQ_INIT_SD) + walk + drift + thermal

    if profile == "sudden_jump":
        y = y + JUMP_SUDDEN * smooth_ramp(days, onset, RAMP_SUDDEN_JUMP)
    elif profile == "degraded":
        # Weak sudden transition at the knee, on top of the elevated drift.
        y = y + JUMP_SUDDEN_WEAK * smooth_ramp(days, onset, RAMP_SUDDEN_JUMP)

    return y


def integrate_bias(freq, threshold=STEERING_THRESHOLD_NS):
    """clock_bias_ns as the steered integral of fractional frequency offset.

    bias(t) = bias0 + y0*t + 0.5*D*t^2 is the closed form; integrating the
    generated y series numerically reproduces it for any drift shape. On top
    of that, ground steering subtracts the accumulated offset whenever it
    exceeds `threshold`, so the series is a bounded sawtooth.

    Returns (bias, steering_reset, steering_correction_ns).

    At the frequency magnitudes modelled here one day of drift accumulates
    ~90-180 ns, so a 50 ns bound is crossed daily by every satellite and the
    reset *count* saturates at 1/day -- it separates healthy from degraded by
    1.0x, i.e. not at all. The correction *magnitude* is the signal that
    survives: a healthy clock needs ~180 ns/day of correction, a degraded one
    ~2600 ns/day. steering_correction_ns records the offset subtracted on each
    upload day (0.0 otherwise).
    """
    n = len(freq)
    bias = np.empty(n)
    reset = np.zeros(n, dtype=bool)
    correction = np.zeros(n)
    accumulated = 0.0
    for t in range(n):
        accumulated += freq[t] * SECONDS_PER_DAY * NS_PER_SECOND
        # Record the PRE-correction offset: that is the bias the clock actually
        # carried that day, and it makes the series a true sawtooth whose tooth
        # amplitude carries the drift rate. Recording the post-correction zero
        # instead would leave the channel ~91% constant-zero -- unlearnable.
        bias[t] = accumulated
        if abs(accumulated) > threshold:
            reset[t] = True
            correction[t] = accumulated   # offset removed by the upload
            accumulated = 0.0             # correction uploaded
    return bias, reset, correction


def build_signal_level(rng, days, profile, onset):
    """RAFS signal level: nominal 1.0, degrading ahead of the frequency fault."""
    signal = SIGNAL_NOMINAL + rng.normal(0.0, SIGNAL_NOISE_SD, len(days))
    if profile == "healthy":
        return signal, None

    # The early-warning lead: the signal starts falling SIGNAL_LEAD_DAYS before
    # the frequency fault begins. This lead is the product's whole value.
    signal_onset = onset - SIGNAL_LEAD_DAYS
    drop = SIGNAL_DROP[profile]
    signal = signal - drop * linear_ramp(days, signal_onset, SIGNAL_RAMP_DAYS)
    return signal, signal_onset


def build_bus_voltage(rng, days, eclipse):
    """Regulated 50 V bus with load variation and eclipse-season dips."""
    load = mean_reverting_walk(rng, np.full(len(days), BUS_LOAD_SD))
    return (
        BUS_NOMINAL_V
        - BUS_ECLIPSE_DIP_V * eclipse
        + load
        + rng.normal(0.0, BUS_NOISE_SD, len(days))
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def generate_satellite(sat, index):
    rng = np.random.default_rng(RANDOM_SEED + index)
    days = np.arange(N_DAYS)
    profile = sat["profile"]

    # Healthy units have no onset; degraded units are already drifting at day 0
    # and use anomaly_start_day as the knee where degradation worsens.
    onset = sat["anomaly_start_day"]
    onset = N_DAYS * 2 if onset is None else onset

    temp, eclipse = build_temperature(rng, days)
    freq = build_frequency(rng, days, temp, profile, onset)
    bias, reset, correction = integrate_bias(freq)
    signal, signal_onset = build_signal_level(rng, days, profile, onset)
    voltage = build_bus_voltage(rng, days, eclipse)

    frame = pd.DataFrame({
        "day": days,
        "satellite_id": sat["id"],
        "clock_type": sat["clock_type"],
        "profile": profile,
        "anomaly_start_day": sat["anomaly_start_day"],
        "freq_offset_y": freq,
        "clock_bias_ns": bias,
        "steering_reset": reset,
        "steering_correction_ns": correction,
        "trb_core_temp_c": temp,
        "rafs_signal_level": signal,
        "bus_voltage_v": voltage,
    })
    return frame, signal_onset


def generate_all():
    frames, leads = [], {}
    for i, sat in enumerate(SATELLITES):
        frame, signal_onset = generate_satellite(sat, i)
        frames.append(frame)
        leads[sat["id"]] = signal_onset
    data = pd.concat(frames, ignore_index=True)
    data["anomaly_start_day"] = data["anomaly_start_day"].astype("Int64")
    return data, leads


def summarise(data, leads):
    width = 104
    print()
    print(f"Generated {len(data):,} rows -- {len(SATELLITES)} satellites x {N_DAYS} days")
    print("=" * width)
    print(f"{'ID':<10} {'PROFILE':<15} {'|y| max':>10} {'bias range (ns)':>22} "
          f"{'temp C':>15} {'sig min':>9} {'resets':>8} {'ns/fix':>9}")
    print("-" * width)
    for sat in SATELLITES:
        sub = data[data["satellite_id"] == sat["id"]]
        y_max = sub["freq_offset_y"].abs().max()
        b_lo, b_hi = sub["clock_bias_ns"].min(), sub["clock_bias_ns"].max()
        t_lo, t_hi = sub["trb_core_temp_c"].min(), sub["trb_core_temp_c"].max()
        s_min = sub["rafs_signal_level"].min()
        resets = int(sub["steering_reset"].sum())
        corr = sub.loc[sub["steering_reset"], "steering_correction_ns"].abs().median()
        lead = leads[sat["id"]]
        lead = "--" if lead is None else f"day {lead}"
        bias_range = f"{b_lo:.3g}..{b_hi:.3g}"
        temp_range = f"{t_lo:.1f}..{t_hi:.1f}"
        print(f"{sat['id']:<10} {sat['profile']:<15} {y_max:>10.2e} {bias_range:>22} "
              f"{temp_range:>15} {s_min:>9.3f} {resets:>8} {corr:>9.0f}")
    print("-" * width)
    inside = bool(data["trb_core_temp_c"].between(*TEMP_RANGE_C).all())
    print(f"temperature within TEMP_RANGE_C {TEMP_RANGE_C}: {inside}")
    print(f"channels: {', '.join(CHANNELS)}")
    print()


if __name__ == "__main__":
    data, leads = generate_all()
    data.to_csv(OUTPUT_CSV, index=False)
    summarise(data, leads)
    print(f"wrote {OUTPUT_CSV}")
