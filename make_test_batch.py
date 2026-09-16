"""
NavIC Pulse -- test batch for live analysis on the Upload screen.

Invents IRNSS-1Q, an imported-clock satellite with 1460 consecutive days of
telemetry: healthy for 900 days, then a slow clock drift. The fault amplitude
is not guessed. It is tuned by scoring the batch with frontend/api/_lib, the
same code the serverless function runs, until the final window is WARNING
with its score comfortably inside (threshold, 2.5 x threshold).

Construction reuses generate_telemetry.py's builders, in its order:
  1. temperature (annual eclipse-season cycle)
  2. freq_offset_y as a healthy unit (random walk, residual drift, thermal term)
  3. the fault: a quadratic ramp on freq_offset_y from FAULT_ONSET_DAY
  4. clock_bias_ns / steering integrated from the FAULTED frequency
  5. rafs_signal_level as a healthy unit, minus a matching quadratic decline
  6. bus_voltage_v

Every random draw happens once, before tuning, so iterations differ only in
the ramp amplitude and the search is not chasing noise.

The ramp is quadratic, zero value and zero slope at onset: no step anywhere.
One amplitude `a` drives both channels, as a fraction of the project's own
slow-drift fault (DRIFT_SLOW_LARGE end offset, SIGNAL_DROP["slow_drift"]).

Outputs:
  frontend/public/test-batch-irnss-1q.csv
"""

import hashlib
import io
import os
import sys

sys.dont_write_bytecode = True  # leave no __pycache__ beside existing files

import numpy as np
import pandas as pd

import generate_telemetry as gen
from config import CHANNELS, DRIFT_SLOW_LARGE, RANDOM_SEED, SATELLITES

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frontend", "api"))
from _lib import pipeline  # noqa: E402 -- the function's own pipeline
from _lib.meta import SEVERITY_CRITICAL, THRESHOLD, WINDOW  # noqa: E402

INPUT_TELEMETRY = os.path.join(HERE, "telemetry.csv")
OUTPUT_CSV = os.path.join(HERE, "frontend", "public", "test-batch-irnss-1q.csv")
MAX_BYTES = 4 * 1024 * 1024

SAT_ID = "IRNSS-1Q"
CLOCK_TYPE = "imported"
PROFILE = "slow_drift"
N_DAYS = 1460
FAULT_ONSET_DAY = 900
# The next index after the roster, as if IRNSS-1Q were the eleventh satellite.
SEED = RANDOM_SEED + len(SATELLITES)

FORM_METADATA = {"Satellite name": SAT_ID, "Launch date": "2022-09-01",
                 "Clock type": "Imported", "Orbit": "IGSO"}

# Tuning target, as multiples of the static threshold. The requested band is
# (1, 2.5); these margins keep the final score clear of both edges.
BAND = (1.0, 2.5)
COMFORT = (1.3, 2.0)
START_AMPLITUDE = 0.1
MAX_ITERATIONS = 25

# Healthy-fleet statistics reported by baseline_ool.py (population sd).
FLEET = {
    "freq_offset_y": (-4.412e-13, 2.146e-12),
    "trb_core_temp_c": (10.01, 3.184),
    "rafs_signal_level": (1.0, 0.003988),
    "bus_voltage_v": (49.73, 0.645),
    "steering_correction_ns": (-38.12, 185.8),
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def healthy_base():
    """Every stochastic component, drawn once in generate_telemetry's order."""
    rng = np.random.default_rng(SEED)
    days = np.arange(N_DAYS)
    no_onset = N_DAYS * 2  # generate_satellite's convention for "never"
    temp, eclipse = gen.build_temperature(rng, days)
    freq = gen.build_frequency(rng, days, temp, "healthy", no_onset)
    signal, _ = gen.build_signal_level(rng, days, "healthy", no_onset)
    voltage = gen.build_bus_voltage(rng, days, eclipse)
    return {"days": days, "temp": temp, "freq": freq, "signal": signal, "voltage": voltage}


def fault_shape(days):
    """0 before onset, then ((d - onset) / span)^2 reaching 1 on the last day."""
    span = float(N_DAYS - 1 - FAULT_ONSET_DAY)
    return np.clip((days - FAULT_ONSET_DAY) / span, 0.0, None) ** 2


def build_frame(base, amplitude, header):
    shape = fault_shape(base["days"])
    freq = base["freq"] + amplitude * DRIFT_SLOW_LARGE * shape
    bias, reset, correction = gen.integrate_bias(freq)
    signal = base["signal"] - amplitude * gen.SIGNAL_DROP[PROFILE] * shape

    frame = pd.DataFrame({
        "day": base["days"],
        "satellite_id": SAT_ID,
        "clock_type": CLOCK_TYPE,
        "profile": PROFILE,
        "anomaly_start_day": FAULT_ONSET_DAY,
        "freq_offset_y": freq,
        "clock_bias_ns": bias,
        "steering_reset": reset,
        "steering_correction_ns": correction,
        "trb_core_temp_c": base["temp"],
        "rafs_signal_level": signal,
        "bus_voltage_v": base["voltage"],
    })
    missing = [c for c in header if c not in frame.columns]
    extra = [c for c in frame.columns if c not in header]
    if missing or extra:
        raise SystemExit(f"telemetry.csv header changed: missing {missing}, unexpected {extra}")
    return frame[header]


def to_csv_bytes(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Scoring -- through the serverless function's pipeline, CSV bytes in
# ---------------------------------------------------------------------------
def score(data):
    run = pipeline.run(data, "daily")
    sat = run["satellites"][0]
    parsed = pipeline.ingest.parse(data, "daily")
    days = parsed[SAT_ID]["days"]

    def end_day(index):
        return None if index is None else int(days[index + WINDOW - 1])

    return {
        "score": sat["anomaly_score"],
        "ratio": sat["anomaly_score"] / THRESHOLD,
        "severity": sat["severity"],
        "first_flagged_index": sat["first_flagged_index"],
        "first_flagged_day": end_day(sat["first_flagged_index"]),
        "confirmed_index": sat["persistence_confirmed_index"],
        "confirmed_day": end_day(sat["persistence_confirmed_index"]),
    }


# ---------------------------------------------------------------------------
def tune(base, header):
    """Geometric bisection on amplitude until the final ratio is inside COMFORT."""
    print()
    print(f"Tuning {SAT_ID}: final-window score target {COMFORT[0]:g}-{COMFORT[1]:g} x threshold "
          f"({COMFORT[0] * THRESHOLD:.4f}-{COMFORT[1] * THRESHOLD:.4f}), "
          f"requested band {BAND[0]:g}-{BAND[1]:g} x")
    print("=" * 100)
    print(f"  {'ITER':>4}  {'AMPLITUDE':>10}  {'FREQ RAMP':>10}  {'SIGNAL DROP':>11}  "
          f"{'SCORE':>9}  {'x THRESH':>8}  {'SEVERITY':<9} DECISION")
    print("-" * 100)

    lo = hi = None
    amplitude = START_AMPLITUDE
    for iteration in range(1, MAX_ITERATIONS + 1):
        frame = build_frame(base, amplitude, header)
        data = to_csv_bytes(frame)
        result = score(data)
        ratio = result["ratio"]

        if COMFORT[0] <= ratio <= COMFORT[1] and result["severity"] == "warning":
            decision = "accept"
        elif ratio < COMFORT[0]:
            lo = amplitude
            decision = "normal / low edge -> increase"
        else:
            hi = amplitude
            decision = ("critical" if ratio >= BAND[1] else "high edge") + " -> reduce"

        print(f"  {iteration:>4}  {amplitude:>10.5f}  {amplitude * DRIFT_SLOW_LARGE:>10.3e}  "
              f"{amplitude * gen.SIGNAL_DROP[PROFILE]:>11.5f}  {result['score']:>9.4f}  "
              f"{ratio:>8.3f}  {result['severity']:<9} {decision}")

        if decision == "accept":
            print("-" * 100)
            return amplitude, frame, data, result, iteration
        if lo is not None and hi is not None:
            amplitude = float(np.sqrt(lo * hi))
        elif lo is not None:
            amplitude = lo * 2.0
        else:
            amplitude = hi / 2.0

    raise SystemExit(f"no amplitude landed in the target band after {MAX_ITERATIONS} iterations")


def plausibility(frame):
    """Healthy-span channel statistics against the fleet figures."""
    healthy = frame[frame["day"] < FAULT_ONSET_DAY]
    print(f"Healthy span (days 0-{FAULT_ONSET_DAY - 1}) against healthy-fleet statistics")
    print("-" * 100)
    print(f"  {'CHANNEL':<24}{'FLEET MEAN':>13}{'FLEET SD':>12}{'MEAN':>13}{'SD':>12}"
          f"{'MEAN OFF (fleet sd)':>21}")
    for channel in CHANNELS:
        f_mean, f_sd = FLEET[channel]
        values = healthy[channel].to_numpy(dtype=float)
        mean, sd = float(values.mean()), float(values.std())
        print(f"  {channel:<24}{f_mean:>13.4g}{f_sd:>12.4g}{mean:>13.4g}{sd:>12.4g}"
              f"{(mean - f_mean) / f_sd:>+21.2f}")
    print()


def main():
    header = pd.read_csv(INPUT_TELEMETRY, nrows=0).columns.tolist()
    base = healthy_base()
    amplitude, frame, data, result, iterations = tune(base, header)

    if len(data) >= MAX_BYTES:
        raise SystemExit(f"{len(data):,} bytes is over the 4 MB upload limit")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "wb") as handle:
        handle.write(data)

    print()
    plausibility(frame)
    print("Result")
    print("-" * 100)
    print(f"  amplitude              {amplitude:.5f} of the slow-drift fault "
          f"(freq ramp {amplitude * DRIFT_SLOW_LARGE:.3e} by day {N_DAYS - 1}, "
          f"signal drop {amplitude * gen.SIGNAL_DROP[PROFILE]:.5f}), {iterations} iterations")
    print(f"  anomaly score          {result['score']:.6f}  (final window, days "
          f"{N_DAYS - WINDOW}-{N_DAYS - 1})")
    print(f"  threshold              {THRESHOLD:.6f}  -> {result['ratio']:.3f} x threshold; "
          f"critical above {SEVERITY_CRITICAL:.4f}")
    print(f"  severity               {result['severity']}")
    for label, index, day in (("first flagged", result["first_flagged_index"], result["first_flagged_day"]),
                              ("persistence confirmed", result["confirmed_index"], result["confirmed_day"])):
        text = "never" if index is None else f"day {day}  (window {index}, ends that day)"
        print(f"  {label:<23}{text}")
    print(f"  fault onset            day {FAULT_ONSET_DAY}")
    print(f"  wrote                  {os.path.relpath(OUTPUT_CSV, HERE)}  "
          f"({len(frame)} rows, {len(data):,} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]})")
    print()
    print("Upload form metadata")
    print("-" * 100)
    for label, value in FORM_METADATA.items():
        print(f"  {label:<16}{value}")
    print(f"  {'Cadence':<16}Daily")
    print()


if __name__ == "__main__":
    main()
