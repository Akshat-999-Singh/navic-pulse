"""
NavIC Pulse -- export the trained model as plain arrays.

Lets inference run without scikit-learn (e.g. in a serverless function): the
autoencoder becomes four weight matrices and four bias vectors, the scaler two
per-channel vectors, and everything needed to rebuild the feature vector and
apply the detectors goes in a JSON sidecar.

Reads (never modifies):
  model.joblib, scaler.joblib   trained artefacts (train_model.py, preprocess.py)
  preprocessed.npz              X_all, the scaled windows the model was scored on
  telemetry.csv                 raw channels, for the end-to-end check
  scores.csv                    the published anomaly_score per window
  detections.csv                the published CUSUM statistic, alarms and severity
  navic_recalibration.json      the shipped static threshold

Writes:
  artifacts/model_weights.npz   w0, b0, w1, b1, ... plus scaler_mean, scaler_scale
  artifacts/model_meta.json     layer sizes, activations, feature layout, detector
                                parameters

Verification, before anything is written:
  A. the pure-NumPy forward pass on X_all (already scaled) reproduces every
     anomaly_score -- checks the weights, layer order and activations;
  B. raw telemetry -> exported scaler -> day-major windows -> forward pass
     reproduces them too -- checks the scaling and the channel order, which A
     cannot, because X_all was scaled and laid out by preprocess.py already;
  C. the exported CUSUM (mu, k, h), threshold and critical cut reproduce the
     cusum_s, cusum_alarm, threshold_alarm and severity columns of detections.csv.
Any failure stops the script without writing artefacts.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from config import CHANNELS, CUSUM_PERSISTENCE, K_SIGMA, SATELLITES, WINDOW
from detect import MAX_HEALTHY_ALARM_RATE, WARNING_SHARE, calibrate_cusum, cusum

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_MODEL = os.path.join(HERE, "model.joblib")
INPUT_SCALER = os.path.join(HERE, "scaler.joblib")
INPUT_NPZ = os.path.join(HERE, "preprocessed.npz")
INPUT_TELEMETRY = os.path.join(HERE, "telemetry.csv")
INPUT_SCORES = os.path.join(HERE, "scores.csv")
INPUT_DETECTIONS = os.path.join(HERE, "detections.csv")
INPUT_RECAL = os.path.join(HERE, "navic_recalibration.json")

OUTPUT_DIR = os.path.join(HERE, "artifacts")
OUTPUT_WEIGHTS = os.path.join(OUTPUT_DIR, "model_weights.npz")
OUTPUT_META = os.path.join(OUTPUT_DIR, "model_meta.json")

TOLERANCE = 1e-9

ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
}


def fail(message):
    print(f"\n  FAILED: {message}\n  No artefacts written.\n")
    sys.exit(1)


def forward(x_scaled, weights, biases, hidden, output):
    """Autoencoder forward pass: matmul, add bias, hidden activation on every
    layer except the last, which uses the output activation."""
    h = x_scaled
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w + b
        h = ACTIVATIONS[output if i == last else hidden](h)
    return h


def reconstruction_error(x_scaled, weights, biases, hidden, output):
    return np.mean((x_scaled - forward(x_scaled, weights, biases, hidden, output)) ** 2, axis=1)


def windows_from_telemetry(telemetry, mean, scale):
    """Raw telemetry -> scaled, flattened windows, laid out as preprocess.py does.

    Per satellite in config.SATELLITES order, sorted by day: scale each channel
    with (value - mean) / scale, take every WINDOW-day window, and flatten it
    DAY-MAJOR -- [d0: c0..c4, d1: c0..c4, ...] -- with channels in CHANNELS order.
    """
    blocks = []
    for sat in SATELLITES:
        frame = telemetry[telemetry["satellite_id"] == sat["id"]].sort_values("day")
        scaled = (frame[CHANNELS].to_numpy(dtype=float) - mean) / scale  # (days, channels)
        n = len(scaled) - WINDOW + 1
        idx = np.arange(WINDOW)[None, :] + np.arange(n)[:, None]          # (windows, WINDOW)
        blocks.append(scaled[idx].reshape(n, WINDOW * len(CHANNELS)))     # day-major
    return np.concatenate(blocks, axis=0)


def main():
    model = joblib.load(INPUT_MODEL)
    scaler = joblib.load(INPUT_SCALER)

    weights = [np.asarray(w, dtype=np.float64) for w in model.coefs_]
    biases = [np.asarray(b, dtype=np.float64) for b in model.intercepts_]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    hidden, output = model.activation, model.out_activation_
    layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]

    width = 78
    print()
    print("Model export")
    print("=" * width)
    print(f"  model        {type(model).__name__}   hidden activation {hidden!r}, output {output!r}")
    print(f"  layer sizes  {'-'.join(map(str, layer_sizes))}")
    for i, (w, b) in enumerate(zip(weights, biases)):
        print(f"  w{i} {str(w.shape):>11}  b{i} {str(b.shape):>7}  {w.dtype}")
    print(f"  scaler_mean  {mean.shape}   scaler_scale {scale.shape}   ({type(scaler).__name__})")

    if hidden not in ACTIVATIONS or output not in ACTIVATIONS:
        fail(f"unsupported activation: hidden {hidden!r}, output {output!r}")
    if layer_sizes != [WINDOW * len(CHANNELS), 64, 12, 64, WINDOW * len(CHANNELS)]:
        fail(f"layer sizes {layer_sizes} are not the expected 150-64-12-64-150")
    if mean.shape != (len(CHANNELS),):
        fail(f"scaler has {mean.shape} entries, expected one per channel ({len(CHANNELS)})")

    # -- A: forward pass on X_all as saved ------------------------------------
    bundle = np.load(INPUT_NPZ, allow_pickle=True)
    x_all = bundle["X_all"]
    feature_names = [str(name) for name in bundle["feature_names"]]
    scores = pd.read_csv(INPUT_SCORES)
    published = scores["anomaly_score"].to_numpy(dtype=np.float64)
    if len(published) != len(x_all):
        fail("scores.csv and preprocessed.npz have different row counts")

    diff_a = np.abs(reconstruction_error(x_all, weights, biases, hidden, output) - published)

    # -- B: raw telemetry through the exported scaler -------------------------
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    x_raw = windows_from_telemetry(telemetry, mean, scale)
    diff_b_features = np.abs(x_raw - x_all).max()
    diff_b = np.abs(reconstruction_error(x_raw, weights, biases, hidden, output) - published)

    expected_names = [f"{c}_t{o - WINDOW + 1:+d}" for o in range(WINDOW) for c in CHANNELS]
    names_match = feature_names == expected_names

    print()
    print("Verification against scores.csv")
    print("-" * width)
    print(f"  A  X_all (pre-scaled) -> forward pass      max |diff| {diff_a.max():.3e}  over {len(diff_a):,} windows")
    print(f"  B  raw telemetry -> scaler -> windows       max |diff| features {diff_b_features:.3e}")
    print(f"                   -> forward pass            max |diff| scores   {diff_b.max():.3e}")
    print(f"     feature layout matches preprocess.py feature_names: {names_match}")

    if diff_a.max() >= TOLERANCE:
        fail(f"forward pass on X_all differs by {diff_a.max():.3e} (>= {TOLERANCE:g}): "
             "layer order or activation is wrong")
    if not names_match:
        fail("feature layout differs from preprocess.py: channel order is wrong")
    if diff_b.max() >= TOLERANCE:
        fail(f"raw-telemetry path differs by {diff_b.max():.3e} (>= {TOLERANCE:g}): "
             "scaling or channel order is wrong")

    # -- C: detector parameters reproduce detections.csv ----------------------
    with open(INPUT_RECAL, encoding="utf-8") as handle:
        threshold = float(json.load(handle)["calibration"]["static_threshold"])

    # Exactly as detect.py: calibration statistics from the healthy windows.
    healthy = scores.loc[scores["profile"] == "healthy", "anomaly_score"].to_numpy()
    mu, sd = float(healthy.mean()), float(healthy.std())
    if abs((mu + K_SIGMA * sd) - threshold) > TOLERANCE:
        fail("navic_recalibration.json threshold is not mu + K_SIGMA * sd of the healthy scores")

    ordered = scores.sort_values(["satellite_id", "end_day"]).reset_index(drop=True)
    best, _, _ = calibrate_cusum(ordered, mu, sd)
    k, h = float(best["k"]), float(best["h"])
    critical = float(np.quantile(ordered.loc[ordered["anomaly_score"] > threshold, "anomaly_score"], WARNING_SHARE))

    detections = pd.read_csv(INPUT_DETECTIONS).sort_values(["satellite_id", "end_day"]).reset_index(drop=True)
    cusum_s = np.concatenate([
        cusum(g.sort_values("end_day")["anomaly_score"].to_numpy(), mu, k)
        for _, g in detections.groupby("satellite_id", sort=True)
    ])
    diff_cusum = np.abs(cusum_s - detections["cusum_s"].to_numpy()).max()
    values = detections["anomaly_score"].to_numpy()
    severity = np.where(values > critical, "Critical", np.where(values > threshold, "Warning", "Normal"))
    checks = {
        "cusum_s": diff_cusum < 1e-9,
        "cusum_alarm": bool(((cusum_s > h) == detections["cusum_alarm"].to_numpy()).all()),
        "threshold_alarm": bool(((values > threshold) == detections["threshold_alarm"].to_numpy()).all()),
        "severity": bool((severity == detections["severity"].to_numpy()).all()),
    }
    print(f"  C  detector parameters vs detections.csv    cusum_s max |diff| {diff_cusum:.3e}; "
          + ", ".join(f"{name} {'ok' if ok else 'MISMATCH'}" for name, ok in checks.items()))
    if not all(checks.values()):
        fail("exported detector parameters do not reproduce detections.csv")

    # -- write ----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arrays = {f"w{i}": w for i, w in enumerate(weights)}
    arrays.update({f"b{i}": b for i, b in enumerate(biases)})
    arrays.update(scaler_mean=mean, scaler_scale=scale)
    np.savez_compressed(OUTPUT_WEIGHTS, **arrays)

    meta = {
        "model": "autoencoder",
        "layer_sizes": layer_sizes,
        "hidden_activation": hidden,
        "output_activation": output,
        "arrays": {
            "weights": [f"w{i}" for i in range(len(weights))],
            "biases": [f"b{i}" for i in range(len(biases))],
            "layer_math": "h = activation(h @ w_i + b_i); hidden activation on all but the last layer",
            "scaler": {"mean": "scaler_mean", "scale": "scaler_scale", "apply": "(raw - mean) / scale, per channel"},
        },
        "features": {
            "window_days": WINDOW,
            "channels": list(CHANNELS),
            "layout": "day-major: for each day from oldest (t-29) to newest (t+0), every channel in 'channels' order",
            "names": expected_names,
        },
        "score": "mean over the 150 features of (x_scaled - reconstruction)^2",
        "detection": {
            "threshold": threshold,
            "threshold_rule": f"healthy mu + {K_SIGMA:g} * healthy sd (navic_recalibration.json)",
            "healthy_mu": mu,
            "healthy_sd": sd,
            "severity_critical": critical,
            "severity_rule": "Normal <= threshold < Warning <= severity_critical < Critical",
            "cusum": {
                "rule": "S = max(0, S_prev + (score - mu) - k), per satellite, starting at 0",
                "mu": mu,
                "k": k,
                "k_sigma": float(best["k_sigma"]),
                "h": h,
                "max_healthy_alarm_rate": MAX_HEALTHY_ALARM_RATE,
            },
            "persistence_windows": CUSUM_PERSISTENCE,
        },
    }
    with open(OUTPUT_META, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print()
    print(f"  wrote {os.path.relpath(OUTPUT_WEIGHTS, HERE)}  {os.path.getsize(OUTPUT_WEIGHTS):,} bytes")
    print(f"  wrote {os.path.relpath(OUTPUT_META, HERE)}  {os.path.getsize(OUTPUT_META):,} bytes")
    print()


if __name__ == "__main__":
    main()
