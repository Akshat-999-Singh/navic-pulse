"""
NavIC Pulse -- windowing and scaling.

Turns telemetry.csv into the arrays the autoencoder trains on:

  * StandardScaler is fitted on HEALTHY satellites only, then applied to
    everything. Faulted units are deliberately left out of the fit -- the whole
    point is that their values fall outside the healthy distribution.
  * Each satellite is cut into overlapping WINDOW-day windows, flattened to
    WINDOW * len(CHANNELS) features.
  * X_train is the healthy windows; X_all is every window.
  * is_true_anomaly is set from each satellite's own anomaly_start_day, for
    every profile. Degraded units were previously labelled anomalous across
    their whole record; see the comment at the label construction for why that
    was changed and what it cost. Training is unaffected either way -- the
    scaler and X_train are selected by profile == "healthy", not by label.

Outputs:
  preprocessed.npz     X_train, X_all, feature_names, train_mask
  window_metadata.csv  one row per window in X_all, same order
  scaler.joblib        the fitted scaler, for reuse at inference
"""

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from config import CHANNELS, SATELLITES, WINDOW

INPUT_CSV = "telemetry.csv"
OUTPUT_NPZ = "preprocessed.npz"
OUTPUT_META = "window_metadata.csv"
OUTPUT_SCALER = "scaler.joblib"


def build_windows(values):
    """(n_days, n_channels) -> (n_windows, WINDOW, n_channels)."""
    # sliding_window_view gives (n_windows, n_channels, WINDOW); move the time
    # axis back in front of the channel axis so a flattened window reads
    # day-major: [d0c0..d0c4, d1c0..d1c4, ...].
    windows = sliding_window_view(values, WINDOW, axis=0)
    return np.transpose(windows, (0, 2, 1))


def feature_names():
    return [f"{channel}_t{offset - WINDOW + 1:+d}"
            for offset in range(WINDOW)
            for channel in CHANNELS]


def main():
    data = pd.read_csv(INPUT_CSV)
    missing = [c for c in CHANNELS if c not in data.columns]
    if missing:
        raise SystemExit(f"telemetry.csv is missing channels: {missing}")

    # -- scaler: fitted on healthy satellites only -------------------------
    healthy_rows = data[data["profile"] == "healthy"]
    scaler = StandardScaler().fit(healthy_rows[CHANNELS].to_numpy())
    scaled = scaler.transform(data[CHANNELS].to_numpy())
    data_scaled = data[["day", "satellite_id"]].copy()
    data_scaled[CHANNELS] = scaled

    # -- windows, per satellite (never spanning two satellites) -----------
    blocks, meta_rows = [], []
    for sat in SATELLITES:
        sub = data_scaled[data_scaled["satellite_id"] == sat["id"]].sort_values("day")
        windows = build_windows(sub[CHANNELS].to_numpy())
        blocks.append(windows)

        end_days = sub["day"].to_numpy()[WINDOW - 1:]
        onset = sat["anomaly_start_day"]
        # Every profile is scored from its own anomaly_start_day. Degraded units
        # used to be labelled anomalous for their whole record, on the grounds
        # that build_drift_rate multiplies their drift by DEGRADED_DRIFT_MULT
        # from day 0 and anomaly_start_day is only the knee where they worsen.
        # That is still true of the generator, but it made 1,242 windows
        # anomalous whose scores are indistinguishable from the healthy fleet
        # (median 0.160 against 0.148), and they accounted for 100% of the
        # pipeline's false negatives -- pinning recall at 0.86 while measuring
        # the labelling rather than the detector. Scoring from the knee makes
        # the label mean "detectably faulty", uniformly across profiles.
        is_anomaly = (np.zeros(len(end_days), dtype=bool) if onset is None
                      else end_days >= onset)
        meta_rows.append(pd.DataFrame({
            "satellite_id": sat["id"],
            "end_day": end_days,
            "clock_type": sat["clock_type"],
            "profile": sat["profile"],
            "anomaly_start_day": onset,
            "is_true_anomaly": is_anomaly,
        }))

    X_all = np.concatenate(blocks, axis=0)
    X_all = X_all.reshape(len(X_all), -1)
    meta = pd.concat(meta_rows, ignore_index=True)
    meta["anomaly_start_day"] = meta["anomaly_start_day"].astype("Int64")

    train_mask = (meta["profile"] == "healthy").to_numpy()
    X_train = X_all[train_mask]

    names = feature_names()
    assert X_all.shape[1] == len(names) == WINDOW * len(CHANNELS)
    assert len(meta) == len(X_all)

    np.savez_compressed(
        OUTPUT_NPZ,
        X_train=X_train,
        X_all=X_all,
        train_mask=train_mask,
        feature_names=np.array(names),
    )
    meta.to_csv(OUTPUT_META, index=False)
    joblib.dump(scaler, OUTPUT_SCALER)

    report(data, meta, X_train, X_all, scaler)


def report(data, meta, X_train, X_all, scaler):
    width = 72
    healthy_sats = [s["id"] for s in SATELLITES if s["profile"] == "healthy"]

    print()
    print("Preprocessing")
    print("=" * width)
    print(f"  source rows          {len(data):>10,}")
    print(f"  channels             {len(CHANNELS):>10}  {', '.join(CHANNELS)}")
    print(f"  window length        {WINDOW:>10} days")
    print(f"  features per window  {X_all.shape[1]:>10}  ({len(CHANNELS)} channels x {WINDOW} days)")
    print()
    print(f"  scaler fitted on     {len(healthy_sats)} healthy satellites "
          f"({', '.join(healthy_sats)})")
    print(f"  {'channel':<24}{'mean':>14}{'scale':>16}")
    for name, mean, scale in zip(CHANNELS, scaler.mean_, scaler.scale_):
        print(f"  {name:<24}{mean:>14.4g}{scale:>16.4g}")
    print()
    print("Shapes")
    print("-" * width)
    print(f"  X_train              {str(X_train.shape):>20}   healthy windows only")
    print(f"  X_all                {str(X_all.shape):>20}   every window")
    print(f"  metadata             {str(meta.shape):>20}   aligned with X_all")
    print()
    print("Window counts")
    print("-" * width)
    n_anom = int(meta["is_true_anomaly"].sum())
    print(f"  healthy (label False)  {len(meta) - n_anom:>8,}   {100*(1-n_anom/len(meta)):>5.1f}%")
    print(f"  anomalous (label True) {n_anom:>8,}   {100*n_anom/len(meta):>5.1f}%")
    print()
    print(f"  {'satellite':<11}{'profile':<16}{'windows':>9}{'anomalous':>11}{'first flagged':>15}")
    print("-" * width)
    for sat in SATELLITES:
        sub = meta[meta["satellite_id"] == sat["id"]]
        n = int(sub["is_true_anomaly"].sum())
        first = sub.loc[sub["is_true_anomaly"], "end_day"]
        first = "--" if first.empty else f"day {int(first.iloc[0])}"
        print(f"  {sat['id']:<11}{sat['profile']:<16}{len(sub):>9,}{n:>11,}{first:>15}")
    print("-" * width)
    print(f"  wrote {OUTPUT_NPZ}, {OUTPUT_META}, {OUTPUT_SCALER}")
    print()


if __name__ == "__main__":
    main()
