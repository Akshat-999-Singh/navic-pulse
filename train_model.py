"""
NavIC Pulse -- autoencoder training and scoring.

An MLPRegressor is trained to reconstruct its own input, using healthy windows
only. Windows it reconstructs badly are, by construction, windows that do not
look like a healthy clock -- per-window reconstruction MSE is the anomaly
score.

Outputs:
  scores.csv                 window metadata + anomaly_score
  model.joblib               the fitted autoencoder
  demo/score_histogram.png   score distribution, healthy vs anomalous
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor

from config import BOTTLENECK, K_SIGMA, RANDOM_SEED

INPUT_NPZ = "preprocessed.npz"
INPUT_META = "window_metadata.csv"
OUTPUT_SCORES = "scores.csv"
OUTPUT_MODEL = "model.joblib"
OUTPUT_DIR = "demo"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "score_histogram.png")

HIDDEN_LAYERS = (64, BOTTLENECK, 64)
MAX_ITER = 200

# Validated categorical slots 1 and 2 (all-pairs CVD dE 24.7, normal 33.6).
COLOR_HEALTHY = "#2a78d6"
COLOR_ANOMALY = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def train(X_train):
    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, X_train)      # reconstruct the input
    return model


def anomaly_scores(model, X):
    """Per-window reconstruction MSE."""
    reconstructed = model.predict(X)
    return np.mean((X - reconstructed) ** 2, axis=1)


def plot_histogram(scores, meta, threshold):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    healthy = scores[~meta["is_true_anomaly"].to_numpy()]
    anomalous = scores[meta["is_true_anomaly"].to_numpy()]

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d9d8d4",
        "text.color": INK,
        "axes.labelcolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 70)

    for values, color, label in (
        (healthy, COLOR_HEALTHY, f"healthy  (n={len(healthy):,})"),
        (anomalous, COLOR_ANOMALY, f"anomalous  (n={len(anomalous):,})"),
    ):
        ax.hist(values, bins=bins, color=color, alpha=0.62,
                edgecolor=color, linewidth=0.6, label=label)

    ax.axvline(threshold, color=INK_MUTED, linestyle="--", linewidth=2, zorder=5)
    ax.annotate(
        f"threshold\nmean + {K_SIGMA:g}$\\sigma$ = {threshold:.3g}",
        xy=(threshold, ax.get_ylim()[1] * 0.92),
        xytext=(threshold * 1.25, ax.get_ylim()[1] * 0.92),
        fontsize=14, color=INK_MUTED, va="top", linespacing=1.35,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction MSE (anomaly score, log scale)", fontsize=16)
    ax.set_ylabel("Windows", fontsize=16)
    ax.set_title("Anomaly score distribution — healthy vs anomalous windows",
                 fontsize=21, color=INK, pad=18)
    ax.legend(frameon=False, fontsize=15, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    bundle = np.load(INPUT_NPZ, allow_pickle=True)
    X_train, X_all = bundle["X_train"], bundle["X_all"]
    meta = pd.read_csv(INPUT_META)
    if len(meta) != len(X_all):
        raise SystemExit("metadata and X_all are out of step -- re-run preprocess.py")

    width = 72
    print()
    print("Training autoencoder")
    print("=" * width)
    print(f"  architecture   {X_train.shape[1]} -> {' -> '.join(map(str, HIDDEN_LAYERS))} "
          f"-> {X_train.shape[1]}   (relu, adam)")
    print(f"  training on    {X_train.shape[0]:,} healthy windows")

    model = train(X_train)
    print(f"  converged      {model.n_iter_} iterations "
          f"(early stopping, max {MAX_ITER})")
    print(f"  final loss     {model.loss_:.6f}")

    scores = anomaly_scores(model, X_all)
    meta["anomaly_score"] = scores
    meta.to_csv(OUTPUT_SCORES, index=False)
    joblib.dump(model, OUTPUT_MODEL)

    is_anom = meta["is_true_anomaly"].to_numpy()
    healthy, anomalous = scores[~is_anom], scores[is_anom]
    train_scores = scores[bundle["train_mask"]]
    threshold = train_scores.mean() + K_SIGMA * train_scores.std()

    print()
    print("Anomaly score by group")
    print("-" * width)
    print(f"  {'group':<14}{'n':>9}{'mean':>14}{'median':>14}")
    print("-" * width)
    print(f"  {'healthy':<14}{len(healthy):>9,}{healthy.mean():>14.5f}{np.median(healthy):>14.5f}")
    print(f"  {'anomalous':<14}{len(anomalous):>9,}{anomalous.mean():>14.5f}{np.median(anomalous):>14.5f}")
    print("-" * width)
    print(f"  {'ratio':<14}{'':>9}{anomalous.mean()/healthy.mean():>13.1f}x"
          f"{np.median(anomalous)/np.median(healthy):>13.1f}x")
    print()
    print(f"  threshold (healthy mean + {K_SIGMA:g} sigma) = {threshold:.5f}")
    print(f"  wrote {OUTPUT_SCORES}, {OUTPUT_MODEL}")

    plot_histogram(scores, meta, threshold)
    print(f"  wrote {OUTPUT_PNG}")
    print()


if __name__ == "__main__":
    main()
