"""
NavIC Pulse -- presentation figures.

Writes three PNGs to demo/:
  anomaly_timeline.png   score vs day for one drifting clock, with threshold,
                         onset and the sustained detection marked
  score_histogram.png    healthy vs anomalous score distributions, log x
  rul_projection.png     the signal-level decline, its fitted trend, and the
                         projected crossing of the failure limit

demo/score_histogram.png is also written by train_model.py; this module is the
presentation copy and both produce the same figure from the same scores.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CUSUM_PERSISTENCE, K_SIGMA, SATELLITES
from rul import (PROGNOSIS_LAG_DAYS, SIGNAL_FAILURE_LIMIT, TREND_WINDOW_DAYS,
                 ols_crossing)

INPUT_DETECTIONS = "detections.csv"
INPUT_TELEMETRY = "telemetry.csv"
OUTPUT_DIR = "demo"

FOCUS_SAT = "IRNSS-1I"      # slow_drift, onset day 1750

# Validated categorical slots 1-3 (all-pairs CVD dE 9.2, normal-vision 24.0).
# Slot 3 sits below 3:1 contrast on this surface, so every series carrying it is
# directly labelled as well as legended -- the relief rule.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e8e7e3"


def theme():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d9d8d4",
        "axes.labelcolor": INK_MUTED,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 16,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load():
    detections = pd.read_csv(INPUT_DETECTIONS)
    telemetry = pd.read_csv(INPUT_TELEMETRY)
    # By profile, not by a positional mask -- detections.csv is written in a
    # different row order than preprocessed.npz.
    train_scores = detections.loc[detections["profile"] == "healthy",
                                  "anomaly_score"].to_numpy()
    threshold = float(train_scores.mean() + K_SIGMA * train_scores.std())
    return detections, telemetry, threshold


# ---------------------------------------------------------------------------
# 1. anomaly timeline
# ---------------------------------------------------------------------------
def plot_timeline(detections, threshold):
    sat = next(s for s in SATELLITES if s["id"] == FOCUS_SAT)
    onset = sat["anomaly_start_day"]
    g = detections[detections["satellite_id"] == FOCUS_SAT].sort_values("end_day")
    hits = g.loc[g["cusum_sustained"], "end_day"]
    detect_day = int(hits.min())
    detect_score = float(g.loc[g["end_day"] == detect_day, "anomaly_score"].iloc[0])

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    ax.plot(g["end_day"], g["anomaly_score"], color=BLUE, linewidth=1.4,
            label="anomaly score", zorder=3)
    ax.axhline(threshold, color=INK_MUTED, linestyle=":", linewidth=2.2, zorder=2)
    ax.axvline(onset, color=ORANGE, linestyle="--", linewidth=2.4, zorder=2)
    ax.scatter([detect_day], [detect_score], s=190, color=ORANGE,
               edgecolor=SURFACE, linewidth=2.5, zorder=6)

    ax.set_yscale("log")
    ax.set_xlabel("Day")
    ax.set_ylabel("Reconstruction MSE (log scale)")
    ax.set_title(f"{FOCUS_SAT} — anomaly score over the mission "
                 f"({sat['profile'].replace('_', ' ')})",
                 fontsize=22, color=INK, pad=18)

    ax.annotate(f"threshold  mean + {K_SIGMA:g}σ = {threshold:.3f}",
                xy=(60, threshold * 1.55), fontsize=14, color=INK_MUTED)
    ax.annotate(f"fault onset\nday {onset}", xy=(onset - 55, 4.0),
                ha="right", fontsize=14, color=ORANGE, linespacing=1.35)
    ax.annotate(f"sustained detection\nday {detect_day}  ({onset - detect_day:+d} d)",
                xy=(detect_day, detect_score),
                xytext=(detect_day + 130, detect_score * 0.16),
                fontsize=14, color=INK, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.4))

    handles = [
        plt.Line2D([], [], color=BLUE, linewidth=2.5, label="anomaly score"),
        plt.Line2D([], [], color=INK_MUTED, linestyle=":", linewidth=2.2,
                   label=f"threshold ({threshold:.3f})"),
        plt.Line2D([], [], color=ORANGE, linestyle="--", linewidth=2.4, label="fault onset"),
        plt.Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=11,
                   label=f"sustained detection ({CUSUM_PERSISTENCE} windows)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    tidy(ax)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. score histogram
# ---------------------------------------------------------------------------
def plot_histogram(detections, threshold):
    scores = detections["anomaly_score"].to_numpy()
    truth = detections["is_true_anomaly"].to_numpy()
    healthy, anomalous = scores[~truth], scores[truth]

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 70)
    for values, color, label in (
        (healthy, BLUE, f"healthy  (n={len(healthy):,})"),
        (anomalous, ORANGE, f"anomalous  (n={len(anomalous):,})"),
    ):
        ax.hist(values, bins=bins, color=color, alpha=0.62,
                edgecolor=color, linewidth=0.6, label=label)

    ax.axvline(threshold, color=INK_MUTED, linestyle="--", linewidth=2.2, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction MSE (anomaly score, log scale)")
    ax.set_ylabel("Windows")
    ax.set_title("Anomaly score distribution — healthy vs anomalous windows",
                 fontsize=22, color=INK, pad=18)
    ax.annotate(f"threshold\nmean + {K_SIGMA:g}σ = {threshold:.3f}",
                xy=(threshold, ax.get_ylim()[1] * 0.9),
                xytext=(threshold * 1.3, ax.get_ylim()[1] * 0.9),
                fontsize=15, color=INK_MUTED, va="top", linespacing=1.35)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "score_histogram.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. RUL projection -- the closing figure
# ---------------------------------------------------------------------------
def plot_rul(detections, telemetry):
    sat = next(s for s in SATELLITES if s["id"] == FOCUS_SAT)
    onset = sat["anomaly_start_day"]
    g = detections[detections["satellite_id"] == FOCUS_SAT]
    detect_day = int(g.loc[g["cusum_sustained"], "end_day"].min())
    reference_day = detect_day + PROGNOSIS_LAG_DAYS

    frame = telemetry[telemetry["satellite_id"] == FOCUS_SAT].sort_values("day")
    lo = reference_day - TREND_WINDOW_DAYS + 1
    window = frame[(frame["day"] >= lo) & (frame["day"] <= reference_day)]
    fit = ols_crossing(window["day"].to_numpy(float),
                       window["rafs_signal_level"].to_numpy(float),
                       SIGNAL_FAILURE_LIMIT, declining=True)
    crossing, se = fit["crossing"], fit["se"]
    rul_days = int(round(crossing - reference_day))

    x_hi = crossing + 90
    known = frame[frame["day"] <= reference_day]
    future = frame[(frame["day"] > reference_day) & (frame["day"] <= x_hi)]

    fig, ax = plt.subplots(figsize=(15.5, 8.5), dpi=150)

    # observed history, and what actually happened afterwards
    ax.plot(known["day"], known["rafs_signal_level"], color=BLUE, linewidth=1.3,
            zorder=3)
    ax.plot(future["day"], future["rafs_signal_level"], color=AQUA, linewidth=1.3,
            alpha=0.9, zorder=3)

    # the fit window and the fitted line, extrapolated to the limit
    ax.axvspan(lo, reference_day, color=BLUE, alpha=0.07, zorder=1)
    fit_x = np.array([lo, reference_day], dtype=float)
    proj_x = np.array([reference_day, crossing], dtype=float)
    line = lambda xs: fit["intercept"] + fit["slope"] * xs
    ax.plot(fit_x, line(fit_x), color=ORANGE, linewidth=3.4, zorder=5)
    ax.plot(proj_x, line(proj_x), color=ORANGE, linewidth=3.0, linestyle="--", zorder=5)

    # the failure limit and the projected crossing
    ax.axhline(SIGNAL_FAILURE_LIMIT, color=INK_MUTED, linestyle=":", linewidth=2.4, zorder=2)
    ax.axvspan(crossing - se, crossing + se, color=ORANGE, alpha=0.22, zorder=2)
    ax.scatter([crossing], [SIGNAL_FAILURE_LIMIT], s=200, color=ORANGE,
               edgecolor=SURFACE, linewidth=2.5, zorder=7)
    ax.axvline(onset, color=INK_MUTED, linestyle="--", linewidth=1.6, alpha=0.65, zorder=2)

    ax.set_xlim(lo - 130, x_hi)
    ax.set_ylim(0.83, 1.012)
    ax.set_xlabel("Day")
    ax.set_ylabel("RAFS signal level")
    ax.set_title(f"{FOCUS_SAT} — remaining useful life from the signal-level decline",
                 fontsize=22, color=INK, pad=18)

    ax.annotate(f"fault onset\nday {onset}", xy=(onset, 0.995),
                xytext=(onset - 12, 0.997), ha="right", va="top",
                fontsize=14, color=INK_MUTED, linespacing=1.35)
    ax.annotate(f"90-day fit window\nends day {reference_day}",
                xy=(lo + 4, 0.862), fontsize=14, color=BLUE, linespacing=1.35)
    ax.annotate(f"failure limit  {SIGNAL_FAILURE_LIMIT}",
                xy=(lo - 120, SIGNAL_FAILURE_LIMIT + 0.004),
                fontsize=15, color=INK_MUTED)
    ax.annotate(f"projected crossing\nday {crossing:.0f} ± {se:.0f}\n"
                f"{rul_days} days remaining",
                xy=(crossing + 6, SIGNAL_FAILURE_LIMIT + 0.0015),
                xytext=(crossing + 18, 0.8615), ha="left", va="bottom",
                fontsize=16, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=1.8))
    ax.annotate("observed after the projection —\nthe decline plateaus near 0.91",
                xy=(crossing - 20, 0.9155), xytext=(crossing - 12, 0.950),
                ha="center", fontsize=13.5, color=INK_MUTED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=1.6))

    handles = [
        plt.Line2D([], [], color=BLUE, linewidth=2.6, label="signal level, known at prognosis"),
        plt.Line2D([], [], color=AQUA, linewidth=2.6, label="signal level, after prognosis"),
        plt.Line2D([], [], color=ORANGE, linewidth=3.2, label="fitted decline, extrapolated"),
        plt.Line2D([], [], color=INK_MUTED, linestyle=":", linewidth=2.4,
                   label=f"failure limit {SIGNAL_FAILURE_LIMIT}"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left",
              bbox_to_anchor=(0.0, 0.42))
    tidy(ax)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rul_projection.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    theme()
    detections, telemetry, threshold = load()
    for path in (plot_timeline(detections, threshold),
                 plot_histogram(detections, threshold),
                 plot_rul(detections, telemetry)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
