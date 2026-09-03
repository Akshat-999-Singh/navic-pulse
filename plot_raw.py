"""
NavIC Pulse -- raw telemetry sanity check.

Plots all six model channels for one healthy satellite and one slow-drift
satellite side by side, so the two can be compared row by row before any
modelling starts.

Output: demo/raw_check.png
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CHANNELS, SATELLITES

# clock_bias_ns is no longer a model input (it correlates with
# steering_correction_ns at r = 1.000), but it stays in the CSV and on this
# plot for reference, so the raw check still shows all six recorded series.
PLOT_CHANNELS = [
    "freq_offset_y",
    "clock_bias_ns",
    "trb_core_temp_c",
    "rafs_signal_level",
    "bus_voltage_v",
    "steering_correction_ns",
]
assert set(CHANNELS).issubset(PLOT_CHANNELS)

# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "telemetry.csv"
OUTPUT_DIR = "demo"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "raw_check.png")

LEFT_SAT = "IRNSS-1C"     # healthy reference
RIGHT_SAT = "IRNSS-1I"    # slow_drift, onset day 1750

# Validated categorical slots 1 and 2 (all-pairs CVD dE 24.7, normal 33.6).
COLOR_HEALTHY = "#2a78d6"
COLOR_FAULT = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
ONSET_LINE = "#52514e"

# Per-channel display label and scale factor. freq_offset_y is rescaled so the
# axis reads in units of 1e-12 rather than as a wall of exponents.
CHANNEL_DISPLAY = {
    "freq_offset_y":          ("Frequency offset\n($\\times 10^{-12}$)", 1e12),
    "clock_bias_ns":          ("Clock bias\n(ns, reference only)", 1.0),
    "trb_core_temp_c":        ("TRB core temp\n(°C)", 1.0),
    "rafs_signal_level":      ("RAFS signal level\n(normalised)", 1.0),
    "bus_voltage_v":          ("Bus voltage\n(V)", 1.0),
    "steering_correction_ns": ("Steering correction\n(ns per upload)", 1.0),
}


def satellite_label(sat_id):
    """'IRNSS-1I -- slow drift, onset day 1750' style column heading."""
    sat = next(s for s in SATELLITES if s["id"] == sat_id)
    profile = sat["profile"].replace("_", " ")
    onset = sat["anomaly_start_day"]
    detail = f"{profile}, onset day {onset}" if onset is not None else profile
    return f"{sat_id}\n{detail} · {sat['clock_type']}"


def load(sat_id):
    data = pd.read_csv(INPUT_CSV)
    sub = data[data["satellite_id"] == sat_id].sort_values("day")
    if sub.empty:
        raise SystemExit(f"{sat_id} not found in {INPUT_CSV}")
    return sub


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    left, right = load(LEFT_SAT), load(RIGHT_SAT)
    onset = next(s["anomaly_start_day"] for s in SATELLITES if s["id"] == RIGHT_SAT)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d9d8d4",
        "axes.labelcolor": INK_MUTED,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.8,
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    n = len(PLOT_CHANNELS)
    fig, axes = plt.subplots(
        n, 2, figsize=(19, 15), dpi=150,
        sharex=True, sharey="row",       # shared row scale == honest comparison
    )

    columns = [
        (left, LEFT_SAT, COLOR_HEALTHY, False),
        (right, RIGHT_SAT, COLOR_FAULT, True),
    ]

    for col, (frame, sat_id, color, mark_onset) in enumerate(columns):
        for row, channel in enumerate(PLOT_CHANNELS):
            ax = axes[row, col]
            label, scale = CHANNEL_DISPLAY[channel]

            ax.plot(frame["day"], frame[channel] * scale,
                    color=color, linewidth=1.0, solid_capstyle="round")

            # Onset marker only on the faulted column -- day 1750 has no
            # meaning for the healthy satellite.
            if mark_onset:
                ax.axvline(onset, color=ONSET_LINE, linestyle="--",
                           linewidth=1.8, alpha=0.75, zorder=1)

            if col == 0:
                ax.set_ylabel(label, fontsize=14, color=INK_MUTED)
            if row == 0:
                ax.set_title(satellite_label(sat_id), fontsize=17,
                             color=INK, pad=16, linespacing=1.4)
            if row == n - 1:
                ax.set_xlabel("Day", fontsize=15, color=INK_MUTED)

            ax.margins(x=0.01)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    # Annotate the onset line once, on the top-right panel.
    top_right = axes[0, 1]
    top_right.annotate(
        f"anomaly onset\nday {onset}",
        xy=(onset, top_right.get_ylim()[1]),
        xytext=(onset - 120, top_right.get_ylim()[1]),
        ha="right", va="top", fontsize=13, color=INK_MUTED, linespacing=1.3,
    )

    handles = [
        plt.Line2D([], [], color=COLOR_HEALTHY, linewidth=3, label=f"{LEFT_SAT} (healthy)"),
        plt.Line2D([], [], color=COLOR_FAULT, linewidth=3, label=f"{RIGHT_SAT} (slow drift)"),
        plt.Line2D([], [], color=ONSET_LINE, linewidth=1.8, linestyle="--", label="anomaly onset"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=15, bbox_to_anchor=(0.5, 0.006))

    fig.suptitle(
        "Raw telemetry check — healthy vs slow-drift clock, six channels",
        fontsize=23, color=INK, y=0.985,
    )
    fig.text(0.5, 0.955,
             "Each row shares a y-axis across both columns, so amplitudes are directly comparable.",
             ha="center", fontsize=14, color=INK_MUTED)

    fig.tight_layout(rect=(0, 0.028, 1, 0.945))
    fig.savefig(OUTPUT_PNG, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}  ({n} channels x 2 satellites)")


if __name__ == "__main__":
    main()
