"""
NavIC Pulse -- central configuration.

Every tunable constant for the project lives here. Nothing else imports
hard-coded numbers.

Reference: ISRO iRAFS (Indian Rubidium Atomic Frequency Standard) spec figures
and the four measured on-orbit failure magnitudes.
"""

# ===========================================================================
# REFERENCE SPECIFICATION -- ISRO iRAFS
# Source: Bandi & Arora, ICG-14 (2019), published iRAFS documentation.
# Verbatim reference figures. Do not delete; the constants below are derived
# from these.
# ---------------------------------------------------------------------------
#
# Frequency drift target ....... < 5e-13 per day
# Initial frequency accuracy ... +/- 1e-9
#
# Allan deviation (drift removed):
#       1 s ..... 5.0e-12
#      10 s ..... 1.5e-12
#     100 s ..... 5.0e-13
#    1000 s ..... 1.5e-13
#   10000 s ..... 5.0e-14
#
# Frequency vs temperature ..... <= +/- 1e-13 per degree C,
#                                over -5 to +15 degrees C
#
# Four measured RAFS failure signatures:
#   1. sudden transition, weak jump ....... 7.2e-13
#   2. sudden transition, jump ............ 5.1e-12
#   3. slow transition, large jump ........ 2.5e-11
#   4. sudden increase in noise ........... no published magnitude
#
# Physics:
#   bias(t) = bias0 + y0 * t + 0.5 * D * t**2
#   1 ns clock error == 30 cm position error
#
# Real telemetry channel names (as used in ISRO housekeeping telemetry):
#   Clock ON/OFF, Light, Signal, TRB core package, TCB-EPC, Lock/Unlock
#
# ===========================================================================

# ---------------------------------------------------------------------------
# Simulation horizon
# ---------------------------------------------------------------------------
N_DAYS = 2900          # one timestep == one day (~7.9 years of telemetry)
RANDOM_SEED = 42       # fixed for reproducible runs

# ---------------------------------------------------------------------------
# Model / detector hyper-parameters
# ---------------------------------------------------------------------------
WINDOW = 30            # sliding window length, in days
BOTTLENECK = 12        # autoencoder latent dimension
K_SIGMA = 3.0          # residual threshold = mean + K_SIGMA * std
CUSUM_PERSISTENCE = 60 # consecutive alarm windows required to declare a
                       # detection. A single-window alarm at a 1% false
                       # alarm rate fires on healthy clocks by day 60-429
                       # purely by chance; healthy units alarm in short
                       # episodes (longest run 6-53 windows) while faults
                       # run 595-2547. At 60 no healthy unit fires at all.

# ---------------------------------------------------------------------------
# ISRO iRAFS specification
# ---------------------------------------------------------------------------
NOMINAL_DRIFT_PER_DAY = 5e-13   # fractional frequency drift per day
TEMP_COEFF = 1e-13              # fractional frequency shift per degree C
SPEC_LIMIT_DRIFT = 5e-13        # drift above this is out of spec

FREQ_ACCURACY_INITIAL = 1e-9    # initial frequency accuracy, +/- this value
TEMP_RANGE_C = (-5, 15)         # degrees C over which TEMP_COEFF is specified

# Allan deviation, drift removed: averaging time tau (s) -> sigma_y(tau)
ALLAN_DEV = {
    1: 5e-12,
    10: 1.5e-12,
    100: 5e-13,
    1000: 1.5e-13,
    10000: 5e-14,
}

# ---------------------------------------------------------------------------
# Measured failure magnitudes (four observed anomaly types)
# ---------------------------------------------------------------------------
JUMP_SUDDEN_WEAK = 7.2e-13      # sudden frequency jump, weak
JUMP_SUDDEN = 5.1e-12           # sudden frequency jump
DRIFT_SLOW_LARGE = 2.5e-11      # slow drift, large accumulated offset
NOISE_INCREASE_FACTOR = 4.0     # noise-increase type: sigma multiplier.
                                # PLACEHOLDER -- no published magnitude for
                                # this signature in the iRAFS reference.

# ---------------------------------------------------------------------------
# Telemetry channels
#
# Provenance -- these five are NOT all real iRAFS housekeeping telemetry:
#   rafs_signal_level ... maps to ISRO's "Signal"
#   trb_core_temp_c ..... maps to ISRO's "TRB core package"
#   freq_offset_y ....... ground-derived, not onboard housekeeping; comes from
#                         clock offset estimation in the ground segment, which
#                         is where this system runs
#   bus_voltage_v ....... generic spacecraft power-bus channel, not from the
#                         iRAFS list
#   steering_correction_ns  ground-derived: the clock offset removed by each
#                         correction upload. Correction *demand* is a
#                         degradation signal (healthy ~170 ns/upload, degraded
#                         ~2200-3400). The companion steering_reset flag stays
#                         in telemetry.csv for reference but is NOT a model
#                         input -- at a 50 ns threshold it is true ~90% of days
#                         for every satellite, and a near-constant boolean
#                         gives an autoencoder nothing.
#
# Two further columns are written to telemetry.csv for plotting and reference
# but are NOT model inputs:
#   clock_bias_ns ....... ground-derived, same origin as freq_offset_y. Dropped
#                         because it correlates with steering_correction_ns at
#                         r = 1.000 (identical on 77-99% of days): once a
#                         correction fires the recorded bias IS the correction
#                         applied, so the two channels carry one signal.
#   steering_reset ...... near-constant boolean, see above.
#
# The binary iRAFS flags (Clock ON/OFF, Lock/Unlock) are deliberately excluded:
# they are state indicators, not continuous signals, and an MLP autoencoder over
# 30-day windows learns nothing from a channel that is a constant 1.
# ---------------------------------------------------------------------------
CHANNELS = [
    "freq_offset_y",
    "trb_core_temp_c",
    "rafs_signal_level",
    "bus_voltage_v",
    "steering_correction_ns",
]

# ---------------------------------------------------------------------------
# Satellite roster
#   clock_type : 'imported' | 'indigenous'
#   profile    : 'healthy' | 'slow_drift' | 'sudden_jump' | 'noise_increase'
#                | 'degraded'
#   anomaly_start_day : day the fault begins, None for healthy units
# ---------------------------------------------------------------------------
SATELLITES = [
    # -- healthy: 3 imported, 2 indigenous ---------------------------------
    {"id": "IRNSS-1B", "clock_type": "imported",   "profile": "healthy",        "anomaly_start_day": None},
    {"id": "IRNSS-1C", "clock_type": "imported",   "profile": "healthy",        "anomaly_start_day": None},
    {"id": "IRNSS-1F", "clock_type": "imported",   "profile": "healthy",        "anomaly_start_day": None},
    {"id": "NVS-01",   "clock_type": "indigenous", "profile": "healthy",        "anomaly_start_day": None},
    {"id": "NVS-02",   "clock_type": "indigenous", "profile": "healthy",        "anomaly_start_day": None},
    # -- distinct failure profiles -----------------------------------------
    # Every failing unit is imported by design: the Clock Trust Score compares
    # the mean reconstruction error of the imported population against the
    # indigenous one. A failing indigenous unit would inflate the indigenous
    # mean and destroy that comparison, which is the core result of the
    # project. Both indigenous units therefore stay clean.
    {"id": "IRNSS-1I", "clock_type": "imported",   "profile": "slow_drift",     "anomaly_start_day": 1750},
    {"id": "IRNSS-1E", "clock_type": "imported",   "profile": "sudden_jump",    "anomaly_start_day": 2100},
    # IRNSS-1A lost all three of its atomic clocks on orbit in 2016; carrying a
    # failure profile here matches the real record.
    {"id": "IRNSS-1A", "clock_type": "imported",   "profile": "noise_increase", "anomaly_start_day": 2350},
    # -- long-degraded -----------------------------------------------------
    {"id": "IRNSS-1G", "clock_type": "imported",   "profile": "degraded",       "anomaly_start_day": 400},
    {"id": "IRNSS-1D", "clock_type": "imported",   "profile": "degraded",       "anomaly_start_day": 900},
]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def _rule(width, char="-"):
    return char * width


def print_config():
    """Print every constant in this module as readable tables."""
    scalars = [
        ("N_DAYS", N_DAYS, "days of telemetry (1 step = 1 day)"),
        ("RANDOM_SEED", RANDOM_SEED, "reproducibility seed"),
        ("WINDOW", WINDOW, "sliding window length (days)"),
        ("BOTTLENECK", BOTTLENECK, "autoencoder latent dimension"),
        ("K_SIGMA", K_SIGMA, "threshold = mean + k*std"),
        ("NOMINAL_DRIFT_PER_DAY", NOMINAL_DRIFT_PER_DAY, "iRAFS spec: drift/day"),
        ("TEMP_COEFF", TEMP_COEFF, "iRAFS spec: per degree C"),
        ("SPEC_LIMIT_DRIFT", SPEC_LIMIT_DRIFT, "iRAFS spec: drift limit"),
        ("FREQ_ACCURACY_INITIAL", FREQ_ACCURACY_INITIAL, "iRAFS spec: initial accuracy"),
        ("TEMP_RANGE_C", TEMP_RANGE_C, "iRAFS spec: valid temp range (C)"),
        ("JUMP_SUDDEN_WEAK", JUMP_SUDDEN_WEAK, "measured: sudden jump, weak"),
        ("JUMP_SUDDEN", JUMP_SUDDEN, "measured: sudden jump"),
        ("DRIFT_SLOW_LARGE", DRIFT_SLOW_LARGE, "measured: slow drift, large"),
        ("NOISE_INCREASE_FACTOR", NOISE_INCREASE_FACTOR, "measured: sigma multiplier"),
    ]

    width = 78
    print()
    print("NavIC Pulse -- configuration")
    print(_rule(width, "="))
    print(f"{'CONSTANT':<24} {'VALUE':>14}  {'MEANING'}")
    print(_rule(width))
    for name, value, note in scalars:
        shown = f"{value:.3g}" if isinstance(value, float) else str(value)
        print(f"{name:<24} {shown:>14}  {note}")

    print()
    print("ALLAN_DEV (drift removed)")
    print(_rule(width))
    print("  " + "  ".join(f"{tau:>7}s" for tau in ALLAN_DEV))
    print("  " + "  ".join(f"{sig:>8.1e}" for sig in ALLAN_DEV.values()))

    print()
    print(f"CHANNELS ({len(CHANNELS)})")
    print(_rule(width))
    for i, ch in enumerate(CHANNELS):
        print(f"  [{i}] {ch}")

    print()
    print(f"SATELLITES ({len(SATELLITES)})")
    print(_rule(width))
    print(f"{'ID':<12} {'CLOCK':<12} {'PROFILE':<16} {'ANOMALY START':>13}")
    print(_rule(width))
    for sat in SATELLITES:
        start = sat["anomaly_start_day"]
        start = "--" if start is None else f"day {start}"
        print(f"{sat['id']:<12} {sat['clock_type']:<12} {sat['profile']:<16} {start:>13}")
    print(_rule(width))

    healthy = sum(1 for s in SATELLITES if s["profile"] == "healthy")
    indigenous = sum(1 for s in SATELLITES if s["clock_type"] == "indigenous")
    print(f"{healthy} healthy / {len(SATELLITES) - healthy} faulted   |   "
          f"{indigenous} indigenous / {len(SATELLITES) - indigenous} imported")
    print()


if __name__ == "__main__":
    print_config()
