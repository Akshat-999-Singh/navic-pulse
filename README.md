# NavIC Pulse

Predictive health monitoring for NavIC atomic clocks: detect a degrading
rubidium frequency standard from ground-segment telemetry before it drifts out
of usable service, and estimate how long it has left.

The system trains an autoencoder on **healthy satellites only**, scores every
30-day telemetry window by reconstruction error, requires a detection to
persist before it is declared, and then projects remaining useful life from the
degradation curve.

## Why it is built this way

The design choices that matter are the ones made against a first instinct:

- **Trained on healthy units only.** Faulted satellites are excluded from the
  scaler fit and the training set, so their telemetry falls outside the healthy
  distribution by construction rather than by tuning.
- **A detection must persist.** A single-window alarm at a 1% false-alarm rate
  fires on a *healthy* clock by day 60–429 purely by chance. Requiring 60
  consecutive alarm windows is what makes the lead times real: healthy units
  then never fire at all.
- **Every fault is on an imported clock.** The fleet comparison contrasts
  imported against indigenous units, so a failing indigenous satellite would
  inflate the indigenous mean and destroy the comparison.
- **Remaining life is projected from signal level, not drift rate.** The iRAFS
  5e-13/day figure is a *free-running* specification, while `freq_offset_y`
  here is a steered ground-segment residual — the two are not commensurate, and
  90-day drift slopes are dominated by random-walk noise. `rafs_signal_level`
  is a genuine wear curve and leads the frequency fault by ~60 days.

## Results

| metric | value |
|---|---|
| ROC AUC (threshold-free) | 0.9348 |
| Precision / Recall / F1 (CUSUM, sustained) | 1.0000 / 0.8425 / 0.9145 |
| False positives across 20,468 healthy windows | 0 |
| Faults caught | 5 / 5 |
| Healthy units falsely flagged | 0 / 5 |
| Detection latency after onset | 5–25 days, on ramps of 100–300 days |

Recall of 0.84 is detection *latency*, not missed faults: the 60 windows a
detection must accumulate are scored as misses even though every fault is
eventually caught.

Remaining useful life, projected at detection + 30 days:

| satellite | profile | RUL (days) |
|---|---|---|
| IRNSS-1G | degraded | 50 ± 2 |
| IRNSS-1D | degraded | 53 ± 2 |
| IRNSS-1I | slow drift | 93 ± 3 |
| IRNSS-1A | noise increase | 151 ± 6 |
| IRNSS-1E | sudden jump | 208 ± 8 |
| 5 healthy units | — | no finite RUL |

## Running the pipeline

Requires `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
python config.py              # print every constant as a table (no side effects)

python generate_telemetry.py  # -> telemetry.csv          (29,000 rows, 10 satellites x 2,900 days)
python plot_raw.py            # -> demo/raw_check.png     (optional sanity check)
python preprocess.py          # -> preprocessed.npz, window_metadata.csv, scaler.joblib
python train_model.py         # -> scores.csv, model.joblib
python detect.py              # -> detections.csv
python rul.py                 # -> rul.json
python validate.py            # -> summary.json
python make_plots.py          # -> demo/*.png            (run last)
```

Every stage is deterministic given `RANDOM_SEED` in `config.py`, so the
generated CSVs are reproducible and are not tracked in git.

Run `make_plots.py` last: it and `train_model.py` both write
`demo/score_histogram.png` from the same scores, and whichever runs later wins.

## Layout

| file | role |
|---|---|
| `config.py` | every constant, plus the ISRO iRAFS reference specification |
| `generate_telemetry.py` | synthetic telemetry with four injected failure signatures |
| `preprocess.py` | healthy-only scaler fit, 30-day windows, 150 features |
| `train_model.py` | MLP autoencoder 150 → 64 → 12 → 64 → 150, per-window MSE |
| `detect.py` | threshold, severity bands, CUSUM with empirical (k, h) |
| `rul.py` | signal-level decline fit and remaining-life projection |
| `validate.py` | precision / recall / F1, lead times, `summary.json` |
| `make_plots.py` | the three presentation figures |
| `plot_raw.py` | six-channel raw telemetry check |

## Known limitations

These are real and worth stating rather than discovering in a demo:

- **The dataset is synthetic.** Failure magnitudes come from the published
  iRAFS record; everything else is modelled.
- **The RUL error bars are model-fit precision, not accuracy.** The generated
  signal decline plateaus near 0.91 and never actually reaches the 0.85 failure
  limit, so every RUL extrapolates a decline that stops. The figure answers
  "if the current rate continued, when would it cross" — the honest question a
  linear projection can answer.
- **`CUSUM_PERSISTENCE = 60` is fitted to five healthy satellites.** It is the
  point where healthy units go quiet on *this* dataset, which is demo-grade
  calibration, not an operationally justified constant.
- **CUSUM barely beats the plain threshold.** At matched false-alarm rates it
  wins by 5 days on `sudden_jump` and 11 on `noise_increase`, and loses by 2
  elsewhere. The accumulation helps least where per-window scores are already
  separable.
- **Severity bands are effectively satellite-level.** The 60th-percentile split
  lands on the degraded/non-degraded boundary, so "Critical" means "this is a
  degraded unit" rather than grading severity within one satellite's history.
- **The secondary RUL indicator does not work.** `steering_correction_ns` is
  too noisy at a 90-day window to fit a trend; it is scaffolding, not a result.
