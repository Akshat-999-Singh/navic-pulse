"""model_meta.json, loaded once per cold start.

Every constant the runtime needs (channel order, window length, threshold,
CUSUM parameters, persistence) comes from this file, never from a copy in code,
so the runtime cannot drift from the model it serves.
"""

import json
import os

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "artifacts")
META_PATH = os.path.join(ARTIFACTS, "model_meta.json")
WEIGHTS_PATH = os.path.join(ARTIFACTS, "model_weights.npz")

with open(META_PATH, encoding="utf-8") as _handle:
    META = json.load(_handle)

CHANNELS = list(META["features"]["channels"])
WINDOW = int(META["features"]["window_days"])
FEATURE_NAMES = list(META["features"]["names"])
LAYER_SIZES = [int(n) for n in META["layer_sizes"]]

DETECTION = META["detection"]
THRESHOLD = float(DETECTION["threshold"])
SEVERITY_CRITICAL = float(DETECTION["severity_critical"])
CUSUM_MU = float(DETECTION["cusum"]["mu"])
CUSUM_K = float(DETECTION["cusum"]["k"])
CUSUM_H = float(DETECTION["cusum"]["h"])
PERSISTENCE = int(DETECTION["persistence_windows"])

# Optional: fixed OOL limits shipped with the model, {channel: {low, high}}.
# Absent from the current export, in which case baseline.py calibrates from the
# batch itself.
BASELINE_OOL = META.get("baseline_ool")
