"""NavIC Pulse inference runtime: numpy only, no training code.

The root pipeline (generate -> preprocess -> train -> detect -> rul) is the
source of truth. These modules port its inference half, reading the frozen
model from frontend/artifacts/ (written by the root export_weights.py).
"""
