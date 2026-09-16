"""POST /api/analyze -- score an uploaded telemetry batch with the frozen model.

multipart/form-data fields:
  file         telemetry CSV (required)
  cadence      daily | weekly | monthly (required; only daily can be scored)
  name, launch_date, clock_type, orbit   optional source-satellite metadata
  synthetic    optional "true" / "false"; null in the response when omitted

200  the demo-run.json shape, "source": "computed", no "plot", measured "stages"
413  body over 4.5 MB (Vercel's request limit)
422  {"errors": [{row, column, message}]}
4xx/500  {"error": message}; no stack trace ever leaves the function
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True  # the function filesystem is ephemeral and read-only

MAX_BODY_BYTES = 4_500_000

# Imported at module scope so the .npz and meta load once per cold start. A
# failure here (missing artifact, layout mismatch) is kept and reported per
# request as JSON instead of crashing the function.
try:
    from _lib import ingest, multipart, pipeline
    LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001 -- reported, never raised
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def analyze(content_type, content_length, read_body):
    """-> (status, payload). Separate from the handler class so it can be tested."""
    if LOAD_ERROR is not None:
        return 500, {"error": f"Model runtime failed to load: {LOAD_ERROR}"}
    if content_length is None:
        return 411, {"error": "Content-Length is required."}
    if content_length > MAX_BODY_BYTES:
        return 413, {"error": f"Upload is {content_length / 1_000_000:.1f} MB; the limit is "
                              f"{MAX_BODY_BYTES / 1_000_000:.1f} MB. Split the batch by "
                              f"satellite or date range."}

    try:
        fields = multipart.parse(content_type, read_body(content_length))
    except multipart.FormError as exc:
        return 400, {"error": str(exc)}

    if "file" not in fields:
        return 422, {"errors": [{"row": None, "column": "file",
                                 "message": "No file was uploaded."}]}
    cadence = multipart.text(fields, "cadence")
    if cadence is None:
        return 422, {"errors": [{"row": None, "column": "cadence",
                                 "message": "Cadence is required."}]}
    metadata = {k: multipart.text(fields, k) for k in pipeline.METADATA_FIELDS}
    synthetic = multipart.text(fields, "synthetic")
    synthetic = None if synthetic is None else synthetic.lower() == "true"

    try:
        return 200, pipeline.run(fields["file"]["value"], cadence, metadata, synthetic)
    except ingest.IngestError as exc:
        return 422, {"errors": exc.errors}


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = self.headers.get("Content-Length")
            try:
                length = None if length is None else int(length)
            except ValueError:
                length = None
            status, payload = analyze(self.headers.get("Content-Type"), length, self.rfile.read)
            self._send(status, payload)
        except Exception as exc:  # noqa: BLE001 -- the catch-all is the point
            try:
                self._send(500, {"error": "Analysis failed.", "type": type(exc).__name__})
            except Exception:  # noqa: BLE001 -- connection already gone
                pass

    def do_GET(self):
        self._send(405, {"error": "Use POST with multipart/form-data."})

    def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
        pass
