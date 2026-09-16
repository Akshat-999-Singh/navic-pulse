"""Parse and validate an uploaded telemetry CSV.

The first block of checks mirrors parseTelemetryCsv in src/data.js, so a file
the browser accepts is not rejected here for a different reason:

  * required columns day, satellite_id, clock_type, and at least one channel
  * every row has as many fields as the header
  * day is an integer
  * no duplicate (satellite_id, day)
  * day never decreases within a satellite

The browser only inspects a batch; this module feeds a model, so it adds the
checks scoring cannot do without, each reported the same way:

  * every model channel is present (the browser accepts any one channel)
  * model channel values are finite numbers
  * daily cadence (the model was trained on daily windows), with no gaps
  * each satellite has at least one full window of rows

Rows are 1-based file line numbers including the header, as in data.js.
"""

import csv
import math
import re

from .meta import CHANNELS, WINDOW

REQUIRED_COLUMNS = ["day", "satellite_id", "clock_type"]

# Same list as CHANNEL_COLUMNS in src/data.js.
CHANNEL_COLUMNS = [
    "freq_offset_y",
    "clock_bias_ns",
    "trb_core_temp_c",
    "rafs_signal_level",
    "bus_voltage_v",
    "steering_correction_ns",
]

CADENCES = ("daily", "weekly", "monthly")
MODEL_CADENCE = "daily"

INTEGER = re.compile(r"^[+-]?\d+$")


class IngestError(Exception):
    """Validation failed; `errors` is the full [{row, column, message}] list."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} validation error(s)")
        self.errors = errors


def _error(row, column, message):
    return {"row": row, "column": column, "message": message}


def _split(line):
    # One physical line is one record, as in data.js; csv handles the splitting.
    return [cell.strip() for cell in next(csv.reader([line]))]


def parse(data, cadence):
    """bytes -> {satellite_id: {"days", "lines", "clock_type", channel: [...]}}.

    Satellites keep first-appearance order. Raises IngestError listing every
    problem found.
    """
    errors = []
    if cadence not in CADENCES:
        errors.append(_error(None, "cadence",
                             f'Cadence "{cadence}" is not one of {", ".join(CADENCES)}.'))
    elif cadence != MODEL_CADENCE:
        errors.append(_error(None, "cadence",
                             f"The model was trained on daily telemetry; a {cadence} "
                             f"batch would be windowed over {WINDOW} {cadence} samples "
                             f"and scored as if they were {WINDOW} days."))

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise IngestError(errors + [_error(None, None, "File is not UTF-8 text.")])

    source = re.split(r"\r?\n", text)
    header_index = next((i for i, line in enumerate(source) if line.strip()), None)
    if header_index is None:
        raise IngestError(errors + [_error(1, None, "File is empty.")])

    header_row = header_index + 1
    columns = _split(source[header_index])
    for name in REQUIRED_COLUMNS:
        if name not in columns:
            errors.append(_error(header_row, name, f'Missing required column "{name}".'))
    if not any(name in columns for name in CHANNEL_COLUMNS):
        errors.append(_error(header_row, None, "Needs at least one telemetry channel: "
                                               f"{', '.join(CHANNEL_COLUMNS)}."))
    else:
        for name in CHANNELS:
            if name not in columns:
                errors.append(_error(header_row, name,
                                     f'Missing channel "{name}", which the model scores.'))
    if any(e["row"] == header_row for e in errors):
        raise IngestError(errors)

    position = {name: columns.index(name) for name in columns}
    satellites = {}
    last_day = {}
    seen = set()

    for i in range(header_index + 1, len(source)):
        if not source[i].strip():
            continue
        row = i + 1
        cells = _split(source[i])
        if len(cells) != len(columns):
            errors.append(_error(row, None,
                                 f"Expected {len(columns)} fields, found {len(cells)}."))
            continue

        satellite = cells[position["satellite_id"]]
        raw_day = cells[position["day"]]
        day = None
        if not INTEGER.match(raw_day):
            errors.append(_error(row, "day", f'Day "{raw_day}" is not an integer.'))
        else:
            day = int(raw_day)
            key = (satellite, day)
            if key in seen:
                errors.append(_error(row, "day", f"Duplicate day {day} for {satellite}."))
            seen.add(key)

            previous = last_day.get(satellite)
            if previous is not None and day < previous:
                errors.append(_error(row, "day",
                                     f"Day {day} comes after day {previous} for {satellite}; "
                                     "days must not decrease."))
            else:
                if previous is not None and day - previous > 1:
                    errors.append(_error(row, "day",
                                         f"Day {day} follows day {previous} for {satellite}; "
                                         "daily telemetry must not skip days."))
                last_day[satellite] = day

        values = {}
        for name in CHANNELS:
            raw = cells[position[name]]
            try:
                value = float(raw)
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                errors.append(_error(row, name, f'Value "{raw}" is not a finite number.'))
            values[name] = value

        record = satellites.setdefault(
            satellite, {"days": [], "lines": [], "clock_type": None,
                        **{name: [] for name in CHANNELS}})
        record["days"].append(day)
        record["lines"].append(row)
        # The last row's clock is the one in service at the end of the batch.
        record["clock_type"] = cells[position["clock_type"]]
        for name in CHANNELS:
            record[name].append(values[name])

    for satellite, record in satellites.items():
        if len(record["days"]) < WINDOW:
            errors.append(_error(record["lines"][-1], "day",
                                 f"{satellite} has {len(record['days'])} rows; at least "
                                 f"{WINDOW} are needed for one {WINDOW}-day window."))

    if errors:
        raise IngestError(errors)
    return satellites
