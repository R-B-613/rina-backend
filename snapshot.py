"""
snapshot.py

Per-run configuration snapshot ("עותק הגדרות של המערכת").

WHY
---
A saved timetable (schedule_runs + schedule rows) stores only NUMBERS: which
timeslot, which teacher_assignment id, which room id. Everything that gives
those numbers a meaning - who the teacher is, what the subject / class / room
are called, the curriculum, the day structure, the constraints - lives in the
LIVE settings tables. So changing a live setting silently rewrites every
timetable in the history.

A snapshot fixes that. At generation time we store, inside the schedule_runs
row itself (column config_snapshot), an exact copy of the dict that
fetch_all_data() returned - the very same data the solvers worked on. Readers
resolve names from the snapshot and never from the live tables, so a saved
timetable is static data. Because the snapshot is a column of the run's own
row, deleting the run deletes its snapshot too: nothing to clean up.

STORED FORMAT (JSONB)
---------------------
{
  "version":    1,
  "source":     "generation" | "backfill",
  "created_at": "2026-10-07T18:00:00+00:00",
  "data":       <the fetch_all_data() dict>
}

config_snapshot IS NULL  ->  a legacy run saved before this feature existed.
Readers fall back to the live tables for such runs (until they are backfilled).

This module only builds / parses snapshots. It does not touch the database.
"""

import datetime as dt
import json
from decimal import Decimal

SNAPSHOT_VERSION = 1


def _json_default(obj):
    """
    Types psycopg2 can return that json cannot serialise natively.
    Decimal -> number (NOT a string: the scorer does arithmetic on weights),
    time/date/datetime -> ISO text ("08:00:00" - the format the week-structure
    code already parses).
    """
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (dt.time, dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"snapshot: cannot serialise {type(obj).__name__}")


def _dumps(obj):
    return json.dumps(obj, default=_json_default, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalized(data):
    """`data` after a JSON round trip - i.e. exactly what a snapshot of it would
    contain. Used to compare "live now" against a stored snapshot."""
    return json.loads(_dumps(data))


def make_config_snapshot(data, source="generation"):
    """
    Returns the snapshot as JSON TEXT.

    Text on purpose: it is an immutable copy taken at this very moment, so a
    solver that later modifies `data` in place cannot change what gets stored,
    and the same text can be saved with every run of one batch without
    serialising again.
    """
    return _dumps({
        "version": SNAPSHOT_VERSION,
        "source": source,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data": data,
    })


def safe_make_config_snapshot(data, source="generation"):
    """
    Same, but NEVER raises: a snapshot problem must not stop a 1-3 minute
    generation. On failure it logs loudly and returns None - the run is then
    saved without a snapshot (and snapshot_check.py will list it).
    """
    try:
        return make_config_snapshot(data, source)
    except Exception as exc:  # noqa: BLE001
        print(f"[snapshot] FAILED to build the config snapshot - run will be saved WITHOUT one: {exc!r}", flush=True)
        return None


def _parse(raw):
    if raw is None:
        return None
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def load_config_snapshot(raw):
    """
    `raw` = the value of schedule_runs.config_snapshot (psycopg2 gives a dict
    for jsonb; text is accepted too). Returns the inner fetch_all_data()-shaped
    dict, or None when there is no usable snapshot (NULL, unreadable, or a
    version this code does not know) - the caller then falls back to live data.
    """
    snap = _parse(raw)
    if snap is None or snap.get("version") != SNAPSHOT_VERSION:
        return None
    data = snap.get("data")
    return data if isinstance(data, dict) else None


def snapshot_meta(raw):
    """(version, source, created_at) of a stored snapshot, or None."""
    snap = _parse(raw)
    if snap is None:
        return None
    return snap.get("version"), snap.get("source"), snap.get("created_at")
