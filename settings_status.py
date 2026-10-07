"""
api/settings_status.py
"Have the settings changed since this timetable was generated?"

Every saved run carries a snapshot of the settings it was generated from (schedule_runs.config_snapshot).
This compares TODAY's settings with the snapshot of the CURRENT run and of the PUBLISHED run, so the
schedule screen can warn the admin that the settings no longer match the timetable that is shown.

Read-only. Nothing is stored and no timetable is touched.
"""

import json

from data_access import fetch_all_data
from snapshot import normalized
from api.run_snapshot import get_run_snapshot_data
from api.schedule_db import get_current_run, get_published_run

# Fields that are NOT scheduling settings: changing them does not change the meaning of any timetable,
# so they must not trigger the warning. (Add a field here to stop it from counting as a change.)
IGNORED_FIELDS = {
    "teachers": {"email", "is_admin", "teacher_color"},
}


def _canonical(table, rows):
    """A table as a sorted list of canonical JSON strings - independent of row order and key order."""
    ignore = IGNORED_FIELDS.get(table, ())
    out = []
    for row in rows or []:
        if isinstance(row, dict):
            row = {k: v for k, v in row.items() if k not in ignore}
        out.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    out.sort()
    return out


def differing_tables(snapshot_data: dict, live_data: dict):
    """Names of the settings tables whose content differs (only tables the snapshot has are compared)."""
    return [t for t in snapshot_data if _canonical(t, snapshot_data[t]) != _canonical(t, live_data.get(t))]


def settings_status():
    """
    {
      "current":   {"run_id": 238, "changed": true,  "tables": ["teacher_assignments"]}  | null (no current run),
      "published": {"run_id": 230, "changed": false, "tables": []}                      | null (nothing published),
    }
    changed = null  ->  that run has no snapshot, so there is nothing to compare against.
    """
    live = None
    verdict = {}          # run_id -> list of differing tables, or None (no snapshot)
    result = {}
    for name, run in (("current", get_current_run()), ("published", get_published_run())):
        if run is None:
            result[name] = None
            continue
        rid = run["id"]
        if rid not in verdict:
            snap = get_run_snapshot_data(rid)
            if snap is None:
                verdict[rid] = None
            else:
                if live is None:
                    live = normalized(fetch_all_data())
                verdict[rid] = differing_tables(snap, live)
        diff = verdict[rid]
        result[name] = {"run_id": rid, "changed": None if diff is None else bool(diff), "tables": diff or []}
    return result
