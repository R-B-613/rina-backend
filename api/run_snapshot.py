"""
api/run_snapshot.py
Reading a saved run THROUGH ITS OWN settings snapshot.

A saved run stores only NUMBERS (timeslot id, teacher_assignment id, room id). What they MEAN -
the teacher's name, the subject, the class, the room - is read from the run's snapshot
(schedule_runs.config_snapshot, built by snapshot.py), NOT from the live tables. So changing the
settings later can never alter a saved timetable.

A run with no snapshot (saved before snapshots existed and not yet backfilled) has
get_run_snapshot_data() == None; the caller then falls back to the live tables, as before.
"""

from psycopg2.extras import RealDictCursor

from data_access import get_db_connection
from snapshot import load_config_snapshot


def run_exists(run_id: int) -> bool:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM schedule_runs WHERE id = %s;", (run_id,))
            return cursor.fetchone() is not None
    finally:
        conn.close()


def get_run_snapshot_data(run_id: int):
    """The fetch_all_data()-shaped settings this run was generated from, or None if it has no snapshot."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT config_snapshot FROM schedule_runs WHERE id = %s;", (run_id,))
            row = cursor.fetchone()
    finally:
        conn.close()
    return load_config_snapshot(row[0]) if row else None


def entries_from_snapshot(run_id: int, data: dict, teacher_id: int = None):
    """The lessons of a run, in the same shape as the live-table query, resolved from `data` (its snapshot)."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, timeslot_id, tea_assignment_id, room_id FROM schedule WHERE run_id = %s ORDER BY id;",
                (run_id,),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    by_id = lambda key: {r["id"]: r for r in data.get(key, [])}
    timeslot, assignment, requirement = by_id("timeslots"), by_id("teacher_assignments"), by_id("curriculum_requirements")
    teacher, subject, group, room = by_id("teachers"), by_id("subjects"), by_id("student_groups"), by_id("rooms")

    entries, skipped = [], 0
    for row in rows:
        ts = timeslot.get(row["timeslot_id"])
        ta = assignment.get(row["tea_assignment_id"])
        req = requirement.get(ta["cur_requirement_id"]) if ta else None
        te = teacher.get(ta["teacher_id"]) if ta else None
        sub = subject.get(req["subject_id"]) if req else None
        grp = group.get(req["student_group_id"]) if req else None
        if any(x is None for x in (ts, ta, req, te, sub, grp)):
            skipped += 1                      # same as the old INNER JOINs: unresolvable rows are not shown
            continue
        if teacher_id is not None and te["id"] != teacher_id:
            continue
        rm = room.get(row["room_id"]) if row["room_id"] is not None else None
        entries.append({
            "id": row["id"],
            "timeslot_id": row["timeslot_id"],
            "tea_assignment_id": row["tea_assignment_id"],
            "room_id": row["room_id"],
            "day_of_week": ts["day_of_week"],
            "hour_of_day": ts["hour_of_day"],
            "teacher_id": te["id"],
            "teacher_first_name": te.get("first_name"),
            "teacher_last_name": te.get("last_name"),
            "teacher_color": te.get("teacher_color"),
            "subject_id": sub["id"],
            "subject_name": sub.get("subject_name"),
            "group_id": grp["id"],
            "group_name": grp.get("group_name"),
            "room_name": rm.get("room_name") if rm else None,
        })
    if skipped:
        print(f"[run_snapshot] run {run_id}: {skipped} lesson(s) not shown - not explained by the run's snapshot", flush=True)
    entries.sort(key=lambda e: (e["day_of_week"], e["hour_of_day"]))
    return entries
