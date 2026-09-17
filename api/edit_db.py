"""
api/edit_db.py
Manual-edit support: score a proposed schedule (preview) and persist an
edited run. Reuses the EXACT scorer the generation pipeline uses, so the
violations/score shown while editing match what the algorithms report.
"""

from psycopg2.extras import Json

from data_access import get_db_connection, fetch_all_data
from scoring_violations import score_genetic_schedule_with_violations


def build_lookups(data):
    """
    Build the lookups dict score_genetic_schedule_with_violations expects.
    Kept local so the edit path has no dependency on the solver internals;
    every key here is one the scorer actually reads.
    """
    requirement_by_id = {cr["id"]: cr for cr in data["curriculum_requirements"]}
    timeslot_by_id = {ts["id"]: ts for ts in data["timeslots"]}
    subject_by_id = {s["id"]: s for s in data["subjects"]}
    group_by_id = {g["id"]: g for g in data["student_groups"]}

    constraint_by_teacher_timeslot = {
        (c["teacher_id"], c["timeslot_id"]): c for c in data["teacher_constraints"]
    }
    preferences_by_teacher = {p["teacher_id"]: p for p in data["teacher_preferences"]}

    room_count_by_type = {}
    for r in data["rooms"]:
        rt = r.get("room_type")
        if rt is not None:
            room_count_by_type[rt] = room_count_by_type.get(rt, 0) + 1

    assignments_by_teacher = {}
    for ta in data["teacher_assignments"]:
        assignments_by_teacher.setdefault(ta["teacher_id"], []).append(ta["id"])

    return {
        "requirement_by_id": requirement_by_id,
        "timeslot_by_id": timeslot_by_id,
        "subject_by_id": subject_by_id,
        "group_by_id": group_by_id,
        "constraint_by_teacher_timeslot": constraint_by_teacher_timeslot,
        "preferences_by_teacher": preferences_by_teacher,
        "room_count_by_type": room_count_by_type,
        "assignments_by_teacher": assignments_by_teacher,
    }


def _entries_to_schedule(entries, data):
    """
    Convert [{tea_assignment_id, timeslot_id}, ...] -> {assignment_id: [ts,...]}.
    Every assignment id must be present (even empty) so the weekly-hours
    check can see assignments that ended up with zero lessons.
    """
    schedule = {ta["id"]: [] for ta in data["teacher_assignments"]}
    for e in entries:
        schedule.setdefault(e["tea_assignment_id"], []).append(e["timeslot_id"])
    return schedule


def preview_violations(entries):
    """Score a proposed schedule WITHOUT saving. Returns (score, violations)."""
    data = fetch_all_data()
    lookups = build_lookups(data)
    schedule = _entries_to_schedule(entries, data)
    return score_genetic_schedule_with_violations(schedule, data, lookups)


def save_run_entries(run_id, entries):
    """
    Replace ALL schedule rows of run_id with `entries`, then recompute and
    store score + violations on the run. One transaction. Returns
    (score, violations).
    entries: [{tea_assignment_id, timeslot_id, room_id}, ...]
    """
    data = fetch_all_data()
    lookups = build_lookups(data)
    schedule = _entries_to_schedule(entries, data)
    score, violations = score_genetic_schedule_with_violations(schedule, data, lookups)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM schedule WHERE run_id = %s;", (run_id,))
            for e in entries:
                cursor.execute(
                    """
                    INSERT INTO schedule (timeslot_id, tea_assignment_id, room_id, run_id)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (e["timeslot_id"], e["tea_assignment_id"], e.get("room_id"), run_id),
                )
            cursor.execute(
                "UPDATE schedule_runs SET score = %s, violations = %s WHERE id = %s;",
                (score, Json(violations), run_id),
            )
        conn.commit()
        return score, violations
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
