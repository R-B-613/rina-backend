"""
data_access.py

Single place responsible for talking to the PostgreSQL database and
returning plain Python data (lists of dicts) - no algorithm logic lives
here. CSP, Genetic Algorithm and Hill Climbing all consume the SAME
output of fetch_all_data(), so they're guaranteed to work on identical data.

Credential reading re-uses the same approach as your original main.py
(reads ~/credentials.txt).
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json


def get_db_credentials(filepath="~/credentials.txt"):
    """
    Reads credentials.txt and returns a dict with host/dbname/user/password.
    """
    expanded_path = os.path.expanduser(filepath)

    creds = {
        "host": "localhost",
        "dbname": "",
        "user": "",
        "password": "",
    }

    try:
        with open(expanded_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Database Name:"):
                    creds["dbname"] = line.split(":", 1)[1].strip()
                elif line.startswith("Database User:"):
                    creds["user"] = line.split(":", 1)[1].strip()
                elif line.startswith("Database Password:"):
                    creds["password"] = line.split(":", 1)[1].strip()
                elif line.startswith("Host:"):
                    host_part = line.split(":", 1)[1].strip()
                    creds["host"] = host_part.split()[0].strip()

        return creds
    except Exception as e:
        print(f"Error reading the credentials file: {e}")
        return None


def get_db_connection():
    creds = get_db_credentials()
    if not creds or not creds["dbname"]:
        raise ValueError("Failed to extract connection details from credentials.txt.")

    return psycopg2.connect(
        host=creds["host"],
        database=creds["dbname"],
        user=creds["user"],
        password=creds["password"],
    )


def fetch_all_data():
    """
    Fetches every table needed by the scheduling algorithms and returns
    a single dict of plain Python lists/dicts, e.g.:

    {
        "timeslots": [ {id, day_of_week, hour_of_day}, ... ],
        "rooms": [ {id, room_name, capacity}, ... ],
        "subjects": [ {id, subject_name, required_room_id}, ... ],
        "student_groups": [ {id, group_name, student_count, home_room_id}, ... ],
        "teachers": [ {id, first_name, last_name, weekly_hours_quota, ...}, ... ],
        "curriculum_requirements": [ {id, subject_id, student_group_id, weekly_hours, sync_block_identity}, ... ],
        "teacher_assignments": [ {id, teacher_id, cur_requirement_id}, ... ],
        "teacher_constraints": [ {id, teacher_id, timeslot_id, weight, constraint_type, reason}, ... ],
        "teacher_preferences": [ {id, teacher_id, min_hours, max_hours, ...}, ... ],
    }

    NOTE: this does not do any algorithm logic - it is pure data fetching.
    """
    conn = get_db_connection()
    data = {}

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:

            cursor.execute("""
                SELECT id, day_of_week, hour_of_day
                FROM timeslots
                ORDER BY day_of_week, hour_of_day;
            """)
            data["timeslots"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, room_name, capacity, room_type
                FROM rooms
                ORDER BY id;
            """)
            data["rooms"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, subject_name, required_room_id, required_room_type
                FROM subjects
                ORDER BY id;
            """)
            data["subjects"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, group_name, student_count, home_room_id
                FROM student_groups
                ORDER BY id;
            """)
            data["student_groups"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, first_name, last_name, weekly_hours_quota,
                       is_admin, email
                FROM teachers
                ORDER BY id;
            """)
            data["teachers"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, subject_id, student_group_id, weekly_hours,
                       sync_block_identity
                FROM curriculum_requirements
                ORDER BY id;
            """)
            data["curriculum_requirements"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, teacher_id, cur_requirement_id
                FROM teacher_assignments
                ORDER BY id;
            """)
            data["teacher_assignments"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, teacher_id, timeslot_id, weight, constraint_type, reason
                FROM teacher_constraints
                ORDER BY teacher_id, timeslot_id;
            """)
            data["teacher_constraints"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, teacher_id, min_hours, max_hours,
                       preferred_consecutive, priority_early_finish,
                       priority_no_gaps, priority_free_day, priority_consecutive
                FROM teacher_preferences
                ORDER BY teacher_id;
            """)
            data["teacher_preferences"] = cursor.fetchall()

            cursor.execute("""
                SELECT grade_level, max_lessons_per_day
                FROM grade_schedule_limits
                ORDER BY grade_level;
            """)
            data["grade_schedule_limits"] = cursor.fetchall()

            cursor.execute("""
                SELECT id, constraint_type, subject_a_id, subject_b_id,
                       numeric_value, weight
                FROM pedagogical_constraints
                WHERE is_active = true
                ORDER BY id;
            """)
            data["pedagogical_constraints"] = cursor.fetchall()

    finally:
        conn.close()

    return data


def mark_run_as_selected(run_id: int):
    """
    Sets is_selected = true for the given schedule_runs.id. Intended to be
    called by the comparator AFTER all candidate runs have already been
    saved via save_schedule_run(), once the best one has been determined.

    Does NOT unset is_selected on any other rows - if you re-run the
    pipeline multiple times, older runs keep whatever is_selected value
    they already had. This is intentional: is_selected marks "this run
    was the winner OF ITS OWN comparison batch", not "this is the single
    best run that has ever existed across all time".
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # First clear the flag everywhere — there must be exactly ONE
            # current schedule at a time. Without this, every batch's winner
            # stays "selected", so multiple runs show as "current".
            cursor.execute("UPDATE schedule_runs SET is_selected = false WHERE is_selected = true;")
            cursor.execute(
                "UPDATE schedule_runs SET is_selected = true WHERE id = %s;",
                (run_id,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_schedule_run(algorithm: str, score: float, schedule_entries: list, violations: list = None):
    """
    Inserts a new row into schedule_runs (with the given algorithm name and
    score), then inserts all schedule_entries into the schedule table,
    tagged with the new run's id.

    schedule_entries: list of dicts, each shaped like:
        {"timeslot_id": ..., "tea_assignment_id": ..., "room_id": ...}

    Returns the new run_id.

    NOTE: this is intentionally NOT called from inside the CSP/GA/Hill
    Climbing solvers themselves - it's meant to be called from main.py or
    the comparator, after a solver has returned its result.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO schedule_runs (algorithm, score, violations)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (algorithm, score, Json(violations) if violations is not None else None),
            )
            run_id = cursor.fetchone()[0]

            for entry in schedule_entries:
                cursor.execute(
                    """
                    INSERT INTO schedule (timeslot_id, tea_assignment_id, room_id, run_id)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (entry["timeslot_id"], entry["tea_assignment_id"], entry["room_id"], run_id),
                )

        conn.commit()
        return run_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
