"""
api/schedule_db.py
Read-only queries for the current + published timetable.
"""

from psycopg2.extras import RealDictCursor

from data_access import get_db_connection


def get_current_run():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, algorithm, score, run_at, is_published, violations, published_at
                FROM schedule_runs
                WHERE is_selected = true
                ORDER BY run_at DESC, id DESC
                LIMIT 1;
                """
            )
            return cursor.fetchone()
    finally:
        conn.close()


def get_published_run():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, algorithm, score, run_at, published_at
                FROM schedule_runs
                WHERE is_published = true
                ORDER BY run_at DESC, id DESC
                LIMIT 1;
                """
            )
            return cursor.fetchone()
    finally:
        conn.close()


def publish_run(run_id: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE schedule_runs SET is_published = false WHERE is_published = true;")
            cursor.execute("UPDATE schedule_runs SET is_published = true, published_at = NOW() WHERE id = %s;", (run_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_schedule_entries(run_id: int, teacher_id: int = None):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT
                    s.id                AS id,
                    s.timeslot_id       AS timeslot_id,
                    s.tea_assignment_id AS tea_assignment_id,
                    s.room_id           AS room_id,
                    t.day_of_week,
                    t.hour_of_day,
                    te.id            AS teacher_id,
                    te.first_name    AS teacher_first_name,
                    te.last_name     AS teacher_last_name,
                    sub.id           AS subject_id,
                    sub.subject_name,
                    sg.id            AS group_id,
                    sg.group_name,
                    r.room_name
                FROM schedule s
                JOIN timeslots t             ON s.timeslot_id = t.id
                JOIN teacher_assignments ta  ON s.tea_assignment_id = ta.id
                JOIN teachers te             ON ta.teacher_id = te.id
                JOIN curriculum_requirements cr ON ta.cur_requirement_id = cr.id
                JOIN subjects sub            ON cr.subject_id = sub.id
                JOIN student_groups sg       ON cr.student_group_id = sg.id
                LEFT JOIN rooms r            ON s.room_id = r.id
                WHERE s.run_id = %s
            """
            params = [run_id]
            if teacher_id is not None:
                query += " AND te.id = %s"
                params.append(teacher_id)
            query += " ORDER BY t.day_of_week, t.hour_of_day;"
            cursor.execute(query, params)
            return cursor.fetchall()
    finally:
        conn.close()
