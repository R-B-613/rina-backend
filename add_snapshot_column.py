"""
add_snapshot_column.py  -  STAGE A: adds the config_snapshot column to schedule_runs.

Run once from ~/rina-backend (scheduler_env venv):

    python add_snapshot_column.py

Safe to run again: it only adds the column if it is missing, and never touches
existing rows (the new column is NULL for them). Existing code ignores it.

If it stops with "must be owner of table schedule_runs", run the same statement
as the database owner instead:

    sudo -u postgres psql timetable_db -c "ALTER TABLE schedule_runs ADD COLUMN IF NOT EXISTS config_snapshot JSONB;"
"""

from data_access import get_db_connection

CHECK = """
    SELECT data_type FROM information_schema.columns
    WHERE table_name = 'schedule_runs' AND column_name = 'config_snapshot'
"""


def main():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CHECK)
            before = cur.fetchone()
            if before:
                print(f"העמודה config_snapshot כבר קיימת (סוג: {before[0]}). לא בוצע שינוי.")
            else:
                cur.execute("ALTER TABLE schedule_runs ADD COLUMN config_snapshot JSONB;")
                conn.commit()
                cur.execute(CHECK)
                after = cur.fetchone()
                print(f"נוספה העמודה config_snapshot (סוג: {after[0]})." if after else "משהו השתבש: העמודה לא נוצרה.")
            cur.execute("SELECT count(*) FROM schedule_runs")
            print(f"מערכות קיימות בטבלה: {cur.fetchone()[0]} (לא שונו; העותק שלהן ריק בינתיים).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
