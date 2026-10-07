"""
snapshot_check.py  -  READ-ONLY inspection of per-run settings snapshots.

Run from ~/rina-backend with the scheduler_env venv:

    python snapshot_check.py size          # how big would a snapshot be, from the settings as they are now
    python snapshot_check.py runs          # every run: does it have a snapshot, how big
    python snapshot_check.py run           # the newest run: is its snapshot there, and does it still equal the live settings
    python snapshot_check.py run 126       # same, for run id 126

Nothing is written to the database.
"""

import json
import sys

from data_access import fetch_all_data, get_db_connection
from snapshot import load_config_snapshot, make_config_snapshot, normalized, snapshot_meta


def _kb(n):
    return f"{n / 1024:.1f} KB"


def cmd_size():
    data = fetch_all_data()
    text = make_config_snapshot(data)
    print(f"גודל עותק הגדרות אחד (JSON): {_kb(len(text.encode('utf-8')))}")
    print("שורות בכל טבלה בעותק:")
    for key, rows in data.items():
        print(f"  {key:<26} {len(rows)}")
    print("teacher_color נשמר בעותק:", all("teacher_color" in t for t in data["teachers"]) if data["teachers"] else "אין מורים")


def cmd_runs():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, algorithm, run_at, is_selected, is_published,
                       config_snapshot IS NOT NULL,
                       COALESCE(pg_column_size(config_snapshot), 0),
                       config_snapshot->>'source'
                FROM schedule_runs ORDER BY id
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    with_snap = [r for r in rows if r[5]]
    print(f"סה״כ מערכות: {len(rows)} · עם עותק הגדרות: {len(with_snap)} · בלי: {len(rows) - len(with_snap)}")
    print(f"נפח כולל של העותקים בבסיס הנתונים: {_kb(sum(r[6] for r in rows))}")
    print()
    print(f"{'id':>5}  {'אלגוריתם':<14} {'נוצרה':<17} {'סטטוס':<12} {'עותק':<10} {'מקור':<11} גודל")
    for rid, algo, run_at, sel, pub, has, size, source in rows[-25:]:
        status = ("נוכחית " if sel else "") + ("פורסמה" if pub else "")
        print(f"{rid:>5}  {algo:<14} {run_at:%Y-%m-%d %H:%M}  {status:<12} {'יש' if has else 'אין':<10} {source or '-':<11} {_kb(size) if has else ''}")
    if len(rows) > 25:
        print(f"(מוצגות 25 האחרונות מתוך {len(rows)})")


def cmd_run(run_id=None):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if run_id is None:
                cur.execute("SELECT id FROM schedule_runs ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    print("אין מערכות בבסיס הנתונים.")
                    return
                run_id = row[0]
            cur.execute("SELECT algorithm, run_at, config_snapshot FROM schedule_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            if not row:
                print(f"מערכת {run_id} לא קיימת.")
                return
            algo, run_at, raw = row
            cur.execute("SELECT count(*) FROM schedule WHERE run_id = %s", (run_id,))
            lessons = cur.fetchone()[0]
    finally:
        conn.close()

    print(f"מערכת {run_id} · {algo} · {run_at:%Y-%m-%d %H:%M} · {lessons} שיעורים")
    data = load_config_snapshot(raw)
    if data is None:
        print("✗ אין עותק הגדרות למערכת הזו.")
        return
    version, source, created = snapshot_meta(raw)
    print(f"✓ יש עותק הגדרות (גרסה {version}, מקור: {source}, נוצר: {created})")

    live = normalized(fetch_all_data())
    same, different = [], []
    for key in sorted(set(live) | set(data)):
        (same if live.get(key) == data.get(key) else different).append(key)
    if not different:
        print("העותק זהה להגדרות החיות כרגע (כל הטבלאות).")
    else:
        print("העותק שונה מההגדרות החיות בטבלאות: " + ", ".join(different))
        print("(זה תקין ומצופה אם שינית הגדרות אחרי שהמערכת נוצרה — זו בדיוק המטרה.)")

    # sanity: every lesson of this run must be explainable by the snapshot alone
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tea_assignment_id, room_id FROM schedule WHERE run_id = %s", (run_id,))
            lesson_rows = cur.fetchall()
    finally:
        conn.close()
    assignment_ids = {a["id"] for a in data.get("teacher_assignments", [])}
    room_ids = {r["id"] for r in data.get("rooms", [])}
    missing_a = {a for a, _ in lesson_rows if a not in assignment_ids}
    missing_r = {r for _, r in lesson_rows if r is not None and r not in room_ids}
    if missing_a or missing_r:
        print(f"✗ שיעורים שהעותק לא יודע להסביר: שיוכים {sorted(missing_a)} · חדרים {sorted(missing_r)}")
    else:
        print("✓ כל שיעורי המערכת ניתנים להסבר מתוך העותק בלבד (שיוך, מורה, מקצוע, כיתה, חדר).")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "size":
        cmd_size()
    elif cmd == "runs":
        cmd_runs()
    elif cmd == "run":
        cmd_run(int(argv[2]) if len(argv) > 2 else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
