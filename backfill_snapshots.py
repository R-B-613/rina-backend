"""
backfill_snapshots.py  -  STAGE C: give every run saved BEFORE snapshots existed an
(approximate) settings snapshot.

    python backfill_snapshots.py                  # dry run: shows what would happen, changes NOTHING
    python backfill_snapshots.py --apply          # writes
    python backfill_snapshots.py --apply --force  # writes even though the live settings differ from the newest generated run

IMPORTANT - what a backfilled snapshot is
-----------------------------------------
For an old run the real generation-time settings were never saved, so they cannot be
recovered. The best available is "the settings as they are RIGHT NOW". From this moment the
run stops changing when settings change (that is the point) - but it freezes TODAY's
teacher assignments, names and rooms, not the ones of the day it was generated.
Such snapshots are marked source = "backfill".

So before applying, the live settings must be in the state you want the history to show.
The script helps: it compares the live settings with the snapshot of the newest run that
was generated with snapshots (a trustworthy picture of recent reality) and lists every
difference - most importantly teacher assignments that were swapped since. Fix those in
the UI first, run again until "no differences", then --apply.

Safe to run again: only runs whose snapshot is NULL are ever touched.
"""

import sys

from data_access import fetch_all_data, get_db_connection
from snapshot import load_config_snapshot, make_config_snapshot, normalized, snapshot_meta


def _by_id(rows):
    return {r["id"]: r for r in rows}


def _teacher_name(teachers, tid):
    t = teachers.get(tid)
    return f"{t['first_name']} {t['last_name']}" if t else f"מורה {tid}"


def _assignment_label(data, ta):
    """'כיתה ג1 · מתמטיקה' for an assignment dict, using the given settings dict."""
    req = _by_id(data["curriculum_requirements"]).get(ta["cur_requirement_id"])
    if not req:
        return f"שיוך {ta['id']}"
    group = _by_id(data["student_groups"]).get(req["student_group_id"], {}).get("group_name", "?")
    subject = _by_id(data["subjects"]).get(req["subject_id"], {}).get("subject_name", "?")
    return f"{group} · {subject}"


def describe_differences(ref, live):
    """Human-readable lines describing how `live` differs from `ref` (both normalized dicts)."""
    lines = []

    ref_ta, live_ta = _by_id(ref["teacher_assignments"]), _by_id(live["teacher_assignments"])
    ref_teachers, live_teachers = _by_id(ref["teachers"]), _by_id(live["teachers"])
    for aid in sorted(set(ref_ta) | set(live_ta)):
        a, b = ref_ta.get(aid), live_ta.get(aid)
        if a and b:
            if a["teacher_id"] != b["teacher_id"]:
                lines.append(
                    f"שיוך {aid} ({_assignment_label(live, b)}): היה {_teacher_name(ref_teachers, a['teacher_id'])}, "
                    f"עכשיו {_teacher_name(live_teachers, b['teacher_id'])}"
                )
            elif a["cur_requirement_id"] != b["cur_requirement_id"]:
                lines.append(f"שיוך {aid}: הוצמד לשורה אחרת בתוכנית הלימודים")
        elif b:
            lines.append(f"שיוך {aid} ({_assignment_label(live, b)}): נוסף ({_teacher_name(live_teachers, b['teacher_id'])})")
        else:
            lines.append(f"שיוך {aid} ({_assignment_label(ref, a)}): הוסר (היה {_teacher_name(ref_teachers, a['teacher_id'])})")

    other = [k for k in sorted(set(ref) | set(live)) if k != "teacher_assignments" and ref.get(k) != live.get(k)]
    if other:
        lines.append("טבלאות נוספות ששונו: " + ", ".join(other))
    return lines


def main(argv):
    apply_ = "--apply" in argv
    force = "--force" in argv

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM schedule_runs WHERE config_snapshot IS NULL ORDER BY id")
            legacy = [r[0] for r in cur.fetchall()]
            cur.execute("""
                SELECT id, run_at, config_snapshot FROM schedule_runs
                WHERE config_snapshot IS NOT NULL AND config_snapshot->>'source' = 'generation'
                ORDER BY id DESC LIMIT 1
            """)
            ref_row = cur.fetchone()
    finally:
        conn.close()

    print(f"מערכות בלי עותק הגדרות: {len(legacy)}")
    if not legacy:
        print("אין מה למלא.")
        return 0

    differences = []
    if ref_row:
        ref_id, ref_at, ref_raw = ref_row
        ref = load_config_snapshot(ref_raw)
        live = normalized(fetch_all_data())
        print(f"השוואה להגדרות החיות מול העותק של מערכת {ref_id} (נוצרה {ref_at:%Y-%m-%d %H:%M}):")
        differences = describe_differences(ref, live) if ref else ["העותק של מערכת הייחוס לא נקרא"]
        if differences:
            for line in differences:
                print("  •", line)
        else:
            print("  אין הבדלים — ההגדרות החיות זהות לעותק של המערכת האחרונה שנוצרה.")
    else:
        print("אין עדיין מערכת שנוצרה עם עותק, ולכן אין עם מה להשוות.")

    if not apply_:
        print("\nהרצת בדיקה בלבד — לא שונה דבר.")
        if differences:
            print("יש הבדלים: אם הם לא מכוונים, לתקן בממשק ולהריץ שוב. אם הם מכוונים: --apply --force")
        else:
            print("אפשר להריץ עם --apply.")
        return 0

    if differences and not force:
        print("\nלא בוצע שינוי: יש הבדלים. לתקן ולהריץ שוב, או להריץ עם --force אם ההגדרות החיות הן המצב הנכון.")
        return 1

    snapshot_text = make_config_snapshot(fetch_all_data(), source="backfill")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE schedule_runs SET config_snapshot = %s::jsonb WHERE config_snapshot IS NULL",
                (snapshot_text,),
            )
            updated = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(f"\nמולאו {updated} מערכות (מקור: backfill).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
