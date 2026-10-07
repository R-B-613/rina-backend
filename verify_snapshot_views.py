"""
verify_snapshot_views.py  -  READ-ONLY check of STAGE D.

For every run that has a snapshot it builds the lesson list two ways:
  OLD = the live tables (what the screens used to show)
  NEW = the run's own snapshot (what the screens show now)
and compares them.

    python verify_snapshot_views.py            # all runs
    python verify_snapshot_views.py 238        # one run

Right after the backfill every run must be IDENTICAL (the snapshot equals today's settings),
which proves the new reader returns exactly what the old one did.
Later, a run whose settings were changed since it was generated shows up here as DIFFERENT -
that is the whole point: the screens no longer follow the live settings.
"""

import sys

from data_access import get_db_connection
from api.schedule_db import _get_schedule_entries_live, get_schedule_entries

# fields both readers produce (teacher_color exists only in the snapshot reader)
FIELDS = ("id", "timeslot_id", "tea_assignment_id", "room_id", "day_of_week", "hour_of_day",
          "teacher_id", "teacher_first_name", "teacher_last_name",
          "subject_id", "subject_name", "group_id", "group_name", "room_name")


def _key(rows):
    return {r["id"]: tuple(r[f] for f in FIELDS) for r in rows}


def main(argv):
    only = int(argv[1]) if len(argv) > 1 else None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if only is None:
                cur.execute("SELECT id FROM schedule_runs WHERE config_snapshot IS NOT NULL ORDER BY id")
            else:
                cur.execute("SELECT id FROM schedule_runs WHERE id = %s AND config_snapshot IS NOT NULL", (only,))
            run_ids = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    if not run_ids:
        print("אין מערכות עם עותק הגדרות לבדיקה.")
        return 1

    same = 0
    different = []
    for rid in run_ids:
        old, new = _key(_get_schedule_entries_live(rid)), _key(get_schedule_entries(rid))
        if old == new:
            same += 1
        else:
            different.append((rid, old, new))

    print(f"נבדקו {len(run_ids)} מערכות: זהות {same} · שונות {len(different)}")
    for rid, old, new in different[:10]:
        only_old, only_new = set(old) - set(new), set(new) - set(old)
        changed = [i for i in set(old) & set(new) if old[i] != new[i]]
        print(f"  מערכת {rid}: שיעורים שונים {len(changed)} · רק בישן {len(only_old)} · רק בחדש {len(only_new)}")
        for i in changed[:3]:
            a, b = old[i], new[i]
            diff = [f"{f}: {x!r} → {y!r}" for f, x, y in zip(FIELDS, a, b) if x != y]
            print(f"      שיעור {i}: " + "; ".join(diff))
    if len(different) > 10:
        print(f"  (ועוד {len(different) - 10})")
    return 0 if not different else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
