"""
scoring_violations.py

Companion to genetic/solver.py's _score_schedule. Computes the SAME total
penalty, but also returns a human-readable list of exactly what was
penalised and by how much - so the admin can see WHY a score is what it is.

It reads a Genetic-style schedule: {assignment_id: [timeslot_id, ...]}.

Returns: (total_penalty, violations)
  violations = [ {"type": str, "detail": str (Hebrew), "penalty": number,
                  "severity": "hard" | "soft"}, ... ]

Kept in its own module so the fast per-generation scorer in the GA loop is
untouched; this heavier version runs ONCE, on the final best schedule.
"""

from scoring_config import (
    HARD_CONSTRAINT_PENALTY,
    TEACHER_HARD_CONSTRAINT_PENALTY,
    SOFT_CONSTRAINT_WEIGHT_MULTIPLIER,
    OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR,
    SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR,
    PREFERENCE_WEIGHTS,
)

DAY_NAMES = {1: "ראשון", 2: "שני", 3: "שלישי", 4: "רביעי", 5: "חמישי", 6: "שישי"}

# Grade letter -> grade number. Young grades = 1-3 (א/ב/ג).
GRADE_LETTERS = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6}


def grade_of(group_name):
    """Derive grade number from a class name like 'כיתה א1' -> 1. None if unknown."""
    if not group_name:
        return None
    rest = group_name.replace("כיתה", "").strip()
    return GRADE_LETTERS.get(rest[0]) if rest else None

def _parse_clock_to_min(s):
    """'09:00' / '09:00:00' / time(9,0) -> minutes since midnight. None on failure."""
    try:
        parts = str(s).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, AttributeError, IndexError):
        return None


LESSON_MINUTES = 45  # a teaching period; must match the School-Settings preview


def _periods_until(start_min, end_min, breaks):
    """How many 45-min lessons fit between start and end, inserting the
    configured breaks between lessons — the SAME walk the frontend uses."""
    if start_min is None or end_min is None or end_min <= start_min:
        return None
    by_after = {}
    for b in (breaks or []):
        try:
            by_after[int(b["after_lesson"])] = int(b["duration_minutes"])
        except (KeyError, TypeError, ValueError):
            continue
    t, n = start_min, 0
    while t + LESSON_MINUTES <= end_min:
        n += 1
        t += LESSON_MINUTES
        dur = by_after.get(n)
        if dur is not None and t + dur + LESSON_MINUTES <= end_min:
            t += dur
    return n


def _dismissal_by_grade(data):
    """
    {grade: last_allowed_period} from admin settings (start_time,
    grade_end_times, breaks). The last period is the NUMBER of 45-min
    lessons that fit — breaks included — NOT the raw clock-hour difference.
    """
    from scoring_config import DEFAULT_DISMISSAL
    sr_list = data.get("system_requirements") or []
    sr = sr_list[0] if sr_list else {}
    start_min = _parse_clock_to_min(sr.get("start_time"))
    if start_min is None:
        start_min = 8 * 60
    breaks = sr.get("breaks") or []
    if isinstance(breaks, str):
        import json
        try:
            breaks = json.loads(breaks)
        except ValueError:
            breaks = []
    ends = sr.get("grade_end_times") or {}
    result = {}
    for grade in range(1, 7):
        end_min = _parse_clock_to_min(ends.get(str(grade)))
        if end_min is not None and end_min <= start_min:   # '01:00' = 1 PM
            end_min += 12 * 60
        periods = _periods_until(start_min, end_min, breaks)
        result[grade] = periods if (periods and periods >= 1) else DEFAULT_DISMISSAL
    return result

def student_structure_penalties(schedule, data, lookups, collect=False):
    """
    Student-focused structural rules, shared by ALL three algorithms so the
    score they optimise and the violations they report can never drift:

      - No internal gaps: a class's lessons on a day must have no empty period
        between its first and last lesson.                       (HARD)
      - Start at period 1: a class must begin its day at period 1. (HARD)
      - Young grades (1-3) finish early: no lessons in periods 7-8. (SOFT)

    schedule: {assignment_id: [timeslot_id, ...]}
    Returns (total_penalty, violations_or_None). Pass collect=True to also
    build the human-readable violations list (used only for the report, not
    in the hot optimisation loop).
    """
    from scoring_config import (
        STUDENT_GAP_PENALTY,
        STUDENT_LATE_START_PENALTY,
        GRADE_DISMISSAL,
        DEFAULT_DISMISSAL,
        DISMISSAL_PENALTY_PER_HOUR,
        NO_EMPTY_DAY_PENALTY,
        BALANCE_PENALTY_PER_HOUR,
        BALANCE_TOLERANCE,
        GRADE_MAX_PER_DAY_PENALTY_PER_HOUR,
    )

    requirement_by_id = lookups["requirement_by_id"]
    timeslot_by_id = lookups["timeslot_by_id"]
    group_by_id = lookups["group_by_id"]

    total = 0.0
    violations = [] if collect else None

    def gname_of(gid):
        return group_by_id.get(gid, {}).get("group_name", f"קבוצה {gid}")

    # Gather each group's lesson-hours per day.
    group_day_hours = {}
    for ta in data["teacher_assignments"]:
        req = requirement_by_id[ta["cur_requirement_id"]]
        gid = req["student_group_id"]
        for t in schedule[ta["id"]]:
            ts = timeslot_by_id[t]
            group_day_hours.setdefault((gid, ts["day_of_week"]), []).append(ts["hour_of_day"])

    # Admin-defined max lessons per day, keyed by grade number (from the DB).
    # Grades with no row get no cap (fail-safe).
    max_per_day_by_grade = {
        row["grade_level"]: row["max_lessons_per_day"]
        for row in data.get("grade_schedule_limits", [])
    }

    dismissal_map = _dismissal_by_grade(data)

    for (gid, day), hours in group_day_hours.items():
        hs = sorted(set(hours))
        if not hs:
            continue
        first, last = hs[0], hs[-1]

        # Internal gaps (hard)
        internal = (last - first + 1) - len(hs)
        if internal > 0:
            pen = internal * STUDENT_GAP_PENALTY
            total += pen
            if collect:
                violations.append({"type": "student_gap", "detail": f"{gname_of(gid)}: {internal} חלונות ריקים ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "hard"})

        # Must start at period 1 (hard)
        if first > 1:
            pen = STUDENT_LATE_START_PENALTY
            total += pen
            if collect:
                violations.append({"type": "student_late_start", "detail": f"{gname_of(gid)}: לא מתחיל בשעה 1 ביום {DAY_NAMES.get(day, day)} (מתחיל בשעה {first})", "penalty": pen, "severity": "hard"})

        # Per-grade dismissal: lessons past the grade's last allowed period (strong-soft)
        grade = grade_of(gname_of(gid))
        if grade is not None:
            dismissal = dismissal_map.get(grade, DEFAULT_DISMISSAL)
            late = [h for h in hs if h > dismissal]
            if late:
                pen = len(late) * DISMISSAL_PENALTY_PER_HOUR
                total += pen
                if collect:
                    violations.append({"type": "grade_dismissal", "detail": f"{gname_of(gid)} (שכבה {grade}): {len(late)} שיעורים אחרי שעת הסיום ({dismissal}) ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})

        # Admin-defined max lessons per day for this grade (strong-soft).
        # hs already holds this class's distinct lesson-periods for the day.
        if grade is not None and grade in max_per_day_by_grade:
            cap = max_per_day_by_grade[grade]
            over = len(hs) - cap
            if over > 0:
                pen = over * GRADE_MAX_PER_DAY_PENALTY_PER_HOUR
                total += pen
                if collect:
                    violations.append({"type": "grade_max_per_day", "detail": f"{gname_of(gid)} (שכבה {grade}): {len(hs)} שיעורים ביום {DAY_NAMES.get(day, day)}, מעל המקסימום ({cap})", "penalty": pen, "severity": "soft"})

    # No empty day: every class must have at least one lesson on each school day (hard)
    school_days = sorted({ts["day_of_week"] for ts in data["timeslots"]})
    all_group_ids = {
        requirement_by_id[ta["cur_requirement_id"]]["student_group_id"]
        for ta in data["teacher_assignments"]
    }
    for gid in all_group_ids:
        for day in school_days:
            if (gid, day) not in group_day_hours:
                total += NO_EMPTY_DAY_PENALTY
                if collect:
                    violations.append({"type": "empty_day", "detail": f"{gname_of(gid)}: יום ריק לחלוטין ({DAY_NAMES.get(day, day)})", "penalty": NO_EMPTY_DAY_PENALTY, "severity": "hard"})

    # Balanced daily load: penalise big day-to-day swings in a class's lesson count (soft)
    group_daily_counts = {}
    for (gid, day), hours in group_day_hours.items():
        group_daily_counts.setdefault(gid, []).append(len(set(hours)))
    for gid, counts in group_daily_counts.items():
        # only compare across the days the class actually has lessons
        if len(counts) < 2:
            continue
        spread = max(counts) - min(counts)
        if spread > BALANCE_TOLERANCE:
            pen = (spread - BALANCE_TOLERANCE) * BALANCE_PENALTY_PER_HOUR
            total += pen
            if collect:
                violations.append({"type": "daily_balance", "detail": f"{gname_of(gid)}: פער של {spread} שעות בין היום הארוך לקצר", "penalty": pen, "severity": "soft"})

    # Admin-defined pedagogical constraints (soft) — flows to GA, HC and the report
    _ptot, _pvios = pedagogical_penalties(schedule, data, lookups, collect=collect)
    total += _ptot
    if collect and _pvios:
        violations.extend(_pvios)

    return total, violations


def pedagogical_penalties(schedule, data, lookups, collect=False):
    """
    Admin-defined pedagogical constraints (pedagogical_constraints table,
    managed from the 'הגדרות מוסד' screen). ALL SOFT. Only is_active rows
    reach here (filtered in data_access.fetch_all_data). Each penalty is
    multiplied by the row's own `weight` (default 1).
    Types: max_per_day, not_last, morning_only, not_consecutive, min_gap.
    """
    from scoring_config import (
        PED_MAX_PER_DAY_PENALTY, PED_NOT_LAST_PENALTY,
        PED_MORNING_ONLY_PENALTY, PED_MORNING_ONLY_THRESHOLD,
        PED_NOT_CONSECUTIVE_PENALTY, PED_MIN_GAP_PENALTY,
    )

    constraints = data.get("pedagogical_constraints", [])
    total = 0.0
    violations = [] if collect else None
    if not constraints:
        return total, violations

    requirement_by_id = lookups["requirement_by_id"]
    timeslot_by_id = lookups["timeslot_by_id"]
    group_by_id = lookups["group_by_id"]
    subject_by_id = lookups["subject_by_id"]

    def gname(gid):
        return group_by_id.get(gid, {}).get("group_name", f"קבוצה {gid}")
    def sname(sid):
        return subject_by_id.get(sid, {}).get("subject_name", f"מקצוע {sid}")

    # Per (group, day): list of (subject_id, hour) actually scheduled.
    group_day_items = {}
    for ta in data["teacher_assignments"]:
        req = requirement_by_id[ta["cur_requirement_id"]]
        gid = req["student_group_id"]
        sid = req["subject_id"]
        for t in schedule[ta["id"]]:
            ts = timeslot_by_id[t]
            group_day_items.setdefault((gid, ts["day_of_week"]), []).append((sid, ts["hour_of_day"]))

    for c in constraints:
        ctype = c["constraint_type"]
        a = c.get("subject_a_id")
        b = c.get("subject_b_id")
        n = c.get("numeric_value")
        w = c.get("weight") or 1

        if ctype == "max_per_day" and a is not None and n is not None:
            for (gid, day), items in group_day_items.items():
                cnt = sum(1 for sid, h in items if sid == a)
                if cnt > n:
                    pen = (cnt - n) * PED_MAX_PER_DAY_PENALTY * w
                    total += pen
                    if collect:
                        violations.append({"type": "ped_max_per_day", "detail": f"{gname(gid)} / {sname(a)}: {cnt} שיעורים ביום {DAY_NAMES.get(day, day)}, מעל המותר ({n})", "penalty": pen, "severity": "soft"})

        elif ctype == "not_last" and a is not None:
            for (gid, day), items in group_day_items.items():
                if not items:
                    continue
                last = max(h for sid, h in items)
                bad = sum(1 for sid, h in items if sid == a and h == last)
                if bad:
                    pen = bad * PED_NOT_LAST_PENALTY * w
                    total += pen
                    if collect:
                        violations.append({"type": "ped_not_last", "detail": f"{gname(gid)} / {sname(a)}: בשיעור האחרון ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})

        elif ctype == "morning_only" and a is not None:
            for (gid, day), items in group_day_items.items():
                bad = sum(1 for sid, h in items if sid == a and h > PED_MORNING_ONLY_THRESHOLD)
                if bad:
                    pen = bad * PED_MORNING_ONLY_PENALTY * w
                    total += pen
                    if collect:
                        violations.append({"type": "ped_morning_only", "detail": f"{gname(gid)} / {sname(a)}: {bad} שיעורים אחרי שעה {PED_MORNING_ONLY_THRESHOLD} ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})

        elif ctype == "not_consecutive" and a is not None and b is not None:
            for (gid, day), items in group_day_items.items():
                a_hours = [h for sid, h in items if sid == a]
                b_hours = set(h for sid, h in items if sid == b)
                pairs = sum(1 for h in a_hours if (h - 1) in b_hours or (h + 1) in b_hours)
                if pairs:
                    pen = pairs * PED_NOT_CONSECUTIVE_PENALTY * w
                    total += pen
                    if collect:
                        violations.append({"type": "ped_not_consecutive", "detail": f"{gname(gid)}: {sname(a)} ו{sname(b)} צמודים ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})

        elif ctype == "min_gap" and a is not None:
            b_eff = b if b is not None else a
            req = n if n is not None else 0
            for (gid, day), items in group_day_items.items():
                a_hours = [h for sid, h in items if sid == a]
                b_hours = [h for sid, h in items if sid == b_eff]
                if a == b_eff:
                    prs = [(a_hours[i], a_hours[j]) for i in range(len(a_hours)) for j in range(i + 1, len(a_hours))]
                else:
                    prs = [(ha, hb) for ha in a_hours for hb in b_hours]
                bad = 0
                for ha, hb in prs:
                    sep = abs(ha - hb) - 1          # empty periods between the two lessons
                    if (req == 0 and sep != 0) or (req > 0 and sep < req):
                        bad += 1
                if bad:
                    pen = bad * PED_MIN_GAP_PENALTY * w
                    total += pen
                    if collect:
                        txt = "צמודים" if req == 0 else f"בהפרש {req}+ שעות"
                        violations.append({"type": "ped_min_gap", "detail": f"{gname(gid)} / {sname(a)}↔{sname(b_eff)}: לא {txt} ({bad} ביום {DAY_NAMES.get(day, day)})", "penalty": pen, "severity": "soft"})

    return total, violations


def _day_hour(timeslot_by_id, t):
    ts = timeslot_by_id.get(t)
    if ts is None:
        return f"משבצת {t}"
    return f"{DAY_NAMES.get(ts['day_of_week'], ts['day_of_week'])} שעה {ts['hour_of_day']}"


def score_genetic_schedule_with_violations(schedule, data, lookups):
    total_penalty = 0.0
    violations = []

    teacher_assignments = data["teacher_assignments"]
    requirement_by_id = lookups["requirement_by_id"]
    constraint_by_teacher_timeslot = lookups["constraint_by_teacher_timeslot"]
    preferences_by_teacher = lookups["preferences_by_teacher"]
    timeslot_by_id = lookups["timeslot_by_id"]
    subject_by_id = lookups["subject_by_id"]
    group_by_id = lookups["group_by_id"]
    room_count_by_type = lookups["room_count_by_type"]
    assignments_by_teacher = lookups["assignments_by_teacher"]

    teacher_by_id = {t["id"]: t for t in data["teachers"]}

    def teacher_name(tid):
        t = teacher_by_id.get(tid)
        return f"{t['first_name']} {t['last_name']}" if t else f"מורה {tid}"

    def add(total_add, vtype, detail, severity):
        return total_add, {"type": vtype, "detail": detail, "penalty": total_add, "severity": severity}

    # ---- Structural: teacher double-booking ----
    timeslot_count_by_teacher = {}
    for ta in teacher_assignments:
        counts = timeslot_count_by_teacher.setdefault(ta["teacher_id"], {})
        for t in schedule[ta["id"]]:
            counts[t] = counts.get(t, 0) + 1
    for teacher_id, counts in timeslot_count_by_teacher.items():
        for t, count in counts.items():
            if count > 1:
                pen = (count - 1) * HARD_CONSTRAINT_PENALTY
                total_penalty += pen
                violations.append({
                    "type": "teacher_double_booked",
                    "detail": f"{teacher_name(teacher_id)}: {count} שיעורים באותה משבצת ({_day_hour(timeslot_by_id, t)})",
                    "penalty": pen, "severity": "hard",
                })

    # ---- Structural: student group double-booking ----
    timeslot_count_by_group = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        counts = timeslot_count_by_group.setdefault(req["student_group_id"], {})
        for t in schedule[ta["id"]]:
            counts[t] = counts.get(t, 0) + 1
    for group_id, counts in timeslot_count_by_group.items():
        for t, count in counts.items():
            if count > 1:
                pen = (count - 1) * HARD_CONSTRAINT_PENALTY
                total_penalty += pen
                gname = group_by_id.get(group_id, {}).get("group_name", f"קבוצה {group_id}")
                violations.append({
                    "type": "group_double_booked",
                    "detail": f"{gname}: {count} שיעורים באותה משבצת ({_day_hour(timeslot_by_id, t)})",
                    "penalty": pen, "severity": "hard",
                })

    # ---- Structural: weekly_hours correctness ----
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        expected = req["weekly_hours"]
        actual = len(schedule[ta["id"]])
        if actual != expected:
            pen = abs(expected - actual) * HARD_CONSTRAINT_PENALTY
            total_penalty += pen
            subj = subject_by_id.get(req["subject_id"], {}).get("subject_name", "מקצוע")
            gname = group_by_id.get(req["student_group_id"], {}).get("group_name", "קבוצה")
            violations.append({
                "type": "weekly_hours",
                "detail": f"{teacher_name(ta['teacher_id'])} / {subj} / {gname}: שובצו {actual} מתוך {expected} שעות נדרשות",
                "penalty": pen, "severity": "hard",
            })

    # ---- Room-awareness: simultaneous room-constrained lessons ----
    timeslot_room_demand = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        subject = subject_by_id[req["subject_id"]]
        if subject["required_room_id"] is not None:
            resource = ("specific", subject["required_room_id"])
        elif subject.get("required_room_type") is not None:
            resource = ("type", subject["required_room_type"])
        else:
            continue
        for t in schedule[ta["id"]]:
            demands = timeslot_room_demand.setdefault(t, {})
            demands[resource] = demands.get(resource, 0) + 1
    for t, demands in timeslot_room_demand.items():
        for resource, count in demands.items():
            max_capacity = 1 if resource[0] == "specific" else room_count_by_type.get(resource[1], 0)
            if count > max_capacity:
                pen = (count - max_capacity) * HARD_CONSTRAINT_PENALTY
                total_penalty += pen
                res_desc = "חדר ייעודי" if resource[0] == "specific" else f"חדר מסוג '{resource[1]}'"
                violations.append({
                    "type": "room_capacity",
                    "detail": f"{res_desc}: {count} שיעורים דורשים אותו משאב ({_day_hour(timeslot_by_id, t)}), זמינים {max_capacity}",
                    "penalty": pen, "severity": "hard",
                })

    # ---- Structural: sync_block_identity ----
    sync_blocks = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_blocks.setdefault(sbi, []).append(ta["id"])
    for sbi, assignment_ids in sync_blocks.items():
        if len(assignment_ids) < 2:
            continue
        reference_slots = sorted(schedule[assignment_ids[0]])
        for a_id in assignment_ids[1:]:
            other_slots = sorted(schedule[a_id])
            if reference_slots != other_slots:
                diff_count = len(set(reference_slots).symmetric_difference(set(other_slots)))
                pen = diff_count * HARD_CONSTRAINT_PENALTY
                total_penalty += pen
                violations.append({
                    "type": "sync_block",
                    "detail": f"בלוק מסונכרן {sbi}: שיעורים שאמורים להתקיים יחד שובצו במשבצות שונות",
                    "penalty": pen, "severity": "hard",
                })

    # ---- Data-driven: teacher_constraints (hard/soft) ----
    for ta in teacher_assignments:
        teacher_id = ta["teacher_id"]
        for t in schedule[ta["id"]]:
            constraint = constraint_by_teacher_timeslot.get((teacher_id, t))
            if constraint is not None:
                if constraint["constraint_type"] == "hard":
                    total_penalty += TEACHER_HARD_CONSTRAINT_PENALTY
                    violations.append({
                        "type": "teacher_cannot",
                        "detail": f"{teacher_name(teacher_id)}: שובץ בזמן שסומן 'לא יכול' ({_day_hour(timeslot_by_id, t)})",
                        "penalty": TEACHER_HARD_CONSTRAINT_PENALTY, "severity": "hard",
                    })
                else:
                    pen = constraint["weight"] * SOFT_CONSTRAINT_WEIGHT_MULTIPLIER
                    total_penalty += pen
                    violations.append({
                        "type": "teacher_prefers_not",
                        "detail": f"{teacher_name(teacher_id)}: שובץ בזמן שסומן 'מעדיף שלא' ({_day_hour(timeslot_by_id, t)})",
                        "penalty": pen, "severity": "soft",
                    })

    # ---- teacher_preferences (soft) ----
    for teacher_id, assignment_ids in assignments_by_teacher.items():
        prefs = preferences_by_teacher.get(teacher_id)
        if prefs is None:
            continue
        total_hours = sum(len(schedule[a_id]) for a_id in assignment_ids)
        if prefs["min_hours"] is not None and total_hours < prefs["min_hours"]:
            pen = (prefs["min_hours"] - total_hours) * OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR
            total_penalty += pen
            violations.append({"type": "hours_range", "detail": f"{teacher_name(teacher_id)}: {total_hours} שעות, מתחת למינימום ({prefs['min_hours']})", "penalty": pen, "severity": "soft"})
        if prefs["max_hours"] is not None and total_hours > prefs["max_hours"]:
            pen = (total_hours - prefs["max_hours"]) * OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR
            total_penalty += pen
            violations.append({"type": "hours_range", "detail": f"{teacher_name(teacher_id)}: {total_hours} שעות, מעל המקסימום ({prefs['max_hours']})", "penalty": pen, "severity": "soft"})

        teacher_day_hours = {}
        for a_id in assignment_ids:
            for t in schedule[a_id]:
                ts = timeslot_by_id[t]
                teacher_day_hours.setdefault(ts["day_of_week"], []).append(ts["hour_of_day"])

        teaching_days = len(teacher_day_hours)
        if (6 - teaching_days) == 0 and prefs.get("priority_free_day") and prefs["priority_free_day"] > 0:
            pen = PREFERENCE_WEIGHTS["free_day"] * prefs["priority_free_day"]
            total_penalty += pen
            violations.append({"type": "free_day", "detail": f"{teacher_name(teacher_id)}: אין יום חופשי", "penalty": pen, "severity": "soft"})

        for day, hours in teacher_day_hours.items():
            hs = sorted(hours)
            gaps = (hs[-1] - hs[0] + 1) - len(hs)
            if gaps > 0 and prefs.get("priority_no_gaps") and prefs["priority_no_gaps"] > 0:
                pen = gaps * PREFERENCE_WEIGHTS["no_gaps"] * prefs["priority_no_gaps"]
                total_penalty += pen
                violations.append({"type": "teacher_gaps", "detail": f"{teacher_name(teacher_id)}: {gaps} חלונות ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})
            if hs[-1] > 6 and prefs.get("priority_early_finish") and prefs["priority_early_finish"] > 0:
                pen = (hs[-1] - 6) * PREFERENCE_WEIGHTS["early_finish"] * prefs["priority_early_finish"]
                total_penalty += pen
                violations.append({"type": "early_finish", "detail": f"{teacher_name(teacher_id)}: מסיים בשעה {hs[-1]} ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})
            if gaps > 0 and prefs.get("priority_consecutive") and prefs["priority_consecutive"] > 0:
                pen = gaps * PREFERENCE_WEIGHTS["consecutive"] * prefs["priority_consecutive"]
                total_penalty += pen
                violations.append({"type": "consecutive", "detail": f"{teacher_name(teacher_id)}: שיעורים לא רצופים ביום {DAY_NAMES.get(day, day)}", "penalty": pen, "severity": "soft"})

    # ---- Subject distribution ----
    group_subject_day_counts = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        for t in schedule[ta["id"]]:
            ts = timeslot_by_id[t]
            key = (req["student_group_id"], req["subject_id"], ts["day_of_week"])
            group_subject_day_counts[key] = group_subject_day_counts.get(key, 0) + 1
      # subject_distribution הוסר לבקשת המנהל — הפיזור נשלט דרך "מקסימום ליום" בהגדרות מוסד.

    # ---- Student structure: gaps, start-at-1 (hard) + young grades late (soft) ----
    _stot, _svios = student_structure_penalties(schedule, data, lookups, collect=True)
    total_penalty += _stot
    violations.extend(_svios)

    violations.sort(key=lambda v: v["penalty"], reverse=True)
    return total_penalty, violations
