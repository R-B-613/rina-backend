"""
preflight.py

Pre-generation validation: "can a timetable even exist with this data?"

Runs on the SAME dict that fetch_all_data() returns (the one the solvers
consume), so the admin is told about a problem BEFORE a 1-3 minute run is
wasted on data that cannot work.

Each check returns a list of issues:

    {
        "code":     str,                      # stable id, e.g. "class_over_capacity"
        "severity": "error" | "warning",      # error = blocks generation
        "category": str,                      # structure | curriculum | assignments | rooms | sync
        "message":  str (Hebrew, shown to the admin),
        "entity":   {"type": str, "id": int} | None,
    }

DESIGN RULES
------------
* Only block (ERROR) on what is PROVABLY impossible - a necessary condition
  that the solvers themselves enforce as hard. A false "blocked" is the worst
  outcome here, because the admin cannot override it. Anything softer, or
  anything that is a policy rather than an impossibility, is a WARNING.
* The week structure is NOT recomputed here. It comes from
  scoring_violations.build_structure_limits(), the single source of truth the
  solvers use, so "room for 36 lessons" in a message is exactly the number the
  solvers work with.
* To add a check: write a function (data) -> list[issue] and append it to
  _CHECKS at the bottom.
"""

from collections import Counter, defaultdict

from scoring_violations import build_structure_limits, grade_of

ERROR = "error"      # generation is blocked
WARNING = "warning"  # shown to the admin, does not block


def _issue(code, severity, category, message, entity_type=None, entity_id=None):
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": message,
        "entity": {"type": entity_type, "id": entity_id} if entity_type else None,
    }


# ---------------------------------------------------------------------------
# Check 1: does each class's curriculum fit in the week the admin defined?
# ---------------------------------------------------------------------------

def _check_class_capacity(data):
    """
    For every class:
      * total curriculum hours  >  lessons that fit in the week   -> ERROR
        (a hard upper bound: lessons past a grade's end-of-day or on an
        inactive day are hard-forbidden in CSP and hard-penalised in GA/HC)
      * total curriculum hours  <  number of study days           -> ERROR
        (every active day needs at least one lesson - "no empty day")

    Capacity per day = how many timeslots exist on that day with
    hour <= the grade's ceiling for that day. Using the timeslots table too
    means a long school day can never promise more than 8 lessons.

    NOT checked here on purpose: grade_schedule_limits.max_lessons_per_day.
    In GA/HC it is a strong SOFT penalty, not a hard cap, so it must not block.
    """
    issues = []
    active_days, ceiling = build_structure_limits(data)

    hours_by_day = defaultdict(set)
    for ts in data["timeslots"]:
        hours_by_day[ts["day_of_week"]].add(ts["hour_of_day"])

    def lessons_per_active_day(grade):
        per_day = {}
        for day in active_days:
            limit = ceiling.get((grade, day), 0)
            per_day[day] = sum(1 for h in hours_by_day.get(day, ()) if h <= limit)
        return per_day

    # What the solvers will actually place: one block of weekly_hours per
    # teacher_assignment. A requirement with no teacher yet still counts once
    # (it is flagged separately by check 2); one with two teachers counts twice.
    teachers_per_req = Counter(ta["cur_requirement_id"] for ta in data["teacher_assignments"])
    hours_by_group = defaultdict(int)
    for req in data["curriculum_requirements"]:
        hours = req.get("weekly_hours") or 0
        hours_by_group[req["student_group_id"]] += hours * max(1, teachers_per_req.get(req["id"], 0))

    for group in data["student_groups"]:
        gid = group["id"]
        total = hours_by_group.get(gid, 0)
        if total <= 0:
            continue

        name = group.get("group_name") or f"קבוצה {gid}"
        grade = grade_of(group.get("group_name"))
        if grade is None:
            issues.append(_issue(
                "class_unknown_grade", WARNING, "curriculum",
                f"{name}: לא ניתן לזהות שכבה משם הכיתה, ולכן לא נבדק אם שעות הלימוד שלה נכנסות במערכת.",
                "group", gid,
            ))
            continue

        per_day = lessons_per_active_day(grade)
        capacity = sum(per_day.values())
        study_days = sum(1 for n in per_day.values() if n > 0)

        if total > capacity:
            issues.append(_issue(
                "class_over_capacity", ERROR, "curriculum",
                f"{name}: בתוכנית הלימודים {total} שעות שבועיות, אבל במערכת השעות יש מקום ל־{capacity} שעות בלבד "
                f"({study_days} ימי לימוד, לפי שעות הסיום שהוגדרו). חסרות {total - capacity} שעות — "
                f"יש להקטין את תוכנית הלימודים או להאריך את יום הלימודים.",
                "group", gid,
            ))
        elif total < study_days:
            issues.append(_issue(
                "class_too_few_hours", ERROR, "curriculum",
                f"{name}: בתוכנית הלימודים רק {total} שעות שבועיות, פחות ממספר ימי הלימוד ({study_days}). "
                f"כל יום לימודים חייב לכלול לפחות שיעור אחד.",
                "group", gid,
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 2: every lesson in a curriculum needs a teacher.
# ---------------------------------------------------------------------------

def _check_requirements_without_teacher(data):
    """A curriculum row with weekly_hours > 0 and no teacher_assignment can
    never be scheduled (the solvers schedule teacher_assignments)."""
    issues = []
    assigned = {ta["cur_requirement_id"] for ta in data["teacher_assignments"]}
    subject_by_id = {s["id"]: s for s in data["subjects"]}
    group_by_id = {g["id"]: g for g in data["student_groups"]}

    for req in data["curriculum_requirements"]:
        hours = req.get("weekly_hours") or 0
        if hours <= 0 or req["id"] in assigned:
            continue
        subject = subject_by_id.get(req["subject_id"], {}).get("subject_name", f"מקצוע {req['subject_id']}")
        group = group_by_id.get(req["student_group_id"], {}).get("group_name", f"קבוצה {req['student_group_id']}")
        issues.append(_issue(
            "requirement_without_teacher", ERROR, "assignments",
            f"{group} · {subject} ({hours} שעות שבועיות): לא משויך מורה.",
            "requirement", req["id"],
        ))
    return issues


# ---------------------------------------------------------------------------
# Registry + entry points
# ---------------------------------------------------------------------------

_CHECKS = (
    _check_class_capacity,
    _check_requirements_without_teacher,
)


def run_preflight(data):
    """Runs every check on fetch_all_data()'s dict. Errors first, then warnings."""
    issues = []
    for check in _CHECKS:
        issues.extend(check(data))
    issues.sort(key=lambda i: 0 if i["severity"] == ERROR else 1)  # stable sort
    return issues


def summarize(issues):
    """The payload the API returns / the frontend reads."""
    errors = sum(1 for i in issues if i["severity"] == ERROR)
    return {
        "can_generate": errors == 0,
        "error_count": errors,
        "warning_count": len(issues) - errors,
        "issues": issues,
    }
