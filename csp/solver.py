"""
csp/solver.py

CSP (Constraint Satisfaction Problem) solver for the timetabling problem,
built with Google OR-Tools CP-SAT.

This module does NOT touch the database directly - it receives plain
Python data (as fetched by data_access.fetch_all_data()) and returns a
plain Python result dict. Saving to the DB happens elsewhere (main.py).

-------------------------------------------------------------------------
HOW HARD VS SOFT CONSTRAINTS ARE HANDLED HERE
-------------------------------------------------------------------------
Two different mechanisms are used on purpose:

1) STRUCTURAL hard constraints (the solver is physically not allowed to
   break these - not even scored, just forbidden):
     - A teacher cannot teach two different things at the same timeslot.
     - A student group cannot attend two different lessons at the same
       timeslot.
     - A room cannot host two different lessons at the same timeslot.
     - Every curriculum_requirement must receive exactly its required
       weekly_hours.
   These come "for free" from the timetable structure - any teacher/
   school timetable would consider these absolute rules, not preferences.

2) SCORED hard/soft constraints (from teacher_constraints, weighted very
   high if type='hard', low if type='soft') and teacher_preferences
   (weighted using scoring_config.PREFERENCE_WEIGHTS). These are NOT
   structurally forbidden in the CP-SAT model (CP-SAT could technically
   still produce a solution that violates a "hard" row in
   teacher_constraints, if doing so allows the model to be feasible at
   all) - instead they're folded into one unified penalty score, using
   the EXACT SAME formula that the Genetic Algorithm and Hill Climbing
   solvers will use later, so all three algorithms can be compared fairly
   on the same scale.

This keeps CSP fast and reliable (thanks to true structural constraints)
while still producing a score that's apples-to-apples comparable with
GA/Hill Climbing.

TODO (explicitly skipped for v1, per project decision):
    - Room capacity matching (room.capacity vs student_group.student_count)
    - Room type matching (subjects.required_room_id)
    - sync_block_identity (lessons that must be scheduled simultaneously)
-------------------------------------------------------------------------
"""

from ortools.sat.python import cp_model
from scoring_violations import score_genetic_schedule_with_violations, grade_of, build_structure_limits

from scoring_config import (
    HARD_CONSTRAINT_PENALTY,
    TEACHER_HARD_CONSTRAINT_PENALTY,
    SOFT_CONSTRAINT_WEIGHT_MULTIPLIER,
    PREFERENCE_WEIGHTS,
    OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR,
    SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR,
    CSP_MAX_SOLVE_SECONDS,
    ALGO_CSP,
    YOUNG_GRADE_LATE_PENALTY,
)


def _build_lookup_maps(data: dict):
    """
    Builds convenience lookup dicts/sets from the raw fetched data, so the
    rest of the solver doesn't need to repeatedly loop over lists.
    """
    timeslots = data["timeslots"]
    rooms = data["rooms"]
    curriculum_requirements = data["curriculum_requirements"]
    teacher_assignments = data["teacher_assignments"]
    teacher_constraints = data["teacher_constraints"]
    teacher_preferences = data["teacher_preferences"]

    # cur_requirement_id -> requirement row (for weekly_hours, student_group_id, subject_id)
    requirement_by_id = {req["id"]: req for req in curriculum_requirements}

    # teacher_assignment.id -> assignment row (for teacher_id, cur_requirement_id)
    assignment_by_id = {ta["id"]: ta for ta in teacher_assignments}

    # teacher_id -> list of assignment ids they teach
    assignments_by_teacher = {}
    for ta in teacher_assignments:
        assignments_by_teacher.setdefault(ta["teacher_id"], []).append(ta["id"])

    # student_group_id -> list of assignment ids that belong to that group
    assignments_by_group = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        assignments_by_group.setdefault(req["student_group_id"], []).append(ta["id"])

    # (teacher_id, timeslot_id) -> constraint row, for fast lookup during scoring
    constraint_by_teacher_timeslot = {
        (c["teacher_id"], c["timeslot_id"]): c for c in teacher_constraints
    }

    # teacher_id -> preferences row
    preferences_by_teacher = {p["teacher_id"]: p for p in teacher_preferences}

    # Extra lookups needed by the shared violations scorer.
    subjects = data["subjects"]
    student_groups = data["student_groups"]
    rooms = data["rooms"]
    timeslot_by_id = {ts["id"]: ts for ts in timeslots}
    subject_by_id = {s["id"]: s for s in subjects}
    group_by_id = {g["id"]: g for g in student_groups}
    room_count_by_type = {}
    for room in rooms:
        rt = room.get("room_type")
        if rt is not None:
            room_count_by_type[rt] = room_count_by_type.get(rt, 0) + 1

    return {
        "requirement_by_id": requirement_by_id,
        "assignment_by_id": assignment_by_id,
        "assignments_by_teacher": assignments_by_teacher,
        "assignments_by_group": assignments_by_group,
        "constraint_by_teacher_timeslot": constraint_by_teacher_timeslot,
        "preferences_by_teacher": preferences_by_teacher,
        "teacher_by_id": {t["id"]: t for t in data["teachers"]},
        "timeslot_by_id": timeslot_by_id,
        "subject_by_id": subject_by_id,
        "group_by_id": group_by_id,
        "room_count_by_type": room_count_by_type,
    }


def _create_decision_variables(model, teacher_assignments, timeslots):
    """
    Creates one BoolVar per (assignment, timeslot) combination:
    schedule_vars[(assignment_id, timeslot_id)] == 1 means "this teaching
    assignment happens at this timeslot".
    """
    schedule_vars = {}
    for ta in teacher_assignments:
        for ts in timeslots:
            var_name = f"assign_{ta['id']}_time_{ts['id']}"
            schedule_vars[(ta["id"], ts["id"])] = model.NewBoolVar(var_name)
    return schedule_vars

def _add_student_contiguity_constraints(model, schedule_vars, data, lookups, timeslots):
    """
    HARD structural rule: each group's lessons on a day must form a contiguous
    block starting at period 1 (no late start, no internal gaps). Since group
    double-booking is already forbidden, busy(p) is 0/1, so the single rule
    busy(p) <= busy(p-1) enforces BOTH 'start at 1' AND 'no gaps' at once.
    """
    assignments_by_group = lookups["assignments_by_group"]
    active_days, _ceiling = build_structure_limits(data)
    ts_by_day_hour = {}
    hours_by_day = {}
    for ts in timeslots:
        ts_by_day_hour[(ts["day_of_week"], ts["hour_of_day"])] = ts["id"]
        hours_by_day.setdefault(ts["day_of_week"], []).append(ts["hour_of_day"])

    for group_id, assignment_ids in assignments_by_group.items():
        for day, hours in hours_by_day.items():
            if day not in active_days:
                continue
            hs = sorted(hours)
            for i in range(1, len(hs)):
                cur = sum(schedule_vars[(a_id, ts_by_day_hour[(day, hs[i])])] for a_id in assignment_ids)
                prev = sum(schedule_vars[(a_id, ts_by_day_hour[(day, hs[i - 1])])] for a_id in assignment_ids)
                model.Add(cur <= prev)

            # HARD: no empty day — each class must have >= 1 lesson every school day.
            # This makes an empty day structurally impossible for CSP (forbidden,
            # not merely penalised). HC/GA still score it via NO_EMPTY_DAY_PENALTY.
            day_lessons = sum(
                schedule_vars[(a_id, ts_by_day_hour[(day, h)])]
                for a_id in assignment_ids
                for h in hs
            )
            model.Add(day_lessons >= 1)

def _add_structural_hard_constraints(model, schedule_vars, data, lookups, timeslots):
    """
    Adds constraints that are NEVER allowed to be broken: weekly_hours,
    teacher/group double-booking, room-awareness, and sync_block_identity.
    """
    teacher_assignments = data["teacher_assignments"]
    requirement_by_id = lookups["requirement_by_id"]
    assignments_by_teacher = lookups["assignments_by_teacher"]
    assignments_by_group = lookups["assignments_by_group"]

    # 1. Each curriculum requirement must get EXACTLY its weekly_hours.
    assignments_by_requirement = {}
    for ta in teacher_assignments:
        assignments_by_requirement.setdefault(ta["cur_requirement_id"], []).append(ta["id"])

    for req_id, assignment_ids in assignments_by_requirement.items():
        weekly_hours = requirement_by_id[req_id]["weekly_hours"]
        model.Add(
            sum(
                schedule_vars[(a_id, ts["id"])]
                for a_id in assignment_ids
                for ts in timeslots
            )
            == weekly_hours
        )

    # 2. A teacher cannot teach two different assignments at the same timeslot.
    for teacher_id, assignment_ids in assignments_by_teacher.items():
        for ts in timeslots:
            model.AddAtMostOne(
                schedule_vars[(a_id, ts["id"])] for a_id in assignment_ids
            )

    # 3. A student group cannot attend two different lessons at the same timeslot.
    for group_id, assignment_ids in assignments_by_group.items():
        for ts in timeslots:
            model.AddAtMostOne(
                schedule_vars[(a_id, ts["id"])] for a_id in assignment_ids
            )

    # 4. Room-awareness: limit simultaneous lessons by available rooms.
    subject_by_id = {s["id"]: s for s in data["subjects"]}
    room_count_by_type = {}
    for room in data["rooms"]:
        rt = room.get("room_type")
        if rt is not None:
            room_count_by_type[rt] = room_count_by_type.get(rt, 0) + 1

    room_resource_groups = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        subject = subject_by_id[req["subject_id"]]

        if subject["required_room_id"] is not None:
            resource = ("specific", subject["required_room_id"])
        elif subject.get("required_room_type") is not None:
            resource = ("type", subject["required_room_type"])
        else:
            continue
        room_resource_groups.setdefault(resource, []).append(ta["id"])

    for resource, assignment_ids in room_resource_groups.items():
        if resource[0] == "specific":
            max_simultaneous = 1
        else:
            max_simultaneous = room_count_by_type.get(resource[1], 0)

        for ts in timeslots:
            model.Add(
                sum(schedule_vars[(a_id, ts["id"])] for a_id in assignment_ids)
                <= max_simultaneous
            )

    # 5. sync_block_identity: synced lessons must share identical timeslots.
    sync_blocks = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_blocks.setdefault(sbi, []).append(ta["id"])

    for sbi, assignment_ids in sync_blocks.items():
        if len(assignment_ids) < 2:
            continue
        first_id = assignment_ids[0]
        for other_id in assignment_ids[1:]:
            for ts in timeslots:
                model.Add(
                    schedule_vars[(first_id, ts["id"])]
                    == schedule_vars[(other_id, ts["id"])]
                )

def _add_admin_structure_constraints(model, schedule_vars, data, lookups, timeslots):
    """
    HARD: honour the admin's day structure exactly (School Settings), by forcing
    the matching decision variables to 0 — structurally impossible, not penalised.
      - No lesson on a day outside active_days.
      - No lesson past a grade's end-of-day period; Friday uses friday_end_time.
        Period count = 45-min lessons + breaks (build_structure_limits).
    """
    from scoring_config import DEFAULT_DISMISSAL
    active_days, ceiling = build_structure_limits(data)
    requirement_by_id = lookups["requirement_by_id"]
    group_by_id = lookups["group_by_id"]
    for ta in data["teacher_assignments"]:
        gid = requirement_by_id[ta["cur_requirement_id"]]["student_group_id"]
        grade = grade_of(group_by_id.get(gid, {}).get("group_name"))
        for ts in timeslots:
            day, hour = ts["day_of_week"], ts["hour_of_day"]
            allowed = day in active_days
            if allowed and grade is not None:
                allowed = hour <= ceiling.get((grade, day), DEFAULT_DISMISSAL)
            if not allowed:
                model.Add(schedule_vars[(ta["id"], ts["id"])] == 0)

def _compute_penalty_score(solver, schedule_vars, data, lookups, timeslots):
    """
    Computes the unified penalty score based on the solved CP-SAT model.
    Covers: teacher_constraints (hard/soft), teacher_preferences (full),
    and subject distribution. Room-awareness and sync_block are enforced
    structurally above, so they always read 0 here.
    """
    constraint_by_teacher_timeslot = lookups["constraint_by_teacher_timeslot"]
    preferences_by_teacher = lookups["preferences_by_teacher"]
    teacher_by_id = lookups["teacher_by_id"]
    assignment_by_id = lookups["assignment_by_id"]
    assignments_by_teacher = lookups["assignments_by_teacher"]
    requirement_by_id = lookups["requirement_by_id"]

    total_penalty = 0.0

    # ---- teacher_constraints (hard/soft) ----
    for ta in data["teacher_assignments"]:
        for ts in timeslots:
            var = schedule_vars[(ta["id"], ts["id"])]
            if solver.Value(var) == 1:
                key = (ta["teacher_id"], ts["id"])
                constraint = constraint_by_teacher_timeslot.get(key)
                if constraint is not None:
                    if constraint["constraint_type"] == "hard":
                        total_penalty += TEACHER_HARD_CONSTRAINT_PENALTY
                    else:
                        total_penalty += constraint["weight"] * SOFT_CONSTRAINT_WEIGHT_MULTIPLIER

    # ---- teacher_preferences (full implementation) ----
    for teacher_id, assignment_ids in assignments_by_teacher.items():
        taught_timeslot_ids = []
        for a_id in assignment_ids:
            for ts in timeslots:
                if solver.Value(schedule_vars[(a_id, ts["id"])]) == 1:
                    taught_timeslot_ids.append(ts["id"])

        total_hours = len(taught_timeslot_ids)

        # Hours range: workload band comes from the teacher record (admin-set).
        trec = teacher_by_id.get(teacher_id) or {}
        tmin, tmax = trec.get("min_hours"), trec.get("max_hours")
        if tmin is not None and total_hours < tmin:
            total_penalty += (tmin - total_hours) * OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR
        if tmax is not None and total_hours > tmax:
            total_penalty += (total_hours - tmax) * OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR

        prefs = preferences_by_teacher.get(teacher_id)
        if prefs is None:
            continue

        # Collect (day, hour) pairs for this teacher
        teacher_day_hours = {}
        for t_id in taught_timeslot_ids:
            for ts in timeslots:
                if ts["id"] == t_id:
                    teacher_day_hours.setdefault(ts["day_of_week"], []).append(ts["hour_of_day"])
                    break

        teaching_days = len(teacher_day_hours)

        # Free day
        free_days = 6 - teaching_days
        if free_days == 0 and prefs.get("priority_free_day") and prefs["priority_free_day"] > 0:
            total_penalty += PREFERENCE_WEIGHTS["free_day"] * prefs["priority_free_day"]

        # Per-day: gaps, early finish, consecutive
        for day, hours in teacher_day_hours.items():
            hours_sorted = sorted(hours)
            first_hour = hours_sorted[0]
            last_hour = hours_sorted[-1]
            expected_if_no_gaps = last_hour - first_hour + 1
            actual_count = len(hours_sorted)
            gaps = expected_if_no_gaps - actual_count

            if gaps > 0 and prefs.get("priority_no_gaps") and prefs["priority_no_gaps"] > 0:
                total_penalty += gaps * PREFERENCE_WEIGHTS["no_gaps"] * prefs["priority_no_gaps"]

            if last_hour > 6 and prefs.get("priority_early_finish") and prefs["priority_early_finish"] > 0:
                total_penalty += (last_hour - 6) * PREFERENCE_WEIGHTS["early_finish"] * prefs["priority_early_finish"]

            if gaps > 0 and prefs.get("priority_consecutive") and prefs["priority_consecutive"] > 0:
                total_penalty += gaps * PREFERENCE_WEIGHTS["consecutive"] * prefs["priority_consecutive"]

    # ---- Subject distribution across the week ----
    subject_by_id = {s["id"]: s for s in data["subjects"]}
    group_subject_day_counts = {}
    for ta in data["teacher_assignments"]:
        req = requirement_by_id[ta["cur_requirement_id"]]
        group_id = req["student_group_id"]
        subject_id = req["subject_id"]
        for ts in timeslots:
            if solver.Value(schedule_vars[(ta["id"], ts["id"])]) == 1:
                key = (group_id, subject_id, ts["day_of_week"])
                group_subject_day_counts[key] = group_subject_day_counts.get(key, 0) + 1

    for key, count in group_subject_day_counts.items():
        if count > 1:
            total_penalty += (count - 1) * SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR

    # ---- Young grades (1-3) finishing in periods 7-8 (soft) ----
    group_by_id = lookups["group_by_id"]
    timeslot_by_id = lookups["timeslot_by_id"]
    for ta in data["teacher_assignments"]:
        req = requirement_by_id[ta["cur_requirement_id"]]
        grade = grade_of(group_by_id.get(req["student_group_id"], {}).get("group_name", ""))
        if grade is None or grade > 3:
            continue
        for ts in timeslots:
            if solver.Value(schedule_vars[(ta["id"], ts["id"])]) == 1 and timeslot_by_id[ts["id"]]["hour_of_day"] >= 7:
                total_penalty += YOUNG_GRADE_LATE_PENALTY

    return total_penalty


def _assign_rooms(solver, schedule_vars, data, timeslots):
    """
    Smart room assignment, applied AFTER the CP-SAT solve is complete.
    Same three-tier priority as HC/GA (specific room -> room type ->
    home room -> fallback), adapted for CSP's variable-based schedule
    representation.
    All tiers check: room is free at this timeslot AND capacity >= group size.
    """
    rooms = data["rooms"]
    subjects = data["subjects"]
    student_groups = data["student_groups"]
    curriculum_requirements = data["curriculum_requirements"]

    room_by_id = {r["id"]: r for r in rooms}
    subject_by_id = {s["id"]: s for s in subjects}
    group_by_id = {g["id"]: g for g in student_groups}
    requirement_by_id = {req["id"]: req for req in curriculum_requirements}

    schedule_entries = []
    used_rooms_by_timeslot = {ts["id"]: set() for ts in timeslots}

    for ta in data["teacher_assignments"]:
        req = requirement_by_id[ta["cur_requirement_id"]]
        subject = subject_by_id[req["subject_id"]]
        group = group_by_id[req["student_group_id"]]

        for ts in timeslots:
            var = schedule_vars[(ta["id"], ts["id"])]
            if solver.Value(var) == 1:
                assigned_room = None

                if subject["required_room_id"] is not None:
                    # Priority 1: subject needs ONE specific dedicated room.
                    room_id = subject["required_room_id"]
                    room = room_by_id.get(room_id)
                    if (room
                            and room_id not in used_rooms_by_timeslot[ts["id"]]
                            and room["capacity"] >= group["student_count"]):
                        assigned_room = room_id

                elif subject.get("required_room_type") is not None:
                    # Priority 2: subject needs any room of a certain type.
                    for room in rooms:
                        if (room.get("room_type") == subject["required_room_type"]
                                and room["id"] not in used_rooms_by_timeslot[ts["id"]]
                                and room["capacity"] >= group["student_count"]):
                            assigned_room = room["id"]
                            break

                else:
                    # Priority 3: use the group's home (parent) classroom.
                    home_room_id = group.get("home_room_id")
                    if home_room_id is not None:
                        room = room_by_id.get(home_room_id)
                        if (room
                                and home_room_id not in used_rooms_by_timeslot[ts["id"]]
                                and room["capacity"] >= group["student_count"]):
                            assigned_room = home_room_id

                    # Fallback: any free room with sufficient capacity.
                    if assigned_room is None:
                        for room in rooms:
                            if (room["id"] not in used_rooms_by_timeslot[ts["id"]]
                                    and room["capacity"] >= group["student_count"]):
                                assigned_room = room["id"]
                                break

                if assigned_room is not None:
                    used_rooms_by_timeslot[ts["id"]].add(assigned_room)

                schedule_entries.append(
                    {
                        "timeslot_id": ts["id"],
                        "tea_assignment_id": ta["id"],
                        "room_id": assigned_room,
                    }
                )

    return schedule_entries


def run_csp(data: dict) -> dict:
    """
    Main entry point for the CSP algorithm.

    Parameters
    ----------
    data : dict
        The exact dict returned by data_access.fetch_all_data().

    Returns
    -------
    dict shaped like:
        {
            "algorithm": "CSP",
            "status": "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN",
            "score": float,                # lower is better, 0 = perfect
            "schedule_entries": [ {...}, ... ],  # ready for data_access.save_schedule_run()
        }
    """
    timeslots = data["timeslots"]
    teacher_assignments = data["teacher_assignments"]

    if not timeslots or not teacher_assignments:
        return {
            "algorithm": ALGO_CSP,
            "status": "NO_DATA",
            "score": None,
            "schedule_entries": [],
        }

    lookups = _build_lookup_maps(data)

    model = cp_model.CpModel()
    schedule_vars = _create_decision_variables(model, teacher_assignments, timeslots)

    _add_structural_hard_constraints(model, schedule_vars, data, lookups, timeslots)
    _add_student_contiguity_constraints(model, schedule_vars, data, lookups, timeslots)
    _add_admin_structure_constraints(model, schedule_vars, data, lookups, timeslots)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CSP_MAX_SOLVE_SECONDS
    status = solver.Solve(model)

    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "algorithm": ALGO_CSP,
            "status": status_name,
            "score": None,
            "schedule_entries": [],
        }

    score = _compute_penalty_score(solver, schedule_vars, data, lookups, timeslots)
    schedule_entries = _assign_rooms(solver, schedule_vars, data, timeslots)

    # Convert the solved CP-SAT variables into the same {assignment_id:
    # [timeslot_id, ...]} format the violations scorer expects, then reuse it.
    csp_schedule = {ta["id"]: [] for ta in data["teacher_assignments"]}
    for ta in data["teacher_assignments"]:
        for ts in timeslots:
            if solver.Value(schedule_vars[(ta["id"], ts["id"])]) == 1:
                csp_schedule[ta["id"]].append(ts["id"])
    _vtotal, violations = score_genetic_schedule_with_violations(csp_schedule, data, lookups)

    return {
        "algorithm": ALGO_CSP,
        "status": status_name,
        "score": score,
        "schedule_entries": schedule_entries,
        "violations": violations,
    }


def _add_balance_hard_constraint(model, schedule_vars, data, lookups, timeslots, max_spread):
    """
    HARD balance constraint: for each class, (busiest day - lightest day) in
    lesson count must be <= max_spread. Reuses the same per-(day,hour) BoolVar
    sums as the other constraints. This is the ONLY structural difference from
    the standard CSP.
    """
    assignments_by_group = lookups["assignments_by_group"]
    hours_by_day = {}
    ts_by_day_hour = {}
    for ts in timeslots:
        hours_by_day.setdefault(ts["day_of_week"], []).append(ts["hour_of_day"])
        ts_by_day_hour[(ts["day_of_week"], ts["hour_of_day"])] = ts["id"]

    for group_id, assignment_ids in assignments_by_group.items():
        day_counts = []
        for day, hours in hours_by_day.items():
            cnt = model.NewIntVar(0, len(hours), f"bal_cnt_g{group_id}_d{day}")
            model.Add(cnt == sum(
                schedule_vars[(a_id, ts_by_day_hour[(day, h)])]
                for a_id in assignment_ids for h in hours
            ))
            day_counts.append(cnt)
        max_day = model.NewIntVar(0, 40, f"bal_max_g{group_id}")
        min_day = model.NewIntVar(0, 40, f"bal_min_g{group_id}")
        model.AddMaxEquality(max_day, day_counts)
        model.AddMinEquality(min_day, day_counts)
        model.Add(max_day - min_day <= max_spread)


def run_csp_balanced(data: dict, max_spread: int = 3) -> dict:
    """
    Same as run_csp, but ALSO enforces a hard balance constraint: no class may
    have more than `max_spread` hours difference between its busiest and lightest
    day. Used to produce a BALANCED feasible seed for the memetic hybrid.

    Returns the same dict shape as run_csp, with algorithm still "CSP" so the
    seed conversion and scoring treat it identically. Returns status
    "INFEASIBLE" (score None) if no schedule can satisfy the balance limit.
    """
    timeslots = data["timeslots"]
    teacher_assignments = data["teacher_assignments"]

    if not timeslots or not teacher_assignments:
        return {
            "algorithm": ALGO_CSP,
            "status": "NO_DATA",
            "score": None,
            "schedule_entries": [],
        }

    lookups = _build_lookup_maps(data)

    model = cp_model.CpModel()
    schedule_vars = _create_decision_variables(model, teacher_assignments, timeslots)

    _add_structural_hard_constraints(model, schedule_vars, data, lookups, timeslots)
    _add_student_contiguity_constraints(model, schedule_vars, data, lookups, timeslots)
    _add_admin_structure_constraints(model, schedule_vars, data, lookups, timeslots)
    _add_balance_hard_constraint(model, schedule_vars, data, lookups, timeslots, max_spread)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = CSP_MAX_SOLVE_SECONDS
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "algorithm": ALGO_CSP,
            "status": status_name,
            "score": None,
            "schedule_entries": [],
        }

    score = _compute_penalty_score(solver, schedule_vars, data, lookups, timeslots)
    schedule_entries = _assign_rooms(solver, schedule_vars, data, timeslots)

    csp_schedule = {ta["id"]: [] for ta in data["teacher_assignments"]}
    for ta in data["teacher_assignments"]:
        for ts in timeslots:
            if solver.Value(schedule_vars[(ta["id"], ts["id"])]) == 1:
                csp_schedule[ta["id"]].append(ts["id"])
    _vtotal, violations = score_genetic_schedule_with_violations(csp_schedule, data, lookups)

    return {
        "algorithm": ALGO_CSP,
        "status": status_name,
        "score": score,
        "schedule_entries": schedule_entries,
        "violations": violations,
    }
