"""
genetic/solver.py

Genetic Algorithm (GA) solver for the timetabling problem.

This module does NOT touch the database directly - it receives plain
Python data (as fetched by data_access.fetch_all_data()) and returns a
plain Python result dict, in the EXACT SAME SHAPE as csp.solver.run_csp()
and hill_climbing.solver.run_hill_climbing(), so it can be measured/saved/
compared identically.

-------------------------------------------------------------------------
DESIGN DECISIONS (see project book for full rationale)
-------------------------------------------------------------------------
- Population size: 50 individuals (tunable - see POPULATION_SIZE below)
- Individual representation: SAME as Hill Climbing -
      {assignment_id: [timeslot_id, ...]}
  one timeslot per weekly hour the assignment requires.
- Initial population: each individual generated independently using the
  same conflict-aware random method as Hill Climbing's
  _generate_initial_schedule.
- Fitness/scoring: the SAME unified scoring formula as CSP/Hill Climbing
  (scoring_config.py weights) - GA, like Hill Climbing, has no structural
  enforcement, so structural conflicts (teacher/group double-booking,
  weekly_hours correctness) must be scored too, not just data-driven
  constraints/preferences.
- Selection: TOURNAMENT SELECTION - to pick one parent, sample a small
  random subset of the population, the best individual in that subset
  wins. Chosen over a deterministic best-N/next-N pairing scheme because
  it better preserves population diversity (weaker individuals retain a
  real, if reduced, chance of reproducing) and is a standard, citable GA
  technique.
- Crossover: per-assignment inheritance - for each assignment_id, the
  child inherits that assignment's ENTIRE timeslot list from either
  parent A or parent B (random choice per assignment). This can introduce
  new conflicts where the two parents' choices don't align - expected,
  and handled the same way initial-population conflicts are handled
  (selection pressure + mutation over generations).
- Mutation: conceptually identical to Hill Climbing's neighbor-move -
  one assignment's one weekly-hour slot is moved to a different random
  timeslot. Applied probabilistically (MUTATION_RATE) to each child.
- Elitism: ENABLED - the best ELITE_COUNT individuals from each
  generation are copied UNCHANGED into the next generation, so the best
  score ever found cannot regress due to unlucky crossover/mutation in a
  later generation.
- Stopping condition: GA_TIME_BUDGET_SECONDS (60s, matching CSP and Hill
  Climbing for a fair comparison), with early exit if a perfect score (0)
  is found in any generation.

TODO (explicitly deferred for v1, same as CSP and Hill Climbing):
    - Room capacity / room-type matching (room assignment, like the other
      two algorithms, is decided AFTER the timeslot schedule is finalized)
    - sync_block_identity handling
    - Full preference scoring (gaps/free-day/consecutive/early-finish) -
      only the min/max hours-range preference check is implemented.
-------------------------------------------------------------------------
"""

import random
import time
from scoring_violations import score_genetic_schedule_with_violations, student_structure_penalties
from scoring_config import (
    HARD_CONSTRAINT_PENALTY,
    TEACHER_HARD_CONSTRAINT_PENALTY,
    SOFT_CONSTRAINT_WEIGHT_MULTIPLIER,
    OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR,
    SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR,
    PREFERENCE_WEIGHTS,
    ALGO_GENETIC,
)

# Overall time budget across ALL generations combined - matches CSP and
# Hill Climbing for a fair comparison. Tune independently later.
GA_TIME_BUDGET_SECONDS = 60.0

POPULATION_SIZE = 50

# Number of individuals sampled per tournament when selecting a parent.
TOURNAMENT_SIZE = 5

# Number of best individuals copied unchanged into the next generation.
ELITE_COUNT = 2

# Probability that a given child undergoes mutation after crossover.
MUTATION_RATE = 0.3

# Fraction of mutations that are TARGETED (balance-attacking) vs random.
# The rest stay random, to preserve exploration. Used by _mutate_targeted.
TARGETED_MUTATION_FRACTION = 0.7

# Memetic: number of hill-climbing steps applied to each child before it
# enters the population. Each step keeps a random single-slot move only if
# it lowers the total score. 0 would disable local search entirely.
LOCAL_SEARCH_STEPS = 30


# ---------------------------------------------------------------------------
# Lookup maps (identical structure to csp/solver.py and hill_climbing/solver.py)
# ---------------------------------------------------------------------------

def _build_lookup_maps(data: dict):
    curriculum_requirements = data["curriculum_requirements"]
    teacher_assignments = data["teacher_assignments"]
    teacher_constraints = data["teacher_constraints"]
    teacher_preferences = data["teacher_preferences"]
    timeslots = data["timeslots"]
    subjects = data["subjects"]
    student_groups = data["student_groups"]
    rooms = data["rooms"]

    requirement_by_id = {req["id"]: req for req in curriculum_requirements}
    assignment_by_id = {ta["id"]: ta for ta in teacher_assignments}

    assignments_by_teacher = {}
    for ta in teacher_assignments:
        assignments_by_teacher.setdefault(ta["teacher_id"], []).append(ta["id"])

    assignments_by_group = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        assignments_by_group.setdefault(req["student_group_id"], []).append(ta["id"])

    constraint_by_teacher_timeslot = {
        (c["teacher_id"], c["timeslot_id"]): c for c in teacher_constraints
    }

    preferences_by_teacher = {p["teacher_id"]: p for p in teacher_preferences}

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


# ---------------------------------------------------------------------------
# Initial individual generation (identical method to Hill Climbing's
# _generate_initial_schedule - reused here to build each population member)
# ---------------------------------------------------------------------------

def _generate_random_individual(data, lookups, timeslot_ids):
    """
    Builds ONE candidate schedule, trying to avoid obvious conflicts
    (teacher/group double-booking, room-resource limits) and aligning
    sync_block members to identical timeslots.
    """
    teacher_assignments = data["teacher_assignments"]
    requirement_by_id = lookups["requirement_by_id"]
    subject_by_id = lookups["subject_by_id"]
    room_count_by_type = lookups["room_count_by_type"]

    schedule = {}
    used_timeslots_by_teacher = {}
    used_timeslots_by_group = {}
    room_usage_by_timeslot = {}

    sync_blocks = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_blocks.setdefault(sbi, []).append(ta["id"])

    sync_followers = set()
    for sbi, a_ids in sync_blocks.items():
        for a_id in a_ids[1:]:
            sync_followers.add(a_id)

    shuffled_assignments = teacher_assignments[:]
    random.shuffle(shuffled_assignments)

    for ta in shuffled_assignments:
        if ta["id"] in sync_followers:
            continue

        req = requirement_by_id[ta["cur_requirement_id"]]
        weekly_hours = req["weekly_hours"]
        teacher_id = ta["teacher_id"]
        group_id = req["student_group_id"]
        subject = subject_by_id[req["subject_id"]]

        if subject["required_room_id"] is not None:
            resource_key = ("specific", subject["required_room_id"])
            resource_limit = 1
        elif subject.get("required_room_type") is not None:
            resource_key = ("type", subject["required_room_type"])
            resource_limit = room_count_by_type.get(subject["required_room_type"], 0)
        else:
            resource_key = None
            resource_limit = None

        teacher_used = used_timeslots_by_teacher.setdefault(teacher_id, set())
        group_used = used_timeslots_by_group.setdefault(group_id, set())

        chosen_timeslots = []
        for _ in range(weekly_hours):
            candidate = None
            for _attempt in range(20):
                t = random.choice(timeslot_ids)
                if t in teacher_used or t in group_used:
                    continue
                if resource_key is not None:
                    current_usage = room_usage_by_timeslot.get(t, {}).get(resource_key, 0)
                    if current_usage >= resource_limit:
                        continue
                candidate = t
                break
            if candidate is None:
                candidate = random.choice(timeslot_ids)

            chosen_timeslots.append(candidate)
            teacher_used.add(candidate)
            group_used.add(candidate)
            if resource_key is not None:
                usage = room_usage_by_timeslot.setdefault(candidate, {})
                usage[resource_key] = usage.get(resource_key, 0) + 1

        schedule[ta["id"]] = chosen_timeslots

    for sbi, a_ids in sync_blocks.items():
        if a_ids[0] in schedule:
            leader_slots = schedule[a_ids[0]]
            for follower_id in a_ids[1:]:
                schedule[follower_id] = list(leader_slots)

    return schedule


def _generate_initial_population(data, lookups, timeslot_ids, population_size):
    return [
        _generate_random_individual(data, lookups, timeslot_ids)
        for _ in range(population_size)
    ]


# ---------------------------------------------------------------------------
# Scoring (identical formula to csp/solver.py and hill_climbing/solver.py -
# see hill_climbing/solver.py's module docstring for why structural
# conflicts must be scored here too, unlike in CSP)
# ---------------------------------------------------------------------------

def _score_schedule(schedule, data, lookups):
    """
    Computes the FULL unified penalty score for a candidate schedule.
    Lower is better, 0 = perfect. Covers:
      - Structural: teacher/group double-booking, weekly_hours correctness
      - Room-awareness: simultaneous lessons exceeding available rooms
      - sync_block_identity: synced lessons must share identical timeslots
      - Data-driven: teacher_constraints (hard/soft)
      - Preferences: hours range, gaps, free day, early finish, consecutive
      - Subject distribution: same subject same day same group penalty
    """
    total_penalty = 0.0

    teacher_assignments = data["teacher_assignments"]
    requirement_by_id = lookups["requirement_by_id"]
    constraint_by_teacher_timeslot = lookups["constraint_by_teacher_timeslot"]
    preferences_by_teacher = lookups["preferences_by_teacher"]
    timeslot_by_id = lookups["timeslot_by_id"]
    subject_by_id = lookups["subject_by_id"]
    room_count_by_type = lookups["room_count_by_type"]

    # ---- Structural: teacher double-booking ----
    timeslot_count_by_teacher = {}
    for ta in teacher_assignments:
        teacher_id = ta["teacher_id"]
        counts = timeslot_count_by_teacher.setdefault(teacher_id, {})
        for t in schedule[ta["id"]]:
            counts[t] = counts.get(t, 0) + 1

    for teacher_id, counts in timeslot_count_by_teacher.items():
        for t, count in counts.items():
            if count > 1:
                total_penalty += (count - 1) * HARD_CONSTRAINT_PENALTY

    # ---- Structural: student group double-booking ----
    timeslot_count_by_group = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        group_id = req["student_group_id"]
        counts = timeslot_count_by_group.setdefault(group_id, {})
        for t in schedule[ta["id"]]:
            counts[t] = counts.get(t, 0) + 1

    for group_id, counts in timeslot_count_by_group.items():
        for t, count in counts.items():
            if count > 1:
                total_penalty += (count - 1) * HARD_CONSTRAINT_PENALTY

    # ---- Structural: weekly_hours correctness ----
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        expected_hours = req["weekly_hours"]
        actual_hours = len(schedule[ta["id"]])
        if actual_hours != expected_hours:
            total_penalty += abs(expected_hours - actual_hours) * HARD_CONSTRAINT_PENALTY

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
            if resource[0] == "specific":
                max_capacity = 1
            else:
                max_capacity = room_count_by_type.get(resource[1], 0)
            if count > max_capacity:
                total_penalty += (count - max_capacity) * HARD_CONSTRAINT_PENALTY

    # ---- Structural: sync_block_identity (synced lessons same timeslots) ----
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
                ref_set = set(reference_slots)
                other_set = set(other_slots)
                diff_count = len(ref_set.symmetric_difference(other_set))
                total_penalty += diff_count * HARD_CONSTRAINT_PENALTY

    # ---- Data-driven: teacher_constraints (hard/soft) ----
    for ta in teacher_assignments:
        teacher_id = ta["teacher_id"]
        for t in schedule[ta["id"]]:
            constraint = constraint_by_teacher_timeslot.get((teacher_id, t))
            if constraint is not None:
                if constraint["constraint_type"] == "hard":
                    total_penalty += TEACHER_HARD_CONSTRAINT_PENALTY
                else:
                    total_penalty += constraint["weight"] * SOFT_CONSTRAINT_WEIGHT_MULTIPLIER

    # ---- Data-driven: teacher_preferences (full implementation) ----
    assignments_by_teacher = lookups["assignments_by_teacher"]
    teacher_by_id = lookups["teacher_by_id"]
    for teacher_id, assignment_ids in assignments_by_teacher.items():
        total_hours = sum(len(schedule[a_id]) for a_id in assignment_ids)

        # Hours range: workload band from the teacher record (admin-set).
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
        for a_id in assignment_ids:
            for t in schedule[a_id]:
                ts = timeslot_by_id[t]
                teacher_day_hours.setdefault(ts["day_of_week"], []).append(ts["hour_of_day"])

        teaching_days = len(teacher_day_hours)

        # Free day preference (penalize if teacher has zero free days)
        free_days = 6 - teaching_days
        if free_days == 0 and prefs.get("priority_free_day") and prefs["priority_free_day"] > 0:
            total_penalty += PREFERENCE_WEIGHTS["free_day"] * prefs["priority_free_day"]

        # Per-day preferences: gaps, early finish, consecutive
        for day, hours in teacher_day_hours.items():
            hours_sorted = sorted(hours)
            first_hour = hours_sorted[0]
            last_hour = hours_sorted[-1]
            expected_if_no_gaps = last_hour - first_hour + 1
            actual_count = len(hours_sorted)
            gaps = expected_if_no_gaps - actual_count

            # No gaps
            if gaps > 0 and prefs.get("priority_no_gaps") and prefs["priority_no_gaps"] > 0:
                total_penalty += gaps * PREFERENCE_WEIGHTS["no_gaps"] * prefs["priority_no_gaps"]

            # Early finish (penalize teaching past hour 6)
            if last_hour > 6 and prefs.get("priority_early_finish") and prefs["priority_early_finish"] > 0:
                total_penalty += (last_hour - 6) * PREFERENCE_WEIGHTS["early_finish"] * prefs["priority_early_finish"]

            # Consecutive (also penalizes gaps, from a different preference angle)
            if gaps > 0 and prefs.get("priority_consecutive") and prefs["priority_consecutive"] > 0:
                total_penalty += gaps * PREFERENCE_WEIGHTS["consecutive"] * prefs["priority_consecutive"]

    # ---- Subject distribution across the week ----
    group_subject_day_counts = {}
    for ta in teacher_assignments:
        req = requirement_by_id[ta["cur_requirement_id"]]
        group_id = req["student_group_id"]
        subject_id = req["subject_id"]
        for t in schedule[ta["id"]]:
            ts = timeslot_by_id[t]
            key = (group_id, subject_id, ts["day_of_week"])
            group_subject_day_counts[key] = group_subject_day_counts.get(key, 0) + 1

    for key, count in group_subject_day_counts.items():
        if count > 1:
            total_penalty += (count - 1) * SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR

    total_penalty += student_structure_penalties(schedule, data, lookups)[0]

    return total_penalty


# ---------------------------------------------------------------------------
# Selection: tournament selection
# ---------------------------------------------------------------------------

def _tournament_select(population_with_scores):
    """
    Picks ONE parent via tournament selection: samples TOURNAMENT_SIZE
    individuals at random from the population, returns the one with the
    best (lowest) score among that sample.

    population_with_scores: list of (schedule, score) tuples.
    """
    contenders = random.sample(population_with_scores, TOURNAMENT_SIZE)
    contenders.sort(key=lambda pair: pair[1])
    return contenders[0][0]  # the schedule (not the score) of the winner


# ---------------------------------------------------------------------------
# Crossover: per-assignment inheritance
# ---------------------------------------------------------------------------

def _crossover(parent_a, parent_b, sync_groups=None):
    """
    Builds one child schedule. For sync_block members, all assignments in
    the same block are inherited from the SAME parent (preserving alignment).
    """
    child = {}
    assigned_from = {}

    # First pass: decide parent per sync group (whole group from one parent).
    if sync_groups:
        for sbi, a_ids in sync_groups.items():
            chosen_parent = parent_a if random.random() < 0.5 else parent_b
            for a_id in a_ids:
                child[a_id] = list(chosen_parent[a_id])
                assigned_from[a_id] = True

    # Second pass: non-synced assignments, random per-assignment as before.
    for assignment_id in parent_a:
        if assignment_id in assigned_from:
            continue
        if random.random() < 0.5:
            child[assignment_id] = list(parent_a[assignment_id])
        else:
            child[assignment_id] = list(parent_b[assignment_id])

    return child


# ---------------------------------------------------------------------------
# Mutation: identical idea to Hill Climbing's single-session move
# ---------------------------------------------------------------------------

def _mutate(schedule, timeslot_ids, sync_groups=None):
    """
    Mutates ONE schedule IN PLACE. If the mutated assignment belongs to a
    sync_block, all members of that block are moved to the same new
    timeslot (keeping them aligned).
    """
    assignment_id = random.choice(list(schedule.keys()))
    hour_index = random.randrange(len(schedule[assignment_id]))
    new_timeslot = random.choice(timeslot_ids)
    schedule[assignment_id][hour_index] = new_timeslot

    if sync_groups:
        for sbi, a_ids in sync_groups.items():
            if assignment_id in a_ids:
                for partner_id in a_ids:
                    if partner_id != assignment_id and hour_index < len(schedule[partner_id]):
                        schedule[partner_id][hour_index] = new_timeslot
                break


def _mutate_targeted(schedule, timeslot_ids, lookups, data, sync_groups=None):
    """
    TARGETED mutation aimed at reducing day-to-day imbalance (the dominant
    penalty). With probability TARGETED_MUTATION_FRACTION it performs a
    balance-directed move; otherwise it falls back to a plain random move
    (identical to _mutate) to preserve exploration.

    Balance-directed move:
      1. Pick a random class (student group).
      2. Find that class's lessons grouped by day.
      3. Identify its most-loaded day and least-loaded day.
      4. Move ONE lesson from the most-loaded day to a free slot on the
         least-loaded day (for that class), reducing the spread.

    Sync blocks: if the chosen lesson's assignment is part of a sync block, we
    fall back to a random move for that case (moving a synced block while
    respecting a specific target day/slot is more complex; keeping it simple and
    correct here — sync members are rare in this dataset).

    All moves are IN PLACE, matching _mutate's contract.
    """
    # Fallback to plain random mutation part of the time (exploration).
    if random.random() > TARGETED_MUTATION_FRACTION:
        _mutate(schedule, timeslot_ids, sync_groups=sync_groups)
        return

    requirement_by_id = lookups["requirement_by_id"]
    timeslot_by_id = lookups["timeslot_by_id"]

    # Group assignments by student group, and know each timeslot's (day, hour).
    assignments_by_group = lookups["assignments_by_group"]
    group_ids = list(assignments_by_group.keys())
    if not group_ids:
        _mutate(schedule, timeslot_ids, sync_groups=sync_groups)
        return

    # Build day -> set(all timeslot_ids that day), for finding free target slots.
    slots_by_day = {}
    for ts_id in timeslot_ids:
        ts = timeslot_by_id[ts_id]
        slots_by_day.setdefault(ts["day_of_week"], []).append(ts_id)

    # Try a few random classes until we find one with an imbalance to fix.
    for _try in range(10):
        gid = random.choice(group_ids)
        a_ids = assignments_by_group[gid]

        # Collect this class's occupied (day -> list of (assignment_id, hour_index, timeslot_id)).
        day_lessons = {}
        occupied_slots = set()  # timeslot_ids this class already uses (avoid clashes)
        for a_id in a_ids:
            for idx, t in enumerate(schedule[a_id]):
                ts = timeslot_by_id[t]
                day_lessons.setdefault(ts["day_of_week"], []).append((a_id, idx, t))
                occupied_slots.add(t)

        if len(day_lessons) < 2:
            continue  # need at least two days to move between

        # Most- and least-loaded days for this class.
        days_sorted = sorted(day_lessons.keys(), key=lambda d: len(day_lessons[d]))
        light_day = days_sorted[0]
        heavy_day = days_sorted[-1]
        if len(day_lessons[heavy_day]) - len(day_lessons[light_day]) < 2:
            continue  # already fairly balanced for this class; try another

        # Pick a lesson on the heavy day to move.
        a_id, hour_index, _old_t = random.choice(day_lessons[heavy_day])

        # Skip sync-block members (keep it simple/correct) -> random fallback.
        if sync_groups and any(a_id in ids for ids in sync_groups.values()):
            _mutate(schedule, timeslot_ids, sync_groups=sync_groups)
            return

        # Find a FREE slot on the light day for this class (not already used by it).
        candidates = [t for t in slots_by_day.get(light_day, []) if t not in occupied_slots]
        if not candidates:
            continue  # light day full for this class; try another class

        new_timeslot = random.choice(candidates)
        schedule[a_id][hour_index] = new_timeslot
        return  # done: one targeted move made

    # If we couldn't find a good targeted move, fall back to random.
    _mutate(schedule, timeslot_ids, sync_groups=sync_groups)



# ---------------------------------------------------------------------------
# Room assignment (identical simple v1 approach to CSP/Hill Climbing)
# ---------------------------------------------------------------------------

def _assign_rooms(schedule, data, timeslot_ids):
    """
    Smart room assignment, applied AFTER the timeslot schedule is finalized.
    Three-tier priority per scheduled session:
      1. Subject has a specific required room (required_room_id) -> use it
      2. Subject needs a room TYPE (required_room_type) -> any free room
         of that type
      3. Default -> group's home_room_id, falling back to any free room
         with sufficient capacity
    All tiers check: room is free at this timeslot AND capacity >= group size.
    """
    rooms = data["rooms"]
    subjects = data["subjects"]
    student_groups = data["student_groups"]
    teacher_assignments = data["teacher_assignments"]
    curriculum_requirements = data["curriculum_requirements"]

    room_by_id = {r["id"]: r for r in rooms}
    subject_by_id = {s["id"]: s for s in subjects}
    group_by_id = {g["id"]: g for g in student_groups}
    requirement_by_id = {req["id"]: req for req in curriculum_requirements}
    assignment_by_id = {ta["id"]: ta for ta in teacher_assignments}

    schedule_entries = []
    used_rooms_by_timeslot = {t: set() for t in timeslot_ids}

    for assignment_id, timeslots in schedule.items():
        ta = assignment_by_id[assignment_id]
        req = requirement_by_id[ta["cur_requirement_id"]]
        subject = subject_by_id[req["subject_id"]]
        group = group_by_id[req["student_group_id"]]

        for t in timeslots:
            assigned_room = None

            if subject["required_room_id"] is not None:
                # Priority 1: subject needs ONE specific dedicated room.
                room_id = subject["required_room_id"]
                room = room_by_id.get(room_id)
                if (room
                        and room_id not in used_rooms_by_timeslot[t]
                        and room["capacity"] >= group["student_count"]):
                    assigned_room = room_id
                # If busy or too small: stays None (a real conflict).

            elif subject.get("required_room_type") is not None:
                # Priority 2: subject needs any room of a certain type.
                for room in rooms:
                    if (room.get("room_type") == subject["required_room_type"]
                            and room["id"] not in used_rooms_by_timeslot[t]
                            and room["capacity"] >= group["student_count"]):
                        assigned_room = room["id"]
                        break
                # If all rooms of this type are busy: stays None.

            else:
                # Priority 3: use the group's home (parent) classroom.
                home_room_id = group.get("home_room_id")
                if home_room_id is not None:
                    room = room_by_id.get(home_room_id)
                    if (room
                            and home_room_id not in used_rooms_by_timeslot[t]
                            and room["capacity"] >= group["student_count"]):
                        assigned_room = home_room_id

                # Fallback: any free room with sufficient capacity.
                if assigned_room is None:
                    for room in rooms:
                        if (room["id"] not in used_rooms_by_timeslot[t]
                                and room["capacity"] >= group["student_count"]):
                            assigned_room = room["id"]
                            break

            if assigned_room is not None:
                used_rooms_by_timeslot[t].add(assigned_room)

            schedule_entries.append(
                {
                    "timeslot_id": t,
                    "tea_assignment_id": assignment_id,
                    "room_id": assigned_room,
                }
            )

    return schedule_entries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_genetic(data: dict) -> dict:
    """
    Main entry point for the Genetic Algorithm.

    Parameters
    ----------
    data : dict
        The exact dict returned by data_access.fetch_all_data().

    Returns
    -------
    dict shaped EXACTLY like csp.solver.run_csp() and
    hill_climbing.solver.run_hill_climbing()'s return values:
        {
            "algorithm": "GENETIC",
            "status": "COMPLETED" | "NO_DATA",
            "score": float,
            "schedule_entries": [ {...}, ... ],
        }
    """
    timeslots = data["timeslots"]
    teacher_assignments = data["teacher_assignments"]

    if not timeslots or not teacher_assignments:
        return {
            "algorithm": ALGO_GENETIC,
            "status": "NO_DATA",
            "score": None,
            "schedule_entries": [],
        }

    lookups = _build_lookup_maps(data)
    timeslot_ids = [ts["id"] for ts in timeslots]

    # Pre-compute sync groups for sync-aware crossover and mutation.
    sync_groups = {}
    for ta in teacher_assignments:
        req = lookups["requirement_by_id"][ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_groups.setdefault(sbi, []).append(ta["id"])

    deadline = time.perf_counter() + GA_TIME_BUDGET_SECONDS

    population = _generate_initial_population(data, lookups, timeslot_ids, POPULATION_SIZE)

    best_schedule = None
    best_score = None
    generation_count = 0

    while time.perf_counter() < deadline:
        generation_count += 1

        # Score the whole population once per generation.
        population_with_scores = [
            (individual, _score_schedule(individual, data, lookups))
            for individual in population
        ]
        population_with_scores.sort(key=lambda pair: pair[1])

        # Track the best individual ever seen (across all generations).
        generation_best_schedule, generation_best_score = population_with_scores[0]
        if best_score is None or generation_best_score < best_score:
            best_schedule = generation_best_schedule
            best_score = generation_best_score

        if best_score == 0:
            break

        # Elitism: carry the best ELITE_COUNT individuals over unchanged.
        next_population = [
            individual for individual, _score in population_with_scores[:ELITE_COUNT]
        ]

        # Fill the rest of the next generation via selection + crossover + mutation.
        while len(next_population) < POPULATION_SIZE:
            if time.perf_counter() >= deadline:
                break

            parent_a = _tournament_select(population_with_scores)
            parent_b = _tournament_select(population_with_scores)

            child = _crossover(parent_a, parent_b, sync_groups=sync_groups)

            if random.random() < MUTATION_RATE:
                _mutate(child, timeslot_ids, sync_groups=sync_groups)

            next_population.append(child)

        population = next_population

    schedule_entries = _assign_rooms(best_schedule, data, timeslot_ids)
    
    _vtotal, violations = score_genetic_schedule_with_violations(best_schedule, data, lookups)
    if abs(_vtotal - best_score) > 0.5:
        print(f"[warn] violations total {_vtotal} != best_score {best_score}")

    return {
        "algorithm": ALGO_GENETIC,
        "status": "COMPLETED",
        "score": best_score,
        "schedule_entries": schedule_entries,
        "violations": violations,
    }


def _make_seeded_population(data, lookups, timeslot_ids, seed_schedule,
                            population_size, seed_fraction, seed_mutations,
                            sync_groups):
    """
    Build the initial population: seed_fraction of it as lightly-mutated copies
    of seed_schedule, the rest fully random.

    Each seeded individual is a fresh copy of the CSP seed with `seed_mutations`
    random single-slot moves applied (via the existing _mutate), so the seeded
    portion is diverse rather than identical clones.
    """
    n_seeded = int(round(population_size * seed_fraction))
    n_seeded = max(1, min(population_size, n_seeded))  # keep in [1, population_size]

    population = []

    # Seeded individuals: copy the seed, apply a few mutations for diversity.
    for _ in range(n_seeded):
        individual = {a_id: list(slots) for a_id, slots in seed_schedule.items()}
        for _m in range(seed_mutations):
            _mutate(individual, timeslot_ids, sync_groups=sync_groups)
        population.append(individual)

    # Remaining individuals: fully random (existing generator).
    while len(population) < population_size:
        population.append(
            _generate_random_individual(data, lookups, timeslot_ids)
        )

    return population


def run_genetic_from_seed(data: dict, seed_schedule: dict,
                          time_budget_seconds: float = None,
                          seed_fraction: float = 0.8,
                          seed_mutations: int = 3) -> dict:
    """
    HYBRID entry point: Genetic Algorithm whose initial population is seeded from
    the CSP schedule (default 80% seeded / 20% random), instead of 100% random.

    Reuses the exact same GA loop and operators as run_genetic — only the
    INITIAL POPULATION differs. Because most of the population starts around
    CSP's feasible schedule, the GA optimises soft constraints from a feasible
    base rather than struggling to reach feasibility at all.

    Parameters
    ----------
    data : dict
        The fetch_all_data() dict.
    seed_schedule : dict
        {assignment_id: [timeslot_id, ...]} — CSP's feasible schedule (from
        hybrid_common.get_csp_seed).
    time_budget_seconds : float, optional
        Defaults to GA_TIME_BUDGET_SECONDS.
    seed_fraction : float
        Fraction of the population seeded from CSP (default 0.8 = 80/20).
    seed_mutations : int
        Random single-slot moves applied to each seeded individual for diversity.

    Returns
    -------
    dict shaped like run_genetic()'s result, plus "seed_score":
        {
            "algorithm": "GENETIC_HYBRID",
            "status": "COMPLETED",
            "score": float,             # best score found
            "seed_score": float,        # score of the CSP seed before GA
            "schedule_entries": [...],
            "violations": [...],
        }
    """
    if time_budget_seconds is None:
        time_budget_seconds = GA_TIME_BUDGET_SECONDS

    lookups = _build_lookup_maps(data)
    timeslot_ids = [ts["id"] for ts in data["timeslots"]]

    # Pre-compute sync groups (same as run_genetic).
    sync_groups = {}
    for ta in data["teacher_assignments"]:
        req = lookups["requirement_by_id"][ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_groups.setdefault(sbi, []).append(ta["id"])

    seed_score = _score_schedule(seed_schedule, data, lookups)

    deadline = time.perf_counter() + time_budget_seconds

    population = _make_seeded_population(
        data, lookups, timeslot_ids, seed_schedule,
        POPULATION_SIZE, seed_fraction, seed_mutations, sync_groups
    )

    # Start best = the seed itself, so the hybrid never does worse than CSP.
    best_schedule = {a_id: list(slots) for a_id, slots in seed_schedule.items()}
    best_score = seed_score

    while time.perf_counter() < deadline:
        population_with_scores = [
            (individual, _score_schedule(individual, data, lookups))
            for individual in population
        ]
        population_with_scores.sort(key=lambda pair: pair[1])

        generation_best_schedule, generation_best_score = population_with_scores[0]
        if generation_best_score < best_score:
            best_schedule = generation_best_schedule
            best_score = generation_best_score

        if best_score == 0:
            break

        next_population = [
            individual for individual, _score in population_with_scores[:ELITE_COUNT]
        ]

        while len(next_population) < POPULATION_SIZE:
            if time.perf_counter() >= deadline:
                break
            parent_a = _tournament_select(population_with_scores)
            parent_b = _tournament_select(population_with_scores)
            child = _crossover(parent_a, parent_b, sync_groups=sync_groups)
            if random.random() < MUTATION_RATE:
                _mutate_targeted(child, timeslot_ids, lookups, data, sync_groups=sync_groups)
            next_population.append(child)

        population = next_population

    schedule_entries = _assign_rooms(best_schedule, data, timeslot_ids)
    _vtotal, violations = score_genetic_schedule_with_violations(
        best_schedule, data, lookups
    )

    return {
        "algorithm": "GENETIC_HYBRID",
        "status": "COMPLETED",
        "score": best_score,
        "seed_score": seed_score,
        "schedule_entries": schedule_entries,
        "violations": violations,
    }


def _local_search(schedule, data, lookups, timeslot_ids, sync_groups,
                  steps, current_score=None):
    """
    Embedded hill-climbing: perform up to `steps` random single-slot moves,
    keeping each move ONLY if it strictly lowers the total score. Operates on a
    COPY, returns (best_schedule, best_score).
 
    Reuses the GA's own _mutate (for the move) and _score_schedule (for the
    total), so it optimises the SAME unified objective as everything else —
    balance, dismissal, holy, distribution, constraints, preferences — all at
    once. That's what makes it safe: it can never accept a move that worsens the
    total, unlike a forced targeted move.
    """
    best = {a_id: list(slots) for a_id, slots in schedule.items()}
    best_score = current_score if current_score is not None else _score_schedule(best, data, lookups)
 
    for _ in range(steps):
        if best_score == 0:
            break
        # Propose one random single-slot move on a fresh copy.
        candidate = {a_id: list(slots) for a_id, slots in best.items()}
        _mutate(candidate, timeslot_ids, sync_groups=sync_groups)
        cand_score = _score_schedule(candidate, data, lookups)
        if cand_score < best_score:
            best = candidate
            best_score = cand_score
 
    return best, best_score
 
 
def run_genetic_memetic(data: dict, seed_schedule: dict,
                        time_budget_seconds: float = None,
                        seed_fraction: float = 0.8,
                        seed_mutations: int = 3,
                        local_search_steps: int = None) -> dict:
    """
    MEMETIC hybrid: identical to run_genetic_from_seed (CSP-seeded GA), but each
    child is refined by _local_search (a few hill-climbing steps) before being
    added to the next generation.
 
    Everything else — seeded population, tournament selection, crossover,
    mutation, elitism, best-starts-at-seed — is the same as run_genetic_from_seed
    so the ONLY difference under test is the embedded local search.
 
    Returns the same dict shape as run_genetic_from_seed, with
    algorithm = "GENETIC_MEMETIC".
    """
    if time_budget_seconds is None:
        time_budget_seconds = GA_TIME_BUDGET_SECONDS
    if local_search_steps is None:
        local_search_steps = LOCAL_SEARCH_STEPS
 
    lookups = _build_lookup_maps(data)
    timeslot_ids = [ts["id"] for ts in data["timeslots"]]
 
    sync_groups = {}
    for ta in data["teacher_assignments"]:
        req = lookups["requirement_by_id"][ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_groups.setdefault(sbi, []).append(ta["id"])
 
    seed_score = _score_schedule(seed_schedule, data, lookups)
    deadline = time.perf_counter() + time_budget_seconds
 
    population = _make_seeded_population(
        data, lookups, timeslot_ids, seed_schedule,
        POPULATION_SIZE, seed_fraction, seed_mutations, sync_groups
    )
 
    # Best starts at the seed, so the memetic hybrid never does worse than CSP.
    best_schedule = {a_id: list(slots) for a_id, slots in seed_schedule.items()}
    best_score = seed_score
 
    while time.perf_counter() < deadline:
        population_with_scores = [
            (individual, _score_schedule(individual, data, lookups))
            for individual in population
        ]
        population_with_scores.sort(key=lambda pair: pair[1])
 
        generation_best_schedule, generation_best_score = population_with_scores[0]
        if generation_best_score < best_score:
            best_schedule = generation_best_schedule
            best_score = generation_best_score
 
        if best_score == 0:
            break
 
        next_population = [
            individual for individual, _score in population_with_scores[:ELITE_COUNT]
        ]
 
        while len(next_population) < POPULATION_SIZE:
            if time.perf_counter() >= deadline:
                break
            parent_a = _tournament_select(population_with_scores)
            parent_b = _tournament_select(population_with_scores)
            child = _crossover(parent_a, parent_b, sync_groups=sync_groups)
            if random.random() < MUTATION_RATE:
                _mutate(child, timeslot_ids, sync_groups=sync_groups)
            # --- MEMETIC STEP: polish the child with embedded hill-climbing ---
            child, _child_score = _local_search(
                child, data, lookups, timeslot_ids, sync_groups,
                steps=local_search_steps
            )
            next_population.append(child)
 
        population = next_population
 
    schedule_entries = _assign_rooms(best_schedule, data, timeslot_ids)
    _vtotal, violations = score_genetic_schedule_with_violations(
        best_schedule, data, lookups
    )
 
    return {
        "algorithm": "GENETIC_MEMETIC",
        "status": "COMPLETED",
        "score": best_score,
        "seed_score": seed_score,
        "schedule_entries": schedule_entries,
        "violations": violations,
    }


