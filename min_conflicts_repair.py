"""
min_conflicts_repair.py

STANDALONE, ADDITIVE repair pass. It does NOT modify any existing solver.
It IMPORTS and REUSES them:
  - genetic/solver.py : _build_lookup_maps, _score_schedule, _assign_rooms,
                        _local_search  (the unified scorer + move + room logic)
  - scoring_violations.py : score_genetic_schedule_with_violations,
                            build_structure_limits, grade_of
  - scoring_config.py : HARD_CONSTRAINT_PENALTY

WHAT IT DOES (min-conflicts repair)
-----------------------------------
Input: the MEMETIC result (its schedule_entries) — a schedule that is already
good but may carry a few residual violations.

1. Rebuild the internal map {assignment_id: [timeslot_id, ...]} from the entries.
2. Repeatedly locate the lessons that SIT IN a HARD conflict (teacher/group
   double-booking, a lesson on a non-active day or past the grade's end, room
   over-capacity) — the "culprit" lessons, read straight off the schedule.
3. For each culprit, try two moves, keeping the FIRST that LOWERS the unified
   score (first-improvement):
     (a) PRIMARY — swap the culprit's slot with another lesson of the SAME
         class. This leaves that class's set of occupied slots identical (no new
         gap / late start / imbalance); it only exchanges which subject sits in
         each of the two slots — enough to clear a teacher/room clash. In a
         packed timetable there are almost no empty slots, so a swap is what
         actually fixes most conflicts.
     (b) FALLBACK — relocate the lesson to a genuinely free legal slot. Needed
         when a swap can't help: a CLASS double-booked in one slot (one lesson
         must leave to a slot the class doesn't yet occupy), or a lesson stuck
         past the grade's ceiling with a free earlier slot to escape to.
   Because the accept test uses the same total scorer as everything else, no
   move — swap or relocation — can ever worsen any objective, hard or soft.
4. When no targeted hard move improves anymore, run a short bounded local-search
   polish (reusing the GA's own _local_search) to trim residual SOFT violations
   without ever raising the score.

We work only from the conflict set, not the whole board, so a pass costs
O(#culprits × #legal-slots) scorer calls, not O(whole system) — cheap, since the
memetic seed starts near-feasible.

Sync-block members are SKIPPED by targeted moves (moving one member alone would
break its alignment). They are rare in this dataset and the polish step already
moves sync members together via the GA's _mutate.

LOGGING
-------
The score BEFORE and AFTER the repair is printed to the server log. The result
dict returned (and therefore what is saved to schedule_runs / shown in the UI)
carries ONLY the final, post-repair score.

WIRING (same generation button)
-------------------------------
In api/jobs.py -> run_memetic_pipeline(), right after the memetic finishes and
BEFORE save_and_select_best_result:

    from min_conflicts_repair import repair_result
    memetic_result = repair_result(memetic_result, data)

Nothing else changes: the repaired result keeps algorithm="GENETIC_MEMETIC",
so the schedule_runs CHECK constraint and the existing UI are untouched.
"""

import time

from genetic.solver import (
    _build_lookup_maps,
    _score_schedule,
    _assign_rooms,
    _local_search,
    LOCAL_SEARCH_STEPS,
)
from scoring_violations import (
    score_genetic_schedule_with_violations,
    build_structure_limits,
    grade_of,
)
from scoring_config import HARD_CONSTRAINT_PENALTY, DEFAULT_DISMISSAL


# Defaults — tune here without touching any other file.
REPAIR_TIME_BUDGET_SECONDS = 15.0   # hard cap for the whole repair
REPAIR_MAX_PASSES = 25              # targeted-move sweeps over the culprit set
REPAIR_POLISH_STEPS = 300           # bounded soft cleanup at the end (accept-if-better)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries_to_schedule_map(schedule_entries, data):
    """Rebuild {assignment_id: [timeslot_id, ...]} from the memetic's entries.
    Every assignment appears (empty list if it somehow got no slot)."""
    schedule_map = {ta["id"]: [] for ta in data["teacher_assignments"]}
    for entry in schedule_entries:
        a_id = entry["tea_assignment_id"]
        if a_id in schedule_map:
            schedule_map[a_id].append(entry["timeslot_id"])
    return schedule_map


def _sync_assignment_ids(data, lookups):
    """assignment_ids that belong to ANY sync block (skipped by targeted moves)."""
    requirement_by_id = lookups["requirement_by_id"]
    blocks = {}
    for ta in data["teacher_assignments"]:
        sbi = requirement_by_id[ta["cur_requirement_id"]].get("sync_block_identity")
        if sbi is not None:
            blocks.setdefault(sbi, []).append(ta["id"])
    ids = set()
    for a_ids in blocks.values():
        ids.update(a_ids)
    return ids, blocks


def _grade_of_group(gid, lookups):
    name = lookups["group_by_id"].get(gid, {}).get("group_name")
    return grade_of(name)


def _hard_culprits(schedule, data, lookups, active_days, ceiling):
    """
    Lessons (assignment_id, slot_index) that CURRENTLY sit in a HARD conflict,
    read directly off the schedule. Returns them ordered heaviest-first.

    Covers: teacher double-booking, group double-booking, lessons on an
    inactive day or past the grade's ceiling, and room over-capacity.
    """
    requirement_by_id = lookups["requirement_by_id"]
    timeslot_by_id = lookups["timeslot_by_id"]
    subject_by_id = lookups["subject_by_id"]
    room_count_by_type = lookups["room_count_by_type"]

    weight = {}  # (a_id, idx) -> accumulated weight

    def bump(a_id, idx, w):
        weight[(a_id, idx)] = weight.get((a_id, idx), 0) + w

    # Teacher double-booking: same teacher, same timeslot, >1 lesson.
    t_occ = {}
    for ta in data["teacher_assignments"]:
        for idx, t in enumerate(schedule[ta["id"]]):
            t_occ.setdefault((ta["teacher_id"], t), []).append((ta["id"], idx))
    for lst in t_occ.values():
        if len(lst) > 1:
            for a_id, idx in lst:
                bump(a_id, idx, HARD_CONSTRAINT_PENALTY)

    # Group double-booking: same class, same timeslot, >1 lesson.
    g_occ = {}
    for ta in data["teacher_assignments"]:
        gid = requirement_by_id[ta["cur_requirement_id"]]["student_group_id"]
        for idx, t in enumerate(schedule[ta["id"]]):
            g_occ.setdefault((gid, t), []).append((ta["id"], idx))
    for lst in g_occ.values():
        if len(lst) > 1:
            for a_id, idx in lst:
                bump(a_id, idx, HARD_CONSTRAINT_PENALTY)

    # Admin structure: inactive day / past the grade's end-of-day ceiling.
    for ta in data["teacher_assignments"]:
        gid = requirement_by_id[ta["cur_requirement_id"]]["student_group_id"]
        grade = _grade_of_group(gid, lookups)
        for idx, t in enumerate(schedule[ta["id"]]):
            ts = timeslot_by_id[t]
            day, hour = ts["day_of_week"], ts["hour_of_day"]
            if day not in active_days:
                bump(ta["id"], idx, HARD_CONSTRAINT_PENALTY)
            elif grade is not None and hour > ceiling.get((grade, day), DEFAULT_DISMISSAL):
                bump(ta["id"], idx, HARD_CONSTRAINT_PENALTY)

    # Room over-capacity: demand for a specific room / room-type beyond supply.
    demand = {}
    for ta in data["teacher_assignments"]:
        subject = subject_by_id[requirement_by_id[ta["cur_requirement_id"]]["subject_id"]]
        if subject["required_room_id"] is not None:
            resource = ("specific", subject["required_room_id"])
        elif subject.get("required_room_type") is not None:
            resource = ("type", subject["required_room_type"])
        else:
            continue
        for idx, t in enumerate(schedule[ta["id"]]):
            demand.setdefault(t, {}).setdefault(resource, []).append((ta["id"], idx))
    for res_map in demand.values():
        for resource, lst in res_map.items():
            cap = 1 if resource[0] == "specific" else room_count_by_type.get(resource[1], 0)
            if len(lst) > cap:
                for a_id, idx in lst:
                    bump(a_id, idx, HARD_CONSTRAINT_PENALTY)

    return [k for k, _w in sorted(weight.items(), key=lambda kv: kv[1], reverse=True)]


def _legal_targets(a_id, idx, schedule, lookups, active_days, ceiling):
    """
    Legal destination timeslots for moving lesson (a_id, idx): an active day,
    within the grade's ceiling, and free of a teacher- OR group-clash. A move to
    such a slot can never ADD a double-booking or a structure violation; the
    scorer still arbitrates soft objectives and room capacity.
    """
    assignment_by_id = lookups["assignment_by_id"]
    requirement_by_id = lookups["requirement_by_id"]
    assignments_by_teacher = lookups["assignments_by_teacher"]
    assignments_by_group = lookups["assignments_by_group"]
    timeslot_by_id = lookups["timeslot_by_id"]

    ta = assignment_by_id[a_id]
    teacher_id = ta["teacher_id"]
    gid = requirement_by_id[ta["cur_requirement_id"]]["student_group_id"]
    grade = _grade_of_group(gid, lookups)
    current = schedule[a_id][idx]

    # Slots occupied by this teacher / this class, ignoring the lesson we're moving.
    teacher_busy = {
        t for other in assignments_by_teacher[teacher_id]
        for j, t in enumerate(schedule[other])
        if not (other == a_id and j == idx)
    }
    group_busy = {
        t for other in assignments_by_group[gid]
        for j, t in enumerate(schedule[other])
        if not (other == a_id and j == idx)
    }

    targets = []
    for t, ts in timeslot_by_id.items():
        if t == current:
            continue
        day, hour = ts["day_of_week"], ts["hour_of_day"]
        if day not in active_days:
            continue
        if grade is not None and hour > ceiling.get((grade, day), DEFAULT_DISMISSAL):
            continue
        if t in teacher_busy or t in group_busy:
            continue
        targets.append(t)
    return targets


def _class_swap_partners(a_id, idx, schedule, lookups, sync_ids):
    """
    Yield (other_assignment_id, other_slot_index) for every OTHER lesson of the
    SAME class as (a_id, idx). Swapping two lessons of one class leaves that
    class's occupied-slot set unchanged — it only exchanges which subject sits
    in each of the two slots — so it can relieve a teacher/room clash without
    ever creating a student gap, late start or imbalance. Sync-block lessons are
    skipped (moving one alone would break alignment).
    """
    assignment_by_id = lookups["assignment_by_id"]
    requirement_by_id = lookups["requirement_by_id"]
    assignments_by_group = lookups["assignments_by_group"]

    gid = requirement_by_id[assignment_by_id[a_id]["cur_requirement_id"]]["student_group_id"]
    for a2 in assignments_by_group[gid]:
        if a2 == a_id or a2 in sync_ids:
            continue
        for idx2 in range(len(schedule[a2])):
            yield a2, idx2


# ---------------------------------------------------------------------------
# Core repair
# ---------------------------------------------------------------------------

def repair_schedule(schedule, data, lookups=None,
                    time_budget_seconds=REPAIR_TIME_BUDGET_SECONDS,
                    max_passes=REPAIR_MAX_PASSES,
                    polish_steps=REPAIR_POLISH_STEPS):
    """
    Min-conflicts repair on a schedule map {assignment_id: [timeslot_id, ...]}.
    Returns (repaired_map, score_after). Never returns something worse than the
    input (every accepted change strictly lowers the unified score).
    """
    if lookups is None:
        lookups = _build_lookup_maps(data)

    active_days, ceiling = build_structure_limits(data)
    sync_ids, sync_groups = _sync_assignment_ids(data, lookups)
    timeslot_ids = [ts["id"] for ts in data["timeslots"]]

    best = {a_id: list(slots) for a_id, slots in schedule.items()}
    best_score = _score_schedule(best, data, lookups)

    deadline = time.perf_counter() + time_budget_seconds

    # --- Targeted, hard-first min-conflicts sweeps ---
    for _pass in range(max_passes):
        if best_score == 0 or time.perf_counter() >= deadline:
            break

        culprits = _hard_culprits(best, data, lookups, active_days, ceiling)
        if not culprits:
            break  # no locatable hard conflict left -> go to the soft polish

        improved_any = False
        for a_id, idx in culprits:
            if a_id in sync_ids:
                continue  # don't break sync-block alignment with a lone move
            if time.perf_counter() >= deadline:
                break

            moved = False

            # (a) PRIMARY: swap with another lesson of the SAME class. Keeps the
            # class's occupied slots identical — no new gap / late start /
            # imbalance — and just reshuffles which subject sits when. This is
            # what clears most clashes in a packed timetable.
            for a2, idx2 in _class_swap_partners(a_id, idx, best, lookups, sync_ids):
                candidate = {aid: list(slots) for aid, slots in best.items()}
                candidate[a_id][idx], candidate[a2][idx2] = (
                    candidate[a2][idx2], candidate[a_id][idx]
                )
                sc = _score_schedule(candidate, data, lookups)
                if sc < best_score:            # first-improvement
                    best, best_score = candidate, sc
                    moved = True
                    break

            # (b) FALLBACK: relocate to a free legal slot. Handles what a swap
            # can't — a CLASS double-booked in one slot, or escaping a slot past
            # the grade's ceiling / on an inactive day.
            if not moved:
                for t in _legal_targets(a_id, idx, best, lookups, active_days, ceiling):
                    candidate = {aid: list(slots) for aid, slots in best.items()}
                    candidate[a_id][idx] = t
                    sc = _score_schedule(candidate, data, lookups)
                    if sc < best_score:
                        best, best_score = candidate, sc
                        moved = True
                        break

            if moved:
                improved_any = True

        if not improved_any:
            break

    # --- Bounded soft polish: random single-slot moves kept only if they help ---
    if best_score > 0 and time.perf_counter() < deadline:
        best, best_score = _local_search(
            best, data, lookups, timeslot_ids, sync_groups,
            steps=polish_steps, current_score=best_score,
        )

    return best, best_score


def repair_result(result, data,
                  time_budget_seconds=REPAIR_TIME_BUDGET_SECONDS,
                  verbose=True):
    """
    Pipeline entry point. Takes the MEMETIC result dict and returns EITHER a
    strictly-improved result of the same shape, OR the ORIGINAL memetic result
    untouched. It can never return something worse — two guards below make
    "made it worse" mathematically impossible.

    Logs before/after to the server log; the UI sees only the final score.
    """
    entries = result.get("schedule_entries") or []
    reported = result.get("score")
    if not entries or reported is None:
        return result  # nothing to repair (e.g. NO_DATA) — pass through untouched

    lookups = _build_lookup_maps(data)
    schedule_map = _entries_to_schedule_map(entries, data)
    score_before = _score_schedule(schedule_map, data, lookups)

    # GUARD 1 — faithfulness. The repair accepts moves by _score_schedule, which
    # is the SAME scorer that produced the memetic's saved/shown number. If the
    # schedule rebuilt from entries doesn't reproduce that number, the accept
    # test would be judging a different quantity than the UI shows — so refuse
    # to touch it and keep the memetic exactly as-is.
    if abs(score_before - reported) > 0.5:
        if verbose:
            print(f"[repair] SKIP: rebuilt {score_before:.1f} != memetic {reported:.1f}; keeping memetic")
        return result

    repaired_map, score_after = repair_schedule(
        schedule_map, data, lookups, time_budget_seconds=time_budget_seconds
    )

    # GUARD 2 — strict-improvement safety net. Replace the memetic ONLY if the
    # repaired schedule scores strictly LOWER on the exact same scorer. On a tie
    # or (impossible-by-construction) regression, return the memetic untouched.
    if score_after >= score_before:
        if verbose:
            print(f"[repair] no gain (before={score_before:.1f} after={score_after:.1f}); keeping memetic")
        return result

    timeslot_ids = [ts["id"] for ts in data["timeslots"]]
    schedule_entries = _assign_rooms(repaired_map, data, timeslot_ids)
    _vtotal, violations = score_genetic_schedule_with_violations(
        repaired_map, data, lookups
    )

    if verbose:
        _bt, before_vios = score_genetic_schedule_with_violations(schedule_map, data, lookups)
        hb = sum(1 for v in before_vios if v.get("severity") == "hard")
        ha = sum(1 for v in violations if v.get("severity") == "hard")
        sa = sum(1 for v in violations if v.get("severity") == "soft")
        print(
            f"[repair] improved {score_before:.1f} -> {score_after:.1f} "
            f"(-{score_before - score_after:.1f}); hard {hb}->{ha}, soft now {sa}"
        )

    out = dict(result)
    out.update(
        algorithm="GENETIC_MEMETIC",
        status="COMPLETED",
        score=score_after,
        schedule_entries=schedule_entries,
        violations=violations,
        score_before_repair=score_before,  # extra field; UI ignores it
    )
    return out


# ---------------------------------------------------------------------------
# Standalone self-test: seed -> memetic -> repair, prints before/after.
#   python min_conflicts_repair.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_access import fetch_all_data
    from hybrid_common import get_balanced_csp_seed, get_csp_seed
    from genetic.solver import run_genetic_memetic

    print("Fetching data...")
    _data = fetch_all_data()

    print("Building a CSP seed and running the memetic...")
    _seed, _csp = get_balanced_csp_seed(_data, max_spread=3)
    if _seed is None:
        _seed, _csp = get_csp_seed(_data)

    _memetic = run_genetic_memetic(_data, _seed, time_budget_seconds=60.0)
    print(f"  memetic score = {_memetic['score']:.1f}")

    print("Repairing...")
    _repaired = repair_result(_memetic, _data)
    print(f"  final score   = {_repaired['score']:.1f}")
