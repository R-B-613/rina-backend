"""
sa_repair.py — Simulated Annealing repair pass (post-processing).

Runs AFTER the memetic GA, on its finished schedule, and tries to remove the
remaining violations / lower the score further.

Why Simulated Annealing here:
  The memetic GA already polishes every child with _local_search, which accepts
  ONLY strictly improving moves. So its result usually sits in a local minimum
  that plain hill-climbing cannot leave. Simulated Annealing sometimes accepts a
  WORSE move (probability exp(-delta / T)), which lets it climb out of that
  local minimum and find a deeper one. The temperature T decreases over time, so
  the search ends as ordinary hill-climbing.

Design rules:
  - Does NOT modify any existing file's logic: it only IMPORTS the GA's own move
    operators (_mutate, _mutate_targeted) and objective (_score_schedule), so it
    optimises exactly the same unified objective as every other algorithm.
  - Never worse: tracks the best schedule seen; if nothing beats the memetic
    score, the memetic result is returned UNCHANGED.
  - Fail-safe: any exception is logged and the memetic result is returned
    unchanged, so a bug here can never break generation.
  - Hard violations (10,000) are practically never accepted: exp(-10000/200)
    is about e^-50. SA may trade small soft penalties, never create a hard one.
"""

import math
import random
import time

from genetic.solver import (
    _build_lookup_maps,
    _score_schedule,
    _mutate,
    _mutate_targeted,
    _assign_rooms,
)
from scoring_violations import score_genetic_schedule_with_violations
from hybrid_common import csp_entries_to_schedule_map


# ---- Tunable constants (kept here so no existing config file is touched) ----
SA_TIME_BUDGET_SECONDS = 20
# Start T ~ one balance-hour penalty (BALANCE_PENALTY_PER_HOUR = 200):
# a +200 move is accepted ~37% of the time at the start.
SA_START_TEMPERATURE = 200.0
SA_END_TEMPERATURE = 1.0
# Share of moves aimed at existing violations (rest are random moves).
SA_TARGETED_MOVE_PROBABILITY = 0.7


def _build_sync_groups(data, lookups):
    """Same sync-block grouping the memetic GA builds (copied, not changed)."""
    sync_groups = {}
    for ta in data["teacher_assignments"]:
        req = lookups["requirement_by_id"][ta["cur_requirement_id"]]
        sbi = req.get("sync_block_identity")
        if sbi is not None:
            sync_groups.setdefault(sbi, []).append(ta["id"])
    return sync_groups


def run_simulated_annealing(data, start_schedule, lookups, timeslot_ids, sync_groups,
                            time_budget_seconds=SA_TIME_BUDGET_SECONDS,
                            start_temperature=SA_START_TEMPERATURE,
                            end_temperature=SA_END_TEMPERATURE):
    """
    Core SA loop. Returns (best_schedule, best_score, stats).
    Temperature cools geometrically from start_temperature to end_temperature
    over the time budget.
    """
    current = {a_id: list(slots) for a_id, slots in start_schedule.items()}
    current_score = _score_schedule(current, data, lookups)
    best = {a_id: list(slots) for a_id, slots in current.items()}
    best_score = current_score

    steps = 0
    accepted_better = 0
    accepted_worse = 0

    start = time.perf_counter()
    deadline = start + time_budget_seconds

    while True:
        now = time.perf_counter()
        if now >= deadline or best_score == 0:
            break

        progress = (now - start) / time_budget_seconds          # 0 -> 1
        temperature = start_temperature * (end_temperature / start_temperature) ** progress

        # Propose one move on a copy of the current state.
        candidate = {a_id: list(slots) for a_id, slots in current.items()}
        if random.random() < SA_TARGETED_MOVE_PROBABILITY:
            _mutate_targeted(candidate, timeslot_ids, lookups, data, sync_groups=sync_groups)
        else:
            _mutate(candidate, timeslot_ids, sync_groups=sync_groups)
        cand_score = _score_schedule(candidate, data, lookups)

        delta = cand_score - current_score
        # Metropolis rule: always accept non-worsening; accept worse with exp(-delta/T).
        if delta <= 0 or random.random() < math.exp(-delta / temperature):
            if delta < 0:
                accepted_better += 1
            elif delta > 0:
                accepted_worse += 1
            current = candidate
            current_score = cand_score
            if current_score < best_score:
                best = {a_id: list(slots) for a_id, slots in current.items()}
                best_score = current_score

        steps += 1

    stats = {
        "steps": steps,
        "accepted_better": accepted_better,
        "accepted_worse": accepted_worse,
    }
    return best, best_score, stats


def repair_with_simulated_annealing(data, memetic_result,
                                    time_budget_seconds=SA_TIME_BUDGET_SECONDS):
    """
    Entry point called from api/jobs.py right after the memetic GA.
    Takes the memetic result dict and returns a result dict of the SAME shape
    (algorithm stays "GENETIC_MEMETIC", so no DB CHECK change is needed).
    """
    if memetic_result.get("status") != "COMPLETED":
        return memetic_result

    try:
        lookups = _build_lookup_maps(data)
        timeslot_ids = [ts["id"] for ts in data["timeslots"]]
        sync_groups = _build_sync_groups(data, lookups)

        start_schedule = csp_entries_to_schedule_map(memetic_result["schedule_entries"], data)
        start_score = _score_schedule(start_schedule, data, lookups)

        # Sanity check: the converted schedule must score the same as the memetic result.
        print(f"[SA repair] memetic score={memetic_result.get('score')}, "
              f"recomputed after conversion={start_score}")
        if start_score != memetic_result.get("score"):
            print("[SA repair] WARNING: scores differ — conversion may be lossy.")

        best, best_score, stats = run_simulated_annealing(
            data, start_schedule, lookups, timeslot_ids, sync_groups,
            time_budget_seconds=time_budget_seconds,
        )
        print(f"[SA repair] {start_score} -> {best_score} | "
              f"steps={stats['steps']}, better={stats['accepted_better']}, "
              f"worse_accepted={stats['accepted_worse']}")

        if best_score >= start_score:
            print("[SA repair] no improvement — keeping memetic result unchanged.")
            return memetic_result

        schedule_entries = _assign_rooms(best, data, timeslot_ids)
        _vtotal, violations = score_genetic_schedule_with_violations(best, data, lookups)

        repaired = dict(memetic_result)
        repaired["score"] = best_score
        repaired["schedule_entries"] = schedule_entries
        repaired["violations"] = violations
        repaired["pre_repair_score"] = start_score
        return repaired

    except Exception as exc:  # fail-safe: never break generation
        print(f"[SA repair] failed, keeping memetic result: {exc!r}")
        return memetic_result
