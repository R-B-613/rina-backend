"""
hybrid_common.py

Shared helpers for the HYBRID pipeline (CSP feasible seed -> HC/GA soft-constraint
optimisation). This file is ENTIRELY NEW and ADDITIVE: it imports and reuses the
existing solvers without modifying any of them. The existing per-algorithm
pipeline (main.py, jobs.py, the UI) is completely unaffected.

The one genuinely new piece of logic here is `csp_entries_to_schedule_map`,
which turns CSP's returned `schedule_entries` (a list of
{timeslot_id, tea_assignment_id, room_id}) into the
`{assignment_id: [timeslot_id, ...]}` dict that Hill Climbing and the Genetic
Algorithm use internally as a candidate schedule. We reconstruct it from CSP's
public output rather than modifying run_csp to expose its internal map, so CSP
stays byte-for-byte untouched.
"""

from csp.solver import run_csp, run_csp_balanced
from genetic.solver import _build_lookup_maps
from scoring_violations import score_genetic_schedule_with_violations


def csp_entries_to_schedule_map(schedule_entries, data):
    """
    Rebuild the {assignment_id: [timeslot_id, ...]} representation from CSP's
    `schedule_entries` output.

    Every teacher_assignment starts with an empty list (so an assignment that
    somehow got no slot still appears, exactly as the solvers expect), then we
    append each entry's timeslot to its assignment.

    Parameters
    ----------
    schedule_entries : list of {"timeslot_id", "tea_assignment_id", "room_id"}
        The `schedule_entries` field from run_csp()'s result.
    data : dict
        The fetch_all_data() dict (used to enumerate all assignment ids).

    Returns
    -------
    dict: {assignment_id: [timeslot_id, ...]}
    """
    schedule_map = {ta["id"]: [] for ta in data["teacher_assignments"]}
    for entry in schedule_entries:
        a_id = entry["tea_assignment_id"]
        if a_id in schedule_map:
            schedule_map[a_id].append(entry["timeslot_id"])
    return schedule_map


def get_csp_seed(data):
    """
    Runs CSP once and returns (seed_schedule_map, csp_result).

    seed_schedule_map is in the HC/GA internal format, ready to be used as a
    starting point. csp_result is CSP's full result dict (so the caller can see
    CSP's own score/violations for comparison).

    Returns (None, csp_result) if CSP failed to produce a schedule (e.g.
    INFEASIBLE) - the caller should handle that.
    """
    csp_result = run_csp(data)
    if csp_result.get("score") is None or not csp_result.get("schedule_entries"):
        return None, csp_result
    seed_map = csp_entries_to_schedule_map(csp_result["schedule_entries"], data)
    return seed_map, csp_result


def verify_seed_matches_csp(data):
    """
    SELF-TEST for step 1. Confirms the reconstructed seed_map scores IDENTICALLY
    to CSP's own reported violations - i.e. the conversion is faithful.
    """
    seed_map, csp_result = get_csp_seed(data)
    if seed_map is None:
        print("FAIL: CSP did not produce a schedule (status="
              f"{csp_result.get('status')}). Cannot verify seed.")
        return False

    lookups = _build_lookup_maps(data)
    seed_total, seed_violations = score_genetic_schedule_with_violations(
        seed_map, data, lookups
    )

    # Compare against CSP's own violations count (same scorer, so should match).
    csp_violations = csp_result.get("violations") or []
    csp_hard = sum(1 for v in csp_violations if v.get("severity") == "hard")
    csp_soft = sum(1 for v in csp_violations if v.get("severity") == "soft")
    seed_hard = sum(1 for v in seed_violations if v.get("severity") == "hard")
    seed_soft = sum(1 for v in seed_violations if v.get("severity") == "soft")

    print("=" * 56)
    print("STEP 1 SELF-TEST: CSP schedule -> seed map conversion")
    print("=" * 56)
    print(f"  CSP reported : hard={csp_hard}  soft={csp_soft}  "
          f"total_violations={len(csp_violations)}")
    print(f"  Seed rescored: hard={seed_hard}  soft={seed_soft}  "
          f"total_violations={len(seed_violations)}")
    print(f"  Seed penalty total (via violations scorer): {seed_total}")

    ok = (csp_hard == seed_hard and csp_soft == seed_soft
          and len(csp_violations) == len(seed_violations))
    print("\n  RESULT:", "PASS - seed is a faithful copy of CSP's schedule."
          if ok else "FAIL - seed does NOT match CSP; do not proceed.")
    print("=" * 56)
    return ok


def get_balanced_csp_seed(data, max_spread=3):
    """
    Like get_csp_seed, but runs the BALANCE-CONSTRAINED CSP (no class has more
    than `max_spread` day-spread). Returns (seed_schedule_map, csp_result), or
    (None, csp_result) if the balanced CSP is infeasible.
    """
    csp_result = run_csp_balanced(data, max_spread=max_spread)
    if csp_result.get("score") is None or not csp_result.get("schedule_entries"):
        return None, csp_result
    seed_map = csp_entries_to_schedule_map(csp_result["schedule_entries"], data)
    return seed_map, csp_result


if __name__ == "__main__":
    # Running this file directly performs the step-1 self-test.
    from data_access import fetch_all_data
    print("Fetching data and verifying CSP->seed conversion...\n")
    _data = fetch_all_data()
    verify_seed_matches_csp(_data)
