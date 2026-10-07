"""
comparator.py

Takes the results of all 3 scheduling algorithms (CSP, Hill Climbing,
Genetic Algorithm), saves EVERY one of them to the database as its own
schedule_runs + schedule rows, then marks the single best-scoring run as
is_selected = true.

Kept as a separate module (rather than inlined in main.py) so it can be
reused identically from main.py (the command-line script) AND, later,
from a FastAPI endpoint - both callers just need to call
save_and_select_best_result(results), with no duplicated logic.
"""

from data_access import save_schedule_run, mark_run_as_selected


def save_and_select_best_result(results: list, config_snapshot: str = None) -> dict:
    """
    Parameters
    ----------
    results : list of result dicts, each shaped like:
        {
            "algorithm": "CSP" | "HILL_CLIMBING" | "GENETIC",
            "status": ...,
            "score": float or None,
            "schedule_entries": [ {...}, ... ],
        }
        (the exact return shape of run_csp / run_hill_climbing / run_genetic)

    Behavior
    --------
    - Every result with a non-None score gets saved to the database via
      save_schedule_run() - so ALL algorithm runs are visible afterward
      in schedule_runs/schedule, not just the winner.
    - Results with score=None (e.g. status="NO_DATA") are skipped - there
      is nothing meaningful to save.
    - The result with the LOWEST score among the saved ones is marked
      is_selected = true via mark_run_as_selected().

    Returns
    -------
    A dict describing what happened:
        {
            "saved_run_ids": {algorithm_name: run_id, ...},
            "best_algorithm": str or None,
            "best_run_id": int or None,
            "best_score": float or None,
        }
    """
    saved_run_ids = {}
    best_algorithm = None
    best_run_id = None
    best_score = None

    for result in results:
        if result["score"] is None:
            # Nothing meaningful to save (e.g. NO_DATA) - skip.
            continue

        run_id = save_schedule_run(
            algorithm=result["algorithm"],
            score=result["score"],
            schedule_entries=result["schedule_entries"],
            violations=result.get("violations"),
            config_snapshot=config_snapshot,
        )
        saved_run_ids[result["algorithm"]] = run_id

        if best_score is None or result["score"] < best_score:
            best_score = result["score"]
            best_algorithm = result["algorithm"]
            best_run_id = run_id

    if best_run_id is not None:
        mark_run_as_selected(best_run_id)

    return {
        "saved_run_ids": saved_run_ids,
        "best_algorithm": best_algorithm,
        "best_run_id": best_run_id,
        "best_score": best_score,
    }
