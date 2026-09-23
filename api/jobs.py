"""
api/jobs.py

Runs the full scheduling pipeline (the same sequence main.py runs) as a
BACKGROUND job, and tracks its status in memory so the frontend can poll.

WHY A BACKGROUND JOB AT ALL
---------------------------
Each algorithm runs for up to its 60-second budget, so a full run is up
to ~3 minutes. You cannot hold an HTTP request open that long. So the
trigger endpoint starts the work in a thread and returns a job_id
immediately; the frontend polls GET /generation/{job_id} until status
becomes "completed" or "failed".

--- DELIBERATE LIMITATIONS OF THIS MINIMAL DESIGN ---------------------
* The job registry is an IN-MEMORY dict. That means:
    - Run uvicorn with a SINGLE worker (the default). With >1 worker,
      each process has its own registry and polling would hit the wrong
      one. (`uvicorn api.app:app` is single-worker; do NOT add --workers N.)
    - Job status is lost on server restart. The actual results are NOT
      lost - they're already saved to schedule_runs/schedule by the
      comparator - only the in-memory "is it done yet" record is.
* Only ONE generation may run at a time. A second trigger while one is
  running is rejected (409). This is intentional: running two pipelines
  in the same process at once would interleave their schedule_runs writes
  AND further contaminate the process-wide peak-memory measurement that
  performance.py already warns about. One-at-a-time keeps the academic
  comparison numbers meaningful.

The pipeline orchestration below duplicates ~15 lines from main.py rather
than importing main.main(). That's the same self-containment trade-off
used across the solvers, and it lets this return structured metrics
(which main.py only prints) without changing the working script.
"""

import threading
import uuid
import datetime as dt

from data_access import fetch_all_data
from performance import measure_performance
from comparator import save_and_select_best_result
from csp.solver import run_csp
from hill_climbing.solver import run_hill_climbing
from genetic.solver import run_genetic
from genetic.solver import run_genetic_memetic
from hybrid_common import get_csp_seed, get_balanced_csp_seed


# (name, function) in the same order main.py runs them.
_ALGORITHMS = [
    ("CSP", run_csp),
    ("HILL_CLIMBING", run_hill_climbing),
    ("GENETIC", run_genetic),
]

# job_id -> job dict. Guarded by _LOCK for the check-and-insert in
# start_generation_job and for all reads/writes.
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()

# A job still "running" past this many seconds is considered dead (its thread
# crashed or was killed without updating status). After this, a new generation
# is allowed instead of being blocked forever. The full pipeline is ~3x60s, so
# 6 minutes leaves generous headroom above a legitimate long run.
_MAX_JOB_SECONDS = 360


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run_full_pipeline() -> dict:
    """
    Fetches data once, runs all three solvers on the SAME data (each wrapped
    in measure_performance), saves every result and marks the best, then
    returns a structured summary.

    Returns
    -------
    {
        "comparison": { ... output of save_and_select_best_result ... },
        "metrics": {
            "CSP":           {"status", "score", "runtime_seconds", "peak_memory_mb"},
            "HILL_CLIMBING": {...},
            "GENETIC":       {...},
        },
    }

    NOTE: peak_memory_mb is process-wide and cumulative (see performance.py).
    Because all three run in one process here, later algorithms inherit the
    high-water mark of earlier ones - same caveat main.py already has. Left
    as-is on purpose rather than silently "fixing" the measurement.
    """
    data = fetch_all_data()

    results = []
    metrics = {}
    for name, algorithm_fn in _ALGORITHMS:
        result, perf = measure_performance(algorithm_fn, data)
        results.append(result)
        metrics[name] = {
            "status": result["status"],
            "score": result["score"],
            "runtime_seconds": perf["runtime_seconds"],
            "peak_memory_mb": perf["peak_memory_mb"],
        }

    comparison = save_and_select_best_result(results)
    return {"comparison": comparison, "metrics": metrics}


def _worker(job_id: str) -> None:
    try:
        output = run_full_pipeline()
        with _LOCK:
            _JOBS[job_id].update(
                status="completed",
                finished_at=_now(),
                result=output,
            )
    except Exception as exc:  # noqa: BLE001 - we want any failure surfaced to the poller
        with _LOCK:
            _JOBS[job_id].update(
                status="failed",
                finished_at=_now(),
                error=str(exc),
            )


def start_generation_job():
    """
    Starts a generation run in a background thread.

    Returns the new job_id, or None if a generation is ALREADY running
    (the caller turns None into an HTTP 409).
    """
    with _LOCK:
        now = _now()
        # A job is "really" running only if it's marked running AND started
        # recently. A stale running job (crashed thread that never updated its
        # status) is ignored so generation can't be blocked permanently.
        really_running = any(
            j["status"] == "running"
            and (now - j["started_at"]).total_seconds() < _MAX_JOB_SECONDS
            for j in _JOBS.values()
        )
        if really_running:
            return None
        # Mark any stale running jobs as failed, so they stop lingering.
        for j in _JOBS.values():
            if j["status"] == "running" and (now - j["started_at"]).total_seconds() >= _MAX_JOB_SECONDS:
                j["status"] = "failed"
                j["finished_at"] = now
                j["error"] = "Job timed out or was interrupted (stale)."
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(target=_worker, args=(job_id,), daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str):
    """Returns a COPY of the job dict, or None if unknown."""
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


# ---------------------------------------------------------------------------
# MEMETIC pipeline (CSP seed -> memetic GA). Saves ONLY the memetic result as
# the current schedule. The standard three-algorithm run already provides the
# deterministic CSP baseline, so the memetic button saves just the optimised
# result.
# ---------------------------------------------------------------------------

MEMETIC_TIME_BUDGET_SECONDS = 120.0


def run_memetic_pipeline() -> dict:
    """
    Fetches data once, gets a feasible CSP seed, runs the memetic GA from that
    seed, saves the memetic result (and marks it current) via the comparator,
    and returns a structured summary.
    """
    data = fetch_all_data()

    seed, _csp_result = get_balanced_csp_seed(data, max_spread=3)
    if seed is None:
        # Balanced CSP infeasible -> fall back to the plain CSP seed.
        seed, _csp_result = get_csp_seed(data)

    memetic_result, perf = measure_performance(
        run_genetic_memetic, data, seed,
        time_budget_seconds=MEMETIC_TIME_BUDGET_SECONDS,
    )

    from min_conflicts_repair import repair_result
    memetic_result = repair_result(memetic_result, data)
    comparison = save_and_select_best_result([memetic_result])
    

    return {
        "comparison": comparison,
        "metrics": {
            "GENETIC_MEMETIC": {
                "status": memetic_result["status"],
                "score": memetic_result["score"],
                "seed_score": memetic_result.get("seed_score"),
                "runtime_seconds": perf["runtime_seconds"],
                "peak_memory_mb": perf["peak_memory_mb"],
            }
        },
    }


def _memetic_worker(job_id: str) -> None:
    try:
        output = run_memetic_pipeline()
        with _LOCK:
            _JOBS[job_id].update(
                status="completed",
                finished_at=_now(),
                result=output,
            )
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _JOBS[job_id].update(
                status="failed",
                finished_at=_now(),
                error=str(exc),
            )


def start_memetic_job():
    """
    Starts a memetic generation as a background job. Shares the _JOBS registry
    and _LOCK with the standard pipeline, so the one-at-a-time rule covers BOTH:
    if any generation (standard OR memetic) is running, this returns None and
    the caller turns that into a 409.
    """
    with _LOCK:
        already_running = any(j["status"] == "running" for j in _JOBS.values())
        if already_running:
            return None
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(target=_memetic_worker, args=(job_id,), daemon=True)
    thread.start()
    return job_id

