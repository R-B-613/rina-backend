"""
api/routers/generation.py

Trigger a timetable generation run and poll its status.

  GET  /generation/preflight  admin only  -> check the data BEFORE generating
  POST /generation/run        admin only  -> start a run, get a job_id (202)
  POST /generation/run-memetic admin only -> start a memetic run (202)
  GET  /generation/{job_id}   any teacher -> poll status/result

Authorization choice: triggering is admin-only (it's heavy and writes new
schedule_runs - this matches the principal-driven generation workflow).
Polling is allowed for any authenticated teacher, since seeing "a new
timetable is being generated" is harmless. Tighten the GET to admin-only
by swapping get_current_teacher for get_current_admin if you prefer.

PRE-GENERATION CHECK
--------------------
Both POST endpoints run preflight.run_preflight() first and answer 422 (with
the full issue list) if any issue has severity "error". The check lives here,
on the server, and not only in the UI, because the data is edited from many
screens at different times and there are two ways to start a run.
"""

import traceback

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import GenerationStartedResponse, GenerationStatusResponse
from api.deps import get_current_admin, get_current_teacher
from api.jobs import start_generation_job, start_memetic_job, get_job
from data_access import fetch_all_data
from preflight import run_preflight, summarize

router = APIRouter(prefix="/generation", tags=["generation"])


def _block_if_preflight_fails():
    """
    Runs the pre-generation checks and raises 422 {message, issues} if any
    ERROR is found. Warnings never block.

    If the CHECK ITSELF crashes (a bug in preflight.py, not bad school data),
    it is logged and generation is allowed to proceed: a safety net must never
    be the reason the main flow is down. The traceback shows in the 8001 log.
    """
    try:
        report = summarize(run_preflight(fetch_all_data()))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return
    if not report["can_generate"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "לא ניתן להתחיל ביצירת המערכת",
                "issues": report["issues"],
            },
        )


# NOTE: this route MUST stay above "/{job_id}" - otherwise FastAPI matches
# "/generation/preflight" as a job_id of "preflight".
@router.get("/preflight")
def generation_preflight(admin: dict = Depends(get_current_admin)):
    return summarize(run_preflight(fetch_all_data()))


@router.post(
    "/run",
    response_model=GenerationStartedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_generation(admin: dict = Depends(get_current_admin)):
    _block_if_preflight_fails()
    job_id = start_generation_job()
    if job_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A generation run is already in progress",
        )
    return GenerationStartedResponse(job_id=job_id, status="running")

@router.post(
    "/run-memetic",
    response_model=GenerationStartedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_memetic_generation(admin: dict = Depends(get_current_admin)):
    _block_if_preflight_fails()
    job_id = start_memetic_job()
    if job_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A generation run is already in progress",
        )
    return GenerationStartedResponse(job_id=job_id, status="running")


@router.get("/{job_id}", response_model=GenerationStatusResponse)
def generation_status(
    job_id: str,
    teacher: dict = Depends(get_current_teacher),
):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found (it may have been lost on a server restart)",
        )
    return GenerationStatusResponse(**job)
