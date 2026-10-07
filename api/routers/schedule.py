"""
api/routers/schedule.py
Schedule views + the admin "publish to staff" action.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_current_teacher, get_current_admin
from api.schedule_db import (
    get_current_run,
    get_published_run,
    get_schedule_entries,
    publish_run,
)
from api.edit_db import preview_violations, save_run_entries
from api.run_snapshot import run_exists
from api.settings_status import settings_status
from api.schemas import ScheduleEditRequest

router = APIRouter(prefix="/schedule", tags=["schedule"])


def _payload(run, teacher_id=None):
    if run is None:
        return {"run": None, "entries": []}
    entries = get_schedule_entries(run["id"], teacher_id=teacher_id)
    return {"run": run, "entries": entries}


@router.get("/current")
def current_schedule(admin: dict = Depends(get_current_admin)):
    return _payload(get_current_run())


@router.get("/me")
def my_schedule(teacher: dict = Depends(get_current_teacher)):
    return _payload(get_published_run(), teacher_id=teacher["id"])


@router.post("/publish")
def publish_current(admin: dict = Depends(get_current_admin)):
    run = get_current_run()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is no schedule to publish yet",
        )
    publish_run(run["id"])
    return {"detail": "published", "run_id": run["id"]}

@router.get("/settings-status")
def current_settings_status(admin: dict = Depends(get_current_admin)):
    """Do today's settings still match those the current / published run was generated from?"""
    return settings_status()

@router.get("/violations")
def current_violations(admin: dict = Depends(get_current_admin)):
    run = get_current_run()
    if run is None:
        return {"run": None, "violations": []}
    return {
        "run": {"id": run["id"], "algorithm": run["algorithm"], "score": run["score"], "is_published": run["is_published"]},
        "violations": run.get("violations") or [],
    }


@router.post("/preview-violations")
def preview_schedule_violations(payload: ScheduleEditRequest, admin: dict = Depends(get_current_admin)):
    """Score a proposed (unsaved) schedule and return its violations."""
    entries = [e.model_dump() for e in payload.entries]
    current = get_current_run()
    score, violations = preview_violations(entries, run_id=current["id"] if current else None)
    return {"score": score, "violations": violations}

@router.get("/run/{run_id}/entries")
def run_entries(run_id: int, admin: dict = Depends(get_current_admin)):
    """Lessons of ANY saved run (history viewer), resolved from that run's own settings snapshot."""
    if not run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule run not found")
    return {"entries": get_schedule_entries(run_id)}

@router.put("/run/{run_id}/entries")
def save_schedule_edits(run_id: int, payload: ScheduleEditRequest, admin: dict = Depends(get_current_admin)):
    """Persist an edited schedule. Only the current selected run may be edited."""
    current = get_current_run()
    if current is None or current["id"] != run_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only the current schedule can be edited",
        )
    entries = [e.model_dump() for e in payload.entries]
    score, violations = save_run_entries(run_id, entries)
    return {"detail": "saved", "run_id": run_id, "score": score, "violations": violations}
