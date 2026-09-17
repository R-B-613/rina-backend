"""
api/schemas.py

Pydantic models for request bodies and responses. Keeping them in one
file makes the entire API surface readable at a glance.

These are written for Pydantic v2 (the version FastAPI installs today).
"""

from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str          # matched against auth_db.LOGIN_COLUMN (email by default)
    password: str


class TeacherProfile(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False
    email: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    teacher: TeacherProfile


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class AdminResetPasswordRequest(BaseModel):
    teacher_id: int
    new_password: str = Field(min_length=8)


class MessageResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class GenerationStartedResponse(BaseModel):
    job_id: str
    status: str            # "running"


class GenerationStatusResponse(BaseModel):
    job_id: str
    status: str            # "running" | "completed" | "failed"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    # On success, result holds the comparator output plus per-algorithm
    # runtime/memory metrics. On failure, error holds the message.
    result: Optional[dict] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Manual schedule editing (admin drag & drop)
# ---------------------------------------------------------------------------

class ScheduleEntryEdit(BaseModel):
    tea_assignment_id: int
    timeslot_id: int
    room_id: Optional[int] = None


class ScheduleEditRequest(BaseModel):
    entries: List[ScheduleEntryEdit]
