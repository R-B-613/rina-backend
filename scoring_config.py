"""
scoring_config.py

Central place for every weight/penalty constant used to score a generated
schedule. ALL THREE algorithms (CSP, Genetic Algorithm, Hill Climbing) import
from this file, so that the final scores they produce are comparable on the
same scale.

Design idea (as discussed):
- Hard constraints get a very large penalty if violated, so the algorithms
  always prefer to satisfy them first.
- Soft constraints (from teacher_constraints with type='soft') use their
  own per-row `weight` column (1-10) from the DB.
- teacher_preferences has no weight column in the DB on purpose - the
  weights for each kind of preference are defined here in code instead,
  so they can be tuned without touching the database.
"""

# ---------------------------------------------------------------------------
# Hard constraint penalty
# ---------------------------------------------------------------------------
# Used for any hard rule violation (e.g. a hard row in teacher_constraints).
# Kept extremely high on purpose: even satisfying *every* soft constraint
# and preference perfectly should never be able to outweigh breaking one
# hard constraint.
HARD_CONSTRAINT_PENALTY = 10_000

# A teacher's "cannot teach" — strong, but below structural 10,000 so the
# timetable stays solvable even if a teacher's hard block can't be honored.
TEACHER_HARD_CONSTRAINT_PENALTY = 500

# Student structural rules
STUDENT_GAP_PENALTY = 10_000          # internal empty period in a class's day (hard)
STUDENT_LATE_START_PENALTY = 10_000   # class not starting at period 1 (hard)
YOUNG_GRADE_LATE_PENALTY = 500        # grades 1-3 with a lesson in period 7-8 (soft)

# ---------------------------------------------------------------------------
# Soft constraints (teacher_constraints where constraint_type = 'soft')
# ---------------------------------------------------------------------------
# These rows already carry their own weight (1-10) from the DB.
# This multiplier just lets us scale all soft-constraint penalties up/down
# relative to preference penalties below, without editing DB rows.
SOFT_CONSTRAINT_WEIGHT_MULTIPLIER = 1

# ---------------------------------------------------------------------------
# teacher_preferences weights (no DB column -> defined here)
# ---------------------------------------------------------------------------
# Each of these is multiplied by the teacher's own priority_* value (1-5)
# from teacher_preferences, then added as a penalty if the preference is
# not met for that teacher in the generated schedule.
PREFERENCE_WEIGHTS = {
    "early_finish": 2,      # priority_early_finish
    "no_gaps": 2,            # priority_no_gaps
    "free_day": 3,            # priority_free_day
    "consecutive": 2,        # priority_consecutive / preferred_consecutive
}

# Penalty applied per hour a teacher is outside their [min_hours, max_hours]
# range from teacher_preferences.
OUTSIDE_HOURS_RANGE_PENALTY_PER_HOUR = 5

# פיזור מקצועות אותו-יום מנוטרל במנוע כדי להתיישר עם scorer ההפרות (B),
# שהמנהל כבר הסיר ממנו את הסעיף. הפיזור עדיין נשלט דרך "מקסימום ליום" בהגדרות מוסד.
# (לשחזור מספרי הספר: החזר ל-5 + checkout לתגית research-v1.)
SUBJECT_DISTRIBUTION_PENALTY_PER_EXTRA_HOUR = 0

# ---------------------------------------------------------------------------
# CSP solver settings
# ---------------------------------------------------------------------------
CSP_MAX_SOLVE_SECONDS = 60.0

# ---------------------------------------------------------------------------
# Algorithm name constants (must match schedule_runs.algorithm CHECK constraint)
# ---------------------------------------------------------------------------
ALGO_CSP = "CSP"
ALGO_GENETIC = "GENETIC"
ALGO_HILL_CLIMBING = "HILL_CLIMBING"


# ============================================================
# STUDENT-STRUCTURE & SCHOOL RULES (added for real-data quality)
# ============================================================

# Subject categories — drive "holy in the morning" and (later) hard-subject rules.
# Looked up by subject NAME; anything not listed defaults to "easy" (fails safe:
# no morning preference, no penalty). Move to a DB column + admin UI later.
SUBJECT_CATEGORY = {
    # holy (קודש) — prefer morning hours
    "תורה": "holy",
    "נביא": "holy",
    "הלכה": "holy",
    "חינוך מתוך אמונה": "holy",
    "פרשת שבוע": "holy",
    # hard / core — heavier subjects
    "חשבון": "hard",
    "עברית": "hard",
    "אנגלית": "hard",
    "מדעים": "hard",
    # easy — everything else (also the default)
    "אומנות": "easy",
    "חינוך גופני": "easy",
    "כישורי חיים": "easy",
    "מולדת": "easy",
    "זה\"ב": "easy",
}

def category_of(subject_name):
    """Category for a subject name; defaults to 'easy' if unlisted (fail-safe)."""
    return SUBJECT_CATEGORY.get(subject_name, "easy")

# Per-grade dismissal: the last teaching period allowed for each grade on a normal day.
# א,ב,ג finish by period 6; ד,ה,ו by period 8. (Friday is already capped at 4 by timeslots.)
# Move to a DB table + admin UI later. Grade number from scoring_violations.grade_of().
GRADE_DISMISSAL = {1: 6, 2: 6, 3: 6, 4: 8, 5: 8, 6: 8}
DEFAULT_DISMISSAL = 8  # if a grade is somehow unknown

# ---- New penalty weights (fit inside the existing hierarchy) ----
# HARD: an empty class-day is never acceptable.
NO_EMPTY_DAY_PENALTY = HARD_CONSTRAINT_PENALTY

# STRONG-SOFT: lessons past a grade's dismissal — strongly avoided, but breakable
# so tight-hour grades stay feasible. Per hour past dismissal, per class, per day.
DISMISSAL_PENALTY_PER_HOUR = 60

# SOFT: a holy subject placed in the afternoon (later than a threshold). Per lesson.
HOLY_MORNING_PENALTY = 15
HOLY_MORNING_THRESHOLD = 4  # holy lessons after period 4 are penalised

# SOFT: day-to-day imbalance in a class's daily lesson count. Per hour of spread
# (longest day minus shortest day, beyond a tolerance). 
BALANCE_PENALTY_PER_HOUR = 200
BALANCE_TOLERANCE = 2  # a spread of up to 2 hours between days is free

# STRONG-SOFT: a grade exceeding its admin-defined max lessons per day.
# Per lesson over the cap, per class, per day. Set per grade in the DB table
# grade_schedule_limits (managed from the admin UI). Grades with no row have
# no cap (fail-safe). Soft on purpose — keeps tight schedules feasible.
GRADE_MAX_PER_DAY_PENALTY_PER_HOUR = 100


# ---------------------------------------------------------------------------
# Pedagogical constraints (admin UI: pedagogical_constraints). All SOFT.
# Each is multiplied by the row's own `weight` (default 1). Tunable here.
# ---------------------------------------------------------------------------
PED_MAX_PER_DAY_PENALTY = 175       # per excess lesson of the subject, per class/day
PED_NOT_LAST_PENALTY = 75           # per lesson of the subject in the day's last period
PED_MORNING_ONLY_PENALTY = 75       # per lesson of the subject after the morning threshold
PED_MORNING_ONLY_THRESHOLD = 4      # lessons after this period count as "afternoon"
PED_NOT_CONSECUTIVE_PENALTY = 150   # per subject-A lesson adjacent to a subject-B lesson
PED_MIN_GAP_PENALTY = 150           # per subject with 2 non-adjacent lessons in a class/day
