"""Immutable project constants and enumerations."""

from enum import IntEnum


class WorkoutAction(IntEnum):
    """Discrete workout-type actions for the RL agent."""

    REST = 0
    UPPER_PUSH = 1
    UPPER_PULL = 2
    LOWER_BODY = 3
    FULL_BODY = 4
    CORE_CARDIO = 5


ACTION_LABELS: dict[int, str] = {
    WorkoutAction.REST: "Rest / Recovery",
    WorkoutAction.UPPER_PUSH: "Upper Push",
    WorkoutAction.UPPER_PULL: "Upper Pull",
    WorkoutAction.LOWER_BODY: "Lower Body",
    WorkoutAction.FULL_BODY: "Full Body",
    WorkoutAction.CORE_CARDIO: "Core / Cardio",
}

# State vector feature indices
STATE_ROLLING_LOAD = 0
STATE_MUSCLE_BALANCE = 1
STATE_DURATION_AVG = 2
STATE_DAY_SIN = 3  # sin(2π·t/28) — cyclic day encoding
STATE_DAY_COS = 4  # cos(2π·t/28) — cyclic day encoding

N_ACTIONS = len(WorkoutAction)
STATE_DIM = 5  # rolling_load, muscle_balance, duration_avg, day_sin, day_cos

# Cycle length for sinusoidal day encoding
CYCLE_LENGTH = 28
