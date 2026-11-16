"""Who moves the toggles.

A *pilot* is anything that looks at the current :class:`~parafoil.dynamics.FlightState`
and returns a :class:`~parafoil.canopy.ControlInput`.  Keeping this behind a
tiny protocol is what lets the same physics core serve a scripted test, an
autopilot, and (eventually) a human at a keyboard.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .canopy import ControlInput
from .dynamics import FlightState


class Pilot(Protocol):
    """Decides the toggle positions for the next step."""

    def control(self, state: FlightState, dt: float) -> ControlInput: ...


@dataclass(slots=True)
class FixedPilot:
    """Holds one control input for the whole flight."""

    setting: ControlInput = ControlInput()

    def control(self, state: FlightState, dt: float) -> ControlInput:
        return self.setting


@dataclass(slots=True)
class ScriptedPilot:
    """Replays a timed sequence of inputs.

    Args:
        schedule: ``(time_seconds, control)`` pairs.  The most recent entry at
            or before the current time is used, so the list acts as a series of
            step changes.  It is sorted on construction.

    Example::

        ScriptedPilot([(0, ControlInput()),
                       (20, ControlInput(right=0.8)),   # start a right turn
                       (35, ControlInput())])           # roll out
    """

    schedule: Sequence[tuple[float, ControlInput]]

    def __post_init__(self) -> None:
        self.schedule = sorted(self.schedule, key=lambda item: item[0])
        if not self.schedule:
            raise ValueError("schedule must contain at least one entry")

    def control(self, state: FlightState, dt: float) -> ControlInput:
        times = [t for t, _ in self.schedule]
        i = bisect_right(times, state.time)
        return self.schedule[max(0, i - 1)][1]


@dataclass(slots=True)
class RateLimitedPilot:
    """Wraps another pilot so the toggles move at a finite speed.

    A real pilot cannot teleport the brakes from 0 to 1: the lines take time to
    pull.  The original program moved each toggle by 0.01 per 60 Hz frame,
    i.e. about 0.6 per second, which is the default here.

    Wrap any pilot to make its commands physically achievable::

        pilot = RateLimitedPilot(HomingAutopilot(...))
    """

    inner: Pilot
    rate: float = 0.6  # units of toggle travel per second
    _current: ControlInput = field(default=ControlInput(), init=False)

    def control(self, state: FlightState, dt: float) -> ControlInput:
        target = self.inner.control(state, dt)
        limit = self.rate * dt
        self._current = ControlInput(
            left=_approach(self._current.left, target.left, limit),
            right=_approach(self._current.right, target.right, limit),
        )
        return self._current


def _approach(current: float, target: float, limit: float) -> float:
    """Move ``current`` towards ``target`` by at most ``limit``."""
    delta = target - current
    if delta > limit:
        return current + limit
    if delta < -limit:
        return current - limit
    return target
