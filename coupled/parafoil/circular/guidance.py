"""Guidance for a round parachute: track a planned drift, do not chase a target.

The whole design follows from one number.  A G-12 descends at about 9 m/s and
can drive itself sideways at about 4 m/s.  A wind of 8 m/s -- ordinary -- is
twice what the canopy can fight.  Flying to a chosen point is therefore
impossible, and any controller that tries will simply saturate and run out of
gas.

What *is* possible is fixing the difference between the forecast wind and the
real one.  So:

1. before the drop, integrate an uncontrolled descent **backwards** from the
   target through the forecast wind.  That gives both the release point and a
   reference path, as a position for every altitude;
2. in flight, compare the actual position at the current altitude with the
   reference;
3. if the error grows past a tolerance that narrows towards the ground, pull
   the riser that drives the system back.

The tolerance is a cone rather than a fixed radius because an error high up
has time to be corrected, while the same error near the ground does not.

The controller is bang-bang with hysteresis: it switches on outside the cone
and off only well inside it.  That is not a simplification for its own sake --
the risers are pneumatic actuators with a fixed gas supply, so a controller
that chatters is not flyable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..atmosphere import WindProfile
from ..vectors import Vec3
from .canopy import RiserCommand, RoundCanopy
from .dynamics import RoundState


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """Where the system should be at each altitude, and where to release it.

    Attributes:
        altitude: Descending sample altitudes, m.
        east, south: Planned horizontal position at each altitude, m.
        release: The computed release point -- the top of the reference path.
        target: The aim point on the ground.
    """

    altitude: np.ndarray
    east: np.ndarray
    south: np.ndarray
    release: Vec3
    target: Vec3

    def at(self, altitude: float) -> Vec3:
        """Planned position at a given altitude, interpolated.

        Outside the planned band the nearest end is held, which keeps the
        controller sane if the canopy is still above the release altitude
        (it never is) or has somehow overshot the ground.
        """
        # `altitude` descends through the table, so the arrays are reversed
        # for np.interp, which wants an increasing x.
        east = float(np.interp(altitude, self.altitude[::-1], self.east[::-1]))
        south = float(np.interp(altitude, self.altitude[::-1], self.south[::-1]))
        return Vec3(east, altitude, south)


@dataclass(frozen=True, slots=True)
class BallisticPlan:
    """Builds a :class:`ReferenceTrajectory` by falling backwards from the target.

    The descent rate comes from the canopy, the drift from the forecast wind.
    No control is assumed: the reference path is what an *unguided* drop
    would do if the forecast were perfect, which is exactly the trajectory the
    guidance then tries to hold.
    """

    canopy: RoundCanopy
    forecast: WindProfile
    air_density: float = 1.23
    gravity: float = 9.81
    #: Altitude step of the plan, m.  100 m matches the resolution of the
    #: wind tables these forecasts come in.
    step: float = 25.0

    def build(self, target: Vec3, release_altitude: float) -> ReferenceTrajectory:
        if release_altitude <= target.y:
            raise ValueError("release altitude must be above the target")

        descent = self.canopy.descent_rate(self.air_density, self.gravity)
        levels = np.arange(target.y, release_altitude + self.step, self.step)

        east = np.empty_like(levels)
        south = np.empty_like(levels)
        east[0], south[0] = target.x, target.z

        for i in range(1, len(levels)):
            # Climbing one step, the canopy would have taken this long to
            # fall it, and the wind would have carried it this far.
            dt = (levels[i] - levels[i - 1]) / descent
            mid = 0.5 * (levels[i] + levels[i - 1])
            wind = self.forecast.at(float(mid))
            east[i] = east[i - 1] - wind.x * dt
            south[i] = south[i - 1] - wind.z * dt

        return ReferenceTrajectory(
            altitude=levels[::-1].copy(),
            east=east[::-1].copy(),
            south=south[::-1].copy(),
            release=Vec3(float(east[-1]), float(levels[-1]), float(south[-1])),
            target=target,
        )


class NoControl:
    """The unguided reference case: never pulls anything."""

    def command(self, state: RoundState, dt: float) -> RiserCommand:
        return RiserCommand.none()


@dataclass(slots=True)
class ConeTracker:
    """Bang-bang tracking of a reference trajectory inside a narrowing cone.

    Attributes:
        reference: The path to hold.
        cone_top / cone_bottom: Tolerance radius at the release altitude and
            at the ground, m.  183 and 18.3 are the published AGAS figures.
        release_radius: Once the error falls below this, control is dropped
            and the system is left to drift.  Hysteresis against the cone --
            without it the actuators would chatter on the boundary.
        activation_threshold: A riser fires only if its axis carries at least
            this share of the unit error vector.  Below ``1/sqrt(2)`` two
            adjacent risers can fire together, giving eight drive directions
            instead of four and half again the glide ratio.

            0.3, the published AGAS value.  It was worth checking: over
            twelve forecast errors it gives a 9.6 m mean miss against 13.5 at
            0.5, for two more actuator cycles.  With so little drive to spare
            the pair is worth firing whenever it points anywhere useful.

            Above ``1/sqrt(2)`` only one riser can ever fire; the drive can
            then be 45 degrees off the direction wanted, and the controller
            hunts -- 24 m mean miss and up to 30 cycles.
        settle_time: Seconds after release before any control is allowed --
            the canopy has to inflate and the risers to untwist first.
    """

    reference: ReferenceTrajectory
    cone_top: float = 183.0
    cone_bottom: float = 18.3
    release_radius: float = 9.1
    activation_threshold: float = 0.3
    settle_time: float = 25.0

    _engaged: bool = field(default=False, init=False)

    def tolerance_at(self, altitude: float) -> float:
        """Cone radius at this altitude, linear between the two ends."""
        top = float(self.reference.altitude[0])
        bottom = float(self.reference.altitude[-1])
        if top <= bottom:
            return self.cone_bottom
        fraction = (altitude - bottom) / (top - bottom)
        fraction = min(1.0, max(0.0, fraction))
        return self.cone_bottom + fraction * (self.cone_top - self.cone_bottom)

    def command(self, state: RoundState, dt: float) -> RiserCommand:
        if state.landed or state.time < self.settle_time:
            self._engaged = False
            return RiserCommand.none()

        planned = self.reference.at(state.altitude)
        error_east = state.position.x - planned.x
        error_south = state.position.z - planned.z
        distance = math.hypot(error_east, error_south)

        # Hysteresis: engage outside the cone, disengage only well inside.
        if distance > self.tolerance_at(state.altitude):
            self._engaged = True
        elif distance < self.release_radius:
            self._engaged = False

        if not self._engaged or distance < 1e-6:
            return RiserCommand.none()

        # Into canopy axes.  The canopy is rotating, so which riser points
        # which way depends on the current heading -- this is why the real
        # system carries a compass.
        cos_h, sin_h = math.cos(state.heading), math.sin(state.heading)
        along = (error_east * cos_h + error_south * sin_h) / distance
        across = (-error_east * sin_h + error_south * cos_h) / distance

        # Releasing a riser drives *away* from it, so to move back towards the
        # reference we release the riser lying on the same side as the error.
        threshold = self.activation_threshold
        return RiserCommand(
            a=along > threshold,
            b=across > threshold,
            c=along < -threshold,
            d=across < -threshold,
        )
