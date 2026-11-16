"""Automatic guidance to a landing point.

This reimplements the algorithm described in the original project's
"Алгоритм автоматического управления" note:

1. Build a **glide slope** -- a ray leaving the intended landing point,
   sloping up and *downwind*, whose gradient is the canopy's glide ratio
   corrected for wind.  Flown backwards, it is the into-wind final approach.
2. Pick three points on it: the highest point the canopy can still reach, the
   landing point itself, and the midpoint between them.
3. Aim at each in turn, steering with differential brake and controlling the
   descent angle with symmetric brake.
4. Below 80 m, stop chasing points and simply line up into wind.
5. At 4 m, pull both toggles fully -- the flare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .atmosphere import WindProfile
from .canopy import CanopyParams, ControlInput
from .dynamics import FlightState
from .vectors import UP, Vec3, signed_angle


class Phase(Enum):
    """Which leg of the approach the autopilot is flying."""

    HIGH = "to high point"
    MIDDLE = "to middle point"
    FINAL = "to landing point"
    INTO_WIND = "aligning into wind"
    FLARE = "flare"
    LANDED = "landed"


@dataclass(frozen=True, slots=True)
class GlideSlope:
    """The three aim points of the approach, from the landing point upwards."""

    landing: Vec3   # L0 -- the target
    middle: Vec3    # L1
    high: Vec3      # L2
    direction: Vec3  # unit vector from landing point towards `high`

    @classmethod
    def build(
        cls,
        target: Vec3,
        entry_altitude: float,
        wind: WindProfile,
        airspeed: float,
        glide_ratio: float,
    ) -> "GlideSlope":
        """Construct the slope for a given entry altitude.

        The ground track of a canopy flying into a headwind is slower than its
        airspeed, so the slope is steeper than the still-air glide ratio.  The
        components used here are the steady-state ones::

            vertical   = airspeed / sqrt(1 + k^2)
            horizontal = k * vertical

        Traced backwards from the landing point, the ray climbs at
        ``vertical`` while moving downwind at ``horizontal - wind_speed``.
        """
        vertical = airspeed / math.sqrt(1.0 + glide_ratio * glide_ratio)
        horizontal = glide_ratio * vertical

        ground_wind = wind.at(target.y)
        wind_speed = ground_wind.norm()
        if wind_speed >= horizontal:
            raise ValueError(
                f"wind of {wind_speed:.1f} m/s exceeds the canopy's horizontal "
                f"airspeed of {horizontal:.1f} m/s, so it cannot penetrate "
                f"upwind and the landing point cannot be reached"
            )

        if wind_speed > 1e-6:
            downwind = ground_wind / wind_speed
            horizontal_leg = downwind * (horizontal - wind_speed)
        else:
            # Calm air: the approach direction is arbitrary, so pick east.
            horizontal_leg = Vec3(horizontal, 0.0, 0.0)

        direction = (horizontal_leg + UP * vertical).unit()

        # Length along the ray that reaches the entry altitude.
        length = (entry_altitude - target.y) / direction.y
        return cls(
            landing=target,
            middle=target + direction * (length / 2.0),
            high=target + direction * length,
            direction=direction,
        )


@dataclass(slots=True)
class HomingAutopilot:
    """Flies the canopy to :attr:`target` and flares.

    The two control channels are independent and simply added:

    * **differential** brake steers, proportionally to the heading error and
      saturating at a 60 degree error;
    * **symmetric** brake trims the glide ratio to whatever the remaining
      distance and height call for, then both toggles get that amount added.

    Attributes:
        capture_radius_high / capture_radius_middle: switch to the next aim
            point once within this distance of the current one.
        align_altitude: below this, give up on aim points and turn into wind.
        flare_altitude: below this, pull both toggles fully down.
    """

    target: Vec3
    wind: WindProfile
    params: CanopyParams
    slope: GlideSlope

    air_density: float = 1.23
    capture_radius_high: float = 150.0
    capture_radius_middle: float = 50.0
    align_altitude: float = 80.0
    flare_altitude: float = 4.0

    #: Heading error at which the steering command saturates.
    saturation_angle: float = math.pi / 3.0

    phase: Phase = field(default=Phase.HIGH, init=False)

    @classmethod
    def for_flight(
        cls,
        target: Vec3,
        state: FlightState,
        wind: WindProfile,
        params: CanopyParams,
        *,
        planned_airspeed: float = 21.0,
        planned_glide_ratio: float = 3.7,
        **kwargs,
    ) -> "HomingAutopilot":
        """Build an autopilot and its glide slope from the current state.

        ``planned_airspeed`` and ``planned_glide_ratio`` describe the canopy's
        nominal cruise performance; they only shape the planned approach, the
        actual flight comes from the physics model.
        """
        slope = GlideSlope.build(
            target=target,
            entry_altitude=state.altitude,
            wind=wind,
            airspeed=planned_airspeed,
            glide_ratio=planned_glide_ratio,
        )
        return cls(target=target, wind=wind, params=params, slope=slope, **kwargs)

    # ---- the Pilot protocol ---------------------------------------------

    def control(self, state: FlightState, dt: float) -> ControlInput:
        if state.landed:
            self.phase = Phase.LANDED
            return ControlInput()

        if state.altitude <= self.flare_altitude:
            self.phase = Phase.FLARE
            return ControlInput(left=1.0, right=1.0)

        if state.altitude <= self.align_altitude:
            self.phase = Phase.INTO_WIND
            return self._steer_towards(state, self._upwind(state))

        aim = self._select_aim_point(state)
        steering = self._steer_towards(state, (aim - state.position).horizontal())
        return self._add_glide_brake(steering, state, aim)

    # ---- guidance --------------------------------------------------------

    def _select_aim_point(self, state: FlightState) -> Vec3:
        """Advance through high -> middle -> landing as each is reached."""
        if self.phase is Phase.HIGH:
            reached = (
                state.altitude < self.slope.high.y
                or (self.slope.high - state.position).norm() < self.capture_radius_high
            )
            if reached:
                self.phase = Phase.MIDDLE

        if self.phase is Phase.MIDDLE:
            reached = (
                state.altitude < self.slope.middle.y
                or (self.slope.middle - state.position).norm() < self.capture_radius_middle
            )
            if reached:
                self.phase = Phase.FINAL

        return {
            Phase.HIGH: self.slope.high,
            Phase.MIDDLE: self.slope.middle,
        }.get(self.phase, self.slope.landing)

    def _upwind(self, state: FlightState) -> Vec3:
        """Horizontal direction to fly to face the wind."""
        wind = self.wind.at(state.altitude).horizontal()
        return -wind if wind.norm() > 1e-6 else state.air_velocity.horizontal()

    def _steer_towards(self, state: FlightState, direction: Vec3) -> ControlInput:
        """Proportional differential brake, saturating at ``saturation_angle``.

        The gain is chosen so the command reaches full deflection exactly at
        the saturation angle, which keeps the law continuous.
        """
        heading = state.air_velocity.horizontal()
        error = signed_angle(heading, direction)

        magnitude = min(1.0, abs(error) / self.saturation_angle)
        if error > 0.0:  # target to the right
            return ControlInput(left=0.0, right=magnitude)
        return ControlInput(left=magnitude, right=0.0)

    def _add_glide_brake(
        self, steering: ControlInput, state: FlightState, aim: Vec3
    ) -> ControlInput:
        """Add symmetric brake so the glide ratio matches what is required.

        The required glide ratio is ``horizontal distance / height to lose``.
        If that is at or above the flattest glide the canopy can manage there
        is nothing to do (hands up).  If it is at or below the steepest, pull
        all the way.  In between, interpolate linearly.

        The comparison is made in the **ground** frame.  The original compared
        a ground-frame requirement against the canopy's air-frame limits, which
        is optimistic whenever there is a headwind -- the canopy would believe
        it could stretch the glide to 4:1 while the air mass was pushing it
        back to nearer 2.7:1, and it consistently landed short.  See
        :meth:`_reachable_glide_ratios`.

        The added amount is clipped so neither toggle exceeds 1, which means
        steering keeps priority over glide control.
        """
        height_to_lose = state.position.y - aim.y
        to_aim = (aim - state.position).horizontal()
        if height_to_lose <= 1e-6:
            required = math.inf  # already at or below the aim point: stretch it
        else:
            required = to_aim.norm() / height_to_lose

        flattest, steepest = self._reachable_glide_ratios(state, to_aim)
        if required >= flattest:
            extra = 0.0
        elif required <= steepest:
            extra = 1.0
        else:
            extra = (flattest - required) / (flattest - steepest)

        extra = min(extra, 1.0 - steering.left, 1.0 - steering.right)
        extra = max(0.0, extra)
        return ControlInput(left=steering.left + extra, right=steering.right + extra)

    def _reachable_glide_ratios(
        self, state: FlightState, towards: Vec3
    ) -> tuple[float, float]:
        """``(flattest, steepest)`` ground glide ratios available right now.

        Steady flight at symmetric brake ``d`` gives horizontal and vertical
        airspeeds ``(Vh, Vv)``.  Flying towards ``towards`` against a headwind
        component ``w``, the canopy actually covers ``Vh - w`` over the ground
        per ``Vv`` of descent, so the achievable ground glide ratio is
        ``(Vh - w) / Vv``.
        """
        wind = self.wind.at(state.altitude).horizontal()
        distance = towards.norm()
        headwind = -wind.dot(towards) / distance if distance > 1e-6 else 0.0

        def ground_ratio(deflection: float) -> float:
            horizontal, vertical = self.params.trim_speeds(
                ControlInput(deflection, deflection), self.air_density
            )
            return (horizontal - headwind) / vertical

        return ground_ratio(0.0), ground_ratio(1.0)
