"""Flight of a round parachute: point mass, drag, weight, and a small drive.

Deliberately a three-degree-of-freedom point-mass model, which is what the
AGAS studies used and for the same reason: there is not enough aerodynamic
data on these canopies to justify anything richer.  The published model notes
explicitly that a round parachute's forces vary strongly with angle of attack
and that a point mass cannot capture it -- so this does not pretend to.

What it does capture is the part that decides where the load lands: terminal
descent, wind drift, and the small horizontal drive that riser control buys.

The integrator, the wind profile and the vector type are shared with the
parafoil model.  Only the force law and the control are new.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..atmosphere import WindProfile
from ..dynamics import VerletIntegrator
from ..simulation import LandingReport
from ..vectors import Vec3
from .canopy import RiserCommand, RoundCanopy


@dataclass(frozen=True, slots=True)
class RoundState:
    """Immutable snapshot of a round-parachute flight."""

    time: float
    position: Vec3
    velocity: Vec3
    wind: Vec3
    heading: float          # canopy rotation, rad
    command: RiserCommand
    landed: bool

    @property
    def altitude(self) -> float:
        return self.position.y

    @property
    def air_velocity(self) -> Vec3:
        return self.velocity - self.wind

    @property
    def descent_rate(self) -> float:
        return -self.velocity.y

    @property
    def ground_speed(self) -> float:
        return self.velocity.horizontal().norm()


class RoundParachute:
    """The vehicle.  Construct, read :attr:`state`, call :meth:`step`.

    Args:
        canopy: Parameters; defaults to the G-12.
        wind: Wind profile.
        position, velocity: Initial conditions, world axes.
        heading: Initial canopy rotation, rad.
        air_density, gravity: Environment.
    """

    __slots__ = ("canopy", "wind", "air_density", "gravity", "_integrator",
                 "_heading", "_time", "_landed", "_command", "_effective_mass")

    def __init__(
        self,
        canopy: RoundCanopy,
        wind: WindProfile,
        position: Vec3,
        velocity: Vec3,
        *,
        heading: float = 0.0,
        air_density: float = 1.23,
        gravity: float = 9.81,
    ) -> None:
        self.canopy = canopy
        self.wind = wind
        self.air_density = air_density
        self.gravity = gravity
        self._integrator = VerletIntegrator(position, velocity)
        self._heading = heading
        self._time = 0.0
        self._landed = position.y <= 0.0
        self._command = RiserCommand.none()
        # Apparent mass resists acceleration along with the real mass.  It
        # only matters while the system is still speeding up after release.
        self._effective_mass = canopy.total_mass + canopy.added_mass(air_density)

    @property
    def state(self) -> RoundState:
        pos = self._integrator.position
        return RoundState(
            time=self._time,
            position=pos,
            velocity=self._integrator.velocity,
            wind=self.wind.at(pos.y),
            heading=self._heading,
            command=self._command,
            landed=self._landed,
        )

    # ---- forces ---------------------------------------------------------

    def aerodynamic_force(self, command: RiserCommand) -> Vec3:
        """Drag plus the drive from any released risers, world axes, newtons.

        Drag opposes the airflow.  Releasing risers deforms the canopy, and
        the deformed canopy is a *different aerodynamic body*: it flies at a
        measured glide ratio, taken from the AGAS flight-test table by way of
        :meth:`~parafoil.circular.canopy.RoundCanopy.drive_ratio_at`.

        So the control does not add a fixed force -- it selects between
        measured states.  One riser and two adjacent risers are genuinely
        different configurations, with glide ratios of 0.37 and 0.52, and the
        difference is not something this model assumes but something the
        flight test measured.
        """
        pos = self._integrator.position
        air_velocity = self._integrator.velocity - self.wind.at(pos.y)
        speed = air_velocity.norm()
        if speed < 1e-9:
            return Vec3(0.0, 0.0, 0.0)

        magnitude = 0.5 * self.air_density * self.canopy.drag_area * speed * speed
        force = air_velocity * (-magnitude / speed)

        east, south = command.drive_vector(self._heading)
        strength = math.hypot(east, south)
        if strength > 1e-9:
            drive = magnitude * self.canopy.drive_ratio_at(strength) / strength
            force = force + Vec3(east * drive, 0.0, south * drive)
        return force

    # ---- integration ----------------------------------------------------

    def step(self, dt: float, command: RiserCommand) -> RoundState:
        """Advance by ``dt`` and return the new state."""
        if self._landed or dt <= 0.0:
            return self.state

        self._command = command
        # Weight is a force on the *real* mass; the apparent mass adds to the
        # inertia that resists the resultant, not to the weight.  Dividing
        # gravity by the effective mass too would make the canopy fall as if
        # it weighed 3.6 tonnes, and the descent rate comes out at 16.7 m/s
        # instead of 9.
        weight = Vec3(0.0, -self.canopy.total_mass * self.gravity, 0.0)
        force = self.aerodynamic_force(command)
        acceleration = (weight + force) / self._effective_mass
        self._integrator.step(acceleration, dt)

        # The canopy turns steadily; nothing the controller does changes it.
        self._heading += self.canopy.rotation_rate * dt
        self._time += dt

        if self._integrator.position.y <= 0.0:
            self._landed = True
            self._integrator.position = self._integrator.position.with_y(0.0)

        return self.state


@dataclass(frozen=True, slots=True)
class RoundTrajectory:
    """Recorded flight, as arrays."""

    time: np.ndarray
    position: np.ndarray      # (n, 3)
    velocity: np.ndarray      # (n, 3)
    wind: np.ndarray          # (n, 3)
    heading: np.ndarray       # (n,)
    command: np.ndarray       # (n, 4) of 0/1
    report: LandingReport

    def __len__(self) -> int:
        return len(self.time)

    @property
    def altitude(self) -> np.ndarray:
        return self.position[:, 1]

    @property
    def descent_rate(self) -> np.ndarray:
        return -self.velocity[:, 1]

    @property
    def ground_speed(self) -> np.ndarray:
        return np.hypot(self.velocity[:, 0], self.velocity[:, 2])

    @property
    def active_risers(self) -> np.ndarray:
        return self.command.sum(axis=1)

    def control_duty(self) -> float:
        """Fraction of the flight with at least one riser released.

        Worth watching: the real system carries enough gas for a limited
        number of activations, so a controller that chatters is not flyable.
        """
        return float((self.active_risers > 0).mean())


def simulate_round(
    canopy: RoundCanopy,
    wind: WindProfile,
    position: Vec3,
    velocity: Vec3,
    pilot,
    *,
    target: Vec3 = Vec3(0.0, 0.0, 0.0),
    dt: float = 0.05,
    max_time: float = 600.0,
    heading: float = 0.0,
    sample_every: int = 1,
    air_density: float = 1.23,
) -> RoundTrajectory:
    """Fly until touchdown or ``max_time``.

    Args:
        pilot: Anything with ``command(state, dt) -> RiserCommand``.
        dt: 0.05 s is ample here -- a round parachute has no fast dynamics in
            this model, and the controller itself runs at a few hertz.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    vehicle = RoundParachute(
        canopy, wind, position, velocity, heading=heading, air_density=air_density
    )

    times: list[float] = []
    positions: list[Vec3] = []
    velocities: list[Vec3] = []
    winds: list[Vec3] = []
    headings: list[float] = []
    commands: list[tuple[int, int, int, int]] = []

    def record(s: RoundState) -> None:
        times.append(s.time)
        positions.append(s.position)
        velocities.append(s.velocity)
        winds.append(s.wind)
        headings.append(s.heading)
        commands.append(tuple(int(x) for x in s.command))

    state = vehicle.state
    record(state)

    steps = 0
    max_steps = int(max_time / dt)
    while not state.landed and steps < max_steps:
        state = vehicle.step(dt, pilot.command(state, dt))
        steps += 1
        if state.landed or steps % sample_every == 0:
            record(state)

    report = LandingReport(
        touchdown=state.position,
        miss_distance=(state.position - target).horizontal().norm(),
        ground_speed=state.ground_speed,
        descent_rate=state.descent_rate,
        flight_time=state.time,
        landed=state.landed,
    )
    return RoundTrajectory(
        time=np.asarray(times, dtype=np.float64),
        position=np.asarray(positions, dtype=np.float64),
        velocity=np.asarray(velocities, dtype=np.float64),
        wind=np.asarray(winds, dtype=np.float64),
        heading=np.asarray(headings, dtype=np.float64),
        command=np.asarray(commands, dtype=np.int8),
        report=report,
    )
