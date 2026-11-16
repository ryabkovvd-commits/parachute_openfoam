"""Running a flight and collecting the result.

This is the only module that knows about NumPy: the integration loop works on
plain floats (fast for 3-vectors), and the recorded trajectory is converted to
arrays once at the end, where NumPy pays off for analysis and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Protocol

import numpy as np

from .canopy import ControlInput
from .dynamics import FlightState, Parafoil
from .pilots import Pilot
from .scenario import Scenario
from .vectors import Vec3

#: Which physics to fly with.  ``"point-mass"`` is the reconstruction of the
#: original program; ``"rigid"`` is the 6-DOF model in :mod:`parafoil.rigid`.
Model = Literal["point-mass", "rigid"]


class Vehicle(Protocol):
    """Anything that can be flown: a state to read and a step to take.

    Both :class:`~parafoil.dynamics.Parafoil` and
    :class:`~parafoil.rigid.body.RigidParafoil` satisfy this, which is what
    lets one simulation loop drive either.
    """

    @property
    def state(self) -> FlightState: ...

    def step(self, dt: float, control: ControlInput) -> FlightState: ...


def build_vehicle(scenario: Scenario, model: Model = "point-mass", **kwargs) -> Vehicle:
    """Construct the vehicle a scenario describes.

    Args:
        model: ``"point-mass"`` for the original 3-DOF physics, ``"rigid"``
            for the 6-DOF one.
        **kwargs: Passed through to the model, e.g. ``damping_ratio`` or
            ``include_trapped_air`` for the rigid model.
    """
    common = dict(
        params=scenario.canopy,
        wind=scenario.wind,
        position=scenario.position,
        velocity=scenario.velocity,
        air_density=scenario.air_density,
        gravity=scenario.gravity,
    )
    if model == "point-mass":
        if kwargs:
            raise TypeError(f"the point-mass model takes no options, got {kwargs}")
        return Parafoil(**common)
    if model == "rigid":
        from .rigid.body import RigidParafoil

        return RigidParafoil(**common, **kwargs)
    raise ValueError(f"unknown model {model!r}; expected 'point-mass' or 'rigid'")


@dataclass(frozen=True, slots=True)
class LandingReport:
    """What happened at the end of the flight."""

    touchdown: Vec3
    miss_distance: float      # horizontal distance from the target, m
    ground_speed: float       # m/s at touchdown
    descent_rate: float       # m/s, positive down
    flight_time: float        # s
    landed: bool              # False if the run hit its time limit first

    def summary(self) -> str:
        if not self.landed:
            return f"did not land within {self.flight_time:.0f} s"
        return (
            f"landed after {self.flight_time:.1f} s, "
            f"{self.miss_distance:.1f} m from the target, "
            f"ground speed {self.ground_speed:.1f} m/s, "
            f"descent {self.descent_rate:.1f} m/s"
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Recorded flight history.

    All arrays share the same first dimension.  Positions and velocities are
    ``(n, 3)`` in world axes (east, up, south); ``control`` is ``(n, 2)`` as
    ``(left, right)``.
    """

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    wind: np.ndarray
    control: np.ndarray
    report: LandingReport
    model: str = "point-mass"
    #: ``(n, 3)`` of roll, pitch and yaw in radians, or ``None`` for the
    #: point-mass model, which has no attitude to report.
    attitude: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.time)

    @property
    def altitude(self) -> np.ndarray:
        return self.position[:, 1]

    @property
    def ground_speed(self) -> np.ndarray:
        return np.hypot(self.velocity[:, 0], self.velocity[:, 2])

    @property
    def descent_rate(self) -> np.ndarray:
        """Positive downwards."""
        return -self.velocity[:, 1]

    @property
    def airspeed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity - self.wind, axis=1)

    def distance_to(self, target: Vec3) -> np.ndarray:
        """Horizontal distance to ``target`` at each sample."""
        return np.hypot(
            self.position[:, 0] - target.x, self.position[:, 2] - target.z
        )

    def glide_ratio(self) -> np.ndarray:
        """Instantaneous ground glide ratio, guarding against division by zero."""
        descent = self.descent_rate
        return np.divide(
            self.ground_speed, descent, out=np.zeros_like(descent), where=descent > 1e-6
        )


def simulate(
    scenario: Scenario,
    pilot: Pilot,
    *,
    dt: float = 0.01,
    max_time: float = 600.0,
    sample_every: int = 1,
    model: Model = "point-mass",
    **model_options,
) -> Trajectory:
    """Fly ``scenario`` under ``pilot`` until touchdown or ``max_time``.

    Args:
        dt: Integration step in seconds.  0.01 matches the original program;
            larger steps are faster but less accurate near the flare.
        max_time: Safety limit so a badly behaved pilot cannot loop forever.
        sample_every: Record one sample per this many steps.  Use e.g. 10 to
            keep plots light without changing the physics.
        model: Which physics to use; see :data:`Model`.  The default keeps the
            original reconstruction, so existing results do not move.
        **model_options: Passed to the model, e.g. ``damping_ratio``.

    Returns:
        A :class:`Trajectory` including a :class:`LandingReport`.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")

    vehicle = build_vehicle(scenario, model, **model_options)

    times: list[float] = []
    positions: list[Vec3] = []
    velocities: list[Vec3] = []
    winds: list[Vec3] = []
    controls: list[tuple[float, float]] = []
    attitudes: list[tuple[float, float, float]] = []

    def record(s: FlightState) -> None:
        times.append(s.time)
        positions.append(s.position)
        velocities.append(s.velocity)
        winds.append(s.wind)
        controls.append((s.control.left, s.control.right))
        if s.attitude is not None:
            attitudes.append(s.attitude.to_euler())

    state = vehicle.state
    record(state)

    step_index = 0
    max_steps = int(max_time / dt)
    while not state.landed and step_index < max_steps:
        control = pilot.control(state, dt)
        state = vehicle.step(dt, control)
        step_index += 1
        if state.landed or step_index % sample_every == 0:
            record(state)

    report = LandingReport(
        touchdown=state.position,
        miss_distance=(state.position - scenario.target).horizontal().norm(),
        ground_speed=state.ground_speed,
        descent_rate=state.descent_rate,
        flight_time=state.time,
        landed=state.landed,
    )

    return Trajectory(
        time=np.asarray(times, dtype=np.float64),
        position=np.asarray(positions, dtype=np.float64),
        velocity=np.asarray(velocities, dtype=np.float64),
        wind=np.asarray(winds, dtype=np.float64),
        control=np.asarray(controls, dtype=np.float64),
        report=report,
        model=model,
        attitude=(
            np.asarray(attitudes, dtype=np.float64) if attitudes else None
        ),
    )


def iter_flight(
    scenario: Scenario,
    pilot: Pilot,
    *,
    dt: float = 0.01,
    max_time: float = 600.0,
    model: Model = "point-mass",
    **model_options,
) -> Iterator[FlightState]:
    """Same simulation, yielded step by step.

    Useful for a live display or an interactive trainer, where you want to
    render each frame rather than wait for the whole flight.
    """
    vehicle = build_vehicle(scenario, model, **model_options)
    state = vehicle.state
    yield state
    steps = 0
    max_steps = int(max_time / dt)
    while not state.landed and steps < max_steps:
        state = vehicle.step(dt, pilot.control(state, dt))
        steps += 1
        yield state

