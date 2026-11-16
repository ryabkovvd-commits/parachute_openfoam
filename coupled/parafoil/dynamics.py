"""Flight dynamics: body frame, time integration, and the equation of motion.

The state advanced by :class:`Parafoil` is just a position and a velocity --
the model has no rotational degrees of freedom.  The canopy's attitude is
*derived*: it always points along the airflow and banks so that its "up" axis
follows the resultant aerodynamic force.  That is what makes a differential
brake input produce a coordinated turn without integrating any moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .atmosphere import WindProfile
from .canopy import CanopyParams, ControlInput
from .vectors import UP, Vec3

if TYPE_CHECKING:  # pragma: no cover - avoids a cycle: rigid imports dynamics
    from .rigid.quaternion import Quat

#: How strongly the reference "up" vector is pulled back towards world
#: vertical each step.  Without it the body frame slowly drifts; with it the
#: canopy levels out when the controls are neutral.
_UP_REFERENCE_BLEND = 0.01


@dataclass(frozen=True, slots=True)
class BodyFrame:
    """Right-handed orthonormal frame attached to the canopy."""

    forward: Vec3  # along the airspeed vector
    up: Vec3       # normal to it, towards the canopy
    side: Vec3     # forward x up

    @classmethod
    def from_velocity(cls, velocity: Vec3) -> "BodyFrame":
        """Initial frame: nose along ``velocity``, wings level."""
        forward = velocity.unit()
        side = forward.cross(UP).unit()
        return cls(forward=forward, up=side.cross(forward).unit(), side=side)

    @property
    def bank_angle(self) -> float:
        """Roll angle in radians: how far ``up`` is tipped from world vertical."""
        import math

        return math.acos(max(-1.0, min(1.0, self.up.dot(UP))))


@dataclass(frozen=True, slots=True)
class FlightState:
    """Immutable snapshot of the vehicle at one instant.

    The last two fields are populated only by the 6-DOF model in
    :mod:`parafoil.rigid`.  For the point-mass model they stay ``None``,
    because it has no rotational state to report: its :attr:`frame` is derived
    from the force each step rather than integrated.  Anything that reads them
    should treat ``None`` as "this model does not know", not as "zero".
    """

    time: float
    position: Vec3
    velocity: Vec3          # ground velocity
    wind: Vec3
    frame: BodyFrame
    control: ControlInput
    landed: bool
    attitude: "Quat | None" = None
    angular_velocity: Vec3 | None = None    # body axes, rad/s

    @property
    def altitude(self) -> float:
        return self.position.y

    @property
    def air_velocity(self) -> Vec3:
        """Velocity relative to the air mass -- what the canopy actually feels."""
        return self.velocity - self.wind

    @property
    def airspeed(self) -> float:
        return self.air_velocity.norm()

    @property
    def ground_speed(self) -> float:
        return self.velocity.horizontal().norm()

    @property
    def descent_rate(self) -> float:
        """Positive downwards, m/s."""
        return -self.velocity.y


class Integrator(Protocol):
    """Anything that can advance a position given an acceleration.

    Implement this to swap in a different scheme (Runge-Kutta, symplectic
    Euler, ...) without touching the force model.
    """

    position: Vec3
    velocity: Vec3

    def step(self, acceleration: Vec3, dt: float) -> None: ...


class VerletIntegrator:
    """Variable-step three-point (Stoermer-Verlet) integrator.

    Keeps the previous position and advances with

        x[n+1] = x[n] + (x[n] - x[n-1]) * r + a * dt * (dt + dt_prev) / 2

    where ``r = dt / dt_prev``.  At a constant step this is plain Verlet,
    ``x[n+1] = 2*x[n] - x[n-1] + a*dt^2``.  Velocity is recovered as a backward
    difference, so it is the mean velocity over the step just taken.

    Verlet is used here (rather than, say, Runge-Kutta) because it is what the
    original program used, and because it is stable and cheap for a
    second-order system whose force depends only on position and velocity.

    Two deliberate differences from the original:

    * The **first step** uses the Taylor form ``x + v*dt + a*dt^2/2`` instead of
      back-extrapolating a fake previous position.  The original's seeding left
      a permanent half-step velocity error of ``a*dt/2`` (about 0.05 m/s under
      gravity at dt = 0.01), which accumulated to roughly 10 m of extra descent
      over a long flight.
    * The acceleration term carries ``dt*(dt + dt_prev)/2`` rather than
      ``dt^2``, which is the correct weight when the step size varies.  At a
      fixed step the two are identical, so fixed-step results are unchanged.
    """

    __slots__ = ("position", "velocity", "_previous", "_dt_previous")

    def __init__(self, position: Vec3, velocity: Vec3) -> None:
        self.position = position
        self.velocity = velocity
        self._previous: Vec3 | None = None
        self._dt_previous = 0.0

    def step(self, acceleration: Vec3, dt: float) -> None:
        if self._previous is None:
            nxt = self.position + self.velocity * dt + acceleration * (0.5 * dt * dt)
        else:
            ratio = dt / self._dt_previous
            nxt = (
                self.position
                + (self.position - self._previous) * ratio
                + acceleration * (0.5 * dt * (dt + self._dt_previous))
            )
        self._previous = self.position
        self.position = nxt
        self.velocity = (nxt - self._previous) / dt
        self._dt_previous = dt


class Parafoil:
    """The simulated vehicle: canopy parameters + wind + integration state.

    Typical use::

        vehicle = Parafoil(CanopyParams(), wind, position, velocity)
        while not vehicle.state.landed:
            vehicle.step(0.01, ControlInput(left=0.2, right=0.0))

    ``step`` is the only method that mutates the object; :attr:`state` always
    returns an immutable snapshot that is safe to store.
    """

    __slots__ = ("params", "wind", "air_density", "gravity", "_integrator",
                 "_frame", "_time", "_landed", "_control")

    def __init__(
        self,
        params: CanopyParams,
        wind: WindProfile,
        position: Vec3,
        velocity: Vec3,
        *,
        air_density: float = 1.23,
        gravity: float = 9.81,
    ) -> None:
        self.params = params
        self.wind = wind
        self.air_density = air_density
        self.gravity = Vec3(0.0, -gravity, 0.0)
        self._integrator: Integrator = VerletIntegrator(position, velocity)
        self._frame = BodyFrame.from_velocity(velocity)
        self._time = 0.0
        self._landed = position.y <= 0.0
        self._control = ControlInput()

    # ---- observation ----------------------------------------------------

    @property
    def state(self) -> FlightState:
        pos = self._integrator.position
        return FlightState(
            time=self._time,
            position=pos,
            velocity=self._integrator.velocity,
            wind=self.wind.at(pos.y),
            frame=self._frame,
            control=self._control,
            landed=self._landed,
        )

    # ---- the force model -------------------------------------------------

    def aerodynamic_force(self, control: ControlInput) -> Vec3:
        """Total aerodynamic force in world axes, in newtons.

        Also updates the body frame, because attitude in this model is a
        function of the force rather than an independent state.  The two steps
        are:

        1. Build a provisional frame with ``forward`` along the airflow and
           ``up`` roughly vertical, and evaluate the coefficients in it.
        2. Re-align ``up`` with the resultant force.  A side force therefore
           banks the whole system, exactly as a real canopy rolls into a turn.
        """
        pos = self._integrator.position
        wind = self.wind.at(pos.y)
        air_velocity = self._integrator.velocity - wind
        airspeed = air_velocity.norm()
        if airspeed < 1e-9:
            # No airflow, no aerodynamic force -- free fall for this step.
            return Vec3(0.0, 0.0, 0.0)

        # Step 1: provisional frame aligned with the airflow.
        forward = air_velocity / airspeed
        up_reference = (self._frame.up + UP * _UP_REFERENCE_BLEND).unit()
        side = forward.cross(up_reference).unit()
        up = side.cross(forward).unit()

        coeff = self.params.coefficients(control)
        coefficient_vector = (
            forward * coeff.drag + up * coeff.lift + side * coeff.side
        )

        # Step 2: the resultant force defines the new "up"; rebuild the frame
        # around it, keeping the side axis as the roll hinge.
        new_up = coefficient_vector.unit()
        new_forward = new_up.cross(side).unit()
        self._frame = BodyFrame(
            forward=new_forward,
            up=new_up,
            side=new_forward.cross(new_up).unit(),
        )

        return coefficient_vector * self.params.dynamic_pressure_force(
            self.air_density, airspeed
        )

    # ---- integration -----------------------------------------------------

    def step(self, dt: float, control: ControlInput) -> FlightState:
        """Advance the simulation by ``dt`` seconds and return the new state.

        Once the vehicle has touched down it is frozen: further calls are
        no-ops that return the landing state.
        """
        if self._landed or dt <= 0.0:
            return self.state

        self._control = control
        force = self.aerodynamic_force(control)
        acceleration = self.gravity + force / self.params.total_mass
        self._integrator.step(acceleration, dt)
        self._time += dt

        if self._integrator.position.y <= 0.0:
            self._landed = True
            # Put the vehicle exactly on the ground rather than below it.
            self._integrator.position = self._integrator.position.with_y(0.0)

        return self.state
