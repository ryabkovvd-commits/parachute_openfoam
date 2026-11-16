"""The 6-DOF vehicle: same force law as the point-mass model, real rotation.

How this differs from :class:`~parafoil.dynamics.Parafoil`
----------------------------------------------------------

The point-mass model computes the aerodynamic force, then **defines** the
attitude to be whatever direction that force points.  Attitude is therefore an
output, not a state, and the canopy banks into a turn the instant a toggle
moves.

Here the attitude is a state.  The same force law is evaluated -- the same
``k(d)`` and ``C(d)`` from :class:`~parafoil.canopy.CanopyParams`, the same
side-force gain, so the two models agree on what force a given airflow and
toggle setting produce -- but the force is applied at the canopy, roughly 8.6 m
above the centre of mass, and the resulting moment ``r x F`` is what rotates
the body.  Nothing snaps into place; the canopy has to be rolled into a turn
by the side force, against its own inertia.

That single change is what buys the rotational behaviour the point-mass model
cannot show: lag between the toggle and the bank, overshoot, and the pendulum
oscillation of the load.

What is *not* modelled, and why
-------------------------------

There is no lift-curve slope, no stall, no sideslip derivative and no
aerodynamic damping derivatives, because the original program's data contains
none of them and inventing a dozen numbers would produce a model that looks
authoritative and is not.  Every quantity used here is either measured from
``points_*.txt`` (see :mod:`parafoil.rigid.inertia`) or already part of the
reconstructed force law.

The one exception is rotational damping, which cannot be omitted: an undamped
pendulum oscillates forever, which is worse than a rough estimate.  Rather than
invent plausible-looking values for ``Clp``, ``Cmq`` and ``Cnr``, it is
parameterised by the thing those coefficients would determine --
:attr:`RigidParafoil.damping_ratio`, how quickly the oscillation dies away --
and the coefficient is derived from the measured geometry.  That keeps the
assumption visible and in units anyone can argue with.
"""

from __future__ import annotations

import math

from ..atmosphere import WindProfile
from ..canopy import CanopyParams, ControlInput
from ..dynamics import BodyFrame, FlightState, Parafoil
from ..vectors import UP, Vec3
from .inertia import CanopyGeometry, MassProperties
from .integrators import RigidState, rk4_step
from .quaternion import Quat
from .strips import StripModel

#: Body-frame axes, for readability at the point of use.
_BODY_UP = Vec3(0.0, 1.0, 0.0)

#: Default fraction of critical damping for the hanging system's oscillation.
#: Cargo parafoils are lightly damped -- a disturbed load swings several times
#: before settling -- so this is small, but it is an assertion, not a
#: measurement.  See the module docstring.
DEFAULT_DAMPING_RATIO = 0.3

#: Airspeed the damping is referenced to -- the canopy's nominal trim speed.
#: Using a constant rather than the instantaneous trim speed keeps the damping
#: linear in the rate, which is what makes ``damping_ratio`` mean what it says.
_REFERENCE_AIRSPEED = 21.0


class RigidParafoil:
    """A rigid canopy-and-load system with six degrees of freedom.

    Mirrors the interface of :class:`~parafoil.dynamics.Parafoil` -- construct
    it, read :attr:`state`, call :meth:`step` -- so both can be driven by the
    same :func:`~parafoil.simulation.simulate` and the same pilots.

    Args:
        params: The same aerodynamic parameters the point-mass model uses.
        wind: Wind profile.
        position: Initial position of the **centre of mass**, world axes.
        velocity: Initial velocity, world axes.
        geometry: Canopy shape and rigging.  Defaults to the geometry measured
            from the original mesh.
        air_density: kg/m^3.
        gravity: m/s^2, positive.
        damping_ratio: Fraction of critical damping applied to body rotation.
            Only used by the ``"lumped"`` aerodynamics; the strip model gets
            its damping from the geometry instead.
        include_trapped_air: Whether the air carried inside the canopy counts
            towards the mass and inertia.  It roughly doubles the oscillation
            period, so turning it off is a useful experiment, not a tidy-up.
        aerodynamics: ``"lumped"`` (default) treats the canopy as one surface.
            ``"strip"`` resolves it across the span -- see
            :mod:`parafoil.rigid.strips` and the warning about its lateral
            behaviour there.  The default is the lumped model because it is
            the one that reproduces the trim anchors exactly and holds them:
            in straight flight it settles to a glide ratio of 4.000 and stays.
    """

    __slots__ = (
        "params", "wind", "air_density", "gravity", "geometry", "mass",
        "damping_ratio", "aerodynamics", "strips", "_damping", "_centre_of_mass",
        "_state", "_landed", "_control", "_elapsed",
    )

    def __init__(
        self,
        params: CanopyParams,
        wind: WindProfile,
        position: Vec3,
        velocity: Vec3,
        *,
        geometry: CanopyGeometry | None = None,
        air_density: float = 1.23,
        gravity: float = 9.81,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
        include_trapped_air: bool = True,
        aerodynamics: str = "lumped",
        strips: "StripModel | None" = None,
    ) -> None:
        if damping_ratio < 0.0:
            raise ValueError("damping_ratio cannot be negative")
        if aerodynamics not in ("strip", "lumped"):
            raise ValueError(
                f"unknown aerodynamics {aerodynamics!r}; expected 'strip' or 'lumped'"
            )

        self.params = params
        self.wind = wind
        self.air_density = air_density
        self.gravity = gravity
        self.geometry = geometry or CanopyGeometry.measured()
        self.mass = MassProperties.build(
            params,
            self.geometry,
            air_density,
            include_trapped_air=include_trapped_air,
        )
        self.damping_ratio = damping_ratio
        self._damping = self._damping_coefficients()

        self.aerodynamics = aerodynamics
        self.strips = (
            (strips or StripModel.measured(params)) if aerodynamics == "strip" else None
        )
        # Strip levers are measured from the mesh origin, so the strip model
        # needs to know where the centre of mass sits in those coordinates.
        self._centre_of_mass = self.geometry.aero_centre - self.mass.aero_offset

        self._state = RigidState(
            position=position,
            velocity=velocity,
            attitude=self._trimmed_attitude(position, velocity),
            angular_velocity=Vec3(0.0, 0.0, 0.0),
        )
        self._landed = position.y <= 0.0
        self._control = ControlInput()
        self._elapsed = 0.0

    # ---- set-up ----------------------------------------------------------

    def _damping_coefficients(self) -> Vec3:
        """Damping moment per unit body rate, one per axis, N m s/rad.

        Derived from :attr:`damping_ratio` rather than from stability
        derivatives: for each axis, ``c = 2 * zeta * sqrt(k * I)`` with ``k``
        the pendulum stiffness the geometry already implies and ``I`` that
        axis's inertia.  Expressed this way the assumption is legible -- "the
        oscillation decays with this damping ratio" -- instead of hidden
        inside a coefficient nobody can check.
        """
        stiffness = self.mass.aero_offset.norm() * self.mass.mass * self.gravity
        rows = self.mass.inertia.rows
        return Vec3(
            *(
                2.0 * self.damping_ratio * math.sqrt(max(stiffness * rows[i][i], 0.0))
                for i in range(3)
            )
        )

    def _trimmed_attitude(self, position: Vec3, velocity: Vec3) -> Quat:
        """Start at the attitude the system would hang at, not an arbitrary one.

        The canopy's lever arm points essentially straight up in body axes, so
        equilibrium is where the aerodynamic force is parallel to it.  The
        point-mass model computes exactly that frame -- it is the frame it
        assigns itself every step -- so it is borrowed here.  Without this the
        flight opens with a 14 degree pitch transient that has nothing to do
        with what is being simulated.
        """
        probe = Parafoil(
            params=self.params,
            wind=self.wind,
            position=position,
            velocity=velocity,
            air_density=self.air_density,
            gravity=self.gravity,
        )
        probe.aerodynamic_force(ControlInput())
        frame = probe.state.frame
        return Quat.from_basis(frame.forward, frame.up, frame.side)

    # ---- observation -----------------------------------------------------

    @property
    def state(self) -> FlightState:
        """Immutable snapshot, in the same shape the point-mass model returns."""
        s = self._state
        forward, up, side = s.attitude.basis()
        return FlightState(
            time=self._elapsed,
            position=s.position,
            velocity=s.velocity,
            wind=self.wind.at(s.position.y),
            frame=BodyFrame(forward=forward, up=up, side=side),
            control=self._control,
            landed=self._landed,
            attitude=s.attitude,
            angular_velocity=s.angular_velocity,
        )

    # ---- forces and moments ---------------------------------------------

    def force_and_moment(
        self, state: RigidState, control: ControlInput
    ) -> tuple[Vec3, Vec3]:
        """Aerodynamic force (world axes) and moment about the CG (body axes).

        The force is the point-mass model's, evaluated in body axes:

        * drag along the airflow,
        * lift perpendicular to it, in the canopy's plane of symmetry -- which
          is where this model differs, because that plane comes from the
          integrated attitude rather than from the force itself,
        * side force along the body's spanwise axis.

        The moment is then ``r x F`` about the centre of mass, plus the
        rotational damping described in the module docstring.
        """
        wind = self.wind.at(state.position.y)
        air_velocity = state.velocity - wind
        airspeed = air_velocity.norm()
        if airspeed < 1e-9:
            return Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)

        air_velocity_body = state.attitude.inverse_rotate(air_velocity)

        if self.strips is not None:
            force_body, moment = self.strips.force_and_moment(
                self.params,
                control,
                air_velocity_body,
                state.angular_velocity,
                self._centre_of_mass,
                self.air_density,
            )
            return state.attitude.rotate(force_body), moment

        # Airflow in body axes: the canopy feels the flow in its own frame.
        flow = air_velocity_body / airspeed

        # Same construction as the point-mass model, but "up" is the body's.
        side_dir = flow.cross(_BODY_UP)
        if side_dir.norm() < 1e-9:
            # Flying exactly along the body vertical: the symmetry plane no
            # longer picks out a lift direction.  Drag alone is the honest
            # answer for this step.
            coefficients = self.params.coefficients(control)
            force_body = flow * coefficients.drag
        else:
            side_dir = side_dir.unit()
            lift_dir = side_dir.cross(flow).unit()
            coefficients = self.params.coefficients(control)
            force_body = (
                flow * coefficients.drag
                + lift_dir * coefficients.lift
                + side_dir * coefficients.side
            )

        force_body = force_body * self.params.dynamic_pressure_force(
            self.air_density, airspeed
        )

        moment = self.mass.aero_offset.cross(force_body)

        # Damping scales with airspeed: no airflow, no aerodynamic damping.
        scale = airspeed / _REFERENCE_AIRSPEED
        omega = state.angular_velocity
        moment = moment - Vec3(
            self._damping.x * omega.x * scale,
            self._damping.y * omega.y * scale,
            self._damping.z * omega.z * scale,
        )

        return state.attitude.rotate(force_body), moment

    def _derivative(self, control: ControlInput):
        """Build the right-hand side for the integrator at a fixed control input."""

        def derivative(time: float, state: RigidState) -> RigidState:
            force, moment = self.force_and_moment(state, control)
            acceleration = Vec3(0.0, -self.gravity, 0.0) + force / self.mass.mass

            omega = state.angular_velocity
            gyroscopic = omega.cross(self.mass.inertia.mul(omega))
            angular_acceleration = self.mass.inertia.solve(moment - gyroscopic)

            return RigidState(
                position=state.velocity,
                velocity=acceleration,
                attitude=state.attitude.derivative(omega),
                angular_velocity=angular_acceleration,
            )

        return derivative

    # ---- integration -----------------------------------------------------

    def step(self, dt: float, control: ControlInput) -> FlightState:
        """Advance by ``dt`` seconds and return the new state.

        Once landed the vehicle is frozen, matching the point-mass model.
        """
        if self._landed or dt <= 0.0:
            return self.state

        self._control = control
        self._state = rk4_step(
            self._derivative(control), self._elapsed, self._state, dt
        )
        self._elapsed += dt

        if self._state.position.y <= 0.0:
            self._landed = True
            self._state = self._state._replace(
                position=self._state.position.with_y(0.0)
            )

        return self.state
