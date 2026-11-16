"""Fourth-order Runge-Kutta over the rigid-body state.

Why not Verlet, which the point-mass model uses?  Verlet advances a *second
order* equation ``x'' = a(x, x')`` by keeping two positions.  Attitude has no
such form: the kinematic relation ``q' = q * (0, w) / 2`` is first order in the
quaternion and the dynamics are first order in the angular velocity, and the
two are coupled.  Written as one first-order system, RK4 is the obvious choice
-- self-starting, fourth-order accurate, and it needs no history, which matters
because the quaternion has to be renormalised every step and a history-based
scheme would be carrying stale un-normalised copies.

The cost is four force evaluations per step instead of one.  Measured on this
project that is 13 700 steps/s down to roughly 3 400, which is still 34 times
faster than real time at ``dt = 0.01``.

The state is a :class:`RigidState` rather than a bare 13-element array on
purpose.  A flat vector saves nothing here (the arithmetic is on 3- and
4-element objects either way, see :mod:`parafoil.vectors`) and costs the one
thing that actually goes wrong in this kind of code: getting a component's
position or sign wrong somewhere between packing and unpacking, silently.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from ..vectors import Vec3
from .quaternion import Quat


class RigidState(NamedTuple):
    """Full 6-DOF state: 13 numbers, grouped so they cannot be misordered.

    Attributes:
        position: World position of the centre of mass, m.
        velocity: World velocity of the centre of mass, m/s.
        attitude: Body-to-world rotation.
        angular_velocity: **Body-frame** angular rate, rad/s.  Body frame is
            where the Euler equations naturally live, so keeping it there
            avoids a rotation on every evaluation.
    """

    position: Vec3
    velocity: Vec3
    attitude: Quat
    angular_velocity: Vec3

    # ---- vector-space operations, needed by the RK4 stages ---------------
    #
    # The quaternion is treated as a plain 4-vector during the stages: the
    # intermediate values are not unit quaternions and are not meant to be.
    # Only the final result of a step is projected back onto the unit sphere.

    def __add__(self, other: "RigidState") -> "RigidState":  # type: ignore[override]
        return RigidState(
            self.position + other.position,
            self.velocity + other.velocity,
            Quat(*(a + b for a, b in zip(self.attitude, other.attitude))),
            self.angular_velocity + other.angular_velocity,
        )

    def __mul__(self, scalar: float) -> "RigidState":  # type: ignore[override]
        return RigidState(
            self.position * scalar,
            self.velocity * scalar,
            Quat(*(a * scalar for a in self.attitude)),
            self.angular_velocity * scalar,
        )

    __rmul__ = __mul__  # type: ignore[assignment]

    def normalized(self) -> "RigidState":
        """Same state with the attitude projected back onto the unit sphere."""
        return self._replace(attitude=self.attitude.normalized())


#: A function giving the time derivative of the state.  It receives the time
#: and the state and returns a :class:`RigidState` holding *derivatives*:
#: velocity, acceleration, quaternion rate and angular acceleration.
Derivative = Callable[[float, RigidState], RigidState]


def rk4_step(
    derivative: Derivative, time: float, state: RigidState, dt: float
) -> RigidState:
    """Advance one step of classical RK4 and renormalise the attitude.

    The renormalisation is deliberately outside the stages: applying it to the
    intermediate ``y + k1*dt/2`` values would distort them, because those are
    not states but points along a derivative estimate.
    """
    k1 = derivative(time, state)
    k2 = derivative(time + 0.5 * dt, state + k1 * (0.5 * dt))
    k3 = derivative(time + 0.5 * dt, state + k2 * (0.5 * dt))
    k4 = derivative(time + dt, state + k3 * dt)

    increment = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
    return (state + increment).normalized()
