"""Unit quaternion for attitude.

Why quaternions rather than Euler angles: the attitude here is *integrated*,
and Euler angles have a coordinate singularity at 90 degrees of pitch where
the rates blow up.  A cargo parafoil is unlikely to get there, but "unlikely"
is a poor foundation for an integrator that runs 20 000 steps.  Quaternions
have no singularity, cost four numbers instead of three, and need only a
normalisation each step to stay on the unit sphere.

Convention, chosen to match the rest of the project rather than any textbook:

* the scalar part comes **first**, ``(w, x, y, z)``;
* a quaternion represents the rotation taking **body axes to world axes**, so
  :meth:`rotate` maps a body-frame vector into the world, and
  :meth:`inverse_rotate` maps the other way;
* composition reads left to right in the usual way, ``a * b`` meaning "apply
  ``b``, then ``a``".

Like :class:`~parafoil.vectors.Vec3`, this is a :class:`~typing.NamedTuple` of
plain floats: the integrator touches it four times per step on a four-element
object, which is exactly the size where NumPy costs more than it saves.

Euler angles are provided by :meth:`to_euler` for display only -- they are
never fed back into the integration, so the singularity cannot bite.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..vectors import Vec3


class Quat(NamedTuple):
    """Unit quaternion ``w + xi + yj + zk`` representing a body-to-world rotation."""

    w: float
    x: float
    y: float
    z: float

    # ---- construction ---------------------------------------------------

    @classmethod
    def identity(cls) -> "Quat":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: float) -> "Quat":
        """Rotation of ``angle`` radians about ``axis`` (need not be unit length).

        A zero-length axis gives the identity rather than raising: callers
        build these from angular velocities, which are legitimately zero.
        """
        length = axis.norm()
        if length < 1e-15:
            return cls.identity()
        half = 0.5 * angle
        s = math.sin(half) / length
        return cls(math.cos(half), axis.x * s, axis.y * s, axis.z * s)

    @classmethod
    def from_basis(cls, forward: Vec3, up: Vec3, side: Vec3) -> "Quat":
        """Quaternion for the rotation whose body axes are these world vectors.

        The body axes are ordered ``(forward, up, side)`` to match
        :class:`~parafoil.dynamics.BodyFrame` and the canopy mesh's local axes,
        so the rotation matrix has those three as its *columns*.

        Uses Shepperd's method: pick the largest of the four possible divisors
        so the square root is never taken of something near zero.  The naive
        formula loses all precision at 180 degrees.
        """
        # Columns are the body axes expressed in world coordinates.
        m00, m01, m02 = forward.x, up.x, side.x
        m10, m11, m12 = forward.y, up.y, side.y
        m20, m21, m22 = forward.z, up.z, side.z

        trace = m00 + m11 + m22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            return cls(
                0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
            ).normalized()
        if m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            return cls(
                (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
            ).normalized()
        if m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            return cls(
                (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
            ).normalized()
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        return cls(
            (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
        ).normalized()

    # ---- algebra ---------------------------------------------------------

    def __mul__(self, other: "Quat") -> "Quat":  # type: ignore[override]
        """Hamilton product: ``a * b`` applies ``b`` first, then ``a``."""
        aw, ax, ay, az = self
        bw, bx, by, bz = other
        return Quat(
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    def conjugate(self) -> "Quat":
        """Inverse rotation (the same thing for a unit quaternion)."""
        return Quat(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Quat":
        """Project back onto the unit sphere.

        Integration nudges the quaternion off the unit sphere a little every
        step; without this the represented rotation slowly acquires a scaling.

        Raises:
            ValueError: if the quaternion has collapsed to zero, which means
                the integration has already diverged and silently returning
                the identity would hide it.
        """
        n = self.norm()
        if n < 1e-12:
            raise ValueError("cannot normalise a zero quaternion")
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    # ---- action on vectors ----------------------------------------------

    def rotate(self, v: Vec3) -> Vec3:
        """Take a body-frame vector into world axes.

        Uses the expanded ``v + 2 * q_vec x (q_vec x v + w * v)`` form rather
        than ``q * v * q'``: same result, roughly half the multiplications and
        no temporary quaternions.
        """
        qv = Vec3(self.x, self.y, self.z)
        t = qv.cross(v) * 2.0
        return v + t * self.w + qv.cross(t)

    def inverse_rotate(self, v: Vec3) -> Vec3:
        """Take a world-frame vector into body axes."""
        return self.conjugate().rotate(v)

    # ---- kinematics ------------------------------------------------------

    def derivative(self, angular_velocity: Vec3) -> "Quat":
        """``dq/dt`` for a body-frame angular velocity.

        The standard kinematic relation ``q' = q * (0, w) / 2``.  The angular
        velocity is in **body** axes, which is where the Euler equations
        naturally produce it.
        """
        wx, wy, wz = angular_velocity
        aw, ax, ay, az = self
        return Quat(
            0.5 * (-ax * wx - ay * wy - az * wz),
            0.5 * (aw * wx + ay * wz - az * wy),
            0.5 * (aw * wy - ax * wz + az * wx),
            0.5 * (aw * wz + ax * wy - ay * wx),
        )

    # ---- readout ---------------------------------------------------------

    def basis(self) -> tuple[Vec3, Vec3, Vec3]:
        """The body axes ``(forward, up, side)`` expressed in world coordinates."""
        return (
            self.rotate(Vec3(1.0, 0.0, 0.0)),
            self.rotate(Vec3(0.0, 1.0, 0.0)),
            self.rotate(Vec3(0.0, 0.0, 1.0)),
        )

    def to_euler(self) -> tuple[float, float, float]:
        """``(roll, pitch, yaw)`` in radians, **for display only**.

        Defined against the project's axes (X east, Y up, Z south) with the
        body's ``forward`` along local X: yaw is the heading measured about the
        vertical, pitch the elevation of the nose, roll the rotation about the
        nose axis.

        Never feed these back into the state -- pitch has the usual singularity
        at +-90 degrees, which is exactly why the state is a quaternion.  At
        that singularity the split between roll and yaw is arbitrary, so this
        puts the whole rotation into yaw rather than returning a NaN.
        """
        forward, up, _ = self.basis()

        yaw = math.atan2(-forward.z, forward.x)  # -Z is north
        pitch = math.asin(max(-1.0, min(1.0, forward.y)))

        if abs(forward.y) > 1.0 - 1e-9:  # nose straight up or straight down
            return 0.0, pitch, yaw

        # Roll: how far `up` is rotated out of the vertical plane containing
        # the nose.  `level_up` is what `up` would be at zero roll -- world
        # vertical with its along-the-nose part removed -- and `level_side`
        # completes the pair, so the two dot products are the cosine and sine
        # of the roll angle.  Positive is a right bank, matching the sign
        # convention of :func:`~parafoil.vectors.signed_angle`.
        world_up = Vec3(0.0, 1.0, 0.0)
        level_up = (world_up - forward * world_up.dot(forward)).unit()
        level_side = forward.cross(level_up)
        return math.atan2(up.dot(level_side), up.dot(level_up)), pitch, yaw

    @property
    def bank_angle(self) -> float:
        """Tilt of the body ``up`` axis away from world vertical, in radians."""
        up = self.rotate(Vec3(0.0, 1.0, 0.0))
        return math.acos(max(-1.0, min(1.0, up.y)))
