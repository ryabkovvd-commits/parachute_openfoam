"""Immutable 3D vector.

Why a hand-rolled class instead of ``numpy.ndarray``?  The simulation loop does
a few dozen operations on *three-element* vectors, tens of thousands of times.
At that size NumPy's per-call overhead dominates and buys nothing, because
there is no bulk work to amortise it over.  Measured on this machine, per
operation: addition is a wash, scaling ~1.9x faster here, dot ~3.7x, norm
~4.3x, and cross product ~43x (``np.cross`` is notoriously slow on single
vectors).  NumPy is still used, but where it belongs: storing and
post-processing whole trajectories (see :mod:`parafoil.simulation`).

``Vec3`` subclasses :class:`tuple`, so it is hashable, immutable and cheap to
create, and it unpacks naturally::

    x, y, z = position

Coordinate frame used throughout the project (inherited from the original
program): **X points east, Y points up (altitude), Z points south.**
"""

from __future__ import annotations

import math
from typing import NamedTuple


class Vec3(NamedTuple):
    """A 3D vector of floats with the usual arithmetic."""

    x: float
    y: float
    z: float

    # ---- arithmetic ---------------------------------------------------
    # `tuple` already defines __add__/__mul__ (concatenation and repetition),
    # so these overrides intentionally change the inherited behaviour.

    def __add__(self, other: "Vec3") -> "Vec3":  # type: ignore[override]
        return Vec3(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, scalar: float) -> "Vec3":  # type: ignore[override]
        return Vec3(self[0] * scalar, self[1] * scalar, self[2] * scalar)

    __rmul__ = __mul__  # type: ignore[assignment]

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self[0] / scalar, self[1] / scalar, self[2] / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self[0], -self[1], -self[2])

    # ---- products and norms -------------------------------------------

    def dot(self, other: "Vec3") -> float:
        return self[0] * other[0] + self[1] * other[1] + self[2] * other[2]

    def cross(self, other: "Vec3") -> "Vec3":
        ax, ay, az = self
        bx, by, bz = other
        return Vec3(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    def norm(self) -> float:
        """Euclidean length."""
        return math.sqrt(self[0] ** 2 + self[1] ** 2 + self[2] ** 2)

    def unit(self) -> "Vec3":
        """Same direction, unit length.

        Raises:
            ValueError: if the vector is (numerically) zero, which would make
                the direction undefined.  Failing loudly here beats silently
                propagating NaNs through the whole trajectory.
        """
        n = self.norm()
        if n < 1e-12:
            raise ValueError("cannot normalise a zero-length vector")
        return Vec3(self[0] / n, self[1] / n, self[2] / n)

    # ---- convenience ---------------------------------------------------

    def horizontal(self) -> "Vec3":
        """Copy with the vertical (Y) component removed."""
        return Vec3(self[0], 0.0, self[2])

    def with_y(self, y: float) -> "Vec3":
        return Vec3(self[0], y, self[2])


ZERO = Vec3(0.0, 0.0, 0.0)
UP = Vec3(0.0, 1.0, 0.0)
NORTH = Vec3(0.0, 0.0, -1.0)  # Z points south, so north is -Z


def signed_angle(a: Vec3, b: Vec3) -> float:
    """Angle from ``a`` to ``b`` in radians, signed about the vertical axis.

    The result is in ``[-pi, pi]`` and is **positive when ``b`` lies to the
    right of ``a``** for someone facing along ``a``.  Because Z points south,
    "right of east" is south, so ``signed_angle(east, south) == +pi/2``.

    That sign is not arbitrary: it has to agree with the direction the side
    force pushes.  Pulling the right toggle produces a force along
    ``forward x up``, which for an eastbound canopy points south -- so a
    positive angle means "pull the right toggle", and the steering law can use
    the sign directly.

    Zero-length inputs give ``0.0`` rather than raising, because the caller
    (the autopilot) legitimately hits that case when it is exactly on target.
    """
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    ua, ub = a / na, b / nb
    # Clamp guards against |cos| slightly exceeding 1 from rounding.
    angle = math.acos(max(-1.0, min(1.0, ua.dot(ub))))
    return angle if ub.cross(ua).y > 0.0 else -angle
