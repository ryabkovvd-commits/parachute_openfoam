"""Where the mass is, and how hard it is to rotate.

The point-mass model needs one number, the total mass.  A rigid body needs to
know how that mass is *arranged*, and this module works it out.

Almost everything here is **measured from the original data** rather than
assumed.  ``points_*.txt`` is a 2730-node mesh of the real canopy, so the span,
the chord, the section thickness, the distance from the load to the canopy and
the distribution of canopy area are all readable off it.  The one genuinely
estimated quantity is the mass of air the canopy carries with it, and it is
isolated in :attr:`CanopyGeometry.section_fill` with its uncertainty spelled
out, because it turns out to dominate the rotational response.

Frame convention
----------------

Body axes are the canopy mesh's own local axes -- ``(along course, up,
spanwise)`` -- which :mod:`parafoil.mesh` already maps onto
``(forward, up, side)``.  The mesh's *origin* is not a physical point: the
nearest vertex to it is 4.4 m away.  The load hangs at the bottom of the
rigging, which the mesh does contain, so that point is measured rather than
assumed.

The tensors below are expressed in those axes with offsets measured **from the
centre of mass**, which is the origin the rigid-body equations want.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..canopy import CanopyParams, ControlInput
from ..vectors import Vec3

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..mesh import CanopyMesh


class Mat3:
    """A 3x3 matrix stored as nine floats, with its inverse precomputed.

    Only two operations are ever needed -- ``I * w`` and ``I^-1 * M`` -- and
    both happen four times per integration step.  NumPy is used once, at
    construction, to invert; the per-step arithmetic is plain Python for the
    same reason :class:`~parafoil.vectors.Vec3` is (see that module's note on
    measured overheads at this size).
    """

    __slots__ = ("rows", "_inverse_rows")

    def __init__(self, rows: tuple[tuple[float, float, float], ...]) -> None:
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise ValueError("Mat3 needs three rows of three")
        self.rows = tuple(tuple(float(x) for x in row) for row in rows)
        self._inverse_rows: tuple[tuple[float, float, float], ...] | None = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Mat3({self.rows})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mat3) and self.rows == other.rows

    def __hash__(self) -> int:
        # The cached inverse is a cache, not state, so it stays out of both
        # equality and the hash -- and defining one without the other would
        # make any frozen dataclass holding a Mat3 unhashable.
        return hash(self.rows)

    def mul(self, v: Vec3) -> Vec3:
        """Matrix-vector product."""
        (a, b, c), (d, e, f), (g, h, i) = self.rows
        return Vec3(
            a * v.x + b * v.y + c * v.z,
            d * v.x + e * v.y + f * v.z,
            g * v.x + h * v.y + i * v.z,
        )

    @property
    def inverse(self) -> "Mat3":
        """The inverse, computed once on first use.

        Raises:
            ValueError: if the matrix is singular.  For an inertia tensor that
                means the mass distribution has collapsed onto a line, which
                is a modelling error worth failing loudly on.
        """
        if self._inverse_rows is None:
            import numpy as np

            try:
                inv = np.linalg.inv(np.array(self.rows, dtype=np.float64))
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"matrix is singular and cannot be inverted: {exc}")
            self._inverse_rows = tuple(tuple(float(x) for x in row) for row in inv)
        result = Mat3.__new__(Mat3)
        result.rows = self._inverse_rows
        result._inverse_rows = self.rows
        return result

    def solve(self, v: Vec3) -> Vec3:
        """``M^-1 * v``."""
        return self.inverse.mul(v)

    @classmethod
    def zeros(cls) -> "Mat3":
        return cls(((0.0, 0.0, 0.0),) * 3)

    def __add__(self, other: "Mat3") -> "Mat3":
        return Mat3(
            tuple(
                tuple(a + b for a, b in zip(row_a, row_b))
                for row_a, row_b in zip(self.rows, other.rows)
            )
        )

    @classmethod
    def point_inertia(cls, mass: float, offset: Vec3) -> "Mat3":
        """Inertia tensor of a point mass at ``offset`` about the origin.

        ``I = m * (|r|^2 * delta - r r^T)`` -- the parallel-axis term, which is
        all a point mass has.
        """
        x, y, z = offset
        r2 = x * x + y * y + z * z
        return cls(
            (
                (mass * (r2 - x * x), -mass * x * y, -mass * x * z),
                (-mass * y * x, mass * (r2 - y * y), -mass * y * z),
                (-mass * z * x, -mass * z * y, mass * (r2 - z * z)),
            )
        )


@dataclass(frozen=True, slots=True)
class CanopyGeometry:
    """Shape and rigging of the canopy, in body axes from the confluence point.

    Attributes:
        span: Tip to tip, m.
        chord: Leading to trailing edge, m.
        thickness: Median section depth, m.
        planform_area: Span times chord, m^2 -- a cross-check on the reference
            area the aerodynamic model uses.
        canopy_centre: Area centroid of the fabric, in mesh coordinates.
        aero_centre: Where the resultant aerodynamic force is taken to act.
        payload_centre: Where the load hangs -- the bottom of the rigging.
        canopy_inertia_per_mass: The canopy's own inertia tensor about its
            centroid, divided by its mass (so it is in m^2 and can be scaled
            by whatever mass the fabric plus trapped air comes to).

            This is **not** a refinement that can be skipped.  The payload and
            the canopy sit almost exactly on one line, so treating both as
            point masses makes the system a dumbbell -- and a dumbbell has no
            inertia at all about its own axis.  The resulting tensor is
            near-singular, its inverse is enormous, and the angular
            acceleration diverges on the first step.  The 17 m of span is what
            makes the problem well posed.
        section_fill: Fraction of the ``chord x thickness`` bounding box that
            the aerofoil section actually encloses, used only to estimate the
            trapped air mass.  **This is the one estimated number here.**  See
            :meth:`enclosed_air_volume`.
    """

    span: float
    chord: float
    thickness: float
    planform_area: float
    canopy_centre: Vec3
    aero_centre: Vec3
    payload_centre: Vec3
    canopy_inertia_per_mass: Mat3
    section_fill: float = 0.7

    def __post_init__(self) -> None:
        for name in ("span", "chord", "thickness", "planform_area"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.section_fill <= 1.0:
            raise ValueError("section_fill is a fraction and must lie in [0, 1]")

    @property
    def aspect_ratio(self) -> float:
        return self.span / self.chord

    def enclosed_air_volume(self) -> float:
        """Volume of air carried inside the canopy, m^3.

        Estimated as ``planform area x thickness x section_fill``.  Two
        independent checks on the default 0.7 for the original canopy:

        * summing the mesh polygons with the divergence theorem gives 106 m^3,
          but the ram-air cells are open at the leading edge so that surface is
          not closed (the closure residual is 65 m^2 out of 282) and the figure
          is an upper bound;
        * the default gives 58 m^3, i.e. 71 kg of air against the canopy's own
          23.3 kg.

        The true value is somewhere between, and it matters: the trapped air
        sits at the end of a 10 m lever and dominates the rotational inertia,
        so it roughly doubles the oscillation period.  Anyone with better data
        should set this rather than accept the estimate.
        """
        return self.planform_area * self.thickness * self.section_fill

    @classmethod
    def from_mesh(cls, mesh: "CanopyMesh", **overrides) -> "CanopyGeometry":
        """Measure the geometry off a loaded canopy mesh, hands up.

        The fabric panels give the planform and the area centroid; the rigging
        lines give where the load hangs; the spread in height across the
        surface gives the section thickness.
        """
        import numpy as np

        vertices = mesh.deform(ControlInput())
        quads = vertices[mesh.polygons]

        # Area and centroid of each panel, via its two triangular halves.
        a, b, c, d = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
        n1 = 0.5 * np.cross(b - a, c - a)
        n2 = 0.5 * np.cross(c - a, d - a)
        area1, area2 = np.linalg.norm(n1, axis=1), np.linalg.norm(n2, axis=1)
        panel_area = area1 + area2
        panel_centre = (
            area1[:, None] * (a + b + c) / 3.0 + area2[:, None] * (a + c + d) / 3.0
        ) / np.maximum(panel_area, 1e-12)[:, None]

        fabric = vertices[np.unique(mesh.polygons.ravel())]
        span = float(np.ptp(fabric[:, 2]))
        chord = float(np.ptp(fabric[:, 0]))

        # Section thickness: spread in height within each cell of a coarse
        # planform grid, taken as the median so the open leading edge and the
        # tips do not skew it.
        cell = 0.4
        keys = np.round(fabric[:, 0] / cell).astype(np.int64) * 100000 + np.round(
            fabric[:, 2] / cell
        ).astype(np.int64)
        order = np.argsort(keys)
        heights = fabric[order, 1]
        splits = np.flatnonzero(np.diff(keys[order])) + 1
        spreads = [np.ptp(g) for g in np.split(heights, splits) if len(g) > 1]
        thickness = float(np.median(spreads)) if spreads else chord * 0.15

        total_area = panel_area.sum()
        centre = panel_centre * panel_area[:, None]
        canopy_centre = Vec3(*(centre.sum(axis=0) / total_area))

        # The canopy's own inertia, per unit mass, about its centroid: mass is
        # spread over the panels in proportion to their area.  This is what
        # keeps the assembled tensor well conditioned; see the class docstring.
        offsets = panel_centre - np.asarray(canopy_centre, dtype=np.float64)
        weights = panel_area / total_area
        r2 = (offsets * offsets).sum(axis=1)
        canopy_inertia = Mat3(
            tuple(
                tuple(
                    float(
                        (weights * ((r2 if i == j else 0.0)
                                    - offsets[:, i] * offsets[:, j])).sum()
                    )
                    for j in range(3)
                )
                for i in range(3)
            )
        )

        # The load hangs at the bottom of the rigging.
        line_vertices = vertices[np.unique(mesh.lines.ravel())]
        lowest = line_vertices[
            line_vertices[:, 1] < line_vertices[:, 1].min() + 0.05
        ].mean(axis=0)

        # Aerodynamic centre at quarter chord aft of the leading edge, at
        # mid-span, on the mean surface.  This sets the attitude at which the
        # system hangs; it does not affect the trajectory, because in this
        # model the aerodynamic force depends on the airflow rather than on
        # the body's angle of attack.
        leading_edge = float(fabric[:, 0].max())
        quarter = leading_edge - 0.25 * chord
        near = fabric[np.abs(fabric[:, 0] - quarter) < 0.5 * cell]
        aero_height = float(near[:, 1].mean()) if len(near) else float(canopy_centre.y)

        geometry = cls(
            span=span,
            chord=chord,
            thickness=thickness,
            planform_area=span * chord,
            canopy_centre=canopy_centre,
            aero_centre=Vec3(quarter, aero_height, 0.0),
            payload_centre=Vec3(float(lowest[0]), float(lowest[1]), 0.0),
            canopy_inertia_per_mass=canopy_inertia,
        )
        return geometry.replace(**overrides) if overrides else geometry

    @classmethod
    def measured(cls, **overrides) -> "CanopyGeometry":
        """The original canopy's geometry, as measured from ``points_*.txt``.

        Hard-coded so the rigid model runs without the 1.3 MB of mesh files --
        the point-mass model works without them too, and that property is
        worth keeping.  Regenerate with :meth:`from_mesh` if the data changes.
        """
        return cls(
            span=17.4124,
            chord=5.8024,
            thickness=0.8245,
            planform_area=101.0344,
            canopy_centre=Vec3(2.9680, 8.6988, 0.0),
            aero_centre=Vec3(4.3492, 8.4355, 0.0),
            payload_centre=Vec3(4.3793, -0.6172, 0.0),
            # Per unit mass, m^2.  The two large diagonals are the 17.4 m
            # span (b^2/12 = 25.3); the small one is the 5.8 m chord.
            canopy_inertia_per_mass=Mat3(
                (
                    (28.2748, -0.0751, 0.0006),
                    (-0.0751, 28.8167, -0.0003),
                    (0.0006, -0.0003, 4.1648),
                )
            ),
        ).replace(**overrides)

    def replace(self, **changes) -> "CanopyGeometry":
        import dataclasses

        return dataclasses.replace(self, **changes) if changes else self


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Total mass, inertia tensor, and the lever the aerodynamic force acts on.

    All offsets are in body axes measured **from the centre of mass**, which is
    the origin the rigid-body equations use.

    Attributes:
        mass: Canopy + payload + trapped air, kg.  The trapped air is included
            because it is accelerated along with the canopy.
        inertia: Tensor about the centre of mass, kg m^2.
        aero_offset: Centre of mass to the aerodynamic centre.  This is the
            lever that turns force into moment, and it is the whole reason the
            model has any rotational dynamics at all.
        included_air_mass: The estimated part of :attr:`mass`, kept separate so
            its influence can be seen and switched off.
    """

    mass: float
    inertia: Mat3
    aero_offset: Vec3
    included_air_mass: float

    @classmethod
    def build(
        cls,
        params: CanopyParams,
        geometry: CanopyGeometry,
        air_density: float = 1.23,
        *,
        include_trapped_air: bool = True,
    ) -> "MassProperties":
        """Assemble the mass properties from the canopy parameters and geometry.

        Three contributions, each a point mass at its own place:

        * the payload, at the bottom of the rigging -- 1880 kg of the 1975 kg
          total, so the centre of mass sits essentially at the load;
        * the canopy fabric, at its area centroid some 9 m above;
        * the trapped air, at the same place as the fabric, and about three
          times its mass.

        A check that falls out of this and was not designed for: the resulting
        centre of mass ends up within 4 cm, horizontally, of the aerodynamic
        centre.  That is what a correctly rigged canopy should do -- it hangs
        its load directly under the lift -- and it comes purely from the
        measured mesh, so it is real corroboration that the geometry has been
        read off correctly.

        On top of those parallel-axis terms comes the canopy's own inertia
        about its centroid, from
        :attr:`CanopyGeometry.canopy_inertia_per_mass`.  It is tempting to
        skip that term -- against the ``10^4 kg m^2`` the lever arm produces
        it looks like a rounding error -- but skipping it is fatal.  The load
        and the canopy lie almost on one line, so without the canopy's 17 m of
        span the tensor is a dumbbell's: singular about the rigging axis, and
        the first inversion sends the angular acceleration to infinity.
        """
        air_mass = (
            air_density * geometry.enclosed_air_volume() if include_trapped_air else 0.0
        )

        payload = params.payload_mass
        canopy = params.canopy_mass + air_mass
        total = payload + canopy
        if total <= 0.0:
            raise ValueError("total mass must be positive")

        centre_of_mass = (
            geometry.payload_centre * payload + geometry.canopy_centre * canopy
        ) / total

        spread = Mat3(
            tuple(
                tuple(value * canopy for value in row)
                for row in geometry.canopy_inertia_per_mass.rows
            )
        )
        inertia = (
            Mat3.point_inertia(payload, geometry.payload_centre - centre_of_mass)
            + Mat3.point_inertia(canopy, geometry.canopy_centre - centre_of_mass)
            + spread
        )

        return cls(
            mass=total,
            inertia=inertia,
            aero_offset=geometry.aero_centre - centre_of_mass,
            included_air_mass=air_mass,
        )

    def pendulum_period(self, gravity: float = 9.81) -> float:
        """Small-oscillation period of the hanging system, seconds.

        In steady flight the aerodynamic force balances the weight, so tilting
        the body by a small angle produces a restoring moment
        ``|r| * m * g * theta``.  With the inertia about the side axis that
        gives ``T = 2 pi sqrt(I / (|r| m g))``.

        This is the number to sanity-check the model against: it is set
        entirely by geometry and mass, owes nothing to any aerodynamic
        coefficient, and is what an oscillating load on a real system would
        show.

        Two caveats on reading it.  It assumes **steady flight** -- away from
        trim the aerodynamic force is not the weight, and the stiffness scales
        with the square of airspeed.  And it is the *uncoupled* rotational
        mode: in a real descent this fast swing (about 1.4 s here) damps out
        within a couple of cycles and leaves a much slower pitch-and-speed
        oscillation of the whole trajectory, nearer 4 s, which this formula
        says nothing about.
        """
        lever = self.aero_offset.norm()
        stiffness = lever * self.mass * gravity
        side_inertia = self.inertia.rows[2][2]
        return 2.0 * math.pi * math.sqrt(side_inertia / stiffness)
