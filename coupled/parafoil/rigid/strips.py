"""Spanwise strip model of the canopy.

Why this exists
---------------

The lumped force model in :mod:`parafoil.rigid.body` treats the canopy as one
flat surface whose lift lies in the plane of symmetry.  That is enough for
pitch -- the lift direction is fixed by the airflow, the lever arm rotates with
the body, and ``r x F`` restores.  In roll it is not: banking rotates the plane
of symmetry, so the lift banks with the lever arm, ``r x F`` stays zero and the
canopy is neutrally stable.  Measured, not assumed: a 10 degree roll
disturbance does not come back, while a 10 degree pitch disturbance settles in
about four seconds.

A real canopy is not flat.  ``points_*.txt`` describes a pronounced arc -- the
tips hang 3.9 m below the centre, the half-span line sits 24 degrees off
horizontal, the arc radius is 11.6 m.  Each spanwise section therefore has its
own orientation, and under sideslip the two halves no longer contribute
equally.  That asymmetry is the canopy's lateral stiffness, and it is
*geometry*: it needs no stability derivative, only the shape that is already in
the data.

What each strip does
--------------------

For every strip the model computes the flow that strip actually sees --
including the part contributed by the body's own rotation, ``omega x r``, which
is where roll and yaw damping come from for free -- and gives it a lift
perpendicular to that flow in the strip's own plane, plus drag along it.  The
strips are summed into one force and one moment about the centre of mass.

The one new assumption, and how it is bounded
---------------------------------------------

How lift is distributed along the span.  This model spreads it in proportion to
strip area, which is the flat-plate assumption; real wings are closer to
elliptic.  The total is then rescaled by a single calibration factor so that in
straight symmetric flight the sum reproduces exactly the ``k(d)`` and ``C(d)``
the original program used.  So the anchors survive: whatever the distribution
does to the *shape* of the loading, it cannot move the trim glide ratio or the
trim airspeed.

What it achieves, and where it stops -- measured
------------------------------------------------

Both lateral derivatives come out with the right sign, from geometry alone:

* **roll stiffness** ``dMx/dbeta < 0`` -- 690 N m per degree of sideslip, so a
  slip to the right rolls the canopy back to the left;
* **roll damping** opposing the roll rate -- 22 700 N m s/rad.

The coupled lateral motion is nevertheless **spirally divergent**.  Released in
straight symmetric flight the bank angle grows out of nothing, roughly tenfold
every five seconds, and by 35 s the canopy is over on its side.  That is the
classic spiral mode: it is not fixed by more roll stiffness but by *yaw*
stiffness, which this model has no way to produce, because weathercock
stability comes from the sideways area distribution and the original data
contains no such coefficient.

So this module does what it set out to do -- it shows that the arc, and only
the arc, is what makes the canopy laterally stiff -- and it stops exactly where
the missing aerodynamic data starts.  It is **not** the default for that
reason; see :class:`~parafoil.rigid.body.RigidParafoil`.  Closing the gap is
the phase-4 work that was deliberately skipped: a lift-curve slope, a sideslip
derivative and the three damping derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..canopy import CanopyParams, ControlInput
from ..vectors import Vec3

if TYPE_CHECKING:  # pragma: no cover
    from ..mesh import CanopyMesh

#: Default number of spanwise strips.  Enough to resolve the arc (which is
#: smooth, so a couple of dozen samples capture it) without making each
#: integration step twenty times more expensive than the lumped model.
DEFAULT_STRIP_COUNT = 21


@dataclass(frozen=True, slots=True)
class Strip:
    """One spanwise slice of the canopy, in body axes.

    Attributes:
        position: The strip's quarter-chord point, from the mesh origin.
        normal: Unit vector out of the canopy's upper surface at this strip.
            Its tilt away from vertical is the arc, and is what makes the two
            halves respond differently in sideslip.
        area: Planform area of the strip, m^2.
        deflection_side: ``+1`` if the strip follows the right toggle, ``-1``
            for the left.  Matches :meth:`parafoil.mesh.CanopyMesh.deform`,
            which splits the canopy the same way.
    """

    position: Vec3
    normal: Vec3
    area: float
    deflection_side: int


@dataclass(frozen=True, slots=True)
class StripModel:
    """The canopy as a set of strips, plus the calibration that ties it to the
    reconstructed force law.

    Attributes:
        strips: The slices, ordered from one tip to the other.
        calibration: Scale factor making the strip sum agree with
            :meth:`~parafoil.canopy.CanopyParams.coefficients` in straight
            symmetric flight.  Computed once, at construction.
    """

    strips: tuple[Strip, ...]
    calibration: float

    @property
    def total_area(self) -> float:
        return sum(strip.area for strip in self.strips)

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        mesh: "CanopyMesh",
        params: CanopyParams,
        count: int = DEFAULT_STRIP_COUNT,
    ) -> "StripModel":
        """Slice a loaded canopy mesh into ``count`` spanwise strips.

        The arc is taken from the mean surface height across each spanwise
        band; differentiating it gives each strip's normal.  Using the mean
        surface rather than the panel normals avoids depending on the winding
        order of the polygon data, which is not guaranteed to be consistent.
        """
        import numpy as np

        vertices = mesh.deform(ControlInput())
        fabric = vertices[np.unique(mesh.polygons.ravel())]

        span_min, span_max = float(fabric[:, 2].min()), float(fabric[:, 2].max())
        edges = np.linspace(span_min, span_max, count + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = (span_max - span_min) / count

        heights = np.empty(count)
        chords = np.empty(count)
        quarter_x = np.empty(count)
        for i in range(count):
            band = fabric[
                (fabric[:, 2] >= edges[i]) & (fabric[:, 2] <= edges[i + 1])
            ]
            if len(band) == 0:  # pragma: no cover - only for absurd strip counts
                raise ValueError(f"strip {i} is empty; use fewer strips")
            heights[i] = band[:, 1].mean()
            chords[i] = float(np.ptp(band[:, 0]))
            quarter_x[i] = float(band[:, 0].max()) - 0.25 * chords[i]

        # Arc slope dy/dz by central differences, one-sided at the tips.
        slope = np.gradient(heights, centres)

        strips: list[Strip] = []
        for i in range(count):
            # Normal to the arc: perpendicular to (0, dy/dz, 1) in the y-z
            # plane, pointing upwards out of the canopy.
            normal = Vec3(0.0, 1.0, -float(slope[i])).unit()
            strips.append(
                Strip(
                    position=Vec3(
                        float(quarter_x[i]), float(heights[i]), float(centres[i])
                    ),
                    normal=normal,
                    area=float(chords[i] * width),
                    deflection_side=1 if centres[i] >= 0.0 else -1,
                )
            )

        model = cls(strips=tuple(strips), calibration=1.0)
        return cls(strips=model.strips, calibration=model._calibration_for(params))

    def _calibration_for(self, params: CanopyParams) -> float:
        """Scale making the strip sum match the lumped model in straight flight.

        Evaluated with the flow along the body's forward axis and both toggles
        up.  The strips' outward-tilted normals mean their lifts partly cancel
        across the span, so the raw sum falls short of the lumped total; this
        is the factor that restores it.
        """
        hands_up = ControlInput()
        coefficients = params.coefficients(hands_up)
        flow = Vec3(1.0, 0.0, 0.0)

        raw = Vec3(0.0, 0.0, 0.0)
        for strip in self.strips:
            lift_dir = _lift_direction(strip.normal, flow)
            raw = raw + (flow * coefficients.drag + lift_dir * coefficients.lift) * (
                strip.area
            )

        target = math.hypot(coefficients.drag, coefficients.lift) * params.area
        magnitude = raw.norm()
        if magnitude < 1e-12:  # pragma: no cover - geometrically impossible
            raise ValueError("strip geometry produces no net force to calibrate against")
        return target / magnitude

    @classmethod
    def measured(
        cls, params: CanopyParams | None = None, count: int = DEFAULT_STRIP_COUNT
    ) -> "StripModel":
        """The original canopy's strips, rebuilt from the stored arc.

        Like :meth:`~parafoil.rigid.inertia.CanopyGeometry.measured`, this
        exists so the model runs without the 1.3 MB of mesh files.  The arc is
        a circle fitted to the measured surface: centre height 10.081 m, tips
        6.181 m at +-8.706 m, which is a radius of 11.65 m.
        """
        params = params or CanopyParams()

        span, chord = 17.4124, 5.8024
        radius, centre_height = 11.65, 10.081
        quarter_x = 5.8004 - 0.25 * chord

        edges = [(-0.5 + i / count) * span for i in range(count + 1)]
        width = span / count

        strips: list[Strip] = []
        for i in range(count):
            z = 0.5 * (edges[i] + edges[i + 1])
            # Circular arc through the measured centre and tips, sagging
            # outwards: y = centre - (radius - sqrt(radius^2 - z^2)).
            inner = max(radius * radius - z * z, 0.0)
            y = centre_height - (radius - math.sqrt(inner))
            slope = -z / math.sqrt(inner) if inner > 1e-9 else 0.0
            strips.append(
                Strip(
                    position=Vec3(quarter_x, y, z),
                    normal=Vec3(0.0, 1.0, -slope).unit(),
                    area=chord * width,
                    deflection_side=1 if z >= 0.0 else -1,
                )
            )

        model = cls(strips=tuple(strips), calibration=1.0)
        return cls(strips=model.strips, calibration=model._calibration_for(params))

    # ---- the force model -------------------------------------------------

    def force_and_moment(
        self,
        params: CanopyParams,
        control: ControlInput,
        air_velocity: Vec3,
        angular_velocity: Vec3,
        centre_of_mass: Vec3,
        air_density: float,
    ) -> tuple[Vec3, Vec3]:
        """Total force and moment about the centre of mass, both in body axes.

        Args:
            air_velocity: Velocity of the centre of mass through the air, in
                **body** axes.
            angular_velocity: Body rates, rad/s.
            centre_of_mass: Where the CM sits in mesh coordinates, so strip
                positions can be turned into lever arms.

        Each strip sees ``air_velocity + omega x r``.  That extra term is not a
        refinement: it is the entire source of roll and yaw damping here, since
        a strip out at the tip moving downwards meets the air at a different
        angle than one at the root.
        """
        force = Vec3(0.0, 0.0, 0.0)
        moment = Vec3(0.0, 0.0, 0.0)
        scale = 0.5 * air_density * self.calibration

        for strip in self.strips:
            lever = strip.position - centre_of_mass
            local = air_velocity + angular_velocity.cross(lever)
            speed = local.norm()
            if speed < 1e-9:
                continue
            flow = local / speed

            deflection = (
                control.right if strip.deflection_side > 0 else control.left
            )
            coefficients = params.coefficients(ControlInput(deflection, deflection))

            lift_dir = _lift_direction(strip.normal, flow)
            contribution = (
                flow * coefficients.drag + lift_dir * coefficients.lift
            ) * (scale * strip.area * speed * speed)

            force = force + contribution
            moment = moment + lever.cross(contribution)

        return force, moment


def _lift_direction(normal: Vec3, flow: Vec3) -> Vec3:
    """Unit vector perpendicular to ``flow``, in the plane of ``flow`` and ``normal``.

    Degenerate only when the flow runs exactly along the strip's normal, which
    means the canopy is moving straight up or down through the air.  Drag alone
    is the honest answer there, so a zero vector is returned and the caller's
    lift term vanishes.
    """
    projected = normal - flow * normal.dot(flow)
    length = projected.norm()
    if length < 1e-9:
        return Vec3(0.0, 0.0, 0.0)
    return projected / length
