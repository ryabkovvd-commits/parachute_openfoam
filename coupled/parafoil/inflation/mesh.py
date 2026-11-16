"""Discretisation of a canopy into nodal masses and tension-only rods.

The grid follows the construction of a real canopy rather than a convenient
mathematical one: radial lines along the seams between gores, annular lines
along the reinforcing bands, and suspension lines from the skirt to the
confluence point.  That way the reinforcing framework and the fabric can be
given different stiffnesses, which is the point of the two superimposed grids
in Leonov's method.

Positions are held as one flat ``(n, 3)`` array and the rods as index pairs,
so a solver step is a handful of array operations rather than a Python loop
over elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .materials import Fabric, RodMaterial, Webbing


@dataclass(slots=True)
class CanopyMesh:
    """A canopy ready to be inflated.

    Attributes:
        flat: ``(n, 3)`` node positions of the **cut pattern**, laid out flat.
            The rod rest lengths come from here, so this is what encodes the
            canopy's construction.
        position: ``(n, 3)`` current node positions, the thing that gets
            solved for.
        mass: ``(n,)`` nodal masses, kg.
        rods: ``(m, 2)`` node index pairs.
        rest_length: ``(m,)`` unstretched rod lengths, from :attr:`flat`.
        stiffness: ``(m,)`` rod stiffnesses, N per unit relative elongation.
        breaking_load: ``(m,)`` N, for safety factors.
        kind: ``(m,)`` labels -- ``radial``, ``annular``, ``line``.
        quads: ``(q, 4)`` node indices of the fabric panels, used to work out
            the area each node carries and the local normal.
        skirt: node indices of the outer edge.
        vent: node indices of the apex vent edge.
        confluence: where the lines converge; held fixed while the shape is
            solved, so the reaction there is the load the canopy carries.
        rings, gores: grid size, kept for reshaping results.
    """

    flat: np.ndarray
    position: np.ndarray
    mass: np.ndarray
    rods: np.ndarray
    rest_length: np.ndarray
    stiffness: np.ndarray
    breaking_load: np.ndarray
    kind: np.ndarray
    quads: np.ndarray
    skirt: np.ndarray
    vent: np.ndarray
    confluence: np.ndarray
    rings: int
    gores: int
    nominal_diameter: float = 0.0
    line_length: float = 0.0
    #: Index into :attr:`rods` of the suspension lines, cached for reporting.
    line_rods: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def node_count(self) -> int:
        return len(self.position)

    def projected_diameter(self) -> float:
        """Diameter of the skirt as seen from below, m.

        The number to compare against the canopy's nominal diameter: for a
        flat-circular parachute the inflated projected diameter comes out at
        roughly two thirds of the nominal, and that ratio is the main check
        on whether a shape calculation is behaving.
        """
        skirt = self.position[self.skirt]
        centre = skirt.mean(axis=0)
        radii = np.hypot(skirt[:, 0] - centre[0], skirt[:, 1] - centre[1])
        return 2.0 * float(radii.mean())

    def projected_area(self) -> float:
        return np.pi * (self.projected_diameter() / 2.0) ** 2

    def nominal_area(self) -> float:
        return np.pi * (self.nominal_diameter / 2.0) ** 2


def flat_circular(
    nominal_diameter: float,
    line_length: float,
    *,
    gores: int = 24,
    rings: int = 12,
    line_stride: int = 1,
    shear_ratio: float | None = None,
    vent_fraction: float = 0.05,
    fabric: Fabric | None = None,
    radial_band: Webbing | None = None,
    line: Webbing | None = None,
) -> CanopyMesh:
    """Build a flat-circular canopy: a disc with a vent at the apex.

    Args:
        nominal_diameter: Diameter of the flat cut pattern, m.  This is the
            ``D0`` parachute drag data is referred to.
        line_length: Suspension line length from skirt to confluence, m.
        gores: Number of nodes around the circumference.
        rings: Radial subdivisions between vent and skirt.
        shear_ratio: Overrides the fabric's own, so the influence of the panel
            diagonals can be measured rather than assumed.  Zero leaves them
            out entirely, which is the behaviour this mesh had before they
            existed.
        line_stride: Attach a suspension line every this many circumferential
            nodes, so the line count is ``gores // line_stride``.  Leave it at
            one and every node carries a line, which makes the fabric between
            lines a single quad wide and therefore unable to bulge.

            That bulging is not a detail.  Nosarev (TsAGI 1735, eq. 5) splits
            the projected area into the polygon through the attachment points
            plus the circular segments the fabric forms between them, and for
            sixteen lines those segments are 21% of it.  Measured on a mesh
            with ``line_stride=1``, the skirt radius varies by 0.05% around the
            circumference: the segments are simply absent, and the projected
            area comes out 20% low.  Give a gore a few nodes across and the
            fabric can take its arc.
        vent_fraction: Vent diameter as a fraction of the nominal.
        fabric, radial_band, line: Materials; sensible defaults if omitted.

    The canopy starts flat and horizontal, positioned so the lines are just
    taut.  Pressure then does the inflating, which is both the physically
    honest starting point and a decent test: if the solver cannot get from a
    flat disc to a dome, it is not going to manage anything harder.
    """
    if nominal_diameter <= 0.0:
        raise ValueError("nominal diameter must be positive")
    if not 0.0 < vent_fraction < 1.0:
        raise ValueError("vent fraction must lie in (0, 1)")
    if line_stride < 1 or gores % line_stride:
        raise ValueError(
            f"line stride {line_stride} must divide {gores} gores evenly"
        )

    fabric = fabric or Fabric.ripstop_type_i()
    if shear_ratio is not None:
        fabric = replace(fabric, shear_ratio=shear_ratio)
    radial_band = radial_band or Webbing.radial_band()
    line = line or Webbing.suspension_cord()

    outer = nominal_diameter / 2.0
    inner = outer * vent_fraction
    if line_length <= outer:
        raise ValueError(
            f"lines of {line_length} m cannot reach a skirt at {outer} m radius"
        )

    radii = np.linspace(inner, outer, rings + 1)
    angles = np.arange(gores) * (2.0 * np.pi / gores)

    # Flat pattern, and the starting three-dimensional guess: the same disc
    # lifted to where the lines are exactly taut.
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    flat = np.stack([rr * np.cos(aa), rr * np.sin(aa), np.zeros_like(rr)], axis=-1)
    flat = flat.reshape(-1, 3)

    height = float(np.sqrt(line_length**2 - outer**2))
    position = flat.copy()
    position[:, 2] = height

    def node(i: int, j: int) -> int:
        return i * gores + (j % gores)

    rods: list[tuple[int, int]] = []
    rest: list[float] = []
    stiff: list[float] = []
    strength: list[float] = []
    kinds: list[str] = []

    # Radial rods: fabric strip plus its share of a radial reinforcing band.
    # Each gore seam carries one band, so the band is split between the two
    # radial rod lines that meet along it.  Only seams get a band: with a
    # stride the intermediate radials run through the middle of a gore, where
    # there is nothing but fabric, and banding them would stiffen the very
    # arcs we added those nodes to allow.
    band = radial_band.rod()
    for i in range(rings):
        width = 2.0 * np.pi * 0.5 * (radii[i] + radii[i + 1]) / gores
        strip = fabric.strip(width)
        for j in range(gores):
            seam = j % line_stride == 0
            rods.append((node(i, j), node(i + 1, j)))
            rest.append(float(radii[i + 1] - radii[i]))
            stiff.append(strip.stiffness + (band.stiffness if seam else 0.0))
            strength.append(strip.breaking_load
                            + (band.breaking_load if seam else 0.0))
            kinds.append("radial")

    # Annular rods: fabric only, unless a real design puts bands here.
    #
    # The width a rod stands for is the mean of the two perpendicular rods
    # either side of it -- Leonov's equation (2.1), the tributary width.  At
    # the vent and at the skirt there is only one, so those rings get half
    # the width; treating them like interior rings makes the edges twice as
    # stiff as they should be.
    for i in range(rings + 1):
        span = 2.0 * np.pi * radii[i] / gores
        if rings == 0:
            width = 1.0
        elif i == 0 or i == rings:
            width = 0.5 * (radii[1] - radii[0])
        else:
            width = radii[1] - radii[0]
        strip = fabric.strip(width)
        for j in range(gores):
            rods.append((node(i, j), node(i, j + 1)))
            rest.append(float(span))
            stiff.append(strip.stiffness)
            strength.append(strip.breaking_load)
            kinds.append("annular")

    # Panel diagonals -- shear, and nothing else.
    #
    # Without them a panel is held by its radial and annular rods alone, so it
    # skews with no resistance at all, and since the rods are tension-only it
    # can pull along those two directions and no other.  Measured on tilted
    # canopies, the largest strain runs along a diagonal in 27% to 43% of
    # panels: load in a direction the mesh has no member for.  At zero
    # incidence the figure is 0%, as axial symmetry demands -- which is why
    # this went unnoticed while only symmetric cases were run.
    #
    # Both diagonals, each tension-only: a wrinkled panel then carries along
    # whichever one happens to be taut, which is what a membrane in a tension
    # field actually does.  One diagonal alone would make the panel stiffer one
    # way than the other for no physical reason.
    if fabric.shear_ratio > 0.0:
        for i in range(rings):
            width = 2.0 * np.pi * 0.5 * (radii[i] + radii[i + 1]) / gores
            cross = fabric.diagonal(width)
            for j in range(gores):
                a, b = node(i, j), node(i, j + 1)
                c, d = node(i + 1, j + 1), node(i + 1, j)
                for start, end in ((a, c), (b, d)):
                    rods.append((start, end))
                    rest.append(float(np.linalg.norm(flat[end] - flat[start])))
                    stiff.append(cross.stiffness)
                    strength.append(cross.breaking_load)
                    kinds.append("diagonal")

    # Suspension lines from every skirt node to the confluence point, which is
    # the last node in the array and is held fixed.
    confluence_index = (rings + 1) * gores
    position = np.vstack([position, np.zeros((1, 3))])
    flat = np.vstack([flat, np.zeros((1, 3))])
    line_rod = line.rod()
    first_line = len(rods)
    for j in range(0, gores, line_stride):
        rods.append((node(rings, j), confluence_index))
        rest.append(float(line_length))
        stiff.append(line_rod.stiffness)
        strength.append(line_rod.breaking_load)
        kinds.append("line")

    rods_a = np.asarray(rods, dtype=np.int64)
    rest_a = np.asarray(rest, dtype=np.float64)

    # Fabric panels, for pressure area and local normals.
    quads = []
    for i in range(rings):
        for j in range(gores):
            quads.append([node(i, j), node(i, j + 1),
                          node(i + 1, j + 1), node(i + 1, j)])
    quads_a = np.asarray(quads, dtype=np.int64)

    # Nodal mass: the fabric each node carries, plus half of every rod at it.
    mass = np.zeros(len(position))
    corners = flat[quads_a]
    panel_area = _quad_areas(corners)
    for c in range(4):
        np.add.at(mass, quads_a[:, c], fabric.areal_density * panel_area / 4.0)
    # Diagonals weigh nothing: the cloth's mass is already on the nodes by
    # panel area, and charging it again through the diagonals would make the
    # canopy heavier than the cloth it is cut from.
    densities = np.array(
        [0.0 if k == "diagonal"
         else line_rod.linear_density if k == "line"
         else fabric.areal_density * 0.1
         for k in kinds]
    )
    rod_mass = densities * rest_a
    np.add.at(mass, rods_a[:, 0], rod_mass / 2.0)
    np.add.at(mass, rods_a[:, 1], rod_mass / 2.0)
    mass[confluence_index] = np.inf   # fixed point

    return CanopyMesh(
        flat=flat,
        position=position,
        mass=mass,
        rods=rods_a,
        rest_length=rest_a,
        stiffness=np.asarray(stiff, dtype=np.float64),
        breaking_load=np.asarray(strength, dtype=np.float64),
        kind=np.asarray(kinds),
        quads=quads_a,
        skirt=np.array([node(rings, j) for j in range(gores)]),
        vent=np.array([node(0, j) for j in range(gores)]),
        confluence=np.array([confluence_index]),
        rings=rings,
        gores=gores,
        nominal_diameter=nominal_diameter,
        line_length=line_length,
        line_rods=np.arange(first_line, len(rods)),
    )


def _quad_areas(corners: np.ndarray) -> np.ndarray:
    """Area of each quad, as the sum of its two triangles."""
    a, b, c, d = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    t2 = 0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=1)
    return t1 + t2


def panel_area_and_normal(
    position: np.ndarray, quads: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Area and unit normal of each fabric panel in its current shape.

    Args:
        reference: A point inside the canopy -- in practice the confluence of
            the lines.  Normals are oriented to point **away from it**.

    That reference matters more than it looks.  Orienting normals by "make the
    z component positive" works while the canopy is flat and fails as soon as
    it domes: near the skirt of an inflated canopy the surface is close to
    vertical, the sign of the z component becomes arbitrary, and the pressure
    starts pushing neighbouring panels in opposite directions.  The canopy
    then collapses inwards instead of filling.  Orienting away from a point
    known to be on the inside has no such ambiguity.
    """
    corners = position[quads]
    a, b, c, d = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    normal = np.cross(c - a, d - b)          # diagonals: robust for warped quads
    length = np.linalg.norm(normal, axis=1)
    area = 0.5 * length
    unit = np.divide(
        normal, length[:, None], out=np.zeros_like(normal), where=length[:, None] > 0
    )
    outward = corners.mean(axis=1) - reference
    flip = np.einsum("ij,ij->i", unit, outward) < 0.0
    unit[flip] *= -1.0
    return area, unit
