"""Обмен между расчётом формы и расчётом обтекания.

Leonov's system keeps the elastic and the aerodynamic sides as two separate
modules and passes data between them: the converged shape goes over "as a
rigid, undeformable structure", the aerodynamic module returns the pressure
distribution on it, and the shape is recomputed.  That exchange is the whole
mechanism by which the two are made consistent, and this module implements
its two halves.

Why an interface rather than a solver
-------------------------------------

Leonov's aerodynamic module uses the method of discrete closed vortex frames.
For a parachute that method needs a **free vortex wake** shed from the skirt:
a canopy is a bluff body, all of its drag comes from the separated region
behind it, and potential flow without a wake gives exactly zero drag.
Building that -- sheet shedding, roll-up, core regularisation -- is a serious
piece of work in its own right.

Meanwhile a CFD run on a given shape is something that can already be done.
So what is provided here is the exchange, not the aerodynamics: export a shape
in a form a flow solver will accept, read a pressure field back, and iterate.
The aerodynamic module is whatever is to hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh import CanopyMesh, panel_area_and_normal
from .pressure import PressureField


def _catmull_rom(control: np.ndarray, steps: int, periodic: bool) -> np.ndarray:
    """Interpolate along the first axis, passing through every control point.

    Catmull-Rom rather than a plain average: it is local, needs no solve, and
    goes through the given points rather than near them, so the subdivided
    surface still carries the shape the structural solver computed.
    """
    n = len(control)
    if periodic:
        take = lambda k: control[k % n]                       # noqa: E731
        segments = n
    else:
        take = lambda k: control[min(max(k, 0), n - 1)]       # noqa: E731
        segments = n - 1

    out = []
    for i in range(segments):
        p0, p1, p2, p3 = take(i - 1), take(i), take(i + 1), take(i + 2)
        for s in range(steps):
            t = s / steps
            out.append(
                0.5 * (
                    2.0 * p1
                    + (p2 - p0) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
                )
            )
    if not periodic:
        out.append(control[-1])
    return np.asarray(out)


def _smooth_grid(mesh: CanopyMesh, subdivide: int) -> np.ndarray:
    """The canopy's nodes, refined ``subdivide`` times in each direction."""
    rings, gores = mesh.rings, mesh.gores
    grid = mesh.position[: (rings + 1) * gores].reshape(rings + 1, gores, 3)

    # Round the canopy first -- periodic -- then along it, where both ends
    # are free edges: the vent hole and the skirt.
    around = np.stack(
        [_catmull_rom(grid[i], subdivide, periodic=True) for i in range(rings + 1)]
    )
    return np.stack(
        [_catmull_rom(around[:, j], subdivide, periodic=False)
         for j in range(around.shape[1])],
        axis=1,
    )


def export_surface(
    mesh: CanopyMesh, path: str | Path, *, subdivide: int = 1
) -> Path:
    """Write the current shape as an ASCII STL, for a flow solver to read.

    Each fabric panel becomes two triangles.  Normals point away from the
    confluence point, i.e. outwards, which is the convention every mesher
    expects.

    Args:
        subdivide: Split each panel this many times in each direction,
            interpolating smoothly between the computed nodes.

    Why smoothing matters, and why splitting flat would not
    -------------------------------------------------------

    The exported surface is not only geometry -- it is the boundary a mesher
    snaps its cells to, and it therefore sets a resolution of its own.  A
    canopy of 16 by 8 panels has facets about 1.4 m across; ask for 0.5 m
    cells and the mesher faithfully resolves **the facet edges**, returning a
    ragged surface that is worse, not better, than the one the coarser cells
    gave.  The geometry, not the flow mesh, is what binds.

    Splitting each facet into four would not help: the polyhedron and its
    creases are unchanged, and the mesher still finds them.  What is needed is
    a surface with no creases, which is what the interpolation gives -- and it
    is also the more honest shape, since fabric under tension runs smoothly
    between its attachments rather than in flats.

    The cost is a one-off geometric operation per shape.  Nothing in the
    structural calculation is redone, and no information is invented beyond
    the assumption that the canopy is smooth between computed nodes.
    """
    path = Path(path)
    reference = mesh.position[mesh.confluence[0]]

    if subdivide > 1:
        grid = _smooth_grid(mesh, subdivide)
        rows, columns = grid.shape[0], grid.shape[1]
        flat = grid.reshape(-1, 3)
        quads = np.array(
            [
                [i * columns + j,
                 i * columns + (j + 1) % columns,
                 (i + 1) * columns + (j + 1) % columns,
                 (i + 1) * columns + j]
                for i in range(rows - 1)
                for j in range(columns)
            ]
        )
        corners = flat[quads]
        _, normals = panel_area_and_normal(flat, quads, reference)
    else:
        corners = mesh.position[mesh.quads]
        _, normals = panel_area_and_normal(mesh.position, mesh.quads, reference)

    lines = ["solid canopy"]
    for quad, normal in zip(corners, normals):
        # Wound to agree with the stated normal.
        #
        # The panel corners run the other way round from the outward normal
        # ``panel_area_and_normal`` gives, so taking them in order produces a
        # facet whose right-hand rule points *into* the canopy while the line
        # above it says the opposite.  Every facet in the file disagreed with
        # its own winding, which a reader is free to resolve either way.  It
        # costs nothing while the canopy is a plain baffle and the mesher only
        # snaps to the triangles, but a porous baffle takes the direction of
        # its pressure jump from the surface orientation, and there a silent
        # flip loads the canopy backwards.
        for triangle in ((quad[0], quad[2], quad[1]), (quad[0], quad[3], quad[2])):
            lines.append(
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}"
            )
            lines.append("    outer loop")
            for vertex in triangle:
                lines.append(
                    f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}"
                )
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append("endsolid canopy")

    path.write_text("\n".join(lines), encoding="ascii")
    return path


def export_panels(mesh: CanopyMesh, path: str | Path) -> Path:
    """Write panel centroids, normals and areas as CSV.

    Often more convenient than a surface mesh: a flow solver can be asked for
    the pressure at these exact points, and the result comes back needing no
    interpolation at all.
    """
    path = Path(path)
    reference = mesh.position[mesh.confluence[0]]
    area, normal = panel_area_and_normal(mesh.position, mesh.quads, reference)
    centroid = mesh.position[mesh.quads].mean(axis=1)

    rows = ["panel,x,y,z,nx,ny,nz,area"]
    for i, (c, n, a) in enumerate(zip(centroid, normal, area)):
        rows.append(
            f"{i},{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
            f"{n[0]:.6f},{n[1]:.6f},{n[2]:.6f},{a:.6f}"
        )
    path.write_text("\n".join(rows), encoding="ascii")
    return path


def pressure_from_points(
    points: np.ndarray,
    values: np.ndarray,
    dynamic_pressure: float,
    *,
    neighbours: int = 4,
    restrict: bool = True,
) -> PressureField:
    """Build a pressure field from a scattered set of ``(x, y, z) -> dCp``.

    This is the return leg: a flow solver gives the pressure difference at
    points on the surface, generally not the ones the structural mesh uses, so
    the values are carried across by inverse-distance weighting over the
    nearest few.

    Args:
        points: ``(k, 3)`` positions where the coefficient is known.
        values: ``(k,)`` pressure difference coefficients, inside minus
            outside.
        neighbours: How many nearest points to average.  Four smooths over
            solver noise without flattening real structure; one gives nearest
            neighbour.

    Cost, and why the weights are held on to
    ----------------------------------------

    The obvious implementation redoes the search on every call, and inside a
    relaxation that is ruinous: the solver asks for the pressure once per time
    step, so a few hundred panels against a few thousand samples becomes
    hundreds of thousands of distances, several hundred thousand times over.
    Measured on a coupled pass, it was **the flow solver that finished
    quickly** and this that did not -- an hour of arithmetic carrying a field
    that had already been computed.

    So the neighbours and their weights are worked out once, on the first
    call, and reused.  What that assumes is that panels keep the same
    neighbours as the shape settles.  Inside one coupled pass they do: the
    shape moves by centimetres against a sample spacing of tens of them, and
    the pass exists precisely because the shape is nearly right already.  Pass
    ``restrict=False`` to search afresh every time, at full cost.

    The assumption is checked rather than trusted: the neighbours are thrown
    away and found again as soon as the panels have moved an appreciable
    fraction of the spacing between samples.  That check has to be on how far
    the **panels** have travelled since the search, not on how far they now
    sit from their samples -- a relaxation starting from the flat cutting
    pattern begins with every panel metres away from the canopy surface, so a
    threshold set from that never trips, and the field returns the neighbours
    of the flat pattern for the whole run.  Measured: 0.646 against 0.636 for
    the same canopy, a 1.6 % error bought for a 20-fold speed-up.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (k, 3) array")
    if values.shape != (len(points),):
        raise ValueError("one value is needed per point")
    if neighbours < 1:
        raise ValueError("neighbours must be at least one")

    k = min(neighbours, len(points))

    # How far apart the samples sit, as the median distance to the nearest
    # other one.  A panel may drift by a fraction of that before its
    # neighbours are worth finding again.
    spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(spread, np.inf)
    # A tenth, not a quarter: at a quarter the shape came out 0.14 % different
    # from a fresh search every step.  Small, but it is error bought with
    # nothing but speed, and the early part of a relaxation re-searches every
    # time anyway -- the saving comes from the settled part, where the panels
    # have stopped moving and the threshold never binds.
    tolerance = 0.1 * float(np.median(spread.min(axis=1)))

    held: dict[str, np.ndarray] = {}

    def search(centroids):
        # Distances from every panel to every sample.  The meshes here are
        # small -- hundreds of panels against thousands of samples -- so the
        # dense form is quicker than building a tree.
        delta = centroids[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
        rows = np.arange(len(centroids))[:, None]
        return nearest, distance[rows, nearest]

    def coefficient(centroids, normals, reference_radius) -> np.ndarray:
        reuse = (
            restrict
            and "at" in held
            and len(centroids) == len(held["at"])
            and np.linalg.norm(centroids - held["at"], axis=1).max() <= tolerance
        )
        if reuse:
            nearest = held["nearest"]
            rows = np.arange(len(centroids))[:, None]
            picked = np.linalg.norm(centroids[rows] - points[nearest], axis=2)
        else:
            nearest, picked = search(centroids)
            if restrict:
                held["nearest"] = nearest
                held["at"] = centroids.copy()

        weight = 1.0 / np.maximum(picked, 1e-9)
        return (weight * values[nearest]).sum(axis=1) / weight.sum(axis=1)

    return PressureField(coefficient=coefficient, dynamic_pressure=dynamic_pressure)


def read_panel_pressures(path: str | Path, dynamic_pressure: float) -> PressureField:
    """Read back a CSV of ``x,y,z,dcp`` -- the counterpart of :func:`export_panels`.

    A header line is skipped if present.  Columns beyond the fourth are
    ignored, so a solver's own export can usually be fed in unedited.
    """
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace(";", ",").split(",")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(p) for p in parts[:4]])
        except ValueError:
            continue          # header or comment
    if not rows:
        raise ValueError(f"no numeric rows of x,y,z,dcp found in {path}")

    table = np.asarray(rows, dtype=np.float64)
    return pressure_from_points(table[:, :3], table[:, 3], dynamic_pressure)
