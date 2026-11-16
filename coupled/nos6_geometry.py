"""Closed-pole fabric geometry for the 6 m² Nosarev configuration."""

from dataclasses import replace

import numpy as np

from parafoil.inflation import Fabric, Webbing, flat_circular
from parafoil.inflation.exchange import _smooth_grid, export_surface


def closed_circular(diameter, length, *, gores, rings, line_stride, shear_ratio):
    """Add a structural fan at the pole, retaining the existing node indices."""
    fabric = Fabric.ripstop_type_i()
    band = Webbing.radial_band()
    mesh = flat_circular(
        diameter, length, gores=gores, rings=rings, line_stride=line_stride,
        shear_ratio=shear_ratio, vent_fraction=1 / (rings + 1),
        fabric=fabric, radial_band=band,
    )
    pole = len(mesh.position)
    first = np.arange(gores)
    following = np.roll(first, -1)
    pole_nodes = np.full(gores, pole)
    fan = np.column_stack((pole_nodes, pole_nodes, following, first))
    # Match the radial strip discretization used on the other rings.
    radius = diameter / (2 * (rings + 1))
    strip = fabric.strip(np.pi * radius / gores)
    seam = first % line_stride == 0
    rods = np.column_stack((pole_nodes, first))
    position = np.vstack((mesh.position, mesh.position[first].mean(axis=0)))
    flat = np.vstack((mesh.flat, np.zeros(3)))
    triangle_area = np.linalg.norm(np.cross(flat[first], flat[following]), axis=1) / 2
    panel_mass = triangle_area * fabric.areal_density
    mass = np.append(mesh.mass, panel_mass.sum() / 2)
    mass[first] += (panel_mass + np.roll(panel_mass, 1)) / 4
    member_mass = radius * (0.1 * fabric.areal_density + seam * band.linear_density)
    mass[first] += member_mass / 2
    mass[pole] += member_mass.sum() / 2
    return replace(
        mesh, flat=flat, position=position, mass=mass,
        rods=np.vstack((mesh.rods, rods)),
        rest_length=np.append(mesh.rest_length, np.full(gores, radius)),
        stiffness=np.append(mesh.stiffness, strip.stiffness + seam * band.stiffness),
        breaking_load=np.append(mesh.breaking_load, strip.breaking_load + seam * band.breaking_load),
        kind=np.append(mesh.kind, np.full(gores, "radial")),
        quads=np.vstack((mesh.quads, fan)),
    )


def export_closed_surface(mesh, path, *, subdivide=12):
    """Mesh the annulus with the standard exporter and close it with the movable pole."""
    if subdivide < 2:
        raise ValueError("Closed canopy export requires subdivide >= 2")
    stl = export_surface(mesh, path, subdivide=subdivide)
    crown = _smooth_grid(mesh, subdivide)[0]
    pole = mesh.position[-1]
    reference = mesh.position[mesh.confluence[0]]
    facets = []
    for first, second in zip(crown, np.roll(crown, -1, axis=0)):
        normal = np.cross(first - pole, second - pole)
        if normal @ (pole - reference) < 0:
            first, second, normal = second, first, -normal
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length == 0:
            raise ValueError("Degenerate STL triangle at the pole")
        normal /= length
        facets.append("  facet normal %.6e %.6e %.6e\n    outer loop" % tuple(normal))
        for vertex in (pole, first, second):
            facets.append("      vertex %.6e %.6e %.6e" % tuple(vertex))
        facets.append("    endloop\n  endfacet")
    with stl.open("rb+") as stream:
        end = b"endsolid canopy"
        stream.seek(-len(end), 2)
        assert stream.read() == end
        stream.seek(-len(end), 2)
        stream.write(("\n".join(facets) + "\n" + end.decode("ascii")).encode("ascii"))
    return stl
