"""The pressure difference that inflates the canopy.

In Leonov's method the pressure is an **input**, not something the structural
solver works out: the shape is found for a given loading.  Getting that
loading right is a separate problem -- a wind tunnel, a CFD run, or an
iteration between the two -- and keeping the two apart is what makes the
structural side tractable.

The loading is given as a coefficient, ``dCp``, so it is independent of speed:
the actual pressure is ``dCp * q`` with ``q = rho V^2 / 2``.  A uniform
``dCp`` around 1.5 is a reasonable stand-in for a round canopy, and is what
the measured distributions here average to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class PressureField:
    """Pressure difference across the fabric, as a coefficient.

    Attributes:
        coefficient: Callable receiving the panel centroids ``(q, 3)``, their
            unit normals ``(q, 3)`` and the canopy's reference radius, and
            returning ``dCp`` per panel.  Positive means the inside is at
            higher pressure, i.e. inflating.
        dynamic_pressure: ``rho V^2 / 2``, Pa.

    The normals are passed because a loading that ignores them cannot
    represent the thing that matters most about a deformed canopy: a panel
    tilted away from the flow catches less.  Without them a skewed canopy
    keeps generating full force from its collapsed side.
    """

    coefficient: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    dynamic_pressure: float

    def at(
        self, centroids: np.ndarray, normals: np.ndarray, reference_radius: float
    ) -> np.ndarray:
        """Pressure difference on each panel, Pa."""
        return (
            self.coefficient(centroids, normals, reference_radius)
            * self.dynamic_pressure
        )


def uniform_pressure(dcp: float = 1.5, dynamic_pressure: float = 500.0) -> PressureField:
    """The same ``dCp`` everywhere.

    Crude, but it is what the source paper uses for its own examples ("the
    pressure difference was uniformly distributed over the entire canopy
    surface"), and it is the right starting point: any structure in the
    loading should be added only once the uniform case is known to behave.
    """
    def coefficient(centroids, normals, reference_radius) -> np.ndarray:
        return np.full(len(centroids), dcp)

    return PressureField(coefficient=coefficient, dynamic_pressure=dynamic_pressure)


def radial_profile(
    radii: np.ndarray,
    values: np.ndarray,
    dynamic_pressure: float = 500.0,
) -> PressureField:
    """``dCp`` as a function of distance from the axis, interpolated.

    This is the shape a measured or computed distribution usually arrives in:
    a value along the canopy's meridian.  Beyond the tabulated range the
    nearest end is held, which keeps the skirt sensible when the canopy pulls
    in during inflation and panels move outside the original table.
    """
    radii = np.asarray(radii, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if radii.ndim != 1 or radii.shape != values.shape:
        raise ValueError("radii and values must be one-dimensional and match")
    order = np.argsort(radii)
    radii, values = radii[order], values[order]

    def coefficient(centroids, normals, reference_radius) -> np.ndarray:
        r = np.hypot(centroids[:, 0], centroids[:, 1])
        return np.interp(r, radii, values)

    return PressureField(coefficient=coefficient, dynamic_pressure=dynamic_pressure)


def measured_incidence(
    transverse: np.ndarray,
    angles: np.ndarray,
    table: np.ndarray,
    travel_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    *,
    dynamic_pressure: float = 49.8,
) -> PressureField:
    """Loading measured across the canopy at a range of angles of attack.

    This is the established way to give a parachute structural calculation its
    loading, and it is the one that fixes the defect the simpler fields have:
    the pressure is tied to the **flow**, and its response to the canopy
    turning is measured rather than assumed.

    Why the simpler fields fail
    --------------------------

    A loading given as a function of position on the canopy follows the canopy.
    Panel normals are oriented outwards from the confluence point, so a rigid
    rotation about that point carries every force round with it, the resultant
    turns too, and no restoring moment appears.  On an **open** surface such a
    load is non-conservative -- uniform pressure has a potential (``p V``) only
    on a closed one -- and the system is then free to swing away.  Measured on
    a skewed canopy it did exactly that: 142 degrees of tilt per metre of line
    stroke, the same at every stroke, mesh and damping, where geometry allows
    about 5.

    What makes this field different is that the distribution is asymmetric
    across the canopy once it is tilted, so the resultant no longer lies along
    the canopy axis and the moment is real.

    Args:
        transverse: Stations across the canopy section, m, running from one
            skirt edge through the centre to the other.  Signed.
        angles: Angles of attack the table was measured at, degrees.
        table: ``dCp`` with shape ``(len(angles), len(transverse))``.
        travel_direction: Which way the canopy is going, not the air.

    The section is extended to three dimensions the usual way: a panel is
    given the loading of the station at its own distance along the
    windward-leeward axis.  That makes the load a function of one coordinate
    rather than two, which is an approximation -- but it is the asymmetry
    along that axis that carries the moment, and that part is measured.

    Which end of the table is windward is worked out from the table itself,
    by seeing which half carries more at the largest angle, rather than being
    assumed.  Guessing it would reverse the moment and turn a canopy that
    rights itself into one that tips over faster, with nothing to show that
    anything was wrong.
    """
    transverse = np.asarray(transverse, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    if transverse.ndim != 1 or angles.ndim != 1:
        raise ValueError("transverse and angles must be one-dimensional")
    if table.shape != (angles.size, transverse.size):
        raise ValueError(
            f"table must be {(angles.size, transverse.size)}, got {table.shape}"
        )

    direction = np.asarray(travel_direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("travel direction cannot be a zero vector")
    onset = -direction / norm            # where the air comes from

    # Which end of the table is the windward one, decided by the data: at the
    # largest angle the windward half carries the most.
    #
    # The sign is the sign of the STATION that windward panels are sent to.
    # If the table's heavy end is at negative stations, a windward panel must
    # be given a negative station -- hence the minus.  Getting this backwards
    # loads the sheltered flank instead of the exposed one, which turns the
    # restoring moment into an overturning one; the canopy then tips over
    # faster than under a load that ignores the flow entirely, and nothing in
    # the run says so.
    steepest = table[np.argmax(angles)]
    half = transverse.size // 2
    windward_sign = -1.0 if steepest[:half].mean() > steepest[half:].mean() else 1.0
    span = np.abs(transverse).max()

    def coefficient(centroids, normals, reference_radius) -> np.ndarray:
        centre = centroids.mean(axis=0)
        axis = normals.mean(axis=0)
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-12 else onset

        alpha = np.degrees(np.arccos(np.clip(axis @ onset, -1.0, 1.0)))

        # A panel's station is its DISTANCE from the axis, signed by which
        # side of the canopy it is on.
        #
        # Distance and sign have to be taken separately.  Projecting straight
        # onto the windward direction conflates them, and then the symmetric
        # case -- where that direction is undefined, the flow lying along the
        # axis -- collapses every station to zero and hands the whole canopy
        # the value at the centre of the table.  The measured radial variation
        # vanishes and the load silently goes uniform.  Nothing shows it,
        # because the shape barely depends on the profile: the symmetric run
        # still lands on Dp/D0 = 0.636, right for the wrong reason.
        offset = centroids - centre
        radial = offset - np.outer(offset @ axis, axis)
        distance = np.linalg.norm(radial, axis=1)

        across = onset - (onset @ axis) * axis
        length = np.linalg.norm(across)
        if length < 1e-9:
            # Flow along the axis: no windward side, and the table is
            # symmetric there, so either sign gives the same answer.
            sign = np.ones(len(centroids))
        else:
            sign = np.sign(radial @ (across / length)) * windward_sign
            sign[sign == 0.0] = 1.0

        reach = distance.max()
        station = distance * sign * (span / reach if reach > 1e-9 else 1.0)

        # Bilinear: interpolate across the section at each bracketing angle,
        # then between them.
        k = np.clip(np.searchsorted(angles, alpha), 1, angles.size - 1)
        lo, hi = angles[k - 1], angles[k]
        weight = 0.0 if hi <= lo else np.clip((alpha - lo) / (hi - lo), 0.0, 1.0)
        below = np.interp(station, transverse, table[k - 1])
        above = np.interp(station, transverse, table[k])
        return below + weight * (above - below)

    return PressureField(coefficient=coefficient, dynamic_pressure=dynamic_pressure)



def local_incidence(
    peak_dcp: float,
    travel_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    *,
    dynamic_pressure: float = 49.8,
    base_fraction: float = 0.35,
) -> PressureField:
    """Loading that follows how each panel is turned to the flow.

    A stand-in for a flow solver, and labelled as such.  Each panel gets

        dCp = peak * (cos t)^2 + base       for cos t > 0

    where ``t`` is the angle between the panel's outward normal and the
    direction the air arrives from, and ``base`` is the suction on the leeward
    side, which for a bluff body persists whichever way the surface faces.

    This is Newtonian impact theory with a base term.  It is crude, and for an
    attached flow it would be wrong.  For a parachute -- a bluff body whose
    drag is essentially the projected area times a constant -- it captures the
    one thing that matters here: **a panel tilted away from the flow catches
    less**, and a canopy skewed by a pulled riser therefore loses force on the
    collapsed side instead of keeping it.

    Args:
        travel_direction: Which way the **canopy** is going, not which way the
            air is going -- the two are opposite, and getting them the wrong
            way round is silent.  A canopy descends, so the default points
            down; the air then arrives from below, and the crown, whose
            outward normal points up, is the most loaded panel.

            Named for the canopy rather than the flow deliberately.  Called
            ``flow_direction`` it invites the reading "the air comes up at the
            canopy, so ``(0, 0, 1)``" -- which would leave every panel with
            nothing but the base term, collapse the canopy, and raise no error
            at all.

    Use it to close the loop and see the mechanism.  Replace it with a real
    solver's output, through :mod:`parafoil.inflation.exchange`, before
    believing a number.
    """
    direction = np.asarray(travel_direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("travel direction cannot be a zero vector")
    direction = direction / norm
    if peak_dcp <= 0.0:
        raise ValueError("peak dCp must be positive")

    def coefficient(centroids, normals, reference_radius) -> np.ndarray:
        # Positive where the panel meets the oncoming air, which arrives from
        # the direction the canopy is travelling towards.
        facing = -np.einsum("ij,j->i", normals, direction)
        windward = np.clip(facing, 0.0, 1.0)
        return peak_dcp * (windward**2 + base_fraction)

    return PressureField(coefficient=coefficient, dynamic_pressure=dynamic_pressure)
