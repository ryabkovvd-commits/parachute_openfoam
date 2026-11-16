"""Wind field.

The original program stored a fixed table of 101 rows (altitude 0..10000 m in
100 m steps) and linearly interpolated between them.  This module keeps the
same physics but drops the fixed size: any monotonically increasing table
works, and lookups are a binary search instead of a linear scan.

Wind is purely horizontal -- the model has no updrafts or thermals.  That is a
property of the original model, not an oversight; see :class:`WindProfile` if
you want to add one.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Sequence

from .vectors import Vec3


@dataclass(frozen=True, slots=True)
class WindProfile:
    """Horizontal wind as a function of altitude, linearly interpolated.

    Args:
        altitudes: Altitudes in metres, strictly increasing.
        east: Wind component towards +X (east), m/s, one per altitude.
        south: Wind component towards +Z (south), m/s, one per altitude.

    Outside the tabulated range the nearest end value is held constant, which
    matches the original behaviour and avoids nonsense extrapolation.
    """

    altitudes: tuple[float, ...]
    east: tuple[float, ...]
    south: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.altitudes)
        if n == 0:
            raise ValueError("wind profile needs at least one row")
        if len(self.east) != n or len(self.south) != n:
            raise ValueError("wind component arrays must match the altitude array")
        if any(b <= a for a, b in zip(self.altitudes, self.altitudes[1:])):
            raise ValueError("altitudes must be strictly increasing")

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[float, float, float]]) -> "WindProfile":
        """Build from ``(altitude, east, south)`` triples."""
        ordered = sorted(rows, key=lambda r: r[0])
        return cls(
            altitudes=tuple(r[0] for r in ordered),
            east=tuple(r[1] for r in ordered),
            south=tuple(r[2] for r in ordered),
        )

    @classmethod
    def uniform(cls, east: float = 0.0, south: float = 0.0) -> "WindProfile":
        """A single wind vector at every altitude -- handy for tests."""
        return cls(altitudes=(0.0,), east=(east,), south=(south,))

    def at(self, altitude: float) -> Vec3:
        """Wind velocity vector at ``altitude`` metres."""
        alts = self.altitudes
        if altitude <= alts[0]:
            return Vec3(self.east[0], 0.0, self.south[0])
        if altitude >= alts[-1]:
            return Vec3(self.east[-1], 0.0, self.south[-1])

        i = bisect.bisect_right(alts, altitude)
        lo, hi = alts[i - 1], alts[i]
        t = (altitude - lo) / (hi - lo)
        return Vec3(
            self.east[i - 1] + t * (self.east[i] - self.east[i - 1]),
            0.0,
            self.south[i - 1] + t * (self.south[i] - self.south[i - 1]),
        )

    def speed_at(self, altitude: float) -> float:
        return self.at(altitude).norm()
