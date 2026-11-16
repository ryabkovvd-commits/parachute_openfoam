"""Scenario definition and loading of the original ``Init.txt`` files.

A :class:`Scenario` is everything needed to run a flight: where the canopy
starts, how fast, what the wind is doing, what it is aiming at, and which
canopy it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .atmosphere import WindProfile
from .canopy import CanopyParams
from .vectors import Vec3

#: The landing point used by the original simulator.
DEFAULT_TARGET = Vec3(-100.0, 0.0, 100.0)

_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(-?[\d.eE+-]+)\s*$")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete, runnable flight setup."""

    position: Vec3 = Vec3(0.0, 1000.0, 0.0)
    velocity: Vec3 = Vec3(20.0, 0.0, 5.0)
    wind: WindProfile = field(default_factory=lambda: WindProfile.uniform())
    target: Vec3 = DEFAULT_TARGET
    canopy: CanopyParams = field(default_factory=CanopyParams)
    air_density: float = 1.23
    gravity: float = 9.81

    @classmethod
    def from_init_file(cls, path: str | Path, **overrides) -> "Scenario":
        """Read an original ``Init.txt`` / ``Init_reserv.txt``.

        The format is a handful of ``NAME = value`` lines for the initial
        position (X, Y, Z) and velocity (VX, VY, VZ), followed by a table of
        ``altitude  wind_east  wind_south`` rows.  Russian comment lines are
        ignored, and the file is decoded as UTF-8 with a fallback to CP1251,
        which is what the 2004-era files actually use.

        Args:
            path: File to read.
            **overrides: Any :class:`Scenario` field to replace afterwards,
                e.g. ``target=Vec3(0, 0, 0)``.
        """
        text = _read_text(Path(path))

        values: dict[str, float] = {}
        wind_rows: list[tuple[float, float, float]] = []

        for line in text.splitlines():
            match = _ASSIGNMENT.match(line)
            if match:
                try:
                    values[match.group(1).upper()] = float(match.group(2))
                except ValueError:
                    pass
                continue

            parts = line.replace(",", ".").split()
            if len(parts) == 3:
                try:
                    wind_rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass  # a header row such as "H  WX  WY"

        scenario = cls(
            position=Vec3(values.get("X", 0.0), values.get("Y", 1000.0), values.get("Z", 0.0)),
            velocity=Vec3(values.get("VX", 20.0), values.get("VY", 0.0), values.get("VZ", 5.0)),
            wind=WindProfile.from_rows(wind_rows) if wind_rows else WindProfile.uniform(),
        )
        return scenario.replace(**overrides) if overrides else scenario

    def replace(self, **changes) -> "Scenario":
        """Return a copy with the given fields replaced."""
        import dataclasses

        return dataclasses.replace(self, **changes)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")
