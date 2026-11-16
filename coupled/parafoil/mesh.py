"""Canopy geometry, loaded from the original data files.

The original program shipped the canopy as a finite-element style mesh in three
kinds of plain-text file:

* ``points_<d>.txt`` -- 2730 vertices, one file for each brake deflection
  ``d`` in 0.0, 0.1, ... 1.0.  Columns are ``index, along-course, up, spanwise``
  in a canopy-local frame.
* ``lines.txt``      -- 2020 two-point elements (the rigging lines), as
  ``index, p1, p2, r, g, b``.
* ``poligons.txt``   -- 2992 four-point elements (the fabric panels), as
  ``index, p1, p2, p3, p4, id, r, g, b``.

Point indices in the element files are 1-based; they are converted to 0-based
on load.  Deforming the canopy for an arbitrary toggle setting is a linear
blend between the two nearest deflection states -- which is exactly the kind of
bulk array work NumPy is for, so this module uses it throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .canopy import ControlInput

#: Deflection values for which the original supplies a ``points_*.txt`` file.
DEFLECTION_STATES = tuple(round(i / 10, 1) for i in range(11))


@dataclass(frozen=True, slots=True)
class CanopyMesh:
    """Canopy surface and rigging as index-based elements.

    Attributes:
        states: ``(n_states, n_points, 3)`` vertex positions, one slice per
            entry of :data:`DEFLECTION_STATES`.
        lines: ``(n_lines, 2)`` vertex indices.
        line_colors: ``(n_lines, 3)`` RGB in 0..1.
        polygons: ``(n_polygons, 4)`` vertex indices.
        polygon_colors: ``(n_polygons, 3)`` RGB in 0..1.
    """

    states: np.ndarray
    lines: np.ndarray
    line_colors: np.ndarray
    polygons: np.ndarray
    polygon_colors: np.ndarray

    @property
    def point_count(self) -> int:
        return self.states.shape[1]

    @classmethod
    def load(cls, directory: str | Path) -> "CanopyMesh":
        """Load every ``points_*.txt`` plus ``lines.txt`` and ``poligons.txt``."""
        base = Path(directory)

        states = np.stack(
            [_load_table(base / f"points_{_state_name(d)}.txt", 4)[:, 1:4]
             for d in DEFLECTION_STATES]
        )

        line_table = _load_table(base / "lines.txt", 6)
        polygon_table = _load_table(base / "poligons.txt", 9)

        return cls(
            states=states,
            lines=line_table[:, 1:3].astype(np.int32) - 1,
            line_colors=line_table[:, 3:6],
            polygons=polygon_table[:, 1:5].astype(np.int32) - 1,
            polygon_colors=polygon_table[:, 6:9],
        )

    def deform(self, control: ControlInput) -> np.ndarray:
        """Vertex positions for a given toggle setting, shape ``(n_points, 3)``.

        Each half of the canopy follows its own toggle: vertices with a
        non-negative spanwise coordinate use ``control.right``, the rest use
        ``control.left``.  Within a half, the two bracketing deflection states
        are blended linearly.
        """
        span = self.states[0, :, 2]
        deflection = np.where(span >= 0.0, control.right, control.left)

        n_states = self.states.shape[0]
        scaled = np.clip(deflection, 0.0, 1.0) * (n_states - 1)
        lower = np.minimum(np.floor(scaled).astype(np.intp), n_states - 2)
        blend = (scaled - lower)[:, None]

        columns = np.arange(self.states.shape[1])
        return (
            self.states[lower, columns] * (1.0 - blend)
            + self.states[lower + 1, columns] * blend
        )

    def to_world(self, vertices: np.ndarray, position, frame) -> np.ndarray:
        """Rotate and translate canopy-local vertices into world axes.

        Args:
            vertices: ``(n, 3)`` local positions, as returned by :meth:`deform`.
            position: World position of the canopy origin (a ``Vec3``).
            frame: A :class:`~parafoil.dynamics.BodyFrame`.

        The local axes are ``(along course, up, spanwise)``, which map onto the
        body frame's ``(forward, up, side)``.
        """
        rotation = np.array([frame.forward, frame.up, frame.side], dtype=np.float64)
        return vertices @ rotation + np.asarray(position, dtype=np.float64)


def _state_name(deflection: float) -> str:
    """``0.0 -> "0"``, ``0.5 -> "0.5"``, ``1.0 -> "1"`` -- the original naming."""
    return str(int(deflection)) if deflection in (0.0, 1.0) else f"{deflection:g}"


def _load_table(path: Path, columns: int) -> np.ndarray:
    """Read a whitespace-separated numeric table, skipping Russian headers.

    The files start with two or three comment lines and may contain a lone
    count line, so rows are filtered by column count rather than by position.
    """
    rows: list[list[float]] = []
    with path.open(encoding="cp1251") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != columns:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # header text that happens to have the right width

    if not rows:
        raise ValueError(f"no {columns}-column data rows found in {path}")
    return np.asarray(rows, dtype=np.float64)
