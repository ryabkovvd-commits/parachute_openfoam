"""Driving an out-of-process flow solver from the shape loop.

``couple`` needs an aerodynamic module: something it can hand a shape and get
a pressure field back from.  When that module is a CFD run the exchange stops
being a function call and becomes a conversation with another program --
write a geometry, start it, wait, read a result -- and every step of that can
fail in a way that leaves a plausible-looking file behind.

This module is that conversation, wrapped so the failures are loud.  It is
deliberately solver-agnostic: what it knows is how to write the geometry, how
to check what came back, and how to refuse it if it is not usable.  Which
solver runs, and how its case is set up, is supplied from outside.

The traps this guards against
-----------------------------

**A stale result read as a fresh one.**  The single most dangerous failure in
an unattended loop.  The solver crashes, last pass's pressure file is still
sitting there, the loop reads it, the shape does not move, and the loop
reports convergence.  Every pass therefore deletes the result file before
starting the solver and refuses to continue if it does not reappear.

**Pressure handed over in the wrong units.**  The exchange is in ``dCp``, but
a solver will happily write pascals, and OpenFOAM's incompressible solvers
write *kinematic* pressure -- p/rho, in m^2/s^2 -- which is off by a factor
of about 1.2.  All three look like ordinary numbers.  A dCp outside a few
units is not physical for a canopy, so a range check catches it on the first
pass instead of after ten.

**A converged-looking run that never converged.**  Left to the caller: the
runner it supplies should check the solver's own residuals.  Nothing here can
tell a converged field from an unconverged one.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .exchange import export_surface, read_panel_pressures
from .mesh import CanopyMesh, panel_area_and_normal
from .pressure import PressureField

#: The widest range of pressure difference coefficient that could still be a
#: real reading on a canopy.  Stagnation is 1, the measured distributions here
#: peak near 2, and the base suction adds something under 1.  Six is generous
#: by any reading; a hundred means somebody sent pascals.
PLAUSIBLE_DCP = (-6.0, 6.0)


class FlowSolverError(RuntimeError):
    """The external solver did not return a usable pressure field."""


def shell_runner(
    command: Sequence[str] | str,
    *,
    timeout: float = 3600.0,
    shell: bool = False,
) -> Callable[[Path], None]:
    """Run a command in the case directory, and raise if it fails.

    The command is whatever drives the solver end to end -- typically a shell
    script that meshes, solves and samples the surface.  Its exit status is
    taken seriously: a non-zero return stops the loop rather than letting it
    continue on whatever happens to be on disk.
    """

    def run(case: Path) -> None:
        try:
            result = subprocess.run(
                command,
                cwd=case,
                timeout=timeout,
                capture_output=True,
                text=True,
                shell=shell,
            )
        except subprocess.TimeoutExpired as exc:
            raise FlowSolverError(
                f"the flow solver did not finish within {timeout:.0f} s in {case}"
            ) from exc
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-20:]
            raise FlowSolverError(
                f"the flow solver failed in {case} (exit {result.returncode}):\n"
                + "\n".join(tail)
            )

    return run


def waiting_runner(
    pressure_name: str = "cp.csv",
    *,
    poll: float = 5.0,
    timeout: float | None = None,
    announce: Callable[[str], None] = print,
) -> Callable[[Path], None]:
    """Half-automatic: export the geometry, then wait for a human to run the
    solver and drop the result in.

    Worth using before automating anything.  It exercises the whole chain --
    export, exchange, re-solve, convergence -- and shows how many passes the
    problem actually needs, which is the number that decides whether
    automating the solver is worth the effort at all.
    """

    def run(case: Path) -> None:
        target = case / pressure_name
        announce(
            f"Геометрия выгружена в {case}.\n"
            f"Посчитайте обтекание и положите результат в {target}.\n"
            f"Жду..."
        )
        waited = 0.0
        while not target.exists():
            time.sleep(poll)
            waited += poll
            if timeout is not None and waited > timeout:
                raise FlowSolverError(f"nothing appeared at {target} within {timeout:.0f} s")
        announce(f"Прочитан {target}.")

    return run


def read_openfoam_raw(path: str | Path, dynamic_pressure: float) -> PressureField:
    """Read OpenFOAM's ``sampledSurface`` raw format.

    The layout is ``# x y z field`` with comment lines, one sample per row.

    Nothing here converts units: whatever the fourth column holds is taken as
    ``dCp`` directly.  Making the solver write a coefficient rather than a
    pressure is the case's business, and doing it there keeps the one number
    that could silently be wrong in one place instead of two.
    """
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append([float(p) for p in parts[:4]])
    if not rows:
        raise ValueError(f"no samples found in {path}")

    from .exchange import pressure_from_points

    data = np.asarray(rows, dtype=np.float64)
    return pressure_from_points(
        data[:, :3], data[:, 3], dynamic_pressure=dynamic_pressure
    )


@dataclass
class ExternalFlow:
    """An aerodynamic module backed by a solver in another process.

    Call it with a mesh and it returns the pressure field on that shape, which
    is exactly the interface ``couple`` wants::

        flow = ExternalFlow(workdir="run", run=shell_runner(["./solve.sh"]),
                            dynamic_pressure=49.8)
        result, history = couple(mesh, flow, passes=12)

    Attributes:
        workdir: Directory the passes are run in.
        run: Callable given the pass's directory; it must leave the pressure
            file there.  Build one with :func:`shell_runner` or
            :func:`waiting_runner`.
        dynamic_pressure: ``rho V^2 / 2``, Pa -- the scale the returned
            coefficients are turned into pressures with.
        geometry_name: What the shape is written as, in the pass directory.
        pressure_name: What the result is expected to be called.
        reader: How to turn that file into a field.  Defaults to the CSV
            reader; pass :func:`read_openfoam_raw` for OpenFOAM's own format.
        separate_passes: Keep each pass in its own subdirectory.  On by
            default -- when a coupled run goes wrong the sequence of shapes is
            usually the only way to see where.
    """

    workdir: str | Path
    run: Callable[[Path], None]
    dynamic_pressure: float
    geometry_name: str = "canopy.stl"
    pressure_name: str = "cp.csv"
    reader: Callable[[Path, float], PressureField] = read_panel_pressures
    separate_passes: bool = True
    passes_run: int = field(default=0, init=False)

    def __call__(self, mesh: CanopyMesh) -> PressureField:
        self.passes_run += 1
        root = Path(self.workdir)
        case = root / f"pass{self.passes_run:02d}" if self.separate_passes else root
        case.mkdir(parents=True, exist_ok=True)

        export_surface(mesh, case / self.geometry_name)

        # Never let last pass's answer stand in for this one.
        result = case / self.pressure_name
        if result.exists():
            result.unlink()

        self.run(case)

        if not result.exists():
            raise FlowSolverError(
                f"the flow solver returned no pressure field: expected "
                f"{result}, which does not exist.  The solver may have "
                f"finished without writing it, or written it elsewhere."
            )

        pressure = self.reader(result, self.dynamic_pressure)
        self._check(pressure, mesh, result)
        return pressure

    def _check(self, pressure: PressureField, mesh: CanopyMesh, source: Path) -> None:
        """Sample the returned field on the shape it was computed for, and
        refuse anything that cannot be a pressure coefficient."""
        anchor = mesh.position[mesh.confluence[0]]
        _, normals = panel_area_and_normal(mesh.position, mesh.quads, anchor)
        centroids = mesh.position[mesh.quads].mean(axis=1)
        dcp = pressure.at(centroids, normals, mesh.nominal_diameter / 2.0)
        dcp = dcp / pressure.dynamic_pressure

        if not np.all(np.isfinite(dcp)):
            raise FlowSolverError(
                f"{source} gives a pressure field that is not finite on "
                f"{np.count_nonzero(~np.isfinite(dcp))} of {len(dcp)} panels."
            )

        low, high = PLAUSIBLE_DCP
        if dcp.min() < low or dcp.max() > high:
            raise FlowSolverError(
                f"{source} gives dCp from {dcp.min():.3g} to {dcp.max():.3g}, "
                f"outside the plausible range {low} to {high}.  The usual "
                f"cause is units: the exchange is in coefficient, not "
                f"pressure, and OpenFOAM's incompressible solvers write "
                f"kinematic pressure (p/rho).  Divide by q = "
                f"{self.dynamic_pressure:.1f} Pa, and by rho as well if the "
                f"solver is incompressible."
            )
