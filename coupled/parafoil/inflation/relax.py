"""The lumped-mass solver: explicit stepping damped to a static shape.

This is Leonov's scheme, equations (2.4) to (2.12).  Each node is a point mass
pushed by the pressure on the fabric around it and pulled by the rods meeting
at it; the equation of motion is integrated explicitly, and the velocity is
scaled down every step so the motion dies away and what is left is the
equilibrium shape.

The damping is not physical.  It is the pseudoviscosity trick: the transient
is of no interest, only the shape it settles into, so the fastest route there
is to drain energy as aggressively as stability allows.  Leonov says the
factor is "selected in the process of the computational experiment", and that
is exactly right -- too much and convergence crawls, too little and the canopy
rings for a long time.

Two things make this work on limp fabric where finite elements struggle:

* rods carry tension only, so slack panels simply stop pulling;
* the time step comes from the Courant condition on the stiffest, shortest
  rod, so nothing has a chance to overshoot.

Correspondence with the paper's equation (2.6)
----------------------------------------------

Checked term by term against the published formula::

    a = [ (N2 - N1) + (N3 - N4)
          + dp * (l1+l2)(l3+l4)/4 ] / [ rho_s*(l1+l2)(l3+l4)/4
                                        + rho_l*(l1+l2+l3+l4)/2 ] - g

* the rod term is the net pull of the four rods at the node -- here the same
  sum, assembled over all rods rather than written out as two differences;
* the pressure term uses the node's **tributary area**, the rectangle made of
  half of each adjacent rod, at its *current* length.  This code sums a
  quarter of each surrounding panel's area instead, which is identical on a
  regular grid and better behaved on a warped one;
* the denominator is the nodal mass: surface density over the tributary area
  plus half of every adjacent rod.  Same here.

Gravity is a separate term, not part of the numerator: the transcription
available reads ``(... - g) / mass``, which would divide an acceleration by a
mass.  ``F/m - g`` is the reading that carries the right units, and it is the
one used.

What the formula does **not** fix is how the area's normal is oriented, and
that turned out to matter -- see :func:`~parafoil.inflation.mesh.panel_area_and_normal`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import CanopyMesh, panel_area_and_normal
from .pressure import PressureField, uniform_pressure
from .constraints import MeanAxisConstraint


#: How often the canopy's attitude is checked, in steps.  The drift being
#: watched for is slow -- degrees over hundreds of thousands of steps -- so
#: measuring it every step would drown the signal in the round-off of a single
#: node displacement.
_AXIS_EVERY = 500


def _canopy_axis(position, quads, area, anchor):
    """Unit vector from the confluence point to the fabric's area centroid.

    This is what "which way the canopy is pointing" means here.  It is the
    quantity a rigid rotation about the confluence point moves and the force
    residual does not see.
    """
    centre = (position[quads].mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    axis = centre - anchor
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


@dataclass(slots=True)
class InflationResult:
    """The converged shape and what it tells us.

    Note that :attr:`mesh` is the caller's own mesh, not a copy, and the
    geometric properties below are worked out from it when they are read.
    Move the mesh after inflating and they will quietly report the new shape
    while the recorded forces still describe the old one -- a result whose
    halves disagree.  Anything that inflates and then goes on to modify the
    mesh has to put the settled shape back before handing the result on;
    :func:`couple` does exactly that.
    """

    mesh: CanopyMesh
    steps: int
    converged: bool
    residual: float
    tension: np.ndarray            # (m,) rod tensions, N
    #: Resultant the suspension lines pull the confluence point with, N.
    #: Points up, towards the canopy; the equal and opposite force is what the
    #: payload hangs from.
    confluence_load: np.ndarray
    pressure_resultant: np.ndarray  # (3,) integral of the pressure, N
    history: np.ndarray            # (k, 2) step and mean node movement
    other_applied_resultant: np.ndarray | None = None
    constraint_force: np.ndarray | None = None

    @property
    def projected_ratio(self) -> float:
        """Inflated projected diameter over nominal.

        For a flat-circular canopy this lands near 0.67 in practice, and it is
        the single most useful check on a shape calculation because it needs
        no material data at all -- just the geometry the solver produced.
        """
        return self.mesh.projected_diameter() / self.mesh.nominal_diameter

    @property
    def balance_error(self) -> float:
        """Relative mismatch between the pressure resultant and the line load.

        Both vectors point the same way -- :attr:`confluence_load` is the pull
        the lines exert *on the confluence point*, which is upwards, and so is
        the pressure resultant on the canopy.  Equilibrium of the canopy is
        pressure up balanced by the lines pulling it down, so the check is
        their **difference**.

        Nothing in the solver enforces this, so it is a genuine independent
        test that the shape has converged rather than merely stopped moving.
        """
        expected=self.pressure_resultant.copy()
        if self.other_applied_resultant is not None:
            expected+=self.other_applied_resultant
        scale = np.linalg.norm(expected)
        if scale < 1e-9:
            return 0.0
        return float(
            np.linalg.norm(expected - self.confluence_load) / scale
        )

    def safety_factors(self) -> np.ndarray:
        """Breaking load over tension for every rod; ``inf`` where slack."""
        return np.divide(
            self.mesh.breaking_load,
            self.tension,
            out=np.full_like(self.tension, np.inf),
            where=self.tension > 1e-9,
        )

    def drag_coefficient(self, dcp_mean: float) -> float:
        """Drag coefficient referred to the **nominal** area.

        The closing check of the whole chain: multiply the pressure
        coefficient the loading was computed with by the area ratio the shape
        came out at.  For a flat-circular canopy the answer should land near
        the textbook 0.75-0.80.
        """
        return dcp_mean * self.mesh.projected_area() / self.mesh.nominal_area()


def _rod_tensions(mesh: CanopyMesh, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tension in each rod and the unit vector from its first node to its second."""
    delta = position[mesh.rods[:, 1]] - position[mesh.rods[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    direction = np.divide(
        delta, length[:, None], out=np.zeros_like(delta), where=length[:, None] > 1e-12
    )
    strain = np.divide(
        length, mesh.rest_length,
        out=np.ones_like(length), where=mesh.rest_length > 1e-12,
    ) - 1.0
    # Tension only: a rod that would be compressed carries nothing at all.
    # This single line is what produces slack panels and wrinkling.
    tension = np.where(strain > 0.0, mesh.stiffness * strain, 0.0)
    return tension, direction


def _courant_step(mesh: CanopyMesh, safety: float = 0.15) -> float:
    """Largest stable explicit step, from the stiffest **node**.

    The obvious estimate -- the period of one rod between its two end masses
    -- is wrong, and wrong in the dangerous direction.  Four rods meet at an
    interior node, so the stiffness it actually feels is their sum and its
    period is half what a single rod suggests.  Using the single-rod figure
    puts the step above the stability limit and the shape blows up after a
    couple of hundred steps.

    So: sum the axial rates ``stiffness / rest_length`` of every rod at each
    node, take the shortest resulting period, and keep a safety fraction of
    it.
    """
    rate = mesh.stiffness / np.maximum(mesh.rest_length, 1e-9)
    node_rate = np.zeros(len(mesh.position))
    np.add.at(node_rate, mesh.rods[:, 0], rate)
    np.add.at(node_rate, mesh.rods[:, 1], rate)

    movable = np.isfinite(mesh.mass) & (node_rate > 0.0)
    if not movable.any():
        raise ValueError("mesh has no movable nodes to step")

    period = 2.0 * np.pi * np.sqrt(mesh.mass[movable] / node_rate[movable])
    return safety * float(period.min())


def couple(
    mesh: CanopyMesh,
    aerodynamics,
    *,
    passes: int = 12,
    shape_tolerance: float = 1e-3,
    relaxation: float = 0.5,
    **options,
) -> tuple[InflationResult, list[float]]:
    """Iterate shape and loading against each other until both settle.

    This is the exchange the source paper describes between its elastic and
    aerodynamic modules, driven from this side.  Each pass:

    1. relax the canopy under the current loading;
    2. hand the resulting shape to ``aerodynamics``, which returns the
       loading on **that** shape;
    3. go round again.

    Args:
        aerodynamics: Callable taking the mesh in its current shape and
            returning a :class:`~parafoil.inflation.pressure.PressureField`.
            That may be a model such as
            :func:`~parafoil.inflation.pressure.local_incidence`, or a wrapper
            that writes the shape out, waits for a flow solver, and reads the
            answer back through :mod:`parafoil.inflation.exchange`.
        passes: Maximum coupled iterations.
        shape_tolerance: Convergence of the **outer** loop: mean node movement
            between successive passes, relative to the nominal diameter.
            Named apart from :func:`inflate`'s ``tolerance``, which governs
            the inner relaxation, because the two travel together through
            ``**options`` and a shared name silently collides.
        relaxation: How much of each new shape to accept, 0 to 1.  Taking the
            new shape whole tends to oscillate: a stronger loading inflates
            the canopy, the fuller canopy catches more, and the two overshoot
            past each other.  Half of each step damps that out.

    Returns:
        The final result and the movement recorded at each pass, so a
        convergence that stalled can be told from one that finished.
    """
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")

    history: list[float] = []
    previous = mesh.position.copy()
    previous_settled = previous
    result = None

    for _ in range(passes):
        field = aerodynamics(mesh)
        result = inflate(mesh, field, **options)
        settled = mesh.position.copy()

        # Convergence is "the answer stopped changing", so it compares one
        # settled shape with the last settled shape.
        #
        # Two near-misses, both of which cost passes without saying so.
        # Comparing against the *blended* position instead measures how far
        # under-relaxation still has to travel, not whether the answer has
        # moved: with a loading that is a plain function of geometry the
        # solver returns the same shape every pass -- the answer is right
        # after the first -- while that distance only halves each time, so
        # the loop runs ten passes to reach a tolerance it met at two.  With
        # a flow solver on the other end those are eight real runs.  And
        # measuring after the blend rather than before folds the damping into
        # the criterion, which then stops earlier the harder it is damped.
        moved = float(
            np.linalg.norm(settled - previous_settled, axis=1).mean()
            / mesh.nominal_diameter
        )
        previous_settled = settled
        history.append(moved)

        if moved < shape_tolerance:
            break

        if relaxation < 1.0:
            mesh.position = previous + relaxation * (settled - previous)
        previous = mesh.position.copy()

    # Leave the mesh holding the shape the result describes.
    #
    # ``InflationResult`` keeps a reference to this mesh and works its
    # geometric properties out on demand, so whatever ``mesh.position`` holds
    # when the caller reads ``projected_ratio`` is what they get.  Returning
    # with the blended position still in place -- which is where running out
    # of passes leaves it -- reports a shape that was never an equilibrium of
    # anything, alongside forces that came from the settled one.
    mesh.position = settled

    return result, history


def inflate_to_load(
    mesh: CanopyMesh,
    target_load: float,
    *,
    shape: PressureField | None = None,
    passes: int = 6,
    load_tolerance: float = 0.005,
    **options,
) -> InflationResult:
    """Find the shape that carries a **given load at the confluence point**.

    This is how the source paper poses the problem for strength work,
    equation (3.2): the canopy is shaped for the design load

        P_design = f_safety * P_operating

    rather than for an assumed pressure.  That is the better way round.  The
    load a canopy has to carry is known -- it is the suspended weight times a
    safety factor -- while the pressure coefficient is exactly the quantity
    one is least sure of, and guessing it wrong shows up as a shape that
    quietly carries the wrong load.

    Args:
        target_load: Resultant the lines must carry, N.
        shape: A pressure field giving the **distribution**; only its shape
            matters, since the magnitude is what gets solved for.  Uniform if
            omitted.
        passes: Maximum outer iterations on the pressure scale.
        load_tolerance: Relative accuracy on the load.
        **options: Passed through to :func:`inflate`.

    The outer loop is cheap: for a nearly inextensible canopy the shape barely
    moves, so the load is almost linear in the pressure and a proportional
    correction converges in two or three passes.
    """
    if target_load <= 0.0:
        raise ValueError("target load must be positive")

    field = shape or uniform_pressure(1.0, 1.0)
    scale = 1.0
    result = None

    for _ in range(passes):
        trial = PressureField(
            coefficient=field.coefficient,
            dynamic_pressure=field.dynamic_pressure * scale,
        )
        result = inflate(mesh, trial, **options)
        carried = float(np.linalg.norm(result.confluence_load))
        if carried < 1e-9:
            raise ValueError("the canopy carries no load; check the pressure field")
        error = carried / target_load - 1.0
        if abs(error) < load_tolerance:
            break
        scale *= target_load / carried

    return result


def inflate(
    mesh: CanopyMesh,
    pressure: PressureField | None = None,
    *,
    node_force: np.ndarray | None = None,
    damping: float = 0.98,
    gravity: float = 0.0,
    max_steps: int = 60_000,
    tolerance: float = 1e-6,
    drift_tolerance: float = 1e-9,
    record_every: int = 200,
    attitude_deg: float | None = None,
) -> InflationResult:
    """Relax the canopy to its shape under ``pressure``.

    Args:
        node_force: Loads already resolved onto the nodes, in newtons, as an
            alternative to ``pressure``.  Exactly one of the two is required.

            This exists because routing a CFD load through a panel loses more
            than the averaging does.  Measured against exact integration over
            14 269 CFD faces, collapsing the load to a coefficient per panel
            and rebuilding the force as ``dp * area * normal`` cost 2.8 % of
            the force and 14 % of the moment -- and only 0.23 % of that came
            from the averaging.  The rest was geometric: 128 flat quads are
            not the surface the flow saw.  Summing the face forces instead and
            handing them over as nodal loads reproduced the force to
            0.000 % and the moment to 2.2 %.

            The two also differ in kind, not just accuracy.  A ``PressureField``
            is attached to the body: it is re-evaluated against the current
            normals every step, so a rigid rotation about the confluence point
            carries the load around with it and leaves the moment's magnitude
            untouched -- there is then no equilibrium orientation at all, and a
            skewed canopy tumbles without settling.  Nodal loads are fixed in
            space, like the oncoming flow that produced them, so rotating the
            canopy does change the moment and an equilibrium can exist.  They
            are a dead load, though, valid only while the shape stays near the
            one the CFD saw; the outer coupling pass is what keeps that true.
        damping: The pseudoviscosity factor, applied to the velocity every
            step.  Below about 0.9 the shape creeps towards equilibrium far
            too slowly; above 0.995 it rings.
        gravity: Usually left at zero.  The canopy's own weight is four orders
            below the aerodynamic load, and including it only slows
            convergence.
        tolerance: Convergence is the mean out-of-balance force on a node,
            as a fraction of the mean pressure force on a node.  1e-3 means
            the residual is a thousandth of the load being applied.
        drift_tolerance: The *second* condition, in radians per step: how fast
            the canopy's axis may still be turning for the shape to count as
            settled.  Both must be met.

            This exists because the force residual on its own is not a test
            of equilibrium for anything asymmetric.  It measures elastic
            imbalance, and the mode that runs away on a skewed canopy is a
            near-rigid rotation about the confluence point -- which strains
            almost nothing and so shows up in the residual almost not at all.
            Left to the force test alone, a canopy quietly turning through
            seven degrees reported a *lower* residual than the same canopy
            stopped early at a tenth of a degree.
        record_every: How often to record progress, for plotting.

    Returns:
        An :class:`InflationResult`.  Check :attr:`~InflationResult.converged`
        and :attr:`~InflationResult.balance_error` before believing anything
        else in it.
    """
    if not 0.0 < damping <= 1.0 or max_steps<1 or tolerance<=0 or drift_tolerance<0 or record_every<1:
        raise ValueError("damping must lie in (0, 1]")
    if (pressure is None) == (node_force is None):
        raise ValueError("give either pressure or node_force, not both")
    if node_force is not None:
        node_force = np.asarray(node_force, dtype=np.float64)
        if node_force.shape != mesh.position.shape:
            raise ValueError(
                f"node_force must be {mesh.position.shape}, got {node_force.shape}"
            )
        if not np.isfinite(node_force).all():
            raise ValueError('Non-finite structural loads')

    position = mesh.position.copy()
    velocity = np.zeros_like(position)
    movable = np.isfinite(mesh.mass)
    inv_mass = np.zeros(len(position))
    inv_mass[movable] = 1.0 / mesh.mass[movable]
    if not np.isfinite(position).all() or np.any(mesh.mass[movable]<=0):
        raise ValueError('Invalid coordinates or movable masses')
    constraint=MeanAxisConstraint(mesh,attitude_deg) if attitude_deg is not None else None
    if constraint is not None:
        position=constraint.align_initial(position)
        constraint.project_state(position,velocity)
    constraint_force=np.zeros_like(position)

    dt = _courant_step(mesh)
    # The confluence point sits inside the canopy, so it is what panel normals
    # are oriented away from -- see `panel_area_and_normal`.
    anchor = position[mesh.confluence[0]].copy()

    # Weight as a force, computed only where the mass is finite: the fixed
    # confluence point carries an infinite mass, and `inf * 0` is a NaN that
    # quietly poisons the whole array.
    weight = np.zeros_like(position)
    weight[movable, 2] = -gravity * mesh.mass[movable]

    history: list[tuple[int, float]] = []
    converged = False
    residual = np.inf
    drift = np.inf
    previous_axis = None
    area = np.zeros(len(mesh.quads))
    tension = np.zeros(len(mesh.rods))
    pressure_force = np.zeros_like(position)

    for step in range(1, max_steps + 1):
        # --- pressure on each panel, shared out to its four corners --------
        area, normal = panel_area_and_normal(position, mesh.quads, anchor)
        if node_force is None:
            centroids = position[mesh.quads].mean(axis=1)
            dp = pressure.at(centroids, normal, mesh.nominal_diameter / 2.0)
            panel_force = (dp * area)[:, None] * normal

            pressure_force.fill(0.0)
            for corner in range(4):
                np.add.at(pressure_force, mesh.quads[:, corner], panel_force / 4.0)
        else:
            # Already on the nodes, and deliberately not re-evaluated against
            # the current normals -- see the note on ``node_force``.
            pressure_force = node_force

        # --- rod tensions ---------------------------------------------------
        tension, direction = _rod_tensions(mesh, position)
        pull = tension[:, None] * direction
        force = pressure_force + weight
        np.add.at(force, mesh.rods[:, 0], pull)
        np.add.at(force, mesh.rods[:, 1], -pull)
        if constraint is not None:
            force,constraint_force=constraint.project_force(force)

        # --- convergence on the out-of-balance force ------------------------
        # Not on how far the nodes moved this step: that scales with the time
        # step, so a small enough step makes any state look converged.  What
        # equilibrium actually means is that the forces on each node cancel,
        # and that test is independent of how the solver got there.
        out_of_balance = np.linalg.norm(force[movable], axis=1)
        scale = np.linalg.norm(pressure_force[movable], axis=1).mean()
        # Below 1 N/node use an explicit 1-N reference, never a fabricated zero.
        residual = float(out_of_balance.mean() / max(scale,1.0))

        # --- and on the canopy having stopped turning ------------------------
        # The force residual alone is not enough, and the way it fails is
        # nasty: it measures ELASTIC imbalance, while the mode that runs away
        # on a skewed canopy is a near-rigid rotation about the confluence
        # point.  Rigid rotation barely strains the fabric, so it produces
        # almost no out-of-balance force -- the residual cannot see it.
        #
        # Measured on a canopy with a quarter of its lines lengthened by
        # 0.05 m: stopping on the force residual alone gave 0.38 % and a tilt
        # of 0.12 degrees at 6913 steps, while running on to 400 000 steps
        # left the tilt at 7.12 degrees with the residual *lower*, 0.32 %.
        # The canopy turned sixty times further and the old test reported it
        # as better converged.
        if step % _AXIS_EVERY == 0 or step == 1:
            axis = _canopy_axis(position, mesh.quads, area, anchor)
            if previous_axis is not None:
                # Radians per step, scaled up to a comparable footing with the
                # force residual by the same window it was measured over.
                turn = float(np.arccos(np.clip(axis @ previous_axis, -1.0, 1.0)))
                drift = turn / _AXIS_EVERY
            previous_axis = axis

        # --- explicit step with pseudoviscosity -----------------------------
        acceleration = force * inv_mass[:, None]
        velocity = (velocity + acceleration * dt) * damping
        velocity[~movable] = 0.0
        position += velocity * dt
        if constraint is not None:
            constraint.project_state(position,velocity)
        if not np.isfinite(position).all():
            raise FloatingPointError('Structural coordinates diverged')

        if step % record_every == 0 or step == 1:
            history.append((step, residual))
        if not np.isfinite(residual):
            raise FloatingPointError(
                f"the shape diverged at step {step}.  Usually the loading is "
                f"too large for the structure: check that the dynamic pressure "
                f"is right (about 50 Pa at a 9 m/s descent, not 500), and that "
                f"the line strength can carry the resulting load."
            )
        if residual < tolerance:
            if not np.isfinite(drift) and previous_axis is not None and step > 1:
                # A canopy that settles inside one checking interval would
                # otherwise never be called converged: the periodic check has
                # not come round, so the drift is still infinite.  Measure it
                # over everything since the start instead.
                axis = _canopy_axis(position, mesh.quads, area, anchor)
                turn = float(np.arccos(np.clip(axis @ previous_axis, -1.0, 1.0)))
                drift = turn / (step - 1)
            if drift < drift_tolerance:
                converged = True
                break

    mesh.position = position

    if node_force is None:
        area, normal = panel_area_and_normal(position, mesh.quads, anchor)
        centroids = position[mesh.quads].mean(axis=1)
        dp = pressure.at(centroids, normal, mesh.nominal_diameter / 2.0)
        pressure_resultant = ((dp * area)[:, None] * normal).sum(axis=0)
        pressure_force=np.zeros_like(position)
        for corner in range(4):
            np.add.at(pressure_force,mesh.quads[:,corner],((dp*area)[:,None]*normal)/4)
    else:
        pressure_resultant = node_force.sum(axis=0)

    lines = mesh.line_rods
    tension, direction = _rod_tensions(mesh, position)
    final_force=pressure_force+weight
    np.add.at(final_force,mesh.rods[:,0],tension[:,None]*direction)
    np.add.at(final_force,mesh.rods[:,1],-tension[:,None]*direction)
    if constraint is not None:
        final_force,constraint_force=constraint.project_force(final_force)
    scale=max(float(np.linalg.norm(pressure_force[movable],axis=1).mean()),1.0)
    residual=float(np.linalg.norm(final_force[movable],axis=1).mean()/scale)
    converged=converged and residual<tolerance
    confluence_load = -(tension[lines][:, None] * direction[lines]).sum(axis=0)
    confluence_load += pressure_force[mesh.confluence].sum(0)+weight[mesh.confluence].sum(0)

    return InflationResult(
        mesh=mesh,
        steps=step,
        converged=converged,
        residual=residual,
        tension=tension,
        confluence_load=confluence_load,
        pressure_resultant=pressure_resultant,
        history=np.asarray(history, dtype=np.float64),
        other_applied_resultant=weight.sum(0)+constraint_force.sum(0),
        constraint_force=constraint_force,
    )
