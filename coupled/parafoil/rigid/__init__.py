"""Rigid-body (6-DOF) flight model, built alongside the point-mass one.

The package in :mod:`parafoil` proper is a faithful reconstruction of the
original program: a point mass whose attitude is *assigned* each step from the
direction of the resultant force.  That model cannot show anything rotational
-- no lag rolling into a turn, no oscillation after a control input, no pitch
change during the flare -- because it has no rotational state to oscillate.

This package adds one.  It integrates the full rigid-body equations::

    r' = v
    (m + A) v' = F + m*g - w x (m + A) v
    q' = q * (0, w) / 2
    (I + I_a) w' = M - w x (I + I_a) w

and is deliberately kept **separate** rather than replacing anything: the
original model, its parameters and its 33 tests stay exactly as they were, and
both are selectable at run time.

What this is *not*
------------------

A full 6-DOF aerodynamic model needs a set of stability derivatives -- CL(a),
Cm(a), Clp, Cmq, Cnr and the rest -- and the original program's data contains
none of them.  Rather than invent a dozen numbers, this model reuses the
force model that *is* documented (:class:`~parafoil.canopy.CanopyParams`) and
gets its moments from geometry alone:

    the aerodynamic force acts at the canopy, the centre of mass sits at the
    payload some 8.4 m below it, so ``M = r x F`` follows without a single new
    aerodynamic coefficient.

That is enough for the pendulum mode, for roll-in lag and for turn dynamics.
It is *not* enough for stall, for pitch-damping-driven phugoid detail, or for
anything that needs lift to vary with angle of attack.  Those need real wind
tunnel or flight data; see :mod:`parafoil.rigid.aerodynamics` for exactly
which terms are geometric and which are lumped.
"""

from .body import DEFAULT_DAMPING_RATIO, RigidParafoil
from .inertia import CanopyGeometry, MassProperties, Mat3
from .integrators import RigidState, rk4_step
from .quaternion import Quat

__all__ = [
    "CanopyGeometry",
    "DEFAULT_DAMPING_RATIO",
    "MassProperties",
    "Mat3",
    "Quat",
    "RigidParafoil",
    "RigidState",
    "rk4_step",
]
