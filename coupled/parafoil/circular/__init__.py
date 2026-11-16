"""Flat-circular (round) parachute with riser-pull control.

A completely different vehicle from the ram-air canopy the rest of this
package models, added **alongside** it rather than replacing anything: the
parafoil code, its parameters and its tests are untouched.

What is different
-----------------

A round parachute is a drag device.  It has no meaningful glide, so it cannot
be flown to a target -- it falls at roughly 9 m/s and goes wherever the wind
takes it.  Pulling on a riser deforms the canopy and tilts the drag vector,
which buys a *small* horizontal drive: about 4 m/s for a G-12, against a 9 m/s
descent.  That is far less than a typical wind.

The consequence shapes everything.  You cannot steer to the target; you can
only correct the error between where the wind actually took you and where you
predicted it would.  So the guidance is **trajectory tracking** against a
reference path computed from a wind forecast, not homing.  This is the AGAS
(Affordable Guided Airdrop System) concept, and the model here follows the
published descriptions of it.

Two further differences from the parafoil:

* the canopy **rotates continuously** -- about 1.8 deg/s for a G-12 -- so a
  riser that pulls "north" now will pull "north-east" a minute later, and the
  controller has to know the heading;
* control is **four discrete on/off risers**, not two continuous toggles, so
  the law is bang-bang with a dead band rather than proportional.

Sources
-------

* Dellicker, Benney & Brown, "Guidance and Control for Flat-Circular
  Parachutes", J. Aircraft 38(5), 2001 -- concept, point-mass model, the
  tolerance-cone controller, G-12 rotation rate.
* Yakimenko et al., "Synthesis of Optimal Control and Flight Testing of an
  Autonomous Circular Parachute" -- G-12 figures, the ``u, v in {-V, 0, V}``
  control model with V ~ 4 m/s, and the note that coning motion averages out
  so heading alone suffices.
"""

from .canopy import RiserCommand, RoundCanopy
from .dynamics import RoundParachute, RoundState, RoundTrajectory, simulate_round
from .guidance import BallisticPlan, ConeTracker, NoControl, ReferenceTrajectory

__all__ = [
    "BallisticPlan",
    "ConeTracker",
    "NoControl",
    "ReferenceTrajectory",
    "RiserCommand",
    "RoundCanopy",
    "RoundParachute",
    "RoundState",
    "RoundTrajectory",
    "simulate_round",
]
