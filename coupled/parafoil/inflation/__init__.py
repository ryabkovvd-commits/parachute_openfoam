"""Shape of an inflated canopy, by the lumped-mass method.

Everything else in this project takes the canopy's aerodynamics as given --
either reconstructed from the original program, or read off a flight-test
table.  This package computes where those numbers come from: it works out the
shape a fabric canopy takes under a given pressure difference, and what the
fabric and the reinforcing bands carry while it holds that shape.

The method is Leonov, Morozov & Ponomarev, "Shape Modeling and Strength
Analysis of Parachutes", Mechanics of Solids 46(2), 2011 -- the lumped-mass
method behind the Russian "Kupol" and "Shaping" systems.  In outline:

* the canopy becomes nodal point masses joined by weightless elastic rods,
  two superimposed grids -- the reinforcing framework and the fabric;
* the rods **work in tension only**.  A rod that would be compressed simply
  carries nothing, and that one rule is what produces wrinkling and slack
  panels without any special treatment;
* each node is integrated explicitly under the pressure difference, the rod
  tensions and its own weight, with the time step from the Courant condition;
* the motion is damped by the pseudoviscosity method until the shape stops
  changing.

Why this method rather than a finite-element one: parachute fabric is limp.
It has no bending stiffness worth modelling and it goes slack under the least
compression, which is exactly the case finite elements handle badly.  A
tension-only rod network handles it by construction.

Why this method rather than the nonlinear mass-spring model of Anton et al.
(2023): that one needs the full force-elongation curve of the fabric in three
directions, and no such data is available here.  Leonov's needs a linear
stiffness and the tension-only rule, which is a far lower bar.

State of this package
---------------------

The geometry is a **parametric** flat-circular canopy, not any real one.  The
purpose is to get the method working and validated while the real cutting
pattern and material data are still being gathered; the mesh and the material
properties are inputs, so swapping in a real canopy is a matter of data, not
of code.
"""

from .materials import Fabric, RodMaterial, Webbing
from .mesh import CanopyMesh, flat_circular
from .pressure import (
    PressureField,
    local_incidence,
    measured_incidence,
    radial_profile,
    uniform_pressure,
)
from .relax import InflationResult, couple, inflate, inflate_to_load

__all__ = [
    "CanopyMesh",
    "Fabric",
    "InflationResult",
    "PressureField",
    "RodMaterial",
    "Webbing",
    "couple",
    "flat_circular",
    "inflate",
    "inflate_to_load",
    "local_incidence",
    "measured_incidence",
    "radial_profile",
    "uniform_pressure",
]
