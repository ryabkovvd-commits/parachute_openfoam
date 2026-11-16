"""Ram-air parachute (parafoil) flight simulation.

A Python rewrite of the C#/XNA "Тренажёр парашютиста" landing trainer, with the
physics separated from any display code.

Quick start::

    from parafoil import Scenario, HomingAutopilot, simulate, Vec3

    scenario = Scenario.from_init_file("Init.txt")
    autopilot = HomingAutopilot.for_flight(
        target=scenario.target,
        state=... ,          # see examples/fly.py, or use the CLI
        wind=scenario.wind,
        params=scenario.canopy,
    )
    trajectory = simulate(scenario, autopilot)
    print(trajectory.report.summary())

The command line entry point wires all of that together::

    python -m parafoil --init "Init.txt" --plot
"""

from .atmosphere import WindProfile
from .autopilot import GlideSlope, HomingAutopilot, Phase
from .canopy import AeroCoefficients, CanopyParams, ControlInput
from .dynamics import BodyFrame, FlightState, Parafoil, VerletIntegrator
from .mesh import CanopyMesh
from .pilots import FixedPilot, Pilot, RateLimitedPilot, ScriptedPilot
from .scenario import DEFAULT_TARGET, Scenario
from .simulation import (
    LandingReport,
    Trajectory,
    Vehicle,
    build_vehicle,
    iter_flight,
    simulate,
)
from .vectors import NORTH, UP, ZERO, Vec3, signed_angle

__version__ = "1.0.0"

__all__ = [
    "AeroCoefficients",
    "BodyFrame",
    "CanopyMesh",
    "CanopyParams",
    "ControlInput",
    "DEFAULT_TARGET",
    "FixedPilot",
    "FlightState",
    "GlideSlope",
    "HomingAutopilot",
    "LandingReport",
    "NORTH",
    "Parafoil",
    "Phase",
    "Pilot",
    "RateLimitedPilot",
    "Scenario",
    "ScriptedPilot",
    "Trajectory",
    "UP",
    "Vec3",
    "Vehicle",
    "VerletIntegrator",
    "WindProfile",
    "ZERO",
    "build_vehicle",
    "iter_flight",
    "signed_angle",
    "simulate",
]
