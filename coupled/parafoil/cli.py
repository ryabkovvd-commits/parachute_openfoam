"""Command line front end.

Run a flight and print, plot, or export the result::

    python -m parafoil --init "D:/Trainer/Init.txt" --plot
    python -m parafoil --pilot glide --csv flight.csv
    python -m parafoil --target -100 100 --save landing.png --theme dark

Like :mod:`parafoil.ui`, this module is presentation only: it parses arguments,
calls :func:`~parafoil.simulation.simulate`, and formats the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .autopilot import HomingAutopilot
from .canopy import ControlInput
from .dynamics import Parafoil
from .pilots import FixedPilot, Pilot, RateLimitedPilot
from .scenario import Scenario
from .simulation import Trajectory, simulate
from .vectors import Vec3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parafoil",
        description="Simulate the flight of a ram-air parachute to a landing point.",
    )
    parser.add_argument(
        "--init", type=Path, metavar="FILE",
        help="original Init.txt with start conditions and the wind table",
    )
    parser.add_argument(
        "--target", type=float, nargs=2, metavar=("EAST", "SOUTH"),
        help="landing point in metres (default: -100 100, as in the original)",
    )
    parser.add_argument(
        "--pilot", choices=("auto", "glide", "flare"), default="auto",
        help="auto: guided approach and flare; glide: hands up; flare: both toggles down",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, metavar="SECONDS",
        help="integration step (default: 0.01, matching the original)",
    )
    parser.add_argument(
        "--max-time", type=float, default=600.0, metavar="SECONDS",
        help="give up if the canopy has not landed by then (default: 600)",
    )
    parser.add_argument(
        "--instant-toggles", action="store_true",
        help="let the autopilot move the brakes instantly instead of at 0.6/s",
    )
    parser.add_argument(
        "--model", choices=("point-mass", "rigid"), default="point-mass",
        help="point-mass: the original 3-DOF reconstruction (default); "
             "rigid: 6-DOF with real attitude. See README for what the rigid "
             "model does and does not yet do",
    )
    parser.add_argument(
        "--aerodynamics", choices=("lumped", "strip"), default="lumped",
        help="rigid model only: lumped treats the canopy as one surface, "
             "strip resolves it across the span (experimental, spirally "
             "divergent -- see README)",
    )
    parser.add_argument("--plot", action="store_true", help="show an interactive figure")
    parser.add_argument("--save", type=Path, metavar="PNG", help="write the figure to a file")
    parser.add_argument("--csv", type=Path, metavar="FILE", help="export the trajectory")
    parser.add_argument(
        "--theme", choices=("light", "dark"), default="light",
        help="figure colour scheme (default: light)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scenario = Scenario.from_init_file(args.init) if args.init else Scenario()
    if args.target:
        scenario = scenario.replace(target=Vec3(args.target[0], 0.0, args.target[1]))

    pilot = _build_pilot(args, scenario)
    options = {"aerodynamics": args.aerodynamics} if args.model == "rigid" else {}
    trajectory = simulate(
        scenario, pilot, dt=args.dt, max_time=args.max_time,
        model=args.model, **options,
    )

    print(_describe(scenario, trajectory))

    if args.csv:
        _write_csv(trajectory, args.csv)
        print(f"trajectory written to {args.csv}")

    if args.plot or args.save:
        _draw(trajectory, scenario.target, args)

    return 0 if trajectory.report.landed else 1


def _build_pilot(args, scenario: Scenario) -> Pilot:
    if args.pilot == "glide":
        return FixedPilot(ControlInput())
    if args.pilot == "flare":
        return FixedPilot(ControlInput(left=1.0, right=1.0))

    # The autopilot needs to know where it starts in order to lay out its
    # glide slope, so build the vehicle once to read its initial state.
    probe = Parafoil(
        params=scenario.canopy,
        wind=scenario.wind,
        position=scenario.position,
        velocity=scenario.velocity,
        air_density=scenario.air_density,
        gravity=scenario.gravity,
    )
    autopilot = HomingAutopilot.for_flight(
        target=scenario.target,
        state=probe.state,
        wind=scenario.wind,
        params=scenario.canopy,
    )
    return autopilot if args.instant_toggles else RateLimitedPilot(autopilot)


def _describe(scenario: Scenario, trajectory: Trajectory) -> str:
    r = trajectory.report
    ground_wind = scenario.wind.at(0.0)
    return "\n".join(
        [
            f"release      : {_fmt(scenario.position)} at {scenario.velocity.norm():.1f} m/s",
            f"target       : {_fmt(scenario.target)}",
            f"surface wind : {ground_wind.norm():.1f} m/s",
            f"model        : {trajectory.model}",
            f"samples      : {len(trajectory)}",
            f"result       : {r.summary()}",
            f"touchdown    : {_fmt(r.touchdown)}",
        ]
    )


def _fmt(v: Vec3) -> str:
    return f"({v.x:.1f}, {v.y:.1f}, {v.z:.1f})"


def _write_csv(trajectory: Trajectory, path: Path) -> None:
    import numpy as np

    table = np.column_stack(
        [
            trajectory.time,
            trajectory.position,
            trajectory.velocity,
            trajectory.wind,
            trajectory.control,
        ]
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="time,x,y,z,vx,vy,vz,wx,wy,wz,left,right",
        comments="",
        fmt="%.4f",
    )


def _draw(trajectory: Trajectory, target: Vec3, args) -> None:
    try:
        import matplotlib.pyplot as plt

        from .ui.plots import Theme, plot_flight
    except ImportError:
        print("matplotlib is not installed; run: pip install matplotlib", file=sys.stderr)
        return

    theme = Theme.dark() if args.theme == "dark" else Theme.light()
    figure = plot_flight(trajectory, target, theme=theme)
    if args.save:
        figure.savefig(args.save, dpi=150, facecolor=theme.surface, bbox_inches="tight")
        print(f"figure written to {args.save}")
    if args.plot:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())

