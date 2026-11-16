"""Matplotlib views of a flight.

Kept deliberately thin: these functions take a finished
:class:`~parafoil.simulation.Trajectory` and draw it.  They never run physics,
never own state, and can be replaced wholesale by a different front end.

Every panel that shows more than one series shows series that share a unit, so
no plot here has two y-scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import Trajectory
from ..vectors import Vec3


@dataclass(frozen=True, slots=True)
class Theme:
    """Colours for one rendering mode.

    The three series colours are the first three slots of a categorical palette
    validated for colour-vision deficiency; keep them in this order rather than
    cycling through arbitrary hues.
    """

    surface: str
    ink: str
    muted: str
    grid: str
    series: tuple[str, str, str]
    target: str

    @classmethod
    def light(cls) -> "Theme":
        return cls(
            surface="#fcfcfb",
            ink="#0b0b0b",
            muted="#898781",
            grid="#e1e0d9",
            series=("#2a78d6", "#eb6834", "#1baf7a"),
            target="#d03b3b",
        )

    @classmethod
    def dark(cls) -> "Theme":
        return cls(
            surface="#1a1a19",
            ink="#ffffff",
            muted="#898781",
            grid="#2c2c2a",
            series=("#3987e5", "#d95926", "#199e70"),
            target="#d03b3b",
        )


def _style_axes(ax, theme: Theme) -> None:
    """Push the chrome into the background so the data reads first."""
    ax.set_facecolor(theme.surface)
    ax.grid(True, color=theme.grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.xaxis.label.set_color(theme.muted)
    ax.yaxis.label.set_color(theme.muted)
    ax.title.set_color(theme.ink)


def plot_ground_track(trajectory: Trajectory, target: Vec3, ax=None,
                      theme: Theme | None = None):
    """Top-down view of the flight path, with the aim point marked.

    East is right, north is up -- note that the simulation's Z axis points
    south, so it is negated here to match map convention.
    """
    import matplotlib.pyplot as plt

    theme = theme or Theme.light()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    east = trajectory.position[:, 0]
    north = -trajectory.position[:, 2]

    ax.plot(east, north, color=theme.series[0], linewidth=2, zorder=3)
    ax.plot(east[0], north[0], "o", color=theme.series[0], markersize=8, zorder=4)
    ax.plot(target.x, -target.z, "X", color=theme.target, markersize=12,
            markeredgewidth=0, zorder=5)
    ax.plot(east[-1], north[-1], "o", color=theme.target, markersize=8,
            markerfacecolor="none", markeredgewidth=2, zorder=5)

    # Direct labels beat a legend for three one-off markers.  Target and
    # touchdown are usually only tens of metres apart, so their labels are
    # pushed to opposite sides to stop them colliding.
    ax.annotate("release", (east[0], north[0]), textcoords="offset points",
                xytext=(10, 6), color=theme.ink, fontsize=9)
    ax.annotate("target", (target.x, -target.z), textcoords="offset points",
                xytext=(12, 6), color=theme.ink, fontsize=9,
                va="bottom", ha="left")
    ax.annotate(f"touchdown, {trajectory.report.miss_distance:.0f} m off",
                (east[-1], north[-1]), textcoords="offset points",
                xytext=(12, -8), color=theme.ink, fontsize=9,
                va="top", ha="left")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east, m")
    ax.set_ylabel("north, m")
    ax.set_title("Ground track")
    _style_axes(ax, theme)
    return ax


def plot_profiles(trajectory: Trajectory, axes=None, theme: Theme | None = None):
    """Three stacked time histories: altitude, speeds, and toggle positions.

    Args:
        axes: A sequence of three matplotlib axes, or ``None`` to create them.
    """
    import matplotlib.pyplot as plt

    theme = theme or Theme.light()
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    altitude_ax, speed_ax, control_ax = axes
    t = trajectory.time

    altitude_ax.plot(t, trajectory.altitude, color=theme.series[0], linewidth=2)
    altitude_ax.set_ylabel("altitude, m")
    altitude_ax.set_title("Altitude")

    # All three share m/s, so they belong on one axis.
    speed_ax.plot(t, trajectory.airspeed, color=theme.series[0], linewidth=2,
                  label="airspeed")
    speed_ax.plot(t, trajectory.ground_speed, color=theme.series[1], linewidth=2,
                  label="ground speed")
    speed_ax.plot(t, trajectory.descent_rate, color=theme.series[2], linewidth=2,
                  label="descent rate")
    speed_ax.set_ylabel("speed, m/s")
    speed_ax.set_title("Speeds")
    # Headroom so the three-column legend sits above the traces, not on them.
    speed_ax.set_ylim(top=float(trajectory.airspeed.max()) * 1.35)
    speed_ax.legend(frameon=False, labelcolor=theme.ink, fontsize=9,
                    ncol=3, loc="upper center")

    control_ax.plot(t, trajectory.control[:, 0], color=theme.series[0],
                    linewidth=2, label="left toggle")
    control_ax.plot(t, trajectory.control[:, 1], color=theme.series[1],
                    linewidth=2, label="right toggle")
    control_ax.set_ylim(-0.05, 1.05)
    control_ax.set_ylabel("deflection")
    control_ax.set_xlabel("time, s")
    control_ax.set_title("Control input")
    control_ax.legend(frameon=False, labelcolor=theme.ink, fontsize=9)

    for ax in axes:
        _style_axes(ax, theme)
    return axes


def plot_attitude(trajectory: Trajectory, ax=None, theme: Theme | None = None):
    """Roll, pitch and yaw against time -- only the 6-DOF model has these.

    All three share degrees, so they belong on one axis.  Yaw is unwrapped
    before plotting: a canopy in a turn passes through the +-180 degree seam
    repeatedly, and the sawtooth that produces hides the thing you are looking
    at, which is the turn rate.
    """
    import numpy as np

    theme = theme or Theme.light()
    if trajectory.attitude is None:
        raise ValueError(
            f"the {trajectory.model} model records no attitude; "
            "run with model='rigid' to get one"
        )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(8, 3))

    t = trajectory.time
    angles = np.degrees(trajectory.attitude)
    for column, colour, label in (
        (0, theme.series[0], "roll"),
        (1, theme.series[1], "pitch"),
        (2, theme.series[2], "yaw"),
    ):
        values = angles[:, column]
        if column == 2:
            values = np.degrees(np.unwrap(trajectory.attitude[:, 2]))
        ax.plot(t, values, color=colour, linewidth=2, label=label)

    ax.set_ylabel("angle, deg")
    ax.set_xlabel("time, s")
    ax.set_title("Attitude")
    ax.legend(frameon=False, labelcolor=theme.ink, fontsize=9, ncol=3,
              loc="upper center")
    _style_axes(ax, theme)
    return ax


def plot_flight(trajectory: Trajectory, target: Vec3, theme: Theme | None = None):
    """One figure combining the ground track and the time histories.

    Gains a fourth right-hand panel when the trajectory carries attitude, i.e.
    when it came from the 6-DOF model.

    Returns the matplotlib ``Figure``; call ``fig.savefig(...)`` or
    ``matplotlib.pyplot.show()`` yourself.
    """
    import matplotlib.pyplot as plt

    theme = theme or Theme.light()
    has_attitude = trajectory.attitude is not None
    rows = 4 if has_attitude else 3

    fig = plt.figure(figsize=(13, 9 if rows == 3 else 10.5), facecolor=theme.surface)
    grid = fig.add_gridspec(rows, 2, width_ratios=(1.1, 1.0), hspace=0.5, wspace=0.25)

    track_ax = fig.add_subplot(grid[:, 0])
    profile_axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]

    plot_ground_track(trajectory, target, ax=track_ax, theme=theme)
    plot_profiles(trajectory, axes=profile_axes, theme=theme)
    if has_attitude:
        plot_attitude(trajectory, ax=fig.add_subplot(grid[3, 1]), theme=theme)

    title = trajectory.report.summary()
    if trajectory.model != "point-mass":
        title = f"{title}   [{trajectory.model}]"
    fig.suptitle(title, color=theme.ink, fontsize=12)
    return fig
