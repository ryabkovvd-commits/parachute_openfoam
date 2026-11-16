"""Presentation layer.

Nothing in :mod:`parafoil` imports this package, and nothing here is needed to
run a simulation -- that separation is deliberate.  Matplotlib is only required
if you actually draw something, so it is an optional dependency::

    pip install "parafoil[plot]"
"""

from .plots import plot_attitude, plot_flight, plot_ground_track, plot_profiles

__all__ = ["plot_attitude", "plot_flight", "plot_ground_track", "plot_profiles"]
