"""Aerodynamics of the ram-air canopy.

The model is a *point-mass* one: the canopy and its payload are treated as a
single body, and the whole aerodynamic effect of pulling the control lines
("клеванты", brake toggles) is expressed through three scalars that depend only
on the two toggle positions.

Sign and axis conventions for the body frame (see :mod:`parafoil.dynamics`):

* ``forward`` -- along the airspeed vector,
* ``up``      -- normal to it, in the plane of symmetry,
* ``side``    -- completing a right-handed set.

Drag acts along ``-forward``, lift along ``+up``, and the turn force along
``side``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlInput:
    """Brake toggle positions, each in ``[0, 1]``.

    ``0`` is hands-up (full glide), ``1`` is fully pulled down (full flare).
    """

    left: float = 0.0
    right: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.left <= 1.0 and 0.0 <= self.right <= 1.0):
            raise ValueError(f"toggle positions must lie in [0, 1], got {self}")

    @property
    def symmetric(self) -> float:
        """Mean deflection -- controls glide ratio and speed."""
        return 0.5 * (self.left + self.right)

    @property
    def differential(self) -> float:
        """``right - left``; positive means a right turn is commanded."""
        return self.right - self.left

    def clamped(self) -> "ControlInput":
        """Same input with both toggles forced into ``[0, 1]``."""
        return ControlInput(
            left=min(1.0, max(0.0, self.left)),
            right=min(1.0, max(0.0, self.right)),
        )


@dataclass(frozen=True, slots=True)
class AeroCoefficients:
    """Dimensionless force coefficients in the body frame."""

    drag: float  # along -forward (already negative)
    lift: float  # along +up
    side: float  # along +side

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.drag**2 + self.lift**2 + self.side**2)


@dataclass(frozen=True, slots=True)
class CanopyParams:
    """Physical and aerodynamic properties of the parachute system.

    Defaults reproduce the cargo parafoil modelled by the original program: a
    100 m^2 canopy carrying a 1880 kg load, glide ratio tunable between 2 and 4.
    """

    area: float = 100.0             # S, reference area [m^2]
    canopy_mass: float = 23.3       # m_p [kg]
    payload_mass: float = 1880.0    # m_l [kg]

    glide_ratio_max: float = 4.0    # k at hands-up
    glide_ratio_min: float = 2.0    # k at full symmetric brake
    force_coeff_min: float = 0.6    # C at hands-up
    force_coeff_max: float = 1.3    # C at full brake

    turn_gain: float = 0.0055       # side-force coefficient per unit of
                                    # differential brake

    def __post_init__(self) -> None:
        if self.area <= 0:
            raise ValueError("reference area must be positive")
        if self.total_mass <= 0:
            raise ValueError("total mass must be positive")
        if self.glide_ratio_min > self.glide_ratio_max:
            raise ValueError("glide_ratio_min must not exceed glide_ratio_max")

    @property
    def total_mass(self) -> float:
        return self.canopy_mass + self.payload_mass

    # ---- the model ------------------------------------------------------

    def glide_ratio(self, control: ControlInput) -> float:
        """Lift-to-drag ratio ``k``.

        Falls off quadratically with symmetric brake: hands-up gives the best
        glide, full flare the steepest descent.
        """
        d = control.symmetric
        return self.glide_ratio_max - (self.glide_ratio_max - self.glide_ratio_min) * d * d

    def force_coefficient(self, control: ControlInput) -> float:
        """Total force coefficient ``C``, growing with brake deflection.

        The effective deflection blends the two toggles as
        ``sqrt(|l*r + (r-l)^2 / 32|)``.  For symmetric input it reduces to the
        common deflection; the small asymmetric term keeps ``C`` from
        collapsing during a hard one-sided turn.
        """
        l, r = control.left, control.right
        effective = math.sqrt(abs(l * r + (r - l) ** 2 / 32.0))
        spread = self.force_coeff_max - self.force_coeff_min
        return self.force_coeff_max - spread * (effective - 1.0) ** 2

    def coefficients(self, control: ControlInput) -> AeroCoefficients:
        """Split the total coefficient into drag, lift and side components.

        The total aerodynamic coefficient has magnitude ``C`` and is shared
        between drag and lift so that ``lift / |drag| == k``::

            drag = -C / sqrt(1 + k^2)
            lift = +C * k / sqrt(1 + k^2)

        The side component is proportional to the differential brake and is
        what makes the canopy turn.
        """
        k = self.glide_ratio(control)
        c = self.force_coefficient(control)
        scale = c / math.sqrt(1.0 + k * k)
        return AeroCoefficients(
            drag=-scale,
            lift=k * scale,
            side=self.turn_gain * control.differential,
        )

    def dynamic_pressure_force(self, air_density: float, airspeed: float) -> float:
        """``rho * S * V^2 / 2`` -- the scale factor turning coefficients into newtons."""
        return air_density * self.area * airspeed * airspeed / 2.0

    # ---- derived performance figures (useful for planning and for the UI) --

    def trim_airspeed(self, control: ControlInput, air_density: float,
                      gravity: float = 9.81) -> float:
        """Steady-state airspeed at which lift+drag balance weight.

        In equilibrium the resultant aerodynamic force equals the weight, so
        ``C * rho * S * V^2 / 2 == m * g``.
        """
        c = self.force_coefficient(control)
        return math.sqrt(2.0 * self.total_mass * gravity / (c * air_density * self.area))

    def trim_speeds(self, control: ControlInput, air_density: float,
                    gravity: float = 9.81) -> tuple[float, float]:
        """``(horizontal, vertical)`` components of the steady-state airspeed."""
        v = self.trim_airspeed(control, air_density, gravity)
        k = self.glide_ratio(control)
        vertical = v / math.sqrt(1.0 + k * k)
        return k * vertical, vertical
