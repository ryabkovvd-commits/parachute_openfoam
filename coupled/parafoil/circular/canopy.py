"""Parameters of a round canopy and the command that steers it.

Everything here is for a *drag* device.  There is no lift curve, no glide
ratio worth the name and no angle of attack: a flat-circular parachute
descends at its terminal velocity and drifts.  The only aerodynamic number
that matters is the drag area, and the only control effect is a small tilt of
that drag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

#: The four risers, in canopy-fixed order.  Riser 0 is the one the compass
#: reference is rigged to; the rest follow anticlockwise seen from above.
RISER_COUNT = 4


class RiserCommand(NamedTuple):
    """Which risers are released, as four booleans.

    Releasing a riser lengthens it, the canopy deforms, and the system is
    driven **away** from the released riser.  That sign is the one the AGAS
    papers describe ("creating drive in the opposite direction of the control
    action") and it is easy to get backwards, so it is asserted by a test.

    One or two adjacent risers may be released at once, giving eight
    achievable drive directions at 45 degree spacing.  Releasing opposite
    risers cancels out and is treated as no command.
    """

    a: bool = False
    b: bool = False
    c: bool = False
    d: bool = False

    @property
    def active(self) -> int:
        return sum(self)

    def drive_vector(self, heading: float) -> tuple[float, float]:
        """Drive in world axes (east, south), in units of one riser's worth.

        Args:
            heading: Canopy rotation angle in radians.  Riser 0 points along
                this heading; the canopy turns, so the same riser drives in a
                different world direction as the flight goes on.

        The contributions are summed and **not** normalised, which matters:

        * one riser gives magnitude 1;
        * two adjacent risers give ``sqrt(2)``, not 1 -- half the canopy is
          deformed instead of a quarter, so the drive really is larger.  This
          matches the published AGAS control model, where the two lateral
          axes take values in ``{-V, 0, +V}`` independently and the diagonal
          therefore reaches ``V*sqrt(2)``;
        * two opposite risers cancel to zero, as they do physically.
        """
        east = south = 0.0
        for index, released in enumerate(self):
            if not released:
                continue
            angle = heading + index * (math.pi / 2.0)
            # Drive is away from the released riser, hence the minus.
            east -= math.cos(angle)
            south -= math.sin(angle)

        if math.hypot(east, south) < 1e-9:
            return 0.0, 0.0
        return east, south

    @classmethod
    def none(cls) -> "RiserCommand":
        return cls()


@dataclass(frozen=True, slots=True)
class RoundCanopy:
    """A flat-circular parachute.

    Defaults are the G-12, the canopy the AGAS programme actually flew.  From
    the manufacturer's data: 64 ft nominal diameter, 64 gores, 64 suspension
    lines of 51 ft in Type IV PIA-C-7515 cord, canopy of 1.1 oz PIA-C-7020
    Type I ripstop nylon, payload to 2200 lb.  Descends at about 9 m/s.

    Attributes:
        diameter: Nominal (constructed) diameter, m.  64 ft = 19.51 m.
        drag_coefficient: Referred to the nominal canopy area.  0.699 is what
            the published descent rate implies at sea level, and sits in the
            usual 0.6-0.8 band for flat-circular canopies.  It is not an
            independent measurement: only the product ``C_D * S`` is pinned,
            by the descent rate.
        canopy_mass, payload_mass: kg.
        glide_by_risers: Glide ratio with 0, 1 and 2 risers released,
            **measured in AGAS flight test**.  Pulling a riser deforms the
            canopy, and a deformed canopy is aerodynamically a different
            thing -- so the control does not add a fixed force, it selects
            between measured aerodynamic states.  See
            :meth:`glide_ratio_at`.
        rotation_rate: How fast the canopy turns about the vertical, rad/s.
            Measured at 1.8 deg/s mean for the G-12 in flight test, with a
            standard deviation of 1.0 deg/s.  This is not a control input; it
            just happens, and the guidance has to live with it.
        added_mass_factor: Apparent mass, as a fraction of the mass of a
            sphere of air of the canopy diameter.  Affects the transient after
            release, not the steady descent.
    """

    diameter: float = 19.507        # 64 ft
    drag_coefficient: float = 0.699
    canopy_mass: float = 60.0
    payload_mass: float = 1000.0
    glide_by_risers: tuple[float, float, float] = (0.0, 0.37, 0.52)
    rotation_rate: float = math.radians(1.8)
    added_mass_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.diameter <= 0.0:
            raise ValueError("diameter must be positive")
        if self.drag_coefficient <= 0.0:
            raise ValueError("drag coefficient must be positive")
        if self.total_mass <= 0.0:
            raise ValueError("total mass must be positive")
        if len(self.glide_by_risers) != 3:
            raise ValueError("glide_by_risers needs entries for 0, 1 and 2 risers")
        if any(g < 0.0 for g in self.glide_by_risers):
            raise ValueError("glide ratios cannot be negative")

    @property
    def area(self) -> float:
        """Nominal canopy area, m^2."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def drag_area(self) -> float:
        """``C_D * S`` -- the only aerodynamic number this model needs."""
        return self.drag_coefficient * self.area

    @property
    def total_mass(self) -> float:
        return self.canopy_mass + self.payload_mass

    def added_mass(self, air_density: float = 1.23) -> float:
        """Apparent mass of the air the canopy drags along, kg.

        Estimated as a fraction of the air in a sphere of the canopy
        diameter.  For a parachute this is genuinely large -- of the order of
        the payload -- but it acts only when the system accelerates, so it
        barely touches a steady descent.  It is kept as an explicit parameter
        rather than buried, because it is an estimate.
        """
        sphere = math.pi * self.diameter**3 / 6.0
        return self.added_mass_factor * air_density * sphere

    # ---- derived performance -------------------------------------------

    def descent_rate(self, air_density: float = 1.23, gravity: float = 9.81) -> float:
        """Steady descent rate in still air, m/s.

        From ``0.5 * rho * C_D * S * V^2 = m * g``.
        """
        return math.sqrt(
            2.0 * self.total_mass * gravity / (air_density * self.drag_area)
        )

    def glide_ratio_at(self, drive_magnitude: float) -> float:
        """Glide ratio for a given net drive, interpolated on the measured table.

        The magnitude is in units of one riser, so 0, 1 and ``sqrt(2)``
        correspond to the three columns of the AGAS flight-test table
        (0 risers, 1 riser, 2 adjacent risers).  Values in between are
        interpolated; beyond ``sqrt(2)`` the table is held, since two adjacent
        risers is the most the hardware can usefully do.

        Two things worth noting about that table:

        * the glide ratio does scale as roughly ``sqrt(2)`` between one riser
          and two -- 0.52 against 0.37 is a factor of 1.405 -- which is what
          you would expect from deforming half the canopy instead of a
          quarter;
        * the measured descent rates (8.43, 8.16, 7.93 m/s) follow from these
          glide ratios alone, through ``Vz(n) = Vz(0)*(1+GR^2)^(-1/4)``, to
          within 0.13%.  So the drag *area* does not change -- only the tilt
          of the resultant does.  That is a real constraint the table
          confirms, not an assumption made here.
        """
        knots = (0.0, 1.0, math.sqrt(2.0))
        values = self.glide_by_risers
        if drive_magnitude <= knots[0]:
            return values[0]
        if drive_magnitude >= knots[2]:
            return values[2]
        i = 1 if drive_magnitude <= knots[1] else 2
        span = knots[i] - knots[i - 1]
        fraction = (drive_magnitude - knots[i - 1]) / span
        return values[i - 1] + fraction * (values[i] - values[i - 1])

    def drive_ratio_at(self, drive_magnitude: float) -> float:
        """Horizontal drive as a fraction of the drag, for a given net drive.

        Derived from the glide ratio rather than tabulated separately.  In
        steady flight the horizontal drive balances the horizontal component
        of the drag, which gives ``r = GR / sqrt(1 + GR^2)`` -- so a glide
        ratio of 0.37 needs a drive fraction of 0.347, not 0.37.
        """
        glide = self.glide_ratio_at(drive_magnitude)
        return glide / math.sqrt(1.0 + glide * glide)

    def drive_speed(
        self, risers: int = 1, air_density: float = 1.23, gravity: float = 9.81
    ) -> float:
        """Steady horizontal speed with ``risers`` released, m/s.

        One riser gives about 3 m/s for a G-12 and two give about 4 -- the
        published "roughly 4 m/s of drive" refers to the two-riser case, which
        is easy to misread.
        """
        magnitude = {0: 0.0, 1: 1.0}.get(risers, math.sqrt(2.0))
        glide = self.glide_ratio_at(magnitude)
        return glide * self.descent_rate(air_density, gravity) / (
            1.0 + glide * glide
        ) ** 0.25

    def glide_ratio(self, risers: int = 2) -> float:
        """Best available glide ratio, i.e. with two adjacent risers released.

        0.52 for a G-12, against 4.0 for the ram-air canopy modelled in the
        rest of this package -- a factor of nearly eight, and the reason one
        can be flown to a target and the other cannot.
        """
        magnitude = {0: 0.0, 1: 1.0}.get(risers, math.sqrt(2.0))
        return self.glide_ratio_at(magnitude)
