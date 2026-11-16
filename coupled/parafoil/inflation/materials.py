"""Material properties of the rods the canopy is discretised into.

Leonov's method needs one number per element: a **linear stiffness**, the
force per unit relative elongation, in newtons.  Not a Young's modulus and not
a stress-strain curve -- the rods stand in for strips of fabric or lengths of
webbing, so what matters is how hard that strip pulls back when stretched by a
given fraction of its length.

That is a much lower data requirement than a nonlinear fabric model, and it is
deliberate: it is the difference between needing a laboratory campaign and
needing a specification sheet.

Every material here also carries a **breaking load**, because the point of
computing the stress-strain state is usually to find out whether anything is
close to letting go.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RodMaterial:
    """What a single discretised rod is made of.

    Attributes:
        stiffness: Force at 100% relative elongation, N.  A rod stretched by
            a fraction ``e`` of its initial length pulls back with
            ``stiffness * e``.
        breaking_load: N.  Used for safety factors, never for the shape.
        linear_density: kg/m, for the nodal masses and the Courant step.
    """

    stiffness: float
    breaking_load: float
    linear_density: float

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError("stiffness must be positive")
        if self.breaking_load <= 0.0:
            raise ValueError("breaking load must be positive")
        if self.linear_density < 0.0:
            raise ValueError("linear density cannot be negative")

    def tension(self, initial_length: float, current_length: float) -> float:
        """Tension in a rod, N -- **zero if it is not stretched**.

        This one-sided law is the whole reason the method suits parachutes.
        Fabric carries no compression: a panel that would be squashed goes
        slack and wrinkles instead, and setting the tension to zero is exactly
        that behaviour.  Leonov writes it as condition (2.11).
        """
        if initial_length <= 0.0:
            return 0.0
        elongation = current_length / initial_length - 1.0
        return self.stiffness * elongation if elongation > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Fabric:
    """Canopy cloth, described per unit width.

    Attributes:
        stiffness_per_width: N per unit relative elongation, per metre of
            width.  A strip ``w`` metres wide gives a rod of stiffness
            ``stiffness_per_width * w``.
        breaking_load_per_width: N/m.  For PIA-C-7020 Type I this is the
            42 lbf/inch of the specification, i.e. about 7.4 kN/m.
        areal_density: kg/m^2.  Type I is 1.1 oz/yd^2 = 0.0373 kg/m^2.
        shear_ratio: Shear stiffness as a fraction of the tensile stiffness,
            for the panel diagonals.  A woven cloth resists shear one to two
            orders of magnitude more weakly than it resists stretching -- that
            is why fabric drapes at all -- so this is a small number, and
            giving a diagonal the full strip stiffness would make the panel an
            order of magnitude stiffer than real cloth.  Like
            ``stiffness_per_width`` it is a placeholder to be replaced rather
            than trusted, and its influence is meant to be measured.
    """

    stiffness_per_width: float
    breaking_load_per_width: float
    areal_density: float
    shear_ratio: float = 0.02

    def strip(self, width: float) -> RodMaterial:
        """The rod standing in for a strip of this cloth ``width`` metres wide."""
        return RodMaterial(
            stiffness=self.stiffness_per_width * width,
            breaking_load=self.breaking_load_per_width * width,
            linear_density=self.areal_density * width,
        )

    def diagonal(self, width: float) -> RodMaterial:
        """A panel diagonal, carrying shear only.

        Weightless on purpose: the cloth's mass is already handed to the nodes
        by panel area, and giving the diagonals mass as well would quietly make
        the canopy heavier than its own specification.
        """
        return RodMaterial(
            stiffness=self.stiffness_per_width * width * self.shear_ratio,
            breaking_load=self.breaking_load_per_width * width,
            linear_density=0.0,
        )

    @classmethod
    def ripstop_type_i(cls, stiffness_per_width: float = 60_000.0) -> "Fabric":
        """1.1 oz ripstop to PIA-C-7020 Type I -- the G-12 canopy cloth.

        Breaking load and weight are from the specification.  The **stiffness
        is not**: specifications state an acceptance minimum at break, one
        point, and say nothing about the slope on the way there.  60 kN/m per
        unit strain is an order-of-magnitude placeholder, equivalent to about
        8% elongation at the breaking load, which is the right ballpark for a
        woven nylon but is a number to be replaced, not trusted.

        Everything that depends on it is reported by
        :func:`~parafoil.inflation.relax.inflate` so its influence can be
        seen rather than assumed.
        """
        return cls(
            stiffness_per_width=stiffness_per_width,
            breaking_load_per_width=7_355.0,   # 42 lbf/in
            areal_density=0.0373,              # 1.1 oz/yd^2
        )


@dataclass(frozen=True, slots=True)
class Webbing:
    """Reinforcing band or suspension cord -- a discrete structural member.

    Named after the Russian convention in the source paper, where the radial
    and annular framework is built from ``LTKP`` bands and the lines from
    ``ShKP`` cord.
    """

    stiffness: float
    breaking_load: float
    linear_density: float

    def rod(self) -> RodMaterial:
        return RodMaterial(self.stiffness, self.breaking_load, self.linear_density)

    @classmethod
    def suspension_cord(cls, breaking_load: float = 4_450.0) -> "Webbing":
        """A suspension line.  4450 N is the 1000 lbf of the G-12's Type IV cord.

        Stiffness taken as 25 times the breaking load, i.e. about 4%
        elongation at failure, which is typical for a braided nylon cord and
        is stiff enough that the lines are nearly inextensible next to the
        fabric -- which is the behaviour that matters here.
        """
        return cls(
            stiffness=25.0 * breaking_load,
            breaking_load=breaking_load,
            linear_density=0.004,
        )

    @classmethod
    def radial_band(cls, breaking_load: float = 6_670.0) -> "Webbing":
        """A radial reinforcing band, roughly an LTKP-25-150."""
        return cls(
            stiffness=25.0 * breaking_load,
            breaking_load=breaking_load,
            linear_density=0.010,
        )
