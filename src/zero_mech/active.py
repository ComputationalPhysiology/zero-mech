from dataclasses import dataclass, field
import sympy as sp

from .atoms import AbstractStrainEnergy


@dataclass(slots=True, frozen=True)
class ActiveStress(AbstractStrainEnergy):
    """An active tension along the fibres, as a strain energy.

    The energy is :math:`\\frac{T_a}{2} (I_{4f} - 1)`, so that the second
    Piola-Kirchhoff stress it adds is :math:`T_a \\, f_0 \\otimes f_0`. For a
    fibre stretch :math:`\\lambda` the first Piola-Kirchhoff stress along the
    fibre is :math:`T_a \\lambda` and the Cauchy stress :math:`T_a \\lambda^2 / J`.

    This is the ``invariant`` formulation of ``pulse.ActiveStress``. See
    :class:`StretchActiveStress` for the ``stretch`` formulation, whose stress
    is smaller by a factor of :math:`\\lambda`.
    """

    Ta: sp.Symbol = sp.Symbol("Ta")
    f0: sp.Matrix = field(default_factory=lambda: sp.Matrix([1, 0, 0]))

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        f = F @ self.f0
        I4f = f.T.dot(f)
        return self.Ta / 2 * (I4f - 1)

    def default_parameters(self):
        return {self.Ta: 0.0}

    @staticmethod
    def str() -> str:
        return "Ta / 2 * (I4f - 1)"


@dataclass(slots=True, frozen=True)
class StretchActiveStress(AbstractStrainEnergy):
    """An active tension along the fibres that is a nominal stress, as a strain energy.

    The energy is :math:`T_a (\\lambda - 1)`, with :math:`\\lambda = \\sqrt{I_{4f}}`
    the fibre stretch, so that the second Piola-Kirchhoff stress it adds is
    :math:`\\frac{T_a}{\\lambda} f_0 \\otimes f_0` and the first Piola-Kirchhoff
    stress is :math:`T_a \\, F f_0 \\otimes f_0 / |F f_0|`. The force per unit
    reference area along the fibre is therefore :math:`T_a` at any stretch, and the
    Cauchy stress along the fibre is :math:`T_a \\lambda / J`.

    This is the ``stretch`` formulation of ``pulse.ActiveStress``, the normalisation
    of Regazzoni and Quarteroni (2021), under which the active stiffness of a
    force-generation model and its tension refer to the same stretch.
    :class:`ActiveStress` is the ``invariant`` formulation, whose stress is larger
    by a factor of :math:`\\lambda`.
    """

    Ta: sp.Symbol = sp.Symbol("Ta")
    f0: sp.Matrix = field(default_factory=lambda: sp.Matrix([1, 0, 0]))

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        f = F @ self.f0
        I4f = f.T.dot(f)
        return self.Ta * (sp.sqrt(I4f) - 1)

    def default_parameters(self):
        return {self.Ta: 0.0}

    @staticmethod
    def str() -> str:
        return "Ta * (sqrt(I4f) - 1)"


@dataclass(slots=True, frozen=True)
class ActiveStrain:
    γ: sp.Symbol = sp.Symbol("γ")
    f0: sp.Matrix = field(default_factory=lambda: sp.Matrix([1, 0, 0]))

    def Fe(self, F: sp.Matrix) -> sp.Expr:
        Fa = (1 + self.γ) * self.f0 @ self.f0.T + (1 + self.γ) ** (-0.5) * (
            sp.eye(3) - self.f0 @ self.f0.T
        )
        return F * Fa.inv()

    def default_parameters(self):
        return {self.γ: 0.0}

    @staticmethod
    def str() -> str:
        return "(1 + γ) * f0 @ f0.T + (1 + γ) ** (-0.5) * (eye(3) - f0 @ f0.T)"


@dataclass
class Passive(AbstractStrainEnergy):
    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        return sp.S.Zero

    def default_parameters(self):
        return {}

    @staticmethod
    def str() -> str:
        return "0"
