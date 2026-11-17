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
