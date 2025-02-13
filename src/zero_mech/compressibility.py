from abc import abstractmethod
from dataclasses import dataclass
import sympy as sp


from .atoms import AbstractStrainEnergy


class Compressibility(AbstractStrainEnergy):
    @abstractmethod
    def is_compressible(self) -> bool: ...


@dataclass
class Incompressible(Compressibility):
    p: float = sp.Symbol("p")

    def is_compressible(self) -> bool:
        return False

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        J = F.det()
        return self.p * (J - 1)

    def default_parameters(self):
        return {}


@dataclass
class Compressible1(Compressibility):
    kappa = sp.Symbol("kappa")

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        J = F.det()
        return self.kappa / 2 * (J - 1) ** 2

    def default_parameters(self):
        return {self.kappa: 1e3}
