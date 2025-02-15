from abc import abstractmethod
from dataclasses import dataclass
import sympy as sp


from .atoms import AbstractStrainEnergy


class Compressibility(AbstractStrainEnergy):
    @abstractmethod
    def is_compressible(self) -> bool: ...


@dataclass(slots=True, frozen=True)
class Incompressible(Compressibility):
    p: float = sp.Symbol("p")

    def is_compressible(self) -> bool:
        return False

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        J = F.det()
        return self.p * (J - 1)

    def default_parameters(self):
        return {}

    @staticmethod
    def str() -> str:
        return "p * (J - 1)"


@dataclass(slots=True, frozen=True)
class Compressible1(Compressibility):
    kappa = sp.Symbol("kappa")

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        J = F.det()
        return self.kappa / 2 * (J - 1) ** 2

    def is_compressible(self) -> bool:
        return True

    def default_parameters(self):
        return {self.kappa: 1e3}

    @staticmethod
    def str() -> str:
        return "kappa / 2 * (J - 1) ** 2"
