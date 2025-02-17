from dataclasses import dataclass, field
from typing import Protocol
import sympy as sp


from .experiments import full_matrix
from .atoms import Atom

from .active import Passive
from .compressibility import Incompressible
from .material import NeoHookean


class StrainEnergy(Protocol):
    def strain_energy(self, F: sp.Matrix) -> sp.Expr: ...


class Compressibility(StrainEnergy, Protocol):
    def is_compressible(self) -> bool: ...


@dataclass(slots=True, frozen=True)
class Model(Atom):
    material: StrainEnergy = field(default_factory=NeoHookean)
    compressibility: Compressibility = field(default_factory=Incompressible)
    active: StrainEnergy = field(default_factory=Passive)

    _full: sp.Matrix = field(default_factory=full_matrix, init=False, repr=False)

    def _subs(self, A, F):
        return A.subs(
            {
                self._full.F[0, 0]: F[0, 0],
                self._full.F[1, 1]: F[1, 1],
                self._full.F[2, 2]: F[2, 2],
                self._full.F[0, 1]: F[0, 1],
                self._full.F[0, 2]: F[0, 2],
                self._full.F[1, 0]: F[1, 0],
                self._full.F[1, 2]: F[1, 2],
                self._full.F[2, 0]: F[2, 0],
                self._full.F[2, 1]: F[2, 1],
            }
        )

    @property
    def _strain_energy(self):
        F = self._full.F
        if self.compressibility.is_compressible():
            J = F.det()
            F_dev = F * J ** (-1 / 3)
        else:
            F_dev = F
        return (
            self.material.strain_energy(F_dev)
            + self.compressibility.strain_energy(F)
            + self.active.strain_energy(F)
        )

    def strain_energy(self, F):
        return self._subs(self._strain_energy, F)

    def first_piola_kirchhoff(self, F):
        return self._subs(sp.diff(self._strain_energy, self._full.F), F)

    def cauchy_stress(self, F):
        J = F.det()
        P = self.first_piola_kirchhoff(F)
        return P @ F.T / J

    def second_piola_kirchhoff(self, F):
        P = self.first_piola_kirchhoff(F)
        return F.inv() @ P
