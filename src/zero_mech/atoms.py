from abc import ABC, abstractmethod
import sympy as sp


class Atom(ABC):
    def variables(self) -> dict[str, sp.Symbol]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, sp.Symbol):
                d[v.name] = v

            if isinstance(v, Atom):
                d.update(v.variables())
        return d

    def __getitem__(self, key):
        return self.variables()[key]


class AbstractStrainEnergy(Atom, ABC):
    @abstractmethod
    def strain_energy(self, F: sp.Matrix) -> sp.Expr: ...

    @abstractmethod
    def default_parameters(self): ...
