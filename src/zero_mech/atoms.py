from typing import Any, Sequence
from abc import ABC, abstractmethod
import sympy as sp


class Atom(ABC):
    def variables(self) -> dict[str, sp.Symbol]:
        d = {}
        if hasattr(self, "__slots__"):
            slots: Sequence[str] = self.__slots__
            items: list[tuple[str, Any]] = [(k, getattr(self, k)) for k in slots]
        else:
            items = list(self.__dict__.items())

        for k, v in items:
            if isinstance(v, sp.Symbol):
                d[v.name] = v

            if isinstance(v, Atom):
                d.update(v.variables())
        return d

    def __getitem__(self, key):
        return self.variables()[key]

    def __contains__(self, key):
        return key in self.variables()


class AbstractStrainEnergy(Atom, ABC):
    @abstractmethod
    def strain_energy(self, F: sp.Matrix) -> sp.Expr: ...

    @abstractmethod
    def default_parameters(self): ...

    @staticmethod
    @abstractmethod
    def str() -> str: ...
