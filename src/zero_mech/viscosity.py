from dataclasses import dataclass
import sympy as sp


@dataclass
class KelvinVoigt:
    η: sp.Symbol = sp.Symbol("η")

    def viscosity(self, F: sp.Matrix) -> sp.Expr:
        return self.η * F.norm()
