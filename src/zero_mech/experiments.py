from dataclasses import dataclass
from typing import Literal
import sympy as sp


from .atoms import Atom


def I1(C: sp.Matrix) -> sp.Expr:
    return sp.trace(C)


def I2(C: sp.Matrix) -> sp.Expr:
    return 0.5 * (I1(C) ** 2 - sp.trace(C**2))


def I3(C: sp.Matrix) -> sp.Expr:
    return C.det()


def I4a(F: sp.Matrix, a: sp.Matrix) -> sp.Expr:
    return a.T @ (F.T @ F) @ a


def I8ab(F: sp.Matrix, a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
    return a.T @ (F.T @ F) @ b


@dataclass(slots=True, frozen=True)
class MechanicalExperiment(Atom):
    F11: sp.Expr = sp.S.One
    F22: sp.Expr = sp.S.One
    F33: sp.Expr = sp.S.One
    F12: sp.Expr = sp.S.Zero
    F13: sp.Expr = sp.S.Zero
    F21: sp.Expr = sp.S.Zero
    F23: sp.Expr = sp.S.Zero
    F31: sp.Expr = sp.S.Zero
    F32: sp.Expr = sp.S.Zero

    @property
    def F(self) -> sp.Matrix:
        return sp.Matrix(
            [
                [self.F11, self.F12, self.F13],
                [self.F21, self.F22, self.F23],
                [self.F31, self.F32, self.F33],
            ]
        )

    @property
    def F_inv(self) -> sp.Matrix:
        return self.F.inv()

    @property
    def J(self) -> sp.Expr:
        return self.F.det()

    @property
    def C(self) -> sp.Matrix:
        return self.F.T * self.F

    @property
    def B(self) -> sp.Matrix:
        return self.F @ self.F.T

    @property
    def E(self) -> sp.Matrix:
        return 0.5 * (self.C - sp.eye(3))

    @property
    def I1(self) -> sp.Expr:
        return sp.trace(self.C)

    @property
    def I2(self) -> sp.Expr:
        return 0.5 * (self.I1**2 - sp.trace(self.C**2))

    @property
    def I3(self) -> sp.Expr:
        return self.C.det()

    def principal_stretches(self) -> list[sp.Expr]:
        return [sp.sqrt(s) for s in self.C.eigenvals(multiple=True)]


def simple_shear(plane: Literal["fs", "sn", "fn", "sf", "ns", "nf"] = "fs") -> MechanicalExperiment:
    gamma = sp.Symbol("gamma", real=True, postive=True)
    if plane == "fs":
        return MechanicalExperiment(F21=gamma)
    elif plane == "fn":
        return MechanicalExperiment(F31=gamma)
    elif plane == "sn":
        return MechanicalExperiment(F32=gamma)
    elif plane == "sf":
        return MechanicalExperiment(F12=gamma)
    elif plane == "nf":
        return MechanicalExperiment(F13=gamma)
    elif plane == "ns":
        return MechanicalExperiment(F23=gamma)
    else:
        raise ValueError(f"plane must be fs, sn, fn, sf, ns, or nf, got {plane}")


def uniaxial_tension(axis: int = 0) -> MechanicalExperiment:
    lmbda = sp.Symbol("λ", real=True, positive=True)

    F_tension = lmbda
    F_trans = 1.0 / sp.sqrt(lmbda)

    if axis == 0:
        F11 = F_tension
        F22 = F33 = F_trans
    elif axis == 1:
        F22 = F_tension
        F11 = F33 = F_trans
    elif axis == 2:
        F33 = F_tension
        F11 = F22 = F_trans
    else:
        raise ValueError("axis must be 0, 1, or 2")

    return MechanicalExperiment(F11=F11, F22=F22, F33=F33)


def full_matrix() -> MechanicalExperiment:
    """Return a MechanicalExperiment with all components
    initialized as symbols.
    """
    F11, F22, F33, F12, F13, F21, F23, F31, F32 = sp.symbols("F11 F22 F33 F12 F13 F21 F23 F31 F32")
    return MechanicalExperiment(
        F11=F11,
        F22=F22,
        F33=F33,
        F12=F12,
        F13=F13,
        F21=F21,
        F23=F23,
        F31=F31,
        F32=F32,
    )
