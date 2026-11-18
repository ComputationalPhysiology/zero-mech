import logging
from dataclasses import dataclass, field
import sympy as sp
from mpmath.libmp import prec_to_dps
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.pycode import MpmathPrinter

from .atoms import AbstractStrainEnergy

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class NeoHookean(AbstractStrainEnergy):
    mu: sp.Symbol = sp.Symbol("mu")

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        C = F.T @ F
        I1 = sp.trace(C)
        return self.mu / 2 * (I1 - 3)

    def default_parameters(self):
        return {self.mu: 15.0}

    @staticmethod
    def str() -> str:
        return "μ / 2 * (I1 - 3)"


# class LinearElastic(AbstractStrainEnergy):
#     mu: sp.Symbol = sp.Symbol("mu")
#     lmbda: sp.Symbol = sp.Symbol("lmbda")

#     def strain_energy(self, F: sp.Matrix) -> sp.Expr:
#         C = F.T @ F
#         I1 = sp.trace(C)


#: Below this magnitude an exponent ``b`` of :class:`HolzapfelOgden` is taken as zero, as in pulse
B_ZERO = 1e-10


class expm1_div(sp.Function):
    """``(exp(b x) - 1) / b``, continued by its limit ``x`` where ``|b| <= B_ZERO``.

    Called as ``expm1_div(b, x)``. A function rather than a ``Piecewise`` in ``b``, so
    that expressions stay free of conditions on the parameters: ``sympy.solve`` cannot
    convert a solution whose ``Piecewise`` conditions combine into an ``And`` to floats.
    With a number for ``b`` it evaluates to ``x`` or to the exponential expression. Its
    derivative in ``x`` is ``exp(b x)``, regular at ``b = 0``, and it lambdifies to
    ``expm1`` with a switch that does not divide by zero.
    """

    nargs = 2

    @classmethod
    def eval(cls, b, x):
        if b.is_Number:
            if abs(b) <= B_ZERO:
                return x
            return (sp.exp(b * x) - 1) / b
        return None

    def fdiff(self, argindex=1):
        b, x = self.args
        if argindex == 1:
            return (x * sp.exp(b * x) - self) / b
        if argindex == 2:
            return sp.exp(b * x)
        raise sp.ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return self.func(*(arg.evalf(dps) for arg in self.args))

    def _pythoncode(self, printer):
        b, x = (printer._print(arg) for arg in self.args)
        if isinstance(printer, NumPyPrinter):
            np = printer._module
            where = printer._module_format(f"{np}.where")
            abs_ = printer._module_format(f"{np}.abs")
            expm1 = printer._module_format(f"{np}.expm1")
            nonzero = f"{abs_}({b}) > {B_ZERO!r}"
            safe_b = f"{where}({nonzero}, {b}, 1.0)"
            return f"{where}({nonzero}, {expm1}(({b})*({x}))/{safe_b}, {x})"
        module = "mpmath" if isinstance(printer, MpmathPrinter) else "math"
        expm1 = printer._module_format(f"{module}.expm1")
        return f"(({expm1}(({b})*({x}))/({b})) if abs({b}) > {B_ZERO!r} else ({x}))"

    # The name of the printing method differs between the printers that lambdify uses
    _numpycode = _cupycode = _jaxcode = _lambdacode = _mpmathcode = _pythoncode


@dataclass(slots=True, frozen=True)
class HolzapfelOgden(AbstractStrainEnergy):
    """The Holzapfel-Ogden model.

    Each term is :math:`\\frac{a}{2 b} (e^{b x} - 1)`. Where an exponent :math:`b`
    is zero (:math:`|b|` at most :data:`B_ZERO`) the term is its limit
    :math:`\\frac{a}{2} x`, as in pulse, so that a parameter set with ``b_s = 0`` and
    ``a_s = 0``, such as pulse's ``transversely_isotropic_parameters()``, can be used
    as it is. A term whose :math:`a` is zero vanishes either way. The terms are
    written with :class:`expm1_div`, which evaluates to the familiar expression once
    :math:`b` is a number.
    """

    a: sp.Symbol = sp.Symbol("a")
    b: sp.Symbol = sp.Symbol("b")
    a_f: sp.Symbol = sp.Symbol("a_f")
    b_f: sp.Symbol = sp.Symbol("b_f")
    a_s: sp.Symbol = sp.Symbol("a_s")
    b_s: sp.Symbol = sp.Symbol("b_s")
    a_fs: sp.Symbol = sp.Symbol("a_fs")
    b_fs: sp.Symbol = sp.Symbol("b_fs")
    f0: sp.Matrix = field(default_factory=lambda: sp.Matrix([1, 0, 0]))
    s0: sp.Matrix = field(default_factory=lambda: sp.Matrix([0, 1, 0]))
    use_heaviside: bool = False

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        C = F.T @ F

        I1 = sp.trace(C)
        logger.debug(f"I1: {I1}")
        I4f = self.f0.dot(C * self.f0)
        logger.debug(f"I4f: {I4f}")
        I4s = self.s0.dot(C * self.s0)
        logger.debug(f"I4s: {I4s}")
        I8fs = self.f0.dot(C * self.s0)
        logger.debug(f"I8fs: {I8fs}")

        I4fm1 = I4f - 1 if not self.use_heaviside else sp.Piecewise((0.0, I4f < 1), (I4f - 1, True))
        I4sm1 = I4s - 1 if not self.use_heaviside else sp.Piecewise((0.0, I4s < 1), (I4s - 1, True))

        return (
            self.a / 2 * expm1_div(self.b, I1 - 3)
            + self.a_f / 2 * expm1_div(self.b_f, I4fm1**2)
            + self.a_s / 2 * expm1_div(self.b_s, I4sm1**2)
            + self.a_fs / 2 * expm1_div(self.b_fs, I8fs**2)
        )

    def default_parameters(self):
        return {
            self.a: 0.059,
            self.b: 8.023,
            self.a_f: 18.472,
            self.b_f: 16.026,
            self.a_s: 2.481,
            self.b_s: 11.120,
            self.a_fs: 0.216,
            self.b_fs: 11.436,
        }

    @staticmethod
    def str() -> str:
        return (
            "(a / (2 * b)) * (exp(b * (I1 - 3)) - 1) + "
            "(a_f / (2 * b_f)) * (exp(b_f * H(I4f - 1)**2) - 1) + "
            "(a_s / (2 * b_s)) * (exp(b_s * H(I4s - 1)**2) - 1) + "
            "(a_fs / (2 * b_fs)) * (exp(b_fs * I8fs**2) - 1)"
        )
