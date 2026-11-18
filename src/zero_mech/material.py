import logging
from dataclasses import dataclass, field
import sympy as sp

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


#: Below this value an exponent ``b`` of :class:`HolzapfelOgden` is taken as zero, as in pulse
B_ZERO = 1e-10


def _exponential(a: sp.Expr, b: sp.Expr, x: sp.Expr) -> sp.Expr:
    """``a / (2 b) (exp(b x) - 1)``, or its limit ``a / 2 x`` when ``b`` is (close to) zero."""
    return sp.Piecewise(
        ((a / (2 * b)) * (sp.exp(b * x) - 1), b > B_ZERO),
        (a / 2 * x, True),
    )


@dataclass(slots=True, frozen=True)
class HolzapfelOgden(AbstractStrainEnergy):
    """The Holzapfel-Ogden model.

    Each term is :math:`\\frac{a}{2 b} (e^{b x} - 1)`. Where an exponent :math:`b`
    is zero (at most :data:`B_ZERO`) the term is its limit :math:`\\frac{a}{2} x`,
    as in pulse, so that a parameter set with ``b_s = 0`` and ``a_s = 0``, such as
    pulse's ``transversely_isotropic_parameters()``, can be used as it is. A term
    whose :math:`a` is zero vanishes either way.

    ``sympy.lambdify`` with numpy evaluates both branches of the switch, so the
    energy itself, evaluated that way at :math:`b = 0`, warns about a division by
    zero in the branch it does not use. The stresses do not, since :math:`b`
    cancels in their exponential branch.
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
            _exponential(self.a, self.b, I1 - 3)
            + _exponential(self.a_f, self.b_f, I4fm1**2)
            + _exponential(self.a_s, self.b_s, I4sm1**2)
            + _exponential(self.a_fs, self.b_fs, I8fs**2)
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
