"""Thick walls: a cylinder and a sphere, solved exactly.

The ventricles of :mod:`zero_mech.ventricle` reduce the wall to a single stretch,
an approximation whose error grows with the thickness of the wall. The classes
here do not. For an incompressible wall whose deformation keeps the symmetry of
its geometry, the deformation is known in closed form, and radial equilibrium
gives the cavity pressure as one integral through the wall. That is the solution
of the full three-dimensional problem, so it can serve as a reference for a finite
element solver, for instance on a mesh of a quarter of an annulus or an eighth of
a sphere. The integral is computed by Gauss-Legendre quadrature in the reference
radius.

Cylinder
--------

An incompressible tube in plane strain (no axial stretch), with inner and outer
radii :math:`R_i` and :math:`R_o` when unloaded. Each cross-section stays
circular, so a point at radius :math:`R` moves to

.. math::
    r(R)^2 = r_i^2 + R^2 - R_i^2, \\qquad \\lambda = \\frac{r}{R},

with :math:`\\lambda` the circumferential stretch, :math:`1/\\lambda` the radial
one and 1 the axial one. With :math:`\\hat\\psi(\\lambda)` the strain energy at
that deformation, :math:`\\sigma_{\\theta\\theta} - \\sigma_{rr} = \\lambda \\,
d\\hat\\psi/d\\lambda`, and radial equilibrium with a traction-free outer surface
gives the inner pressure

.. math::
    p = \\int_{r_i}^{r_o} (\\sigma_{\\theta\\theta} - \\sigma_{rr}) \\frac{dr}{r}
      = \\int_{R_i}^{R_o} \\lambda \\frac{d\\hat\\psi}{d\\lambda} \\frac{R}{r^2} \\, dR.

This is exact when the stress is diagonal in the radial, circumferential and
axial directions: an isotropic material, or one whose fibre and sheet directions
lie along those directions. Circumferential fibres may carry an active stress, so
the tube can contract, with an active tension that varies through the wall.

Sphere
------

A point at radius :math:`R` moves to :math:`r(R)^3 = r_i^3 + R^3 - R_i^3`, with
the stretch :math:`\\lambda = r / R` in both directions of the wall plane and
:math:`\\lambda^{-2}` through it (:func:`zero_mech.experiments.equibiaxial`).
With :math:`\\hat\\psi(\\lambda)` the strain energy at that deformation,
:math:`\\sigma_{\\theta\\theta} + \\sigma_{\\phi\\phi} - 2 \\sigma_{rr} = \\lambda
\\, d\\hat\\psi/d\\lambda`, and

.. math::
    p = \\int_{r_i}^{r_o} (\\sigma_{\\theta\\theta} + \\sigma_{\\phi\\phi} - 2 \\sigma_{rr})
        \\frac{dr}{r}
      = \\int_{R_i}^{R_o} \\lambda \\frac{d\\hat\\psi}{d\\lambda} \\frac{R^2}{r^3} \\, dR.

The deformation is spherically symmetric only if the material is isotropic in the
wall plane (:math:`\\sigma_{\\theta\\theta} = \\sigma_{\\phi\\phi}`). The sphere
therefore takes no fibres, sheets or active stress along a direction of the wall
plane: no single direction is tangent to a sphere everywhere. It is a reference
for a passive wall that is isotropic in its plane.

Common to both
--------------

The argument of the methods is the cavity volume: :math:`\\frac{4}{3} \\pi r_i^3`
for the sphere and, for the cylinder, the volume per unit length,
:math:`\\pi r_i^2`. Lengths can be in any unit, and pressures come out in the unit
of the material parameters.

A parameter of the model can be given one value per quadrature point (an array
shaped like :attr:`ThickCylinder.R`), to vary through the wall. An active tension
computed from the stretch :meth:`ThickCylinder.stretch` of each point is one
example.

The quadrature converges exponentially for a smooth integrand, and the default
number of points then gives round-off accuracy. The integrand has a kink where a
stretch crosses a switch of the material inside the wall, such as the heaviside
function of :class:`zero_mech.material.HolzapfelOgden` with
``use_heaviside=True``. The convergence is then algebraic, so check the result
with more points.

References
----------
Ogden RW (1984). Non-linear Elastic Deformations. Ellis Horwood, Chichester.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from .model import Model
from .ventricle import _Ventricle

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]

__all__ = ["ThickCylinder", "ThickSphere"]

#: Stretches at which the symmetry of the stress is checked
_CHECK_STRETCHES = (0.8, 0.95, 1.05, 1.25)


class _ThickWall(_Ventricle):
    """What the cylinder and the sphere share. ``_n`` is 2 for the cylinder and 3 for the
    sphere: the cavity volume is ``_c * r_i**_n``, and ``r**_n - R**_n`` is the same at
    every point of the wall."""

    _n: int
    _c: float

    def _setup(
        self,
        model: Model | None,
        Ri: float,
        Ro: float,
        diagonal: Callable[[sp.Symbol], Sequence[sp.Expr]],
        parameters: Mapping[Any, Any] | None,
        num_points: int,
    ) -> None:
        if _np is None:  # pragma: no cover
            raise ImportError("The thick walls need numpy")
        if not 0 < Ri < Ro:
            raise ValueError(f"The radii must satisfy 0 < Ri < Ro, got Ri = {Ri} and Ro = {Ro}")
        if int(num_points) < 1:
            raise ValueError(f"num_points must be positive, got {num_points}")
        self.model = model if model is not None else Model()
        if self.model.compressibility.is_compressible():
            raise ValueError("The wall must be incompressible")

        self.Ri = float(Ri)
        self.Ro = float(Ro)
        x, w = _np.polynomial.legendre.leggauss(int(num_points))
        #: Reference radii of the quadrature points through the wall
        self.R = 0.5 * (self.Ro - self.Ri) * x + 0.5 * (self.Ro + self.Ri)
        #: Quadrature weights of the points :attr:`R`
        self.weights = 0.5 * (self.Ro - self.Ri) * w
        #: Cavity volume of the unloaded wall
        self.V0 = self.volume(self.Ri)

        n = self._n
        lmbda = sp.Symbol("λ", real=True, positive=True)
        F = sp.diag(*diagonal(lmbda))
        # lmbda dpsi/dlmbda: sigma_tt - sigma_rr (cylinder), sigma_tt + sigma_pp - 2 sigma_rr
        # (sphere). The pressure of an incompressible model drops out, since J = 1.
        stress = lmbda * sp.diff(self.model.strain_energy(F), lmbda)
        self._R = sp.Symbol("R", real=True, positive=True)
        self._ri = sp.Symbol("r_i", real=True, positive=True)
        r = (self._ri**n + self._R**n - sp.Float(self.Ri) ** n) ** sp.Rational(1, n)
        #: The integrand of the pressure, :math:`p = \\int_{R_i}^{R_o} f \\, dR`, in the
        #: reference radius ``R``, the deformed inner radius ``r_i`` and the parameters
        #: of the model
        self.integrand_expr = stress.subs(lmbda, r / self._R) * self._R ** (n - 1) / r**n
        dintegrand = sp.diff(self.integrand_expr, self._ri)

        self._set_parameters(parameters, (self.integrand_expr, dintegrand), (self._R, self._ri))
        args = (self._R, self._ri, *self._parameter_symbols)
        self._integrand: Callable[..., Any] = sp.lambdify(args, self.integrand_expr, "numpy")
        self._dintegrand: Callable[..., Any] = sp.lambdify(args, dintegrand, "numpy")
        self._build_symmetry_check(lmbda, self.model.first_piola_kirchhoff(F))
        self._check_symmetry(self._defaults)

    def _symmetry_violations(self, P: sp.Matrix) -> list[sp.Expr]:
        """Expressions that must vanish for the deformation to be symmetric: the
        off-diagonal entries of the stress."""
        return [P[i, j] for i in range(3) for j in range(3) if i != j]

    _symmetry_error: str

    def _build_symmetry_check(self, lmbda: sp.Symbol, P: sp.Matrix) -> None:
        p = getattr(self.model.compressibility, "p", None)
        zero = {p: 0} if p is not None else {}
        exprs = [e.subs(zero) for e in self._symmetry_violations(P)]
        diagonal = [P[i, i].subs(zero) for i in range(3)]
        symbols = set().union(*(e.free_symbols for e in exprs + diagonal)) - {lmbda}
        self._check_symbols = sorted(symbols, key=lambda s: s.name)
        self._num_violations = len(exprs)
        self._check: Callable[..., Any] = sp.lambdify(
            (lmbda, *self._check_symbols), [*exprs, *diagonal], "numpy"
        )

    def _check_symmetry(self, values: Mapping[str, Any]) -> None:
        """Raise if, at these parameter values, the stress of the model breaks the symmetry
        that the deformation assumes."""
        stretches = _np.array(_CHECK_STRETCHES).reshape(-1, 1)
        args = [values.get(s.name, 1.0) for s in self._check_symbols]
        with _np.errstate(all="ignore"):
            results = [_np.max(_np.abs(v)) for v in self._check(stretches, *args)]
        violation = max(results[: self._num_violations])
        scale = max(results[self._num_violations :])
        if not violation <= 1e-12 * max(scale, 1.0):
            raise ValueError(self._symmetry_error)

    def _values(self, parameters: Mapping[Any, Any] | None) -> list[Any]:
        if parameters:
            self._check_symmetry({**self._defaults, **self._checked(parameters)})
        return super()._values(parameters)

    def _value(self, name: str, value: Any) -> Any:
        """A float, or one value per point of the wall."""
        if _np.ndim(value) == 0:
            return float(value)
        values = _np.asarray(value, dtype=float)
        if values.shape != self.R.shape:
            raise ValueError(
                f"A parameter given through the wall needs one value per point, shape "
                f"{self.R.shape}; {name} has shape {values.shape}"
            )
        return values

    def variables(self) -> dict[str, sp.Symbol]:
        """The symbols of the integrand (``R``, ``r_i``) and of the model, by name."""
        return {self._R.name: self._R, self._ri.name: self._ri, **self.model.variables()}

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def volume(self, ri):
        """Cavity volume at the deformed inner radius ``ri``."""
        return self._c * ri**self._n

    def inner_radius(self, V):
        """Deformed inner radius at the cavity volume ``V``."""
        return (V / self._c) ** (1 / self._n)

    def radius(self, V):
        """Deformed radius of each point :attr:`R` at the cavity volume ``V``, with the
        points along the last axis."""
        ri = self.inner_radius(_np.asarray(V, dtype=float))[..., None]
        return (ri**self._n + self.R**self._n - self.Ri**self._n) ** (1 / self._n)

    def stretch(self, V):
        """Stretch of the wall at each point :attr:`R`, along the last axis. Circumferential
        for the cylinder; in both directions of the wall plane for the sphere."""
        return self.radius(V) / self.R

    # ------------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------------

    def pressure(self, V, parameters: Mapping[Any, Any] | None = None):
        """Cavity pressure, with a traction-free outer surface.

        Parameters
        ----------
        V : float or array
            Cavity volume (per unit length, for the cylinder).
        parameters : mapping, optional
            Model parameters to override for this call, by name or by symbol: a
            float, or an array of one value per point :attr:`R`.
        """
        ri = self.inner_radius(_np.asarray(V, dtype=float))[..., None]
        f = self._integrand(self.R, ri, *self._values(parameters))
        return _np.sum(self.weights * f, axis=-1)

    def stiffness(self, V, parameters: Mapping[Any, Any] | None = None):
        """Derivative of the cavity pressure with respect to the cavity volume, at fixed
        parameters. Arguments as for :meth:`pressure`."""
        V = _np.asarray(V, dtype=float)
        ri = self.inner_radius(V)
        df = self._dintegrand(self.R, ri[..., None], *self._values(parameters))
        # V = c ri^n, so dri/dV = ri / (n V)
        return _np.sum(self.weights * df, axis=-1) * ri / (self._n * V)


def _axes(radial_axis: int, axial_axis: int | None = None) -> None:
    given = [radial_axis] if axial_axis is None else [radial_axis, axial_axis]
    if any(a not in (0, 1, 2) for a in given) or len(set(given)) != len(given):
        raise ValueError(f"The axes must be distinct and 0, 1 or 2, got {given}")


class ThickCylinder(_ThickWall):
    """A thick-walled tube in plane strain, solved exactly.

    Parameters
    ----------
    model : zero_mech.Model, optional
        The material of the wall, incompressible. Its stress must be diagonal in
        the radial, circumferential and axial directions, so its fibre and sheet
        directions, if any, must lie along those axes. By default a neo-Hookean
        material.
    Ri, Ro : float
        Inner and outer radius of the unloaded tube, ``0 < Ri < Ro``.
    radial_axis, axial_axis : int, optional
        The axes of the model that are radial and axial, by default 1 and 2. The
        third is circumferential, which by default is axis 0, the default fibre
        direction of the materials.
    parameters : mapping, optional
        Values for the parameters of the model, by name or by symbol: a float, or
        an array of one value per point :attr:`R`. Those not given take the
        defaults of the model's parts.
    num_points : int, optional
        Number of Gauss-Legendre points through the wall, by default 100.

    Notes
    -----
    The volumes of the methods are per unit length: :math:`\\pi r_i^2` for an
    inner radius :math:`r_i`. A quarter of an annulus of height :math:`H` holds a
    quarter of :math:`\\pi r_i^2 H`. The constructor raises ``ValueError`` if, at
    the parameters it is given, the stress breaks the symmetry of the tube.
    """

    _n = 2
    _c = math.pi
    _symmetry_error = (
        "The stress of the model is not diagonal in the radial, circumferential and axial "
        "directions, so the tube does not deform symmetrically. Its fibre and sheet directions "
        "must lie along the axes of the tube."
    )

    def __init__(
        self,
        model: Model | None = None,
        Ri: float = 1.0,
        Ro: float = 1.5,
        radial_axis: int = 1,
        axial_axis: int = 2,
        parameters: Mapping[Any, Any] | None = None,
        num_points: int = 100,
    ) -> None:
        _axes(radial_axis, axial_axis)
        self.radial_axis = radial_axis
        self.axial_axis = axial_axis
        (self.circumferential_axis,) = {0, 1, 2} - {radial_axis, axial_axis}

        def diagonal(lmbda: sp.Symbol) -> list[sp.Expr]:
            d: list[sp.Expr] = [lmbda, lmbda, lmbda]
            d[radial_axis] = 1 / lmbda
            d[axial_axis] = sp.S.One
            return d

        self._setup(model, Ri, Ro, diagonal, parameters, num_points)


class ThickSphere(_ThickWall):
    """A thick-walled sphere, solved exactly.

    Parameters
    ----------
    model : zero_mech.Model, optional
        The material of the wall, incompressible and isotropic in the wall plane.
        By default a neo-Hookean material.
    Ri, Ro : float
        Inner and outer radius of the unloaded sphere, ``0 < Ri < Ro``.
    radial_axis : int, optional
        The axis of the model that is radial, by default 2.
    parameters : mapping, optional
        Values for the parameters of the model, by name or by symbol: a float, or
        an array of one value per point :attr:`R`. Those not given take the
        defaults of the model's parts.
    num_points : int, optional
        Number of Gauss-Legendre points through the wall, by default 100.

    Notes
    -----
    The constructor raises ``ValueError`` if, at the parameters it is given, the
    stress differs between the two directions of the wall plane. A material with
    fibres in the wall plane, such as
    :class:`zero_mech.material.HolzapfelOgden` with ``a_f > 0``, does, and so does
    an active stress along one direction. Compare
    :class:`zero_mech.SphericalVentricle`, which approximates the same sphere by a
    single stretch.
    """

    _n = 3
    _c = 4 * math.pi / 3
    _symmetry_error = (
        "The stress of the model is not the same in the two directions of the wall plane, or "
        "not diagonal, so the sphere does not deform spherically symmetrically. The model must "
        "be isotropic in the wall plane: no fibres, sheets or active stress along a direction "
        "of that plane."
    )

    def __init__(
        self,
        model: Model | None = None,
        Ri: float = 1.0,
        Ro: float = 1.5,
        radial_axis: int = 2,
        parameters: Mapping[Any, Any] | None = None,
        num_points: int = 100,
    ) -> None:
        _axes(radial_axis)
        self.radial_axis = radial_axis
        self._in_plane = sorted({0, 1, 2} - {radial_axis})

        def diagonal(lmbda: sp.Symbol) -> list[sp.Expr]:
            d: list[sp.Expr] = [lmbda, lmbda, lmbda]
            d[radial_axis] = lmbda**-2
            return d

        self._setup(model, Ri, Ro, diagonal, parameters, num_points)

    def _symmetry_violations(self, P: sp.Matrix) -> list[sp.Expr]:
        a, b = self._in_plane
        return [*super()._symmetry_violations(P), P[a, a] - P[b, b]]
