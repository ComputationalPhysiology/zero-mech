"""Zero-dimensional ventricles.

The module turns the stress in a ventricular wall into a pressure in the
cavity. It does so for any :class:`zero_mech.Model`, so the material law is
the same one that is used in the other experiments. There are two geometries:

- :class:`SphericalVentricle`, one ventricle as the wall of a sphere;
- :class:`TriSeg`, the left and right ventricle as three wall segments that
  meet in a junction (Lumens et al. 2009), which captures how the ventricles
  load each other through the septum.

Both rest on the same idea, the one-fibre model: the work done by the cavity
pressures equals the work done by the fibres, :math:`\\sigma_f \\, d\\epsilon_f`
per unit volume of wall, with :math:`\\epsilon_f = \\ln \\lambda` the natural
strain of the fibres. For a :class:`zero_mech.Model` whose wall is stretched by
:math:`\\lambda` in both in-plane directions, the fibre stress in that sense is

.. math::
    \\sigma_f = \\lambda \\frac{d\\psi}{d\\lambda} + T_a,

with :math:`T_a` an active tension along the fibres (a Cauchy stress).

.. note::
    The active tension of the ventricles, the argument ``Ta`` of their methods
    (symbol ``T_a``), is not the parameter ``Ta`` of
    :class:`zero_mech.active.ActiveStress`. If the model of the wall has its own
    active part, that part contributes through the strain energy
    :math:`\\psi` like the material does: its ``Ta`` is a second
    Piola-Kirchhoff stress along one fixed direction ``f0`` (a Cauchy stress
    :math:`T_a \\lambda^2`), it is a parameter of the model, and in
    :class:`TriSeg` it is one value shared by the three walls. The same number
    passed either way therefore gives different pressures. Use a model without
    an active part, and pass the active tension to the methods; the classes
    warn otherwise.

Spherical ventricle
-------------------

**Kinematics.** The wall is incompressible and its fibres lie in the wall
plane. When the cavity is inflated, both in-plane directions are stretched by
:math:`\\lambda` and the wall thins by :math:`\\lambda^{-2}`
(:func:`zero_mech.experiments.equibiaxial`). For a wall of volume
:math:`V_w` around a cavity of volume :math:`V`, with unloaded cavity volume
:math:`V_0`, the stretch is taken as

.. math::
    \\lambda(V) = \\left( \\frac{V + V_w / 3}{V_0 + V_w / 3} \\right)^{1/3},

the one-fibre model of Arts et al. (1991). For a thin wall it reduces to
:math:`(V / V_0)^{1/3}`.

**Pressure.** With the deformation taken as uniform through the wall, the work
done by the cavity pressure equals the change in the strain energy of the wall,
:math:`p \\, dV = V_w \\, d\\psi`, so

.. math::
    p(V) = V_w \\, \\frac{d\\psi}{d\\lambda} \\, \\frac{d\\lambda}{dV},
    \\qquad \\frac{d\\lambda}{dV} = \\frac{\\lambda}{3 V + V_w}.

For a thin wall this is Laplace's law. An active tension :math:`T_a`, a Cauchy
stress along the fibres, does work on the fibre strain :math:`\\ln \\lambda` and
adds

.. math::
    p_a(V) = \\frac{T_a}{1 + 3 V / V_w}.

TriSeg
------

Three wall segments, the left free wall (LW), the septum (SW) and the right free
wall (RW), are spherical caps that meet in a circle of radius :math:`y_m`. Each
is described by its midwall surface, with :math:`x_m` the distance from the plane
of the junction to the top of the cap (negative for the left free wall):

.. math::
    V_m = \\frac{\\pi}{6} x_m (x_m^2 + 3 y_m^2), \\qquad
    A_m = \\pi (x_m^2 + y_m^2), \\qquad
    C_m = \\frac{2 x_m}{x_m^2 + y_m^2}.

The cavity volumes fix the midwall volumes of the free walls,
:math:`V_{m,LW} = -V_{LV} - V_{w,LW}/2 - V_{w,SW}/2 + V_{m,SW}` and
:math:`V_{m,RW} = V_{RV} + V_{w,RW}/2 + V_{w,SW}/2 + V_{m,SW}`. With
:math:`z = 3 C_m V_w / (2 A_m)`, the fibre strain and the midwall tension of a
segment are

.. math::
    \\epsilon_f = \\frac{1}{2} \\ln \\frac{A_m}{A_{m,ref}} - \\frac{z^2}{12} - 0.019 z^4,
    \\qquad
    T_m = \\frac{V_w \\sigma_f}{2 A_m} \\left(1 + \\frac{z^2}{3} + \\frac{z^4}{5}\\right).

The two unknowns :math:`x_{m,SW}` and :math:`y_m` follow from the balance of
the tensions in the junction, along the axis and radially,

.. math::
    \\sum_i T_{m,i} \\frac{2 x_{m,i} y_m}{x_{m,i}^2 + y_m^2} = 0, \\qquad
    \\sum_i T_{m,i} \\frac{y_m^2 - x_{m,i}^2}{x_{m,i}^2 + y_m^2} = 0,

and the pressures from the axial tensions of the free walls,
:math:`p_{LV} = -2 T_{x,LW} / y_m` and :math:`p_{RV} = 2 T_{x,RW} / y_m`.

References
----------
Lumens J, Delhaas T, Kirn B, Arts T (2009). Three-wall segment (TriSeg) model
describing mechanics and hemodynamics of ventricular interaction. Ann Biomed
Eng 37:2234-2255.

Arts T, Bovendeerd PHM, Prinzen FW, Reneman RS (1991). Relation between left
ventricular cavity pressure and volume and systolic fiber stress and strain in
the wall. Biophys J 59:93-102.

Caruel M, Chabiniok R, Moireau P, Lecarpentier Y, Chapelle D (2014).
Dimensional reductions of a cardiac model for effective validation and
calibration. Biomech Model Mechanobiol 13:897-914.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from .active import Passive
from .atoms import Atom
from .experiments import equibiaxial
from .model import Model

__all__ = ["SphericalVentricle", "TriSeg", "TriSegState"]

try:  # arrays where numpy is installed, scalars otherwise
    import numpy as _np

    _MODULES = "numpy"
    _quiet: Callable[[], Any] = lambda: _np.errstate(all="ignore")
except ImportError:  # pragma: no cover
    _MODULES = "math"
    _quiet = contextlib.nullcontext


def _by_name(parameters: Mapping[Any, float]) -> dict[str, float]:
    return {
        (k.name if isinstance(k, sp.Symbol) else str(k)): float(v) for k, v in parameters.items()
    }


def _default_parameters(model: Model) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for part in (model.material, model.compressibility, model.active):
        get = getattr(part, "default_parameters", None)
        if get is not None:
            defaults.update(_by_name(get()))
    return defaults


def _passive_fibre_stress(model: Model, radial_axis: int) -> tuple[sp.Symbol, sp.Expr]:
    """The in-plane stretch and the passive fibre stress ``lmbda * dpsi/dlmbda`` of a wall of
    the given model that is stretched equally in both in-plane directions."""
    lmbda = sp.Symbol("λ", real=True, positive=True)
    experiment = equibiaxial(axis=radial_axis, lmbda=lmbda)
    if experiment.J != 1:  # pragma: no cover
        raise RuntimeError("The deformation is not volume preserving")
    return lmbda, lmbda * sp.diff(model.strain_energy(experiment.F), lmbda)


class _Ventricle(Atom):
    """What the ventricles share: the model of the wall and its parameters."""

    model: Model
    _defaults: dict[str, float]
    _parameter_symbols: list[sp.Symbol]

    def _set_model(self, model: Model | None) -> None:
        self.model = model if model is not None else Model()
        if not isinstance(self.model.active, Passive):
            warnings.warn(
                f"The model of the wall has an active part ({type(self.model.active).__name__}). "
                "Its parameters are not the active tension Ta of the ventricle: they enter "
                "through the strain energy, along one fixed direction and equally in every "
                "wall. Use a model without an active part and pass the active tension to the "
                "methods of the ventricle, unless this is intended.",
                UserWarning,
                stacklevel=3,
            )

    def _set_parameters(
        self,
        parameters: Mapping[Any, float] | None,
        expressions: Sequence[sp.Basic],
        own: Sequence[sp.Symbol],
    ) -> None:
        """Collect the parameters of the model: their values (the defaults of the model's
        parts, then ``parameters``) and, sorted by name, those that ``expressions`` depend on.
        ``own`` are the symbols of the ventricle itself."""
        self._defaults = _default_parameters(self.model)
        free: set[sp.Symbol] = set()
        for expr in expressions:
            free |= expr.free_symbols
        self._parameter_symbols = sorted(free - set(own), key=lambda s: s.name)
        #: Every name that is a parameter of the model, used or not
        self._known = set(self._defaults) | {s.name for s in self._parameter_symbols}
        if parameters is not None:
            self._defaults.update(self._checked(parameters))
        missing = [s.name for s in self._parameter_symbols if s.name not in self._defaults]
        if missing:
            raise ValueError(f"No value for the model parameters {missing}")

    def _checked(self, parameters: Mapping[Any, float]) -> dict[str, float]:
        """``parameters`` by name, after checking that each is a parameter of the model."""
        given = _by_name(parameters)
        unknown = set(given) - self._known
        if unknown:
            raise KeyError(f"Unknown parameters {sorted(unknown)}; known: {sorted(self._known)}")
        return given

    @property
    def parameters(self) -> dict[str, float]:
        """The parameters of the model that the pressure depends on, with their values."""
        return {s.name: self._defaults[s.name] for s in self._parameter_symbols}

    def _values(self, parameters: Mapping[Any, float] | None) -> list[float]:
        values = self._defaults
        if parameters:
            values = {**values, **self._checked(parameters)}
        return [values[s.name] for s in self._parameter_symbols]


class SphericalVentricle(_Ventricle):
    """The wall of a sphere around a cavity.

    Parameters
    ----------
    model : zero_mech.Model
        The material of the wall. Its fibre and sheet directions should lie
        in the wall plane, i.e. not along ``radial_axis``.
    V0 : float
        Volume of the cavity when the wall is unloaded.
    Vw : float
        Volume of the wall, in the same unit as ``V0``. Must be positive.
    radial_axis : int, optional
        The direction through the wall, by default 2.
    parameters : mapping, optional
        Values for the parameters of the model, by name or by symbol. Those
        not given take the defaults of the model's parts. A name that is not
        a parameter of the model raises a ``KeyError``.

    Notes
    -----
    The active tension of the ventricle is the argument ``Ta`` of
    :meth:`pressure` (symbol ``T_a``), a Cauchy stress along the fibres. It is
    not the parameter ``Ta`` of :class:`zero_mech.active.ActiveStress`; see the
    note in the module documentation. The model should have no active part.

    Volumes can be in any unit, as long as ``V0``, ``Vw`` and the volumes
    passed to the methods share it. Pressures come out in the unit of the
    material parameters and of the active tension.

    The pressure and its derivative are available as sympy expressions,
    :attr:`pressure_expr` and :attr:`stiffness_expr`, in the symbols ``V``,
    ``T_a`` and those of the model.
    """

    def __init__(
        self,
        model: Model | None = None,
        V0: float = 1.0,
        Vw: float = 1.0,
        radial_axis: int = 2,
        parameters: Mapping[Any, float] | None = None,
    ) -> None:
        if not V0 > 0:
            raise ValueError(f"V0 must be positive, got {V0}")
        if not Vw > 0:
            raise ValueError(f"Vw must be positive, got {Vw}")

        self._set_model(model)
        self.V0 = float(V0)
        self.Vw = float(Vw)
        self.radial_axis = radial_axis
        #: Cavity volume
        self.V = sp.Symbol("V", real=True)
        #: Active tension along the fibres (Cauchy stress)
        self.Ta = sp.Symbol("T_a", real=True)

        lmbda, sigma_passive = _passive_fibre_stress(self.model, radial_axis)
        #: Stretch as a function of the cavity volume
        self.stretch_expr = ((self.V + self.Vw / 3) / (self.V0 + self.Vw / 3)) ** sp.Rational(1, 3)
        # p = Vw dpsi/dlmbda dlmbda/dV + Ta / (1 + 3 V / Vw), with dlmbda/dV = lmbda / (3 V + Vw)
        pressure = self.Vw * (sigma_passive + self.Ta) / (3 * self.V + self.Vw)
        #: Cavity pressure as a function of the cavity volume and the active tension
        self.pressure_expr = pressure.subs(lmbda, self.stretch_expr)
        #: Derivative of the cavity pressure with respect to the cavity volume
        self.stiffness_expr = sp.diff(self.pressure_expr, self.V)

        self._set_parameters(
            parameters, (self.pressure_expr, self.stiffness_expr), (self.V, self.Ta)
        )
        args = (self.V, self.Ta, *self._parameter_symbols)
        self._pressure: Callable[..., Any] = sp.lambdify(args, self.pressure_expr, _MODULES)
        self._stiffness: Callable[..., Any] = sp.lambdify(args, self.stiffness_expr, _MODULES)

    @classmethod
    def from_radius(cls, R0: float, h0: float, **kwargs: Any) -> SphericalVentricle:
        """A ventricle from the inner radius ``R0`` and the wall thickness ``h0``
        of the unloaded sphere. Other arguments are those of the class."""
        V0 = 4 * sp.pi * R0**3 / 3
        Vw = 4 * sp.pi * ((R0 + h0) ** 3 - R0**3) / 3
        return cls(V0=float(V0), Vw=float(Vw), **kwargs)

    def variables(self) -> dict[str, sp.Symbol]:
        """The symbols of the ventricle (``V``, ``T_a``) and of its model, by name."""
        # Not Atom.variables: that one reads __slots__, which every Atom inherits
        # (empty) from abc.ABC, so it finds nothing on a class without slots.
        return {self.V.name: self.V, self.Ta.name: self.Ta, **self.model.variables()}

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def stretch(self, V):
        """Stretch of the fibres at cavity volume ``V``; one at ``V0``."""
        return ((V + self.Vw / 3) / (self.V0 + self.Vw / 3)) ** (1 / 3)

    def dstretch_dV(self, V):
        """Derivative of the stretch with respect to the cavity volume."""
        return self.stretch(V) / (3 * V + self.Vw)

    def volume(self, stretch):
        """Cavity volume at which the fibres have the given stretch."""
        return stretch**3 * (self.V0 + self.Vw / 3) - self.Vw / 3

    # ------------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------------

    def pressure(self, V, Ta=0.0, parameters: Mapping[Any, float] | None = None):
        """Cavity pressure.

        Parameters
        ----------
        V : float or array
            Cavity volume.
        Ta : float or array, optional
            Active tension along the fibres, a Cauchy stress, by default 0.
        parameters : mapping, optional
            Model parameters to override for this call, by name or by symbol.
            A parameter that the pressure does not depend on has no effect.
        """
        return self._pressure(V, Ta, *self._values(parameters))

    def stiffness(self, V, Ta=0.0, parameters: Mapping[Any, float] | None = None):
        """Derivative of the cavity pressure with respect to the cavity volume,
        at fixed active tension. Arguments as for :meth:`pressure`."""
        return self._stiffness(V, Ta, *self._values(parameters))

    def active_pressure(self, V, Ta):
        """The part of the pressure that the active tension gives, ``Ta / (1 + 3 V / Vw)``."""
        return Ta / (1 + 3 * V / self.Vw)


@dataclass(frozen=True)
class TriSegState:
    """The geometry and the loads of a :class:`TriSeg` model at given cavity volumes.

    The entries of the tuples are for the left free wall, the septum and the right
    free wall, in that order.
    """

    #: Cavity pressures, in the unit of the material parameters
    p_LV: float
    p_RV: float
    #: Radius of the junction circle
    ym: float
    #: Distance from the junction plane to the top of each midwall surface
    xm: tuple[float, float, float]
    #: Stretch of the fibres in each wall
    stretch: tuple[float, float, float]
    #: Midwall area, curvature and tension of each wall
    Am: tuple[float, float, float]
    Cm: tuple[float, float, float]
    Tm: tuple[float, float, float]
    #: Net axial and radial tension in the junction, zero at equilibrium
    residual: tuple[float, float]
    #: Number of Newton iterations used
    iterations: int

    @property
    def guess(self) -> tuple[float, float]:
        """``(xm_SW, ym)``, the starting point for a solve at nearby volumes."""
        return (self.xm[1], self.ym)


class TriSeg(_Ventricle):
    """The left and right ventricle as three wall segments (Lumens et al. 2009).

    Parameters
    ----------
    model : zero_mech.Model
        The material of the walls. Its fibre and sheet directions should lie
        in the wall plane, i.e. not along ``radial_axis``.
    Vw : sequence of three floats
        Wall volumes of the left free wall, the septum and the right free wall.
    Am_ref : sequence of three floats
        Midwall areas at which the fibres of each wall are unstretched.
    radial_axis : int, optional
        The direction through the wall, by default 2.
    parameters : mapping, optional
        Values for the parameters of the model, by name or by symbol. Those
        not given take the defaults of the model's parts. A name that is not
        a parameter of the model raises a ``KeyError``.

    Notes
    -----
    The active tension of each wall is the argument ``Ta`` of :meth:`solve`
    (symbols ``T_a_LW``, ``T_a_SW``, ``T_a_RW``), a Cauchy stress along the
    fibres. It is not the parameter ``Ta`` of
    :class:`zero_mech.active.ActiveStress`, which would be one value shared by
    the three walls; see the note in the module documentation. The model should
    have no active part.

    Lengths, areas and volumes must share a unit of length (e.g. cm, cm² and mL).
    Pressures come out in the unit of the material parameters and of the active
    tension.

    ``Am_ref`` is the area at a fibre stretch of one. Lumens et al. give theirs
    at a sarcomere length of 2.0 µm; for sarcomeres that are slack at
    :math:`L_0`, multiply those by :math:`(L_0 / 2.0)^2`. The defaults are the
    values Lumens et al. start from (wall volumes 75, 40 and 30 mL; areas 80, 45
    and 100 cm²), with the areas scaled to a slack length of 1.85 µm.

    The pericardium is not included: the pressure around the ventricles is zero.
    """

    walls = ("LW", "SW", "RW")

    def __init__(
        self,
        model: Model | None = None,
        Vw: Sequence[float] = (75.0, 40.0, 30.0),
        Am_ref: Sequence[float] = (68.0, 38.0, 85.0),
        radial_axis: int = 2,
        parameters: Mapping[Any, float] | None = None,
    ) -> None:
        if len(Vw) != 3 or len(Am_ref) != 3:
            raise ValueError("Vw and Am_ref must have three entries: LW, SW, RW")
        if not all(v > 0 for v in Vw) or not all(a > 0 for a in Am_ref):
            raise ValueError("Wall volumes and reference areas must be positive")

        self._set_model(model)
        self.Vw = tuple(float(v) for v in Vw)
        self.Am_ref = tuple(float(a) for a in Am_ref)
        self.radial_axis = radial_axis

        #: Cavity volumes
        self.V_LV = sp.Symbol("V_LV", real=True)
        self.V_RV = sp.Symbol("V_RV", real=True)
        #: Active tension along the fibres of each wall (Cauchy stress)
        self.Ta = tuple(sp.Symbol(f"T_a_{w}", real=True) for w in self.walls)
        #: Active stiffness of each wall and the stretch at which ``Ta`` holds
        self.Ka = tuple(sp.Symbol(f"K_a_{w}", real=True) for w in self.walls)
        self.stretch_ref = tuple(sp.Symbol(f"λ_ref_{w}", real=True) for w in self.walls)
        #: The unknowns: septal ``xm`` and the radius of the junction
        self.xm_SW = sp.Symbol("xm_SW", real=True)
        self.ym = sp.Symbol("ym", real=True, positive=True)

        lmbda, sigma_passive = _passive_fibre_stress(self.model, radial_axis)
        Vw_L, Vw_S, Vw_R = self.Vw
        Vm_S = sp.pi / 6 * self.xm_SW * (self.xm_SW**2 + 3 * self.ym**2)
        Vm = (
            -self.V_LV - Vw_L / 2 - Vw_S / 2 + Vm_S,
            Vm_S,
            self.V_RV + Vw_R / 2 + Vw_S / 2 + Vm_S,
        )
        xm = (self._xm(Vm[0]), self.xm_SW, self._xm(Vm[2]))

        Tx, Ty = [], []
        stretch, Am_all, Cm_all, Tm_all = [], [], [], []
        for i in range(3):
            r2 = xm[i] ** 2 + self.ym**2
            Am = sp.pi * r2
            Cm = 2 * xm[i] / r2
            z = 3 * Cm * self.Vw[i] / (2 * Am)
            eps = sp.log(Am / self.Am_ref[i]) / 2 - z**2 / 12 - sp.Float(0.019) * z**4
            lam = sp.exp(eps)
            sigma = sigma_passive.subs(lmbda, lam) + self.Ta[i]
            sigma += self.Ka[i] * (lam - self.stretch_ref[i])
            Tm = self.Vw[i] * sigma / (2 * Am) * (1 + z**2 / 3 + z**4 / 5)
            Tx.append(Tm * 2 * xm[i] * self.ym / r2)
            Ty.append(Tm * (self.ym**2 - xm[i] ** 2) / r2)
            stretch.append(lam)
            Am_all.append(Am)
            Cm_all.append(Cm)
            Tm_all.append(Tm)

        #: Net axial and radial tension in the junction
        self.residual_expr = sp.Matrix([sum(Tx), sum(Ty)])
        unknowns = sp.Matrix([self.xm_SW, self.ym])
        jacobian = self.residual_expr.jacobian(unknowns)
        #: Cavity pressures
        self.p_LV_expr = -2 * Tx[0] / self.ym
        self.p_RV_expr = 2 * Tx[2] / self.ym

        own = (self.V_LV, self.V_RV, self.xm_SW, self.ym, *self.Ta, *self.Ka, *self.stretch_ref)
        self._set_parameters(parameters, (self.residual_expr,), own)

        args = (self.xm_SW, self.ym, self.V_LV, self.V_RV, *self.Ta, *self.Ka, *self.stretch_ref)
        args += tuple(self._parameter_symbols)
        # The midwall tensions come with the residual: they are its scale in the iteration
        newton = [*self.residual_expr, *jacobian, *Tm_all]
        outputs = [self.p_LV_expr, self.p_RV_expr, xm[0], xm[2]]
        outputs += [*stretch, *Am_all, *Cm_all, *Tm_all]
        self._newton: Callable[..., Any] = sp.lambdify(args, newton, _MODULES, cse=True)
        self._outputs: Callable[..., Any] = sp.lambdify(args, outputs, _MODULES, cse=True)

    def _xm(self, Vm: sp.Expr) -> sp.Expr:
        """``xm`` of a cap with midwall volume ``Vm`` on the junction circle: the real
        root of ``xm^3 + 3 ym^2 xm - 6 Vm / pi = 0``."""
        V = 3 * Vm / sp.pi
        Q = (V + sp.sqrt(V**2 + self.ym**6)) ** sp.Rational(1, 3)
        return Q - self.ym**2 / Q

    def variables(self) -> dict[str, sp.Symbol]:
        """The symbols of the ventricles and of their model, by name."""
        own = (self.V_LV, self.V_RV, self.xm_SW, self.ym, *self.Ta, *self.Ka, *self.stretch_ref)
        return {**{s.name: s for s in own}, **self.model.variables()}

    def initial_guess(self, V_LV: float, V_RV: float) -> tuple[float, float]:
        """A starting point ``(xm_SW, ym)`` for :meth:`solve`: the junction of a left
        ventricle that is a sphere, with a septum bulging slightly into the right ventricle."""
        r = self._radius(V_LV)
        return (0.2 * r, 0.9 * r)

    def _radius(self, V_LV: float) -> float:
        """Midwall radius of a spherical left ventricle with cavity volume ``V_LV``."""
        Vw_L, Vw_S, _ = self.Vw
        return (3 * (V_LV + (Vw_L + Vw_S) / 2) / (4 * math.pi)) ** (1 / 3)

    def solve(
        self,
        V_LV: float,
        V_RV: float,
        Ta: float | Sequence[float] = 0.0,
        Ka: float | Sequence[float] = 0.0,
        stretch_ref: float | Sequence[float] = 1.0,
        guess: tuple[float, float] | None = None,
        parameters: Mapping[Any, float] | None = None,
        tol: float = 1e-10,
        max_iterations: int = 100,
    ) -> TriSegState:
        """Find the geometry in which the tensions in the junction balance.

        Parameters
        ----------
        V_LV, V_RV : float
            Cavity volumes.
        Ta : float or sequence of three floats, optional
            Active tension along the fibres, a Cauchy stress: one value for all
            walls, or one each for LW, SW and RW. By default 0.
        Ka, stretch_ref : float or sequence of three floats, optional
            Active stiffness, the derivative of the active tension with respect to
            the stretch, and the stretch at which ``Ta`` holds. The active tension
            of a wall at stretch λ is ``Ta + Ka (λ - stretch_ref)``. By default
            ``Ka`` is 0, i.e. the active tension does not depend on the geometry
            that is solved for; see the notes.
        guess : tuple, optional
            Starting point ``(xm_SW, ym)``; pass ``state.guess`` of a solution at
            nearby volumes. By default :meth:`initial_guess`.
        parameters : mapping, optional
            Model parameters to override for this call, by name or by symbol.
            A parameter that the pressures do not depend on has no effect.
        tol : float, optional
            Relative tolerance: the net tension in the junction is brought below
            ``tol`` times the largest midwall tension.
        max_iterations : int, optional
            Newton iterations allowed from each starting point.

        Returns
        -------
        TriSegState

        Raises
        ------
        RuntimeError
            If the iteration does not converge.

        Notes
        -----
        The tension in a wall is its fibre stress times ``Vw / (2 Am)``, so a wall
        that shortens pulls harder at the same stress. With active tensions that
        do not depend on the stretch the geometry is therefore not stable: one wall
        shortens at the expense of the others, and the solution can jump between
        such states as the load changes. Muscle does not behave that way, because
        its tension falls at once when it shortens. ``Ka`` carries that dependence
        into the solve. When coupling to a model of the sarcomere, pass its tension
        as ``Ta``, its active stiffness as ``Ka`` and the stretch at which the two
        were evaluated as ``stretch_ref``.
        """
        if not (V_LV > 0 and V_RV > 0):
            raise ValueError(f"Cavity volumes must be positive, got {V_LV} and {V_RV}")
        Ta3, Ka3, ref3 = (
            self._per_wall(v, n) for v, n in ((Ta, "Ta"), (Ka, "Ka"), (stretch_ref, "stretch_ref"))
        )
        tail = (float(V_LV), float(V_RV), *Ta3, *Ka3, *ref3, *self._values(parameters))

        # Newton's method converges from a septum that starts close to flat. Other starting
        # points are tried if the first one fails, as a guess from distant volumes may.
        r = self._radius(V_LV)
        starts = [guess if guess is not None else self.initial_guess(V_LV, V_RV)]
        starts += [(fx * r, fy * r) for fx, fy in ((0.2, 0.9), (0.1, 0.8), (0.3, 1.0), (-0.2, 0.9))]
        # Last resort, for loads far from the usual ones (e.g. a passive septum between
        # active free walls, which bulges far into the right ventricle)
        starts += [
            (fx * r, fy * r)
            for fy in (0.9, 0.7, 1.1, 0.5, 1.3)
            for fx in (0.5, 0.8, -0.05, -0.4, 1.1, -0.7)
        ]
        for start in starts:
            found = self._newton_iteration(start, tail, tol, max_iterations)
            if found is not None:
                break
        else:
            raise RuntimeError(f"TriSeg did not converge at V_LV={V_LV}, V_RV={V_RV}, Ta={Ta3}")
        x, y, f, iterations = found

        out = [float(v) for v in self._outputs(x, y, *tail)]
        return TriSegState(
            p_LV=out[0],
            p_RV=out[1],
            ym=y,
            xm=(out[2], x, out[3]),
            stretch=(out[4], out[5], out[6]),
            Am=(out[7], out[8], out[9]),
            Cm=(out[10], out[11], out[12]),
            Tm=(out[13], out[14], out[15]),
            residual=f,
            iterations=iterations,
        )

    @staticmethod
    def _per_wall(value: Any, name: str) -> list[float]:
        """Three floats from one value (numpy scalars and 0-d arrays included) or one per wall."""
        if isinstance(value, numbers.Real) or getattr(value, "ndim", None) == 0:
            values = [float(value)] * 3
        else:
            values = [float(v) for v in value]
        if len(values) != 3:
            raise ValueError(f"{name} must be one value or three: LW, SW, RW")
        return values

    def _newton_iteration(
        self, start: tuple[float, float], tail: tuple[float, ...], tol: float, max_iterations: int
    ) -> tuple[float, float, tuple[float, float], int] | None:
        """Damped Newton iteration on the two tension balances from ``start``.
        Returns ``(xm_SW, ym, residual, iterations)``, or None if it does not converge."""

        def evaluate(x, y):
            try:
                with _quiet():
                    vals = [float(v) for v in self._newton(x, y, *tail)]
            except (OverflowError, ValueError, ZeroDivisionError):
                return None
            norm = math.hypot(vals[0], vals[1])
            scale = max(abs(t) for t in vals[6:9])
            if not (math.isfinite(norm) and math.isfinite(scale)):
                return None
            return (vals[0], vals[1]), vals[2:6], norm, scale

        x, y = start
        if not y > 0:
            return None
        current = evaluate(x, y)
        if current is None:
            return None
        f, J, norm, scale = current
        for iteration in range(max_iterations + 1):
            if norm <= tol * scale:
                return x, y, f, iteration
            det = J[0] * J[3] - J[1] * J[2]
            if det == 0 or not math.isfinite(det):
                return None
            dx = -(J[3] * f[0] - J[1] * f[1]) / det
            dy = -(J[0] * f[1] - J[2] * f[0]) / det
            # Backtrack until the junction radius is positive and the residual falls
            step = 1.0
            while step > 1e-6:
                trial = evaluate(x + step * dx, y + step * dy) if y + step * dy > 0 else None
                if trial is not None and trial[2] < norm:
                    break
                step /= 2
            else:
                return None
            x, y = x + step * dx, y + step * dy
            f, J, norm, scale = trial
        return None

    def pressures(self, V_LV: float, V_RV: float, Ta=0.0, **kwargs: Any) -> tuple[float, float]:
        """Cavity pressures ``(p_LV, p_RV)``; arguments as for :meth:`solve`."""
        state = self.solve(V_LV, V_RV, Ta, **kwargs)
        return state.p_LV, state.p_RV
