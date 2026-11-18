import math

import numpy as np
import pytest
import sympy as sp

import zero_mech
from zero_mech import ThickCylinder, ThickSphere

MU = 15.0  # the default of NeoHookean
RI, RO = 1.0, 1.5

#: Holzapfel-Ogden as in pulse's transversely_isotropic_parameters(), in kPa
TRANSVERSELY_ISOTROPIC = dict(
    a=2.280, b=9.726, a_f=1.685, b_f=15.779, a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0
)


def contracting_wall():
    """Holzapfel-Ogden with circumferential fibres (axis 0) and radial sheets (axis 1),
    the default axes of ThickCylinder, and an active stress along the fibres."""
    return zero_mech.Model(
        material=zero_mech.material.HolzapfelOgden(use_heaviside=True),
        active=zero_mech.StretchActiveStress(),
    )


def stretches(wall, V):
    """The stretch at the inner and the outer surface, from the incompressibility."""
    n = 2 if isinstance(wall, ThickCylinder) else 3
    ri = wall.inner_radius(V)
    ro = (ri**n + wall.Ro**n - wall.Ri**n) ** (1 / n)
    return ri / wall.Ri, ro / wall.Ro


@pytest.mark.parametrize("ratio", [0.6, 0.9, 1.2, 2.0, 4.0])
def test_neo_hookean_cylinder_closed_form(ratio):
    # p = mu (ln(li / lo) - 1 / (2 li^2) + 1 / (2 lo^2)), li and lo the inner and outer stretch
    cylinder = ThickCylinder(Ri=RI, Ro=RO)
    V = ratio * cylinder.V0
    li, lo = stretches(cylinder, V)
    exact = MU * (math.log(li / lo) - 1 / (2 * li**2) + 1 / (2 * lo**2))
    assert cylinder.pressure(V) == pytest.approx(exact, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("ratio", [0.6, 0.9, 1.2, 2.0, 4.0])
def test_neo_hookean_sphere_closed_form(ratio):
    # p = mu (2 / lo + 1 / (2 lo^4) - 2 / li - 1 / (2 li^4))
    sphere = ThickSphere(Ri=RI, Ro=RO)
    V = ratio * sphere.V0
    li, lo = stretches(sphere, V)
    exact = MU * (2 / lo + 1 / (2 * lo**4) - 2 / li - 1 / (2 * li**4))
    assert sphere.pressure(V) == pytest.approx(exact, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("cls", [ThickCylinder, ThickSphere])
def test_unloaded_wall_has_no_pressure(cls):
    wall = cls(Ri=RI, Ro=RO)
    assert wall.pressure(wall.V0) == pytest.approx(0.0, abs=1e-13)
    assert np.allclose(wall.stretch(wall.V0), 1.0)


@pytest.mark.parametrize("cls, n", [(ThickCylinder, 2), (ThickSphere, 3)])
def test_kinematics(cls, n):
    wall = cls(Ri=2.0, Ro=3.0, num_points=7)
    assert wall.V0 == pytest.approx(wall._c * 2.0**n)
    V = np.array([0.5, 1.0, 3.0]) * wall.V0
    assert np.allclose(wall.volume(wall.inner_radius(V)), V)
    r = wall.radius(V)
    assert r.shape == (3, 7)
    # Incompressible: r^n - R^n is the same at every point, the change of the cavity
    assert np.allclose(r**n - wall.R**n, (wall.inner_radius(V) ** n - 2.0**n)[:, None])
    assert np.allclose(wall.stretch(V), r / wall.R)
    assert np.all((wall.R > 2.0) & (wall.R < 3.0))
    assert np.sum(wall.weights) == pytest.approx(1.0)


def test_pressure_accepts_arrays_of_volumes():
    cylinder = ThickCylinder(Ri=RI, Ro=RO)
    V = np.array([0.8, 1.5, 2.5]) * cylinder.V0
    assert np.allclose(cylinder.pressure(V), [cylinder.pressure(v) for v in V], rtol=1e-14)


@pytest.mark.parametrize("ratio", [0.8, 1.3])
def test_active_stress_of_a_cylinder(ratio):
    # Circumferential fibres add their Cauchy stress to sigma_tt - sigma_rr. For
    # StretchActiveStress that is Ta lmbda, so p_a = Ta int dR / r = Ta ln((Ro + ro) / (Ri + ri));
    # for ActiveStress it is Ta lmbda^2, so p_a = Ta int dR / R = Ta ln(Ro / Ri), at any volume
    Ta = 7.0
    stretch = ThickCylinder(zero_mech.Model(active=zero_mech.StretchActiveStress()), RI, RO)
    invariant = ThickCylinder(zero_mech.Model(active=zero_mech.ActiveStress()), RI, RO)
    V = ratio * stretch.V0
    li, lo = stretches(stretch, V)
    p_stretch = stretch.pressure(V, parameters={"Ta": Ta}) - stretch.pressure(V)
    p_invariant = invariant.pressure(V, parameters={"Ta": Ta}) - invariant.pressure(V)
    assert p_stretch == pytest.approx(Ta * math.log((RO * (1 + lo)) / (RI * (1 + li))), rel=1e-12)
    assert p_invariant == pytest.approx(Ta * math.log(RO / RI), rel=1e-12)


def test_parameters_can_vary_through_the_wall():
    cylinder = ThickCylinder(zero_mech.Model(active=zero_mech.StretchActiveStress()), RI, RO)
    V = 0.9 * cylinder.V0
    uniform = np.full(cylinder.R.shape, 7.0)
    assert cylinder.pressure(V, parameters={"Ta": uniform}) == pytest.approx(
        cylinder.pressure(V, parameters={"Ta": 7.0}), rel=1e-14
    )
    # p_a = int Ta(R) lmbda R / r^2 dR = int Ta(R) / r dR
    Ta = 10.0 * (cylinder.R - RI)
    active = cylinder.pressure(V, parameters={"Ta": Ta}) - cylinder.pressure(V)
    assert active == pytest.approx(np.sum(cylinder.weights * Ta / cylinder.radius(V)), rel=1e-12)
    through_constructor = ThickCylinder(cylinder.model, RI, RO, parameters={"Ta": Ta})
    assert through_constructor.pressure(V) == pytest.approx(cylinder.pressure(V, {"Ta": Ta}))


def test_a_parameter_through_the_wall_needs_a_value_per_point():
    cylinder = ThickCylinder(zero_mech.Model(active=zero_mech.StretchActiveStress()), RI, RO)
    with pytest.raises(ValueError, match="one value per point"):
        cylinder.pressure(cylinder.V0, parameters={"Ta": np.ones(3)})
    with pytest.raises(ValueError, match="one value per point"):
        ThickCylinder(cylinder.model, RI, RO, parameters={"Ta": np.ones(3)})


@pytest.mark.parametrize("Ta", [0.0, 30.0])
def test_stiffness_is_the_derivative_of_the_pressure(Ta):
    cylinder = ThickCylinder(contracting_wall(), RI, RO, parameters=TRANSVERSELY_ISOTROPIC)
    sphere = ThickSphere(Ri=RI, Ro=RO)
    for wall, parameters in ((cylinder, {"Ta": Ta}), (sphere, None)):
        V, h = np.array([0.9, 1.2, 1.6]) * wall.V0, 1e-6
        slope = (
            wall.pressure(V * (1 + h), parameters) - wall.pressure(V * (1 - h), parameters)
        ) / (2 * h * V)
        assert np.allclose(wall.stiffness(V, parameters), slope, rtol=1e-6)


def test_thin_sphere_is_the_spherical_ventricle():
    # The one-fibre model of SphericalVentricle is exact in the limit of a thin wall: the
    # difference is first order in the thickness
    differences = []
    for h0 in (1e-2, 1e-3):
        sphere = ThickSphere(Ri=1.0, Ro=1.0 + h0)
        ventricle = zero_mech.SphericalVentricle.from_radius(1.0, h0)
        V = 1.5 * sphere.V0
        differences.append(abs(ventricle.pressure(V) / sphere.pressure(V) - 1))
    assert differences[1] < 2e-3
    assert differences[0] / differences[1] == pytest.approx(10, rel=0.1)


def test_quadrature_of_a_smooth_integrand_converges_to_round_off():
    model = contracting_wall()
    coarse = ThickCylinder(model, RI, RO, parameters=TRANSVERSELY_ISOTROPIC, num_points=60)
    fine = ThickCylinder(model, RI, RO, parameters=TRANSVERSELY_ISOTROPIC, num_points=240)
    V = 1.2 * coarse.V0  # the fibres are stretched everywhere, so the heaviside does not switch
    assert np.all(coarse.stretch(V) > 1)
    assert coarse.pressure(V) == pytest.approx(fine.pressure(V), rel=1e-13)


def test_holzapfel_ogden_with_fibres_in_the_wall_plane_is_refused_by_the_sphere():
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden(use_heaviside=True))
    with pytest.raises(ValueError, match="wall plane"):
        ThickSphere(model, RI, RO, parameters=TRANSVERSELY_ISOTROPIC)
    # Without the fibre term, or with the fibres through the wall, it is symmetric
    ThickSphere(model, RI, RO, parameters={**TRANSVERSELY_ISOTROPIC, "a_f": 0.0})
    ThickSphere(model, RI, RO, radial_axis=0, parameters=TRANSVERSELY_ISOTROPIC)


def test_an_active_stress_along_the_wall_plane_is_refused_by_the_sphere():
    sphere = ThickSphere(zero_mech.Model(active=zero_mech.StretchActiveStress()), RI, RO)
    assert sphere.pressure(sphere.V0) == pytest.approx(0.0, abs=1e-13)  # Ta defaults to 0
    with pytest.raises(ValueError, match="wall plane"):
        sphere.pressure(sphere.V0, parameters={"Ta": 1.0})
    with pytest.raises(ValueError, match="wall plane"):
        sphere.stiffness(sphere.V0, parameters={"Ta": 1.0})


def test_fibres_off_the_axes_are_refused_by_the_cylinder():
    helix = sp.Matrix([math.cos(0.5), 0, math.sin(0.5)])
    material = zero_mech.material.HolzapfelOgden(use_heaviside=True, f0=helix)
    with pytest.raises(ValueError, match="axes of the tube"):
        ThickCylinder(zero_mech.Model(material=material), RI, RO, parameters=TRANSVERSELY_ISOTROPIC)


@pytest.mark.parametrize(
    "kwargs",
    [dict(Ri=0.0), dict(Ri=1.5, Ro=1.0), dict(num_points=0), dict(radial_axis=3)],
)
def test_rejects_bad_geometry(kwargs):
    with pytest.raises(ValueError):
        ThickSphere(**kwargs)


def test_cylinder_rejects_equal_axes():
    with pytest.raises(ValueError, match="distinct"):
        ThickCylinder(radial_axis=2, axial_axis=2)


def test_rejects_a_compressible_wall():
    model = zero_mech.Model(compressibility=zero_mech.compressibility.Compressible1())
    with pytest.raises(ValueError, match="incompressible"):
        ThickCylinder(model)


def test_unknown_parameter_names_are_rejected():
    with pytest.raises(KeyError, match="typo_mu"):
        ThickCylinder(parameters={"typo_mu": 3.0})
    sphere = ThickSphere()
    with pytest.raises(KeyError, match="typo_mu"):
        sphere.pressure(5.0, parameters={"typo_mu": 3.0})


def test_parameters_and_symbols():
    cylinder = ThickCylinder(parameters={"mu": 30.0})
    assert cylinder.parameters == {"mu": 30.0}
    reference = ThickCylinder()
    V = 1.4 * cylinder.V0
    assert cylinder.pressure(V) == pytest.approx(2 * reference.pressure(V))
    assert cylinder.pressure(V, parameters={cylinder.model["mu"]: 15.0}) == pytest.approx(
        reference.pressure(V)
    )
    assert {"R", "r_i", "mu"} <= set(cylinder.variables())
    assert cylinder.integrand_expr.free_symbols == {cylinder["R"], cylinder["r_i"], cylinder["mu"]}
    assert cylinder.circumferential_axis == 0
