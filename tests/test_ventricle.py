import math
import warnings

import numpy as np
import pytest
import sympy as sp

import zero_mech
from zero_mech import SphericalVentricle


SOFT = {"a": 0.3, "b": 5.0, "a_f": 1.0, "b_f": 4.0, "a_s": 0.0, "a_fs": 0.0}


def neo_hookean():
    return zero_mech.Model(material=zero_mech.material.NeoHookean())


@pytest.mark.parametrize("axis", range(3))
def test_equibiaxial_is_volume_preserving(axis):
    experiment = zero_mech.experiments.equibiaxial(axis=axis)
    lmbda = experiment["λ"]
    assert sp.simplify(experiment.J - 1) == 0
    diagonal = [experiment.F[i, i] for i in range(3)]
    assert sp.simplify(diagonal.pop(axis) - lmbda**-2) == 0
    assert all(sp.simplify(d - lmbda) == 0 for d in diagonal)


def test_equibiaxial_rejects_unknown_axis():
    with pytest.raises(ValueError):
        zero_mech.experiments.equibiaxial(axis=3)


def test_stretch_and_volume_are_inverse():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    V = np.linspace(10.0, 300.0, 30)
    assert ventricle.stretch(50.0) == pytest.approx(1.0)
    assert np.allclose(ventricle.volume(ventricle.stretch(V)), V)
    h = 1e-4
    slope = (ventricle.stretch(V + h) - ventricle.stretch(V - h)) / (2 * h)
    assert np.allclose(ventricle.dstretch_dV(V), slope, rtol=1e-7)


def test_thin_wall_stretch_follows_the_cavity_radius():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=1e-6)
    assert ventricle.stretch(100.0) == pytest.approx(2.0 ** (1 / 3), rel=1e-7)


def test_unloaded_ventricle_has_no_pressure():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    assert ventricle.pressure(50.0) == pytest.approx(0.0, abs=1e-12)


def test_passive_pressure_rises_with_volume():
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    ventricle = SphericalVentricle(model, V0=50.0, Vw=150.0)
    V = np.linspace(50.0, 90.0, 40)
    p = ventricle.pressure(V)
    assert np.all(np.diff(p) > 0)
    assert np.all(ventricle.stiffness(V) > 0)


def test_neo_hookean_closed_form():
    mu, V0, Vw = 15.0, 50.0, 150.0
    ventricle = SphericalVentricle(neo_hookean(), V0=V0, Vw=Vw)
    V = np.linspace(20.0, 200.0, 25)
    lmbda = ventricle.stretch(V)
    # psi = mu / 2 (2 lmbda^2 + lmbda^-4 - 3)
    expected = Vw * 2 * mu * (lmbda - lmbda**-5) * lmbda / (3 * V + Vw)
    assert np.allclose(ventricle.pressure(V), expected)


def test_thin_wall_is_laplace():
    # Laplace in the deformed sphere: p = 2 sigma h / r, with the in-plane Cauchy stress
    # sigma = mu (lmbda^2 - lmbda^-4) of a neo-Hookean sheet, r = R0 lmbda, h = h0 / lmbda^2
    mu, R0, h0 = 15.0, 1.0, 1e-5
    ventricle = SphericalVentricle.from_radius(R0, h0, model=neo_hookean())
    for lmbda in (0.9, 1.1, 1.3):
        V = 4 * math.pi * (R0 * lmbda) ** 3 / 3
        sigma = mu * (lmbda**2 - lmbda**-4)
        laplace = 2 * sigma * (h0 / lmbda**2) / (R0 * lmbda)
        assert ventricle.pressure(V) == pytest.approx(laplace, rel=1e-4)


def test_from_radius():
    ventricle = SphericalVentricle.from_radius(2.0, 0.5, model=neo_hookean())
    assert ventricle.V0 == pytest.approx(4 * math.pi * 8 / 3)
    assert ventricle.Vw == pytest.approx(4 * math.pi * (2.5**3 - 8) / 3)


def test_pressure_work_is_the_strain_energy_of_the_wall():
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    ventricle = SphericalVentricle(model, V0=50.0, Vw=150.0)
    V = np.linspace(50.0, 80.0, 4001)
    p = ventricle.pressure(V)
    work = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(V)))

    experiment = zero_mech.experiments.equibiaxial()
    psi = model.strain_energy(experiment.F).subs(model.material.default_parameters())
    energy = ventricle.Vw * float(psi.subs(experiment["λ"], ventricle.stretch(80.0)))
    assert work == pytest.approx(energy, rel=1e-6)


def test_active_tension_is_a_fibre_stress():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    V = np.linspace(30.0, 150.0, 13)
    active = ventricle.pressure(V, Ta=40.0) - ventricle.pressure(V)
    assert np.allclose(active, 40.0 / (1 + 3 * V / 150.0))
    assert np.allclose(active, ventricle.active_pressure(V, 40.0))
    assert np.allclose(ventricle.pressure(V, Ta=80.0) - ventricle.pressure(V), 2 * active)


def test_stiffness_is_the_derivative_of_the_pressure():
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    ventricle = SphericalVentricle(model, V0=50.0, Vw=150.0)
    V, h = np.linspace(40.0, 85.0, 10), 1e-5
    for Ta in (0.0, 30.0):
        slope = (ventricle.pressure(V + h, Ta) - ventricle.pressure(V - h, Ta)) / (2 * h)
        assert np.allclose(ventricle.stiffness(V, Ta), slope, rtol=1e-6)


def test_parameters_can_be_set_and_overridden():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0, parameters={"mu": 30.0})
    reference = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    assert ventricle.parameters == {"mu": 30.0}
    assert ventricle.pressure(80.0) == pytest.approx(2 * reference.pressure(80.0))
    mu = ventricle.model["mu"]
    assert ventricle.pressure(80.0, parameters={mu: 15.0}) == pytest.approx(
        reference.pressure(80.0)
    )
    with pytest.raises(KeyError):
        ventricle.pressure(80.0, parameters={"nu": 1.0})


def test_a_model_with_its_own_active_energy_contributes_through_it():
    # The model's Ta is not the active tension of the ventricle: it enters through the
    # strain energy, psi_active = Ta / 2 (lmbda^2 - 1), and the class warns about it
    model = zero_mech.Model(
        material=zero_mech.material.NeoHookean(), active=zero_mech.active.ActiveStress()
    )
    with pytest.warns(UserWarning, match="active part"):
        ventricle = SphericalVentricle(model, V0=50.0, Vw=150.0)
    assert set(ventricle.parameters) == {"Ta", "mu"}
    V = 90.0
    lmbda = ventricle.stretch(V)
    through_the_model = ventricle.pressure(V, parameters={"Ta": 10.0}) - ventricle.pressure(V)
    assert through_the_model == pytest.approx(150.0 * 10.0 * lmbda * ventricle.dstretch_dV(V))
    as_fibre_tension = ventricle.pressure(V, Ta=10.0) - ventricle.pressure(V)
    assert through_the_model == pytest.approx(lmbda**2 * as_fibre_tension)

    with pytest.warns(UserWarning, match="active part"):
        zero_mech.TriSeg(model)


def test_a_passive_model_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
        zero_mech.TriSeg(neo_hookean())


def test_unknown_parameter_names_are_rejected_everywhere():
    with pytest.raises(KeyError, match="typo_mu"):
        SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0, parameters={"typo_mu": 3.0})
    with pytest.raises(KeyError, match="typo_mu"):
        zero_mech.TriSeg(neo_hookean(), parameters={"typo_mu": 3.0})
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    with pytest.raises(KeyError, match="typo_mu"):
        ventricle.pressure(80.0, parameters={"typo_mu": 3.0})
    with pytest.raises(KeyError, match="typo_mu"):
        ventricle.stiffness(80.0, parameters={"typo_mu": 3.0})


def test_a_parameter_the_pressure_does_not_depend_on_is_accepted():
    # The fibre-sheet shear term of Holzapfel-Ogden vanishes in an equibiaxial deformation,
    # so a_fs is a parameter of the model that the pressure does not depend on
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    ventricle = SphericalVentricle(model, V0=50.0, Vw=150.0, parameters={"a_fs": 0.0})
    assert "a_fs" not in ventricle.parameters
    assert ventricle.pressure(80.0, parameters={"a_fs": 5.0}) == ventricle.pressure(80.0)
    triseg = zero_mech.TriSeg(model, parameters={**SOFT, "b_fs": 1.0})
    assert "b_fs" not in triseg.parameters
    assert triseg.pressures(110.0, 120.0, parameters={"a_fs": 5.0}) == triseg.pressures(
        110.0, 120.0
    )


@pytest.mark.parametrize("kwargs", [dict(V0=0.0, Vw=1.0), dict(V0=1.0, Vw=0.0)])
def test_rejects_non_positive_volumes(kwargs):
    with pytest.raises(ValueError):
        SphericalVentricle(neo_hookean(), **kwargs)


def test_symbols_are_reachable_by_name():
    ventricle = SphericalVentricle(neo_hookean(), V0=50.0, Vw=150.0)
    assert ventricle["V"] is ventricle.V
    assert ventricle["T_a"] is ventricle.Ta
    assert "mu" in ventricle


# ----------------------------------------------------------------------
# TriSeg
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def triseg():
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    return zero_mech.TriSeg(model, parameters=SOFT)


def cap(xm, ym):
    """Midwall volume, area and curvature of a spherical cap (Lumens et al. 2009, Eqs. 9-11)."""
    return (
        math.pi / 6 * xm * (xm**2 + 3 * ym**2),
        math.pi * (xm**2 + ym**2),
        2 * xm / (xm**2 + ym**2),
    )


@pytest.mark.parametrize(
    "V_LV, V_RV, Ta",
    [
        (120.0, 120.0, 0.0),
        (60.0, 70.0, 60.0),
        (100.0, 200.0, (40.0, 40.0, 40.0)),
        (30.0, 40.0, 0.0),
    ],
)
def test_triseg_geometry_and_junction(triseg, V_LV, V_RV, Ta):
    state = triseg.solve(V_LV, V_RV, Ta)
    Vw_L, Vw_S, Vw_R = triseg.Vw
    Vm, Am, Cm = zip(*(cap(x, state.ym) for x in state.xm))

    # The caps enclose the cavities (Eqs. 3-4)
    assert Vm[0] == pytest.approx(-V_LV - Vw_L / 2 - Vw_S / 2 + Vm[1], rel=1e-10)
    assert Vm[2] == pytest.approx(V_RV + Vw_R / 2 + Vw_S / 2 + Vm[1], rel=1e-10)
    assert state.Am == pytest.approx(Am)
    assert state.Cm == pytest.approx(Cm)

    # Strain of each wall (Eq. 15)
    for i in range(3):
        z = 3 * Cm[i] * triseg.Vw[i] / (2 * Am[i])
        eps = 0.5 * math.log(Am[i] / triseg.Am_ref[i]) - z**2 / 12 - 0.019 * z**4
        assert state.stretch[i] == pytest.approx(math.exp(eps))

    # The tensions balance in the junction (Eqs. 5-6, 12-13)
    sin = [2 * x * state.ym / (x**2 + state.ym**2) for x in state.xm]
    cos = [(state.ym**2 - x**2) / (x**2 + state.ym**2) for x in state.xm]
    scale = max(abs(t) for t in state.Tm)
    assert sum(t * s for t, s in zip(state.Tm, sin)) == pytest.approx(0.0, abs=1e-8 * scale)
    assert sum(t * c for t, c in zip(state.Tm, cos)) == pytest.approx(0.0, abs=1e-8 * scale)

    # Each wall carries its transmural pressure by Laplace's law, and the septum carries
    # the difference between the ventricles (Eqs. 7-8, 14)
    assert state.p_LV == pytest.approx(-2 * state.Tm[0] * Cm[0])
    assert state.p_RV == pytest.approx(2 * state.Tm[2] * Cm[2])
    assert state.p_LV - state.p_RV == pytest.approx(2 * state.Tm[1] * Cm[1], abs=1e-7 * scale)


@pytest.mark.parametrize("wall_scale, tolerance", [(1.0, 4e-2), (0.1, 1e-3)])
def test_triseg_pressure_work_is_the_strain_energy_of_the_walls(wall_scale, tolerance):
    # Pump work of both cavities equals the work of the fibres in the three walls. The model
    # uses series in z (wall thickness over radius of curvature) for strain and tension, so
    # this holds to a few percent for a normal heart and exactly in the limit of thin walls.
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    triseg = zero_mech.TriSeg(
        model, Vw=[wall_scale * v for v in (75.0, 40.0, 30.0)], parameters=SOFT
    )
    experiment = zero_mech.experiments.equibiaxial()
    values = {**model.material.default_parameters()}
    values.update({model[k]: v for k, v in SOFT.items()})
    psi = sp.lambdify(experiment["λ"], model.strain_energy(experiment.F).subs(values))

    n = 400
    V_LV, V_RV = np.linspace(60.0, 130.0, n), np.linspace(140.0, 80.0, n)
    states, guess = [], None
    for vl, vr in zip(V_LV, V_RV):
        states.append(triseg.solve(vl, vr, guess=guess))
        guess = states[-1].guess
    p_LV = np.array([s.p_LV for s in states])
    p_RV = np.array([s.p_RV for s in states])
    work = np.sum(0.5 * (p_LV[1:] + p_LV[:-1]) * np.diff(V_LV))
    work += np.sum(0.5 * (p_RV[1:] + p_RV[:-1]) * np.diff(V_RV))
    energy = sum(
        triseg.Vw[i] * (psi(states[-1].stretch[i]) - psi(states[0].stretch[i])) for i in range(3)
    )
    assert work == pytest.approx(energy, rel=tolerance)


def test_triseg_identical_free_walls_give_a_flat_septum():
    model = zero_mech.Model(material=zero_mech.material.NeoHookean())
    triseg = zero_mech.TriSeg(model, Vw=(60.0, 40.0, 60.0), Am_ref=(70.0, 40.0, 70.0))
    state = triseg.solve(100.0, 100.0, Ta=20.0, guess=(0.3, 3.5))
    assert state.xm[1] == pytest.approx(0.0, abs=1e-8)
    assert state.xm[0] == pytest.approx(-state.xm[2])
    assert state.p_LV == pytest.approx(state.p_RV)
    assert state.p_LV > 0


def test_triseg_right_ventricular_load_flattens_the_septum(triseg):
    low = triseg.solve(100.0, 100.0, Ta=40.0)
    high = triseg.solve(100.0, 220.0, Ta=40.0)
    assert high.p_RV > low.p_RV
    assert low.Cm[1] > high.Cm[1]
    # and the left ventricle feels it: same volume, different pressure
    assert high.p_LV != pytest.approx(low.p_LV, rel=1e-3)


def test_triseg_active_tension_raises_both_pressures(triseg):
    passive = triseg.solve(110.0, 120.0)
    active = triseg.solve(110.0, 120.0, Ta=30.0)
    assert active.p_LV > passive.p_LV and active.p_RV > passive.p_RV
    same = triseg.solve(110.0, 120.0, Ta=(30.0, 30.0, 30.0))
    assert same.p_LV == pytest.approx(active.p_LV)
    # A tension in the right free wall alone raises the right pressure most
    right = triseg.solve(110.0, 120.0, Ta=(0.0, 0.0, 30.0))
    assert right.p_RV - passive.p_RV > abs(right.p_LV - passive.p_LV)


def test_triseg_active_stiffness(triseg):
    reference = triseg.solve(110.0, 120.0, Ta=30.0)
    # Around the stretch at which Ta holds, the stiffness changes nothing
    same = triseg.solve(110.0, 120.0, Ta=30.0, Ka=500.0, stretch_ref=reference.stretch)
    assert same.xm == pytest.approx(reference.xm, rel=1e-8)
    assert same.p_LV == pytest.approx(reference.p_LV)
    # A wall that is a spring, sigma = Ka (stretch - 0.8), is the same wall with that tension
    spring = triseg.solve(110.0, 120.0, Ka=150.0, stretch_ref=0.8)
    tension = [150.0 * (s - 0.8) for s in spring.stretch]
    assert all(t > 0 for t in tension)
    explicit = triseg.solve(110.0, 120.0, Ta=tension, guess=spring.guess)
    assert explicit.xm == pytest.approx(spring.xm, rel=1e-7)
    assert explicit.p_LV == pytest.approx(spring.p_LV)
    assert explicit.p_RV == pytest.approx(spring.p_RV)
    with pytest.raises(ValueError):
        triseg.solve(110.0, 120.0, Ka=(1.0, 2.0))


@pytest.mark.parametrize(
    "value",
    [30, 30.0, np.int64(30), np.float32(30.0), np.float64(30.0), np.array(30.0), np.full(3, 30.0)],
    ids=lambda v: f"{type(v).__name__}{getattr(v, 'shape', '')}",
)
def test_triseg_accepts_any_scalar_or_three_values(triseg, value):
    reference = triseg.solve(110.0, 120.0, Ta=(30.0, 30.0, 30.0))
    state = triseg.solve(110.0, 120.0, Ta=value)
    assert state.p_LV == pytest.approx(reference.p_LV)
    assert state.p_RV == pytest.approx(reference.p_RV)


def test_triseg_converges_to_the_relative_tolerance(triseg):
    for tol in (1e-6, 1e-12):
        state = triseg.solve(110.0, 120.0, Ta=30.0, tol=tol)
        assert math.hypot(*state.residual) <= tol * max(abs(t) for t in state.Tm)
    # The unloaded heart: no tension anywhere, which the relative test must accept
    model = zero_mech.Model(material=zero_mech.material.NeoHookean())
    unloaded = zero_mech.TriSeg(model, Vw=(1e-9, 1e-9, 1e-9), Am_ref=(70.0, 40.0, 70.0))
    assert math.isfinite(unloaded.solve(100.0, 100.0).p_LV)


def test_triseg_solution_does_not_depend_on_the_guess(triseg):
    reference = triseg.solve(90.0, 110.0, Ta=25.0)
    nearby = triseg.solve(100.0, 100.0, Ta=25.0)
    continued = triseg.solve(90.0, 110.0, Ta=25.0, guess=nearby.guess)
    assert continued.ym == pytest.approx(reference.ym, rel=1e-8)
    assert continued.xm == pytest.approx(reference.xm, rel=1e-8)
    assert triseg.pressures(90.0, 110.0, 25.0) == pytest.approx((reference.p_LV, reference.p_RV))


def test_triseg_parameters_and_symbols(triseg):
    assert triseg.parameters["a_f"] == 1.0
    assert triseg["V_LV"] is triseg.V_LV
    assert "ym" in triseg and "a" in triseg
    stiffer = triseg.pressures(120.0, 120.0, parameters={"a": 0.6})
    assert stiffer[0] > triseg.pressures(120.0, 120.0)[0]
    with pytest.raises(KeyError):
        triseg.solve(120.0, 120.0, parameters={"nu": 1.0})


def test_triseg_rejects_bad_input(triseg):
    with pytest.raises(ValueError):
        triseg.solve(-1.0, 100.0)
    with pytest.raises(ValueError):
        triseg.solve(100.0, 100.0, Ta=(1.0, 2.0))
    with pytest.raises(ValueError):
        zero_mech.TriSeg(Vw=(1.0, 2.0))
    with pytest.raises(ValueError):
        zero_mech.TriSeg(Am_ref=(1.0, 0.0, 1.0))
