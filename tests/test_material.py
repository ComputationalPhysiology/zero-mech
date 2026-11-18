import numpy as np
import pytest
import sympy as sp

import zero_mech
from zero_mech.material import expm1_div

#: A deformation with every invariant of Holzapfel-Ogden away from its rest value
F_VALUES = sp.Matrix([[1.15, 0.1, 0.05], [0.08, 0.95, 0.02], [0.03, 0.06, 0.93]])


def energy(parameters, use_heaviside=True):
    material = zero_mech.material.HolzapfelOgden(use_heaviside=use_heaviside)
    return float(
        material.strain_energy(F_VALUES).xreplace(
            {getattr(material, k): v for k, v in parameters.items()}
        )
    )


def invariants():
    f0, s0 = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0])
    C = F_VALUES.T @ F_VALUES
    return (
        float(sp.trace(C)),
        float(f0.dot(C @ f0)),
        float(s0.dot(C @ s0)),
        float(f0.dot(C @ s0)),
    )


A = dict(a=2.0, a_f=3.0, a_s=4.0, a_fs=5.0)


def test_holzapfel_ogden_with_zero_exponents_is_the_limit():
    I1, I4f, I4s, I8fs = invariants()
    assert I4f > 1 and I4s < 1  # so that the heaviside of the sheet term switches it off
    zero = {**A, "b": 0.0, "b_f": 0.0, "b_s": 0.0, "b_fs": 0.0}
    expected = A["a"] / 2 * (I1 - 3) + A["a_f"] / 2 * (I4f - 1) ** 2 + A["a_fs"] / 2 * I8fs**2
    assert energy(zero) == pytest.approx(expected, rel=1e-14)
    without_heaviside = expected + A["a_s"] / 2 * (I4s - 1) ** 2
    assert energy(zero, use_heaviside=False) == pytest.approx(without_heaviside, rel=1e-14)


def test_holzapfel_ogden_is_continuous_in_its_exponents():
    for b in ("b", "b_f", "b_s", "b_fs"):
        zero = {**A, "b": 1.0, "b_f": 1.0, "b_s": 1.0, "b_fs": 1.0, b: 0.0}
        small = {**zero, b: 1e-7}
        assert energy(small, use_heaviside=False) == pytest.approx(
            energy(zero, use_heaviside=False), rel=1e-6
        )


def test_pulse_transversely_isotropic_parameters_give_a_finite_stress():
    # a_s = b_s = a_fs = b_fs = 0, which divided zero by zero before the limit was taken
    parameters = dict(a=2.280, b=9.726, a_f=1.685, b_f=15.779, a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0)
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden(use_heaviside=True))
    experiment = zero_mech.experiments.uniaxial_tension(lmbda=sp.Float(1.1))
    P = model.first_piola_kirchhoff(experiment.F).xreplace(
        {model[k]: v for k, v in parameters.items()} | {model["p"]: 0.0}
    )
    values = np.array(P.evalf(), dtype=float)
    assert np.all(np.isfinite(values))
    # The same as dropping the two terms by any positive exponent
    reference = model.first_piola_kirchhoff(experiment.F).xreplace(
        {model[k]: v for k, v in parameters.items()}
        | {model["p"]: 0.0, model["b_s"]: 1.0, model["b_fs"]: 1.0}
    )
    assert np.allclose(values, np.array(reference.evalf(), dtype=float), rtol=1e-14, atol=0)


def test_holzapfel_ogden_lambdifies_with_zero_exponents():
    material = zero_mech.material.HolzapfelOgden()
    symbols = [material.a, material.b, material.a_f, material.b_f]
    f = sp.lambdify(
        symbols,
        material.strain_energy(F_VALUES).xreplace(
            {material.a_s: 0.0, material.b_s: 0.0, material.a_fs: 0.0, material.b_fs: 0.0}
        ),
        "numpy",
    )
    I1, I4f, *_ = invariants()
    with np.errstate(all="raise"):
        value = f(2.0, 0.0, 3.0, 0.0)
    assert value == pytest.approx(1.0 * (I1 - 3) + 1.5 * (I4f - 1) ** 2)


def test_holzapfel_ogden_stress_lambdifies_cleanly_with_zero_exponents():
    material = zero_mech.material.HolzapfelOgden()
    model = zero_mech.Model(material=material)
    P = model.first_piola_kirchhoff(F_VALUES).xreplace({model["p"]: 0.0})
    symbols = [model[k] for k in ("a", "b", "a_f", "b_f", "a_s", "b_s", "a_fs", "b_fs")]
    f = sp.lambdify(symbols, P, "numpy")
    with np.errstate(all="raise"):
        values = np.array(f(2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0), dtype=float)
    assert np.all(np.isfinite(values))


def test_holzapfel_ogden_str_names_the_sheet_invariant():
    assert "exp(b_s * H(I4s - 1)**2)" in zero_mech.material.HolzapfelOgden.str()


def test_holzapfel_ogden_lambdifies_with_math_and_zero_exponents():
    material = zero_mech.material.HolzapfelOgden()
    symbols = [material.a, material.b, material.a_f, material.b_f]
    f = sp.lambdify(
        symbols,
        material.strain_energy(F_VALUES).xreplace(
            {material.a_s: 0.0, material.b_s: 0.0, material.a_fs: 0.0, material.b_fs: 0.0}
        ),
        "math",
    )
    I1, I4f, *_ = invariants()
    assert f(2.0, 0.0, 3.0, 0.0) == pytest.approx(1.0 * (I1 - 3) + 1.5 * (I4f - 1) ** 2)
    assert f(2.0, 1.0, 3.0, 2.0) == pytest.approx(
        energy(
            {**A, "b": 1.0, "b_f": 2.0, "a_s": 0.0, "a_fs": 0.0, "b_s": 1.0, "b_fs": 1.0},
            use_heaviside=False,
        )
    )


def test_holzapfel_ogden_with_a_number_for_b_is_the_plain_exponential():
    material = zero_mech.material.HolzapfelOgden()
    x = sp.Symbol("x")
    term = expm1_div(material.b, x)
    assert term.subs(material.b, 2) == (sp.exp(2 * x) - 1) / 2
    assert term.subs(material.b, 0) == x
    # A negative exponent keeps the exponential, which is exact there too
    assert term.subs(material.b, -1) == 1 - sp.exp(-x)
    assert sp.diff(term, x) == sp.exp(material.b * x)


def test_solve_for_the_pressure_in_simple_shear():
    # A Piecewise in b made sympy.solve fail to convert its solution to floats
    model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
    experiment = zero_mech.experiments.simple_shear(plane="fs")
    P = model.first_piola_kirchhoff(experiment.F)
    (p,) = sp.solve(P[0, 0], model["p"])
    assert not p.has(sp.Piecewise)
