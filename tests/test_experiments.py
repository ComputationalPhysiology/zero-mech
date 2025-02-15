import itertools as it
import sympy as sp

import pytest
import zero_mech


def equal(a, b):
    return sp.simplify(a - b) == 0


@pytest.mark.parametrize("axis", range(3))
def test_uniaxial_tension(axis):
    xx = axis
    yy = (axis + 1) % 3
    zz = (axis + 2) % 3

    experiment = zero_mech.experiments.uniaxial_tension(axis=axis)
    mat = zero_mech.material.NeoHookean()
    comp = zero_mech.compressibility.Incompressible()
    act = zero_mech.active.Passive()
    model = zero_mech.Model(material=mat, compressibility=comp, active=act)

    lmbda = experiment["λ"]

    principal_stretches = experiment.principal_stretches()
    assert len(principal_stretches) == 3
    assert equal(principal_stretches[xx], lmbda)
    assert equal(principal_stretches[yy], 1 / sp.sqrt(lmbda))
    assert equal(principal_stretches[zz], 1 / sp.sqrt(lmbda))

    mu = model["μ"]
    p = model["p"]

    P = model.first_piola_kirchhoff(experiment.F)

    assert equal(P[xx, xx], lmbda * mu + p / lmbda)
    assert equal(P[yy, yy], sp.sqrt(lmbda) * p + mu / sp.sqrt(lmbda))
    assert equal(P[zz, zz], sp.sqrt(lmbda) * p + mu / sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(P[i, j], 0)

    T = model.cauchy_stress(experiment.F)
    assert equal(T[xx, xx], lmbda * P[xx, xx])
    assert equal(T[yy, yy], P[yy, yy] / sp.sqrt(lmbda))
    assert equal(T[zz, zz], P[zz, zz] / sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(T[i, j], 0)

    S = model.second_piola_kirchhoff(experiment.F)
    assert equal(S[xx, xx], P[xx, xx] / lmbda)
    assert equal(S[yy, yy], P[yy, yy] * sp.sqrt(lmbda))
    assert equal(S[zz, zz], P[zz, zz] * sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(S[i, j], 0)


@pytest.mark.parametrize("axis", range(3))
def test_active_uniaxial_tension(axis):
    xx = axis
    yy = (axis + 1) % 3
    zz = (axis + 2) % 3

    f0 = sp.Matrix([0, 0, 0])
    f0[axis] = 1

    experiment = zero_mech.experiments.uniaxial_tension(axis=axis)
    mat = zero_mech.material.NeoHookean()
    comp = zero_mech.compressibility.Incompressible()
    act = zero_mech.active.ActiveStress(f0=f0)
    model = zero_mech.Model(material=mat, compressibility=comp, active=act)

    lmbda = experiment["λ"]

    principal_stretches = experiment.principal_stretches()
    assert len(principal_stretches) == 3
    assert equal(principal_stretches[xx], lmbda)
    assert equal(principal_stretches[yy], 1 / sp.sqrt(lmbda))
    assert equal(principal_stretches[zz], 1 / sp.sqrt(lmbda))

    mu = model["μ"]
    Ta = model["Ta"]
    p = model["p"]

    P = model.first_piola_kirchhoff(experiment.F)
    assert equal(P[xx, xx], 2 * Ta * lmbda + lmbda * mu + p / lmbda)
    assert equal(P[yy, yy], sp.sqrt(lmbda) * p + mu / sp.sqrt(lmbda))
    assert equal(P[zz, zz], sp.sqrt(lmbda) * p + mu / sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(P[i, j], 0)

    T = model.cauchy_stress(experiment.F)
    assert equal(T[xx, xx], lmbda * P[xx, xx])
    assert equal(T[yy, yy], P[yy, yy] / sp.sqrt(lmbda))
    assert equal(T[zz, zz], P[zz, zz] / sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(T[i, j], 0)

    S = model.second_piola_kirchhoff(experiment.F)
    assert equal(S[xx, xx], P[xx, xx] / lmbda)
    assert equal(S[yy, yy], P[yy, yy] * sp.sqrt(lmbda))
    assert equal(S[zz, zz], P[zz, zz] * sp.sqrt(lmbda))
    for i, j in ((x, y) for (x, y) in it.product(range(3), repeat=2) if x != y):
        assert sp.Eq(S[i, j], 0)
