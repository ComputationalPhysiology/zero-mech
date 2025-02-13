import zero_mech
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

experiment = zero_mech.experiments.uniaxial_tension(axis=0)
mat = zero_mech.material.NeoHookean()
comp = zero_mech.compressibility.Incompressible()
act = zero_mech.active.Passive()
model = zero_mech.Model(material=mat, compressibility=comp, active=act)

lmbda = experiment["lmbda"]
p = model["p"]
mu = model["μ"]
mu_value = 10.0

P = model.first_piola_kirchhoff(experiment.F)
p_sym = sp.solve(P[1, 1], p)[0]
# p_sym = sp.solve(P[0, 0], p)[0]
T = model.cauchy_stress(experiment.F)
S = model.second_piola_kirchhoff(experiment.F)
Txx = []
Sxx = []
Exx = []
Syy = []
Eyy = []
stretch = np.concatenate([
    np.linspace(1.0, 1.5, 10),
    np.linspace(1.5, 0.7, 20),
    np.linspace(0.7, 1.0, 10),
])

for lmbda_value in stretch:
    p_value = p_sym.subs({mu: mu_value, lmbda: lmbda_value})
    Exx.append(experiment.E[0, 0].subs({lmbda: lmbda_value}))
    Eyy.append(experiment.E[1, 1].subs({lmbda: lmbda_value}))
    Txx.append(T[0, 0].subs({p: p_value, mu: mu_value, lmbda: lmbda_value}))
    Sxx.append(S[0, 0].subs({p: p_value, mu: mu_value, lmbda: lmbda_value}))
    Syy.append(S[1, 1].subs({p: p_value, mu: mu_value, lmbda: lmbda_value}))


fig, ax = plt.subplots()
ax.plot(Exx, Sxx, label="Sxx")
ax.plot(Eyy, Syy, label="Syy")
ax.legend()
ax.set_xlabel("Strain")
ax.set_ylabel("Stress")
fig.savefig("uniaxial_tension.png")
