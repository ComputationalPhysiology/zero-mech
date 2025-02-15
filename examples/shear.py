# # Shear test

# This example demonstrates how to perform a simple shear test and is a reproduction of Figure 7 in {cite}`holzapfel2009constitutive`

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import zero_mech

mat = zero_mech.material.HolzapfelOgden()
print(zero_mech.material.HolzapfelOgden.str())


comp = zero_mech.compressibility.Incompressible()
act = zero_mech.active.Passive()
model = zero_mech.Model(material=mat, compressibility=comp, active=act)

p = model["p"]

stress = {
    "fs": [],
    "sn": [],
    "fn": [],
    "sf": [],
    "ns": [],
    "nf": [],
}

material_params = mat.default_parameters()

str2index = {"f": 0, "s": 1, "n": 2}
stretch = np.linspace(0.0, 0.5, 20)

for plane in ["fs", "sn", "fn", "sf", "ns", "nf"]:
    experiment = zero_mech.experiments.simple_shear(plane=plane)

    gamma = experiment["gamma"]
    P = model.first_piola_kirchhoff(experiment.F)
    p_sym = sp.solve(P[0, 0], p)[0]

    T = model.cauchy_stress(experiment.F)
    S = model.second_piola_kirchhoff(experiment.F)

    idx1 = str2index[plane[0]]
    idx2 = str2index[plane[1]]

    for gamma_value in stretch:
        p_value = p_sym.subs({gamma: gamma_value, **material_params})
        stress[plane].append(T[idx1, idx2].subs({p: p_value, gamma: gamma_value, **material_params}))


fig, ax = plt.subplots()
for k, v in stress.items():
    ax.plot(stretch, v, label=k)
ax.legend()
ax.set_xlabel("Strain")
ax.set_ylabel("Stress")
fig.savefig("simple_shear.png")

#
# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
