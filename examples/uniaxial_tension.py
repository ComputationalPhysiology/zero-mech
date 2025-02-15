# # Uniaxial test

# This example demonstrates how to perform a simple uniaxial experiment

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import zero_mech
from IPython.display import display


# We can set up the experiment by defining a stretch $\lambda$ along a given axis, say along the $x$-axis. The corresponding stretch along the $y$- and $z$-axis will then be $\lambda^{-1/2}$

experiment = zero_mech.experiments.uniaxial_tension(axis=0)
display(experiment.F)

# To create an experiment with the stretch acting along a different axis you just change the `axis` argument, e.g

display(zero_mech.experiments.uniaxial_tension(axis=1).F)

# Now we need to specify the material model, here we use a NeoHookean model (but we have also commented out the code for using a Holzapfel-Ogden model

# +
mat = zero_mech.material.NeoHookean()

# Here one could also use a different material model such as the Holzapfel Ogden model
#f0 = sp.Matrix([1, 0, 0])
#s0 = sp.Matrix([0, 1, 0])
#mat = zero_mech.material.HolzapfelOgden(f0=f0, s0=s0)
material_params = mat.default_parameters()
print(mat)
# -

# If you want to look at how the strain energy function looks like you can do as follows

mat.strain_energy(experiment.F)

# Now this strain energy function is specific to the uniaxial experiment. To get the full general strain energy function, you can instead create a full experiment

full = zero_mech.experiments.full_matrix()
display(full)

# and pass that the deformation gradient from this experiement in

mat.strain_energy(full.F)

# And to get the general formula you can call the `str` method

print(zero_mech.material.NeoHookean.str())

# We choose an incompressible model and creates a full model (here we also comment out the code for using a compressible model)

comp = zero_mech.compressibility.Incompressible()
#comp = zero_mech.compressibility.Compressible1()
print(comp.str())
model = zero_mech.Model(material=mat, compressibility=comp)
print(model)

# Note also the the model is purely passive (i.e there are no actively contracting cells such as in the case of muscle cells). We can take a look at the stress components

P = model.first_piola_kirchhoff(experiment.F).simplify()  # First Piola
display(P)

S = model.second_piola_kirchhoff(experiment.F).simplify() # Second Piola
display(S)

T = model.cauchy_stress(experiment.F).simplify()  # Cauchy stress
display(T)

# If you choose an incompressible model, then we need to also solve for the hydrostatic pressure `p` which is introduced as a Lagrange multiplier. In this case `p` would be a variable in the model and we can solve for the stress being zero in one of the axis orthogonal to the axis in which we extend / compress the material

if "p" in model:
    p = model["p"]
    p_sym = sp.solve(P[1, 1], p)[0]
    material_params[p] = 0.0

# We can now solve the model by varying the stretch $\lambda$ between 0.9 (10% compression) and 1.1 (10% extension), and plot the Cauchy stress

lmbda = experiment["λ"]
stretch = np.linspace(0.9, 1.1, 20)
stress_xx = []
stress_yy = []
stress_zz = []
ps = []

for lmbda_value in stretch:
    if "p" in model:
        p_value = p_sym.subs({lmbda: lmbda_value, **material_params})
        material_params[p] = p_value
        ps.append(p_value)

    stress_xx.append(T[0, 0].subs({lmbda: lmbda_value, **material_params}))
    stress_yy.append(T[1, 1].subs({lmbda: lmbda_value, **material_params}))
    stress_zz.append(T[2, 2].subs({lmbda: lmbda_value, **material_params}))

fig, ax = plt.subplots()
ax.plot(stretch, stress_xx, label="Txx")
ax.plot(stretch, stress_yy, label="Tyy")
ax.plot(stretch, stress_zz, label="Tzz")
if "p" in model:
    ax.plot(stretch, ps, label="p")
ax.legend()
ax.set_xlabel("Strain")
ax.set_ylabel("Stress")
fig.savefig("uniaxial_tension.png")
