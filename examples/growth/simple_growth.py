# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Simple growth demo
#
# In this example we will see how to simulate growth of a material. Growth and remodelling happens for example in biological tissue when muscle cells grow due to different stimuli (i.e during exercise or disease).
#
# One way to model growth is do decompose the deformation gradient into a growth part ($\mathbf{G}$) and an elastic deformation ($A$) which ensures compatibility of the tissue
#
# $$
# \mathbf{F} = \mathbf{A}\mathbf{G}
# $$
#
# The energy of the system will only depend on the elastic deformation $\mathbf{A}$, and therefore we often would like to compute
#
# $$
# \mathbf{A} = \mathbf{F} \mathbf{G}^{-1}
# $$

import zero_mech
import sympy as sp
from IPython.display import display

# Let us define an uniaxial tension experiment where the material is stretch / compressed along the $x$-axis.

experiment = zero_mech.experiments.uniaxial_tension()
F = experiment.F
lmbda = experiment["λ"]
display(F)


# Now, lets us create a growth tensor which we assume will grow for a certain amount in the $x$-direction and an equal amount (but possible different) in the two other directions. Since we also know that this tensor is diagonal we create a helper function to compute the inverse (which is less expensive than letting `sympy` handle this in a general case)

# +
def g_inv(G):
    return sp.Matrix([[1 / G[0, 0], 0, 0], [0, 1 / G[1, 1], 0], [0, 0, 1 / G[2, 2]]])

Gff, Gtt = sp.symbols("G_ff, G_tt")
G = sp.Matrix([[Gff, 0, 0], [0, Gtt, 0], [0, 0, Gtt]])
display(G)
# -

# We can now copute the elastic deformation

A = F * g_inv(G)
display(A)

# We can set up a simple mechanical model using an incompressible Neo Hookean material

material = zero_mech.material.NeoHookean()
mech_model = zero_mech.Model(
    material=material,
    compressibility=zero_mech.compressibility.Incompressible(),
    active=zero_mech.active.Passive()
)
print(mech_model)

# And we can evalue the strain energy function

mech_model.strain_energy(A).simplify()

# or the stress

T = mech_model.cauchy_stress(A).simplify()
display(T)


# We will now perform a simple experiment where we will see what happens to the material as it grows over time. To do this, we need to define some rule for how the material should grow, i.e and ODE that tells us something about how the growth changes in response to a change in stress or strain
#
# $$
# \frac{\partial \mathbf{G}}{\partial t} = f(t, \mathbf{G}, \mathbf{A}, \sigma)
# $$
#
# where $\sigma$ is the Cauchy stress. Numerically we can discretize this using a forward Euler scheme
#
# $$
# \mathbf{G}_{n+1} = \mathbf{G}_n + \Delta t f(t_n, \mathbf{G}_n, \mathbf{A}_n, \sigma_n)
# $$
#
# Now let us start by defining a simple rule that grow the material in the $x$ direction whenever the elastic Green-Lagrange strain ($\mathbf{E}$) in the $x$-direction ($E_xx$) is above some threshold $s_0$. We would also like to stop the growth when $E_xx$ reaches $s_0$, so if there is growth occurring and this happens then we want to somehow stop the growth.
#
# The Green-Lagrange strain is given by
#
# $$
# \mathbf{E} = \frac{1}{2}(\mathbf{A}^T\mathbf{A} - \mathbf{I})
# $$
#
# and to incorporate this we can write
#
# $$
# f(t, \mathbf{G}, \mathbf{A}, \sigma) = \begin{cases}
# \frac{E_{xx} - s_0}{k_T}   \quad \text{if } E_{xx} > s_0 \\
# \frac{1 - G_{xx}}{k_T} \quad \text{otherwise}
# \end{cases}
# $$
#
# Here $k_T$ is also a parameter with same unit as frequency (1 / time) so make sure we get the correct units (which is 1 / time).
#
# Let us write this in a simple function where we set $s_0 = 0.1$ and $k_T = 1.0 \text{ days}^{-1}$ (note that the chose of days here are completely arbitrary)

def f(t, G, A, T):
    E = 0.5 * (A.T * A - sp.eye(3))
    Exx = E[0, 0]
    s_0 = 0.1
    dG = sp.zeros(3)
    dG[0, 0] += sp.Piecewise((Exx - s_0, Exx > s_0), ((1 - G[0, 0]), True))
    return dG


# We can now look at how $G_{n+1}$ will look like for the numerical experiment we have designed

dt = sp.Symbol("dt")
t = sp.Symbol("t")
G_next = G + dt * f(t, G, A, T)
display(G_next)

# To evaluate a concrete value we can for exaple insert $G_{ff} = G_{tt} = 1.0$ (no initial growth) and $\lambda = 1.3$ (30% stretch), with a time step of $\Delta t = 1.0$ day

G_next.subs({Gff: 1, Gtt: 1, dt: 1.0, lmbda: 1.3})

# and we see that we get some growth in the $x$-direction. Similarly we can take a look at $A_{n+1}$

A_next = F * g_inv(G_next)
display(A_next)

# and see what the corresponding $A_{n+1}$ would look like if we substitute the same values

A_next.subs({Gff: 1, Gtt: 1, dt: 1.0, lmbda: 1.3})

# Note that the value of $A_n$ are as follows

A.subs({Gff: 1, Gtt: 1, dt: 1.0, lmbda: 1.3})

# So we can see that we get a reduction in the elastic strain along the $x$-axis because this is converted into growth. Note also that $\lambda$ in the next grwoth step will be the value of the $x$-component of $A_{n+1}$.
#
# Similarly let us also look at the cauchy stress. Let us also choose $p$ so that $\sigma_{xx} = 0$

mu_value = 10.0
mu = mech_model["mu"]
p = mech_model["p"]
T = mech_model.cauchy_stress(A)
p_value = sp.solve(T[0, 0], p)[0].subs({Gff: 1, Gtt: 1, dt: 1.0, lmbda: 1.3, mu: mu_value})
print(p_value)
T_value = T.subs({Gff: 1, Gtt: 1, dt: 1.0, lmbda: 1.3, mu: mu_value, p: p_value})
display(T_value)

# Now let us run a 100 growth steps with a time step of $\Delta t = 0.1$ days

# +
# Initial values
lmbda_value = 1.3
Gff_value = 1.0
Gtt_value = 1.0
dt_value = 0.1

Axx = [A[0, 0].subs({Gff: Gff_value, Gtt: Gtt_value, dt: 0, lmbda: lmbda_value})]
Ayy = [A[1, 1].subs({Gff: Gff_value, Gtt: Gtt_value, dt: 0, lmbda: lmbda_value})]
Gxx = [Gff_value]
Gyy = [Gtt_value]
Txx = [T[0, 0].subs({Gff: Gff_value, Gtt: Gtt_value, dt: 0, lmbda: lmbda_value, mu: mu_value, p: p_value})]
Tyy = [T[1, 1].subs({Gff: Gff_value, Gtt: Gtt_value, dt: 0, lmbda: lmbda_value, mu: mu_value, p: p_value})]

for i in range(100):
    A_n = A_next.xreplace({Gff: Gff_value, Gtt: Gtt_value, dt: dt_value, lmbda: lmbda_value})
    T_n = mech_model.cauchy_stress(A_n)
    p_value = sp.solve(T_n[0, 0], p)[0].xreplace({Gff: Gff_value, Gtt: Gtt_value, dt: dt_value, lmbda: lmbda_value, mu: mu_value})
    lmbda_value = A_n[0, 0]
    G_n = G_next.xreplace({Gff: Gff_value, Gtt: Gtt_value, dt: dt_value, lmbda: lmbda_value})
    Gff_value = G_n[0, 0]
    Gtt_value = G_n[1, 1]

    Axx.append(lmbda_value)
    Ayy.append(A_n[1, 1])
    Gxx.append(Gff_value)
    Gyy.append(Gtt_value)
    Txx.append(T_n[0, 0].xreplace({Gff: Gff_value, Gtt: Gtt_value, dt: dt_value, lmbda: lmbda_value, mu: mu_value, p: p_value}))
    Tyy.append(T_n[1, 1].xreplace({Gff: Gff_value, Gtt: Gtt_value, dt: dt_value, lmbda: lmbda_value, mu: mu_value, p: p_value}))

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 2, sharex=True, sharey="row", figsize=(10, 10))
ax[0, 0].plot(Axx)
ax[0, 0].set_title("Axx")
ax[0, 1].plot(Ayy)
ax[0, 1].set_title("Ayy")
ax[1, 0].plot(Gxx)
ax[1, 0].set_title("Gxx")
ax[1, 1].plot(Gyy)
ax[1, 1].set_title("Gyy")
ax[2, 0].plot(Txx)
ax[2, 0].set_title("Txx")
ax[2, 1].plot(Tyy)
ax[2, 1].set_title("Tyy")
plt.show()
