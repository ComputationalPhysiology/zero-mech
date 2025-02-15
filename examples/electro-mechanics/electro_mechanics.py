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

# # Theory for cardiac electro-mechanics
#
# In general for strongly coupled electro-mechanics we have the following system of equations for the electrophysiolgy
#
# $$
# \frac{\partial s}{\partial t} = f(v, s, \lambda, \dot{\lambda}) \\
# $$
# $$
# \frac{\partial v}{\partial t} + I_{\text{ion}}(v, s, \lambda, \dot{\lambda}) = \nabla \cdot (M_i \nabla v) + \nabla \cdot (M_i \nabla u_e)
# $$
# $$
# \nabla \cdot ((M_i + M_e) \nabla u_e) = - \nabla \cdot (M_i \nabla v)
# $$
#
# Here the ordinary differential equations (ODEs), defined by $f$ describes the electro-chemical state of the muscle cells, characterized by the state vector $s$. Furthermore $v$ is the transmembrane potential, $\lambda$ is the fiber stretch ratio and $\dot{\lambda} = \frac{\partial \lambda}{\partial t}$ is the stretch rate. The electrical state of the tissue is described by the Bi-domain model, which could also be reduced to the Mondomain model (see {cite}`sundnes2007computing` for more info). The function $I_{\text{ion}}$ describes the ionic current accross the cell membrane, while $M_i$ and $M_e$ are intracellular and extracellular tissue conductivities respectively.
#
# The mechanical equilibrium are described by the following system of equations
#
# $$
# \nabla \cdot \mathbf{P} = 0
# $$
# $$
# \mathbf{P} = \mathbf{P}_p + \mathbf{P}_a
# $$
#
# where $\mathbf{P}$ is the first Piola-Kirrchoff stress tensor, with
#
# $$
# \mathbf{P}_p = \frac{\partial \Psi}{\partial \mathbf{F}}
# $$
#
# and
#
# $$
# \mathbf{P}_a = \mathbf{P}_a(s, \lambda, \dot{\lambda}).
# $$
#
# Here $\Psi$ is a strain energy function which relates stress to strain and depends on the material under consideration (This is often denoted as the "constitutive relation"). For examples we can choose a NeoHookean material model (which is typical for rubber-like materials)
#
# $$
# \Psi(\mathbf{F}) = \frac{\mu}{2}( \mathrm{tr}(\mathbf{F}^T \mathbf{F}) - 3)
# $$
#
# The active stress $\mathbf{P}_a$ depends on the electro-chemical state, and is typically described via a model for the crossbrigde dynamics.
#
# Now the mechanics system is typically solved for the displacement field $\mathbf{u}$, which relates coordinates in the reference configuration ($\mathbf{X}$) to coordinates in the current configuration ($\mathbf{x}$), by $\mathbf{u} = \mathbf{x} - \mathbf{X}$. We also have the deformation gradient $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$. Furthermore the stretch is typically defined as the stretch along the fiber direction. If we denote the fiber orientation in the reference configuration as $\mathbf{f}_0$, then $\lambda = | \mathbf{F}\mathbf{f}_0 | =  | \mathbf{f} | $. For example one can consider a tissue slab with fibers oriented in the $x$-direction, in which case we would have $\mathbf{f}_0 = (1 \; 0 \; 0)^T$
#
# ## Reduction to 0D
#
# To reduce the problem to 0D, one need to consider a simplified experiment. For example we can consider a rectangle of length $l_x$, a heigth of $l_y$ and a depth of $l_z$. Now imaging that we apply a deformation that changes the length from $l_x$ to $l_x + \Delta x = L_x$. We can quantify this deformation using the stretch
#
# $$
# \lambda = \frac{L_x}{l_x} = \frac{l_x + \Delta x}{l_x} = 1 + \frac{\Delta x}{l_x}
# $$
#
# If we now assume that the material is incompressible, and that the compression along the $y$- and $z$-axis are equal then we can write the following deformation gradient
#
# $$
# \mathbf{F} = \begin{pmatrix} \lambda & 0 & 0 \\ 0 & \lambda^{-1/2} & 0 \\ 0 & 0 & \lambda^{-1/2}\end{pmatrix}
# $$
#
# so that
#
# $$
# L_y = l_y \cdot \lambda^{-1/2} = \frac{l_y}{\sqrt{1 + \frac{\Delta x}{l_x}}}
# $$

# +
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(0, 4.5)
ax.set_ylim(0, 3.5)
ax.set_axis_off()

corner = (1, 1)
lx = 2
ly = 2

dx = 0.8
lmbda = (lx + dx) / lx
Lx = lx * lmbda
Ly = lx * (1 / math.sqrt(lmbda))


someX, someY = 2, 5
ax.add_patch(mpatches.Rectangle(corner, lx, ly, facecolor="none", ec='k', lw=2))
ax.add_patch(mpatches.Rectangle(corner, Lx, Ly, facecolor="none", ec='r', lw=2, linestyle="--"))

shift = 0.4
arr = mpatches.FancyArrowPatch((corner[0], corner[1]-shift), (corner[0] + lx, corner[1]-shift), arrowstyle='<->,head_width=.1', mutation_scale=20)
ax.add_patch(arr)
ax.annotate("$l_x$", (.5, .5), xycoords=arr, ha='center', va='bottom')

arr = mpatches.FancyArrowPatch((corner[0] + lx, corner[1]-shift), (corner[0] + lx + dx, corner[1]-shift),  arrowstyle='<->,head_width=.1', mutation_scale=20)
ax.add_patch(arr)
ax.annotate(r"$\Delta x$", (.5, .5), xycoords=arr, ha='center', va='bottom')

shift = 0.2
arr = mpatches.FancyArrowPatch((corner[0]-shift, corner[1]), (corner[0] - shift, corner[1] + ly), arrowstyle='<->,head_width=.1', mutation_scale=20)
ax.add_patch(arr)
ax.annotate("$l_y$", (.5, .5), xycoords=arr, va='center', ha='right')

shift = 0.5
arr = mpatches.FancyArrowPatch((corner[0]-shift, corner[1]), (corner[0] - shift, corner[1] + Ly), arrowstyle='<->,head_width=.1', mutation_scale=20, color="red")
ax.add_patch(arr)
ax.annotate("$L_y$", (.5, .5), xycoords=arr, va='center', ha='right', color="red")

plt.show()
# -

# We can describe this as a uniaxial experiment in the `zero_mech` package

# +
import zero_mech
from IPython.display import display

experiment = zero_mech.experiments.uniaxial_tension()
display(experiment.F)
# -

# If we now create a passive material with a NeoHookean material model then we can compute the strain energy density as a function of $\lambda$

material = zero_mech.material.NeoHookean()
display(material.strain_energy(experiment.F))

# Say that we now assume the material has now active component and is incompressible, then we can create a model

mech_model = zero_mech.Model(
    material=material,
    compressibility=zero_mech.compressibility.Incompressible(),
    active=zero_mech.active.Passive()
)

# and compute the stress as a function of $\lambda$

display(mech_model.first_piola_kirchhoff(experiment.F).simplify())

# Note here that the variable $p$ is introduced. This variable acts as a Lagrange multiplier to enforce incompressibility

print(mech_model.compressibility)
print(mech_model.compressibility.str())

# ## Adding active component (with no feedback)
#
# Now this becomes much more interesting when we allow the material to actively contract. As we saw from the system above, the active tension typically depends on the state from the electro-chemical state of the muscle cells. Assume for the moment that there is no feedback from the mechanics (i.e that the active tension does not depend on $\lambda$ or $\dot{\lambda}$. In this case we can write the active stress and a pure function of $s$
#
# $$
# \mathbf{P}_a = \mathbf{P}_a(s)
# $$
#
# We will now update our model to use an active stress model, which contains a variable $T_a$ that describes the strength of the active force and a fiber direction (which in our case is chosen to be along the $x$-axis)

active_model = zero_mech.active.ActiveStress()
display(active_model)
display(active_model.strain_energy(experiment.F))
mech_model = zero_mech.Model(
    material=material,
    compressibility=zero_mech.compressibility.Incompressible(),
    active=active_model
)

# Let us print the total stress tensor

P = mech_model.first_piola_kirchhoff(experiment.F).simplify()
display(P)

# To model the electro-chemical state of the muscle cells we will use the TorORd model {cite}`tomek2019development` coupled to the Land model for crossbridge dynamics {cite}`land2017model`. We can use [`gotranx`](https://finsberg.github.io/gotranx) to generate code for solving the ODE using the Generalized Rush Larsen scheme.

# +
from pathlib import Path
import gotranx

module_path = Path("ToRORd_dynCl_endo.py")
if not module_path.is_file():
    ode = gotranx.load_ode("ToRORd_dynCl_endo.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
    )
    module_path.write_text(code)

import ToRORd_dynCl_endo
model = ToRORd_dynCl_endo.__dict__
# -

# Let us simulate this for 400 ms with a timestep of 0.1. We will aslo keep track of the Voltage (`V`), the intracellular Calcium (`Ca`) and the active tension. Also note that we set the initial value for $p$ to zero, which is the hydrostatic pressure that enforces incompressibility

# +
import numpy as np

dt = 0.1
BCL = 400
num_beats = 1
t = np.arange(0, num_beats * BCL, dt)

p_value = 0.0

y = model["init_state_values"]()
params = model["init_parameter_values"](i_Stim_Period=BCL)

# Get the index of the membrane potential
V_index = model["state_index"]("v")
Ca_index = model["state_index"]("cai")
# Get the index of the active tension from the land model
Ta_index = model["monitor_index"]("Ta")
fgr = model["generalized_rush_larsen"]
mon = model["monitor_values"]
# -

# We set the initial $\lambda$ to 1.0 and $\dot{\lambda}$ to 0.0. Since we would need to update $\lambda$ and $\dot{\lambda}$, we need to also keep track of the previous value of $\lambda$.

# +
lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")
lmbda_value = 1.0

params[lmbda_index] = lmbda_value
prev_lmbda = lmbda_value
params[dLambda_index] = 0.0


# -

# Now, we will create a function that will solve the mechanics problem (i.e $P_{xx} = P_{yy} = 0$) for $\lambda$ and $p$, given an active stress $T_a$

def func(x, Ta):
    lmbda, p = x

    replace = {
        mech_model["p"]: p,
        mech_model["Ta"]: Ta,
        experiment["λ"]: lmbda,
        **material.default_parameters(),
    }

    P11 = P[0, 0].xreplace(replace)
    P22 = P[1, 1].xreplace(replace)
    return np.array([P11, P22], dtype=np.float64)


# Now let us loop over the time steps and in each iteration we first solve the EP problem by calling the `fgr` (forward generalized rush larsen) function, and then we solve the mechanics problem using a root-finding algorithm. An important observation to make here is that we do not update $\lambda$ or $\dot{\lambda}$ (we have commented out the relevant lines that would update these values in the EP model)

# +
from scipy.optimize import root

V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
lmbdas = np.zeros(len(t))
dLambdas = np.zeros(len(t))
ps = np.zeros(len(t))

for i, ti in enumerate(t):
    y = fgr(y, ti, dt, params)
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, params)
    Ta[i] = monitor[Ta_index]

    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(Ta[i],),
        method="hybr",
    )
    lmbda_value, p_value = res.x
    lmbdas[i] = lmbda_value
    ps[i] = p_value

    dLambda = (lmbda_value - prev_lmbda) / dt
    dLambdas[i] = dLambda

    # Update lmbda and dLambda in the model
    # params[lmbda_index] = lmbda_value
    # params[dLambda_index] = dLambda

    prev_lmbda = lmbda_value
# -

# Let us plot the results

fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, dLambdas)
ax[0, 2].plot(t, lmbdas)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel(r"$\dot{\lambda}$")
ax[0, 2].set_ylabel(r"$\lambda$")
ax[1, 2].set_ylabel("p")
for axi in ax.flatten():
    axi.grid()
fig.tight_layout()
plt.show()


# ## Adding feedback
#
# Now let us try to add feedback, which essentially means commenting out the two lines which will update $\lambda$ and $\dot{\lambda}$ in the EP model

# +
lmbda_value = 1.0
p_value = 0.0

y = model["init_state_values"]()
params = model["init_parameter_values"](i_Stim_Period=BCL)
params[lmbda_index] = lmbda_value
prev_lmbda = lmbda_value
params[dLambda_index] = 0.0

V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
lmbdas = np.zeros(len(t))
dLambdas = np.zeros(len(t))
ps = np.zeros(len(t))

for i, ti in enumerate(t):
    y = fgr(y, ti, dt, params)
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, params)
    Ta[i] = monitor[Ta_index]

    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(Ta[i],),
        method="hybr",
    )
    lmbda_value, p_value = res.x
    lmbdas[i] = lmbda_value
    ps[i] = p_value

    dLambda = (lmbda_value - prev_lmbda) / dt
    dLambdas[i] = dLambda

    # Update lmbda and dLambda in the model
    params[lmbda_index] = lmbda_value
    params[dLambda_index] = dLambda

    prev_lmbda = lmbda_value
# -

fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, dLambdas)
ax[0, 2].plot(t, lmbdas)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel(r"$\dot{\lambda}$")
ax[0, 2].set_ylabel(r"$\lambda$")
ax[1, 2].set_ylabel("p")
for axi in ax.flatten():
    axi.grid()
fig.tight_layout()
plt.show()


# Yikes!! That did not work well. We see that the solution starts oscillating. The reason this happens is because $T_a$ depends heavily on $\lambda$ and $\dot{\lambda}$, but we also have that $\lambda$ and $\dot{\lambda}$ depends heavily on $T_a$. For example if we stretch the material (i.e increase $\lambda$), then the [Frank Starling law](https://en.wikipedia.org/wiki/Frank%E2%80%93Starling_law) states that $T_a$ should be increased, but when $T_a$ increases we get more force that compress the material which will bring $\lambda$ down, and it is not hard do see why this system can start oscillating.
#
# The solution to the problem is to make sure we update $T_a$ whenever we solve the mechanics model, which means that we would need to integrate the ODE systen during every iteration within the root finding algorithm. In other words, let us update the function as follows

def func(x, y, ti, dt, params, new_y, prev_lmbda):
    lmbda, p = x
    dLambda = (lmbda - prev_lmbda) / dt

    # Update lmbda and dLambda in the model
    params[lmbda_index] = lmbda
    params[dLambda_index] = dLambda
    new_y[:] = fgr(y, ti, dt, params)
    monitor = mon(ti, new_y, params)
    Ta = monitor[Ta_index]

    replace = {
        mech_model["p"]: p,
        mech_model["Ta"]: Ta,
        experiment["λ"]: lmbda,
        **material.default_parameters(),
    }

    P11 = P[0, 0].xreplace(replace)
    P22 = P[1, 1].xreplace(replace)
    return np.array([P11, P22], dtype=np.float64)


# Note that whenever we now get a new guess for $\lambda$ (and $p$), we update $T_a$ accordingly. We now need to also pass the previous $\lambda$ (in order to compute $\dot{\lambda}$) as well as the previous state, the parameters and the time step. Our time loop will therefore need to be updated accordingly.

# +
lmbda_value = 1.0
p_value = 0.0

y = model["init_state_values"]()
params = model["init_parameter_values"](i_Stim_Period=BCL)
params[lmbda_index] = lmbda_value
prev_lmbda = lmbda_value
params[dLambda_index] = 0.0

V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
lmbdas = np.zeros(len(t))
dLambdas = np.zeros(len(t))
ps = np.zeros(len(t))

for i, ti in enumerate(t):
    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(y.copy(), ti, dt, params, y, prev_lmbda),
        method="hybr",
    )
    lmbda_value, p_value = res.x
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, params)
    Ta[i] = monitor[Ta_index]
    lmbdas[i] = lmbda_value
    ps[i] = p_value

    dLambda = (lmbda_value - prev_lmbda) / dt
    dLambdas[i] = dLambda

    prev_lmbda = lmbda_value
# -

fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, dLambdas)
ax[0, 2].plot(t, lmbdas)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel(r"$\dot{\lambda}$")
ax[0, 2].set_ylabel(r"$\lambda$")
ax[1, 2].set_ylabel("p")
for axi in ax.flatten():
    axi.grid()
fig.tight_layout()
plt.show()

#
# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
