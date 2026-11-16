# # A spherical ventricle
#
# In this example we build a zero-dimensional model of the left ventricle: a spherical wall around a cavity.
# `zero_mech.SphericalVentricle` turns the stress in the wall into a pressure in the cavity, for any `zero_mech.Model`.
# We look at its passive pressure-volume relation, make it beat against a simple circulation, and change its
# preload and afterload.
#
# The wall is incompressible with its fibres in the wall plane. When the cavity is inflated, both in-plane
# directions are stretched by $\lambda$ and the wall thins by $\lambda^{-2}$. For a wall of volume $V_w$ around
# a cavity of volume $V$, with unloaded cavity volume $V_0$,
#
# $$
# \lambda(V) = \left( \frac{V + V_w/3}{V_0 + V_w/3} \right)^{1/3},
# $$
#
# the one-fibre model of Arts et al. (1991). The work done by the cavity pressure equals the change in the strain
# energy $\psi$ of the wall, $p \, dV = V_w \, d\psi$, so
#
# $$
# p(V) = V_w \, \frac{d\psi}{d\lambda} \, \frac{d\lambda}{dV} + \frac{T_a}{1 + 3V/V_w}.
# $$
#
# The last term is the contribution of an active tension $T_a$ along the fibres (a Cauchy stress). For a thin wall
# the first term is Laplace's law.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display
import zero_mech

# ## The deformation
#
# The deformation of the wall is available as an experiment, like uniaxial tension and simple shear

experiment = zero_mech.experiments.equibiaxial()
display(experiment.F)

# ## The ventricle
#
# We use the Holzapfel-Ogden model with the fibres along the first axis and the sheets along the second, both in
# the wall plane. The default parameters of the model are fitted to shear experiments on excised tissue and are
# very stiff at the stretches of a filling ventricle, so we choose softer ones. These are illustrative values
# that give an end-diastolic pressure near 10 mmHg at 120 mL; they are not fitted to data.

# +
material = zero_mech.material.HolzapfelOgden()
model = zero_mech.Model(material=material)
material_parameters = {"a": 0.3, "b": 5.0, "a_f": 1.0, "b_f": 4.0, "a_s": 0.0, "a_fs": 0.0}  # kPa and dimensionless

V0 = 50.0  # mL, cavity volume of the unloaded ventricle
Vw = 150.0  # mL, wall volume
ventricle = zero_mech.SphericalVentricle(model, V0=V0, Vw=Vw, parameters=material_parameters)
# -

# The pressure is a sympy expression in the cavity volume `V`, the active tension `T_a` and the material parameters,
# so it can be inspected or turned into code for another solver

display(ventricle.pressure_expr)

# ## Passive pressure-volume relation
#
# The material is in kPa, so the pressure is too. We compare the ventricle with a thinner and a thicker wall
# around the same cavity.

# +
kPa_to_mmHg = 7.50062
V = np.linspace(20, 160, 200)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for wall_volume in (75.0, 150.0, 300.0):
    v = zero_mech.SphericalVentricle(model, V0=V0, Vw=wall_volume, parameters=material_parameters)
    ax[0].plot(V, v.pressure(V) * kPa_to_mmHg, label=f"$V_w$ = {wall_volume:.0f} mL")
    ax[1].plot(V, v.stretch(V), label=f"$V_w$ = {wall_volume:.0f} mL")
ax[0].axhline(0, color="k", linewidth=0.5)
ax[0].set_ylim(-5, 30)
ax[0].set_xlabel("Cavity volume [mL]")
ax[0].set_ylabel("Passive pressure [mmHg]")
ax[1].set_xlabel("Cavity volume [mL]")
ax[1].set_ylabel("Fibre stretch")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# -

# Below the unloaded volume the wall is compressed and the pressure is negative: the ventricle sucks.
# A thicker wall is stiffer, and its fibres are stretched less by the same change in cavity volume.

# ## A beating ventricle
#
# To make the ventricle beat we need an active tension and something to pump against. Here the active tension is
# a stand-in for a model of the sarcomere: a prescribed activation in time, the same in every beat, times a linear
# dependence on the fibre stretch. The length dependence matters. The active part of the pressure,
# $T_a / (1 + 3V/V_w)$, grows as the cavity gets smaller, so a tension that did not fall with shortening would
# empty the ventricle completely.

# +
T_cycle = 0.8  # s
T_systole = 0.35  # s
Ta_max = 60.0  # kPa, at full activation and a stretch of 1.15 or more
stretch_zero = 0.8  # stretch at which the active tension vanishes


def active_tension(t, stretch):
    tc = np.mod(t, T_cycle)
    activation = np.where(tc < T_systole, np.sin(np.pi * tc / T_systole) ** 2, 0.0)
    length_dependence = np.clip((stretch - stretch_zero) / (1.15 - stretch_zero), 0.0, 1.0)
    return Ta_max * activation * length_dependence


# -

# and the circulation is as simple as it can be: the ventricle fills from a reservoir at a fixed pressure through
# the mitral valve, and ejects through the aortic valve into a compliant artery that drains through a peripheral
# resistance. Each valve is a resistance that only lets flow through in one direction. Volumes are in mL, pressures
# in mmHg and time in s.

# +
circulation = {
    "p_ven": 8.0,  # mmHg, filling pressure (preload)
    "R_mv": 0.01,  # mmHg s / mL, mitral valve
    "R_av": 0.01,  # mmHg s / mL, aortic valve
    "C_art": 1.5,  # mL / mmHg, arterial compliance
    "R_per": 1.0,  # mmHg s / mL, peripheral resistance (afterload)
}


def rhs(t, y, c):
    V, p_art = y
    p_lv = ventricle.pressure(V, active_tension(t, ventricle.stretch(V))) * kPa_to_mmHg
    Q_mv = max(c["p_ven"] - p_lv, 0.0) / c["R_mv"]
    Q_av = max(p_lv - p_art, 0.0) / c["R_av"]
    return [Q_mv - Q_av, (Q_av - p_art / c["R_per"]) / c["C_art"]]


def simulate(c, num_beats=12):
    """Simulate `num_beats` beats and return time, volume, ventricular and arterial pressure of the last one."""
    t_end = num_beats * T_cycle
    t_eval = np.linspace(t_end - T_cycle, t_end, 801)
    sol = solve_ivp(rhs, (0, t_end), [120.0, 80.0], args=(c,), method="LSODA", max_step=1e-3, t_eval=t_eval)
    V, p_art = sol.y
    p_lv = ventricle.pressure(V, active_tension(sol.t, ventricle.stretch(V))) * kPa_to_mmHg
    return sol.t - sol.t[0], V, p_lv, p_art


# -

# The ventricle needs no equation of its own: its pressure follows from its volume and the active tension, and the
# valves give the rate of change of the volume.

# +
t, V_lv, p_lv, p_art = simulate(circulation)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot(t, p_lv, label="Ventricle")
ax[0].plot(t, p_art, label="Artery")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Pressure [mmHg]")
ax[0].legend()
ax[1].plot(t, V_lv)
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Cavity volume [mL]")
ax[2].plot(V_lv, p_lv)
ax[2].set_xlabel("Cavity volume [mL]")
ax[2].set_ylabel("Pressure [mmHg]")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()

stroke_volume = V_lv.max() - V_lv.min()
print(f"End-diastolic volume {V_lv.max():.0f} mL, end-systolic volume {V_lv.min():.0f} mL")
print(f"Stroke volume {stroke_volume:.0f} mL, ejection fraction {100 * stroke_volume / V_lv.max():.0f} %")
print(f"Peak pressure {p_lv.max():.0f} mmHg, cardiac output {stroke_volume / T_cycle * 60 / 1000:.1f} L/min")
print(f"Fibre stretch from {ventricle.stretch(V_lv.min()):.2f} to {ventricle.stretch(V_lv.max()):.2f}")
# -

# ## Preload and afterload
#
# We now lower the filling pressure, as when blood is lost, and raise the peripheral resistance

# +
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for p_ven in (2.0, 4.0, 8.0, 12.0):
    _, V_i, p_i, _ = simulate({**circulation, "p_ven": p_ven})
    ax[0].plot(V_i, p_i, label=f"{p_ven:.0f} mmHg, SV {V_i.max() - V_i.min():.0f} mL")
for R_per in (0.6, 1.0, 1.6, 2.5):
    _, V_i, p_i, _ = simulate({**circulation, "R_per": R_per})
    ax[1].plot(V_i, p_i, label=f"{R_per:.1f} mmHg s/mL, SV {V_i.max() - V_i.min():.0f} mL")
V_passive = np.linspace(35, 150, 100)
for a in ax:
    a.plot(V_passive, ventricle.pressure(V_passive) * kPa_to_mmHg, "k--", linewidth=0.8)
    a.set_xlabel("Cavity volume [mL]")
    a.grid(alpha=0.3)
ax[0].set_ylabel("Pressure [mmHg]")
ax[0].set_title("Filling pressure")
ax[1].set_title("Peripheral resistance")
ax[0].legend()
ax[1].legend()
fig.tight_layout()
plt.show()
# -

# The dashed line is the passive pressure-volume relation, which every loop follows during filling.
#
# With less filling the ventricle starts each beat smaller, its fibres develop less tension, and it ejects less:
# the Frank-Starling mechanism, here through the linear length dependence we prescribed. Against a higher
# resistance the pressure rises and the ventricle empties less.
#
# In the next example the active tension comes from a model of the cell instead, which takes the fibre stretch
# `ventricle.stretch(V)` and its rate of change as inputs.
#
# ## References
#
# Arts T, Bovendeerd PHM, Prinzen FW, Reneman RS (1991). Relation between left ventricular cavity pressure and
# volume and systolic fiber stress and strain in the wall. Biophys J 59:93-102.
#
# Caruel M, Chabiniok R, Moireau P, Lecarpentier Y, Chapelle D (2014). Dimensional reductions of a cardiac model
# for effective validation and calibration. Biomech Model Mechanobiol 13:897-914.
