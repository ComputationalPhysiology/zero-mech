# # A spherical ventricle driven by a cell model
#
# In the previous example the active tension was a prescribed function of time and stretch. Here it comes from a
# model of the cell: the ToR-ORd model of the human ventricular action potential coupled to the Land model of
# contraction, the same `.ode` file as in the electro-mechanics examples. The ventricle gives the cell its fibre
# stretch and the cell gives the ventricle its active tension. Because the tension of the Land model depends on
# the stretch, the ventricle ejects more when it is filled more: the Frank-Starling mechanism.
#
# The coupled system is a set of ordinary differential equations with no algebraic constraint:
#
# - the pressure in the cavity follows from its volume $V$ and the active tension $T_a$,
#   `ventricle.pressure(V, Ta)`;
# - the valves turn pressures into flows, which give $dV/dt$;
# - the cell needs the stretch $\lambda(V)$ and its rate $d\lambda/dt = \lambda'(V) \, dV/dt$.

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numba
import gotranx
import zero_mech

# ## The cell model
#
# We generate Python code for the cell model with `gotranx`, with a generalized Rush-Larsen scheme

# +
module_path = Path("ToRORd_dynCl_endo.py")
if not module_path.is_file():
    ode = gotranx.load_ode(Path("..") / "electro-mechanics" / "ToRORd_dynCl_endo.ode")
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen])
    module_path.write_text(code)

sys.path.insert(0, str(Path.cwd()))
import ToRORd_dynCl_endo as cell

step_cell = numba.njit(cell.generalized_rush_larsen)
# -

# The Land model's active tension is one of the monitored expressions of the file. We need it in every time step,
# so we evaluate it directly from the four crossbridge states instead (and check it against the file below)

# +
BCL = 800.0  # ms
Tref = 120.0  # kPa, the file's value
parameters = cell.init_parameter_values(i_Stim_Period=BCL, Tref=Tref)
lmbda_index = cell.parameter_index("lmbda")
dLambda_index = cell.parameter_index("dLambda")
XS, XW, Zetas, Zetaw = (cell.state_index(name) for name in ("XS", "XW", "Zetas", "Zetaw"))
rs, Beta0 = (parameters[cell.parameter_index(name)] for name in ("rs", "Beta0"))


def active_tension(y, lmbda):
    lmbda = min(lmbda, 1.2)
    h = max(0.0, 1.0 + Beta0 * (lmbda + min(lmbda, 0.87) - 1.87))
    return h * (Tref / rs) * (y[XS] * (y[Zetas] + 1.0) + y[XW] * y[Zetaw])


# -

# ## The ventricle and the circulation
#
# These are the ones of the previous example: a Holzapfel-Ogden wall of 150 mL around a cavity of 50 mL, filled
# from a reservoir through the mitral valve and ejecting through the aortic valve into a compliant artery.
# The unloaded ventricle has slack sarcomeres, so the stretch of the Land model is the stretch of the ventricle.

# +
material_parameters = {"a": 0.3, "b": 5.0, "a_f": 1.0, "b_f": 4.0, "a_s": 0.0, "a_fs": 0.0}
ventricle = zero_mech.SphericalVentricle(
    zero_mech.Model(material=zero_mech.material.HolzapfelOgden()), V0=50.0, Vw=150.0, parameters=material_parameters
)
kPa_to_mmHg = 7.50062
circulation = {"p_ven": 8.0, "R_mv": 0.01, "R_av": 0.01, "C_art": 1.5, "R_per": 1.0}  # mmHg, mL, s
# -

# ## Time stepping
#
# In each step of 0.05 ms we advance the cell with the stretch and stretch rate of the ventricle, and the volume and
# the arterial pressure with the flows through the valves (explicit Euler)


def simulate(c, num_beats=12, dt=0.05):
    """Simulate `num_beats` beats. Returns the last one: time [ms], volume [mL], ventricular and arterial
    pressure [mmHg], active tension [kPa], stretch and cytosolic calcium [mM]."""
    y = cell.init_state_values()
    p = parameters.copy()
    cai = cell.state_index("cai")
    V, p_art = 120.0, 80.0
    steps_per_beat = int(round(BCL / dt))
    out = np.zeros((7, steps_per_beat))
    for beat in range(num_beats):
        for i in range(steps_per_beat):
            t = beat * BCL + i * dt
            lmbda = ventricle.stretch(V)
            Ta = active_tension(y, lmbda)
            p_lv = ventricle.pressure(V, Ta) * kPa_to_mmHg
            Q_mv = max(c["p_ven"] - p_lv, 0.0) / c["R_mv"]
            Q_av = max(p_lv - p_art, 0.0) / c["R_av"]
            dV_dt = Q_mv - Q_av  # mL / s
            if beat == num_beats - 1:
                out[:, i] = (i * dt, V, p_lv, p_art, Ta, lmbda, y[cai])

            p[lmbda_index] = lmbda
            p[dLambda_index] = ventricle.dstretch_dV(V) * dV_dt * 1e-3  # 1 / ms
            y = step_cell(y, t, dt, p)
            V += dt * 1e-3 * dV_dt
            p_art += dt * 1e-3 * (Q_av - p_art / c["R_per"]) / c["C_art"]
    # The active tension we computed is the one of the file
    monitored = cell.monitor_values(t, y, p)[cell.monitor_index("Ta")]
    assert abs(active_tension(y, p[lmbda_index]) - monitored) < 1e-9 * max(1.0, abs(monitored))
    return out


# ## One beat

# +
t, V_lv, p_lv, p_art, Ta, lmbda, cai = simulate(circulation)

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
ax[0, 0].plot(t, cai * 1e3)
ax[0, 0].set_ylabel("Cytosolic calcium [µM]")
ax[0, 1].plot(t, Ta)
ax[0, 1].set_ylabel("Active tension [kPa]")
ax[0, 2].plot(t, lmbda)
ax[0, 2].set_ylabel("Fibre stretch")
ax[1, 0].plot(t, p_lv, label="Ventricle")
ax[1, 0].plot(t, p_art, label="Artery")
ax[1, 0].set_ylabel("Pressure [mmHg]")
ax[1, 0].legend()
ax[1, 1].plot(t, V_lv)
ax[1, 1].set_ylabel("Cavity volume [mL]")
ax[1, 2].plot(V_lv, p_lv)
ax[1, 2].set_xlabel("Cavity volume [mL]")
ax[1, 2].set_ylabel("Pressure [mmHg]")
for a in ax.flatten()[:5]:
    a.set_xlabel("Time [ms]")
for a in ax.flatten():
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()

print(f"End-diastolic volume {V_lv.max():.0f} mL, end-systolic volume {V_lv.min():.0f} mL")
print(f"Stroke volume {V_lv.max() - V_lv.min():.0f} mL, peak pressure {p_lv.max():.0f} mmHg")
# -

# ## The Frank-Starling mechanism
#
# We repeat the simulation at different filling pressures and plot the stroke volume against the end-diastolic volume

# +
filling_pressures = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
edv, sv = [], []
for p_ven in filling_pressures:
    _, V_i, p_i, _, _, _, _ = simulate({**circulation, "p_ven": p_ven})
    ax[0].plot(V_i, p_i, label=f"{p_ven:.0f} mmHg")
    edv.append(V_i.max())
    sv.append(V_i.max() - V_i.min())
    print(f"Filling pressure {p_ven:4.1f} mmHg: end-diastolic volume {edv[-1]:5.1f} mL, stroke volume {sv[-1]:5.1f} mL")
ax[0].set_xlabel("Cavity volume [mL]")
ax[0].set_ylabel("Pressure [mmHg]")
ax[0].legend(title="Filling pressure")
ax[1].plot(edv, sv, "o-")
ax[1].set_xlabel("End-diastolic volume [mL]")
ax[1].set_ylabel("Stroke volume [mL]")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# -

# The stroke volume rises with the filling of the ventricle. Two things in the Land model produce this: its
# maximal tension grows with the stretch, and so does its sensitivity to calcium.
