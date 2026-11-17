# # Two ventricles: the TriSeg model
#
# A sphere describes one ventricle on its own. The two ventricles of the heart share a wall, the septum, and load
# each other through it: when the pressure in the right ventricle rises, the septum flattens and the left ventricle
# fills less. The TriSeg model of Lumens et al. (2009) captures this with three wall segments, the left free wall
# (LW), the septum (SW) and the right free wall (RW), that meet in a circular junction.
#
# Each segment is a spherical cap. Its midwall surface is described by the radius $y_m$ of the junction, which the
# three share, and the distance $x_m$ from the plane of the junction to the top of the cap. The two cavity volumes fix
# the caps of the free walls once the septum is known, so two unknowns remain, $x_{m,SW}$ and $y_m$. They follow from
# the balance of the wall tensions in the junction, along the axis and radially.
#
# As for the sphere, the wall tension comes from the work of the fibres. `zero_mech.TriSeg` takes any
# `zero_mech.Model` for the wall and an active tension for each segment.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import zero_mech

# ## The model
#
# We use the same Holzapfel-Ogden wall as for the spherical ventricle, with illustrative parameters. The wall volumes
# are those Lumens et al. start from. The reference areas are the midwall areas at which the fibres are unstretched.
# Lengths are in cm, areas in cm$^2$, volumes in mL and stresses in kPa.

# +
material_parameters = {"a": 0.3, "b": 5.0, "a_f": 1.0, "b_f": 4.0, "a_s": 0.0, "a_fs": 0.0}
model = zero_mech.Model(material=zero_mech.material.HolzapfelOgden())
triseg = zero_mech.TriSeg(
    model,
    Vw=(75.0, 40.0, 30.0),  # mL: LW, SW, RW
    Am_ref=(68.0, 38.0, 85.0),  # cm^2
    parameters=material_parameters,
)
kPa_to_mmHg = 7.50062
# -

# For given cavity volumes and active tensions, `solve` finds the geometry in which the junction is in equilibrium

state = triseg.solve(V_LV=120.0, V_RV=130.0)
print(f"Left ventricular pressure  {state.p_LV * kPa_to_mmHg:5.1f} mmHg")
print(f"Right ventricular pressure {state.p_RV * kPa_to_mmHg:5.1f} mmHg")
print(f"Junction radius {state.ym:.2f} cm, xm of LW, SW, RW: " + ", ".join(f"{x:.2f}" for x in state.xm) + " cm")
print("Fibre stretch of LW, SW, RW: " + ", ".join(f"{s:.3f}" for s in state.stretch))

# The midwall surfaces are circular arcs through the junction. Let us draw them, for the passive ventricles above and
# for the same volumes with an active tension of 40 kPa in all three walls


# +
def plot_geometry(ax, state, **kwargs):
    """Draw the midwall surfaces of the three walls in the plane through the axis."""
    y = np.linspace(-state.ym, state.ym, 201)
    for xm in state.xm:
        if abs(xm) < 1e-9:
            x = np.zeros_like(y)
        else:
            radius = (xm**2 + state.ym**2) / (2 * xm)
            x = xm - radius + np.sign(radius) * np.sqrt(radius**2 - y**2)
        ax.plot(x, y, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")


fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
plot_geometry(ax[0], state, color="k")
ax[0].set_title("Passive")
ax[0].set_ylabel("y [cm]")
active = triseg.solve(120.0, 130.0, Ta=40.0)
plot_geometry(ax[1], active, color="k")
ax[1].set_title(f"Active: {active.p_LV * kPa_to_mmHg:.0f} and {active.p_RV * kPa_to_mmHg:.0f} mmHg")
for a in ax:
    a.text(-3.0, 0, "LV", ha="center")
    a.text(3.5, 0, "RV", ha="center")
fig.tight_layout()
plt.show()
# -

# The left ventricle is to the left of the septum (the middle arc) and the right ventricle to the right. With the
# same fibre tension in all walls, the pressure in the left ventricle is several times that in the right: the right
# free wall is thin and wide.
#
# ## A closed circulation
#
# With two ventricles the circulation can be closed. Blood goes from the left ventricle to the systemic arteries (SA),
# through the systemic resistance to the systemic veins (SV), into the right ventricle, to the pulmonary arteries (PA),
# through the pulmonary resistance to the pulmonary veins (PV), and back. Each of the four vessels is a compliance, each
# valve a resistance that lets flow through in one direction. Volumes are in mL, pressures in mmHg and time in s.
# The values are illustrative.

circulation = {
    "C_SA": 2.5, "C_SV": 90.0, "C_PA": 6.0, "C_PV": 9.4,  # mL / mmHg
    "R_sys": 1.33, "R_pul": 0.146,  # mmHg s / mL
    "R_valve": 0.01,  # mmHg s / mL
    "stressed_volume": 1100.0,  # mL, the blood that distends the vessels and the heart
}  # fmt: skip

# The active tension is, as for the sphere, a stand-in for a model of the sarcomere: a prescribed activation in time
# multiplied by a linear dependence on the stretch of the fibres, zero at a stretch of 0.8. Each wall has its own
# stretch, and so its own tension.
#
# The dependence on the stretch is not a detail here. The tension in a wall segment is the fibre stress times the wall
# volume divided by twice its area, so a segment that shortens pulls harder at the same stress. Unless the stress
# falls when the fibres shorten, and falls at once, one wall shortens at the expense of the others. The stretches are
# part of the geometry we solve for, so the dependence has to be inside the solve. `solve` takes it as an active
# stiffness `Ka`: the active tension of a wall at stretch $\lambda$ is `Ta + Ka * (stretch - stretch_ref)`. Here the
# active tension is $a(t) \, T_{max} \, (\lambda - 0.8) / 0.35$, a spring whose stiffness follows the activation.

# +
T_cycle = 0.8  # s
T_systole = 0.35  # s
Ta_max = 65.0  # kPa, at full activation and a stretch of 1.15
stretch_zero = 0.8  # stretch at which the active tension vanishes


def active_stiffness(t):
    tc = t % T_cycle
    activation = np.sin(np.pi * tc / T_systole) ** 2 if tc < T_systole else 0.0
    return Ta_max * activation / (1.15 - stretch_zero)


# -

# The state of the system is the six volumes. For given volumes `triseg.solve` gives the ventricular pressures, and
# the valves and resistances give the flows. We start each solve from the geometry of the previous one.

# +
names = ("V_LV", "V_SA", "V_SV", "V_RV", "V_PA", "V_PV")


def evaluate(t, y, c, guess):
    """Pressures, flows and geometry at time t and state y."""
    V_LV, V_SA, V_SV, V_RV, V_PA, V_PV = y
    state = triseg.solve(V_LV, V_RV, Ka=active_stiffness(t), stretch_ref=stretch_zero, guess=guess)
    p = {"LV": state.p_LV * kPa_to_mmHg, "RV": state.p_RV * kPa_to_mmHg}
    p.update(SA=V_SA / c["C_SA"], SV=V_SV / c["C_SV"], PA=V_PA / c["C_PA"], PV=V_PV / c["C_PV"])
    Q = {
        "mitral": max(p["PV"] - p["LV"], 0.0) / c["R_valve"],
        "aortic": max(p["LV"] - p["SA"], 0.0) / c["R_valve"],
        "tricuspid": max(p["SV"] - p["RV"], 0.0) / c["R_valve"],
        "pulmonary": max(p["RV"] - p["PA"], 0.0) / c["R_valve"],
        "sys": (p["SA"] - p["SV"]) / c["R_sys"],
        "pul": (p["PA"] - p["PV"]) / c["R_pul"],
    }
    return state, p, Q


def simulate(c, num_beats=30):
    """Simulate `num_beats` beats and return the last one as a dictionary of arrays."""
    # Start with the blood spread so that the pressures are roughly 90, 5, 15 and 8 mmHg
    y0 = np.array([110.0, c["C_SA"] * 90.0, c["C_SV"] * 5.0, 120.0, c["C_PA"] * 15.0, c["C_PV"] * 8.0])
    y0[1:3] *= (c["stressed_volume"] - y0[0] - y0[3] - y0[4] - y0[5]) / y0[1:3].sum()
    last = {"guess": None}

    def rhs(t, y):
        state, p, Q = evaluate(t, y, c, last["guess"])
        last["guess"] = state.guess
        dV = [
            Q["mitral"] - Q["aortic"],
            Q["aortic"] - Q["sys"],
            Q["sys"] - Q["tricuspid"],
            Q["tricuspid"] - Q["pulmonary"],
            Q["pulmonary"] - Q["pul"],
            Q["pul"] - Q["mitral"],
        ]
        return dV

    t_end = num_beats * T_cycle
    t_eval = np.linspace(t_end - T_cycle, t_end, 401)
    sol = solve_ivp(rhs, (0, t_end), y0, method="LSODA", max_step=2e-3, rtol=1e-6, atol=1e-8, t_eval=t_eval)
    assert sol.success, sol.message

    out = {"t": sol.t - sol.t[0], "V_LV": sol.y[0], "V_RV": sol.y[3]}
    series = {k: [] for k in ("p_LV", "p_RV", "p_SA", "p_PA", "Cm_SW", "stretch_LW", "stretch_SW", "stretch_RW")}
    guess = None
    for t, y in zip(sol.t, sol.y.T):
        state, p, _ = evaluate(t, y, c, guess)
        guess = state.guess
        values = (p["LV"], p["RV"], p["SA"], p["PA"], state.Cm[1], *state.stretch)
        for k, v in zip(series, values):
            series[k].append(v)
    out.update({k: np.array(v) for k, v in series.items()})
    out["end_diastole"] = triseg.solve(out["V_LV"][0], out["V_RV"][0])
    return out


# -

# ## A normal beat

# +
normal = simulate(circulation)


def summary(label, r):
    sv_l = r["V_LV"].max() - r["V_LV"].min()
    sv_r = r["V_RV"].max() - r["V_RV"].min()
    print(
        f"{label:22s} LV {r['V_LV'].max():4.0f}/{r['V_LV'].min():3.0f} mL, {r['p_LV'].max():4.0f} mmHg, SV {sv_l:3.0f} mL | "
        f"RV {r['V_RV'].max():4.0f}/{r['V_RV'].min():3.0f} mL, {r['p_RV'].max():3.0f} mmHg, SV {sv_r:3.0f} mL | "
        f"cardiac output {sv_l / T_cycle * 0.06:.1f} L/min"
    )


summary("Normal", normal)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot(normal["t"], normal["p_LV"], label="LV")
ax[0].plot(normal["t"], normal["p_SA"], label="Systemic arteries")
ax[0].plot(normal["t"], normal["p_RV"], label="RV")
ax[0].plot(normal["t"], normal["p_PA"], label="Pulmonary arteries")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Pressure [mmHg]")
ax[0].legend()
ax[1].plot(normal["V_LV"], normal["p_LV"], label="LV")
ax[1].plot(normal["V_RV"], normal["p_RV"], label="RV")
ax[1].set_xlabel("Cavity volume [mL]")
ax[1].set_ylabel("Pressure [mmHg]")
ax[1].legend()
ax[2].plot(normal["t"], normal["stretch_LW"], label="LW")
ax[2].plot(normal["t"], normal["stretch_SW"], label="SW")
ax[2].plot(normal["t"], normal["stretch_RW"], label="RW")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Fibre stretch")
ax[2].legend()
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
plt.show()
# -

# The two ventricles eject the same stroke volume, as they must in a closed circulation, at very different pressures.
#
# ## Ventricular interaction
#
# We now load the ventricles in two ways. First we raise the pulmonary resistance four-fold, an acute pulmonary
# hypertension, which is the case Lumens et al. study. Second we remove 500 mL of blood, as in a hemorrhage.

# +
hypertension = simulate({**circulation, "R_pul": 4 * circulation["R_pul"]})
hemorrhage = simulate({**circulation, "stressed_volume": circulation["stressed_volume"] - 500.0})
summary("Pulmonary hypertension", hypertension)
summary("Hemorrhage", hemorrhage)

cases = {"Normal": normal, "Pulmonary hypertension": hypertension, "Hemorrhage": hemorrhage}
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for label, r in cases.items():
    ax[0].plot(r["V_LV"], r["p_LV"], label=label)
    ax[1].plot(r["V_RV"], r["p_RV"], label=label)
    ax[2].plot(r["t"], r["Cm_SW"], label=label)
ax[0].set_title("Left ventricle")
ax[1].set_title("Right ventricle")
ax[2].set_title("Curvature of the septum")
ax[0].set_ylabel("Pressure [mmHg]")
ax[2].set_ylabel("Curvature [1/cm]")
ax[2].set_xlabel("Time [s]")
ax[2].axhline(0, color="k", linewidth=0.5)
for a in ax[:2]:
    a.set_xlabel("Cavity volume [mL]")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend()
fig.tight_layout()
plt.show()
# -

# With a high pulmonary resistance the right ventricle has to build up more pressure. It dilates, and the septum,
# which normally bulges into the right ventricle, flattens. The left ventricle receives less blood and is squeezed by
# the septum, so it fills less and its pressure falls: the ventricles interact. The geometry at the end of diastole
# shows it

fig, ax = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
for a, (label, r) in zip(ax, cases.items()):
    plot_geometry(a, r["end_diastole"], color="k")
    a.set_title(label)
ax[0].set_ylabel("y [cm]")
fig.tight_layout()
plt.show()

# After the loss of blood both ventricles are smaller at the end of diastole, their fibres are shorter and develop
# less tension, and stroke volume and pressures fall.
#
# The active tension in this example is prescribed. As for the spherical ventricle, it can come from a model of the
# cell instead: one cell for each wall, driven by the stretch of that wall, `state.stretch`.
#
# ## References
#
# Lumens J, Delhaas T, Kirn B, Arts T (2009). Three-wall segment (TriSeg) model describing mechanics and hemodynamics
# of ventricular interaction. Ann Biomed Eng 37:2234-2255.
