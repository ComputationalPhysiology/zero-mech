# # Zeta splitting

# In this example, we couple the Neo-Hookean model with the ToRORd and Land model using the so called zeta splitting approach.


from pathlib import Path
import gotranx
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
import sympy  # noqa: F401
import zero_mech


experiment = zero_mech.experiments.uniaxial_tension()
mat = zero_mech.material.NeoHookean()
comp = zero_mech.compressibility.Incompressible()
act = zero_mech.active.ActiveStress()
mech_model = zero_mech.Model(material=mat, compressibility=comp, active=act)

# First Piola-Kirchhoff stress
P = mech_model.first_piola_kirchhoff(experiment.F)

# Generate code and save it to a file if it does not exist
module_path = Path("ToRORd_dynCl_endo_zetasplit.py")
ode = gotranx.load_ode("ToRORd_dynCl_endo_zetasplit.ode")
mechanics_comp = ode.get_component("mechanics")
mechanics_ode = mechanics_comp.to_ode()

ep_ode = ode - mechanics_comp
ep_file = Path("ToRORd_dynCl_endo_zetasplit_ep.py")

# Generate model code from .ode file
rebuild = False
if not ep_file.is_file() or rebuild:
    # Generate code for full model. The full model output is plotted together with the splitting
    code = gotranx.cli.gotran2py.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
    )

    # Generate code for the electrophysiology model
    code_ep = gotranx.cli.gotran2py.get_code(
        ep_ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
        missing_values=mechanics_ode.missing_variables,
    )

    # Generate code for the mechanics model
    code_mechanics = gotranx.cli.gotran2py.get_code(
        mechanics_ode,
        scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
        missing_values=ep_ode.missing_variables,
    )

    # Create ep, mechanics and full model to files:
    ep_file.write_text(code_ep)
    Path("ToRORd_dynCl_endo_zetasplit_mechanics.py").write_text(code_mechanics)

import ToRORd_dynCl_endo_zetasplit_ep
import ToRORd_dynCl_endo_zetasplit_mechanics

ep_model = ToRORd_dynCl_endo_zetasplit_ep.__dict__
mechanics_model = ToRORd_dynCl_endo_zetasplit_mechanics.__dict__

# Now we can use the model dictionary to call the generated functions

# Set time step to 0.1 ms
dt = 0.1
# Simulate model for 1000 ms
BCL = 400
num_beats = 1
t = np.arange(0, num_beats * BCL, dt)

# Get the index of the membrane potential
V_index_ep = ep_model["state_index"]("v")
# Forwared generalized rush larsen scheme for the electrophysiology model
fgr_ep = ep_model["forward_generalized_rush_larsen"]
# Monitor function for the electrophysiology model
mon_ep = ep_model["monitor_values"]
# Missing values function for the electrophysiology model
mv_ep = ep_model["missing_values"]
# Index of the calcium concentration
Ca_index_ep = ep_model["state_index"]("cai")
CaTrpn_index_ep = ep_model["state_index"]("CaTrpn")
dLambda_index_ep = ep_model["parameter_index"]("dLambda")
lmbda_index_ep = ep_model["parameter_index"]("lmbda")

# From split-cai 0D (not in zeta 3D):
# Forwared generalized rush larsen scheme for the mechanics model
fgr_mechanics = mechanics_model["forward_generalized_rush_larsen"]
# Monitor function for the mechanics model
mon_mechanics = mechanics_model["monitor_values"]
# Missing values function for the mechanics model
mv_mechanics = mechanics_model["missing_values"]

Ta_index_mechanics = mechanics_model["monitor_index"]("Ta")
J_TRPN_index_ep = ep_model["monitor_index"]("J_TRPN")
XS_index_ep = ep_model["state_index"]("XS")
TmB_index_ep = ep_model["state_index"]("TmB")
XU_index_ep = ep_model["monitor_index"]("XU")


lmbda_index_mechanics = mechanics_model["parameter_index"]("lmbda")
Zetas_index_mechanics = mechanics_model["state_index"]("Zetas")
dLambda_index_mechanics = mechanics_model["parameter_index"]("dLambda")


# Get initial values from the EP model
y_ep = ep_model["init_state_values"]()
p_ep = ep_model["init_parameter_values"]()

# Setup missing values
ep_missing_values = np.zeros(len(ep_ode.missing_variables))
y_mechanics = mechanics_model["init_state_values"]()
p_mechanics = mechanics_model["init_parameter_values"]()
mechanics_missing_values = np.zeros(len(mechanics_ode.missing_variables))


mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
ep_missing_values[:] = mv_mechanics(
    0, y_mechanics, p_mechanics, mechanics_missing_values
)


def func(x, y, ti, dt, params, new_y, prev_lmbda, missing_values):
    lmbda, p = x
    dLambda = (lmbda - prev_lmbda) / dt

    # Update lmbda and dLambda in the model
    params[dLambda_index_mechanics] = dLambda
    params[lmbda_index_mechanics] = lmbda

    new_y[:] = fgr_mechanics(y, ti, dt, params, missing_values)
    monitor = mon_mechanics(ti, new_y, params, missing_values)
    Ta = monitor[Ta_index_mechanics]

    replace = {
        mech_model["p"]: p,
        mech_model["Ta"]: Ta,
        experiment["λ"]: lmbda,
        **mat.default_parameters()
    }

    P11 = P[0, 0].xreplace(replace)
    P22 = P[1, 1].xreplace(replace)
    return np.array([P11, P22], dtype=np.float64)


lmbda_value = 1.0
p_value = 0.0
prev_lmbda = lmbda_value

V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))
CaTrpn_ep = np.zeros(len(t))
TmB_ep = np.zeros(len(t))
XU_ep = np.zeros(len(t))
J_TRPN_ep = np.zeros(len(t))
XS_ep = np.zeros(len(t))

Ta_mechanics = np.zeros(len(t))
J_TRPN_ep = np.zeros(len(t))
lmbda_mechanics = np.zeros(len(t))
Zetas_mechanics = np.zeros(len(t))
Zetaw_mechanics = np.zeros(len(t))
dLambda_mechanics = np.zeros(len(t))

XS_ep = np.zeros(len(t))
TmB_ep = np.zeros(len(t))
XU_ep = np.zeros(len(t))

ps = np.zeros(len(t))

for i, ti in enumerate(t):
    # Forward step for the EP model (from cai split)
    y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
    V_ep[i] = y_ep[V_index_ep]
    Ca_ep[i] = y_ep[Ca_index_ep]
    CaTrpn_ep[i] = y_ep[CaTrpn_index_ep]
    monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values)
    TmB_ep[i] = y_ep[TmB_index_ep]
    XU_ep[i] = monitor_ep[XU_index_ep]
    J_TRPN_ep[i] = monitor_ep[J_TRPN_index_ep]
    XS_ep[i] = y_ep[XS_index_ep]

    # Update missing values for the mechanics model
    mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values)

    # Calculate lambda and p using root finding
    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(
            y_mechanics.copy(),
            ti,
            dt,
            p_mechanics,
            y_mechanics,
            prev_lmbda,
            mechanics_missing_values,
        ),
        method="hybr",
    )
    lmbda_value, p_value = res.x

    monitor_mechanics = mon_mechanics(
        ti,
        y_mechanics,
        p_mechanics,
        mechanics_missing_values,
    )

    ps[i] = p_value

    Ta_mechanics[i] = monitor_mechanics[Ta_index_mechanics]
    Zetas_mechanics[i] = y_mechanics[Zetas_index_mechanics]
    lmbda_mechanics[i] = lmbda_value
    dLambda = (lmbda_value - prev_lmbda) / dt
    dLambda_mechanics[i] = dLambda

    p_ep[lmbda_index_ep] = lmbda_value
    p_ep[dLambda_index_ep] = dLambda
    p_mechanics[lmbda_index_mechanics] = lmbda_value
    p_mechanics[dLambda_index_mechanics] = dLambda

    # Update missing values for the EP model
    ep_missing_values[:] = mv_mechanics(
        t, y_mechanics, p_mechanics, mechanics_missing_values
    )

    prev_lmbda = lmbda_value


# And plot the results
fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V_ep)
ax[1, 0].plot(t, Ta_mechanics)
ax[0, 1].plot(t, Ca_ep)
ax[1, 1].plot(t, dLambda_mechanics)
ax[0, 2].plot(t, lmbda_mechanics)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel("dLambda")
ax[0, 2].set_ylabel("Lambda")
ax[1, 2].set_ylabel("p")
for axi in ax.flatten():
    axi.grid()
fig.tight_layout()
fig.savefig("neohookean_zeta_split.png")
