from pathlib import Path

from tqdm import tqdm
import gotranx
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt


# Generate code and save it to a file if it does not exist
module_path = Path("ToRORd_dynCl_endo.py")
if not module_path.is_file():
    ode = gotranx.load_ode("ToRORd_dynCl_endo.ode")
    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
    )
    module_path.write_text(code)

import ToRORd_dynCl_endo

model = ToRORd_dynCl_endo.__dict__


# Now we can use the model dictionary to call the generated functions

# Set time step to 0.1 ms
dt = 0.1
# Simulate model for 1000 ms
BCL = 1000
num_beats = 1
t = np.arange(0, num_beats * BCL, dt)

y = model["init_state_values"]()
# Get initial parameter values
params = model["init_parameter_values"](i_Stim_Period=BCL)

# Get the index of the membrane potential
V_index = model["state_index"]("v")
Ca_index = model["state_index"]("cai")
# Get the index of the active tension from the land model
Ta_index = model["monitor_index"]("Ta")
Istim_index = model["monitor_index"]("Istim")
fgr = model["generalized_rush_larsen"]
mon = model["monitor_values"]

lmbda_index = model["parameter_index"]("lmbda")
dLambda_index = model["parameter_index"]("dLambda")


# Material parameters for Neo-Hookean model
mu = 15.0


def func(x, Ta):
    lmbda, p = x

    return np.array(
        [
            2 * Ta * lmbda + 1.0 * lmbda * mu + p / lmbda,
            2 * np.sqrt(lmbda) * p + 2.0 * mu / np.sqrt(lmbda),
        ],
        dtype=np.float64,
    )


def jac(x, Ta):
    lmbda, p = x

    dP11_dlmbda = 2 * Ta + 1.0 * mu - p / lmbda**2
    dP11_dp = 1 / lmbda
    dP22_dlmbda = p / np.sqrt(lmbda) - 1.0 * mu / lmbda ** (3 / 2)
    dP22_dp = 2 * np.sqrt(lmbda)

    return np.array(
        [[dP11_dlmbda, dP11_dp], [dP22_dlmbda, dP22_dp]],
        dtype=np.float64,
    )


lmbda_value = 1.0
p_value = 0.0

params[lmbda_index] = lmbda_value
prev_lmbda = lmbda_value
params[dLambda_index] = 0.0

# Let us simulate the model

V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
Istim = np.zeros(len(t))
lmbdas = np.zeros(len(t))
dLambdas = np.zeros(len(t))
ps = np.zeros(len(t))

for i, ti in tqdm(enumerate(t), total=len(t)):
    y = fgr(y, ti, dt, params)
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, params)
    Ta[i] = monitor[Ta_index]

    # # Calculate lambda and p using root finding
    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(Ta[i],),
        jac=jac,
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
    Istim[i] = monitor[Istim_index]


# And plot the results
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
ax[1, 1].set_ylabel("dLambda")
ax[0, 2].set_ylabel("Lambda")
ax[1, 2].set_ylabel("p")
fig.tight_layout()
fig.savefig("neohookean_strong_no_split.png")
