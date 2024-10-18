from pathlib import Path

import numba
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
num_beats = 2
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


# Just add some values for the Guccione model
K = 2.0
b_ff = 8.0
b_xx = 2.0
# And set eta to 0.0 so that we have no transverse activation
eta = 0.0


@numba.jit(nopython=True)
def func(x, Ta, eta):
    lmbda, p = x
    E11 = 0.5 * (lmbda**2 - 1.0)
    E22 = E33 = 0.5 * (1.0 / lmbda - 1.0)
    W = b_ff * E11**2 + b_xx * (E22**2 + E33**2)

    S11 = 0.5 * K * b_ff * (lmbda**2 - 1.0) * np.exp(W) + p / (lmbda**2)
    S22 = 0.5 * K * b_xx * (1 / lmbda - 1.0) * np.exp(W) + p * lmbda

    return np.array([Ta - S11, eta * Ta - S22], dtype=np.float64)


@numba.jit(nopython=True)
def jac(x, *args):
    lmbda, p = x

    dS11_dlmbda = (
        1.0
        * K
        * b_ff
        * lmbda
        * np.exp(0.25 * b_ff * (lmbda**2 - 1) ** 2 + 0.5 * b_xx * (-1 + 1 / lmbda) ** 2)
        + 0.5
        * K
        * b_ff
        * (lmbda**2 - 1.0)
        * (1.0 * b_ff * lmbda * (lmbda**2 - 1) - 1.0 * b_xx * (-1 + 1 / lmbda) / lmbda**2)
        * np.exp(0.25 * b_ff * (lmbda**2 - 1) ** 2 + 0.5 * b_xx * (-1 + 1 / lmbda) ** 2)
        - 2 * p / lmbda**3
    )
    dS11_dp = lmbda ** (-2)
    dS22_dlmbda = (
        0.5
        * K
        * b_xx
        * (-1.0 + 1 / lmbda)
        * (1.0 * b_ff * lmbda * (lmbda**2 - 1) - 1.0 * b_xx * (-1 + 1 / lmbda) / lmbda**2)
        * np.exp(0.25 * b_ff * (lmbda**2 - 1) ** 2 + 0.5 * b_xx * (-1 + 1 / lmbda) ** 2)
        - 0.5
        * K
        * b_xx
        * np.exp(0.25 * b_ff * (lmbda**2 - 1) ** 2 + 0.5 * b_xx * (-1 + 1 / lmbda) ** 2)
        / lmbda**2
        + p
    )
    dS22_dp = lmbda

    return np.array([[dS11_dlmbda, dS11_dp], [dS22_dlmbda, dS22_dp]], dtype=np.float64)


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
ps = np.zeros(len(t))

for i, ti in tqdm(enumerate(t), total=len(t)):
    y = fgr(y, ti, dt, params)
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, params)
    Ta[i] = monitor[Ta_index]

    # Calculate lambda and p using root finding
    res = root(
        func,
        np.array([lmbda_value, p_value]),
        args=(Ta[i], 0.0),
        jac=jac,
        method="hybr",
    )

    lmbda_value, p_value = res.x
    lmbdas[i] = lmbda_value
    ps[i] = p_value

    params[lmbda_index] = lmbda_value
    params[dLambda_index] = lmbda_value - prev_lmbda
    prev_lmbda = lmbda_value
    Istim[i] = monitor[Istim_index]


# And plot the results
fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, Istim)
ax[0, 2].plot(t, lmbdas)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel("Istim (uA/cm^2)")
ax[0, 2].set_ylabel("Lambda")
ax[1, 2].set_ylabel("p")
fig.tight_layout()
fig.savefig("results.png")
