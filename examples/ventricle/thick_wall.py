# # Thick walls, solved exactly
#
# `zero_mech.SphericalVentricle` reduces the wall of a sphere to a single stretch, the one-fibre model, which is
# exact only for a thin wall. For an incompressible wall whose deformation keeps the symmetry of its geometry, the
# full problem can be solved instead: the deformation is known in closed form, and radial equilibrium gives the
# cavity pressure as one integral through the wall. `zero_mech.ThickSphere` and `zero_mech.ThickCylinder` compute
# that integral. Since the result is the solution of the three-dimensional problem, it is also a reference to
# verify a finite element solver against.
#
# For a sphere with inner radius $R_i$ and outer radius $R_o$ when unloaded, a point at radius $R$ moves to
# $r(R)^3 = r_i^3 + R^3 - R_i^3$, and the pressure is
#
# $$
# p = \int_{R_i}^{R_o} \lambda \frac{d\hat\psi}{d\lambda} \frac{R^2}{r^3} \, dR, \qquad \lambda = \frac{r}{R},
# $$
#
# with $\hat\psi(\lambda)$ the strain energy of the equibiaxial deformation of the wall.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import zero_mech

# ## A thin and a thick sphere
#
# We compare the two models for a neo-Hookean wall, as the wall gets thicker. The sphere must be isotropic in the
# wall plane to deform symmetrically, so it takes no fibres in that plane.

# +
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for h0 in (0.05, 0.2, 0.5):
    exact = zero_mech.ThickSphere(Ri=1.0, Ro=1.0 + h0)
    one_fibre = zero_mech.SphericalVentricle.from_radius(1.0, h0)
    V = np.linspace(1.0, 3.0, 50) * exact.V0
    (line,) = axes[0].plot(V / exact.V0, exact.pressure(V), label=f"exact, $h_0/R_0$ = {h0}")
    axes[0].plot(V / exact.V0, one_fibre.pressure(V), "--", color=line.get_color())
    axes[1].plot(V[1:] / exact.V0, one_fibre.pressure(V[1:]) / exact.pressure(V[1:]) - 1, label=f"{h0}")
axes[0].set_xlabel("$V / V_0$")
axes[0].set_ylabel("Pressure (unit of $\\mu$)")
axes[0].set_title("Exact (solid) and one-fibre (dashed)")
axes[0].legend()
axes[1].set_xlabel("$V / V_0$")
axes[1].set_ylabel("Relative difference")
axes[1].set_title("One-fibre model against the exact solution")
axes[1].legend(title="$h_0 / R_0$")
fig.tight_layout()
plt.show()
# -

# The one-fibre model overestimates the pressure by 2 to 4 percent for a wall a twentieth of the radius thick,
# and by up to about 12 percent for the thicker walls. It is exact only in the limit of a thin wall.
#
# ## A contracting tube
#
# A tube in plane strain can have fibres, as long as they run along its axes. Here the fibres are circumferential
# (the first axis of the model, the default of the materials) and the sheets radial (the second axis). The fibres
# carry an active stress, `zero_mech.StretchActiveStress`, whose force per unit reference area is the active
# tension $T_a$. The parameters are those of the transversely isotropic Holzapfel-Ogden model in kPa.

# +
model = zero_mech.Model(
    material=zero_mech.material.HolzapfelOgden(use_heaviside=True),
    active=zero_mech.StretchActiveStress(),
)
parameters = dict(a=2.280, b=9.726, a_f=1.685, b_f=15.779, a_s=0.0, b_s=0.0, a_fs=0.0, b_fs=0.0)
tube = zero_mech.ThickCylinder(model, Ri=1.0, Ro=1.5, parameters=parameters)
# -

# With no pressure in the cavity, the tube contracts until the passive stress balances the active one. The cavity
# volume where that happens is a root of the pressure

# +
fig, ax = plt.subplots()
for Ta in (2.0, 5.0, 10.0):
    V = brentq(lambda V: tube.pressure(V, parameters={"Ta": Ta}), 0.3 * tube.V0, tube.V0)
    ax.plot(tube.R, tube.stretch(V), label=f"$T_a$ = {Ta} kPa, $V / V_0$ = {V / tube.V0:.3f}")
ax.set_xlabel("Reference radius $R$")
ax.set_ylabel("Circumferential stretch")
ax.set_title("Free contraction of a thick tube")
ax.legend()
plt.show()
# -

# The inner surface shortens more than the outer one. An active tension can also vary through the wall: any
# parameter can be given one value per quadrature point `tube.R`, for instance a tension computed from the stretch
# at each point by a cell model.

Ta = np.linspace(10.0, 2.0, tube.R.size)  # from the inner to the outer surface
V = brentq(lambda V: tube.pressure(V, parameters={"Ta": Ta}), 0.3 * tube.V0, tube.V0)
print(f"V / V0 = {V / tube.V0:.4f}, stretch from {tube.stretch(V)[0]:.4f} to {tube.stretch(V)[-1]:.4f}")
