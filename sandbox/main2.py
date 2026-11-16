import numpy as np
from scipy.integrate import solve_ivp
import pytest
import matplotlib.pyplot as plt
import sympy as sp
import zero_mech
from dataclasses import dataclass, field

# ==============================================================================
# Modular Components
# ==============================================================================


@dataclass(slots=True, frozen=True)
class PaperMaterial(zero_mech.atoms.AbstractStrainEnergy):
    """
    Custom material model representing the specific hyperelastic
    strain energy from Caruel et al. 2014.
    """

    C0: sp.Symbol = sp.Symbol("C0")
    C1: sp.Symbol = sp.Symbol("C1")
    # Fiber oriented along phi1 (index 1)
    f0: sp.Matrix = field(default_factory=lambda: sp.Matrix([0, 1, 0]))

    def strain_energy(self, F: sp.Matrix) -> sp.Expr:
        C = F.T @ F
        I1 = sp.trace(C)
        I4 = self.f0.dot(C * self.f0)
        return self.C0 * sp.exp(self.C1 * (I1 - 3) ** 2) + self.C0 * sp.exp(self.C1 * (I4 - 1) ** 2)

    def default_parameters(self):
        return {self.C0: 1.9e3, self.C1: 1.1e-1}

    @staticmethod
    def str() -> str:
        return "C0 * exp(C1 * (I1 - 3)**2) + C0 * exp(C1 * (I4 - 1)**2)"


class ZeroMechSphericalMaterial:
    """
    Passive material model using `zero_mech` to symbolically derive
    the spherical stresses from the strain energy function.
    """

    # Cache the lambdified function at the class level to avoid recompiling on multiple instantiations
    _eval_elastic_stress = None

    def __init__(self, C0=1.9e3, C1=1.1e-1, eta=70.0):
        self.C0 = C0
        self.C1 = C1
        self.eta = eta

        if ZeroMechSphericalMaterial._eval_elastic_stress is None:
            ZeroMechSphericalMaterial._eval_elastic_stress = self._build_symbolic_stress()

    @staticmethod
    def _build_symbolic_stress():
        lmbda = sp.Symbol("lmbda", real=True, positive=True)
        # Define 0D spherical kinematics (radial, angular1, angular2)
        experiment = zero_mech.experiments.MechanicalExperiment(
            F11=lmbda ** (-2), F22=lmbda, F33=lmbda
        )

        mat = PaperMaterial()
        comp = zero_mech.compressibility.Incompressible()
        act = zero_mech.active.Passive()

        mech_model = zero_mech.Model(material=mat, compressibility=comp, active=act)

        # 2nd Piola-Kirchhoff stress
        S = mech_model.second_piola_kirchhoff(experiment.F)
        p = mech_model["p"]

        # Enforce zero radial stress to find the hydrostatic pressure p
        p_sol = sp.solve(S[0, 0], p)[0]

        # Compute total spherical passive stress (S11 + S22)
        S11 = S[1, 1].subs(p, p_sol)
        S22 = S[2, 2].subs(p, p_sol)
        Sigma_sph_elastic = sp.simplify(S11 + S22)

        return sp.lambdify((lmbda, mat.C0, mat.C1), Sigma_sph_elastic, "numpy")

    def get_passive_spherical_stress(self, C, C_dot):
        """
        Calculates the passive elastic and viscous components of the spherical stress.
        """
        lmbda_val = np.sqrt(C)

        # Evaluate symbolically derived elastic part
        # Note: Called via the class directly to avoid Python's descriptor protocol injecting `self`
        elastic_term = ZeroMechSphericalMaterial._eval_elastic_stress(lmbda_val, self.C0, self.C1)

        # Viscous term (added manually as it acts on the strain rate specifically)
        visc_term = self.eta * C_dot * (1.0 + 2.0 * C ** (-6))

        return elastic_term + visc_term


class TwoStageWindkesselCirculation:
    """
    Two-stage Windkessel model representing the systemic circulation and
    associated piecewise valve dynamics.
    """

    def __init__(
        self,
        Rp=1.0e6,
        Cp=2.5e-9,
        Rd=1.0e8,
        Cd=1.0e-8,
        Kat=9.0e-6,
        Kp=5.0e-10,
        Kar=1.3e-5,
        Pvs=1.0e3,
        T_cycle=0.8,
    ):
        self.Rp = Rp
        self.Cp = Cp
        self.Rd = Rd
        self.Cd = Cd
        self.Kat = Kat
        self.Kp = Kp
        self.Kar = Kar
        self.Pvs = Pvs
        self.T_cycle = T_cycle

    @property
    def num_states(self):
        """Number of ODE state variables required by this circulation model."""
        return 2

    def init_state(self):
        """Initial conditions for [Par, Pd]"""
        # Start at roughly 75 mmHg and 37.5 mmHg to reduce transient time
        return [10000.0, 5000.0]

    def get_atrial_pressure(self, t):
        """Prescribed Atrial Pressure."""
        tc = t % self.T_cycle
        base_P = 800.0  # ~6 mmHg
        # Atrial kick happens at the start of the cycle (t=0.0 to t=0.15)
        if tc < 0.15:
            return base_P + 400.0 * np.sin(np.pi * tc / 0.15)
        return base_P

    def get_ventricular_pressure(self, t, Q, state):
        """
        Inverts the piecewise flow equation Q = q(Pv, Par, Pat) to solve for Pv.
        (Required since Q is kinematically locked to velocity v in the 0D ODEs)
        """
        Par = state[0]
        Pat = self.get_atrial_pressure(t)

        Q_isovol_max = self.Kp * (Par - Pat)

        if Q <= 0:
            # Filling phase (Mitral valve open, Aortic closed)
            Pv = Pat + Q / self.Kat
        elif Q <= Q_isovol_max:
            # Isovolumic phase (Both valves essentially closed)
            Pv = Pat + Q / self.Kp
        else:
            # Ejection phase (Aortic valve open, Mitral closed)
            Pv = Par + (Q - Q_isovol_max) / self.Kar

        return Pv

    def get_derivatives(self, t, Q, state, Pv):
        """Derivatives for the Windkessel ODEs."""
        Par, Pd = state

        # Aortic flow is strictly the positive flow into the aorta.
        # During filling (Q < 0) or isovolumic phase (Pv < Par), the aortic valve is closed.
        if Pv > Par:
            Q_aorta = self.Kar * (Pv - Par)
        else:
            Q_aorta = 0.0

        Par_dot = (1.0 / self.Cp) * (Q_aorta - (Par - Pd) / self.Rp)
        Pd_dot = (1.0 / self.Cd) * ((Par - Pd) / self.Rp - (Pd - self.Pvs) / self.Rd)

        return [Par_dot, Pd_dot]


# ==============================================================================
# Core Cardiac 0D Model
# ==============================================================================


class Cardiac0DModel:
    """
    Modular 0D Spherical Cardiac Model based on Caruel et al. 2014.
    Couples structural kinematics, active sarcomere contraction, and injected
    passive material / systemic circulation models.
    """

    def __init__(self, material_model=None, circulation_model=None, params=None):
        self.material = material_model if material_model else ZeroMechSphericalMaterial()
        self.circulation = (
            circulation_model if circulation_model else TwoStageWindkesselCirculation()
        )

        # Default Active Model & Geometric Parameters
        self.params = {
            "sigma_0": 1.24e5,  # Pa
            "k0": 1e5,  # Pa
            "mu": 70.0,  # Pa.s
            "Es": 3.0e7,  # Pa
            "alpha": 1.5,  # Dimensionless
            "alpha_r": 0.12,  # s
            "R0": 2.36e-2,  # m
            "d0": 1.42e-2,  # m
            "rho": 1.0e3,  # kg/m^3
            "T_cycle": 0.8,  # s
        }
        if params is not None:
            self.params.update(params)

    # --- Active Contraction Functions ---

    def u_bar(self, t):
        """Activation function representing electrical depolarization/repolarization."""
        tc = t % self.params["T_cycle"]

        # Matched to the physiological timing of Figure 10:
        if tc < 0.1:
            return 0.0
        elif tc < 0.15:
            # Fast depolarization rise
            return 30.0 * ((tc - 0.1) / 0.05)
        elif tc < 0.35:
            # Systolic plateau
            return 30.0
        elif tc < 0.45:
            # Repolarization drop
            return 30.0 - 40.0 * ((tc - 0.35) / 0.10)
        elif tc < 0.55:
            # Relaxation phase
            return -10.0
        elif tc < 0.65:
            # Return to resting baseline
            return -10.0 + 10.0 * ((tc - 0.55) / 0.10)
        else:
            return 0.0

    def n0(self, ec):
        """Length-dependence function for contraction (Frank-Starling effect).
        Matches piecewise profile from Figure 7a in Caruel et al."""
        if ec < -0.3:
            return 0.0
        elif ec < 1.0:
            return (ec + 0.3) / 1.3
        elif ec <= 1.3:
            return 1.0
        else:
            return max(0.0, 1.0 - (ec - 1.3) / 0.5)

    def m0(self, ec):
        """Load-dependent relaxation function.
        Matches the decreasing profile from Figure 7a."""
        val = 1.7 - 0.7 * ec
        return np.clip(val, 1.0, 1.7)

    # --- System of ODEs ---

    def system_derivatives(self, t, state):
        """
        Calculates the derivatives for the fully coupled system.
        State vector: [y, v, e_c, k_c, tau_c, w] + [Circulation States...]
        """
        # Unpack state
        y, v, ec, kc, tau_c, w = state[:6]
        circ_state = state[6:]
        p = self.params

        # Kinematics
        R0 = p["R0"]
        C_sqrt = 1.0 + y / R0
        C = C_sqrt**2

        # Guard against unphysical compression
        C = max(C, 1e-4)
        C_dot = 2.0 * C_sqrt * v / R0
        e_1D = (C - 1.0) / 2.0

        # Passive and Active Stress branch
        denom_ec = max(1.0 + 2.0 * ec, 1e-4)  # Guard against singular strain division
        sigma_1D = p["Es"] * (e_1D - ec) / (denom_ec**2)

        Sigma_passive = self.material.get_passive_spherical_stress(C, C_dot)
        Sigma_sph = sigma_1D + Sigma_passive

        # e_c dynamics (algebraic-differential relation)
        ec_dot = (1.0 / p["mu"]) * (
            p["Es"] * (e_1D - ec) * (1.0 + 2.0 * e_1D) / (denom_ec**3) - tau_c
        )

        # Sarcomere activation
        u_val = self.u_bar(t)
        u_plus = max(u_val, 0.0)
        u_minus = max(-u_val, 0.0)

        activation_rate = u_plus + w * u_minus + p["alpha"] * abs(ec_dot)
        kc_dot = -activation_rate * kc + self.n0(ec) * p["k0"] * u_plus
        tau_c_dot = -activation_rate * tau_c + self.n0(ec) * p["sigma_0"] * u_plus + kc * ec_dot
        w_dot = (self.m0(ec) - w) / p["alpha_r"]

        # Fluid Dynamics & Circulation coupling
        Q = -4.0 * np.pi * R0**2 * C_sqrt**2 * v
        Pv = self.circulation.get_ventricular_pressure(t, Q, circ_state)
        circ_derivs = self.circulation.get_derivatives(t, Q, circ_state, Pv)

        # Mechanical equation of motion for y
        y_ddot = (1.0 / (p["rho"] * p["d0"])) * (
            Pv * C_sqrt**2 - (p["d0"] / R0) * C_sqrt * Sigma_sph
        )

        return [v, y_ddot, ec_dot, kc_dot, tau_c_dot, w_dot] + circ_derivs

    def simulate(self, t_span, y0=None, **kwargs):
        """Runs the simulation using SciPy's ODE solver."""
        if y0 is None:
            y_init = 0.007  # Starts near End-Diastole
            R0 = self.params["R0"]
            C_init = (1.0 + y_init / R0) ** 2
            e_1D_init = (C_init - 1.0) / 2.0

            # Initialize ec near e_1D to prevent massive spring shock
            ventricle_initial = [y_init, 0.0, e_1D_init, 0.0, 0.0, 1.0]
            y0 = ventricle_initial + self.circulation.init_state()

        kwargs.setdefault("method", "Radau")
        kwargs.setdefault("max_step", 1e-3)

        return solve_ivp(self.system_derivatives, t_span, y0, **kwargs)


# ==============================================================================
# Unit Tests (pytest)
# ==============================================================================


def test_valve_law_inverse():
    circ = TwoStageWindkesselCirculation()

    # Test filling phase (Q < 0)
    Q = -5e-4
    Par = 10000.0
    Pat = 1000.0
    state = [Par, 5000.0]

    # Mocking atrial pressure internally for deterministic testing
    circ.get_atrial_pressure = lambda t: Pat
    Pv = circ.get_ventricular_pressure(0.0, Q, state)
    assert Q == pytest.approx(circ.Kat * (Pv - Pat), abs=1e-8)

    # Test isovolumic phase (0 <= Q <= K_p*(Par - Pat))
    Q = 1e-7
    Pv = circ.get_ventricular_pressure(0.0, Q, state)
    assert Q == pytest.approx(circ.Kp * (Pv - Pat), abs=1e-8)

    # Test ejection phase (Q > K_p*(Par - Pat))
    Q = 5e-4
    Pv = circ.get_ventricular_pressure(0.0, Q, state)
    Q_isovol_max = circ.Kp * (Par - Pat)
    expected_Q = circ.Kar * (Pv - Par) + Q_isovol_max
    assert Q == pytest.approx(expected_Q, abs=1e-8)


def test_strain_invariants_neutral():
    model = Cardiac0DModel()

    # Reference Configuration (y=0, v=0)
    # State: y, v, e_c, k_c, tau_c, w, P_ar, P_d
    state_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + model.circulation.init_state()
    derivs = model.system_derivatives(0.0, state_0)

    y_ddot = derivs[1]

    # Determine the expected Pv given Q=0 at t=0
    circ_state = state_0[6:]
    Pv = model.circulation.get_ventricular_pressure(0.0, 0.0, circ_state)

    # Since passive stretch is 0, Sigma_sph should be 0, leaving only fluid pressure pushing outward
    expected_y_ddot = Pv / (model.params["rho"] * model.params["d0"])
    assert y_ddot == pytest.approx(expected_y_ddot, abs=1e-8)


def test_short_simulation():
    model = Cardiac0DModel()
    sol = model.simulate((0, 0.05))

    assert sol.success
    assert sol.y.shape[0] == 6 + model.circulation.num_states
    assert np.all(np.isfinite(sol.y))


# ==============================================================================
# Paper Example Implementation
# ==============================================================================


def run_paper_example():
    """
    Runs a simulation of the 0D model and plots the cardiac cycle.
    """
    model = Cardiac0DModel()

    T_cycle = model.params["T_cycle"]
    num_cycles = 6
    t_span = (0, num_cycles * T_cycle)
    t_eval = np.linspace(t_span[0], t_span[1], 1500)

    print(f"Running simulation for {num_cycles} cycles...")
    sol = model.simulate(t_span, t_eval=t_eval)
    print("Simulation finished. Generating plots...")

    # Extract the last full cycle to ensure steady state
    mask = sol.t >= (num_cycles - 1) * T_cycle
    t_last = sol.t[mask] - (num_cycles - 1) * T_cycle
    state_last = sol.y[:, mask]

    y, v = state_last[0], state_last[1]
    Par = state_last[6]  # First circulation state is Par

    R0 = model.params["R0"]
    V = (4.0 / 3.0) * np.pi * (R0 + y) ** 3
    Q = -4.0 * np.pi * (R0 + y) ** 2 * v

    # Fetch pressures dynamically via the circulation interface
    Pat = np.array([model.circulation.get_atrial_pressure(t) for t in t_last])
    Pv = np.array(
        [
            model.circulation.get_ventricular_pressure(t, Q[i], state_last[6:, i])
            for i, t in enumerate(t_last)
        ]
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(t_last, V * 1e4, "k-")
    axs[0, 0].set_title("(a) Left ventricular volume")
    axs[0, 0].set_xlabel("time (s)")
    axs[0, 0].set_ylabel("V ($10^{-4}$ m$^3$)")
    axs[0, 0].set_ylim([0.4, 1.3])
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t_last, Q * 1e4, "k-")
    axs[0, 1].set_title("(b) Cardiac outflow")
    axs[0, 1].set_xlabel("time (s)")
    axs[0, 1].set_ylabel("Q ($10^{-4}$ m$^3$/s)")
    axs[0, 1].set_ylim([-8, 8])
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(t_last, Pv * 1e-4, "k-", label="Ventricular $P_v$")
    axs[1, 0].plot(t_last, Par * 1e-4, "r--", alpha=0.7, label="Aortic $P_{ar}$")
    axs[1, 0].plot(t_last, Pat * 1e-4, "g:", alpha=0.7, label="Atrial $P_{at}$")
    axs[1, 0].set_title("(c) Pressures")
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("Pressure ($10^4$ Pa)")
    axs[1, 0].set_ylim([0, 1.6])
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(V * 1e4, Pv * 1e-4, "k-")
    axs[1, 1].set_title("(d) Pressure-Volume cycle")
    axs[1, 1].set_xlabel("V ($10^{-4}$ m$^3$)")
    axs[1, 1].set_ylabel("$P_v$ ($10^4$ Pa)")
    axs[1, 1].set_xlim([0.3, 1.3])
    axs[1, 1].set_ylim([0, 1.6])
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_paper_example()
