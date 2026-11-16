import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# =============================================================================
# Cardiovascular System Model (Kim et al. 2023 approximations)
# =============================================================================


class BiventricularCVSModel:
    def __init__(self, E_max_LV=2.8, E_max_RV=0.55, disease_state=False):
        # Contractility (active elastance)
        self.E_max_LV = E_max_LV
        self.E_max_RV = E_max_RV
        self.disease_state = disease_state

        TBV_ref = 5000.0

        # Compliances (mL/mmHg) - derived from Table 3 stressed volume / max pressure
        self.C_SA = (1.0 - 0.70) * (0.20 * TBV_ref) / 120.0
        self.C_SV = (1.0 - 0.80) * (0.54 * TBV_ref) / 6.0
        self.C_PA = (1.0 - 0.40) * (0.05 * TBV_ref) / 25.0
        self.C_PV = (1.0 - 0.90) * (0.15 * TBV_ref) / 8.0

        # Unstressed Volumes (mL) - distributed based on TBV fractions (Table 3)
        if disease_state:
            self.V_u_SA = 0.60 * (0.20 * TBV_ref)  # Shifts SA
            self.V_u_SV = 0.90 * (0.54 * TBV_ref)  # Pools blood in systemic veins
            self.V_u_PA = 0.35 * (0.05 * TBV_ref)
            self.V_u_PV = 0.75 * (0.15 * TBV_ref)
        else:
            self.V_u_SA = 0.70 * (0.20 * TBV_ref)
            self.V_u_SV = 0.80 * (0.54 * TBV_ref)
            self.V_u_PA = 0.40 * (0.05 * TBV_ref)
            self.V_u_PV = 0.90 * (0.15 * TBV_ref)

        # Resistances (mmHg s/mL)
        self.R_SA = 1.33  # Systemic Vascular Resistance
        self.R_PA = 0.146  # Pulmonary Vascular Resistance
        self.R_valves = 0.005  # Minimal valve resistance

        # Timing (60 bpm)
        self.T = 1.0
        self.T_S = 0.20  # Systole duration
        self.T_R = 0.20  # Relaxation duration

    def activation(self, t):
        """Piecewise sinusoidal activation function Y(t)"""
        tc = t % self.T
        if tc <= self.T_S:
            return 0.5 * (1.0 - np.cos(np.pi * tc / self.T_S))
        elif tc <= self.T_S + self.T_R:
            return 0.5 * (1.0 + np.cos(np.pi * (tc - self.T_S) / self.T_R))
        return 0.0

    def calc_pressures(self, y, t):
        V_LV, V_SA, V_SV, V_RV, V_PA, V_PV = y

        # Compartment Pressures
        P_SA = max(0, (V_SA - self.V_u_SA) / self.C_SA)
        P_SV = max(0, (V_SV - self.V_u_SV) / self.C_SV)
        P_PA = max(0, (V_PA - self.V_u_PA) / self.C_PA)
        P_PV = max(0, (V_PV - self.V_u_PV) / self.C_PV)

        # Passive Myocardial Pressures (Exponential approximation of Klotz)
        P_pas_LV = 0.06 * (np.exp(0.035 * V_LV) - 1.0)
        P_pas_RV = 0.02 * (np.exp(0.028 * V_RV) - 1.0)

        # Active Myocardial Pressures
        P_act_LV = self.E_max_LV * max(0, V_LV - 10.0)
        P_act_RV = self.E_max_RV * max(0, V_RV - 15.0)

        Y = self.activation(t)

        # Pericardial Constraint (Eq 14 from Kim 2023)
        V_h = V_LV + V_RV
        V_h0 = 280.0  # Approximate combined EDV boundary
        s = 10.0
        P_peri = np.exp(s * (V_h / V_h0 - 1.0))

        # Total Ventricular Pressures
        P_LV = P_pas_LV + Y * P_act_LV + P_peri
        P_RV = P_pas_RV + Y * P_act_RV + P_peri

        return P_LV, P_SA, P_SV, P_RV, P_PA, P_PV

    def derivatives(self, t, y):
        P_LV, P_SA, P_SV, P_RV, P_PA, P_PV = self.calc_pressures(y, t)

        # Valve Flows (diodes)
        Q_m = max(0, P_PV - P_LV) / self.R_valves  # Mitral
        Q_a = max(0, P_LV - P_SA) / self.R_valves  # Aortic
        Q_t = max(0, P_SV - P_RV) / self.R_valves  # Tricuspid
        Q_p = max(0, P_RV - P_PA) / self.R_valves  # Pulmonary

        # Capillary Flows
        Q_sys = (P_SA - P_SV) / self.R_SA
        Q_pul = (P_PA - P_PV) / self.R_PA

        # Volume changes
        dV_LV = Q_m - Q_a
        dV_SA = Q_a - Q_sys
        dV_SV = Q_sys - Q_t
        dV_RV = Q_t - Q_p
        dV_PA = Q_p - Q_pul
        dV_PV = Q_pul - Q_m

        return [dV_LV, dV_SA, dV_SV, dV_RV, dV_PA, dV_PV]

    def simulate(self, TBV=5000.0, num_cycles=15, return_cycles=1):
        """Simulates the CVS until steady state is reached."""
        y0 = [
            100.0,
            self.V_u_SA + self.C_SA * 95,
            self.V_u_SV + self.C_SV * 5,
            100.0,
            self.V_u_PA + self.C_PA * 15,
            self.V_u_PV + self.C_PV * 5,
        ]

        curr_vol = sum(y0)
        y0 = [val * (TBV / curr_vol) for val in y0]  # Conserve exact TBV

        t_span = (0, num_cycles * self.T)
        t_eval = np.linspace(0, num_cycles * self.T, num_cycles * 500)

        # Radau solver needed for stiff valve transitions
        sol = solve_ivp(self.derivatives, t_span, y0, method="Radau", t_eval=t_eval, max_step=0.01)

        # Extract the requested completed cycles
        mask = sol.t >= (num_cycles - return_cycles) * self.T
        t_last = sol.t[mask] - (num_cycles - return_cycles) * self.T
        y_last = sol.y[:, mask]

        pressures = np.array(
            [self.calc_pressures(y_last[:, i], t_last[i]) for i in range(len(t_last))]
        )
        return t_last, y_last, pressures


# =============================================================================
# Helper Functions
# =============================================================================


def extract_loop_metrics(y_last, pressures):
    """Extracts ESP, EDP, ESV, EDV, and SV for both ventricles."""
    V_LV = y_last[0, :]
    P_LV = pressures[:, 0]
    V_RV = y_last[3, :]
    P_RV = pressures[:, 3]

    # Left Ventricle Metrics
    EDV_LV = np.max(V_LV)
    ESV_LV = np.min(V_LV)
    SV_LV = EDV_LV - ESV_LV
    EDP_LV = P_LV[np.argmax(V_LV)]
    ESP_LV = P_LV[np.argmin(V_LV)]

    # Right Ventricle Metrics
    EDV_RV = np.max(V_RV)
    ESV_RV = np.min(V_RV)
    SV_RV = EDV_RV - ESV_RV
    EDP_RV = P_RV[np.argmax(V_RV)]
    ESP_RV = P_RV[np.argmin(V_RV)]

    return {
        "LV": {"EDV": EDV_LV, "ESV": ESV_LV, "SV": SV_LV, "EDP": EDP_LV, "ESP": ESP_LV},
        "RV": {"EDV": EDV_RV, "ESV": ESV_RV, "SV": SV_RV, "EDP": EDP_RV, "ESP": ESP_RV},
    }


def find_compensated_tbv(model, target_P_SA=95.0):
    """Finds the required Total Blood Volume to hit a target mean systemic pressure."""

    def objective(tbv):
        _, _, pressures = model.simulate(TBV=tbv, num_cycles=12)
        mean_P_SA = np.mean(pressures[:, 1])
        return mean_P_SA - target_P_SA

    print(f"  -> Compensating Volume for E_max_LV={model.E_max_LV:.2f}...")
    try:
        res = root_scalar(objective, bracket=[4000, 15000], method="brentq", xtol=50)
        return res.root
    except ValueError:
        print("     Could not reach target P_SA within bounds. Defaulting to 15000 mL.")
        return 15000.0


def generate_loading_curves(model, tbv_range):
    """Simulates the model over a range of TBVs to map the EDPVR/ESPVR."""
    results = {
        "LV_EDV": [],
        "LV_ESP": [],
        "LV_EDP": [],
        "LV_SV": [],
        "LV_ESV": [],
        "RV_EDV": [],
        "RV_ESP": [],
        "RV_EDP": [],
        "RV_SV": [],
        "RV_ESV": [],
    }
    for tbv in tbv_range:
        _, y_last, pressures = model.simulate(TBV=tbv, num_cycles=8)
        m = extract_loop_metrics(y_last, pressures)

        results["LV_EDV"].append(m["LV"]["EDV"])
        results["LV_ESP"].append(m["LV"]["ESP"])
        results["LV_EDP"].append(m["LV"]["EDP"])
        results["LV_SV"].append(m["LV"]["SV"])
        results["LV_ESV"].append(m["LV"]["ESV"])

        results["RV_EDV"].append(m["RV"]["EDV"])
        results["RV_ESP"].append(m["RV"]["ESP"])
        results["RV_EDP"].append(m["RV"]["EDP"])
        results["RV_SV"].append(m["RV"]["SV"])
        results["RV_ESV"].append(m["RV"]["ESV"])
    return results


# =============================================================================
# Plotting Figure 3 (Healthy Baseline)
# =============================================================================


def plot_figure_3():
    model = BiventricularCVSModel(disease_state=False)
    tbv_healthy = find_compensated_tbv(model, target_P_SA=95.0)
    t, y, P = model.simulate(TBV=tbv_healthy, num_cycles=15, return_cycles=2)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Figure 3: Healthy Baseline (Mean PSA ≈ 95 mmHg)", fontsize=14)

    # (A) PV Loops
    axs[0, 0].plot(y[0, :], P[:, 0], "r-", label="LV", linewidth=2)
    axs[0, 0].plot(y[3, :], P[:, 3], "b-", label="RV", linewidth=2)

    v_klotz_lv = np.linspace(40, 200, 100)
    axs[0, 0].plot(
        v_klotz_lv,
        0.06 * (np.exp(0.035 * v_klotz_lv) - 1.0),
        color="gray",
        alpha=0.5,
        label="Klotz Model",
    )

    axs[0, 0].set_title("(A) Ventricular PV Loops")
    axs[0, 0].set_xlabel("Volume (mL)")
    axs[0, 0].set_ylabel("Pressure (mmHg)")
    axs[0, 0].set_xlim([0, 200])
    axs[0, 0].set_ylim([0, 150])
    axs[0, 0].legend()

    # (B) Volumes over time
    axs[0, 1].plot(t, y[0, :], "r-", label="LV")
    axs[0, 1].plot(t, y[3, :], "b-", label="RV")
    axs[0, 1].set_title("(B) Ventricular Volumes")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Volume (mL)")
    axs[0, 1].set_xlim([0, 2])
    axs[0, 1].set_ylim([40, 120])
    axs[0, 1].legend()

    # (C) LV & Systemic Arterial Pressures
    axs[1, 0].plot(t, P[:, 0], "r-", label="$P_{LV}$")
    axs[1, 0].plot(t, P[:, 1], "orange", label="$P_{SA}$")
    axs[1, 0].set_title("(C) LV and Systemic Pressures")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Pressure (mmHg)")
    axs[1, 0].set_xlim([0, 2])
    axs[1, 0].set_ylim([0, 130])
    axs[1, 0].legend()

    # (D) RV & Pulmonary/Venous Pressures
    axs[1, 1].plot(t, P[:, 3], "b-", label="$P_{RV}$")
    axs[1, 1].plot(t, P[:, 4], "cyan", label="$P_{PA}$")
    axs[1, 1].plot(t, P[:, 2], "magenta", label="$P_{SV}$")
    axs[1, 1].plot(t, P[:, 5], "k-", label="$P_{PV}$")
    axs[1, 1].set_title("(D) RV and Right-Heart Pressures")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Pressure (mmHg)")
    axs[1, 1].set_xlim([0, 2])
    axs[1, 1].set_ylim([0, 30])
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig("figure_3_healthy_baseline.png", dpi=300)


# =============================================================================
# Plotting Figure 5 (LV Systolic Dysfunction & Compensation)
# =============================================================================


def plot_figure_5():
    print("\nRunning simulations for Figure 5 (LV Systolic Dysfunction)...")

    # Define models
    model_H = BiventricularCVSModel(E_max_LV=2.8, E_max_RV=0.55, disease_state=False)  # Healthy
    model_M = BiventricularCVSModel(
        E_max_LV=2.8 * 0.6, E_max_RV=0.55, disease_state=True
    )  # Moderate LVSD
    model_S = BiventricularCVSModel(
        E_max_LV=2.8 * 0.4, E_max_RV=0.55, disease_state=True
    )  # Severe LVSD

    # Find compensated volumes (shifts right to maintain PSA)
    tbv_H = find_compensated_tbv(model_H)
    tbv_M = find_compensated_tbv(model_M)
    tbv_S = find_compensated_tbv(model_S)

    # 1. Uncompensated (All evaluated at normal circulating blood volume)
    _, y_H_uncomp, P_H_uncomp = model_H.simulate(TBV=5000.0)
    _, y_M_uncomp, P_M_uncomp = model_M.simulate(TBV=5000.0)
    _, y_S_uncomp, P_S_uncomp = model_S.simulate(TBV=5000.0)

    # 2. Compensated (Volume Loaded)
    _, y_H_comp, P_H_comp = model_H.simulate(TBV=tbv_H)
    _, y_M_comp, P_M_comp = model_M.simulate(TBV=tbv_M)
    _, y_S_comp, P_S_comp = model_S.simulate(TBV=tbv_S)

    # 3. Generate Curves for C, D, G, H (driving up to the Pericardial constraint)
    tbvs = np.linspace(4500, 13000, 10)
    crv_H = generate_loading_curves(model_H, tbvs)
    crv_M = generate_loading_curves(model_M, tbvs)
    crv_S = generate_loading_curves(model_S, tbvs)

    # Extract base markers for compensated state
    m_H = extract_loop_metrics(y_H_comp, P_H_comp)
    m_M = extract_loop_metrics(y_M_comp, P_M_comp)
    m_S = extract_loop_metrics(y_S_comp, P_S_comp)

    # --- Plotting ---
    fig, axs = plt.subplots(4, 2, figsize=(10, 15))
    fig.suptitle("Figure 5: Left Ventricular Systolic Dysfunction (LVSD)", fontsize=14, y=0.98)

    styles = {
        "H": {"ls": "-", "lbl": "H"},
        "M": {"ls": ":", "lbl": "M"},
        "S": {"ls": "--", "lbl": "S"},
    }

    v_klotz_lv = np.linspace(40, 200, 100)
    v_klotz_rv = np.linspace(40, 200, 100)
    p_klotz_lv = 0.06 * (np.exp(0.035 * v_klotz_lv) - 1.0)
    p_klotz_rv = 0.02 * (np.exp(0.028 * v_klotz_rv) - 1.0)

    # (A, B) Uncompensated PV Loops
    for y_dat, P_dat, st in zip(
        [y_H_uncomp, y_M_uncomp, y_S_uncomp], [P_H_uncomp, P_M_uncomp, P_S_uncomp], ["H", "M", "S"]
    ):
        axs[0, 0].plot(
            y_dat[0, :], P_dat[:, 0], color="red", ls=styles[st]["ls"], label=styles[st]["lbl"]
        )
        axs[0, 1].plot(
            y_dat[3, :], P_dat[:, 3], color="blue", ls=styles[st]["ls"], label=styles[st]["lbl"]
        )

    axs[0, 0].plot(v_klotz_lv, p_klotz_lv, color="gray", alpha=0.5)
    axs[0, 1].plot(v_klotz_rv, p_klotz_rv, color="gray", alpha=0.5)
    axs[0, 0].set_title("(A) Uncompensated LV")
    axs[0, 1].set_title("(B) Uncompensated RV")
    axs[0, 0].legend()
    axs[0, 1].legend()

    # (C, D) ESPVR and EDPVR loading curves
    for crv, st in zip([crv_H, crv_M, crv_S], ["H", "M", "S"]):
        # LV Curves (ESV vs ESP for ESPVR)
        axs[1, 0].plot(crv["LV_ESV"], crv["LV_ESP"], color="red", ls=styles[st]["ls"])
        axs[1, 0].plot(crv["LV_EDV"], crv["LV_EDP"], color="blue", ls=styles[st]["ls"])
        # RV Curves
        axs[1, 1].plot(crv["RV_ESV"], crv["RV_ESP"], color="red", ls=styles[st]["ls"])
        axs[1, 1].plot(crv["RV_EDV"], crv["RV_EDP"], color="blue", ls=styles[st]["ls"])

    # Mark the compensated endpoints on the curves
    for m, mark in zip([m_H, m_M, m_S], ["o", "^", "s"]):
        axs[1, 0].plot(m["LV"]["ESV"], m["LV"]["ESP"], marker=mark, color="black", markersize=6)
        axs[1, 0].plot(m["LV"]["EDV"], m["LV"]["EDP"], marker=mark, color="black", markersize=6)
        axs[1, 1].plot(m["RV"]["ESV"], m["RV"]["ESP"], marker=mark, color="black", markersize=6)
        axs[1, 1].plot(m["RV"]["EDV"], m["RV"]["EDP"], marker=mark, color="black", markersize=6)

    axs[1, 0].set_title("(C) LV ESPVR/EDPVR")
    axs[1, 1].set_title("(D) RV ESPVR/EDPVR")

    # (E, F) Compensated PV Loops
    for y_dat, P_dat, st in zip(
        [y_H_comp, y_M_comp, y_S_comp], [P_H_comp, P_M_comp, P_S_comp], ["H", "M", "S"]
    ):
        axs[2, 0].plot(y_dat[0, :], P_dat[:, 0], color="red", ls=styles[st]["ls"])
        axs[2, 1].plot(y_dat[3, :], P_dat[:, 3], color="blue", ls=styles[st]["ls"])

    axs[2, 0].plot(v_klotz_lv, p_klotz_lv, color="gray", alpha=0.5)
    axs[2, 1].plot(v_klotz_rv, p_klotz_rv, color="gray", alpha=0.5)
    axs[2, 0].set_title("(E) Compensated LV")
    axs[2, 1].set_title("(F) Compensated RV")

    # (G, H) Frank-Starling Curves (SV vs EDP)
    for crv, st in zip([crv_H, crv_M, crv_S], ["H", "M", "S"]):
        axs[3, 0].plot(crv["LV_EDP"], crv["LV_SV"], color="green", marker=".", ls=styles[st]["ls"])
        axs[3, 1].plot(crv["RV_EDP"], crv["RV_SV"], color="green", marker=".", ls=styles[st]["ls"])

    # Mark the compensated endpoints on Frank-Starling
    for m, mark in zip([m_H, m_M, m_S], ["o", "^", "s"]):
        axs[3, 0].plot(m["LV"]["EDP"], m["LV"]["SV"], marker=mark, color="black", markersize=6)
        axs[3, 1].plot(m["RV"]["EDP"], m["RV"]["SV"], marker=mark, color="black", markersize=6)

    axs[3, 0].set_title("(G) LV Frank-Starling")
    axs[3, 1].set_title("(H) RV Frank-Starling")

    # Formatting passes
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    for i in range(3):
        axs[i, 0].set_ylabel("Pressure (mmHg)")
        axs[i, 0].set_xlabel("Volume (mL)")
        axs[i, 1].set_ylabel("Pressure (mmHg)")
        axs[i, 1].set_xlabel("Volume (mL)")

    axs[3, 0].set_ylabel("SV (mL)")
    axs[3, 0].set_xlabel("EDP (mmHg)")
    axs[3, 1].set_ylabel("SV (mL)")
    axs[3, 1].set_xlabel("EDP (mmHg)")

    # Standardize axes to match the paper exactly
    axs[0, 0].set_xlim([0, 200])
    axs[0, 0].set_ylim([0, 150])
    axs[0, 1].set_xlim([0, 200])
    axs[0, 1].set_ylim([0, 60])

    axs[1, 0].set_xlim([0, 150])
    axs[1, 0].set_ylim([0, 150])
    axs[1, 1].set_xlim([50, 250])
    axs[1, 1].set_ylim([0, 50])

    axs[2, 0].set_xlim([0, 200])
    axs[2, 0].set_ylim([0, 150])
    axs[2, 1].set_xlim([0, 200])
    axs[2, 1].set_ylim([0, 60])

    axs[3, 0].set_xlim([0, 45])
    axs[3, 0].set_ylim([40, 105])
    axs[3, 1].set_xlim([0, 15])
    axs[3, 1].set_ylim([40, 105])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figure_5_lv_systolic_dysfunction.png", dpi=300)


if __name__ == "__main__":
    # Execute Figure Reproductions
    plot_figure_3()
    plot_figure_5()
