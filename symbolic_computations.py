import sympy as sp


def guccione():
    K, b_ff, b_xx = sp.symbols("K b_ff b_xx")
    lmbda, p = sp.symbols("lmbda p")

    E11 = 0.5 * (lmbda**2 - 1.0)
    E22 = E33 = 0.5 * (1.0 / lmbda - 1.0)
    W = b_ff * E11**2 + b_xx * (E22**2 + E33**2)

    S11 = 0.5 * K * b_ff * (lmbda**2 - 1.0) * sp.exp(W) + p / (lmbda**2)
    S22 = 0.5 * K * b_xx * ((1 / lmbda) - 1.0) * sp.exp(W) + p * lmbda

    dS11_dlmbda = sp.diff(S11, lmbda)
    print(dS11_dlmbda)
    dS11_dp = sp.diff(S11, p)
    print(dS11_dp)
    dS22_dlmbda = sp.diff(S22, lmbda)
    print(dS22_dlmbda)
    dS22_dp = sp.diff(S22, p)
    print(dS22_dp)

    J = sp.Matrix([[dS11_dlmbda, dS11_dp], [dS22_dlmbda, dS22_dp]])
    # print(J)


def neohookean():
    mu, lmbda, p, Ta, lmbda_cross = sp.symbols("mu lmbda p Ta lmbda_cross")

    F11 = lmbda
    # F22 = F33 = 1.0 / sp.sqrt(lmbda)
    F22 = F33 = lmbda_cross

    C11 = F11**2
    C22 = C33 = F22**2
    J = F11 * F22 * F33

    I1 = C11 + C22 + C33
    I4f = F11**2

    psi = 0.5 * mu * (I1 - 3) + p * (J - 1) + Ta * (I4f - 1)

    P11 = sp.diff(psi, F11).subs({lmbda_cross: 1 / sp.sqrt(lmbda)})
    P22 = sp.diff(psi, F22).subs({lmbda_cross: 1 / sp.sqrt(lmbda)})

    # T11 = J * P11 / F11
    # T22 = J * P22 / F22

    print("P11 = ", P11)
    print("P22 = ", P22)

    dP11_dlmbda = sp.diff(P11, lmbda)
    print("dP11_dlmbda = ", dP11_dlmbda)

    dP22_dlmbda = sp.diff(P22, lmbda)
    print("dP22_dlmbda = ", dP22_dlmbda)

    dP11_dp = sp.diff(P11, p)
    print("dP11_dp = ", dP11_dp)

    dP22_dp = sp.diff(P22, p)
    print("dP22_dp = ", dP22_dp)
    breakpoint()


def neohookean_compressible():
    mu, lmbda, kappa, Ta, lmbda_cross = sp.symbols("mu lmbda kappa Ta lmbda_cross")

    F11 = lmbda
    # F22 = F33 = 1.0 / sp.sqrt(lmbda)
    F22 = F33 = lmbda_cross

    C11 = F11**2
    C22 = C33 = F22**2
    J = F11 * F22 * F33

    I1 = C11 + C22 + C33
    I4f = F11**2

    psi = 0.5 * mu * (I1 - 3) + kappa * (J - 1) ** 2 + Ta * (I4f - 1)

    P11 = sp.diff(psi, F11)
    P22 = sp.diff(psi, F22)

    # T11 = J * P11 / F11
    # T22 = J * P22 / F22

    # print("P11 = ", P11.subs({lmbda_cross: 1 / sp.sqrt(lmbda)}))
    print("P11 = ", P11)
    # print("P22 = ", P22.subs({lmbda_cross: 1 / sp.sqrt(lmbda)}))
    print("P22 = ", P22)

    dP11_dlmbda = sp.diff(P11, lmbda)
    print("dP11_dlmbda = ", dP11_dlmbda)
    # print("dP11_dlmbda = ", dP11_dlmbda.subs({lmbda_cross: 1 / sp.sqrt(lmbda)}))

    dP22_dlmbda = sp.diff(P22, lmbda)
    print("dP22_dlmbda = ", dP22_dlmbda)
    # print("dP22_dlmbda = ", dP22_dlmbda.subs({lmbda_cross: 1 / sp.sqrt(lmbda)}))


if __name__ == "__main__":
    neohookean()
    # neohookean_compressible()
