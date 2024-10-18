import sympy as sp

K, b_ff, b_xx = sp.symbols("K b_ff b_xx")
lmbda, p = sp.symbols("lmbda p")

E11 = 0.5 * (lmbda**2 - 1.0)
E22 = E33 = 0.5 * (1.0 / lmbda - 1.0)
W = b_ff * E11**2 + b_xx * (E22**2 + E33**2)

S11 = 0.5 * K * b_ff * (lmbda ** 2 - 1.0) * sp.exp(W) + p / (lmbda ** 2)
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
