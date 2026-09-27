"""Non-rigorous cross-checks of the closed-form D(u) (independent code path):
generic Cole-Hopf evaluation of P(mu_eps) for the 3-atom measure mu_eps=(1-eps)mu*+eps d_u,
central finite differences in eps, and D'(u) = -xi''(u)(Gamma(u)-u)/2 via FD of rigorous D."""
import numpy as np
from fractions import Fraction as Fr
import fgrid
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

B2 = 6.25; X = 554371e-6; Q = 929223e-6
v = lambda s: 1.5 * B2 * s * s
th = lambda s: B2 * s ** 3
g = np.arange(-14, 14 + 1e-12, 0.04); w = 0.04 * np.exp(-g * g / 2) / np.sqrt(2 * np.pi)
lc = lambda z: np.abs(z) + np.log1p(np.exp(-2 * np.abs(z))) - np.log(2)


def parisi(levels, ms):
    """levels: 0=q0<q1<..<qk<=1 (list, len k+1), ms: m_j on [q_j,q_{j+1}) with q_{k+1}=1."""
    qs = list(levels) + [1.0]
    # Phi at level q_k as function of y, built recursively via callables on arrays
    def Phi(j, y):  # value at level qs[j], y array
        if j == len(qs) - 1:
            return lc(y)
        if j == len(qs) - 2 and ms[j] == 1.0:  # m=1 on [q_k,1]: closed form
            return lc(y) + (v(1.0) - v(qs[j])) / 2
        s = np.sqrt(v(qs[j + 1]) - v(qs[j]))
        Z = y[..., None] + s * g
        inner = Phi(j + 1, Z)
        m = ms[j]
        if m == 0:
            return (inner * w).sum(-1)
        mx = inner.max(-1, keepdims=True)
        return (np.log((np.exp(m * (inner - mx)) * w).sum(-1)) + m * mx[..., 0]) / m
    phi0 = Phi(0, np.array([0.0]))[0]
    pen = 0.5 * sum(ms[j] * (th(qs[j + 1]) - th(qs[j])) for j in range(len(ms)))
    return np.log(2) + phi0 - pen


def P_eps(u, eps):
    if u < Q:
        lv = [0.0, u, Q]; ms = [(1 - eps) * X, (1 - eps) * X + eps, 1.0]
    elif u > Q:
        lv = [0.0, Q, u]; ms = [(1 - eps) * X, 1 - eps, 1.0]
    return parisi(lv, ms)


if __name__ == "__main__":
    print("P(mu*) nonrig:", parisi([0.0, Q], [X, 1.0]))
    for u in [Fr(1, 10), Fr(8, 75), Fr(38, 75), Fr(9, 10), Fr(928, 1000), Fr(935, 1000), Fr(97, 100)]:
        uf = float(u); e = 1e-4
        fd = (P_eps(uf, e) - P_eps(uf, -e)) / (2 * e)
        r = fgrid.evaluate(u)
        print(f"u={uf:.6f}  FD-CH D={fd:.10e}   rigorous D=[{r['D'][0]:.10e},{r['D'][1]:.10e}]  diff={fd-r['D'][0]:.2e}")
    # D' check
    for u, hstep in [(Fr(3, 10), Fr(1, 10000)), (Fr(95, 100), Fr(1, 10000)), (Fr(92, 100), Fr(1, 10000))]:
        rp = fgrid.evaluate(u + hstep); rm = fgrid.evaluate(u - hstep); r0 = fgrid.evaluate(u)
        dD = (rp['D'][0] - rm['D'][0]) / (2 * float(hstep))
        pred = -0.5 * 3 * B2 * float(u) * (r0['G'][0] - float(u))
        print(f"u={float(u)}: FD D'={dD:.10e}  -xi''(Gamma-u)/2={pred:.10e}")
