"""Non-rigorous cross-checks (not certificate inputs), independent code path:
(1) generic Cole-Hopf evaluation of the finite-level Parisi functional P(mu_eps), mu_eps = (1-eps) mu* + eps d_u,
    Richardson-extrapolated central differences in eps, compared with the rigorous closed-form D(u);
(2) D'(u) = -(1/2) xi''(u) (Gamma(u) - u) via centred differences of the rigorous D;
(3) exact side conditions: xi'(21/100) <= 21/100 at beta = 10/3 and g(2/5) = beta j0 (2/5)^4 - (2/5)^2/2 < 0.
Peak memory: arrays of 701 x 701 doubles (~4 MB each)."""
import json
import os
import resource
from fractions import Fraction as Fr

import numpy as np

import fgrid4
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
BF = 10 / 3
B2 = BF * BF
X = 930529 / 2000000
Q = 4930831 / 5000000
v = lambda s: 2 * B2 * s ** 3
th = lambda s: 1.5 * B2 * s ** 4
g = np.arange(-14, 14 + 1e-12, 0.04)
w = 0.04 * np.exp(-g * g / 2) / np.sqrt(2 * np.pi)
lc = lambda z: np.abs(z) + np.log1p(np.exp(-2 * np.abs(z))) - np.log(2)


def parisi(levels, ms):
    """levels 0=q0<=q1<..<=qk (list), ms[j] = mu([0,s]) on [q_j, q_{j+1}), q_{k+1} = 1."""
    qs = list(levels) + [1.0]

    def Phi(j, y):
        if j == len(qs) - 1:
            return lc(y)
        if j == len(qs) - 2 and ms[j] == 1.0:
            return lc(y) + (v(1.0) - v(qs[j])) / 2
        s = np.sqrt(v(qs[j + 1]) - v(qs[j]))
        inner = Phi(j + 1, y[..., None] + s * g)
        m = ms[j]
        mx = inner.max(-1, keepdims=True)
        return (np.log((np.exp(m * (inner - mx)) * w).sum(-1)) + m * mx[..., 0]) / m

    phi0 = Phi(0, np.array([0.0]))[0]
    pen = 0.5 * sum(ms[j] * (th(qs[j + 1]) - th(qs[j])) for j in range(len(ms)))
    return np.log(2) + phi0 - pen


def P_eps(u, eps):
    if u < Q:
        return parisi([0.0, u, Q], [(1 - eps) * X, (1 - eps) * X + eps, 1.0])
    return parisi([0.0, Q, u], [(1 - eps) * X, 1 - eps, 1.0])


def fd(u, e):
    return (P_eps(u, e) - P_eps(u, -e)) / (2 * e)


if __name__ == "__main__":
    out = {"P_mu_star_float": parisi([0.0, Q], [X, 1.0])}
    print("P(mu*) float:", out["P_mu_star_float"])
    rows = []
    for u in [Fr(1, 10), Fr(21, 100), Fr(1, 2), Fr(9, 10), Fr(98, 100), Fr(986, 1000), Fr(9865, 10000),
              Fr(99, 100), Fr(1)]:
        uf = float(u)
        d1, d2 = fd(uf, 2e-3), fd(uf, 1e-3)
        rich = (4 * d2 - d1) / 3
        r = fgrid4.evaluate(u)
        rows.append(dict(u=str(u), fd=rich, D=r["D"], diff=rich - 0.5 * (r["D"][0] + r["D"][1])))
        print(f"u={uf:.6f} FD={rich:.10e} rigorous D=[{r['D'][0]:.10e},{r['D'][1]:.10e}] diff={rows[-1]['diff']:.1e}",
              flush=True)
    out["fd_rows"] = rows
    # dP/dx at x (two-atom functional) -> D(0) = (1-x) dP/dx
    e = 1e-3
    c1 = (parisi([0.0, Q], [X + e, 1.0]) - parisi([0.0, Q], [X - e, 1.0])) / (2 * e)
    c2 = (parisi([0.0, Q], [X + e / 2, 1.0]) - parisi([0.0, Q], [X - e / 2, 1.0])) / e
    dPdx = (4 * c2 - c1) / 3
    out["dPdx_fd"] = dPdx
    print("dP/dx FD:", dPdx, " D(0)=(1-x)dP/dx:", (1 - X) * dPdx)
    drows = []
    for u, hs in [(Fr(3, 10), Fr(1, 10000)), (Fr(7, 10), Fr(1, 10000)), (Fr(97, 100), Fr(1, 10000)),
                  (Fr(995, 1000), Fr(1, 10000))]:
        rp, rm, r0 = fgrid4.evaluate(u + hs), fgrid4.evaluate(u - hs), fgrid4.evaluate(u)
        dD = (rp["D"][0] - rm["D"][0]) / (2 * float(hs))
        pred = -0.5 * 6 * B2 * float(u) ** 2 * (r0["G"][0] - float(u))
        drows.append(dict(u=str(u), fd=dD, pred=pred))
        print(f"u={float(u)}: FD D'={dD:.10e}  -xi''(Gamma-u)/2={pred:.10e}", flush=True)
    out["dD_rows"] = drows
    beta, j0 = Fr(10, 3), Fr(8107, 10000)
    U0 = Fr(21, 100)
    out["xi_prime_U0_le_U0"] = 2 * beta ** 2 * U0 ** 3 <= U0
    out["xi_prime_U0"] = float(2 * beta ** 2 * U0 ** 3)
    gMA = beta * j0 * Fr(2, 5) ** 4 - Fr(2, 5) ** 2 / 2
    out["g_MA"] = [str(gMA), float(gMA)]
    out["MA_below_root"] = Fr(2, 5) ** 2 < 1 / (2 * beta * j0)
    out["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print({k: out[k] for k in ("xi_prime_U0_le_U0", "xi_prime_U0", "g_MA", "MA_below_root", "maxrss_MB")})
    json.dump(out, open(os.path.join(HERE, "sanity4_result.json"), "w"), indent=1)
