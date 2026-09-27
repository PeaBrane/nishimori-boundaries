"""(W) Warm certificate, p=4: F^0(beta_w,0) < log2 + beta_w^2/4 at the Nishimori point beta_w = 2 j0,
via the 1RSB trial mu = x delta_0 + (1-x) delta_q:
  Delta_w = (1/x) log E cosh^x(sqrt(L) Z) - L/2 + (1-x) theta(q)/2,  L = xi'(q) = 2 b^2 q^3,
  theta(q) = 3 b^2 q^4 / 2.
Float search: nested bounded Brent, x = 1 - exp(-t) (the optimum sits at 1-x ~ 2.5e-4)."""
import json
import os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize_scalar

from common4 import BETA_W, J0, ivr, dxi, theta, maxrss_mb
from gauss import expect
from iv import IV, ilog, isqrt, tcosh, tpow_pos
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

bf = float(BETA_W)
Zt = np.linspace(-18.0, 18.0, 2881)
Wt = (Zt[1] - Zt[0]) * np.exp(-Zt * Zt / 2) / np.sqrt(2 * np.pi)


def lcf(y):
    return np.abs(y) + np.log1p(np.exp(-2 * np.abs(y))) - np.log(2)


def delta_float(x, q):
    L = 2 * bf * bf * q ** 3
    v = x * lcf(np.sqrt(L) * Zt)
    mx = v.max()
    return (mx + np.log(np.sum(Wt * np.exp(v - mx)))) / x - L / 2 + (1 - x) * 1.5 * bf * bf * q ** 4 / 2


def inner(q):
    r = minimize_scalar(lambda t: delta_float(1 - np.exp(-t), q), bounds=(0.5, 20), method='bounded',
                        options={'xatol': 1e-13})
    return r.fun, 1 - np.exp(-r.x)


if __name__ == "__main__":
    qs = np.linspace(0.90, 0.99, 91)
    k = int(np.argmin([inner(q)[0] for q in qs]))
    r = minimize_scalar(lambda q: inner(q)[0], bounds=(qs[k - 1], qs[k + 1]), method='bounded',
                        options={'xatol': 1e-13})
    qf = r.x
    dfl, xf = inner(qf)
    print(f"float optimum x={xf:.10f} q={qf:.10f} Delta={dfl:.6e}", flush=True)
    den = 10 ** 7
    x = Fr(round(xf * den), den)
    q = Fr(round(qf * den), den)
    print("rational trial x,q =", x, q, "float Delta", delta_float(float(x), float(q)), flush=True)
    B = ivr(BETA_W)
    X, Q = ivr(x), ivr(q)
    Lam = dxi(B, Q)
    s = isqrt(Lam)
    xs = float(X.hi) * float(s.hi)
    out = {}
    for n in [4000, 12000]:
        A = expect(lambda z: tpow_pos(tcosh(z * s), X), 1.0, xs, n=n)
        D = ilog(A) * X.recip() - Lam * 0.5 + (1 - X) * theta(B, Q) * 0.5
        print(f"n={n} A={A} Delta_w in [{D.lo}, {D.hi}] width={D.hi - D.lo:.2e} certified<0: {bool(D.hi < 0)}",
              flush=True)
        out = {"p": 4, "j0": str(J0), "beta_w": str(BETA_W), "x": str(x), "q": str(q), "n": n,
               "Delta_w_lo": float(D.lo), "Delta_w_hi": float(D.hi), "certified": bool(D.hi < 0),
               "m1": f"(2 beta_w j0)^(-1/2) = 1/(2 j0) = {1 / (2 * float(J0)):.10f}"}
    json.dump(out, open(os.path.join(HERE, 'warm4_result.json'), 'w'), indent=1)
    print(f"peak RSS {maxrss_mb():.0f} MB")
