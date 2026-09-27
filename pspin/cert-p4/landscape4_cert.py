"""(L) Cold landscape cover, p=4, beta = 10/3, j0 = 8107/10000.
For every interval [ma, mb] of a cover of [M_A, 1] a trial tau (rational parameters) with
  U(m) = beta j0 m^4 + P_tau(h) - h m     (convex in m: 12 beta j0 m^2 >= 0)
satisfies U(ma), U(mb) < TARGET, hence U < TARGET on [ma, mb].
Trials: RS delta_q with field h; 1RSB x delta_0 + (1-x) delta_q with field h.
  P_RS = log2 + E logcosh(h + sqrt(xi'(q)) Z) + (xi(1) - xi(q) - (1-q) xi'(q))/2
  P_1R = log2 + (1/x) log E cosh^x(h + sqrt(xi'(q)) Z) + (xi(1) - xi'(q) + (1-x) theta(q))/2
Negative m: for even p, H^0(-sigma) = H^0(sigma) and the bias is even, so Z_{[-b,-a]} = Z_{[a,b]} exactly."""
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize

from common4 import BETA_C, J0, M_A, ivr, xi, dxi, theta, maxrss_mb
from gauss import expect
from iv import LOG2, ilog, isqrt, tcosh, tlogcosh, tpow_pos

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

beta = BETA_C
B = ivr(beta)
JJ = ivr(J0)
bf, jf = float(beta), float(J0)
Zt = np.linspace(-18.0, 18.0, 1441)
Wt = (Zt[1] - Zt[0]) * np.exp(-Zt * Zt / 2) / np.sqrt(2 * np.pi)


def lcf(y):
    return np.abs(y) + np.log1p(np.exp(-2 * np.abs(y))) - np.log(2)


def P_rs_f(q, h):
    L = 2 * bf * bf * q ** 3
    return np.log(2) + np.sum(Wt * lcf(h + np.sqrt(L) * Zt)) + 0.5 * (bf * bf / 2 - bf * bf * q ** 4 / 2 - (1 - q) * L)


def P_1r_f(x, q, h):
    L = 2 * bf * bf * q ** 3
    v = x * lcf(h + np.sqrt(L) * Zt)
    mx = v.max()
    return np.log(2) + (mx + np.log(np.sum(Wt * np.exp(v - mx)))) / x + 0.5 * (
        bf * bf / 2 - L + (1 - x) * 1.5 * bf * bf * q ** 4)


def U_f(m, kind, t):
    if kind == 'RS':
        return bf * jf * m ** 4 + P_rs_f(t[1], t[0]) - t[0] * m
    return bf * jf * m ** 4 + P_1r_f(t[1], t[2], t[0]) - t[0] * m


BOUNDS = {'RS': [(0, 40), (0.05, 0.9999)], '1R': [(0, 40), (0.02, 1), (0.05, 0.9999)]}


def best_trial(m, warm=None):
    starts = [('RS', [h0, q0]) for h0 in np.linspace(0, 12, 7) for q0 in [0.6, 0.9, 0.98]]
    starts += [('1R', [h0, x0, q0]) for h0 in np.linspace(0, 12, 7) for q0 in [0.6, 0.9, 0.98] for x0 in [0.3, 0.6]]
    if warm is not None:
        starts = [warm] + starts
    best = None
    for kind, t0 in starts:
        r = minimize(lambda t: U_f(m, kind, t), t0, method='L-BFGS-B', bounds=BOUNDS[kind])
        if best is None or r.fun < best[0]:
            best = (r.fun, kind, r.x)
    return best


def rat(v, d=10 ** 6):
    return Fr(round(float(v) * d), d)


def P_iv(kind, t):
    """rigorous enclosure of P_tau(h) for rational parameters t"""
    h = ivr(t[0])
    if kind == 'RS':
        q = ivr(t[1])
        L = dxi(B, q)
        s = isqrt(L)
        E = expect(lambda z: tlogcosh(z * s + h), float(h.hi) + float(s.hi), 1.0, n=3000)
        return LOG2 + E + (xi(B, 1.0) - xi(B, q) - (1 - q) * L) * 0.5
    x, q = ivr(t[1]), ivr(t[2])
    L = dxi(B, q)
    s = isqrt(L)
    xs = float(x.hi) * float(s.hi)
    E = expect(lambda z: tpow_pos(tcosh(z * s + h), x), float(np.exp(float(x.hi) * float(h.hi))), xs, n=3000)
    return LOG2 + ilog(E) * x.recip() + (xi(B, 1.0) - L + (1 - x) * theta(B, q)) * 0.5


def U_iv(m, kind, t, P):
    M = ivr(m)
    h = ivr(t[0])
    return B * JJ * M * M * M * M + P - h * M


if __name__ == "__main__":
    TARGET = float(sys.argv[1])
    ma0, mb0 = M_A, Fr(1)
    nint = 60
    todo = [(ma0 + (mb0 - ma0) * Fr(k, nint), ma0 + (mb0 - ma0) * Fr(k + 1, nint)) for k in range(nint)]
    cover = []
    worst = -1e9
    warm = None
    while todo:
        ma, mb = todo.pop(0)
        mid = (ma + mb) / 2
        fval, kind, t = best_trial(float(mid), warm)
        warm = (kind, list(t))
        t = [rat(v) for v in t]
        P = P_iv(kind, t)
        ua, ub = U_iv(ma, kind, t, P), U_iv(mb, kind, t, P)
        top = max(float(ua.hi), float(ub.hi))
        if top < TARGET:
            cover.append({"ma": str(ma), "mb": str(mb), "kind": kind, "t": [str(v) for v in t],
                          "U_hi": top, "P": [float(P.lo), float(P.hi)]})
            worst = max(worst, top)
            print(f"ok [{float(ma):.5f},{float(mb):.5f}] {kind} U_hi-TARGET={top - TARGET:+.3e}", flush=True)
        else:
            if mb - ma < Fr(1, 10 ** 5):
                print("FAIL at", float(ma), float(mb), top - TARGET)
                sys.exit(1)
            todo = [(ma, mid), (mid, mb)] + todo
    print("cover intervals:", len(cover), "worst U_hi:", worst, "TARGET:", TARGET, "slack", TARGET - worst)
    json.dump({"p": 4, "beta": str(beta), "j0": str(J0), "m_range": [str(M_A), "1"], "TARGET": TARGET,
               "worst_U_hi": worst, "cover": cover}, open(os.path.join(HERE, 'landscape4_cover.json'), 'w'), indent=1)
    print(f"peak RSS {maxrss_mb():.0f} MB")
