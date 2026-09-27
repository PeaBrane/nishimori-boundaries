"""Float spot check (not a certificate) of the p=4 landscape cover: for every cover interval, recompute
U(m) = beta j0 m^4 + P_tau(h) - h m at both endpoints from the Cole-Hopf recursion (separate code) and compare
with the primary certificate's ("builder") certified U_hi; check contiguity, parameter ranges, h >= 0; also scan U on a fine m grid
inside each interval to confirm endpoint maxima (convexity), and check the warm Delta_w independently."""
import os
import json
import math
from fractions import Fraction as Fr

import numpy as np
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

P4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
H = 0.01
Z = np.arange(-2000, 2001) * H
W = H * np.exp(-Z * Z / 2) / math.sqrt(2 * math.pi)
LOGW = np.log(W)


def lc(y):
    a = np.abs(y)
    return a + np.log1p(np.exp(-2 * a)) - math.log(2)


def P_tau(b, kind, t):
    xi = lambda s: b * b * s ** 4 / 2
    dxi = lambda s: 2 * b * b * s ** 3
    th = lambda s: s * dxi(s) - xi(s)
    if kind == 'RS':
        h, q = t
        # mu = delta_q : CDF 0 on [0,q), 1 on [q,1]
        phi0 = float(W @ lc(h + math.sqrt(dxi(q)) * Z)) + (dxi(1) - dxi(q)) / 2
        return math.log(2) + phi0 - 0.5 * (th(1) - th(q))
    h, x, q = t
    v = x * lc(h + math.sqrt(dxi(q)) * Z) + LOGW
    mx = v.max()
    phi0 = (mx + math.log(np.exp(v - mx).sum())) / x + (dxi(1) - dxi(q)) / 2
    return math.log(2) + phi0 - 0.5 * (x * th(q) + th(1) - th(q))


if __name__ == "__main__":
    cov = json.load(open(P4 + 'landscape4_cover.json'))
    b, j0 = float(Fr(cov['beta'])), float(Fr(cov['j0']))
    prev = Fr(2, 5)
    worst_f, worst_scan, maxdev = -1e9, -1e9, 0.0
    for c in cov['cover']:
        ma, mb = Fr(c['ma']), Fr(c['mb'])
        assert ma == prev and mb > ma
        prev = mb
        t = [float(Fr(v)) for v in c['t']]
        assert t[0] >= 0, "negative field"
        if c['kind'] == '1R':
            assert 0 < t[1] <= 1 and 0 < t[2] < 1
        else:
            assert 0 < t[1] < 1
        P = P_tau(b, c['kind'], t)
        assert c['P'][0] - 1e-11 <= P <= c['P'][1] + 1e-11, (c, P)
        U = lambda m: b * j0 * m ** 4 + P - t[0] * m
        ue = max(U(float(ma)), U(float(mb)))
        us = max(U(m) for m in np.linspace(float(ma), float(mb), 201))
        maxdev = max(maxdev, ue - c['U_hi'])
        worst_f = max(worst_f, ue)
        worst_scan = max(worst_scan, us)
    assert prev == 1
    print(f"intervals {len(cov['cover'])}: contiguous on [2/5,1], all h>=0, P inside enclosures")
    print(f"float worst endpoint U = {worst_f!r}; interior-scan worst = {worst_scan!r}; "
          f"max(U_float - U_hi) = {maxdev:.2e}; certified worst U_hi = {cov['worst_U_hi']!r}")
    # universal region / threshold arithmetic
    bc = 10 / 3
    print("(2 beta_c j0)^(-1/2) =", (2 * bc * j0) ** -0.5, " g(2/5) =", bc * j0 * 0.4 ** 4 - 0.08,
          " argmin g =", (4 * bc * j0) ** -0.5)
    bw = 2 * j0
    print("m1 = (2 beta_w j0)^(-1/2) =", (2 * bw * j0) ** -0.5, " 1/(2 j0) =", 1 / (2 * j0))
    # warm Delta_w (independent float, trapezoid h=0.01)
    w = json.load(open(P4 + 'warm4_result.json'))
    x, q = float(Fr(w['x'])), float(Fr(w['q']))
    Pw = P_tau(bw, '1R', [0.0, x, q])
    print(f"warm P(mu,0) - (log2 + bw^2/4) = {Pw - (math.log(2) + bw * bw / 4):.6e}; "
          f"builder enclosure [{w['Delta_w_lo']:.6e}, {w['Delta_w_hi']:.6e}]")
    # optimize the RS value near the worst interval to see the peak of the FM landscape (float)
