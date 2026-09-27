"""Float spot check (not a certificate): D(u) by Richardson-extrapolated finite differences of the
3-atom Parisi functional P((1-eps) mu* + eps delta_u), computed directly from the Cole-Hopf recursion
(no closed forms), and Gamma(u) from a separate derivation. Compared with the primary enclosures ("builder").
Memory: 1-D grids of ~1600 nodes, outer loop chunked by 200 rows (~2.5 MB per temporary)."""
import os
import json
import math
import resource
import sys
from fractions import Fraction as Fr

import numpy as np

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

P4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') + os.sep
b = 10.0 / 3.0
x = 930529 / 2000000
q = 4930831 / 5000000
xi = lambda t: b * b * t ** 4 / 2
dxi = lambda t: 2 * b * b * t ** 3
th = lambda t: t * dxi(t) - xi(t)   # computed from definition, not the 3/2 closed form

H = 0.02
Z = np.arange(-800, 801) * H          # [-16,16]
W = H * np.exp(-Z * Z / 2) / math.sqrt(2 * math.pi)
LOGW = np.log(W)


def lc(y):
    a = np.abs(y)
    return a + np.log1p(np.exp(-2 * a)) - math.log(2)


def lse_rows(v):
    """log sum_k W_k exp(v[:,k])"""
    v = v + LOGW[None, :]
    m = v.max(axis=1)
    return m + np.log(np.exp(v - m[:, None]).sum(axis=1))


def phi_top(y, m, s, top_const):
    """(1/m) log E_g exp(m lc(y + s g)) + top_const for a 1-D array y"""
    out = np.empty_like(y)
    for i in range(0, len(y), 200):
        yy = y[i:i + 200, None]
        out[i:i + 200] = lse_rows(m * lc(yy + s * Z[None, :])) / m
    return out + top_const


def outer(vals, m):
    v = m * vals + LOGW
    mx = v.max()
    return (mx + math.log(np.exp(v - mx).sum())) / m


def P_eps(u, eps, h=0.0):
    """Parisi functional of (1-eps) mu* + eps delta_u with field h (mu* = x delta_0 + (1-x) delta_q)"""
    if u == 0.0:
        m = x + eps * (1 - x)       # CDF on [0,q)
        s = math.sqrt(dxi(q))
        vals = lc(h + s * Z) + (dxi(1) - dxi(q)) / 2
        return math.log(2) + outer(vals, m) - 0.5 * (m * th(q) + th(1) - th(q))
    if u < q:
        m1, m2 = (1 - eps) * x, x + eps * (1 - x)
        s1, s2 = math.sqrt(dxi(u)), math.sqrt(dxi(q) - dxi(u))
        vals = phi_top(h + s1 * Z, m2, s2, (dxi(1) - dxi(q)) / 2)
        integ = 0.5 * (m1 * th(u) + m2 * (th(q) - th(u)) + th(1) - th(q))
        return math.log(2) + outer(vals, m1) - integ
    m1, m2 = (1 - eps) * x, 1 - eps
    s1, s = math.sqrt(dxi(q)), math.sqrt(dxi(u) - dxi(q))
    vals = phi_top(h + s1 * Z, m2, s, (dxi(1) - dxi(u)) / 2)
    integ = 0.5 * (m1 * th(q) + m2 * (th(u) - th(q)) + th(1) - th(u))
    return math.log(2) + outer(vals, m1) - integ


def D_fd(u, e=1e-3):
    d1 = (P_eps(u, e) - P_eps(u, -e)) / (2 * e)
    d2 = (P_eps(u, 2 * e) - P_eps(u, -2 * e)) / (4 * e)
    return (4 * d1 - d2) / 3


def Gamma_mine(u):
    """Gamma(u) = E_y[(d_y Phi_u)^2 * tilt] from a separate derivation: E over the mu*-tilted path of Phi_x(u,.)^2"""
    s_q = math.sqrt(dxi(q))
    A = float(np.sum(W * np.exp(x * lc(s_q * Z))))
    if u < q:
        s1, s2 = math.sqrt(dxi(u)), math.sqrt(dxi(q) - dxi(u))
        acc = 0.0
        y = s1 * Z
        for i in range(0, len(y), 200):
            Y = y[i:i + 200, None] + s2 * Z[None, :]
            c = np.exp(x * lc(Y))
            I0 = c @ W
            I1 = (c * np.tanh(Y)) @ W
            acc += float(W[i:i + 200] @ (I1 * I1 / I0))
        return acc / A
    s = math.sqrt(dxi(u) - dxi(q))
    y = s_q * Z
    acc = 0.0
    for i in range(0, len(y), 200):
        yy = y[i:i + 200, None]
        Y = yy + s * Z[None, :]
        J1 = (np.exp(lc(Y)) * np.tanh(Y) ** 2) @ W
        pre = np.exp((x - 1) * lc(yy[:, 0]))
        acc += float(W[i:i + 200] @ (pre * J1))
    return acc / (A * math.exp(s * s / 2))


if __name__ == "__main__":
    pts = json.load(open(P4 + 'fw4_points.json'))
    P0 = P_eps(0.3, 0.0)
    P0b = P_eps(1.0, 0.0)
    print(f"P(mu*) recursion = {P0!r} (u>q path {P0b!r}); builder enclosure [2.754581581382262, 2.754581581383308]")
    D0 = D_fd(0.0)
    print(f"D(0) FD = {D0:.6e}; builder enclosure [-1.43816e-07, -1.43805e-07]")
    us = [Fr(21, 100), Fr(31, 100), Fr(61, 100), Fr(97, 100), Fr(4930831, 5000000), Fr(39365817, 40000000), Fr(15762493, 16000000),
          Fr(19742493, 20000000), Fr(99, 100), Fr(1)]
    for u in us:
        e = pts[str(u)]
        uf = float(u)
        d = D_fd(uf)
        g = Gamma_mine(uf)
        print(f"u={uf:.8f} D_fd={d:.12e} encl=[{e[0]:.12e},{e[1]:.12e}] dev={max(e[0]-d, d-e[1], 0):.1e} | "
              f"Gamma={g:.12f} encl=[{e[2]:.12f},{e[3]:.12f}] dev={max(e[2]-g, g-e[3], 0):.1e}", flush=True)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"peak RSS {r:.0f} MB")
