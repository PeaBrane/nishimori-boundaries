"""Validation of the trapezoid engine (not itself a certificate input):
 (1) 1-D and nested enclosures contain high-precision mpmath references;
 (2) the analytic strip majorants used in sg_lb4t.py dominate the complex integrands on Im z = a
     (sampled in complex floating point, with the inner Gaussian integrals done by a fine trapezoid sum);
 (3) enclosed D' consistency: centred differences of D vs -xi''(u)(Gamma(u)-u)/2."""
import math
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np

import sg_lb4t as S
from common4 import maxrss_mb
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.mp.dps = 25
x, q, b = mp.mpf(S.xr.numerator) / S.xr.denominator, mp.mpf(S.qr.numerator) / S.qr.denominator, mp.mpf(10) / 3
dxi = lambda u: 2 * b * b * u ** 3
sq = mp.sqrt(dxi(q))
lc = lambda y: mp.log(mp.cosh(y))
ok = True

# (1a) 1-D references
Aref = mp.quad(lambda z: mp.cosh(sq * z) ** x * mp.npdf(z), mp.linspace(-20, 20, 41))
AKref = mp.quad(lambda z: mp.cosh(sq * z) ** x * lc(sq * z) * mp.npdf(z), mp.linspace(-20, 20, 41))
for name, iv, ref in [("A", S.A, Aref), ("AK", S.AK, AKref)]:
    inside = float(iv.lo) <= float(ref) <= float(iv.hi)
    ok &= inside
    print(f"(1a) {name}: [{float(iv.lo)!r}, {float(iv.hi)!r}] ref {mp.nstr(ref, 20)} inside={inside}", flush=True)


# (1b) nested references at u<q and u>q by an independent float64 method: Gauss-Legendre (20 nodes) on
# panels of width 1/4 over [-16,16] in both variables (panel half-width 1/8 vs singularity distance
# >= pi/(2 s) ~ 0.34: geometric convergence far below float64 rounding). Tolerance: 1e-12 relative.
gx, gw = np.polynomial.legendre.leggauss(20)
edges = np.arange(-16.0, 16.0 + 1e-12, 0.25)
GZ = ((edges[:-1, None] + edges[1:, None]) / 2 + 0.125 * gx[None, :]).ravel()
GW = (0.125 * np.tile(gw, len(edges) - 1)) * np.exp(-GZ ** 2 / 2) / math.sqrt(2 * math.pi)
lcf = lambda y: np.abs(y) + np.log1p(np.exp(-2 * np.abs(y))) - math.log(2)


def nested_ref(u):
    uf, xq, qq, bb = float(u), float(x), float(q), float(b)
    sqf = math.sqrt(2 * bb * bb * qq ** 3)
    acc = np.zeros(2)
    for i0 in range(0, len(GZ), 256):
        z1, w1 = GZ[i0:i0 + 256, None], GW[i0:i0 + 256]
        if uf < qq:
            s1, s2 = math.sqrt(2 * bb * bb * uf ** 3), math.sqrt(2 * bb * bb * (qq ** 3 - uf ** 3))
            Y = s1 * z1 + s2 * GZ[None, :]
            c = np.exp(xq * lcf(Y))
            I0 = c @ GW
            I1 = (c * np.tanh(Y)) @ GW
            acc += [w1 @ (I0 * np.log(I0)), w1 @ (I1 * I1 / I0)]
        else:
            sg = math.sqrt(2 * bb * bb * (uf ** 3 - qq ** 3))
            y = sqf * z1
            Y = y + sg * GZ[None, :]
            chY = np.exp(lcf(Y)) 
            J0 = (chY * lcf(Y)) @ GW
            J1 = (chY * np.tanh(Y) ** 2) @ GW
            pre = np.exp((xq - 1) * lcf(y[:, 0]))
            acc += [w1 @ (pre * J0), w1 @ (pre * J1)]
    return acc


for u in [Fr(3, 5), Fr(97, 100), Fr(99, 100), Fr(1)]:
    D, G, _ = S.D_and_Gamma(u)
    r1, r2 = nested_ref(u)
    uu, A = float(u), float(Aref)
    K = float(AKref) / A
    xq, qq, bb = float(x), float(q), float(b)
    th = lambda v: 1.5 * bb * bb * v ** 4
    if uu < qq:
        Dref = math.log(A) / xq + (1 - xq) * K / xq - r1 / (xq * xq * A) + (th(uu) - (1 - xq) * th(qq)) / 2
        Gref = r2 / A
    else:
        s2 = 2 * bb * bb * (uu ** 3 - qq ** 3)
        Dref = math.log(A) / xq + s2 / 2 - r1 / (A * math.exp(s2 / 2)) + (xq * th(qq) + th(uu) - th(qq)) / 2
        Gref = r2 / (A * math.exp(s2 / 2))
    tolD, tolG = 1e-12 * 30, 1e-12
    i1 = float(D.lo) - tolD <= Dref <= float(D.hi) + tolD
    i2 = float(G.lo) - tolG <= Gref <= float(G.hi) + tolG
    ok &= i1 and i2
    print(f"(1b) u={uu} D=[{float(D.lo)!r},{float(D.hi)!r}] GLref {Dref!r} in={i1};"
          f" Gamma=[{float(G.lo)!r},{float(G.hi)!r}] GLref {Gref!r} in={i2}", flush=True)

# (2) strip majorants, sampled in complex floating point
xf, qf = float(x), float(q)
tt = np.linspace(-14, 14, 4481)
wt = (tt[1] - tt[0]) * np.exp(-tt * tt / 2) / math.sqrt(2 * math.pi)
ch = lambda Y: np.cosh(Y)
cpow = lambda Y, e: np.exp(e * np.log(np.cosh(Y)))                    # principal branch
worst = 0.0
for uf in [0.25, 0.6, 0.9, 0.98, 0.985]:
    s1 = math.sqrt(2 * float(b) ** 2 * uf ** 3)
    s2 = math.sqrt(2 * float(b) ** 2 * (qf ** 3 - uf ** 3))
    a1 = min(math.pi / (4 * s1), 2.0)
    for z in np.linspace(-6, 6, 25):
        y = s1 * complex(z, a1)
        Y = y + s2 * tt
        I0 = np.sum(wt * cpow(Y, xf))
        I1 = np.sum(wt * cpow(Y, xf) * np.tanh(Y))
        c = abs(y.real)
        I0c = 2 * math.exp(xf * c + xf * xf * s2 * s2 / 2)
        bS = I0c * (math.log(2) + xf * xf * s2 * s2 / 2 + xf * c + S.LOGK + math.pi / 2)
        bG = I0c / S.KAPPA
        rS = abs(I0 * np.log(I0)) / bS
        rG = abs(I1 ** 2 / I0) / bG
        rk = (I0.real / np.sum(wt * np.cosh(y.real + s2 * tt) ** xf)) / S.KAPPA     # must be >= 1
        worst = max(worst, rS, rG, 1 / rk)
for uf in [0.987, 0.99, 1.0]:
    sgm = math.sqrt(2 * float(b) ** 2 * (uf ** 3 - qf ** 3))
    sqq = math.sqrt(2 * float(b) ** 2 * qf ** 3)
    a1 = min(math.pi / (4 * sqq), 2.0)
    for z in np.linspace(-6, 6, 25):
        y = sqq * complex(z, a1)
        Y = y + sgm * tt
        J0 = np.sum(wt * np.cosh(Y) * np.log(np.cosh(Y)))
        J1 = np.sum(wt * np.cosh(Y) * np.tanh(Y) ** 2)
        c = abs(y.real)
        pre = abs(cpow(y, xf - 1))
        bT = 2 ** (1.5 * (1 - xf) + 1) * math.exp(sgm ** 2 / 2 + xf * c) * (c + 1.2 + sgm * (0.4 + sgm))
        bG = 2 ** (1.5 * (1 - xf)) * math.exp(sgm ** 2 / 2 + xf * c)
        worst = max(worst, pre * abs(J0) / bT, pre * abs(J1) / bG)
    # inner strip majorants at a sample of real y
    a2 = min(math.pi / (4 * max(sgm, 1e-9)), 2.0)
    for yv in [0.0, 1.0, 5.0, 20.0]:
        Y = yv + sgm * (tt + 1j * a2)
        r0 = np.abs(np.cosh(Y) * np.log(np.cosh(Y))) / (np.exp(yv + sgm * np.abs(tt)) * (yv + 1.2 + sgm * np.abs(tt)))
        r1 = np.abs(np.cosh(Y) * np.tanh(Y) ** 2) / np.exp(yv + sgm * np.abs(tt))
        worst = max(worst, r0.max(), r1.max())
print(f"(2) max ratio |integrand| / majorant over samples = {worst:.4f} (must be <= 1)", flush=True)
ok &= worst <= 1.0

# (3) D' consistency at a few u
for u in [Fr(1, 2), Fr(9, 10), Fr(97, 100), Fr(99, 100)]:
    h = Fr(1, 10000)
    Dp, _, _ = S.D_and_Gamma(u + h)
    Dm, _, _ = S.D_and_Gamma(u - h)
    _, G, _ = S.D_and_Gamma(u)
    fd = (float(Dp.lo) - float(Dm.lo)) / (2 * float(h))
    pred = -0.5 * 6 * float(b) ** 2 * float(u) ** 2 * (float(G.lo) - float(u))
    print(f"(3) u={float(u)} FD D'={fd:+.8e} -xi''(Gamma-u)/2={pred:+.8e} diff={fd - pred:+.2e}", flush=True)
print("ALL CHECKS PASS" if ok else "CHECK FAILED", f"rss={maxrss_mb():.0f}MB")
