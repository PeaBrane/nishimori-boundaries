"""(F) Frank-Wolfe lower bound for the zero-field p=4 free energy at beta = 10/3 (trapezoid engine trap4.py):
  F^0(beta,0) >= P(mu*) + min_u D(u),   mu* = x delta_0 + (1-x) delta_q.
Closed forms (general xi; theta(u) = u xi'(u) - xi(u) = 3 b^2 u^4/2 here), Y_q ~ N(0, xi'(q)),
A = E cosh^x Y_q, K = E[cosh^x Y_q logcosh Y_q]/A, C = dP/dx = -log(A)/x^2 + K/x - theta(q)/2.
 u<q: I0(y) = E_g cosh^x(y + s2 g), I1(y) = E_g[cosh^x tanh](y + s2 g), s2^2 = xi'(q)-xi'(u), y ~ N(0, xi'(u)):
   D(u) = log(A)/x + (1-x)K/x - E_y[I0 log I0]/(x^2 A) + [theta(u) - (1-x) theta(q)]/2
   Gamma(u) = E_y[I1^2/I0]/A
 u>q: s^2 = xi'(u) - xi'(q), Y_u = Y_q + s g, J0 = E_g[cosh logcosh](Y_u), J1 = E_g[cosh tanh^2](Y_u):
   D(u) = log(A)/x + s^2/2 - E[cosh^{x-1}Y_q J0]/(A e^{s^2/2}) + [x theta(q) + theta(u) - theta(q)]/2
   Gamma(u) = E[cosh^{x-1}Y_q J1]/(A e^{s^2/2})
 u=q: D(q) = -x C, Gamma(q) = E[cosh^x Y_q tanh^2 Y_q]/A.
Growth/strip majorants: see trap4.py docstring and the comments below."""
import math
import time
from fractions import Fraction as Fr

import numpy as np

from common4 import BETA_C, ivr, xi, dxi, theta, maxrss_mb
from iv import IV, LOG2, iexp, ilog, isqrt
from trap4 import AMAX, icosh, ilc, itanh, ipow_cosh, inner_expect, outer_expect, pick_h
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

beta = BETA_C
B = ivr(beta)
xr, qr = Fr(930529, 2000000), Fr(4930831, 5000000)
X, Q = ivr(xr), ivr(qr)
xf, qf = float(xr), float(qr)
sq = isqrt(dxi(B, Q))
sqf = float(sq.hi)
LN2 = math.log(2)
KAPPA = 2 ** (-xf / 2) * math.cos(xf * math.pi / 4) * (1 - 1e-12)   # Re I0(c+id) >= KAPPA I0(c)
LOGK = abs(math.log(KAPPA)) * (1 + 1e-12)


def _grid(s):
    a, h = pick_h(s)
    return a, h


def E1(psi, tail, strip, s):
    """1-D E psi(Z) with psi built on Y = s Z; majorants in z."""
    a, h = _grid(s)
    L = h * math.ceil((tail[1] + 13.0) / h)
    return outer_expect(psi, s, L, h, a, tail, strip)


# A = E cosh^x(sq Z):   real & strip majorant e^{x sq |z|}
A = E1(lambda z: ipow_cosh(z * sq, X), (0.0, xf * sqf, 1.0, 0.0), (0.0, xf * sqf, 1.0, 0.0), sqf)
# AK = E cosh^x lc:  real <= e^{x|y|}|y| ; strip <= e^{x|c|}(|c| + 6/5)
AK = E1(lambda z: ipow_cosh(z * sq, X) * ilc(z * sq), (0.0, xf * sqf, 0.0, sqf), (0.0, xf * sqf, 1.2, sqf), sqf)
# Gamma(q) numerator: cosh^x tanh^2 <= e^{x|c|} (|tanh| <= 1 on the strip)
GQ = E1(lambda z: ipow_cosh(z * sq, X) * itanh(z * sq).sq(), (0.0, xf * sqf, 1.0, 0.0), (0.0, xf * sqf, 1.0, 0.0), sqf)
K = AK / A
logA = ilog(A)
P = LOG2 + logA * X.recip() + (xi(B, 1.0) - dxi(B, Q) + (1 - X) * theta(B, Q)) * 0.5
C = -(logA * (X * X).recip()) + K * X.recip() - theta(B, Q) * 0.5
D0 = (1 - X) * C
Dq = -(X * C)
Gq = GQ / A


def D_and_Gamma(u):
    u = Fr(u)
    if u == qr:
        return Dq, Gq, {}
    U = ivr(u)
    if u < qr:
        s1iv = isqrt(dxi(B, U))
        s2iv = isqrt(dxi(B, Q) - dxi(B, U))
        s1, s2 = float(s1iv.hi), float(s2iv.hi)
        # inner: |cosh^x(y+s2 z)|, |cosh^x tanh| <= e^{x|y|} e^{x s2 |z|} (real line and strip)
        gr = lambda ym: ((xf * ym, xf * s2, 1.0, 0.0), (xf * ym, xf * s2, 1.0, 0.0))
        a1, h1 = _grid(s1)
        L1 = h1 * math.ceil((xf * s1 + 13.0) / h1)
        z1 = IV.pt(h1 * np.arange(-int(round(L1 / h1)), int(round(L1 / h1)) + 1, dtype=float))
        (I0, I1), info = inner_expect([lambda Y: ipow_cosh(Y, X), lambda Y: ipow_cosh(Y, X) * itanh(Y)],
                                      z1 * s1iv, s2iv, [gr, gr])
        I0 = IV(np.maximum(I0.lo, 1.0), I0.hi)             # I0 >= 1 on the real line
        c0 = LN2 + xf * xf * s2 * s2 / 2                     # I0(c) <= 2 e^{x|c| + x^2 s2^2/2}
        # real: 0 <= I0 log I0 <= I0 (log I0);  strip: |I0 log I0| <= I0(c)(log I0(c) + |log kappa| + pi/2)
        S = outer_expect(lambda z: I0 * ilog(I0), s1, L1, h1, a1,
                         (c0, xf * s1, c0, xf * s1), (c0, xf * s1, c0 + LOGK + math.pi / 2, xf * s1))
        # real: 0 <= I1^2/I0 <= I0 ; strip: |I1^2/I0| <= I0(c)/kappa
        Gn = outer_expect(lambda z: I1.sq() / I0, s1, L1, h1, a1,
                          (c0, xf * s1, 1.0, 0.0), (c0 + LOGK, xf * s1, 1.0, 0.0))
        D = logA * X.recip() + (1 - X) * K * X.recip() - S * (X * X * A).recip() + (
            theta(B, U) - (1 - X) * theta(B, Q)) * 0.5
        return D, Gn / A, {"h1": h1, "L1": L1, "a1": a1, "inner": info, "n1": len(z1.lo)}
    sgiv = isqrt(dxi(B, U) - dxi(B, Q))
    sg = float(sgiv.hi)
    # inner g0 = cosh lc: real <= e^{|y|+s|z|}(|y|+s|z|); strip <= e^{|y|+s|x|}(|y| + 6/5 + s|x|)
    gr0 = lambda ym: ((ym, sg, ym, sg), (ym, sg, ym + 1.2, sg))
    # inner g1 = cosh tanh^2: real and strip <= e^{|y|+s|z|}
    gr1 = lambda ym: ((ym, sg, 1.0, 0.0), (ym, sg, 1.0, 0.0))
    a1, h1 = _grid(sqf)
    L1 = h1 * math.ceil((xf * sqf + 13.0) / h1)
    z1 = IV.pt(h1 * np.arange(-int(round(L1 / h1)), int(round(L1 / h1)) + 1, dtype=float))
    Y1 = z1 * sq
    (J0, J1), info = inner_expect([lambda Y: icosh(Y) * ilc(Y), lambda Y: icosh(Y) * itanh(Y).sq()],
                                  Y1, sgiv, [gr0, gr1])
    J0 = IV(np.maximum(J0.lo, 0.0), J0.hi)
    J1 = IV(np.maximum(J1.lo, 0.0), J1.hi)
    pre = ipow_cosh(Y1, X - 1)
    s2h = sg * sg / 2
    # real: cosh^{x-1} <= 2^{1-x} e^{-(1-x)|y|};  J0 <= 2 e^{|y| + s^2/2}(|y| + s(0.4+s))
    # strip: |cosh^{x-1}| <= 2^{(1-x)/2} 2^{1-x} e^{-(1-x)|c|};  |J0| <= 2 e^{|c|+s^2/2}(|c| + 6/5 + s(0.4+s))
    T1 = outer_expect(lambda z: pre * J0, sqf, L1, h1, a1,
                      ((2 - xf) * LN2 + s2h, xf * sqf, sg * (0.4 + sg), sqf),
                      ((1.5 * (1 - xf) + 1) * LN2 + s2h, xf * sqf, 1.2 + sg * (0.4 + sg), sqf))
    # J1 <= cosh(y) e^{s^2/2} <= e^{|y| + s^2/2} (real and strip)
    G1 = outer_expect(lambda z: pre * J1, sqf, L1, h1, a1,
                      ((1 - xf) * LN2 + s2h, xf * sqf, 1.0, 0.0),
                      (1.5 * (1 - xf) * LN2 + s2h, xf * sqf, 1.0, 0.0))
    e = iexp((dxi(B, U) - dxi(B, Q)) * 0.5)
    D = logA * X.recip() + (dxi(B, U) - dxi(B, Q)) * 0.5 - T1 * (A * e).recip() + (
        X * theta(B, Q) + theta(B, U) - theta(B, Q)) * 0.5
    return D, G1 / (A * e), {"h1": h1, "L1": L1, "a1": a1, "inner": info, "n1": len(z1.lo)}


if __name__ == "__main__":
    import sys
    print("A", A, "K", K, "Gamma(q)", Gq)
    print("P(mu*)", P, "width", float(P.hi - P.lo))
    print("C", C, "D(0)", D0, "D(q)", Dq)
    us = [Fr(s) for s in sys.argv[1:]] or [Fr(3, 10), Fr(9, 10), Fr(98, 100), Fr(99, 100)]
    for u in us:
        t0 = time.time()
        D, G, info = D_and_Gamma(u)
        print(f"u={float(u)} D=[{D.lo!r},{D.hi!r}] w={D.hi - D.lo:.2e} Gamma=[{G.lo!r},{G.hi!r}] "
              f"gw={G.hi - G.lo:.2e} t={time.time() - t0:.1f}s rss={maxrss_mb():.0f}MB {info}", flush=True)
