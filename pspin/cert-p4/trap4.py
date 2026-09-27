"""Rigorous (nested) Gaussian expectations by the trapezoid rule on R with analytic-strip error bounds.

Trefethen-Weideman (SIAM Rev. 56 (2014), Thm 5.1): if w is analytic in |Im z| < a, w -> 0 uniformly as
|Re z| -> oo in the strip, and int |w(x+ib)| dx <= M for all |b| < a, then for every h > 0
    | h sum_{k in Z} w(kh) - int w |  <=  2M / (e^{2 pi a / h} - 1).
Truncation to |kh| <= L is bounded by the tail integral of a real majorant W(|z|) that is decreasing on
[L-h, oo):  h sum_{kh > L} W(kh) <= int_{L-h}^oo W.

Integrands are built from Y -> cosh^x(Y), logcosh(Y), tanh(Y) with Y = (real shift) + s z. The strip
half-width is chosen so that |Im Y| <= pi/4, where (for Y = c + i d, |d| <= pi/4):
    |cosh Y| <= cosh c,  |cosh Y| >= cos(d) cosh c >= cosh(c)/sqrt2,  |arg cosh Y| <= |d| <= pi/4,
    |tanh Y| <= 1,  |logcosh Y| <= logcosh c + log(2)/2 + pi/4 <= |c| + 6/5,
    Re cosh^x Y >= 2^{-x/2} cos(x pi/4) cosh^x c  (0 < x < 1).
|phi(x+ib)| = phi(x) e^{b^2/2}.

Arithmetic: iv.IV outward-rounded intervals (elementary functions widened by 3 ulp); large sums are done
in float64 on the endpoint arrays with the a priori bound |fl(sum) - sum| <= gamma_{n-1} sum|x_i|
(valid for any summation order), then rounded outward. Error-bound constants are computed in floats and
inflated by 2% (they multiply factors <= 1e-13)."""
import math

import numpy as np

from iv import IV, iexp, ilog, SQRT2PI, _dn, _up

U_ROUND = 2.0 ** -53
AMAX = 2.0


def ic(v):
    return IV(_dn(np.float64(v)), _up(np.float64(v)))


def icosh(Y):
    return (iexp(Y) + iexp(-Y)) * 0.5


def ilc(Y):
    return ilog(icosh(Y))


def itanh(Y):
    ep, em = iexp(Y), iexp(-Y)
    return (ep - em) / (ep + em)


def ipow_cosh(Y, X):
    """cosh(Y)^X for real Y (positive base)"""
    return iexp(ilc(Y) * X)


def fsum_fast(a, axis):
    """Rigorous sum of an IV array along an axis via float sums plus gamma_{n-1} bound."""
    n = a.lo.shape[axis]
    g = (n * U_ROUND) / (1 - n * U_ROUND) * 1.0000001
    slo = np.sum(a.lo, axis=axis)
    shi = np.sum(a.hi, axis=axis)
    elo = g * np.sum(np.abs(a.lo), axis=axis)
    ehi = g * np.sum(np.abs(a.hi), axis=axis)
    return IV(_dn(_dn(slo - elo)), _up(_up(shi + ehi)))


def full_poly_exp(a, b, c, d):
    """upper bound of int_R e^{a+b|z|}(c+d|z|) phi(z) dz (b,c,d >= 0)."""
    return 2.0 * math.exp(a + b * b / 2) * (c + d * (0.4 + b)) * 1.02


def tail_poly_exp(a, b, c, d, L):
    """upper bound of int_{|z|>L} e^{a+b|z|}(c+d|z|) phi(z) dz, requires L >= b+1."""
    t = L - b
    assert t >= 1.0
    ph = math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    return 2.0 * math.exp(a + b * b / 2) * ph * ((c + d * b) / t + d) * 1.02 + 1e-300


def strip_err(M, a, h):
    return 2.0 * M / math.expm1(2 * math.pi * a / h) * 1.02


def nodes(h, L):
    """exact nodes k*h, |k h| <= L, h a power of two; returns (z float array, weights IV = h*phi(z))."""
    kmax = int(math.floor(L / h))
    z = h * np.arange(-kmax, kmax + 1, dtype=float)
    Z = IV.pt(z)
    w = iexp(Z * Z * (-0.5)) * SQRT2PI.recip() * h
    return z, w


def pick_h(s, target=45.0):
    """power-of-two step with 2 pi a/h >= target, a = min(pi/(4 s), AMAX)."""
    a = AMAX if s <= 0 else min(math.pi / (4 * s), AMAX)
    h = 1.0
    while 2 * math.pi * a / h < target:
        h /= 2
    return a, h


def inner_expect(fs, y, s2, growth, chunk=48):
    """Enclosures of I_j(y_i) = E f_j(y_i + s2 Z) for an IV array y (1-D) and IV scalar s2.
    growth[j](ymag) -> (tail_params(a,b,c,d), strip_params(a,b,c,d)) majorants in z for fixed |y| <= ymag,
    valid on the real line and on the strip |Im z| <= a2 (with |Im(s2 z)| <= pi/4) respectively."""
    s2f = float(s2.hi)
    a2, h2 = pick_h(s2f)
    b_max = max(max(growth[j](0.0)[0][1], growth[j](0.0)[1][1]) for j in range(len(fs)))
    L2 = h2 * math.ceil((b_max + 13.0) / h2)
    z2, w2 = nodes(h2, L2)
    Z2 = IV.pt(z2[None, :])
    W2 = IV(w2.lo[None, :], w2.hi[None, :])
    out_lo = [[] for _ in fs]
    out_hi = [[] for _ in fs]
    n = y.lo.shape[0]
    for i0 in range(0, n, chunk):
        yy = IV(y.lo[i0:i0 + chunk][:, None], y.hi[i0:i0 + chunk][:, None])
        Y = yy + Z2 * s2
        for j, f in enumerate(fs):
            S = fsum_fast(f(Y) * W2, axis=1)
            ymag = np.maximum(np.abs(yy.lo[:, 0]), np.abs(yy.hi[:, 0]))
            err = np.empty_like(ymag)
            for r, ym in enumerate(ymag):
                (ta, tb, tc, td), (sa, sb, sc, sd) = growth[j](float(ym))
                e = tail_poly_exp(ta, tb, tc, td, L2 - h2)
                e += strip_err(math.exp(a2 * a2 / 2) * full_poly_exp(sa, sb, sc, sd), a2, h2)
                err[r] = e
            out_lo[j].append(_dn(S.lo - err))
            out_hi[j].append(_up(S.hi + err))
    return [IV(np.concatenate(out_lo[j]), np.concatenate(out_hi[j])) for j in range(len(fs))], (h2, L2, a2)


def outer_expect(psi_vals_fn, s1f, L1, h1, a1, tail_params, strip_params):
    """E psi(Z) by the trapezoid rule: psi_vals_fn(Z1 IV nodes) -> IV values at nodes."""
    z1, w1 = nodes(h1, L1)
    vals = psi_vals_fn(IV.pt(z1))
    S = fsum_fast(vals * w1, axis=0)
    e = tail_poly_exp(*tail_params, L1 - h1) + strip_err(math.exp(a1 * a1 / 2) * full_poly_exp(*strip_params), a1, h1)
    return IV(_dn(S.lo - e), _up(S.hi + e))
