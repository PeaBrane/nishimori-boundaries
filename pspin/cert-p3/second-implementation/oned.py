"""Rigorous 1D Gaussian expectations E f(c + sqrt(s) g) in mpmath.iv.

Strip bounds used (z = t + i b, |b| < a, a < pi/2 so Re cosh z > 0 and principal
powers/logs of cosh are analytic):
  |cosh z| <= cosh t,  |cosh z| >= cosh t cos b,  |arg cosh z| <= |b|,  |tanh z| <= 1/cos b.
Hence
  |cosh^x z|            <= cosh^x t <= e^{x|t|}
  |log cosh z|          <= log cosh t + l_b <= (1+l_b) e^{|t|},      l_b = -log cos b + b
  |cosh^x z log cosh z| <= (1+l_b) e^{(1+x)|t|}
  |cosh^x z tanh^2 z|   <= cosh^x t / cos^2 b
Integrals of the Gaussian against these majorants (Jensen for x<1, E e^{lam|c+sg|} <= 2 e^{lam|c|} e^{lam^2 s/2})
give M for the trapezoid bound; tails use |f| <= A e^{lam|z|}.
"""
import math
from fractions import Fraction as Fr

from mpmath import iv

import rig
from rig import ivq, up, lower, disc, Qup, logcosh


def _frac_le(val, cap, den=10**6):
    """Rational b = min(cap, floor(val*den)/den) (val float)."""
    b = Fr(math.floor(val * den), den)
    return min(b, cap) if b > 0 else Fr(1, den)


KINDS = ("coshx", "logcosh", "coshx_logcosh", "coshx_tanh2")


def f_iv(kind, x):
    xi = ivq(x) if x is not None else None

    def coshx(z):
        return iv.exp(xi * logcosh(z))

    def lcosh(z):
        return logcosh(z)

    def coshx_lc(z):
        lc = logcosh(z)
        return iv.exp(xi * lc) * lc

    def coshx_t2(z):
        zz = rig.absz(z)
        lc = rig.logcosh_nonneg(zz)
        em = iv.exp(-2 * zz)
        th = (1 - em) / (1 + em)
        return iv.exp(xi * lc) * th * th

    return {"coshx": coshx, "logcosh": lcosh, "coshx_logcosh": coshx_lc, "coshx_tanh2": coshx_t2}[kind]


def bounds(kind, x, c, s, b):
    """Return (M, A, lam) as intervals/Fractions for the strip half-width b (Fraction)."""
    bi = ivq(b)
    si = ivq(s)
    ac = ivq(abs(Fr(c)))
    xi = ivq(x) if x is not None else None
    lb = -iv.log(iv.cos(bi)) + bi
    g = iv.exp(bi * bi / (2 * si))
    if kind == "coshx":
        M = g * iv.exp(xi * ac) * iv.exp(xi * si / 2)
        return M, iv.mpf(1), xi
    if kind == "logcosh":
        M = g * (1 + lb) * 2 * iv.exp(ac) * iv.exp(si / 2)
        return M, iv.mpf(1), iv.mpf(1)
    if kind == "coshx_logcosh":
        lam = 1 + xi
        M = g * (1 + lb) * 2 * iv.exp(lam * ac) * iv.exp(lam * lam * si / 2)
        return M, iv.mpf(1), lam
    if kind == "coshx_tanh2":
        cb = iv.cos(bi)
        M = g * iv.exp(xi * ac) * iv.exp(xi * si / 2) / (cb * cb)
        return M, iv.mpf(1), xi
    raise ValueError(kind)


BMAX = {"coshx": Fr(157, 100), "logcosh": Fr(6, 5), "coshx_logcosh": Fr(6, 5), "coshx_tanh2": Fr(6, 5)}
LAMF = {"coshx": None, "logcosh": 1.0, "coshx_logcosh": None, "coshx_tanh2": None}


def expect(kind, x, c, s, h=Fr(1, 8), r_target=14.0, verbose=False):
    """Rigorous enclosure of E f(c + sqrt(s) g), f >= 0 of the given kind.

    Returns (interval, info dict)."""
    c = Fr(c)
    s = Fr(s)
    assert s > 0
    lamf = {"coshx": float(x) if x is not None else 0, "logcosh": 1.0,
            "coshx_logcosh": 1 + float(x) if x is not None else 0, "coshx_tanh2": float(x) if x is not None else 0}[kind]
    sf = float(s)
    K = math.ceil((lamf * sf + r_target * math.sqrt(sf)) / float(h))
    b = _frac_le(2 * math.pi * sf / float(h), BMAX[kind])
    M, A, lam = bounds(kind, x, c, s, b)
    hi_ = ivq(h)
    si = ivq(s)
    r = (K * hi_ - lam * si) / iv.sqrt(si)
    assert lower(r) > 0
    T = up(2 * A * iv.exp(lam * ivq(abs(c))) * iv.exp(lam * lam * si / 2) * Qup(r))
    E = disc(M, ivq(b), hi_)
    fn = f_iv(kind, x)
    if c == 0:
        S = rig.gauss_1d_even(fn, s, h, K)
    else:
        S = rig.gauss_1d(fn, c, s, h, K)
    # positive integrand: truncated tail in [0, T]; discretization in [-E, E]
    res = S + E * iv.mpf([-1, 1]) + T * iv.mpf([0, 1])
    info = dict(K=K, h=str(h), b=str(b), disc=rig.fhi(E), tail=rig.fhi(T), width=rig.fhi(res) - rig.flo(res))
    if verbose:
        print(kind, info)
    return res, info
