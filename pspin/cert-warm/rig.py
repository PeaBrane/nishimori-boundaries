"""Rigorous building blocks of the second implementation (Section 8, sec:certificates).

Interval arithmetic: mpmath.iv (directed rounding of mpf exp/log/cos).
Large convolutions: numpy float64 with a Higham forward-error bound
|fl(sum a_i b_i) - sum a_i b_i| <= gamma_n sum |a_i b_i|, gamma_n = n u/(1-n u),
valid for any summation order / FMA, IEEE-754 binary64, no overflow; an absolute
slack n*2^-1070 covers possible underflow.

Quadrature: trapezoid rule on the whole line for functions analytic in a strip
|Im z| < a (Trefethen-Weideman 2014, Thm 5.1):
  |h sum_k w(c+kh) - int w| <= 2M/(exp(2 pi a/h)-1),  M >= sup_{|b|<a} int |w(t+ib)| dt,
plus an explicit Gaussian tail bound for truncating the infinite sum:
for |f(z)| <= A e^{lam|z|},  h sum_{|k|>K} phi_s(kh)|f(c+kh)|
   <= 2 A e^{lam|c|} e^{lam^2 s/2} Q((K h - lam s)/sqrt s),   (needs K h >= lam s)
with Q(r) <= exp(-r^2/2)/(r sqrt(2 pi)).
"""
import math
import os
from fractions import Fraction as Fr

import numpy as np
from mpmath import iv
from mpmath.libmp import round_ceiling, round_floor, to_float

iv.prec = int(os.environ.get("RIG_PREC", "80"))
U53 = 2.0 ** -53


def ivq(fr):
    fr = Fr(fr)
    return iv.mpf(fr.numerator) / iv.mpf(fr.denominator)


def flo(I):
    return float(np.nextafter(to_float(I._mpi_[0], rnd=round_floor), -np.inf))


def fhi(I):
    return float(np.nextafter(to_float(I._mpi_[1], rnd=round_ceiling), np.inf))


def ivf(a, b):
    return iv.mpf([float(a), float(b)])


def up(I):
    """Upper endpoint as mpmath point interval (for error bounds)."""
    return iv.make_mpf((I._mpi_[1], I._mpi_[1]))


def lower(I):
    return iv.make_mpf((I._mpi_[0], I._mpi_[0]))


LOG2 = iv.log(iv.mpf(2))
PI = iv.pi
SQ2PI = iv.sqrt(2 * PI)


def logcosh_nonneg(z):
    """log cosh z for z >= 0 (interval, z assumed >= 0)."""
    return z - LOG2 + iv.log(1 + iv.exp(-2 * z))


def absz(z):
    """|z| for an interval not straddling 0."""
    neg_lo = z._mpi_[0][0] == 1 and z._mpi_[0][1] != 0
    neg_hi = z._mpi_[1][0] == 1 and z._mpi_[1][1] != 0
    if neg_lo and not neg_hi and z._mpi_[1][1] != 0:
        raise ValueError("interval straddles 0")
    return -z if neg_lo else z


def logcosh(z):
    return logcosh_nonneg(absz(z))


def cosh(z):
    e = iv.exp(z)
    return (e + 1 / e) / 2


def Qup(r):
    """Upper bound on Gaussian tail P(N(0,1) > r), r > 0 interval."""
    r = lower(r)
    assert r > 0
    return iv.exp(-r * r / 2) / (r * SQ2PI)


def disc(M, b, h):
    """Trapezoid discretization bound 2M/(exp(2 pi b/h)-1)  (b, h intervals)."""
    return up(2 * up(M) / (iv.exp(2 * PI * lower(b) / h) - 1))


def phi(t, s):
    """N(0, s) density at t (intervals)."""
    return iv.exp(-t * t / (2 * s)) / iv.sqrt(2 * PI * s)


# ----------------------------------------------------------------- numpy sums
def gam(n):
    return n * U53 / (1 - n * U53)


def _dn(a):
    return np.nextafter(a, -np.inf)


def _up(a):
    return np.nextafter(a, np.inf)


def conv_pos(Llo, Lhi, wlo, whi):
    """Bounds on sum_j w_j L_{n-j} (valid part) for nonnegative interval arrays."""
    n = len(wlo)
    g = gam(n)
    alo = np.convolve(Llo, wlo, "valid")
    ahi = np.convolve(Lhi, whi, "valid")
    slack = n * 2.0 ** -1070
    lo = _dn(_dn(alo * (1 - 3 * g)) - slack)
    hi = _up(_up(ahi * (1 + 3 * g)) + slack)
    return lo, hi


def conv_signed(Llo, Lhi, wlo, whi):
    """Bounds on sum_j w_j L_{n-j} for signed L intervals, nonnegative w."""
    plo_pos, _ = conv_pos(np.maximum(Llo, 0), np.maximum(Llo, 0), wlo, whi)  # lower of +part: uses wlo
    _, nlo_neg = conv_pos(np.maximum(-Llo, 0), np.maximum(-Llo, 0), wlo, whi)  # upper of (-)part: uses whi
    _, phi_pos = conv_pos(np.maximum(Lhi, 0), np.maximum(Lhi, 0), wlo, whi)
    nhi_neg, _ = conv_pos(np.maximum(-Lhi, 0), np.maximum(-Lhi, 0), wlo, whi)
    lo = _dn(plo_pos - nlo_neg)
    hi = _up(phi_pos - nhi_neg)
    return lo, hi


def dot_pos(alo, ahi, blo, bhi):
    """Bounds on sum a_i b_i for nonnegative interval arrays (floats out)."""
    n = len(alo)
    g = gam(n)
    slo = float(np.dot(alo, blo))
    shi = float(np.dot(ahi, bhi))
    slack = n * 2.0 ** -1070
    return float(_dn(_dn(slo * (1 - 3 * g)) - slack)), float(_up(_up(shi * (1 + 3 * g)) + slack))


def mul_pos(alo, ahi, blo, bhi):
    """Elementwise product of nonnegative interval arrays."""
    return _dn(alo * blo), _up(ahi * bhi)


# ----------------------------------------------------------------- lattice
class Lattice:
    """Rigorous float64 enclosures of functions at z = n*Delta, Delta = 2^-k (n >= 0)."""

    NAMES = ("C", "S", "f", "k", "cx1", "e1")

    def __init__(self, k, x):
        self.k = k
        self.Delta = Fr(1, 2 ** k)
        self.x = x
        self.xi = ivq(x)
        self.N = -1
        self.lo = {nm: [] for nm in self.NAMES}
        self.hi = {nm: [] for nm in self.NAMES}

    def _extend(self, N):
        x = self.xi
        den = iv.mpf(2 ** self.k)
        for n in range(self.N + 1, N + 1):
            z = iv.mpf(n) / den
            lc = logcosh_nonneg(z)
            e1 = iv.exp(z)
            ch = (e1 + 1 / e1) / 2
            C = iv.exp(x * lc)
            em = iv.exp(-2 * z)
            th = (1 - em) / (1 + em)
            vals = {
                "C": C,
                "S": x * C * th,
                "f": ch * lc,
                "k": ch - 1 / ch,
                "cx1": iv.exp((x - 1) * lc),
                "e1": e1,
            }
            for nm in self.NAMES:
                self.lo[nm].append(flo(vals[nm]))
                self.hi[nm].append(fhi(vals[nm]))
        self.N = max(self.N, N)

    def arr(self, name, N):
        """Arrays over n = -N..N."""
        if N > self.N:
            self._extend(N)
        lo = np.array(self.lo[name][: N + 1])
        hi = np.array(self.hi[name][: N + 1])
        if name == "S":  # odd
            flo_ = np.concatenate([-hi[:0:-1], lo])
            fhi_ = np.concatenate([-lo[:0:-1], hi])
        else:  # even
            flo_ = np.concatenate([lo[:0:-1], lo])
            fhi_ = np.concatenate([hi[:0:-1], hi])
        return flo_, fhi_


def kernel(s, Delta, J):
    """Delta*phi_s(j Delta), j=-J..J, as float bounds (s Fraction > 0)."""
    si = ivq(s)
    D = ivq(Delta)
    c = D / iv.sqrt(2 * PI * si)
    lo = np.empty(J + 1)
    hi = np.empty(J + 1)
    for j in range(J + 1):
        t = j * D
        w = c * iv.exp(-t * t / (2 * si))
        lo[j] = flo(w)
        hi[j] = fhi(w)
    return np.concatenate([lo[:0:-1], lo]), np.concatenate([hi[:0:-1], hi])


def gauss_1d(fn, c, s, h, K):
    """sum_{k=-K}^{K} h*phi_s(k h) * fn(c + k h) in iv (fn: iv -> iv)."""
    si = ivq(s)
    hh = ivq(h)
    cc = ivq(c)
    tot = iv.mpf(0)
    for k in range(-K, K + 1):
        t = k * hh
        tot += hh * phi(t, si) * fn(cc + t)
    return tot


def gauss_1d_even(fn, s, h, K):
    """Same with c = 0 and fn even."""
    si = ivq(s)
    hh = ivq(h)
    tot = hh * phi(iv.mpf(0), si) * fn(iv.mpf(0))
    for k in range(1, K + 1):
        t = k * hh
        tot += 2 * hh * phi(t, si) * fn(t)
    return tot
