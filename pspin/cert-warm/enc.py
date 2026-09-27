"""Interval enclosures of the scalar-channel quantities used by the RS warm certificate.

psi(r) = E log cosh(r + sqrt(r) Z),  M(r) = E tanh(r + sqrt(r) Z),
Mp(R)  = E[sech^2(Y)(1 - tanh Y)], Y = r + sqrt(r) Z, enclosed over an r-interval R.
All via gauss.expect (composite Simpson + interval Taylor remainder + analytic Gaussian tails).
"""
import math
from fractions import Fraction as Fr

from iv import IV, isqrt, tlogcosh, ttanh
from gauss import expect


def fr_iv(x):
    x = Fr(x)
    return IV.exact_rational(x.numerator, x.denominator)


def psi(r, n=1000):
    r = Fr(r)
    if r == 0:
        return IV.pt(0.0)
    R = fr_iv(r)
    s = isqrt(R)
    # log cosh(r + sqrt(r) z) <= r + sqrt(r)|z| <= (r + sqrt r) e^{|z|}
    C = (float(r) + math.sqrt(float(r))) * (1 + 1e-9) + 1e-12
    return expect(lambda z: tlogcosh(z * s + R), C, 1.0, n=n)


def M(r, n=1000):
    r = Fr(r)
    if r == 0:
        return IV.pt(0.0)
    R = fr_iv(r)
    s = isqrt(R)
    return expect(lambda z: ttanh(z * s + R), 1.0, 0.0, n=n)


def Mp_hull(r_lo, r_hi, n=1000):
    """Enclosure of {E[sech^2(Y)(1-tanh Y)] : r in [r_lo, r_hi]}, r_lo > 0."""
    a, b = fr_iv(r_lo), fr_iv(r_hi)
    R = IV(a.lo, b.hi)
    s = isqrt(R)

    def F(z):
        t = ttanh(z * s + R)
        return (1 - t * t) * (1 - t)
    # 0 <= (1-t)^2 (1+t) <= 32/27
    return expect(F, 32.0 / 27.0 * (1 + 1e-9), 0.0, n=n)
