"""Shared parameters and xi-dependent interval helpers for the p=4 certificate.

Model: J_I ~ N(j0 p!/N^{p-1}, p!/(2 N^{p-1})), Gibbs weight e^{beta H}, p = 4.
Centred part: covariance N xi(R) + O(1) with xi(q) = beta^2 q^4 / 2.
xi'(q) = 2 beta^2 q^3, xi''(q) = 6 beta^2 q^2, theta(q) = q xi'(q) - xi(q) = 3 beta^2 q^4 / 2."""
import resource
import sys
from fractions import Fraction as Fr

from iv import IV

P_SPIN = 4
J0 = Fr(8107, 10000)
BETA_W = 2 * J0                  # Nishimori point
BETA_C = Fr(10, 3)               # cold point, T = 0.3
U0 = Fr(21, 100)                 # xi'(t) <= t on [0, U0] at BETA_C (exact root 1/(sqrt2 beta) = 0.21213...)
M_A = Fr(2, 5)                   # universal bound covers 0 < m <= M_A at BETA_C


def ivr(f):
    f = Fr(f)
    return IV.exact_rational(f.numerator, f.denominator)


def xi(B, u):
    return B * B * u * u * u * u * 0.5


def dxi(B, u):
    return B * B * u * u * u * 2.0


def ddxi(B, u):
    return B * B * u * u * 6.0


def theta(B, u):
    return B * B * u * u * u * u * 1.5


def maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


# exact rational sanity checks of the elementary side conditions
assert 2 * BETA_C ** 2 * U0 ** 3 <= U0                       # xi'(U0) <= U0
g = lambda m, b: b * J0 * m ** 4 - m * m / 2
assert g(M_A, BETA_C) < 0                                     # universal bound strictly negative at M_A
assert 2 * BETA_W * J0 * Fr(1) > 0
