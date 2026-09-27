"""Rigorous Gaussian integrals via composite Simpson with interval 4th-derivative remainders."""
import math
import numpy as np
from iv import IV, T, isum, tphi, _up

K4 = 4


def simpson(gfun, L, n):
    """Enclose int_{-L}^{L} g(z) dz. gfun maps a Taylor variable to the Taylor jet of g.
    n Simpson cells of width w=2L/n. Error term -(w^5/2880) g''''(xi) per cell."""
    w = 2.0 * L / n
    nodes = -L + (w / 2.0) * np.arange(2 * n + 1)          # endpoints and midpoints
    gv = gfun(T.var(IV.pt(nodes), 0)).c[0]                  # point enclosures
    wts = np.ones(2 * n + 1)
    wts[1:-1:2] = 4.0
    wts[2:-1:2] = 2.0
    # exact weights w/6 * {1,4,2,...}; w/6 is not exact in binary, so enclose it
    w6 = IV(np.nextafter(w / 6.0, -np.inf), np.nextafter(w / 6.0, np.inf))
    main = isum(gv * IV.pt(wts)) * w6
    lo_c = -L + w * np.arange(n)
    cell = IV(lo_c, np.minimum(lo_c + w, L))
    g4 = gfun(T.var(cell, K4)).c[4] * 24.0                   # encloses g'''' on cells
    c5 = IV(np.nextafter(w ** 5 / 2880.0, -np.inf), np.nextafter(w ** 5 / 2880.0, np.inf))
    rem = isum(g4) * c5
    return main - rem


def tail_exp(C, k, L):
    """Upper bound for 2*int_L^inf C e^{k z} phi(z) dz (both tails), L > k + 1."""
    t = L - k
    assert t > 1
    val = 2.0 * C * math.exp(k * k / 2.0) * math.exp(-t * t / 2.0) / (math.sqrt(2 * math.pi) * t)
    return val * 1.01 + 1e-300


def expect(Ffun, C, k, L=None, n=2000):
    """Enclose E[F(Z)], Z~N(0,1), given |F(z)| <= C e^{k|z|} for |z| >= L."""
    if L is None:
        L = max(12.0, k + 12.0)
    g = lambda z: Ffun(z) * tphi(z)
    core = simpson(g, L, n)
    tb = tail_exp(C, k, L)
    return IV(np.nextafter(core.lo - tb, -np.inf), np.nextafter(core.hi + tb, np.inf))
