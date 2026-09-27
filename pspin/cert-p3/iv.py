"""Vectorized interval arithmetic with outward rounding, and interval Taylor arithmetic.

Rounding model: IEEE-754 double; +,-,*,/ are correctly rounded, so one nextafter step outward
encloses the exact result. numpy exp/log/log1p are assumed accurate to < 2 ulp; results of
elementary functions are widened by 3 ulp outward. Sums use math.fsum (correctly rounded)
followed by one outward ulp.
"""
import math
import numpy as np

INF = np.inf


def _dn(x, k=1):
    for _ in range(k):
        x = np.nextafter(x, -INF)
    return x


def _up(x, k=1):
    for _ in range(k):
        x = np.nextafter(x, INF)
    return x


class IV:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=float)
        hi = lo if hi is None else np.asarray(hi, dtype=float)
        self.lo, self.hi = lo, hi

    @staticmethod
    def pt(x):
        x = np.asarray(x, dtype=float)
        return IV(x, x)

    @staticmethod
    def exact_rational(num, den):
        """Enclosure of num/den for Python ints."""
        v = num / den
        return IV(_dn(v), _up(v))

    def __repr__(self):
        return f"IV({self.lo}, {self.hi})"

    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def width(self):
        return self.hi - self.lo

    # arithmetic
    def __add__(self, o):
        o = _as(o)
        return IV(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return IV(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _as(o)
        return IV(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return _as(o) - self

    def __mul__(self, o):
        o = _as(o)
        a, b, c, d = self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi
        lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
        hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
        # 0*inf guard not needed: no infinities expected
        return IV(_dn(lo), _up(hi))

    __rmul__ = __mul__

    def recip(self):
        if np.any((self.lo <= 0) & (self.hi >= 0)):
            raise ZeroDivisionError("interval contains 0")
        lo = 1.0 / self.hi
        hi = 1.0 / self.lo
        return IV(_dn(lo), _up(hi))

    def __truediv__(self, o):
        return self * _as(o).recip()

    def __rtruediv__(self, o):
        return _as(o) * self.recip()

    def sq(self):
        a2, b2 = self.lo * self.lo, self.hi * self.hi
        straddle = (self.lo <= 0) & (self.hi >= 0)
        lo = np.where(straddle, 0.0, _dn(np.minimum(a2, b2)))
        lo = np.maximum(lo, 0.0)
        return IV(lo, _up(np.maximum(a2, b2)))

    def abs(self):
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(np.abs(self.lo), np.abs(self.hi)))
        hi = np.maximum(np.abs(self.lo), np.abs(self.hi))
        return IV(lo, hi)

    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def hull(self, o):
        return IV(np.minimum(self.lo, o.lo), np.maximum(self.hi, o.hi))

    def __getitem__(self, k):
        return IV(self.lo[k], self.hi[k])


def _as(o):
    return o if isinstance(o, IV) else IV.pt(o)


ULP_F = 3


def iexp(a):
    return IV(np.maximum(_dn(np.exp(a.lo), ULP_F), 0.0), _up(np.exp(a.hi), ULP_F))


def ilog(a):
    if np.any(a.lo <= 0):
        raise ValueError("log of nonpositive")
    return IV(_dn(np.log(a.lo), ULP_F), _up(np.log(a.hi), ULP_F))


def isqrt(a):
    return IV(_dn(np.sqrt(np.maximum(a.lo, 0.0))), _up(np.sqrt(a.hi)))


def isum(a, axis=None):
    """Rigorous sum of an IV array (fsum-based) along an axis (None = all)."""
    if axis is None:
        lo = math.fsum(np.ravel(a.lo))
        hi = math.fsum(np.ravel(a.hi))
        return IV(_dn(np.float64(lo)), _up(np.float64(hi)))
    lo = np.apply_along_axis(math.fsum, axis, a.lo)
    hi = np.apply_along_axis(math.fsum, axis, a.hi)
    return IV(_dn(lo), _up(hi))


SQRT2PI = IV(_dn(np.float64(math.sqrt(2 * math.pi)), 2), _up(np.float64(math.sqrt(2 * math.pi)), 2))
LOG2 = IV(_dn(np.float64(math.log(2.0)), 2), _up(np.float64(math.log(2.0)), 2))


# ---------------------------------------------------------------------------------------
# Interval Taylor arithmetic: T.c[k] encloses f^{(k)}/k! over the base interval.
# ---------------------------------------------------------------------------------------
class T:
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = c

    @property
    def K(self):
        return len(self.c) - 1

    @staticmethod
    def var(x, K):
        x = _as(x)
        if K == 0:
            return T([x])
        zero = IV.pt(np.zeros_like(x.lo))
        one = IV.pt(np.ones_like(x.lo))
        return T([x, one] + [zero] * (K - 1))

    @staticmethod
    def const(x, K):
        x = _as(x)
        zero = IV.pt(np.zeros_like(x.lo))
        return T([x] + [zero] * K)

    def __add__(self, o):
        if not isinstance(o, T):
            return T([self.c[0] + o] + self.c[1:])
        return T([a + b for a, b in zip(self.c, o.c)])

    __radd__ = __add__

    def __neg__(self):
        return T([-a for a in self.c])

    def __sub__(self, o):
        return self + (-o if isinstance(o, T) else -_as(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, T):
            o = _as(o)
            return T([a * o for a in self.c])
        K = self.K
        out = []
        for k in range(K + 1):
            s = self.c[0] * o.c[k]
            for i in range(1, k + 1):
                s = s + self.c[i] * o.c[k - i]
            out.append(s)
        return T(out)

    __rmul__ = __mul__

    def recip(self):
        K = self.K
        b = self.c
        inv0 = b[0].recip()
        d = [inv0]
        for k in range(1, K + 1):
            s = b[1] * d[k - 1]
            for j in range(2, k + 1):
                s = s + b[j] * d[k - j]
            d.append(-(s * inv0))
        return T(d)

    def __truediv__(self, o):
        if not isinstance(o, T):
            return self * _as(o).recip()
        return self * o.recip()


def texp(a):
    K = a.K
    e = [iexp(a.c[0])]
    for k in range(1, K + 1):
        s = a.c[1] * e[k - 1]
        for j in range(2, k + 1):
            s = s + (j * a.c[j]) * e[k - j]
        e.append(s * IV.exact_rational(1, k))
    return T(e)


def tlog(a):
    K = a.K
    inv0 = a.c[0].recip()
    l = [ilog(a.c[0])]
    for k in range(1, K + 1):
        s = a.c[k]
        for j in range(1, k):
            s = s - (IV.exact_rational(j, k) * l[j]) * a.c[k - j]
        l.append(s * inv0)
    return T(l)


def tcosh(a):
    return (texp(a) + texp(-a)) * 0.5


def tsinh(a):
    return (texp(a) - texp(-a)) * 0.5


def tlogcosh(a):
    return tlog(tcosh(a))


def ttanh(a):
    return tsinh(a) / tcosh(a)


def tpow_pos(a, x):
    return texp(tlog(a) * x)


def tphi(z):
    """standard normal density of a Taylor variable"""
    return texp(z * z * (-0.5)) * SQRT2PI.recip()


def iphi(z):
    """enclosure of the standard normal density over an interval z"""
    straddle = (z.lo <= 0) & (z.hi >= 0)
    zmin2 = np.where(straddle, 0.0, np.minimum(z.lo * z.lo, z.hi * z.hi))
    zmax2 = np.maximum(z.lo * z.lo, z.hi * z.hi)
    c = 1.0 / math.sqrt(2 * math.pi)
    return IV(np.exp(-zmax2 * 0.5) * c * (1 - 1e-12), np.exp(-zmin2 * 0.5) * c * (1 + 1e-12))
