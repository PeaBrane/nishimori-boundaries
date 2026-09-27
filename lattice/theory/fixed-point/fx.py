"""Fixed-point integer interval arithmetic with directed rounding (used by h4_fixed.py).

A value is an interval [lo, hi] * 2^-P with integer lo <= hi. Every operation rounds lo down
and hi up, so the true real value always stays inside. No floats are used.
"""
from fractions import Fraction
from math import isqrt

P = 384
S = 1 << P


def fdiv_floor(a, b):
    return a // b


def fdiv_ceil(a, b):
    return -((-a) // b)


class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        assert lo <= hi, (lo, hi)
        self.lo, self.hi = lo, hi

    @staticmethod
    def frac(q):
        q = Fraction(q)
        return Iv(fdiv_floor(q.numerator * S, q.denominator), fdiv_ceil(q.numerator * S, q.denominator))

    @staticmethod
    def int(n):
        return Iv(n * S, n * S)

    def __add__(self, o):
        o = _c(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, o):
        o = _c(o)
        return Iv(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        return _c(o) - self

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __mul__(self, o):
        o = _c(o)
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(fdiv_floor(min(c), S), fdiv_ceil(max(c), S))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _c(o)
        assert o.lo > 0, "division by interval not strictly positive"
        c = [(self.lo, o.lo), (self.lo, o.hi), (self.hi, o.lo), (self.hi, o.hi)]
        lo = min(fdiv_floor(a * S, b) for a, b in c)
        hi = max(fdiv_ceil(a * S, b) for a, b in c)
        return Iv(lo, hi)

    def __rtruediv__(self, o):
        return _c(o) / self

    def pow(self, n):
        assert self.lo >= 0
        lo = _ipow(self.lo, n, floor=True)
        hi = _ipow(self.hi, n, floor=False)
        return Iv(lo, hi)

    def sqrt(self):
        assert self.lo >= 0
        lo = isqrt(self.lo * S)
        h = isqrt(self.hi * S)
        if h * h < self.hi * S:
            h += 1
        return Iv(lo, h)

    def flo(self):
        return Fraction(self.lo, S)

    def fhi(self):
        return Fraction(self.hi, S)

    def width(self):
        return Fraction(self.hi - self.lo, S)

    def __repr__(self):
        return f"[{float(self.flo()):.15g}, {float(self.fhi()):.15g}]"

    def dec(self, nd=15):
        return f"[{fmt_down(self.flo(), nd)}, {fmt_up(self.fhi(), nd)}]"


def _c(o):
    if isinstance(o, Iv):
        return o
    if isinstance(o, int):
        return Iv.int(o)
    return Iv.frac(o)


def _ipow(a, n, floor):
    # a is scaled by S; compute a^n scaled by S with directed rounding, a >= 0.
    result = S
    base = a
    rnd = fdiv_floor if floor else fdiv_ceil
    while n:
        if n & 1:
            result = rnd(result * base, S)
        n >>= 1
        if n:
            base = rnd(base * base, S)
    return result


def fmt_down(q, nd):
    q = Fraction(q)
    s = 10 ** nd
    v = (q.numerator * s) // q.denominator
    return _fmt(v, nd)


def fmt_up(q, nd):
    q = Fraction(q)
    s = 10 ** nd
    v = -((-q.numerator * s) // q.denominator)
    return _fmt(v, nd)


def _fmt(v, nd):
    sign = "-" if v < 0 else ""
    v = abs(v)
    ip, fp = divmod(v, 10 ** nd)
    return f"{sign}{ip}.{fp:0{nd}d}"
