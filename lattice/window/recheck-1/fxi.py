"""Fixed-point interval arithmetic with directed rounding, on Python integers only.

An interval I(lo, hi) encloses the real set [lo / 2**W, hi / 2**W].  Every operation rounds the
lower endpoint down (floor) and the upper endpoint up (ceil), so the result always encloses the
exact image of the operand sets.  exp, log and sqrt are implemented here from scratch (Taylor
series for exp, atanh series for log, integer isqrt), with explicit truncation remainders.
No floating point and no third-party library is used anywhere in this module.
"""
from fractions import Fraction
from math import isqrt

W = 224  # working precision in bits; values are ints scaled by 2**W
ONE = 1 << W


def cdiv(a, b):
    return -((-a) // b)


def _c(x):
    if isinstance(x, I):
        return x
    return I.frac(Fraction(x))


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        if lo > hi:
            raise ValueError("empty interval")
        self.lo = lo
        self.hi = hi

    @staticmethod
    def frac(fr):
        fr = Fraction(fr)
        n = fr.numerator << W
        d = fr.denominator
        return I(n // d, cdiv(n, d))

    def __add__(s, o):
        o = _c(o)
        return I(s.lo + o.lo, s.hi + o.hi)

    __radd__ = __add__

    def __sub__(s, o):
        o = _c(o)
        return I(s.lo - o.hi, s.hi - o.lo)

    def __rsub__(s, o):
        return _c(o) - s

    def __neg__(s):
        return I(-s.hi, -s.lo)

    def __mul__(s, o):
        o = _c(o)
        if s.lo >= 0 and o.lo >= 0:
            return I((s.lo * o.lo) >> W, cdiv(s.hi * o.hi, ONE))
        ps = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return I(min(ps) >> W, cdiv(max(ps), ONE))

    __rmul__ = __mul__

    def __truediv__(s, o):
        o = _c(o)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("divisor interval contains 0")
        lo = min((a << W) // b for a in (s.lo, s.hi) for b in (o.lo, o.hi))
        hi = max(cdiv(a << W, b) for a in (s.lo, s.hi) for b in (o.lo, o.hi))
        return I(lo, hi)

    def __rtruediv__(s, o):
        return _c(o) / s

    def __pow__(s, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("integer power only")
        if s.lo < 0:
            raise ValueError("nonnegative base only")
        r = I(ONE, ONE)
        b = s
        while n:
            if n & 1:
                r = r * b
            n >>= 1
            if n:
                b = b * b
        return r

    def hull(s, o):
        return I(min(s.lo, o.lo), max(s.hi, o.hi))

    def upper(self):
        """Exact rational upper endpoint."""
        return Fraction(self.hi, ONE)

    def lower(self):
        return Fraction(self.lo, ONE)

    def width(self):
        return Fraction(self.hi - self.lo, ONE)

    def __repr__(self):
        return f"I[{float(self.lower())!r}, {float(self.upper())!r}]"


# ---------------------------------------------------------------- exp
def _exp_nonneg(Z):
    """Enclosure (lo, hi), scaled by 2**W, of exp(Z / 2**W) for an integer Z >= 0."""
    k = max(0, Z.bit_length() - W + 10)  # w = z / 2**k <= 2**-10
    G = W + k + 48
    one = 1 << G
    wi = Z << (G - W - k)  # w * 2**G, exact
    lo = hi = one
    tlo = thi = one
    j = 1
    while True:
        tlo = ((tlo * wi) >> G) // j
        thi = cdiv(cdiv(thi * wi, one), j)
        lo += tlo
        hi += thi
        j += 1
        if thi <= 1:
            break
    # remainder sum_{i>=j} w^i/i! <= t_{j-1} * w/(1-w) <= t_{j-1} <= thi
    hi += thi + 1
    for _ in range(k):
        lo = (lo * lo) >> G
        hi = cdiv(hi * hi, one)
    sh = G - W
    return lo >> sh, cdiv(hi, 1 << sh)


def _exp_scalar(Z):
    if Z >= 0:
        return _exp_nonneg(Z)
    L, H = _exp_nonneg(-Z)
    return (ONE * ONE) // H, cdiv(ONE * ONE, L)


def exp(x):
    x = _c(x)
    return I(_exp_scalar(x.lo)[0], _exp_scalar(x.hi)[1])


# ---------------------------------------------------------------- log
def _atanh_nonneg(num, den, G):
    """Enclosure at scale 2**G of atanh(num/den), 0 <= num/den <= 1/3 exactly."""
    one = 1 << G
    zlo = (num << G) // den
    zhi = cdiv(num << G, den)
    z2lo = (zlo * zlo) >> G
    z2hi = cdiv(zhi * zhi, one)
    plo, phi = zlo, zhi
    slo = shi = 0
    j = 0
    while True:
        slo += plo // (2 * j + 1)
        shi += cdiv(phi, 2 * j + 1)
        plo = (plo * z2lo) >> G
        phi = cdiv(phi * z2hi, one)
        j += 1
        if phi <= 1:
            break
    # remainder sum_{i>=j} z^{2i+1}/(2i+1) <= z^{2j+1}/(1-z^2) <= phi * 9/8
    shi += cdiv(9 * phi, 8) + 1
    return slo, shi


_G_LOG = W + 48
_LN2 = tuple(2 * v for v in _atanh_nonneg(1, 3, _G_LOG))  # ln 2 = 2 atanh(1/3)


def _log_scalar(Y, upper):
    """Directed bound (floor if not upper, else ceil), scaled by 2**W, of log(Y / 2**W), Y > 0."""
    if Y <= 0:
        raise ValueError("log of nonpositive")
    G = _G_LOG
    b = Y.bit_length()
    e = b - 1 - W  # Y/2^W = m * 2^e with m = Y / 2^(b-1) in [1, 2)
    half = 1 << (b - 1)
    if Y * Y > 2 * half * half:  # m > sqrt 2: use m/2 in (1/sqrt2, 1)
        half <<= 1
        e += 1
    num = Y - half
    den = Y + half
    if num >= 0:
        alo, ahi = _atanh_nonneg(num, den, G)
    else:
        blo, bhi = _atanh_nonneg(-num, den, G)
        alo, ahi = -bhi, -blo
    lnlo, lnhi = _LN2
    if upper:
        v = 2 * ahi + (e * lnhi if e >= 0 else e * lnlo)
        return cdiv(v, 1 << (G - W))
    v = 2 * alo + (e * lnlo if e >= 0 else e * lnhi)
    return v >> (G - W)


def log(x):
    x = _c(x)
    return I(_log_scalar(x.lo, False), _log_scalar(x.hi, True))


def sqrt(x):
    x = _c(x)
    if x.lo < 0:
        raise ValueError("sqrt of possibly negative interval")
    lo = isqrt(x.lo << W)
    t = x.hi << W
    hi = isqrt(t)
    if hi * hi < t:
        hi += 1
    return I(lo, hi)


def powr(x, s):
    """x**s for x > 0 and real s (interval), via exp(s log x)."""
    return exp(_c(s) * log(x))


def self_test(n=2000, seed=1):
    """Containment test against mpmath at 600 bits (float evidence for the library).

    Returns (number of containment failures, worst relative width of exp/log/sqrt on [-20, 60])."""
    import random
    import mpmath

    mpmath.mp.prec = 600
    rng = random.Random(seed)
    bad = 0
    worst = 0
    S = mpmath.mpf(2) ** W

    def inside(iv, ref):
        return mpmath.mpf(iv.lo) / S <= ref <= mpmath.mpf(iv.hi) / S

    def mp(fr):
        return mpmath.mpf(fr.numerator) / fr.denominator

    for _ in range(n):
        a = Fraction(rng.randint(-80 * 10**6, 60 * 10**6), rng.randint(10**6, 2 * 10**6))
        e = exp(I.frac(a))
        ref = mpmath.exp(mp(a))
        bad += not inside(e, ref)
        if a > -20:
            worst = max(worst, (mpmath.mpf(e.hi - e.lo) / S) / ref)
        y = Fraction(rng.randint(1, 10**12), rng.randint(1, 10**12)) * Fraction(1, 10 ** rng.randint(0, 20))
        lg = log(I.frac(y))
        ref = mpmath.log(mp(y))
        bad += not inside(lg, ref)
        worst = max(worst, (mpmath.mpf(lg.hi - lg.lo) / S) / max(abs(ref), 1))
        sq = sqrt(I.frac(y))
        ref = mpmath.sqrt(mp(y))
        bad += not inside(sq, ref)
        if y > Fraction(1, 10**6):
            worst = max(worst, (mpmath.mpf(sq.hi - sq.lo) / S) / ref)
        # signed interval arithmetic: random corner points must land inside
        u = [Fraction(rng.randint(-10**9, 10**9), rng.randint(1, 10**6)) for _ in range(4)]
        A = I.frac(min(u[0], u[1])).hull(I.frac(max(u[0], u[1])))
        B = I.frac(min(u[2], u[3])).hull(I.frac(max(u[2], u[3])))
        for xa in (u[0], u[1], (u[0] + u[1]) / 2):
            for xb in (u[2], u[3], (u[2] + u[3]) / 2):
                bad += not inside(A * B, mp(xa * xb))
                bad += not inside(A - B, mp(xa - xb))
                if B.lo > 0 or B.hi < 0:
                    bad += not inside(A / B, mp(xa / xb))
        pw = rng.randint(0, 1200)
        base = Fraction(rng.randint(1, 10**6), 10**6)
        bad += not inside(I.frac(base) ** pw, mp(base) ** pw)
    return bad, float(worst)
