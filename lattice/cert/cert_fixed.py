"""Independent second certificate: pure-Python fixed-point interval arithmetic with directed
rounding (integers scaled by 2^D), exact Fractions for the disorder law, no Arb and no mpmath.

Differences from cert_arb.py (on purpose):
  * variable s = e^{-2 beta} with a rational s-partition instead of a beta-partition;
    transcendental inputs are avoided entirely: e^{-2 beta} is bounded by rational Taylor
    bounds, t = (1-s)/(1+s) and z = tanh(b0 atanh s) = (1 - t^b0)/(1 + t^b0) are rational in s
    and monotone, so endpoint evaluation plus monotonicity encloses them;
  * the unit law is obtained by enumerating (J_e, J_a J_b, path signs) rather than by the
    closed-form class formulas, and the chain law by a k-step convolution over class counts
    (dynamic programming, tracking signed mass) rather than by the multinomial formula;
  * u_H is summed over the chain law rather than using u_H = t^2 u_U^k.
Usage: python3 cert_fixed.py > out/cert_fixed.txt
"""
import sys
import time
from fractions import Fraction as Fr
from math import isqrt

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

D = 480
ONE = 1 << D


def fdiv_floor(a, b):
    return a // b


def fdiv_ceil(a, b):
    return -((-a) // b)


class I:
    """Closed interval [lo/2^D, hi/2^D] with integer endpoints; all ops round outward."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        assert lo <= hi, (lo, hi)
        self.lo, self.hi = lo, hi

    @staticmethod
    def of(x):
        if isinstance(x, I):
            return x
        x = Fr(x)
        n, d = x.numerator, x.denominator
        return I(fdiv_floor(n << D, d), fdiv_ceil(n << D, d))

    @staticmethod
    def hull(a, b):
        a, b = I.of(a), I.of(b)
        return I(min(a.lo, b.lo), max(a.hi, b.hi))

    def __add__(self, o):
        o = I.of(o)
        return I(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-I.of(o))

    def __rsub__(self, o):
        return I.of(o) + (-self)

    def __mul__(self, o):
        o = I.of(o)
        c = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return I(min(c) >> D, -((-max(c)) >> D))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = I.of(o)
        assert o.lo > 0, "division by an interval not bounded away from 0"
        num = (self.lo << D, self.hi << D)
        return I(min(fdiv_floor(a, b) for a in num for b in (o.lo, o.hi)),
                 max(fdiv_ceil(a, b) for a in num for b in (o.lo, o.hi)))

    def __rtruediv__(self, o):
        return I.of(o) / self

    def __pow__(self, n):
        assert n >= 0
        res, base = I(ONE, ONE), self
        while n:
            if n & 1:
                res = res * base
            base = base * base
            n >>= 1
        return res

    def sqrt(self):
        assert self.lo >= 0
        return I(isqrt(self.lo << D), isqrt(self.hi << D) + 1)

    def lt(self, c):
        c = Fr(c)
        return self.hi * c.denominator < (c.numerator << D)

    def gt(self, c):
        c = Fr(c)
        return self.lo * c.denominator > (c.numerator << D)

    def f(self):
        return (self.lo + self.hi) / 2 / ONE

    def lof(self):
        return self.lo / ONE

    def hif(self):
        return self.hi / ONE


# ------------------------------------------------------------------ rational exp bounds
def exp_neg_bounds(x, n=None):
    """Rational [lo, hi] with lo <= e^{-x} <= hi for rational x >= 0 (Taylor with remainder)."""
    x = Fr(x)
    if n is None:
        n = int(3 * float(x)) + 60
    S, term = Fr(0), Fr(1)
    for j in range(n + 1):
        S += term
        term = term * x / (j + 1)
    # term = x^{n+1}/(n+1)!;  e^x - S_n <= term / (1 - x/(n+2))
    assert x < n + 2
    R = term / (1 - x / (n + 2))
    return 1 / (S + R), 1 / S


# ------------------------------------------------------------------ v_P
def h_exact(v):
    d = 1 - 9 * v * v
    return 36 * v ** 4 / d ** 2 + 36 * v ** 3 / d


def vp_lower(digits=15):
    scale = 10 ** digits
    lo, hi = 0, scale // 3
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if h_exact(Fr(mid, scale)) < 1:
            lo = mid
        else:
            hi = mid
    assert h_exact(Fr(lo, scale)) < 1 < h_exact(Fr(hi, scale))
    return Fr(lo, scale), Fr(hi, scale)


# ------------------------------------------------------------------ law by enumeration
def unit_law_enum(p, b0):
    """Enumerate J_e, J_a, J_b and the two path sign products (each path sign is the parity of
    Binomial(b0,p) negatives: P(-) = (1 - (1-2p)^b0)/2).  Class 1: branch nonzero and same sign
    as J_e; class 2: branch zero (paths disagree); class 3: opposite sign.  The unit coupling has
    the sign of J_e in all classes (|atanh(branch)| < beta), so signed mass = sum P * J_e."""
    q = (1 - (1 - 2 * p) ** b0) / 2
    PJ = {1: 1 - p, -1: p}
    Ps = {1: 1 - q, -1: q}
    prob = [Fr(0)] * 3
    sgn = [Fr(0)] * 3
    for Je in (1, -1):
        for Ja in (1, -1):
            for Jb in (1, -1):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        w = PJ[Je] * PJ[Ja] * PJ[Jb] * Ps[s1] * Ps[s2]
                        if s1 != s2:
                            c = 1
                        else:
                            branch_sign = Ja * Jb * s1
                            c = 0 if branch_sign == Je else 2
                        prob[c] += w
                        sgn[c] += w * Je
    assert sum(prob) == 1
    return prob, sgn


def chain_law(prob, sgn, k):
    """DP over k units: returns dict (n1, n3) -> (probability, signed mass) as Fractions."""
    law = {(0, 0): (Fr(1), Fr(1))}
    for _ in range(k):
        new = {}
        for (n1, n3), (pr, sg) in law.items():
            for c, dn in ((0, (1, 0)), (1, (0, 0)), (2, (0, 1))):
                key = (n1 + dn[0], n3 + dn[1])
                a, b = new.get(key, (Fr(0), Fr(0)))
                new[key] = (a + pr * prob[c], b + sg * sgn[c])
        law = new
    assert sum(v[0] for v in law.values()) == 1
    return law


# ------------------------------------------------------------------ magnitudes on s-intervals
def t_of(s):
    return (1 - s) / (1 + s)


def z_enclosure_at(s, b0):
    """Interval containing z(s) = (1 - t^b0)/(1 + t^b0) at a rational point s > 0."""
    T = I.of(t_of(s)) ** b0
    return I((ONE - T.hi) * ONE // (ONE + T.hi), -((-(ONE - T.lo) * ONE) // (ONE + T.lo)))


def mags_s(slo, shi, b0):
    """Enclosures of t, w1, w3 valid for all s in [slo, shi] (0 <= slo < shi < 1, rational)."""
    s = I.hull(slo, shi)
    t = I.hull(t_of(shi), t_of(slo))                  # t decreasing in s
    if slo > 0:
        zlo = z_enclosure_at(slo, b0).lo               # z increasing in s
        zhi = z_enclosure_at(shi, b0).hi
        Z = I(zlo, zhi) * I(zlo, zhi) / s
    else:
        Z = I.hull(0, Fr(b0) ** 2 * shi / (1 - shi * shi) ** 2)
    s2 = s * s
    r = (2 + Z * (1 + s2)) / (1 + s2 + 2 * s2 * Z)
    w1 = 2 / (1 + r * s2) - 1
    w3 = 1 - 2 / (r + 1)
    return t, w1, w3


class Design:
    def __init__(self, name, b0, k, p):
        self.name, self.b0, self.k, self.p = name, b0, k, Fr(p)
        self.prob, self.sgn = unit_law_enum(self.p, b0)
        law = chain_law(self.prob, self.sgn, k)
        self.law = [(n1, k - n1 - n3, n3, I.of(pr), I.of(sg)) for (n1, n3), (pr, sg) in sorted(law.items())]
        self.exact_law = law

    def functionals(self, t, w1, w3):
        k = self.k
        pt = [t ** i for i in range(k + 3)]
        p1 = [w1 ** i for i in range(k + 1)]
        p3 = [w3 ** i for i in range(k + 1)]
        u = I(0, 0)
        pbar = I(0, 0)
        for n1, n2, n3, pr, _ in self.law:
            x = pt[2 + n2] * p1[n1] * p3[n3]
            u = u + pr * x
            pbar = pbar + pr * (2 - 2 / (1 + x))
        return dict(u=u, pbar=pbar)

    def warm(self):
        p, b0 = self.p, self.b0
        s = p / (1 - p)                    # e^{-2 gamma} exactly
        t = I.of(1 - 2 * p)                # tanh gamma exactly (rounded outward once)
        tb = t ** b0
        a = t * t * (2 * tb / (1 + tb * tb))
        w1 = (t + a) / (1 + t * a)
        w3 = (t - a) / (1 - t * a)
        # same magnitudes through the s-form (consistency)
        t2, w1b, w3b = mags_s(s, s, b0)
        term = I.of(1 - 2 * p) ** 2
        v = I(0, 0)
        vsech = I(0, 0)
        for n1, n2, n3, pr, sg in self.law:
            x = t ** (2 + n2) * w1 ** n1 * w3 ** n3
            A = term * sg                  # signed mass incl. the two terminal edges
            g = ((1 - x) / (1 + x)).sqrt()  # e^{-|K|}
            v = v + (pr + A) / 2 * g + (pr - A) / 2 / g
            vsech = vsech + pr * (1 - x * x).sqrt()
        return v, vsech, (w1, w1b, w3, w3b)

    def half_line(self, key, thr, S0, upper, s_min_beta=64, ratio=Fr(31, 32)):
        """F < thr (upper) or F > thr on s in (0, S0], i.e. beta >= -(1/2) log S0."""
        ok_fn = (lambda F: F.lt(thr)) if upper else (lambda F: F.gt(thr))
        s_min = exp_neg_bounds(2 * s_min_beta)[1]
        s_min = Fr(fdiv_ceil(s_min.numerator << D, s_min.denominator), ONE)
        grid = [S0]
        while grid[-1] > s_min:
            nxt = Fr(fdiv_floor(grid[-1].numerator * ratio.numerator << D, grid[-1].denominator * ratio.denominator), ONE)
            grid.append(max(nxt, s_min))
        stack = [(grid[i + 1], grid[i]) for i in range(len(grid) - 1)]
        n_ok, n_eval, ext, fails = 0, 0, None, []
        covered = []
        while stack:
            lo, hi = stack.pop()
            F = self.functionals(*mags_s(lo, hi, self.b0))[key]
            n_eval += 1
            if ok_fn(F):
                n_ok += 1
                covered.append((lo, hi))
                val = F.hif() if upper else F.lof()
                ext = val if ext is None else (max(ext, val) if upper else min(ext, val))
            elif hi - lo > Fr(1, 10 ** 40) and hi / lo > 1 + Fr(1, 10 ** 14):
                mid = (lo + hi) / 2
                mid = Fr(fdiv_floor(mid.numerator << D, mid.denominator), ONE)
                stack.append((lo, mid))
                stack.append((mid, hi))
            else:
                fails.append((float(lo), float(hi), F.lof(), F.hif()))
                break
        Ft = self.functionals(*mags_s(Fr(0), s_min, self.b0))[key]
        tail_ok = ok_fn(Ft)
        covered.sort()
        cover = covered[0][0] == s_min and covered[-1][1] == S0 and all(
            covered[i][1] == covered[i + 1][0] for i in range(len(covered) - 1))
        ext = max(ext, Ft.hif()) if upper else min(ext, Ft.lof())
        return dict(ok=(not fails) and tail_ok and cover, n=n_ok, n_eval=n_eval, ext=ext,
                    tail=(Ft.lof(), Ft.hif()), fails=fails)

    def point(self, key, beta, thr, above):
        lo, hi = exp_neg_bounds(2 * Fr(beta))
        F = self.functionals(*mags_s(lo, hi, self.b0))[key]
        return (F.gt(thr) if above else F.lt(thr)), (F.lof(), F.hif())


def run(name, b0, k, p, halflines, points, vp_lo):
    t0 = time.time()
    Dz = Design(name, b0, k, p)
    print(f"=== design {name}: b0={b0} k={k} p={p}")
    print(f"  unit law by enumeration: P = {[float(x) for x in Dz.prob]}, signed = {[float(x) for x in Dz.sgn]}; chain classes = {len(Dz.law)}")
    v, vsech, (w1, w1b, w3, w3b) = Dz.warm()
    print(f"  s-form vs t-form magnitudes at gamma: w1 {w1.f():.15f} / {w1b.f():.15f}, w3 {w3.f():.15f} / {w3b.f():.15f}")
    okW = v.lt(vp_lo)
    print(f"  (W) v_H in [{v.lof():.15f}, {v.hif():.15f}]  E sech in [{vsech.lof():.15f}, {vsech.hif():.15f}]  < vP_lo={float(vp_lo):.15f}: {okW}")
    V = Fr(v.hi + 1, ONE)
    print(f"      h(v_H upper) = {float(h_exact(V)):.12f} < 1: {h_exact(V) < 1}")
    allok = okW and h_exact(V) < 1
    for key, thr, beta0, upper in halflines:
        if beta0 == "gamma":
            S0 = Fr(p) / (1 - Fr(p))            # e^{-2 gamma} exactly
            lab = "gamma"
        else:
            S0 = exp_neg_bounds(2 * Fr(beta0))[1]
            S0 = Fr(fdiv_ceil(S0.numerator << D, S0.denominator), ONE)
            lab = str(Fr(beta0))
        HL = Dz.half_line(key, Fr(thr), S0, upper)
        rel = "<" if upper else ">"
        print(f"  (C) {key}_H {rel} {thr} for all beta >= {lab} (s in (0, {float(S0):.6e}]): {HL['ok']}  pieces={HL['n']} evals={HL['n_eval']}"
              f" {'sup<=' if upper else 'inf>='}{HL['ext']:.12f} tail=[{HL['tail'][0]:.12f}, {HL['tail'][1]:.12f}] {HL['fails'] if HL['fails'] else ''}")
        allok = allok and HL["ok"]
    for key, beta, thr, above in points:
        ok, F = Dz.point(key, Fr(beta), Fr(thr), above)
        print(f"  (PT) {key}_H({Fr(beta)}) {'>' if above else '<'} {thr}: {ok}  [{F[0]:.12f}, {F[1]:.12f}]")
        allok = allok and ok
    print(f"  design {name} all ok: {allok}   [{time.time()-t0:.1f}s]")
    sys.stdout.flush()
    return allok


if __name__ == "__main__":
    t0 = time.time()
    vp = vp_lower(15)
    print(f"fixed point D={D} bits; v_P in [{vp[0]}, {vp[1]}]")
    ok = True
    ok &= run("P", 1000, 10, Fr(9, 10000),
              [("u", Fr(1, 3), Fr(741995, 100000), True), ("pbar", Fr(1, 2), Fr(698554, 100000), True),
               ("pbar", Fr(1, 3), "gamma", False)],
              [("u", Fr(741994, 100000), Fr(1, 3), True), ("pbar", Fr(698553, 100000), Fr(1, 2), True)], vp[0])
    ok &= run("E", 1500, 10, Fr(9, 10000),
              [("pbar", Fr(1, 3), Fr(790643, 100000), True), ("u", Fr(1, 3), Fr(722772, 100000), True),
               ("pbar", Fr(1, 2), Fr(706442, 100000), True)],
              [("pbar", Fr(790642, 100000), Fr(1, 3), True), ("u", Fr(722771, 100000), Fr(1, 3), True),
               ("pbar", Fr(706441, 100000), Fr(1, 2), True)], vp[0])
    ok &= run("B", 1500, 8, Fr(87373, 10 ** 8),
              [("u", Fr(1, 3), Fr(783140, 100000), True), ("pbar", Fr(1, 2), Fr(740433, 100000), True),
               ("pbar", Fr(1, 3), "gamma", False)],
              [("u", Fr(783139, 100000), Fr(1, 3), True), ("pbar", Fr(740432, 100000), Fr(1, 2), True)], vp[0])
    print(f"ALL CLAIMS CERTIFIED (fixed-point): {ok}   total {time.time()-t0:.1f}s")
