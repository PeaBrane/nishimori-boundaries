"""Design P = H4(1000,10): rigorous enclosures of v_H(gamma(p)) on the Nishimori line, used for input
(C8) of Appendix C.7 (Theorem 9.1(e), thm:lat-main) and for the reach of the criterion v_H(gamma(p)) < w_+.

Method (trust base: Python integers and fractions.Fraction; every decision in parts 1-4 is exact,
floats are used only to print widths):
- At beta = gamma(p), tanh(beta) = 1 - 2p =: th is rational, so the unit magnitudes m_c, the class
  probabilities P_c and the sign means rho_c of claims.json `law_of_K_H` are exact rationals.
- Part 1 checks the Nishimori identities rho_c = m_c exactly at each p used.
- v_H(gamma(p)) = sum_n w_n sqrt(1 - x_n^2) (sech form, valid by part 1), and also the signed form
  sum_n w_n [(1+A_n)/2 sqrt((1-x_n)/(1+x_n)) + (1-A_n)/2 sqrt((1+x_n)/(1-x_n))], are enclosed with
  outward-rounded integer interval arithmetic at scale 10^-D.
- Part 2 cross-validates against the certified claims.json display enclosures (Arb) at 8 values of p.
- Parts 3-4: exact threshold checks. P(w) = 9w^4/(1-9w^2)^2; 2P(w) < 1 with 9w^2 < 1 iff w < w_+.
- Part 5 (float, mpmath): crossings of p -> v_H(gamma(p)) with w_+ and with the HM plus-state root.
The law formulas are transcribed from claims.json `law_of_K_H` (checked equal to a separate
derivation by ../recheck-1/). Run:
  python nishimori_reach.py > nishimori_reach.out
"""
import json
import os
import time
from fractions import Fraction as Fr
from math import comb, isqrt

T0 = time.time()
B0, K = 1000, 10
D = 70
S = 10 ** D
W_STAR = Fr(274797, 10 ** 6)
HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS = os.environ.get("CLAIMS_JSON", os.path.join(HERE, "..", "out", "claims.json"))


# ---------- outward-rounded integer intervals [lo/S, hi/S] ----------
def cdiv(a, b):  # ceil(a/b) for b > 0
    return -((-a) // b)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        assert lo <= hi
        self.lo, self.hi = lo, hi

    @staticmethod
    def of(q):
        q = Fr(q)
        return I((q.numerator * S) // q.denominator, cdiv(q.numerator * S, q.denominator))

    def __add__(a, b):
        return I(a.lo + b.lo, a.hi + b.hi)

    def __sub__(a, b):
        return I(a.lo - b.hi, a.hi - b.lo)

    def __mul__(a, b):  # nonnegative operands only
        assert a.lo >= 0 and b.lo >= 0
        return I((a.lo * b.lo) // S, cdiv(a.hi * b.hi, S))

    def __truediv__(a, b):  # a >= 0, b > 0
        assert a.lo >= 0 and b.lo > 0
        return I((a.lo * S) // b.hi, cdiv(a.hi * S, b.lo))

    def sqrt(a):
        assert a.lo >= 0
        lo = isqrt(a.lo * S)
        n = a.hi * S
        hi = isqrt(n)
        if hi * hi < n:
            hi += 1
        return I(lo, hi)

    def pow(a, n):
        r = I(S, S)
        for _ in range(n):
            r = r * a
        return r

    def frs(a):
        return Fr(a.lo, S), Fr(a.hi, S)


ONE = I(S, S)
HALF = I.of(Fr(1, 2))


# ---------- exact law at the Nishimori temperature ----------
def exact_law(p):
    p = Fr(p)
    th = 1 - 2 * p  # tanh(gamma(p)) and also 1 - 2p in the class probabilities
    T = th ** B0
    tau = 2 * th ** 2 * T / (1 + T * T)
    m = (th, (th + tau) / (1 + th * tau), (th - tau) / (1 - th * tau))
    q = (1 - T) / 2
    a = (1 + th ** 3) / 2
    P1 = 2 * q * (1 - q)
    P3 = (1 - q) ** 2 * (1 - a) + q ** 2 * a
    P2 = 1 - P1 - P3
    cp, cm = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)
    rho = (1 - 2 * p,
           ((1 - q) ** 2 * ((1 - p) * cp - p * cm) + q ** 2 * ((1 - p) * cm - p * cp)) / P2,
           ((1 - q) ** 2 * ((1 - p) * cm - p * cp) + q ** 2 * ((1 - p) * cp - p * cm)) / P3)
    assert 0 < tau < th, "tau < theta (so m_3 > 0)"
    return th, m, (P1, P2, P3), rho


def enclose_v(p):
    th, m, Pc, rho = exact_law(p)
    nish = [rho[c] == m[c] for c in range(3)]
    thI = I.of(th)
    mI = [I.of(x) for x in m]
    PI = [I.of(x) for x in Pc]
    rI = [I.of(x) for x in rho]
    sech = I(0, 0)
    signed = I(0, 0)
    wsum = I(0, 0)
    for n1 in range(K + 1):
        for n2 in range(K + 1 - n1):
            n3 = K - n1 - n2
            M = comb(K, n1) * comb(K - n1, n2)
            w = I(M * S, M * S) * PI[0].pow(n1) * PI[1].pow(n2) * PI[2].pow(n3)
            x = thI * thI * mI[0].pow(n1) * mI[1].pow(n2) * mI[2].pow(n3)
            A = thI * thI * rI[0].pow(n1) * rI[1].pow(n2) * rI[2].pow(n3)
            sech = sech + w * (ONE - x * x).sqrt()
            em = ((ONE - x) / (ONE + x)).sqrt()  # e^{-k}, k = atanh x
            ep = ((ONE + x) / (ONE - x)).sqrt()  # e^{+k}
            signed = signed + w * ((ONE + A) * HALF * em + (ONE - A) * HALF * ep)
            wsum = wsum + w
    return nish, sech, signed, wsum


def P(w):
    return 9 * w ** 4 / (1 - 9 * w * w) ** 2


def below_wplus(w):  # w < w_+ = (9+3 sqrt 2)^(-1/2), exactly
    return w > 0 and 9 * w * w < 1 and 2 * P(w) < 1


def above_wplus(w):  # w > w_+ (P increasing on (0,1/3))
    return 9 * w * w < 1 and 2 * P(w) > 1


def fstr(q, nd=18):
    # exact decimal truncation of a nonnegative Fraction to nd digits
    s = (q.numerator * 10 ** nd) // q.denominator
    ip, fp = divmod(s, 10 ** nd)
    return "%d.%0*d" % (ip, nd, fp)


print("design P = H4(%d,%d); interval scale 10^-%d" % (B0, K, D))
claims = json.load(open(CLAIMS))
disp = claims["nishimori_line"]["display"]

print("\n(1)+(2) Nishimori identities and cross-validation against claims.json display enclosures")
res = {}
for e in disp:
    p = Fr(e["p"])
    nish, sech, signed, wsum = enclose_v(p)
    slo, shi = sech.frs()
    glo, ghi = signed.frs()
    clo, chi = Fr(e["v_lower"]), Fr(e["v_upper"])
    overlap = max(slo, glo, clo) <= min(shi, ghi, chi)
    inside = clo <= slo and shi <= chi
    res[p] = (slo, shi)
    print("  p = %-14s rho_c == m_c exact: %s  sech [%s, %s]  width %.1e  signed/sech/claims overlap: %s  sech inside claims: %s"
          % (e["p"], nish, fstr(slo), fstr(shi), float(shi - slo), overlap, inside))
    assert all(nish) and overlap

print("\n(3) input (C8) at p_2 = 3/2500 from the claims.json display value (exact)")
V2 = Fr(disp[-1]["v_upper"])
assert disp[-1]["p"] == "3/2500"
print("  v_upper = %s = %s" % (disp[-1]["v_upper"], fstr(V2, 15)))
print("  v_upper < W_STAR: %s;  3 v_upper < 1: %s;  2P(v_upper) < 1: %s" % (V2 < W_STAR, 3 * V2 < 1, 2 * P(V2) < 1))
print("  1 - 2P(v_upper) = %s;  >= 770991/10^6: %s" % (fstr(1 - 2 * P(V2), 12), 1 - 2 * P(V2) >= Fr(770991, 10 ** 6)))
print("  W_STAR - v_upper = %s" % fstr(W_STAR - V2, 12))

print("\n(4) the reach of the criterion v_H(gamma(p)) < w_+ (integer intervals)")
for ps, want in (("163/100000", "below"), ("1633/1000000", "below"), ("1634/1000000", "above")):
    p = Fr(ps)
    nish, sech, signed, wsum = enclose_v(p)
    slo, shi = sech.frs()
    glo, ghi = signed.frs()
    assert all(nish) and max(slo, glo) <= min(shi, ghi)
    if want == "below":
        print("  p = %-13s rho_c == m_c: %s  v_H in [%s, %s]; signed form overlaps: True" % (ps, nish, fstr(slo), fstr(shi)))
        print("      v_upper < W_STAR: %s;  v_upper < w_+ (2P < 1, 9v^2 < 1): %s;  1 - 2P(v_upper) = %s;  w_+ margin via W_STAR - v_upper = %s"
              % (shi < W_STAR, below_wplus(shi), fstr(1 - 2 * P(shi), 9), fstr(W_STAR - shi, 12) if shi < W_STAR else "n/a"))
    else:
        print("  p = %-13s rho_c == m_c: %s  v_H in [%s, %s]; signed form overlaps: True" % (ps, nish, fstr(slo), fstr(shi)))
        print("      v_lower > w_+ (2P(v_lower) > 1, 9v^2 < 1): %s;  2P(v_lower) - 1 = %s" % (above_wplus(slo), fstr(2 * P(slo) - 1, 9)))
print("  mpmath.iv upper value at 163/100000: 0.27451885576412675 (../recheck-3/nishimori_iv.out)")

print("\n(5) float evidence (mpmath, 30 digits): crossings of p -> v_H(gamma(p)) (sech form)")
import mpmath as mp  # noqa: E402  (float part only)
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.mp.dps = 30


def vf(p):
    th = 1 - 2 * p
    T = th ** B0
    tau = 2 * th ** 2 * T / (1 + T * T)
    m = (th, (th + tau) / (1 + th * tau), (th - tau) / (1 - th * tau))
    q = (1 - T) / 2
    a = (1 + th ** 3) / 2
    P1 = 2 * q * (1 - q)
    P3 = (1 - q) ** 2 * (1 - a) + q ** 2 * a
    P2 = 1 - P1 - P3
    tot = mp.mpf(0)
    for n1 in range(K + 1):
        for n2 in range(K + 1 - n1):
            n3 = K - n1 - n2
            w = comb(K, n1) * comb(K - n1, n2) * P1 ** n1 * P2 ** n2 * P3 ** n3
            x = th ** 2 * m[0] ** n1 * m[1] ** n2 * m[2] ** n3
            tot += w * mp.sqrt(1 - x * x)
    return tot


def root(target, lo=mp.mpf("0.0009"), hi=mp.mpf("0.004")):
    for _ in range(90):
        mid = (lo + hi) / 2
        if vf(mid) < target:
            lo = mid
        else:
            hi = mid
    return lo


wplus = 1 / mp.sqrt(9 + 3 * mp.sqrt(2))
vP = mp.mpf("0.2224074394521915")
# HM plus-state root: 2 P_HM(w) = 1 with P_HM(w) = P(w)(2 - 9w^2)/4 (Horiguchi-Morita form)
whm = mp.findroot(lambda w: 2 * (9 * w ** 4 / (1 - 9 * w ** 2) ** 2) * (2 - 9 * w ** 2) / 4 - 1, mp.mpf("0.297"))
print("  v_H(gamma(9/10000)) = %s (regression: cert/ value 0.2052950459896849...)" % mp.nstr(vf(mp.mpf(9) / 10000), 18))
print("  v_H(gamma(p)) = v_P   at p = %s" % mp.nstr(root(vP), 8))
print("  v_H(gamma(p)) = w_+   at p = %s   (w_+ = %s)" % (mp.nstr(root(wplus), 8), mp.nstr(wplus, 12)))
print("  v_H(gamma(p)) = W_STAR at p = %s" % mp.nstr(root(mp.mpf(274797) / 10 ** 6), 8))
print("  v_H(gamma(p)) = w_HM  at p = %s   (w_HM = %s, root of 2 P_HM = 1)" % (mp.nstr(root(whm), 8), mp.nstr(whm, 12)))
print("\nelapsed %.1f s" % (time.time() - T0))
