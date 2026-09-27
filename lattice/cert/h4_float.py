"""Float (mpmath) closed forms for the max-degree-4 gadget
    H4 = e - S_k(U) - e,   U = e || (e - (P_b0 || P_b0) - e),
iid +-J couplings with P(J=-1)=p at inverse temperature beta.

Closed form (Proposition 9.3, prop:lat-law, of the paper):
  unit U has 3 magnitude classes c=1,2,3 with |tanh K_U| = w1, t, w3 and
  probabilities P1, P2, P3 (independent of beta); signs are conditionally
  independent given the class, E[sign | c] = rho_c.  In the variables
  s = e^{-2 beta}, Z = tanh(b0 atanh s)^2 / s:
     t  = (1-s)/(1+s)
     r  = (2 + Z(1+s^2)) / (1 + s^2 + 2 s^2 Z)      (= e^{-2K_long}/s)
     w1 = (1 - r s^2) / (1 + r s^2)
     w3 = (1-s^2)(1+Z) / (3 + s^2 + Z(1+3 s^2))    (= (r-1)/(r+1))
  |tanh K_H| = t^{2+n2} w1^{n1} w3^{n3} with multinomial class counts n.
Usage: h4_float.py b0 k p   (p may be a fraction like 9/10000)
"""
import sys
from fractions import Fraction
from mpmath import mp, mpf, tanh, atanh, log, sqrt, findroot, exp

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.dps = 40


def unit_law(p, b0):
    p = mpf(p)
    t_g = 1 - 2 * p
    q = (1 - t_g ** b0) / 2
    cplus, cminus = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)
    tpos = cplus * (1 - q) ** 2 + cminus * q ** 2
    tzero = 2 * q * (1 - q)
    tneg = cplus * q ** 2 + cminus * (1 - q) ** 2
    P1 = (1 - p) * tpos + p * tneg
    P2 = tzero
    P3 = (1 - p) * tneg + p * tpos
    rho1 = ((1 - p) * tpos - p * tneg) / P1
    rho2 = 1 - 2 * p
    rho3 = ((1 - p) * tneg - p * tpos) / P3
    return dict(q=q, tpos=tpos, tzero=tzero, tneg=tneg, P=(P1, P2, P3), rho=(rho1, rho2, rho3))


def mags(beta, b0):
    s = exp(-2 * beta)
    Z = tanh(b0 * atanh(s)) ** 2 / s
    t = (1 - s) / (1 + s)
    r = (2 + Z * (1 + s * s)) / (1 + s * s + 2 * s * s * Z)
    w1 = (1 - r * s * s) / (1 + r * s * s)
    w3 = (1 - s * s) * (1 + Z) / (3 + s * s + Z * (1 + 3 * s * s))
    return t, w1, w3


def mags_direct(beta, b0):
    t = tanh(beta)
    a = t * t * tanh(2 * atanh(t ** b0))
    return t, (t + a) / (1 + t * a), (t - a) / (1 - t * a)


def classes(k):
    from math import comb
    for n1 in range(k + 1):
        for n3 in range(k + 1 - n1):
            n2 = k - n1 - n3
            yield (n1, n2, n3), comb(k, n1) * comb(k - n1, n3)


def functionals(p, b0, k, beta, direct=False, warm=False):
    L = unit_law(p, b0)
    P1, P2, P3 = L["P"]
    r1, r2, r3 = L["rho"]
    t, w1, w3 = (mags_direct if direct else mags)(beta, b0)
    uU = P1 * w1 + P2 * t + P3 * w3
    uH = t * t * uU ** k
    pbar = mpf(0)
    v = mpf(0)
    vsech = mpf(0)
    for (n1, n2, n3), M in classes(k):
        w = M * P1 ** n1 * P2 ** n2 * P3 ** n3
        x = t ** (2 + n2) * w1 ** n1 * w3 ** n3
        A = (1 - 2 * mpf(p)) ** 2 * r1 ** n1 * r2 ** n2 * r3 ** n3
        pbar += w * 2 * x / (1 + x)
        if warm:
            g = sqrt((1 - x) / (1 + x))
            v += w * ((1 + A) / 2 * g + (1 - A) / 2 / g)
            vsech += w * sqrt(1 - x * x)
    return dict(uU=uU, uH=uH, pbar=pbar, v=v, vsech=vsech)


def hP(v):
    return 36 * v ** 4 / (1 - 9 * v * v) ** 2 + 36 * v ** 3 / (1 - 9 * v * v)


if __name__ == "__main__":
    b0, k = int(sys.argv[1]), int(sys.argv[2])
    p = mpf(Fraction(sys.argv[3]).numerator) / Fraction(sys.argv[3]).denominator
    gam = log((1 - p) / p) / 2
    vP = findroot(lambda v: hP(v) - 1, mpf("0.2224"))
    L = unit_law(p, b0)
    print(f"b0={b0} k={k} p={p} gamma={mp.nstr(gam, 12)} q={mp.nstr(L['q'], 10)} P={[mp.nstr(x, 10) for x in L['P']]} vP={mp.nstr(vP, 14)}")
    F = functionals(p, b0, k, gam, warm=True)
    Fd = functionals(p, b0, k, gam, direct=True, warm=True)
    print(f"gamma: v=E e^-K={mp.nstr(F['v'], 12)} Esech={mp.nstr(F['vsech'], 12)} (direct {mp.nstr(Fd['v'], 12)}) h(v)={mp.nstr(hP(F['v']), 10)} uH={mp.nstr(F['uH'], 8)} pbar={mp.nstr(F['pbar'], 8)}")
    for m in ["1.5", "2", "2.05", "2.1", "2.15", "2.2", "2.5", "3", "4", "5", "10", "30"]:
        beta = mpf(m) * gam
        F = functionals(p, b0, k, beta)
        try:
            Fd = functionals(p, b0, k, beta, direct=True)
        except ZeroDivisionError:  # t rounds to 1: the direct form is 0/0, the (s,Z) form is not
            Fd = dict(uH=mpf("nan"), pbar=mpf("nan"))
        print(f"{m:>5} gamma beta={mp.nstr(beta, 8):>10}: uU={mp.nstr(F['uU'], 8)} uH={mp.nstr(F['uH'], 8)} pbar={mp.nstr(F['pbar'], 8)} | direct uH={mp.nstr(Fd['uH'], 8)} pbar={mp.nstr(Fd['pbar'], 8)}")
    fu = lambda b: functionals(p, b0, k, b)["uH"] - mpf(1) / 3
    fp = lambda b: functionals(p, b0, k, b)["pbar"] - mpf(1) / 2
    bu = findroot(fu, 2.1 * gam)
    bp = findroot(fp, 2.0 * gam)
    print(f"crossing u_H=1/3 at beta={mp.nstr(bu, 12)} = {mp.nstr(bu / gam, 8)} gamma")
    print(f"crossing pbar_H=1/2 at beta={mp.nstr(bp, 12)} = {mp.nstr(bp / gam, 8)} gamma")
    # limit beta -> infinity
    P1, P2, P3 = L["P"]
    uinf = (P1 + P2 + P3 / 3) ** k
    pinf = sum(M * P1 ** a * P2 ** b * P3 ** c * 2 * mpf(3) ** (-c) / (1 + mpf(3) ** (-c)) for (a, b, c), M in classes(k))
    print(f"beta->inf: uH={mp.nstr(uinf, 12)} pbar={mp.nstr(pinf, 12)}")
    # scan min of pbar and max of uH on [bu, 60 gamma]
    import mpmath
    grid = [bu + (60 * gam - bu) * (mpf(i) / 4000) ** 3 for i in range(4001)]
    vals = [(functionals(p, b0, k, b)["uH"], functionals(p, b0, k, b)["pbar"], b) for b in grid[::10]]
    print("max uH on scan beyond crossing:", mp.nstr(max(v[0] for v in vals), 10))
    mn = min(vals, key=lambda v: v[1])
    print("min pbar on scan:", mp.nstr(mn[1], 10), "at beta", mp.nstr(mn[2], 8), "=", mp.nstr(mn[2] / gam, 6), "gamma")
