"""Interval enclosure (mpmath.iv), for input (C8), of v_H(gamma(p)) = F(gamma(p), 1/2) for design P at larger p,
signed form (no use of the Nishimori identity) and sech form as a cross-check. At beta = gamma(p), theta = tanh gamma = 1 - 2p exactly.
Class law as in claims.json law_of_K_H. Checks v_upper < W_STAR = 274797/10^6 exactly (then 2P(v) < 1 and 3v < 1)."""
from fractions import Fraction as Fr
from math import comb
import mpmath as mp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.iv.dps = 60
iv = mp.iv
b0, k = 1000, 10
WSTAR = Fr(274797, 10**6)
P = lambda w: 9*w**4/(1-9*w**2)**2
def enclose(p):
    p = iv.mpf(p.numerator)/p.denominator
    th = 1 - 2*p
    q = (1 - th**b0)/2; a = (1 + th**3)/2
    P1 = 2*q*(1-q); P3 = (1-q)**2*(1-a) + q**2*a; P2 = 1 - P1 - P3
    cp = (1-p)**2 + p**2; cm = 2*p*(1-p)
    r1 = 1 - 2*p
    r2 = ((1-q)**2*((1-p)*cp - p*cm) + q**2*((1-p)*cm - p*cp))/P2
    r3 = ((1-q)**2*((1-p)*cm - p*cp) + q**2*((1-p)*cp - p*cm))/P3
    T = th**b0; tau = 2*th**2*T/(1+T*T)
    m1, m2, m3 = th, (th+tau)/(1+th*tau), (th-tau)/(1-th*tau)
    Fs = iv.mpf(0); Fsech = iv.mpf(0); wsum = iv.mpf(0)
    for n1 in range(k+1):
        for n2 in range(k+1-n1):
            n3 = k-n1-n2
            w = comb(k, n1)*comb(k-n1, n2)*P1**n1*P2**n2*P3**n3
            x = th**2*m1**n1*m2**n2*m3**n3
            A = (1-2*p)**2*r1**n1*r2**n2*r3**n3
            r = (1-x)/(1+x)
            Fs += w*((1+A)/2*iv.sqrt(r) + (1-A)/2/iv.sqrt(r))
            Fsech += w*iv.sqrt(1-x*x)
            wsum += w
    return Fs, Fsech, wsum
for p in (Fr(9, 10000), Fr(3, 2500), Fr(3, 2000), Fr(1, 625), Fr(163, 100000)):
    Fs, Fsech, wsum = enclose(p)
    mm, ee = mp.mpf(Fs.b).man_exp; hi = Fr(mm)*Fr(2)**ee
    print("p = %-12s signed F(gamma,1/2) in [%s, %s]; sech form in [%s, %s]; sum w in [%s, %s]" % (
        p, mp.nstr(Fs.a, 15), mp.nstr(Fs.b, 15), mp.nstr(Fsech.a, 15), mp.nstr(Fsech.b, 15), mp.nstr(wsum.a, 8), mp.nstr(wsum.b, 8)))
    print("    exact: v_upper < W_STAR: %s; 2P(v_upper) < 1: %s; 1-2P(v_upper) = %.6f; margin W_STAR - v_upper = %.3e" % (
        hi < WSTAR, 2*P(hi) < 1, float(1-2*P(hi)), float(WSTAR-hi)))
