"""(F) p=4 rigorous enclosures of D(u) and Gamma(u) for u in (0,q) and (q,1].
Adapted from the p=3 second implementation fgrid.py: only xi-dependent inputs changed
(v(s)=xi'(s)=2b^2 s^3, theta(s)=3b^2 s^4/2, beta=10/3, mu* parameters). The closed forms below hold for general xi.

Closed forms (derived by differentiating the 3-atom Cole-Hopf formula in eps at 0):
 u in [0,q]:  D(u) = (1/x^2)[x log F0 + x(1-x) G0/F0 - H(u)/F0] + theta(u)/2 - (1-x) theta(q)/2
              H(u) = E psi(F1(Y)),  psi(t)=t log t,  Y~N(0,v(u)),  F1(y) = E cosh^x(y + sqrt(v(q)-v(u)) g)
              Gamma(u) = E[F1'(Y)^2/F1(Y)] / (x^2 F0)
 u in [q,1]:  D(u) = (1/x) log F0 + s/2 - e^{-s/2} J(u)/F0 + theta(u)/2 - (1-x) theta(q)/2,  s = v(u)-v(q)
              J(u) = E[cosh^{x-1}(Y) K(Y)],  K(y) = E f(y + sqrt(s) g),  f = cosh * log cosh,  Y~N(0,v(q))
              Gamma(u) = e^{-s/2} E[cosh^{x-1}(Y) E k(Y+sqrt(s)g)] / F0,  k = sinh^2/cosh
Inner integrals: trapezoid on the lattice n*Delta (discrete Gaussian convolution in numpy with a Higham
bound); outer integrals: trapezoid on the same lattice; errors bounded via strip analyticity (|Im|<b<pi/2).
"""
import json
import math
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np
from mpmath import iv

import rig
from rig import ivq, up, lower, disc, Qup, flo, fhi, conv_pos, conv_signed, dot_pos, mul_pos, kernel

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

beta = Fr(10, 3)
b2 = beta * beta
x = Fr(930529, 2000000)
q = Fr(4930831, 5000000)
v = lambda s: 2 * b2 * s ** 3
th = lambda s: Fr(3, 2) * b2 * s ** 4
XF = float(x)
R_T = 14.0  # target tail parameter
TOL = 1e-19

_C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "F_const4.json")))


def _ivs(d):
    from mpmath.libmp import MPZ
    a, b = d["raw"]
    return iv.make_mpf(((a[0], MPZ(a[1]), a[2], a[3]), (b[0], MPZ(b[1]), b[2], b[3])))


F0 = _ivs(_C["F0"])
G0 = _ivs(_C["G0"])
LOGF0 = iv.log(F0)
XI = ivq(x)
LATS = {}


def lat(k):
    if k not in LATS:
        LATS[k] = rig.Lattice(k, x)
    return LATS[k]


def bchoice(s, Delta, bmax):
    b = 2 * math.pi * s / Delta
    return min(Fr(math.floor(b * 10**6), 10**6), bmax)


def est_disc(s, Delta, bmax, extra_log=0.0):
    b = float(bchoice(s, Delta, bmax))
    return math.log(2) + b * b / (2 * s) + extra_log - 2 * math.pi * b / Delta


def pick_k(errs):
    for k in range(3, 16):
        D = 2.0 ** -k
        if all(e(D) < math.log(TOL) for e in errs):
            return k
    raise RuntimeError("no Delta")


def sym(E):
    return E * iv.mpf([-1, 1])


def pos(T):
    return T * iv.mpf([0, 1])


def below(u):
    """u Fraction in (0,q)."""
    s0 = v(u)
    s1 = v(q) - v(u)
    f0, f1 = float(s0), float(s1)
    B12 = Fr(6, 5)
    BC = Fr(157, 100)
    ki = pick_k([
        lambda D: est_disc(f1, D, BC, XF * f1 / 2),
        lambda D: est_disc(f1, D, B12, XF * f1 / 2 - math.log(0.36)),
    ])
    ko = pick_k([
        lambda D: est_disc(f0, D, B12, 2 * XF * XF * f0 + XF * f1 / 2 + math.log(8)),
        lambda D: est_disc(f0, D, B12, XF * XF * f0 / 2 + XF * f1 / 2 + math.log(40)),
    ])
    ki = max(ki, ko)
    m = 2 ** (ki - ko)
    Delta = Fr(1, 2 ** ki)   # inner lattice step
    Dout = Fr(1, 2 ** ko)    # outer node step (multiple of Delta)
    Df = float(Delta)
    J = math.ceil((XF * f1 + R_T * math.sqrt(f1)) / Df)
    K = math.ceil((2 * XF * f0 + R_T * math.sqrt(f0)) / float(Dout))
    N = K * m + J
    L = lat(ki)
    Clo, Chi = L.arr("C", N)
    Slo, Shi = L.arr("S", N)
    wlo, whi = kernel(s1, Delta, J)
    Dl = ivq(Delta)
    s0i, s1i = ivq(s0), ivq(s1)
    # ---- inner error scalars (relative to C(y) = cosh^x y)
    bC = bchoice(f1, Df, BC)
    bS = bchoice(f1, Df, B12)
    r1 = (J * Dl - XI * s1i) / iv.sqrt(s1i)
    assert lower(r1) > 0
    epsC = disc(iv.exp(ivq(bC) ** 2 / (2 * s1i)) * iv.exp(XI * s1i / 2), ivq(bC), Dl)
    tauC = up(4 * iv.exp(XI * XI * s1i / 2) * Qup(r1))
    cbS = iv.cos(ivq(bS))
    epsS = disc(iv.exp(ivq(bS) ** 2 / (2 * s1i)) * XI * iv.exp(XI * s1i / 2) / cbS, ivq(bS), Dl)
    tauS = up(4 * XI * iv.exp(XI * XI * s1i / 2) * Qup(r1))
    eC, tC, eS = fhi(epsC), fhi(tauC), fhi(epsS + tauS)
    # ---- inner sums on the fine lattice, then subsample every m-th node (outer nodes k*Dout)
    F1lo, F1hi = conv_pos(Clo, Chi, wlo, whi)
    Pl, Ph = conv_signed(Slo, Shi, wlo, whi)
    F1lo, F1hi, Pl, Ph = F1lo[::m], F1hi[::m], Pl[::m], Ph[::m]
    Cn = Chi[J:J + 2 * K * m + 1][::m]
    assert len(F1lo) == 2 * K + 1 == len(Cn)
    F1lo = np.nextafter(F1lo - np.nextafter(eC * Cn, np.inf), -np.inf)
    F1hi = np.nextafter(F1hi + np.nextafter((eC + tC) * Cn, np.inf), np.inf)
    ePn = np.nextafter(eS * Cn, np.inf)
    Pl = np.nextafter(Pl - ePn, -np.inf)
    Ph = np.nextafter(Ph + ePn, np.inf)
    Dl = ivq(Dout)
    Df = float(Dout)
    # ---- outer node values (iv)
    GHlo = np.empty(2 * K + 1); GHhi = np.empty(2 * K + 1)
    GGlo = np.empty(2 * K + 1); GGhi = np.empty(2 * K + 1)
    for i in range(2 * K + 1):
        F = iv.mpf([float(F1lo[i]), float(F1hi[i])])
        a, b = float(Pl[i]), float(Ph[i])
        if a <= 0 <= b:
            P2 = iv.mpf([0, 1]) * (iv.mpf(max(abs(a), abs(b))) ** 2)
        else:
            mn = min(abs(a), abs(b)); mx = max(abs(a), abs(b))
            P2 = iv.mpf([mn, mx]) ** 2
        gh = F * iv.log(F)
        gg = P2 / F
        GHlo[i] = max(flo(gh), 0.0); GHhi[i] = fhi(gh)
        GGlo[i] = max(flo(gg), 0.0); GGhi[i] = fhi(gg)
    olo, ohi = kernel(s0, Dout, K)
    Hs = dot_pos(olo, ohi, GHlo, GHhi)
    Gs = dot_pos(olo, ohi, GGlo, GGhi)
    # ---- outer error bounds
    bo = ivq(bchoice(f0, Df, B12))
    cb = iv.cos(bo)
    a1 = XI * s1i / 2
    cprime = -iv.log(iv.cos(XI * bo)) - XI * iv.log(cb)
    AH = iv.exp(a1) * (1 + a1 + cprime + XI * bo)
    AH0 = iv.exp(a1) * (1 + a1)
    ro = (K * Dl - 2 * XI * s0i) / iv.sqrt(s0i)
    assert lower(ro) > 0
    EH = disc(iv.exp(bo * bo / (2 * s0i)) * AH * 2 * iv.exp(2 * XI * XI * s0i), bo, Dl)
    TH = up(2 * AH0 * iv.exp(2 * XI * XI * s0i) * Qup(ro))
    AG = XI * XI * iv.exp(a1) / (cb * cb * iv.cos(XI * bo) * iv.exp(XI * iv.log(cb)))
    EG = disc(iv.exp(bo * bo / (2 * s0i)) * AG * 2 * iv.exp(XI * XI * s0i / 2), bo, Dl)
    TG = up(2 * XI * XI * iv.exp(a1) * iv.exp(XI * XI * s0i / 2) * Qup(ro))
    H = iv.mpf([Hs[0], Hs[1]]) + sym(EH) + pos(TH)
    Gsum = iv.mpf([Gs[0], Gs[1]]) + sym(EG) + pos(TG)
    D = (XI * LOGF0 + XI * (1 - XI) * G0 / F0 - H / F0) / (XI * XI) + ivq(th(u) / 2 - (1 - x) * th(q) / 2)
    G = Gsum / (XI * XI * F0)
    return D, G, dict(ki=ki, ko=ko, J=J, K=K, eC=eC, tC=tC, eS=eS, EH=fhi(EH), TH=fhi(TH), EG=fhi(EG), TG=fhi(TG))


def above(u):
    """u Fraction in (q,1]."""
    s = v(u) - v(q)
    sY = v(q)
    fs, fY = float(s), float(sY)
    B12 = Fr(6, 5)
    ki = pick_k([
        lambda D: est_disc(fs, D, B12, 2 * fs + math.log(8) + 2 * fY + 3 * math.log(2)),
        lambda D: est_disc(fs, D, B12, fs / 2 - 2 * math.log(0.36) + math.log(2)),
    ])
    ko = pick_k([
        lambda D: est_disc(fY, D, B12, 2 * fs + 2 * fY + math.log(40)),
        lambda D: est_disc(fY, D, B12, fs / 2 + XF * XF * fY / 2 + math.log(80)),
    ])
    ki = max(ki, ko)
    m = 2 ** (ki - ko)
    Delta = Fr(1, 2 ** ki)
    Dout = Fr(1, 2 ** ko)
    Df = float(Delta)
    J = math.ceil((2 * fs + R_T * math.sqrt(fs)) / Df)
    K = math.ceil((2 * fY + R_T * math.sqrt(fY)) / float(Dout))
    N = K * m + J
    L = lat(ki)
    Lo = lat(ko)
    flo_, fhi_ = L.arr("f", N)
    klo, khi = L.arr("k", N)
    elo, ehi = Lo.arr("e1", K)
    cxlo, cxhi = Lo.arr("cx1", K)
    wlo, whi = kernel(s, Delta, J)
    Dl = ivq(Delta)
    si, sYi = ivq(s), ivq(sY)
    b = ivq(bchoice(fs, Df, B12))
    cb = iv.cos(b)
    lb = -iv.log(cb) + b
    r2 = (J * Dl - 2 * si) / iv.sqrt(si)
    assert lower(r2) > 0
    # inner f: disc <= e^{b^2/2s}(1+lb) 2 e^{2s} e^{2|y|} * 2/(e^{2 pi b/D}-1); tail <= 2 e^{2|y|} e^{2s} Q(r2)
    ef = disc(iv.exp(b * b / (2 * si)) * (1 + lb) * 2 * iv.exp(2 * si), b, Dl)
    tf = up(2 * iv.exp(2 * si) * Qup(r2))
    # inner k: disc <= e^{b^2/2s} cosh(y) e^{s/2}/cos^2 b ...; tail <= 2 e^{|y|} e^{s/2} Q(r1'), r1'=(J D - s)/sqrt s >= r2
    ek = disc(iv.exp(b * b / (2 * si)) * iv.exp(si / 2) / (cb * cb), b, Dl)
    tk = up(2 * iv.exp(si / 2) * Qup(r2))
    Kflo, Kfhi = conv_pos(flo_, fhi_, wlo, whi)
    Kklo, Kkhi = conv_pos(klo, khi, wlo, whi)
    Kflo, Kfhi, Kklo, Kkhi = Kflo[::m], Kfhi[::m], Kklo[::m], Kkhi[::m]
    assert len(Kflo) == 2 * K + 1
    e2 = np.nextafter(ehi * ehi, np.inf)
    Kflo = np.nextafter(Kflo - np.nextafter(fhi(ef) * e2, np.inf), -np.inf)
    Kfhi = np.nextafter(Kfhi + np.nextafter(fhi(ef + tf) * e2, np.inf), np.inf)
    Kklo = np.nextafter(Kklo - np.nextafter(fhi(ek) * ehi, np.inf), -np.inf)
    Kkhi = np.nextafter(Kkhi + np.nextafter(fhi(ek + tk) * ehi, np.inf), np.inf)
    Kflo = np.maximum(Kflo, 0.0); Kklo = np.maximum(Kklo, 0.0)
    Dl = ivq(Dout)
    Df = float(Dout)
    olo, ohi = kernel(sY, Dout, K)
    wlo2, whi2 = mul_pos(olo, ohi, cxlo, cxhi)
    Js = dot_pos(wlo2, whi2, Kflo, Kfhi)
    Gs = dot_pos(wlo2, whi2, Kklo, Kkhi)
    # outer bounds
    bo = ivq(bchoice(fY, Df, B12))
    cbo = iv.cos(bo)
    lbo = -iv.log(cbo) + bo
    ro = (K * Dl - 2 * sYi) / iv.sqrt(sYi)
    assert lower(ro) > 0
    AJ = iv.exp((XI - 1) * iv.log(cbo)) * (1 + lbo) * 2 * iv.exp(2 * si)
    EJ = disc(iv.exp(bo * bo / (2 * sYi)) * AJ * 2 * iv.exp(2 * sYi), bo, Dl)
    TJ = up(2 * 2 * iv.exp(2 * si) * iv.exp(2 * sYi) * Qup(ro))
    AG = iv.exp((XI - 3) * iv.log(cbo)) * iv.exp(si / 2)
    EG = disc(iv.exp(bo * bo / (2 * sYi)) * AG * 2 * iv.exp(XI * XI * sYi / 2), bo, Dl)
    TG = up(2 * iv.exp(si / 2) * iv.exp(XI * XI * sYi / 2) * Qup(ro))
    Jv = iv.mpf([Js[0], Js[1]]) + sym(EJ) + pos(TJ)
    Gv = iv.mpf([Gs[0], Gs[1]]) + sym(EG) + pos(TG)
    es = iv.exp(-si / 2)
    D = LOGF0 / XI + si / 2 - es * Jv / F0 + ivq(th(u) / 2 - (1 - x) * th(q) / 2)
    G = es * Gv / F0
    return D, G, dict(ki=ki, ko=ko, J=J, K=K, ef=fhi(ef), tf=fhi(tf), ek=fhi(ek), tk=fhi(tk), EJ=fhi(EJ), TJ=fhi(TJ), EG=fhi(EG), TG=fhi(TG))


def evaluate(u):
    u = Fr(u)
    t = time.time()
    if u == 0:
        D = _ivs(_C["D0"]); G = iv.mpf(0); info = {"closed": "u=0"}
    elif u == q:
        D = _ivs(_C["Dq"]); G = _ivs(_C["Gq"]); info = {"closed": "u=q"}
    elif u < q:
        D, G, info = below(u)
    else:
        D, G, info = above(u)
    return dict(u=str(u), uf=float(u), D=[flo(D), fhi(D)], G=[flo(G), fhi(G)], info=info, secs=time.time() - t)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        r = evaluate(Fr(u))
        print(r)
