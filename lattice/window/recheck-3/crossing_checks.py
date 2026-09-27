"""Float and exact checks around input (C8) of Appendix C.7.
(A) float brute force of the general gauge floor E_p e^{-2sK_H(beta)} >= v_H(gamma(p)) on small gadgets (incl. non-series-parallel).
(B) float crossing of v_H(gamma(p)) with w_+ and W_STAR for design P (class form, Nishimori sech form).
(C) exact: claims.json display value at p = 3/2500 is below W_STAR, so 2P < 1 there.
(D) the running minimum w(beta) = min_{t in [0,beta]} E e^{-2tJ} on nested grids."""
import itertools, json, os
from fractions import Fraction as Fr
from math import comb
import numpy as np
import mpmath as mp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

rng = np.random.default_rng(12345)
def KH(edges, nv, poles, betaJ):
    # K_H from Z(s0,s1) by exhaustive interior sum; vertices 0..nv-1, poles=(a,b)
    inter = [v for v in range(nv) if v not in poles]
    Z = {}
    for s0 in (1, -1):
        for s1 in (1, -1):
            tot = 0.0
            for cfg in itertools.product((1, -1), repeat=len(inter)):
                sig = [0]*nv; sig[poles[0]] = s0; sig[poles[1]] = s1
                for v, c in zip(inter, cfg): sig[v] = c
                tot += np.exp(sum(bj*sig[u]*sig[v] for (u, v), bj in zip(edges, betaJ)))
            Z[(s0, s1)] = tot
    return 0.25*np.log(Z[(1, 1)]*Z[(-1, -1)]/(Z[(1, -1)]*Z[(-1, 1)]))
gadgets = {
    "wheatstone": (4, (0, 3), [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
    "K4": (4, (0, 1), [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    "grid2x3": (6, (0, 5), [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]),
    "H4(2,1)-like chain": (6, (0, 5), [(0, 1), (1, 4), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)]),
}
worst = 1e9; ncase = 0
for name, (nv, poles, edges) in gadgets.items():
    Js = list(itertools.product((1, -1), repeat=len(edges)))
    for p in (0.02, 0.1, 0.25, 0.4):
        g = 0.5*np.log((1-p)/p)
        prob = np.array([np.prod([(1-p) if j == 1 else p for j in J]) for J in Js])
        Kg = np.array([KH(edges, nv, poles, [g*j for j in J]) for J in Js])
        vH = float(np.sum(prob*np.exp(-Kg)))
        for beta in (0.1, 0.5, g, 1.3*g, 3.0):
            Kb = np.array([KH(edges, nv, poles, [beta*j for j in J]) for J in Js])
            for s in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.7):
                F = float(np.sum(prob*np.exp(-2*s*Kb)))
                worst = min(worst, F - vH); ncase += 1
print("(A) gauge floor: cases", ncase, " min(F - v_H(gamma(p))) =", worst)

# (B) design P class form at general p
b0, k = 1000, 10
def law(pf):
    qq = (1-(1-2*pf)**b0)/2; aa = (1+(1-2*pf)**3)/2
    P1 = 2*qq*(1-qq); P3 = (1-qq)**2*(1-aa) + qq**2*aa; P2 = 1-P1-P3
    ns = [(n1, n2, k-n1-n2) for n1 in range(k+1) for n2 in range(k+1-n1)]
    wn = np.array([comb(k, n1)*comb(k-n1, n2)*P1**n1*P2**n2*P3**n3 for n1, n2, n3 in ns])
    return ns, wn
def vH(pf):
    ns, wn = law(pf)
    th = 1-2*pf; T = th**b0; tau = 2*th**2*T/(1+T*T)
    m = (th, (th+tau)/(1+th*tau), (th-tau)/(1-th*tau))
    x = np.array([th**2*m[0]**a*m[1]**b*m[2]**c for a, b, c in ns])
    return float(np.sum(wn*np.sqrt(1-x**2)))  # E sech K on the Nishimori line
for pf in (9e-4, 1e-3, 1.058e-3, 1.1e-3, 1.2e-3):
    print("   v_H(gamma(%.4g)) ~ %.9f" % (pf, vH(pf)))
def root(target):
    lo, hi = 1e-3, 5e-3
    for _ in range(80):
        mid = (lo+hi)/2
        if vH(mid) < target: lo = mid
        else: hi = mid
    return lo
wplus = (9+3*np.sqrt(2))**-0.5
print("(B) float crossing v_H(gamma(p)) = w_+ at p ~ %.7g; = W_STAR at p ~ %.7g; = v_P at p ~ %.7g" % (root(wplus), root(0.274797), root(0.2224074394521915)))
# (C) exact
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "claims.json")))
P = lambda w: 9*w**4/(1-9*w**2)**2
for e in d["nishimori_line"]["display"]:
    vu = Fr(e["v_upper"])
    print("(C) p=%-14s v_upper=%.12f  < W_STAR: %s  2P(v_upper)<1: %s  1-2P(v_upper) = %.6f" % (e["p"], float(vu), vu < Fr(274797, 10**6), 2*P(vu) < 1, float(1-2*P(vu))))
# (D) w(beta) = min_{t in [0,beta]} E e^{-2tJ} for +-J at p0; nested grid
pf = 9e-4
tg = np.linspace(0, 10, 20001)
f = (1-pf)*np.exp(-2*tg) + pf*np.exp(2*tg)
cm = np.minimum.accumulate(f)
print("(D) running min nonincreasing:", bool(np.all(np.diff(cm) <= 0)), " w(10) =", cm[-1], " 2sqrt(p(1-p)) =", 2*np.sqrt(pf*(1-pf)))
