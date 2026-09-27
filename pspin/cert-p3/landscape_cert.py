"""Cold landscape cover at beta=5/2, j0=3841/5000:
for every interval [ma,mb] in the cover, a trial tau with
  U(m) = beta j0 m^3 + P_tau(h) - h m  (convex in m)  satisfies U(ma), U(mb) < TARGET.
Trials: RS (q,h) and 1RSB with atoms {0,q} (x,q,h). Rational parameters; interval-enclosed P_tau."""
import json, os, sys
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize
from iv import *
from gauss import expect

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

beta = Fr(5, 2); j0 = Fr(3841, 5000)
B = IV.exact_rational(5, 2); J0 = IV.exact_rational(3841, 5000)
Zf, Wf = np.polynomial.hermite_e.hermegauss(200); Wf = Wf / Wf.sum()
bf, jf = float(beta), float(j0)
def lcf(y): return np.abs(y) + np.log1p(np.exp(-2*np.abs(y))) - np.log(2)
def P_rs_f(q, h):
    L = 1.5*bf*bf*q*q
    return np.log(2) + np.sum(Wf*lcf(h + np.sqrt(L)*Zf)) + 0.5*(bf*bf/2 - bf*bf*q**3/2 - (1-q)*L)
def P_1r_f(x, q, h):
    L = 1.5*bf*bf*q*q; v = x*lcf(h + np.sqrt(L)*Zf); mx = v.max()
    return np.log(2) + (mx + np.log(np.sum(Wf*np.exp(v-mx))))/x + 0.5*(bf*bf/2 - L + (1-x)*bf*bf*q**3)
def U_f(m, kind, t):
    if kind == 'RS': return bf*jf*m**3 + P_rs_f(t[1], t[0]) - t[0]*m
    return bf*jf*m**3 + P_1r_f(t[1], t[2], t[0]) - t[0]*m
def best_trial(m):
    best = None
    for h0 in np.linspace(0, 6, 7):
        for q0 in [0.6, 0.85, 0.93]:
            r = minimize(lambda t: U_f(m, 'RS', t), [h0, q0], method='L-BFGS-B', bounds=[(0, 20), (0.05, 0.999)])
            if best is None or r.fun < best[0]: best = (r.fun, 'RS', r.x)
            for x0 in [0.3, 0.6]:
                r = minimize(lambda t: U_f(m, '1R', t), [h0, x0, q0], method='L-BFGS-B',
                             bounds=[(0, 20), (0.02, 1), (0.05, 0.999)])
                if best is None or r.fun < best[0]: best = (r.fun, '1R', r.x)
    return best
def rat(v, d=10**6): return Fr(round(float(v)*d), d)
def P_iv(kind, t):
    """rigorous enclosure of P_tau(h) with rational parameters t"""
    h = IV.exact_rational(t[0].numerator, t[0].denominator)
    if kind == 'RS':
        q = IV.exact_rational(t[1].numerator, t[1].denominator)
        L = B*B*q*q*1.5; s = isqrt(L)
        E = expect(lambda z: tlogcosh(z*s + h), float(h.hi)+float(s.hi), 1.0, n=3000)
        return LOG2 + E + (B*B*0.5 - B*B*q*q*q*0.5 - (1-q)*L)*0.5
    x = IV.exact_rational(t[1].numerator, t[1].denominator); q = IV.exact_rational(t[2].numerator, t[2].denominator)
    L = B*B*q*q*1.5; s = isqrt(L); xs = float(x.hi)*float(s.hi)
    E = expect(lambda z: tpow_pos(tcosh(z*s + h), x), float(np.exp(float(x.hi)*float(h.hi))), xs, n=3000)
    return LOG2 + ilog(E)*x.recip() + (B*B*0.5 - L + (1-x)*B*B*q*q*q)*0.5
def U_iv(m, kind, t, P):
    M = IV.exact_rational(m.numerator, m.denominator)
    h = IV.exact_rational(t[0].numerator, t[0].denominator)
    return B*J0*M*M*M + P - h*M

if __name__ == "__main__":
    TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0457539532688025 - 5e-5
    ma0, mb0 = Fr(1, 4), Fr(1)
    todo = [(ma0 + (mb0-ma0)*Fr(k, 75), ma0 + (mb0-ma0)*Fr(k+1, 75)) for k in range(75)]
    cover = []
    worst = -1e9
    while todo:
        ma, mb = todo.pop(0)
        mid = (ma + mb) / 2
        fval, kind, t = best_trial(float(mid))
        t = [rat(v) for v in t]
        P = P_iv(kind, t)
        ua, ub = U_iv(ma, kind, t, P), U_iv(mb, kind, t, P)
        top = max(float(ua.hi), float(ub.hi))
        if top < TARGET:
            cover.append({"ma": str(ma), "mb": str(mb), "kind": kind, "t": [str(v) for v in t],
                          "U_hi": top, "P": [float(P.lo), float(P.hi)]})
            worst = max(worst, top)
            print(f"ok [{float(ma):.5f},{float(mb):.5f}] {kind} U_hi-TARGET={top-TARGET:+.3e}", flush=True)
        else:
            if mb - ma < Fr(1, 10**5):
                print("FAIL at", float(ma), float(mb), top - TARGET); sys.exit(1)
            todo = [(ma, mid), (mid, mb)] + todo
    print("cover intervals:", len(cover), "worst U_hi:", worst, "TARGET:", TARGET, "margin", TARGET - worst)
    json.dump({"beta": str(beta), "j0": str(j0), "TARGET": TARGET, "worst_U_hi": worst, "cover": cover},
              open(os.path.join(HERE, 'landscape_cover.json'), 'w'), indent=1)
