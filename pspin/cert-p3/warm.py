"""Warm certificate: F^0(beta_w,0) < log2 + beta_w^2/4 at beta_w = 2 j0, via a 1RSB trial.
Delta_w = (1/x) log E cosh^x(sqrt(L) Z) - L/2 + (1-x) theta(q)/2 < 0, L = 3 b^2 q^2/2, theta = b^2 q^3."""
import os, sys, json
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize
from iv import *
from gauss import expect

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

j0 = Fr(3841, 5000)
bw = 2 * j0
def delta_float(x, q, b=float(bw)):
    Z, W = np.polynomial.hermite_e.hermegauss(300); W = W/W.sum()
    L = 1.5*b*b*q*q
    lc = np.abs(np.sqrt(L)*Z) + np.log1p(np.exp(-2*np.abs(np.sqrt(L)*Z))) - np.log(2)
    return np.log(np.sum(W*np.exp(x*lc)))/x - L/2 + (1-x)*b*b*q**3/2
r = minimize(lambda t: delta_float(t[0], t[1]), [0.99, 0.81], method='L-BFGS-B', bounds=[(0.5,1),(0.5,0.99)],
             options={'ftol':1e-18,'gtol':1e-14})
print("float optimum", r.x, r.fun)
x = Fr(round(r.x[0]*10**6), 10**6); q = Fr(round(r.x[1]*10**6), 10**6)
print("rational trial x,q =", x, q, float(delta_float(float(x), float(q))))
b = IV.exact_rational(bw.numerator, bw.denominator)
X = IV.exact_rational(x.numerator, x.denominator)
Q = IV.exact_rational(q.numerator, q.denominator)
Lam = b * b * Q * Q * 1.5
theta = b * b * Q * Q * Q
s = isqrt(Lam)
xs = float(X.hi) * float(s.hi)
for n in [2000, 8000]:
    A = expect(lambda z: tpow_pos(tcosh(z * s), X), 1.0, xs, n=n)
    D = ilog(A) * X.recip() - Lam * 0.5 + (1 - X) * theta * 0.5
    print(f"n={n} A={A} Delta_w in [{D.lo}, {D.hi}]  certified<0: {bool(D.hi < 0)}")
json.dump({"j0": str(j0), "beta_w": str(bw), "x": str(x), "q": str(q), "Delta_w_hi": float(D.hi), "Delta_w_lo": float(D.lo)},
          open(os.path.join(HERE, 'warm_result.json'),'w'), indent=1)
