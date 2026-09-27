"""Checks of the final (elementary) part of Lemma 6 of Zhou (2024) (T convex; Remark 7.4, rem:nm-chen), and
float checks of its first part. Writes zc_lemma6.json next to this file.

Exact (sympy, rational arithmetic):
  E1. G1'(t) equals Zhou's displayed formula (sqrt(1-t^2) B + A) / (2 t^2 (1-t^2)(t^4-3t^2+3)^2)
      with G1(t) = artanh t - 3t / (1 + 2(1-t^2) + 3/(1 + 2/sqrt(1-t^2))).
  E2. A^2 - (1-t^2) B^2 = t^2 (t^4-3t^2+3)^2 (25t^10+90t^8-309t^6+324t^4-153t^2+27).
  E3. p(s) = 25s^5+90s^4-309s^3+324s^2-153s+27 has exactly 2 roots in [0,1] (exact root count),
      and the sign claims at s = 0.65^2, 0.66^2, 0.94^2, 0.95^2.
  E4. (x = artanh t substitution) the RHS of Zhou (14) at p = 3 equals 3t/(...) above.
Interval (mpmath.iv, directed rounding):
  I1. artanh(0.94) - 3*0.95/(1 + 2(1-0.95^2) + 3/(1 + 2/sqrt(1-0.95^2))) > 0.
Float (numpy Gauss-Hermite):
  F1. T''(u) from Zhou's closed form (p-1) a1 Y^2 G0 / (u (a1 - a_{-1})^3) vs finite differences.
  F2. Zhou's rewrite of G0 vs its first form.
"""

import json
import pathlib

import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.hermite_e import hermegauss
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

out = {}
t = sp.Symbol("t", positive=True)
r = sp.sqrt(1 - t**2)
G1 = sp.atanh(t) - 3 * t / (1 + 2 * (1 - t**2) + 3 / (1 + 2 / r))
A = 5 * t**10 - 6 * t**8 - 51 * t**6 + 117 * t**4 - 90 * t**2 + 27
B = 36 * t**6 - 99 * t**4 + 81 * t**2 - 27
claimed = (r * B + A) / (2 * t**2 * (1 - t**2) * (t**4 - 3 * t**2 + 3) ** 2)
d = sp.diff(G1, t)
diffE1 = sp.simplify(sp.radsimp(d - claimed))
if diffE1 != 0:
    # try numeric spot checks at rational points as a fallback diagnostic
    vals = [sp.N((d - claimed).subs(t, sp.Rational(k, 10)), 30) for k in range(1, 10)]
    out["E1 G1' formula"] = f"symbolic simplify != 0; spot values {[str(v) for v in vals]}"
else:
    out["E1 G1' formula"] = "PASS (exact)"
lhs = sp.expand(A**2 - (1 - t**2) * B**2)
rhs = sp.expand(t**2 * (t**4 - 3 * t**2 + 3) ** 2 * (25 * t**10 + 90 * t**8 - 309 * t**6 + 324 * t**4 - 153 * t**2 + 27))
out["E2 resultant identity"] = "PASS (exact)" if sp.expand(lhs - rhs) == 0 else f"FAIL: {sp.factor(lhs - rhs)}"
s = sp.Symbol("s")
ps = sp.Poly(25 * s**5 + 90 * s**4 - 309 * s**3 + 324 * s**2 - 153 * s + 27, s)
out["E3 roots of p in [0,1] (exact count)"] = int(ps.count_roots(0, 1))
out["E3 roots of p in (0,1) intervals (exact isolation)"] = [[str(a), str(b)] for (a, b), _ in ps.intervals() if 0 <= a <= 1]
signs = {}
for tv in ("0.65", "0.66", "0.94", "0.95"):
    sv = sp.Rational(tv) ** 2
    signs[tv] = int(sp.sign(ps.eval(sv)))
out["E3 sign p(t^2) at t=0.65,0.66,0.94,0.95 (claimed +,-,-,+)"] = signs
# E4: with x = artanh t: cosh^-2 x = 1-t^2, cosh x = 1/sqrt(1-t^2), x/tanh x = artanh(t)/t.
x = sp.atanh(t)
rhs14 = 2 * 3 / ((3 - 1) * (1 + 2 / sp.cosh(x) ** 2 + 3 / (1 + 2 * sp.cosh(x))))
out["E4 (14) at p=3 <=> G1>=0"] = "PASS (exact)" if sp.simplify(sp.simplify((rhs14 * t) - 3 * t / (1 + 2 * (1 - t**2) + 3 / (1 + 2 / r)))) == 0 else "CHECK"

# E5: Gaussian IBP step before (13): with F(x) = sinh x + 2 sinh x cosh^-2 x + atan(sinh x),
# F'(x) = tanh^2 x cosh x + 4 cosh^-3 x, so Y E[tanh^2 cosh + 4 cosh^-3](Yg) = E[g F(Yg)] and
# Y^2 E[...] = E[(Yg) F(Yg)]; (13) must be read with x multiplying the whole bracket.
xx = sp.Symbol("x", real=True)
F = sp.sinh(xx) + 2 * sp.sinh(xx) / sp.cosh(xx) ** 2 + sp.atan(sp.sinh(xx))
e5 = sp.simplify((sp.diff(F, xx) - (sp.tanh(xx) ** 2 * sp.cosh(xx) + 4 / sp.cosh(xx) ** 3)).rewrite(sp.exp))
out["E5 F' = tanh^2 cosh + 4 cosh^-3"] = "PASS (exact)" if e5 == 0 else f"FAIL {e5}"
# E6: (sinh cosh^-4)' = -3 cosh^-3 + 4 cosh^-5 (so (1/Y) E[g sinh cosh^-4] = 4a_-5 - 3a_-3 > 0)
e6 = sp.simplify((sp.diff(sp.sinh(xx) / sp.cosh(xx) ** 4, xx) - (-3 / sp.cosh(xx) ** 3 + 4 / sp.cosh(xx) ** 5)).rewrite(sp.exp))
out["E6 (sinh cosh^-4)' identity"] = "PASS (exact)" if e6 == 0 else f"FAIL {e6}"

# E7: Zhou's T'' = (p-1) a1 Y^2 G0 / (u (a1 - a_-1)^3), symbolically, from T = u a1/(a1 - a_-1) - 1,
# Y = sqrt(xi_Z'(u)) (dY/du = (p-1) Y/(2u)), and the Gaussian IBP rule
# d a_k/dY = k^2 Y a_k - k(k-1) Y a_{k-2}  (a_k = E cosh^k(Yg); rule itself: Gaussian integration by parts, checked in E8).
uu, pp_ = sp.symbols("u p", positive=True)
Yf = sp.Function("Y")(uu)
ak = {k: sp.Function(f"a{k}".replace("-", "m"))(uu) for k in (1, -1, -3, -5)}
dY = (pp_ - 1) * Yf / (2 * uu)
rule = {1: Yf * ak[1], -1: Yf * ak[-1] - 2 * Yf * ak[-3], -3: 9 * Yf * ak[-3] - 12 * Yf * ak[-5]}


def D_u(expr):
    e = sp.diff(expr, uu)
    e = e.subs(sp.Derivative(Yf, uu), dY)
    for k, rk in rule.items():
        e = e.subs(sp.Derivative(ak[k], uu), rk * dY)
    return e


Tsym = uu * ak[1] / (ak[1] - ak[-1]) - 1
Tpp = D_u(D_u(Tsym))
a1, am1, am3, am5, Ys = ak[1], ak[-1], ak[-3], ak[-5], Yf
G0sym = (-pp_ * a1 * am3 + pp_ * am1 * am3 + 2 * (pp_ - 1) * Ys**2 * am3**2 + 4 * (pp_ - 1) * Ys**2 * am1 * am3
         - 4 * (pp_ - 1) * Ys**2 * a1 * am3 + 6 * (pp_ - 1) * Ys**2 * a1 * am5 - 6 * (pp_ - 1) * Ys**2 * am1 * am5)
e7 = sp.simplify(sp.together(Tpp - (pp_ - 1) * a1 * Ys**2 * G0sym / (uu * (a1 - am1) ** 3)))
out["E7 T'' closed form (Zhou (11)-G0)"] = "PASS (exact)" if e7 == 0 else f"FAIL residual {sp.factor(e7)}"
# E8: IBP rule check at the integrand level: d/dY cosh^k(Yx) = k x cosh^(k-1) sinh, and
# E[g f(Yg)] = Y E[f'(Yg)] turns it into k^2 Y a_k - k(k-1) Y a_{k-2}:
kk_ = sp.Symbol("k")
f_ = kk_ * sp.cosh(xx) ** (kk_ - 1) * sp.sinh(xx)
e8 = sp.simplify((sp.diff(f_, xx) - (kk_**2 * sp.cosh(xx) ** kk_ - kk_ * (kk_ - 1) * sp.cosh(xx) ** (kk_ - 2))).rewrite(sp.exp))
out["E8 IBP rule integrand (k cosh^(k-1) sinh)' = k^2 cosh^k - k(k-1) cosh^(k-2)"] = "PASS (exact)" if e8 == 0 else f"FAIL {e8}"

# E9: Zhou's rewrite of G0 with E[cosh tanh^2] = a1 - a_-1.
ct = a1 - am1
G0rw = sp.Rational(3, 2) * (pp_ - 1) * Ys**2 * ct * (4 * am5 - 3 * am3) + am3 * (
    sp.Rational(1, 2) * (pp_ - 1) * Ys**2 * ct + 2 * (pp_ - 1) * Ys**2 * am3 - pp_ * ct)
e9 = sp.expand(G0sym - G0rw)
out["E9 G0 rewrite"] = "PASS (exact)" if e9 == 0 else f"FAIL residual {sp.factor(e9)}"

iv = mp.iv
iv.dps = 30
lo = iv.mpf("0.94")
hi = iv.mpf("0.95")
val = iv.atanh(lo) if hasattr(iv, "atanh") else (iv.log((1 + lo) / (1 - lo)) / 2)
den = 1 + 2 * (1 - hi**2) + 3 / (1 + 2 / iv.sqrt(1 - hi**2))
I1 = val - 3 * hi / den
out["I1 G1(tm) lower bound interval"] = [str(I1.a), str(I1.b)]
out["I1 positive"] = bool(I1.a > 0)
# I2 (repair): the same monotone bound on the ten cells [0.94+k/1000, 0.94+(k+1)/1000] covering
# [0.94, 0.95]; since artanh and h(t) = 3t/den(t) are increasing, G1 >= artanh(a) - h(b) on [a,b].
cells = []
for kk in range(10):
    a_ = iv.mpf(94000 + 100 * kk) / 100000
    b_ = iv.mpf(94000 + 100 * (kk + 1)) / 100000
    va = iv.log((1 + a_) / (1 - a_)) / 2
    denb = 1 + 2 * (1 - b_**2) + 3 / (1 + 2 / iv.sqrt(1 - b_**2))
    lb = va - 3 * b_ / denb
    cells.append(float(mp.mpf(lb.a)))
out["I2 repaired cell lower bounds (interval, rounded down)"] = cells
out["I2 all positive"] = all(c > 0 for c in cells)

# F1/F2: float check of the T'' closed form and G0 rewrite
Z, W = hermegauss(200)
W = W / W.sum()


def a(k, Y):
    return W @ np.cosh(Y * Z) ** k


def T(u, p, bZ):
    Y = np.sqrt(p * bZ**2 * u ** (p - 1))
    return u * a(1, Y) / (W @ (np.tanh(Y * Z) ** 2 * np.cosh(Y * Z))) - 1


res = []
for p in (3, 4, 6):
    for bZ in (0.8, 1.1):
        for u in (0.3, 0.6, 0.9):
            Y = np.sqrt(p * bZ**2 * u ** (p - 1))
            a1, am1, am3, am5 = a(1, Y), a(-1, Y), a(-3, Y), a(-5, Y)
            G0 = (-p * a1 * am3 + p * am1 * am3 + 2 * (p - 1) * Y**2 * am3**2 + 4 * (p - 1) * Y**2 * am1 * am3
                  - 4 * (p - 1) * Y**2 * a1 * am3 + 6 * (p - 1) * Y**2 * a1 * am5 - 6 * (p - 1) * Y**2 * am1 * am5)
            ct = W @ (np.cosh(Y * Z) * np.tanh(Y * Z) ** 2)
            G0b = 1.5 * (p - 1) * Y**2 * ct * (4 * am5 - 3 * am3) + am3 * (0.5 * (p - 1) * Y**2 * ct + 2 * (p - 1) * Y**2 * am3 - p * ct)
            Tpp_cf = (p - 1) * a1 * Y**2 * G0 / (u * (a1 - am1) ** 3)
            h = 1e-3
            Tpp_fd = (T(u + h, p, bZ) - 2 * T(u, p, bZ) + T(u - h, p, bZ)) / h**2
            res.append({"p": p, "bZ": bZ, "u": u, "Tpp_closed": Tpp_cf, "Tpp_fd": Tpp_fd, "G0": G0, "G0_rewrite": G0b})
out["F1_F2"] = res
out["F1 max rel err"] = max(abs(q["Tpp_closed"] - q["Tpp_fd"]) / abs(q["Tpp_fd"]) for q in res)
out["F2 max abs diff"] = max(abs(q["G0"] - q["G0_rewrite"]) for q in res)
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "F1_F2"}, indent=1))
