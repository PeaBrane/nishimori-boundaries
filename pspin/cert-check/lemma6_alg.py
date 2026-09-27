#!/usr/bin/env python3
"""Exact re-checks of the algebra in the last step of Lemma 6 of Zhou (2024) (Remark 7.4, rem:nm-chen)
and of Shafer's inequality. Pure sympy, exact rational arithmetic; no floats decide anything.

A1  G1'(t) = (sqrt(1-t^2) B + A) / (2 t^2 (1-t^2)(t^4-3t^2+3)^2), checked by clearing
    denominators and reducing modulo r^2 - (1 - t^2) with r = sqrt(1-t^2) (no simplify()).
A2  A^2 - (1-t^2) B^2 = t^2 (t^4-3t^2+3)^2 p(t^2) (polynomial identity).
A3  Sturm sequences: number of roots in (0,1] of p(s), A(s), B(s), with s = t^2.
A4  Shafer: d/dx[atan x - 3x/(1+2 sqrt(1+x^2))] = (s-1)^2 / (s^2 (1+2s)^2), s = sqrt(1+x^2).
A5  den(t) = 1 + 2(1-t^2) + 3/(1+2/r) has den'(t) < 0 on (0,1) (exact form of den').
A6  Zhou's (14) with 2p/(p-1) replaced by 3 is G1 >= 0 (x = artanh t), and 2p/(p-1) <= 3 for p >= 3.
"""

import json

import sympy as sp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

t, r, s, x, P = sp.symbols("t r s x P", positive=True)
out = {}

# A1: G1' with r = sqrt(1 - t^2) treated as a symbol, dr/dt = -t/r
A = 5 * t**10 - 6 * t**8 - 51 * t**6 + 117 * t**4 - 90 * t**2 + 27
B = 36 * t**6 - 99 * t**4 + 81 * t**2 - 27
den = 1 + 2 * r**2 + 3 / (1 + 2 / r)          # 1 - t^2 = r^2
h = 3 * t / den
dh = sp.diff(h, t) + sp.diff(h, r) * (-t / r)
G1p = 1 / r**2 - dh                            # artanh'(t) = 1/(1-t^2)
claimed = (r * B + A) / (2 * t**2 * r**2 * (t**4 - 3 * t**2 + 3) ** 2)
num, dnm = sp.fraction(sp.together(G1p - claimed))
num = sp.expand(num)
red = sp.expand(sp.rem(sp.Poly(num, r), sp.Poly(r**2 - (1 - t**2), r)).as_expr())
out["A1 G1' formula (reduction mod r^2-(1-t^2))"] = "PASS (exact)" if red == 0 else f"FAIL {sp.factor(red)}"

# A2
lhs = sp.expand(A**2 - (1 - t**2) * B**2)
ps = 25 * s**5 + 90 * s**4 - 309 * s**3 + 324 * s**2 - 153 * s + 27
rhs = sp.expand(t**2 * (t**4 - 3 * t**2 + 3) ** 2 * ps.subs(s, t**2))
out["A2 resultant identity"] = "PASS (exact)" if sp.expand(lhs - rhs) == 0 else "FAIL"


def sturm_count(poly, a, b):
    seq = sp.sturm(sp.Poly(poly, s))
    def v(pt):
        vals = [q.eval(pt) for q in seq]
        vals = [w for w in vals if w != 0]
        return sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
    return v(a) - v(b)   # roots in (a, b]


As = sp.expand(A.subs(t, sp.sqrt(s)))
Bs = sp.expand(B.subs(t, sp.sqrt(s)))
out["A3 Sturm roots in (0,1]: p, A, B"] = [sturm_count(ps, 0, 1), sturm_count(As, 0, 1), sturm_count(Bs, 0, 1)]
out["A3 values at s=0 and s=1: p, A, B"] = [[int(ps.subs(s, 0)), int(ps.subs(s, 1))],
                                            [int(As.subs(s, 0)), int(As.subs(s, 1))],
                                            [int(Bs.subs(s, 0)), int(Bs.subs(s, 1))]]
out["A3 sign p at 0.65^2,0.66^2,0.94^2,0.95^2"] = [int(sp.sign(ps.subs(s, sp.Rational(k, 100) ** 2))) for k in (65, 66, 94, 95)]
out["A3 Sturm roots of p in (0.65^2,0.66^2] and (0.94^2,0.95^2]"] = [
    sturm_count(ps, sp.Rational(65, 100) ** 2, sp.Rational(66, 100) ** 2),
    sturm_count(ps, sp.Rational(94, 100) ** 2, sp.Rational(95, 100) ** 2)]

# A4 Shafer
S_ = sp.sqrt(1 + x**2)
f = sp.atan(x) - 3 * x / (1 + 2 * S_)
d = sp.diff(f, x)
target = (S_ - 1) ** 2 / (S_**2 * (1 + 2 * S_) ** 2)
chk = sp.simplify(sp.radsimp(d - target))
# independent: numerator after clearing, with S as a symbol and S^2 = 1 + x^2
Ssym = sp.Symbol("S", positive=True)
dS = 1 / (1 + x**2) - 3 * sp.diff(x / (1 + 2 * Ssym), x) - 3 * sp.diff(x / (1 + 2 * Ssym), Ssym) * (x / Ssym)
n2, _ = sp.fraction(sp.together(dS - (Ssym - 1) ** 2 / (Ssym**2 * (1 + 2 * Ssym) ** 2)))
n2 = sp.expand(n2.subs(x**2, Ssym**2 - 1))
n2 = sp.expand(sp.rem(sp.Poly(sp.expand(n2), x), sp.Poly(x**2 - (Ssym**2 - 1), x)).as_expr())
out["A4 Shafer derivative identity"] = {"simplify": str(chk), "reduction": "PASS (exact)" if n2 == 0 else f"FAIL {n2}"}

# A5 den'(t) with r = sqrt(1-t^2): den' = -4t - 6t / (r^3 (1+2/r)^2) < 0 for t in (0,1)
dden = sp.diff(den, r) * (-t / r)
dden = dden + sp.diff(den, t)
claim = -4 * t - 6 * t / (r**3 * (1 + 2 / r) ** 2)
out["A5 den' closed form"] = "PASS (exact)" if sp.simplify(sp.together(dden - claim)) == 0 else f"FAIL {sp.simplify(dden - claim)}"

# A6 (14) at general p: 2P/(P-1) <= 3 iff P >= 3 (P > 1)
out["A6 3 - 2P/(P-1) = (P-3)/(P-1)"] = "PASS (exact)" if sp.simplify(3 - 2 * P / (P - 1) - (P - 3) / (P - 1)) == 0 else "FAIL"

print(json.dumps(out, indent=1))
