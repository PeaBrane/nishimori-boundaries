"""Exact checks of the Peierls thresholds of Section 9.3 used by inputs (C3) and (C5).

P(v) = 9 v^4 / (1 - 9 v^2)^2 on (0, 1/3).
- w_plus: root of 2P(w) = 1, claimed = (9 + 3 sqrt 2)^(-1/2) = 0.274797...
- w_bb:   root of 4P(w) = 1, claimed = 15^(-1/2) = 0.258198...
- v_P:    root of h(v) = 4P + 4Q = 1 (bracket from ../../cert/cert_arb.py).
- V = 20529504599/10^11 >= v_H(gamma(9/10000)) (cert/out/cert_arb.txt): 2P(V), 4P(V), lower bounds.
All comparisons in exact rationals (fractions) or exact symbolic algebra (sympy).
"""
from fractions import Fraction as F
import sympy as sp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")


def P(v):
    return 9 * v**4 / (1 - 9 * v * v) ** 2


def Q(v):
    return 9 * v**3 / (1 - 9 * v * v)


def h(v):
    return 4 * P(v) + 4 * Q(v)


w = sp.symbols('w', positive=True)
Ps = 9 * w**4 / (1 - 9 * w**2) ** 2
wp = 1 / sp.sqrt(9 + 3 * sp.sqrt(2))
wb = 1 / sp.sqrt(15)
print("2P(w_plus) - 1 simplifies to:", sp.simplify(2 * Ps.subs(w, wp) - 1))
print("4P(w_bb) - 1 simplifies to:", sp.simplify(4 * Ps.subs(w, wb) - 1))
print("w_plus =", sp.N(wp, 30))
print("w_bb   =", sp.N(wb, 30))
print("dP/dw =", sp.factor(sp.diff(Ps, w)), "(positive on (0,1/3))")

# Rational brackets
lo, hi = F(274797, 10**6), F(274798, 10**6)
print("2P(0.274797) < 1:", 2 * P(lo) < 1, " 2P(0.274798) > 1:", 2 * P(hi) > 1)
lo, hi = F(258198, 10**6), F(258199, 10**6)
print("4P(0.258198) < 1:", 4 * P(lo) < 1, " 4P(0.258199) > 1:", 4 * P(hi) > 1)
vPlo, vPhi = F(222407439452191, 10**15), F(222407439452192, 10**15)
print("h(vP_lo) < 1 < h(vP_hi):", h(vPlo) < 1 < h(vPhi))
print("ordering vP_hi < 0.258198 < 0.274797 < 1/3:", vPhi < F(258198, 10**6) < F(274797, 10**6) < F(1, 3))

V = F(20529504599, 10**11)
print("V =", float(V), " 3V < 1:", 3 * V < 1)
twoP, fourP, hV = 2 * P(V), 4 * P(V), h(V)
print("2P(V) =", float(twoP), " 1-2P(V) >= 0.91700:", 1 - twoP >= F(91700, 10**5), " 1-2P(V) < 0.91701:", 1 - twoP < F(91701, 10**5))
print("4P(V) =", float(fourP), " 1-4P(V) >= 0.83401:", 1 - fourP >= F(83401, 10**5), " 1-4P(V) < 0.83402:", 1 - fourP < F(83402, 10**5))
print("h(V) =", float(hV), " (<1:", hV < 1, ")")
print("V < w_plus bracket lo:", V < F(274797, 10**6))
