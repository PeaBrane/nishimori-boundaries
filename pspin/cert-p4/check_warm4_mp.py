"""High-precision (mpmath, non-interval) cross-check of the (W) value Delta_w at the rational trial."""
import json
import os
import mpmath as mp
from fractions import Fraction as Fr
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))
w = json.load(open(os.path.join(HERE, 'warm4_result.json')))
b = mp.mpf(Fr(w['beta_w']).numerator) / Fr(w['beta_w']).denominator
x = mp.mpf(Fr(w['x']).numerator) / Fr(w['x']).denominator
q = mp.mpf(Fr(w['q']).numerator) / Fr(w['q']).denominator
L = 2 * b * b * q ** 3
A = mp.quad(lambda z: mp.cosh(mp.sqrt(L) * z) ** x * mp.npdf(z), mp.linspace(-25, 25, 51))
D = mp.log(A) / x - L / 2 + (1 - x) * 1.5 * b * b * q ** 4 / 2
print("mpmath Delta_w =", mp.nstr(D, 15), " interval:", [w['Delta_w_lo'], w['Delta_w_hi']],
      "inside:", w['Delta_w_lo'] <= float(D) <= w['Delta_w_hi'])
