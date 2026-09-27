import math, mpmath
import numpy as np
from iv import *
from gauss import expect
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mpmath.mp.dps = 40
s, x, a = 2.3, 0.55, 0.7
r = expect(lambda z: tcosh(z * s), 1.0, s, n=1000)
print("E cosh(sZ)", r, "exact", math.exp(s*s/2), r.width())
r = expect(lambda z: tlogcosh(z * s + a), abs(a)+s, 1.0, n=1000)
ex = mpmath.quad(lambda z: mpmath.log(mpmath.cosh(a+s*z))*mpmath.npdf(z), [-mpmath.inf, 0, mpmath.inf])
print("E logcosh", r, "mp", ex, r.width())
r = expect(lambda z: tpow_pos(tcosh(z * s), x), 1.0, x*s, n=1000)
ex = mpmath.quad(lambda z: mpmath.cosh(s*z)**x*mpmath.npdf(z), [-mpmath.inf, 0, mpmath.inf])
print("E cosh^x", r, "mp", ex, r.width(), float(r.lo) <= ex <= float(r.hi))
