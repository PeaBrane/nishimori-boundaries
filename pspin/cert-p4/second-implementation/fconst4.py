"""(F) p=4 1D constants for mu* = x d_0 + (1-x) d_q, beta = 10/3, h = 0.
v(s) = xi'(s) = 2 b^2 s^3, theta(s) = 3 b^2 s^4 / 2.
F0 = E cosh^x(Y), G0 = E cosh^x(Y) log cosh(Y), T0 = E cosh^x(Y) tanh^2(Y), Y ~ N(0, v(q)).
P(mu*) = log2 + (1/x) log F0 + (v(1)-v(q))/2 - (1/2)(theta(1) - (1-x) theta(q))
dP/dx  = -(1/x^2) log F0 + G0/(x F0) - theta(q)/2
D(q)   = -x dP/dx = (1/x) log F0 - G0/F0 + (x/2) theta(q)     (u = q: x -> (1-eps) x)
D(0)   = (1-x) dP/dx = -((1-x)/x) D(q)                          (u = 0: x -> x + eps (1-x))
Gamma(q) = T0/F0 (both one-sided limits agree), Gamma(0) = 0 (h = 0, symmetry).
"""
import json
import os
import time
from fractions import Fraction as Fr

os.environ.setdefault("RIG_PREC", "100")
from mpmath import iv  # noqa: E402

import oned  # noqa: E402
import rig  # noqa: E402
from rig import ivq  # noqa: E402
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
beta = Fr(10, 3)
b2 = beta * beta
x = Fr(930529, 2000000)
q = Fr(4930831, 5000000)
v = lambda s: 2 * b2 * s ** 3
th = lambda s: Fr(3, 2) * b2 * s ** 4


def s_(I):
    (a, b) = I._mpi_
    return {"raw": [[a[0], int(a[1]), a[2], a[3]], [b[0], int(b[1]), b[2], b[3]]], "str": str(I)}


if __name__ == "__main__":
    t0 = time.time()
    F0, i1 = oned.expect("coshx", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    G0, i2 = oned.expect("coshx_logcosh", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    T0, i3 = oned.expect("coshx_tanh2", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    xi = ivq(x)
    P = rig.LOG2 + iv.log(F0) / xi + ivq((v(1) - v(q)) / 2 - (th(1) - (1 - x) * th(q)) / 2)
    dPdx = -iv.log(F0) / (xi * xi) + G0 / (xi * F0) - ivq(th(q)) / 2
    Dq = iv.log(F0) / xi - G0 / F0 + ivq(x * th(q) / 2)
    D0 = -ivq((1 - x) / x) * Dq
    Gq = T0 / F0
    res = dict(F0=s_(F0), G0=s_(G0), T0=s_(T0), P=s_(P), Dq=s_(Dq), D0=s_(D0), Gq=s_(Gq), dPdx=s_(dPdx),
               P_float=[rig.flo(P), rig.fhi(P)], Dq_float=[rig.flo(Dq), rig.fhi(Dq)],
               D0_float=[rig.flo(D0), rig.fhi(D0)], Gq_float=[rig.flo(Gq), rig.fhi(Gq)],
               dPdx_float=[rig.flo(dPdx), rig.fhi(dPdx)], info=[i1, i2, i3], runtime_s=time.time() - t0)
    for k in ("F0", "G0", "T0", "P", "dPdx", "Dq", "D0", "Gq"):
        print(k, res[k]["str"])
    gP = (2.754581581382262, 2.754581581383308)
    print("generator P(mu*) interval contains mine:", gP[0] <= res["P_float"][0] and res["P_float"][1] <= gP[1])
    json.dump(res, open(os.path.join(HERE, "F_const4.json"), "w"), indent=1)
