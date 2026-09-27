"""(F) 1D constants for mu* = x d_0 + (1-x) d_q, beta=5/2, h=0.
F0 = E cosh^x(Y), G0 = E cosh^x(Y) log cosh(Y), T0 = E cosh^x(Y) tanh^2(Y), Y ~ N(0, v(q)).
P(mu*) = log2 + (1/x) log F0 + (v(1)-v(q))/2 - (1/2)(theta(1) - (1-x) theta(q))
D(q)   = (1/x) log F0 - G0/F0 + (x/2) theta(q)          (= -x dP/dx)
D(0)   = -((1-x)/x) D(q)                                  (= (1-x) dP/dx)
Gamma(q) = T0/F0, Gamma(0) = 0.
"""
import json, os, time
os.environ.setdefault("RIG_PREC", "100")
from fractions import Fraction as Fr
from mpmath import iv
import rig, oned
from rig import ivq
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

beta = Fr(5, 2); b2 = beta * beta
x = Fr(554371, 10**6); q = Fr(929223, 10**6)
v = lambda s: Fr(3, 2) * b2 * s * s
th = lambda s: b2 * s ** 3


def s_(I):
    """Exact raw endpoints (sign, man, exp, bc) plus a display string."""
    (a, b) = I._mpi_
    return {"raw": [[a[0], int(a[1]), a[2], a[3]], [b[0], int(b[1]), b[2], b[3]]], "str": str(I)}


if __name__ == "__main__":
    t0 = time.time()
    F0, i1 = oned.expect("coshx", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    G0, i2 = oned.expect("coshx_logcosh", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    T0, i3 = oned.expect("coshx_tanh2", x, 0, v(q), h=Fr(1, 16), r_target=16.0)
    xi = ivq(x)
    P = rig.LOG2 + iv.log(F0) / xi + ivq((v(1) - v(q)) / 2 - (th(1) - (1 - x) * th(q)) / 2)
    Dq = iv.log(F0) / xi - G0 / F0 + ivq(x * th(q) / 2)
    D0 = -ivq((1 - x) / x) * Dq
    Gq = T0 / F0
    res = dict(F0=s_(F0), G0=s_(G0), T0=s_(T0), P=s_(P), Dq=s_(Dq), D0=s_(D0), Gq=s_(Gq),
               P_float=[rig.flo(P), rig.fhi(P)], Dq_float=[rig.flo(Dq), rig.fhi(Dq)], D0_float=[rig.flo(D0), rig.fhi(D0)],
               Gq_float=[rig.flo(Gq), rig.fhi(Gq)], info=[i1, i2, i3], runtime_s=time.time() - t0)
    for k in ("F0", "G0", "T0", "P", "Dq", "D0", "Gq"):
        print(k, res[k])
    print("generator claim P in [2.04575395212, 2.04575395228]:",
          2.04575395212 <= rig.flo(P) and rig.fhi(P) <= 2.04575395228)
    json.dump(res, open(os.path.join(HERE, "F_const.json"), "w"), indent=1)
