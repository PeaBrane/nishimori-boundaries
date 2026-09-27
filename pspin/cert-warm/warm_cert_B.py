"""Warm predicate (W) of Theorem 6.10 (thm:reduction) -- second implementation (Table 1, tab:cert).

Same quantities as warm_cert_A.py (see its docstring), computed without numpy intervals, Simpson
rules, libm or rational series:
  psi         -- oned.expect("logcosh"): mpmath.iv (directed rounding), trapezoid rule on the whole
                 line with the Trefethen-Weideman strip bound and explicit Gaussian truncation tails.
  logarithms  -- mpmath.iv.log (directed rounding).
Working precision: RIG_PREC bits (default 80 in rig.py; this script sets 120 unless overridden).
Usage: python warm_cert_B.py
"""
import json
import os

os.environ.setdefault("RIG_PREC", "120")

from fractions import Fraction as Fr  # noqa: E402

from mpmath import iv  # noqa: E402

import rig  # noqa: E402
from rig import flo, fhi, ivq  # noqa: E402
from oned import expect as oexpect  # noqa: E402
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {
    3: dict(j0=Fr(3841, 5000), q0=Fr(814809, 10**6), m1=Fr(11, 25)),
    4: dict(j0=Fr(8107, 10000), q0=Fr(948189, 10**6), m1=Fr(16, 25)),
}


def psiB(r):
    val, info = oexpect("logcosh", None, r, r)
    return val, info


def run(p, j0, q0, m1):
    c = 2 * j0 * j0
    xi = lambda q: c * Fr(q) ** p
    xip = lambda q: c * p * Fr(q) ** (p - 1)
    th = lambda q: (p - 1) * c * Fr(q) ** p
    out = dict(p=p, q0=str(q0), m1=str(m1), prec=iv.prec)
    # side conditions, recomputed here (exact integers)
    if p % 2 == 1:
        # q0 in Q_p <=> g(-1) = -1 + p q0^(p-1) + (p-1) q0^p >= 0 (g concave on [-1,0], g(0) >= 0)
        assert -1 + p * q0 ** (p - 1) + (p - 1) * q0 ** p >= 0
    assert p * m1 * m1 <= p - 2

    def Phi(q):
        v, info = psiB(xip(q))
        return v - ivq((xip(q) + th(q)) / 2), info

    P0, info = Phi(q0)
    out["Phi_q0"] = [flo(P0), fhi(P0)]
    out["psi_info"] = info
    m = ivq(m1)
    I = (1 + m) / 2 * iv.log(1 + m) + (1 - m) / 2 * iv.log(1 - m)
    f = ivq(xi(m1)) - I
    out["I_m1"] = [flo(I), fhi(I)]
    out["f_m1"] = [flo(f), fhi(f)]
    bar = f if fhi(f) > 0 else iv.mpf(0)
    marg = P0 - bar
    out["margin_W_lo"] = flo(marg)
    assert flo(P0) > 0 and flo(marg) > 0
    if p == 3:
        Ph, _ = Phi(Fr(1, 2))
        B = ivq(c * (m1 - Fr(1, 2)) ** 2 * (m1 + 1))
        out["Phi_half"] = [flo(Ph), fhi(Ph)]
        out["Phi_half_plus_B_hi"] = fhi(Ph + B)
        out["margin_half_lo"] = flo(P0 - (Ph + B))
        assert flo(P0 - (Ph + B)) > 0
    return out


if __name__ == "__main__":
    res = {}
    for p, kw in CASES.items():
        res[p] = run(p, **kw)
        print(json.dumps(res[p], indent=1, default=str), flush=True)
    json.dump(res, open(os.path.join(HERE, "warm_cert_B.json"), "w"), indent=1, default=str)
    print("rig.iv.prec =", rig.iv.prec)
