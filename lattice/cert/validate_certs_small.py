"""Close the loop: evaluate the two certificate codes (cert_arb.py, cert_fixed.py) on small instances
of the same gadget family and compare with exhaustive brute force (all J, all interior spins)
from validate_small.py.  Needs python-flint, numpy, mpmath."""
import itertools
import math
from fractions import Fraction as Fr

import numpy as np
from flint import fmpq

import cert_arb as CA
import cert_fixed as CF
from validate_small import build, bf_K
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")


def brute(b0, k, p, beta):
    nv, edges, _ = build(b0, k)
    m = len(edges)
    allJ = np.array(list(itertools.product([1, -1], repeat=m)), dtype=np.float64).T
    nneg = (allJ < 0).sum(axis=0)
    wts = p ** nneg * (1 - p) ** (m - nneg)
    K = np.concatenate([bf_K(nv, edges, allJ[:, i:i + 1024], beta) for i in range(0, allJ.shape[1], 1024)])
    return dict(u=(wts * np.abs(np.tanh(K))).sum(), pbar=(wts * (1 - np.exp(-2 * np.abs(K)))).sum(),
                v=(wts * np.exp(-K)).sum())


worst = 0.0
tol = 1e-11
ok = True
for b0, k, p in [(2, 1, Fr(1, 5)), (3, 1, Fr(3, 20)), (2, 2, Fr(1, 5))]:
    A = CA.Design("t", b0, k, fmpq(p.numerator, p.denominator))
    X = CF.Design("t", b0, k, p)
    for beta in (Fr(2, 5), Fr(17, 10), Fr(3)):
        bf = brute(b0, k, float(p), float(beta))
        Fa = A.eval_interval(fmpq(beta.numerator, beta.denominator), fmpq(beta.numerator, beta.denominator))
        slo, shi = CF.exp_neg_bounds(2 * beta)
        Fx = X.functionals(*CF.mags_s(slo, shi, b0))
        for key in ("u", "pbar"):
            lo_a, hi_a = (float(z) for z in CA.bounds(Fa[key]))
            lo_x, hi_x = Fx[key].lof(), Fx[key].hif()
            good = (lo_a - tol <= bf[key] <= hi_a + tol) and (lo_x - tol <= bf[key] <= hi_x + tol)
            ok &= good
            worst = max(worst, abs(bf[key] - (lo_a + hi_a) / 2), abs(bf[key] - (lo_x + hi_x) / 2))
            print(f"b0={b0} k={k} p={p} beta={beta} {key}: brute={bf[key]:.14f} arb=[{lo_a:.14f},{hi_a:.14f}] fixed=[{lo_x:.14f},{hi_x:.14f}] {'ok' if good else 'MISMATCH'}")
    gam = 0.5 * math.log((1 - float(p)) / float(p))
    bf = brute(b0, k, float(p), gam)
    Wa = A.warm(fmpq(1, 3))
    va = [float(z) for z in Wa["v_bounds"]]
    vx, _, _ = X.warm()
    good = (va[0] - tol <= bf["v"] <= va[1] + tol) and (vx.lof() - tol <= bf["v"] <= vx.hif() + tol)
    ok &= good
    worst = max(worst, abs(bf["v"] - va[0]), abs(bf["v"] - vx.lof()))
    print(f"b0={b0} k={k} p={p} beta=gamma v: brute={bf['v']:.14f} arb=[{va[0]:.14f},{va[1]:.14f}] fixed=[{vx.lof():.14f},{vx.hif():.14f}] Nishimori exact {Wa['nishimori_exact']} {'ok' if good else 'MISMATCH'}")
print(f"max |brute - certificate midpoint| = {worst:.3e};  CERT CODES MATCH BRUTE FORCE: {ok}")
