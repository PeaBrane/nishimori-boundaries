"""(W) p=4: certify P(x d_0 + (1-x) d_q, 0) - (log 2 + beta_w^2/4) < 0.

xi(q) = b^2 q^4/2, v(q) = xi'(q) = 2 b^2 q^3, theta(q) = q xi'(q) - xi(q) = 3 b^2 q^4/2.
P(mu) = log2 + (1/x) log F + (v(1)-v(q))/2 - (1/2)(theta(1) - (1-x) theta(q)),  F = E cosh^x(sqrt(v(q)) g).
P(delta_0) = log2 + v(1)/2 - theta(1)/2 = log2 + b^2/4.  Hence
  Delta = (1/x) log F - v(q)/2 + (1/2)(1-x) theta(q).
"""
import json
import os
import time
from fractions import Fraction as Fr

os.environ["RIG_PREC"] = "140"
from mpmath import iv  # noqa: E402

import oned  # noqa: E402
import rig  # noqa: E402
from rig import ivq  # noqa: E402
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
beta = Fr(8107, 5000)
x = Fr(2499377, 2500000)
q = Fr(9481123, 10000000)
b2 = beta * beta
v = lambda s: 2 * b2 * s ** 3
th = lambda s: Fr(3, 2) * b2 * s ** 4
assert v(1) / 2 - th(1) / 2 == b2 / 4
out = {}
for h in (Fr(1, 8), Fr(1, 16)):
    F, info = oned.expect("coshx", x, 0, v(q), h=h, r_target=16.0)
    Delta = iv.log(F) / ivq(x) - ivq(v(q)) / 2 + ivq((1 - x) * th(q)) / 2
    out[str(h)] = dict(F=[str(F.a), str(F.b)], Delta=[str(Delta.a), str(Delta.b)],
                       Delta_lo_float=rig.flo(Delta), Delta_hi_float=rig.fhi(Delta), info=info)
    print("h=", h, "Delta=", Delta, info, flush=True)
out["verdict"] = "PASS" if all(r["Delta_hi_float"] < 0 for r in out.values()) else "FAIL"
lo = max(r["Delta_lo_float"] for r in out.values() if isinstance(r, dict))
hi = min(r["Delta_hi_float"] for r in out.values() if isinstance(r, dict))
out["Delta_enclosure"] = [lo, hi]
out["generator_claim"] = [-2.839937107376213e-08, -2.8398939063431838e-08]
out["generator_claim_overlaps"] = lo <= out["generator_claim"][1] and out["generator_claim"][0] <= hi
out["runtime_s"] = time.time() - t0
print(json.dumps({k: out[k] for k in ("verdict", "Delta_enclosure", "generator_claim_overlaps", "runtime_s")}))
json.dump(out, open(os.path.join(HERE, "W4_result.json"), "w"), indent=1)
