"""(W): certify P(mu_w,0) - (log 2 + beta_w^2/4) < 0.

With xi(q)=b^2 q^3/2: xi'(q)=3b^2q^2/2 =: v(q), theta(q)=q xi'(q)-xi(q)=b^2 q^3.
mu = x d_0 + (1-x) d_q:  P = log2 + (1/x) log E cosh^x(sqrt(v(q)) g) + (v(1)-v(q))/2 - (1/2) b^2 (1-(1-x) q^3)
annealed log2 + b^2/4 = P(delta_0).  Difference:
  Delta = (1/x) log F - v(q)/2 + (1/2) b^2 (1-x) q^3,   F = E cosh^x(sqrt(v(q)) g).
"""
import json, os, sys, time
os.environ["RIG_PREC"] = "140"
from fractions import Fraction as Fr
from mpmath import iv
import rig, oned
from rig import ivq

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
beta = Fr(3841, 2500)
x = Fr(499411, 500000)
q = Fr(203463, 250000)
b2 = beta * beta
vq = Fr(3, 2) * b2 * q * q
out = {}
for h in (Fr(1, 8), Fr(1, 16)):
    F, info = oned.expect("coshx", x, 0, vq, h=h, r_target=16.0)
    xi = ivq(x)
    Delta = iv.log(F) / xi - ivq(vq) / 2 + ivq(b2 * (1 - x) * q ** 3) / 2
    out[str(h)] = dict(F=[str(F.a), str(F.b)], Delta=[str(Delta.a), str(Delta.b)],
                       Delta_hi_float=rig.fhi(Delta), Delta_lo_float=rig.flo(Delta), info=info)
    print("h=", h, "F=", F, "\nDelta=", Delta, info)
Delta_hi = max(v["Delta_hi_float"] for v in out.values())
out["verdict"] = "PASS" if all(v["Delta_hi_float"] < 0 for k, v in out.items() if k != "verdict") else "FAIL"
out["runtime_s"] = time.time() - t0
print(out["verdict"], out["runtime_s"])
json.dump(out, open(os.path.join(HERE, "W_result.json"), "w"), indent=1)
