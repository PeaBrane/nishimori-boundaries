"""Assemble the (F) decision: certified lower bound on P(mu*) + min_u D(u) vs certified (L) max."""
import json
import os
from fractions import Fraction as Fr
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
ld = lambda f: json.load(open(os.path.join(HERE, f)))
C = ld("F_const.json"); Fc = ld("F_cells_result.json")
L = ld("L_result.json")["summary"]; W = ld("W_result.json")
P_lo = Fr(C["P_float"][0]); P_hi = Fr(C["P_float"][1])     # rigorous float bounds (outward rounded)
minD_lo = Fr(Fc["min_D_lower_exact"])
LB = P_lo + minD_lo
Lmax = Fr(L["certified_max_U_hi"])
res = dict(P_mu_star=[float(P_lo), float(P_hi)], min_D_lower=float(minD_lo), min_D_upper_grid=Fc["grid_min_D_upper"],
           P_plus_minD_lower=float(LB), L_certified_max=float(Lmax), margin=float(LB - Lmax),
           decision_LB_exceeds_Lmax=LB > Lmax, exceeds_2_0456862=LB > Fr(20456862, 10**7),
           generator_claims_ok=dict(P_in_claim=Fr("2.04575395212") <= P_lo and P_hi <= Fr("2.04575395228"),
                                    minD_ge_claim=minD_lo >= Fr("-6.78e-5")),
           W_Delta=[W["1/16"]["Delta_lo_float"], W["1/16"]["Delta_hi_float"]])
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(os.path.join(HERE, "FINAL_result.json"), "w"), indent=1, default=str)
