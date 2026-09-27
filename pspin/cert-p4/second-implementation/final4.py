"""Assemble the p=4 decision from this checker's own results (exact rational comparison of float bounds)."""
import json
import os
from fractions import Fraction as Fr
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
ld = lambda f: json.load(open(os.path.join(HERE, f)))
C, Fc, L, W = ld("F_const4.json"), ld("F_cells4_result.json"), ld("L4_result.json")["summary"], ld("W4_result.json")
F6 = ld("F_cells4_result_tol1e-6.json")
P_lo, P_hi = Fr(C["P_float"][0]), Fr(C["P_float"][1])
minD_lo = Fr(Fc["min_D_lower_exact"])
LB = P_lo + minD_lo
Lmax = Fr(L["certified_max_U_hi"])
gen = dict(LB=2.754576695939707, worst_U=2.7532998671863287, margin=0.0012768287533782008,
           P=(2.754581581382262, 2.754581581383308), min_cell=-4.885442554682566e-06,
           D0=(-1.4381600091518418e-07, -1.438049291015172e-07))
res = dict(
    W=dict(Delta_w=W["Delta_enclosure"], pass_=W["Delta_enclosure"][1] < 0),
    L=dict(certified_max_U_hi=float(Lmax), tiling_ok=L["tiling_ok"], worst=L["worst_interval"],
           binbound_max=L["certified_max_binbound_hi"]),
    F=dict(P_mu_star=[float(P_lo), float(P_hi)], min_D_lower=float(minD_lo), min_D_upper=Fc["grid_min_D_upper"],
           min_D_lower_tol1e6=F6["min_D_certified_lower"], D0=C["D0_float"], Dq=C["Dq_float"], Gq=C["Gq_float"],
           LB=float(LB), LB_tol1e6=float(P_lo + Fr(F6["min_D_lower_exact"]))),
    margin=float(LB - Lmax), F_gt_L=LB > Lmax,
    generator_consistency=dict(
        P_inside_gen=gen["P"][0] <= P_lo and P_hi <= gen["P"][1],
        D0_inside_gen=gen["D0"][0] <= C["D0_float"][0] and C["D0_float"][1] <= gen["D0"][1],
        my_LB_ge_gen_LB=LB >= Fr(gen["LB"]),
        my_Lmax_le_gen=Lmax <= Fr(gen["worst_U"]),
        binbound_below_gen_LB=Fr(L["certified_max_binbound_hi"]) < Fr(gen["LB"])),
)
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(os.path.join(HERE, "FINAL4_result.json"), "w"), indent=1, default=str)
