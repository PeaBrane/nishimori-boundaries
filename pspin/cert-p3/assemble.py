"""Assemble the cold certificate:
 LB := P(mu*)_lo + min( D(0)_lo  [valid on [0, 8/75], D nondecreasing there],
                        min_k cell_k )
 cell_k on [u_k,u_{k+1}] = max( D_lo(u_k)   - 1/2 int_{u_k}^{u_{k+1}} xi''(t) (Gam_hi(u_{k+1}) - t)^+ dt,
                                D_lo(u_{k+1}) - 1/2 int_{u_k}^{u_{k+1}} xi''(t) (t - Gam_lo(u_k))^+ dt )
 Then require LB > worst landscape U_hi.  All arithmetic in outward-rounded intervals."""
import json, os, sys
from fractions import Fraction as Fr
import numpy as np
from iv import IV
from sg_lb import P, D0, B

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

res = json.load(open(sys.argv[1]))
land = json.load(open(os.path.join(HERE, 'landscape_cover.json')))
pts = sorted(res, key=lambda r: Fr(r[0]))
us = [Fr(r[0]) for r in pts]
assert us[0] == Fr(8, 75) and us[-1] == 1
b2x3 = B * B * 3.0            # xi''(t) = 3 beta^2 t
third = IV(np.nextafter(1/3, -np.inf), np.nextafter(1/3, np.inf))
def I_plus(a, b, c):
    """enclose int_a^b 3 beta^2 t (c - t)^+ dt, c an interval (use c.hi for upper bound)"""
    ch = c.hi
    if ch <= a: return IV.pt(0.0)
    e = min(b, ch)
    A, E, C = IV.pt(a), IV.pt(e), IV.pt(ch)
    return b2x3 * (C * (E*E - A*A) * 0.5 - (E*E*E - A*A*A) * third)
def I_minus(a, b, c):
    """enclose int_a^b 3 beta^2 t (t - c)^+ dt using c.lo"""
    cl = c.lo
    if cl >= b: return IV.pt(0.0)
    s = max(a, cl)
    S, Bb, C = IV.pt(s), IV.pt(b), IV.pt(cl)
    return b2x3 * ((Bb*Bb*Bb - S*S*S) * third - C * (Bb*Bb - S*S) * 0.5)
cells = []
for k in range(len(pts) - 1):
    a, bnd = float(np.nextafter(float(us[k]), -np.inf)), float(np.nextafter(float(us[k+1]), np.inf))
    Dk = IV(pts[k][2], pts[k][3]); Dk1 = IV(pts[k+1][2], pts[k+1][3])
    Gk = IV(pts[k][4], pts[k][5]); Gk1 = IV(pts[k+1][4], pts[k+1][5])
    left = Dk - I_plus(a, bnd, Gk1) * 0.5
    right = Dk1 - I_minus(a, bnd, Gk) * 0.5
    cells.append((a, bnd, float(left.lo), float(right.lo), max(float(left.lo), float(right.lo))))
mincell = min(c[4] for c in cells)
kmin = min(range(len(cells)), key=lambda i: cells[i][4])
LB = P.lo + min(float(D0.lo), mincell)
LBiv = IV(np.nextafter(LB, -np.inf))
print("P(mu*) lo", P.lo, " D(0) lo", D0.lo)
print("min cell bound", mincell, "at", cells[kmin][:2])
for c in sorted(cells, key=lambda c: c[4])[:6]: print("  cell", c)
print("certified LB F^0(5/2,0) >=", float(LBiv.lo))
print("landscape worst U_hi     =", land['worst_U_hi'])
print("margin", float(LBiv.lo) - land['worst_U_hi'], "CERTIFIED" if float(LBiv.lo) > land['worst_U_hi'] else "NOT certified")
json.dump({"LB": float(LBiv.lo), "worst_U_hi": land['worst_U_hi'], "min_cell": mincell, "cells": cells},
          open(os.path.join(HERE, 'assembly_result.json'), 'w'), indent=1)
