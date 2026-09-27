"""Assemble the p=4 cold certificate:
  LB := P(mu*)_lo + min( D(0)_lo  [valid on [0, U0]: Gamma <= xi' <= id there, so D is nondecreasing],
                         min_k cell_k over the stored grid on [U0, 1] )
and require LB > worst landscape U_hi. Cell bounds are recomputed here from the stored point enclosures
(outward-rounded). Also re-checks the exact side conditions and the warm result."""
import json
import os
from fractions import Fraction as Fr

import numpy as np

from common4 import BETA_C, BETA_W, J0, M_A, U0
from fw_grid4 import cell_bound
from iv import IV
from sg_lb4t import P, D0, Dq, qr, xr
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
grid = json.load(open(os.path.join(HERE, 'fw4_grid.json')))
land = json.load(open(os.path.join(HERE, 'landscape4_cover.json')))
warm = json.load(open(os.path.join(HERE, 'warm4_result.json')))

pts = grid['points']
us = [Fr(p[0]) for p in pts]
assert us == sorted(us) and len(set(us)) == len(us)
assert us[0] == U0 and us[-1] == 1 and qr in us
assert Fr(grid['q']) == qr
cells = []
for k in range(len(pts) - 1):
    cb = cell_bound(us[k], us[k + 1], pts[k][1:5], pts[k + 1][1:5])
    cells.append((str(us[k]), str(us[k + 1])) + cb)
mincell = min(c[4] for c in cells)
kmin = min(range(len(cells)), key=lambda i: cells[i][4])
Glo = [p[3] for p in pts]
Ghi = [p[4] for p in pts]
mono = all(Ghi[k] <= Glo[k + 1] + 1e-12 or Glo[k] <= Ghi[k + 1] for k in range(len(pts) - 1))
LB = float(np.nextafter(P.lo + min(float(D0.lo), mincell), -np.inf))

# exact side conditions
assert 2 * BETA_C ** 2 * U0 ** 3 <= U0, "xi'(U0) <= U0"
gA = BETA_C * J0 * M_A ** 4 - M_A ** 2 / 2
assert gA < 0, "universal bound at M_A"
assert Fr(land['m_range'][0]) == M_A and Fr(land['beta']) == BETA_C and Fr(land['j0']) == J0
assert Fr(warm['beta_w']) == BETA_W and Fr(warm['j0']) == J0 and warm['certified']

print("P(mu*) =", P, " D(0) =", D0, " D(q) =", Dq)
print("grid points", len(pts), " min cell bound", mincell, "on", cells[kmin][:2])
for c in sorted(cells, key=lambda c: c[4])[:6]:
    print("  cell", c)
print("Gamma nondecreasing across nodes (consistency):", mono)
print("certified LB F^0(10/3, 0) >=", LB)
print("landscape worst U_hi      =", land['worst_U_hi'])
margin = LB - land['worst_U_hi']
print("cold margin", margin, "CERTIFIED" if margin > 0 else "NOT certified")
print("universal-bound slack g(M_A) =", float(gA), " warm Delta_w in", [warm['Delta_w_lo'], warm['Delta_w_hi']])
json.dump({"p": 4, "j0": str(J0), "beta_w": str(BETA_W), "beta_c": str(BETA_C), "mu_star": [str(xr), str(qr)],
           "P_mu_star": [float(P.lo), float(P.hi)], "D0": [float(D0.lo), float(D0.hi)],
           "min_cell": mincell, "min_cell_at": cells[kmin][:2], "LB": LB, "worst_U_hi": land['worst_U_hi'],
           "cold_margin": margin, "certified": bool(margin > 0), "g_MA": float(gA),
           "warm_Delta_w": [warm['Delta_w_lo'], warm['Delta_w_hi']], "n_grid": len(pts),
           "n_cover": len(land['cover']), "cells": cells},
          open(os.path.join(HERE, 'assembly4_result.json'), 'w'), indent=1)
