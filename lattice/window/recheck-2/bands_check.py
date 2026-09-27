"""Exact check of the band table (Table 4, tab:lat-bands) in ../out/claims.json."""
import json
import os
from fractions import Fraction as Fr
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "claims.json")))
P = lambda w: 9 * w**4 / (1 - 9 * w**2) ** 2
for key, mult in (("W_plus", 2), ("W_two_point", 4)):
    cells = d[key]["cells"]
    for b in d[key]["bands"]:
        lo, hi = Fr(b["lo"]), Fr(b["hi"])
        meet = [c for c in cells if Fr(c["br"]) > lo and Fr(c["bl"]) < hi]
        covered = Fr(cells[0]["bl"]) <= lo and hi <= Fr(cells[-1]["br"])
        sup = max(Fr(c["W_up"]) for c in meet)
        ok = covered and sup <= Fr(b["sup_W_up"]) and Fr(b["bound"]) <= 1 - mult * P(Fr(b["sup_W_up"]))
        print(key, b["lo"], b["hi"], "cells", len(meet), "sup W_up", float(sup), "bound", b["bound"], float(Fr(b["bound"])), "ok", ok)
