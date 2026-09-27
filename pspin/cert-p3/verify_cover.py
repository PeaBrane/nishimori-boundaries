"""Replay check of the frozen landscape cover (no search): recompute every trial's Parisi value
with the interval engine and re-check both endpoint bounds against the certified SG lower bound."""
import json, os, sys
from fractions import Fraction as Fr
from landscape_cert import P_iv, U_iv

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
cov = json.load(open(os.path.join(HERE, 'landscape_cover.json')))
LB = json.load(open(os.path.join(HERE, 'assembly_result.json')))['LB']
worst = -1e9; prev = Fr(1, 4)
for c in cov['cover']:
    ma, mb = Fr(c['ma']), Fr(c['mb'])
    assert ma == prev, "cover has a gap"; prev = mb
    t = [Fr(v) for v in c['t']]
    P = P_iv(c['kind'], t)
    top = max(float(U_iv(ma, c['kind'], t, P).hi), float(U_iv(mb, c['kind'], t, P).hi))
    worst = max(worst, top)
assert prev == 1
print("intervals", len(cov['cover']), "replayed worst U_hi", worst, "LB", LB, "PASS" if worst < LB else "FAIL")
