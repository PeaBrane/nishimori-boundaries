"""(F) driver: rigorous lower bound of min_{u in [U0,1]} D(u) by an adaptive grid with enclosures of D and
Gamma (trapezoid engine), using monotonicity of Gamma on each cell [a,b]:
  D(t) >= D(a) - 1/2 int_a^b xi''(s) (Gamma(b) - s)^+ ds,   D(t) >= D(b) - 1/2 int_a^b xi''(s) (s - Gamma(a))^+ ds.
Cells whose bound is below -TOL are bisected. Single process; results cached in fw4_points.json (resumable).
Usage: fw_grid4.py [TOL] [max_minutes]"""
import json
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np

from common4 import U0, maxrss_mb
from iv import IV
from sg_lb4t import D_and_Gamma, qr, B

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'fw4_points.json')


def _iv(v):
    """Fraction -> enclosing IV; float -> point IV"""
    return IV.exact_rational(v.numerator, v.denominator) if isinstance(v, Fr) else IV.pt(v)


def I_plus(a, b, c_hi):
    """upper enclosure of int_a^b xi''(t) (c - t)^+ dt, xi'' = 6 beta^2 t^2, for any c <= c_hi."""
    if c_hi <= a:
        return IV.pt(0.0)
    e = b if b <= c_hi else c_hi
    A, E, Cc = _iv(a), _iv(e), IV.pt(c_hi)
    return B * B * (Cc * (E * E * E - A * A * A) * 2.0 - (E * E * E * E - A * A * A * A) * 1.5)


def I_minus(a, b, c_lo):
    """upper enclosure of int_a^b xi''(t) (t - c)^+ dt for any c >= c_lo."""
    if c_lo >= b:
        return IV.pt(0.0)
    s = a if a >= c_lo else c_lo
    S, Bb, Cc = _iv(s), _iv(b), IV.pt(c_lo)
    return B * B * ((Bb * Bb * Bb * Bb - S * S * S * S) * 1.5 - Cc * (Bb * Bb * Bb - S * S * S) * 2.0)


def cell_bound(a, b, Pa, Pb):
    """Pa, Pb = (Dlo, Dhi, Glo, Ghi); returns (left_lo, right_lo, best)."""
    left = IV.pt(Pa[0]) - I_plus(a, b, Pb[3]) * 0.5
    right = IV.pt(Pb[0]) - I_minus(a, b, Pa[2]) * 0.5
    l, r = float(left.lo), float(right.lo)
    return l, r, max(l, r)


def load():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def evaluate(u, cache):
    k = str(u)
    if k not in cache:
        t0 = time.time()
        D, G, _ = D_and_Gamma(u)
        cache[k] = [float(D.lo), float(D.hi), float(G.lo), float(G.hi), time.time() - t0]
        json.dump(cache, open(CACHE + '.tmp', 'w'), indent=1)
        os.replace(CACHE + '.tmp', CACHE)
    return cache[k]


if __name__ == "__main__":
    TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 5e-6
    tmax = 60 * (float(sys.argv[2]) if len(sys.argv) > 2 else 25)
    T0 = time.time()
    cache = load()
    pts = {U0, qr, Fr(1)}
    k = 0
    while U0 + Fr(k, 50) < 1:
        pts.add(U0 + Fr(k, 50))
        k += 1
    for u in list(cache):
        pts.add(Fr(u))
    pts = sorted(pts)
    while True:
        for u in pts:
            evaluate(u, cache)
            if time.time() - T0 > tmax:
                print("time budget reached; rerun to resume", flush=True)
                sys.exit(2)
        cells = [(pts[i], pts[i + 1], cell_bound(pts[i], pts[i + 1], cache[str(pts[i])], cache[str(pts[i + 1])]))
                 for i in range(len(pts) - 1)]
        bad = [(a, b) for a, b, cb in cells if cb[2] < -TOL]
        worst = min(cb[2] for _, _, cb in cells)
        print(f"{len(pts)} points, {len(bad)} cells below -TOL, min cell {worst:.3e}, rss={maxrss_mb():.0f}MB",
              flush=True)
        if not bad:
            break
        for a, b in bad:
            if b - a < Fr(1, 10 ** 6):
                print("FAIL: cell too small", float(a), float(b))
                sys.exit(1)
            pts.append((a + b) / 2)
        pts = sorted(set(pts))
    out = {"TOL": TOL, "q": str(qr), "U0": str(U0),
           "points": [[str(u)] + cache[str(u)] for u in pts],
           "cells": [[str(a), str(b)] + list(cb) for a, b, cb in cells],
           "min_cell": worst}
    json.dump(out, open(os.path.join(HERE, 'fw4_grid.json'), 'w'), indent=1)
    print("min cell bound", worst, "points", len(pts), f"elapsed {time.time() - T0:.0f}s")
