"""(F) p=4 certified lower bound on min_{u in [0,1]} D(u), grid enclosures + Gamma monotonicity.

D'(u) = -(1/2) xi''(u) (Gamma(u) - u),  xi''(u) = 6 b^2 u^2 >= 0,  Gamma nondecreasing on [0,1].
On a cell [a,b]:
  L1(u) = D(a) - b^2 [Gb (u^3 - a^3) - (3/4)(u^4 - a^4)] <= D(u)   (Gamma(s) <= Gb := Gamma(b)_hi)
  L2(u) = D(b) + b^2 [Ga (b^3 - u^3) - (3/4)(b^4 - u^4)] <= D(u)   (Gamma(s) >= Ga := Gamma(a)_lo)
L1' = -3 b^2 u^2 (Gb - u), L2' = 3 b^2 u^2 (u - Ga): each is monotone on any piece whose interior avoids
Gb (resp. Ga), so on such pieces the minima are at the endpoints.  All cell arithmetic is exact (Fractions).
The whole interval [0,1] is covered by cells (u=0 and u=q use closed forms); the xi'(u) <= u shortcut on
[0,U0] is not used.  Single process, sequential, resumable cache F_grid4.json.
"""
import json
import math
import os
import resource
import sys
import time
from fractions import Fraction as Fr

import fgrid4

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "F_grid4.json")
q = fgrid4.q
B2 = fgrid4.b2
NSUB = 64
TOL = Fr(os.environ.get("FTOL", "1/1000000"))


def L1(u, a, Da, Gb):
    return Da - B2 * (Gb * (u ** 3 - a ** 3) - Fr(3, 4) * (u ** 4 - a ** 4))


def L2(u, b, Db, Ga):
    return Db + B2 * (Ga * (b ** 3 - u ** 3) - Fr(3, 4) * (b ** 4 - u ** 4))


def cell_lb(a, b, ra, rb):
    Da, Db = Fr(ra["D"][0]), Fr(rb["D"][0])
    Ga, Gb = Fr(ra["G"][0]), Fr(rb["G"][1])
    pts = {a + (b - a) * i / NSUB for i in range(NSUB + 1)}
    pts |= {g for g in (Ga, Gb) if a < g < b}
    pts = sorted(pts)
    best = None
    for p0, p1 in zip(pts, pts[1:]):
        l1 = min(L1(p0, a, Da, Gb), L1(p1, a, Da, Gb))
        l2 = min(L2(p0, b, Db, Ga), L2(p1, b, Db, Ga))
        lb = max(l1, l2)
        if best is None or lb < best[0]:
            best = (lb, p0, p1)
    return best


def load():
    return json.load(open(GRID)) if os.path.exists(GRID) else {}


def save(G):
    tmp = GRID + ".tmp"
    json.dump(G, open(tmp, "w"), indent=0)
    os.replace(tmp, GRID)


def run_points(G, us):
    todo = [u for u in us if str(u) not in G]
    t = time.time()
    for u in todo:
        r = fgrid4.evaluate(u)
        G[r["u"]] = r
        save(G)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"  evaluated {len(todo)} points in {time.time() - t:.1f}s, maxrss {rss:.0f} MB", flush=True)


def cells(G):
    us = sorted(Fr(k) for k in G)
    out = []
    for a, b in zip(us, us[1:]):
        lb, p0, p1 = cell_lb(a, b, G[str(a)], G[str(b)])
        out.append((lb, a, b, p0, p1))
    return out


if __name__ == "__main__":
    t0 = time.time()
    G = load()
    init = [Fr(0), Fr(21, 100)] + [Fr(i, 100) for i in range(5, 99)] + [q]
    init += [q + (1 - q) * Fr(j, 16) for j in range(1, 17)]
    run_points(G, init)
    for it in range(40):
        cs = cells(G)
        ub = min(Fr(r["D"][1]) for r in G.values())
        lbmin = min(c[0] for c in cs)
        bad = [c for c in cs if c[0] < ub - TOL]
        print(f"iter {it}: points={len(G)} cells={len(cs)} min D_hi(grid)={float(ub):.12e} "
              f"certified LB={float(lbmin):.12e} cells to refine={len(bad)}", flush=True)
        if not bad:
            break
        run_points(G, [(c[1] + c[2]) / 2 for c in bad])
    cs = cells(G)
    lbmin, a, b, p0, p1 = min(cs)
    ub = min(Fr(r["D"][1]) for r in G.values())
    argub = min(G.values(), key=lambda r: r["D"][1])["u"]
    us = sorted(Fr(k) for k in G)
    viol = [(str(x1), str(x2)) for x1, x2 in zip(us, us[1:]) if G[str(x2)]["G"][1] < G[str(x1)]["G"][0]]
    res = dict(min_D_certified_lower=math.nextafter(float(lbmin), -math.inf), min_D_lower_exact=str(lbmin),
               worst_cell=[str(a), str(b)], worst_piece=[float(p0), float(p1)], grid_min_D_upper=float(ub),
               grid_argmin=argub, n_points=len(G), covers=[str(us[0]), str(us[-1])],
               gamma_monotone_violations=viol, max_D_width=max(r["D"][1] - r["D"][0] for r in G.values()),
               max_G_width=max(r["G"][1] - r["G"][0] for r in G.values()),
               maxrss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, runtime_s=time.time() - t0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "F_cells4_result.json"), "w"), indent=1)
