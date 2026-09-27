"""(F) certified lower bound on min_{u in [0,1]} D(u) via grid enclosures + Gamma monotonicity.

On a cell [a,b] (D' = -(1/2) xi''(s) (Gamma(s)-s),  xi''(s) = 3 beta^2 s >= 0,  Gamma nondecreasing):
  L1(u) = D(a) - c3 [Gb (u^2-a^2)/2 - (u^3-a^3)/3]  <= D(u)   (Gamma(s) <= Gb := Gamma(b)_hi)
  L2(u) = D(b) + c3 [Ga (b^2-u^2)/2 - (b^3-u^3)/3]  <= D(u)   (Gamma(s) >= Ga := Gamma(a)_lo)
with c3 = 3 beta^2/2.  L1 is monotone on each side of Gb, L2 on each side of Ga, so on a piece not
containing Ga, Gb in its interior the minima are at endpoints.  All cell arithmetic is exact (Fractions).
Resumable: evaluations cached in F_grid.json.
"""
import json
import os
import sys
import time
from fractions import Fraction as Fr
from multiprocessing import Pool

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "F_grid.json")
q = Fr(929223, 10**6)
C3 = Fr(3, 2) * Fr(25, 4)
NSUB = 64
TOL = Fr(os.environ.get("FTOL", "5/100000000"))


def ev(u):
    import fgrid
    return fgrid.evaluate(Fr(u))


def L1(u, a, Da, Gb):
    return Da - C3 * (Gb * (u * u - a * a) / 2 - (u ** 3 - a ** 3) / 3)


def L2(u, b, Db, Ga):
    return Db + C3 * (Ga * (b * b - u * u) / 2 - (b ** 3 - u ** 3) / 3)


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
    if os.path.exists(GRID):
        return json.load(open(GRID))
    return {}


def save(G):
    tmp = GRID + ".tmp"
    json.dump(G, open(tmp, "w"), indent=0)
    os.replace(tmp, GRID)


def run_points(G, us, pool):
    todo = [str(u) for u in us if str(u) not in G]
    todo.sort(key=lambda s: abs(Fr(s) - q))  # similar lattices together
    t = time.time()
    for r in pool.imap_unordered(ev, todo):
        G[r["u"]] = r
    save(G)
    print(f"  evaluated {len(todo)} points in {time.time() - t:.1f}s", flush=True)


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
    init = [Fr(i, 64) for i in range(0, 60)] + [q] + [q + (1 - q) * Fr(j, 16) for j in range(1, 17)]
    init += [Fr(8, 75), Fr(38, 75)]
    with Pool(10) as pool:
        run_points(G, init, pool)
        for it in range(40):
            cs = cells(G)
            ub = min(Fr(r["D"][1]) for r in G.values())
            lbmin = min(c[0] for c in cs)
            bad = [c for c in cs if c[0] < ub - TOL]
            print(f"iter {it}: points={len(G)} cells={len(cs)} min D_hi(grid)={float(ub):.12e} "
                  f"certified LB={float(lbmin):.12e} cells to refine={len(bad)}", flush=True)
            if not bad:
                break
            run_points(G, [(c[1] + c[2]) / 2 for c in bad], pool)
    cs = cells(G)
    lbmin, a, b, p0, p1 = min(cs)
    ub = min(Fr(r["D"][1]) for r in G.values())
    argub = min(G.values(), key=lambda r: r["D"][1])["u"]
    # Gamma monotonicity consistency check on computed enclosures (necessary condition)
    us = sorted(Fr(k) for k in G)
    viol = [(str(x1), str(x2)) for x1, x2 in zip(us, us[1:]) if G[str(x2)]["G"][1] < G[str(x1)]["G"][0]]
    # write in exact decimal-safe form: lbmin is a Fraction; round down to float
    import math
    lb_float = math.nextafter(float(lbmin), -math.inf)
    res = dict(min_D_certified_lower=lb_float, min_D_lower_exact=str(lbmin), worst_cell=[str(a), str(b)],
               worst_piece=[float(p0), float(p1)], grid_min_D_upper=float(ub), grid_argmin=argub,
               n_points=len(G), gamma_monotone_violations=viol, runtime_s=time.time() - t0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "F_cells_result.json"), "w"), indent=1)
