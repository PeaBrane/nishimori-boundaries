"""Compare this implementation's rigorous D, Gamma enclosures with the primary grid (comparison only)."""
import os
import json, time
from fractions import Fraction as Fr
from multiprocessing import Pool
from fcells import ev
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fw_grid_n600.json")
if __name__ == "__main__":
    t = time.time()
    gen = json.load(open(PATH))
    with Pool(10) as p:
        mine = p.map(ev, [r[0] for r in gen])
    rows, bad = [], []
    for g, m in zip(gen, mine):
        okD = g[2] <= m["D"][0] and m["D"][1] <= g[3]
        okG = g[4] <= m["G"][0] and m["G"][1] <= g[5]
        rows.append(dict(u=g[0], gen_D=g[2:4], my_D=m["D"], gen_G=g[4:6], my_G=m["G"], D_inside=okD, G_inside=okG))
        if not (okD and okG):
            bad.append(g[0])
    res = dict(n=len(rows), not_contained=bad, max_my_D_width=max(r["my_D"][1]-r["my_D"][0] for r in rows),
               min_gen_D_width=min(r["gen_D"][1]-r["gen_D"][0] for r in rows), runtime_s=time.time()-t, rows=rows)
    json.dump(res, open(os.path.join(HERE, "F_compare.json"), "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})
