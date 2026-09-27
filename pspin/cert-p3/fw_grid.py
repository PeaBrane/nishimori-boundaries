"""Driver: rigorous lower bound  min_{u in [0,1]} D(u)  for mu* at beta=5/2, and the FW bound on F."""
import os, sys, json, time
from fractions import Fraction as Fr
from multiprocessing import Pool
import numpy as np

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))

def work(args):
    u_str, n1 = args
    from sg_lb import D_and_Gamma
    u = Fr(u_str)
    t0 = time.time()
    D, G = D_and_Gamma(u, n1=n1, n2=300, n2c=300)
    return (u_str, n1, float(D.lo), float(D.hi), float(G.lo), float(G.hi), time.time() - t0)

def grid():
    q = Fr(929223, 1000000)
    pts = set()
    u0 = Fr(8, 75)
    # coarse region
    k = 0
    while True:
        u = u0 + Fr(k, 50)
        if u >= Fr(86, 100): break
        pts.add(u); k += 1
    u = Fr(86, 100)
    while u < q:
        pts.add(u); u += Fr(4, 1000)
    pts.add(q)
    u = q + Fr(4, 1000)
    while u < Fr(99, 100):
        pts.add(u); u += Fr(4, 1000)
    pts.add(Fr(1))
    return sorted(pts)

if __name__ == "__main__":
    n1 = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    pts = grid()
    print(len(pts), "grid points", flush=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        res = []
        for r in pool.imap(work, [(str(u), n1) for u in pts]):
            print(f"u={float(Fr(r[0])):.6f} D=[{r[2]:.3e},{r[3]:.3e}] G=[{r[4]:.5f},{r[5]:.5f}] {r[6]:.0f}s", flush=True)
            res.append(r)
    json.dump(res, open(os.path.join(HERE, f'fw_grid_n{n1}.json'), 'w'), indent=1)
