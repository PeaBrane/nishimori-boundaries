"""Float cross-check by direct Monte Carlo sampling of the microscopic disorder (a population
sampler for the law of K_H), with a numerical series/parallel reduction that does not use the
class decomposition or the (s,Z) closed form:
  * every edge: K = beta*J;  path signs sampled as the parity of Binomial(b0, p) negative edges;
  * series via the dual coupling D(K) = atanh(e^{-2|K|}) (tanh|K| = e^{-2D}): D adds in series,
    sign multiplies;  parallel: signed K adds;
  * u = E tanh|K_H|, pbar = E[1 - e^{-2|K_H|}], v = E e^{-K_H}.
Compared with the exact closed-form values (h4_float, mpmath).
Usage: python mc_crosscheck.py [samples_per_beta]"""
import sys
import time

import numpy as np
from mpmath import mpf, log as mlog

import h4_float as HF

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")


def dual(K):
    """D with tanh D = e^{-2|K|}; involution; stable for small and large |K| (inf at K = 0)."""
    # atanh(y) = (1/2) log1p(2y/(1-y)) and y/(1-y) = 1/expm1(2|K|): no cancellation at any |K|
    with np.errstate(divide="ignore"):
        return 0.5 * np.log1p(2 / np.expm1(2 * np.abs(K)))


primal = dual   # the map is an involution


def sample(rng, b0, k, p, beta, M):
    dB = float(dual(np.array(beta)))
    Kpath = float(primal(np.array(b0 * dB)))               # |K| of a path of b0 edges
    sig = 1 - 2 * (rng.binomial(b0, p, size=(M, k, 2)) & 1)  # path sign products
    Kpp = Kpath * sig.sum(axis=2)                          # two paths in parallel
    Ja = 1 - 2 * (rng.random((M, k)) < p)
    Jb = 1 - 2 * (rng.random((M, k)) < p)
    Je = 1 - 2 * (rng.random((M, k)) < p)
    Dbr = 2 * dB + dual(Kpp)                               # e - (P||P) - e in series
    Kbr = Ja * Jb * np.sign(Kpp) * primal(Dbr)
    KU = Je * beta + Kbr                                   # direct edge in parallel
    DH = 2 * dB + dual(KU).sum(axis=1)                     # terminal edges + chain of k units
    sH = (1 - 2 * (rng.random(M) < p)) * (1 - 2 * (rng.random(M) < p)) * np.prod(np.sign(KU), axis=1)
    KH = sH * primal(DH)
    return np.exp(-2 * DH), 1 - np.exp(-2 * np.abs(KH)), np.exp(-KH)


def run(name, b0, k, pq, betas, M, rng, chunk=200000):
    p = pq[0] / pq[1]
    pm = mpf(pq[0]) / pq[1]
    gam = float(mlog((1 - pm) / pm) / 2)
    print(f"=== {name}: b0={b0} k={k} p={pq[0]}/{pq[1]} gamma={gam:.10f} samples/beta={M}")
    worst = 0.0
    for lab, beta in betas:
        beta = beta if lab.startswith("abs") else gam * beta
        acc = [[], [], []]
        for i in range(0, M, chunk):
            out = sample(rng, b0, k, p, beta, min(chunk, M - i))
            for a, o in zip(acc, out):
                a.append(o)
        u, pb, v = (np.concatenate(a) for a in acc)
        F = HF.functionals(pm, b0, k, mpf(beta), warm=lab.startswith("gamma"))
        line = f"  beta={beta:.6f} ({beta/gam:.4f} gamma):"
        for key, arr, ref in (("u", u, F["uH"]), ("pbar", pb, F["pbar"]), ("v", v, F["v"])):
            if key == "v" and not lab.startswith("gamma"):
                continue
            m, se = arr.mean(), arr.std(ddof=1) / np.sqrt(arr.size)
            z = (m - float(ref)) / se
            worst = max(worst, abs(z))
            line += f"  {key}: MC {m:.6f}+-{se:.1e} exact {float(ref):.6f} (z={z:+.2f})"
        print(line)
        sys.stdout.flush()
    return worst


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    rng = np.random.default_rng(12345)
    t0 = time.time()
    betas = [("gamma", 1.0), ("1.9924g", 1.9924), ("2.1163g", 2.1163), ("2.2g", 2.2), ("2.255g", 2.255),
             ("3g", 3.0), ("5g", 5.0), ("abs60", 60.0)]
    w = 0.0
    w = max(w, run("P", 1000, 10, (9, 10000), betas, M, rng))
    w = max(w, run("E", 1500, 10, (9, 10000), betas, M, rng))
    w = max(w, run("B", 1500, 8, (87373, 10 ** 8), betas, M, rng))
    print(f"max |z| over all comparisons = {w:.2f}  (float cross-check; |z| < ~4 expected)   total {time.time()-t0:.0f}s")
