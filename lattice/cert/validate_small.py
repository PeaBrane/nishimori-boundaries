"""Brute-force validation of the closed form used by the certificates, on small instances of the
same gadget family H4(b0, k) = e - S_k(U) - e, U = e || (e - (P_b0 || P_b0) - e).

The gadget is built as an explicit edge list; K_H(J, beta) = (1/2) log(Z_{++}/Z_{+-}) is computed by
summing over all interior spin configurations (log-sum-exp).  Checks:
  (1) samplewise: brute-force K_H vs the closed-form map J -> class -> tanh K_H (s,Z form),
      random J, several beta, b0 in {2,3,4}, k in {1,2,3};
  (2) law: for tiny instances, exhaustive enumeration of all J (exact weights p^{#neg}(1-p)^{#pos})
      with brute-force K_H, giving u_H, pbar_H, E e^{-K_H}, E sech K_H, compared with the
      closed-form class law of h4_float.functionals (multinomial + rho), at several beta
      including beta = gamma(p) (where E e^{-K} = E sech K must hold).
Runs locally in well under a minute and under 200 MB.
"""
import itertools
import math

import numpy as np
from mpmath import mp, mpf

import h4_float as HF
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.dps = 30


def build(b0, k):
    """Return (n_vertices, edges, index info). Vertex 0 = pole X, 1 = pole Y."""
    edges, info = [], dict(units=[])
    nv = 2
    chain = list(range(nv, nv + k + 1))
    nv += k + 1

    def add(a, b):
        edges.append((a, b))
        return len(edges) - 1

    info["term"] = (add(0, chain[0]), add(chain[k], 1))
    for i in range(k):
        c0, c1 = chain[i], chain[i + 1]
        A, B = nv, nv + 1
        nv += 2
        u = dict(direct=add(c0, c1), ea=add(c0, A), eb=add(B, c1), paths=[])
        for _ in range(2):
            prev, pe = A, []
            for j in range(b0 - 1):
                pe.append(add(prev, nv))
                prev = nv
                nv += 1
            pe.append(add(prev, B))
            u["paths"].append(pe)
        info["units"].append(u)
    return nv, edges, info


def bf_K(nv, edges, Jmat, beta):
    """Brute-force K_H for each column of Jmat (edges x samples)."""
    ni = nv - 2
    spins = 1 - 2 * ((np.arange(2 ** ni)[:, None] >> np.arange(ni)[None, :]) & 1)   # interior spins
    out = []
    for sy in (1, -1):
        full = np.concatenate([np.ones((2 ** ni, 1), dtype=np.int64), sy * np.ones((2 ** ni, 1), dtype=np.int64), spins], axis=1)
        prod = np.stack([full[:, a] * full[:, b] for a, b in edges], axis=1).astype(np.float64)  # configs x edges
        with np.errstate(all="ignore"):   # macOS Accelerate matmul raises spurious FP flags
            E = beta * (prod @ Jmat)      # configs x samples
        assert np.isfinite(E).all()
        m = E.max(axis=0)
        out.append(m + np.log(np.exp(E - m).sum(axis=0)))
    K = 0.5 * (out[0] - out[1])
    assert np.isfinite(K).all()
    return K


def closed_tanh(info, J, beta, b0):
    t, w1, w3 = HF.mags(mpf(beta), b0)
    T = J[info["term"][0]] * J[info["term"][1]] * t * t
    for u in info["units"]:
        Je = J[u["direct"]]
        s1 = int(np.prod(J[u["paths"][0]]))
        s2 = int(np.prod(J[u["paths"][1]]))
        if s1 != s2:
            mag = t
        else:
            mag = w1 if J[u["ea"]] * J[u["eb"]] * s1 == Je else w3
        T *= Je * mag
    return T


def check_samplewise(rng):
    worst = 0.0
    n = 0
    for b0, k in [(2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (4, 1)]:
        nv, edges, info = build(b0, k)
        ns = 40
        Jmat = np.where(rng.random((len(edges), ns)) < 0.35, -1, 1).astype(np.float64)
        for beta in (0.3, 0.9, 2.2):
            Kb = bf_K(nv, edges, Jmat, beta)
            for j in range(ns):
                Tc = closed_tanh(info, Jmat[:, j].astype(int), beta, b0)
                Kc = float(mp.atanh(Tc))
                worst = max(worst, abs(Kc - Kb[j]))
                n += 1
    return n, worst


def check_law():
    rows = []
    worst = 0.0
    for b0, k, p in [(2, 1, 0.2), (3, 1, 0.15), (2, 2, 0.2)]:
        nv, edges, info = build(b0, k)
        m = len(edges)
        allJ = np.array(list(itertools.product([1, -1], repeat=m)), dtype=np.float64).T      # m x 2^m
        nneg = (allJ < 0).sum(axis=0)
        wts = p ** nneg * (1 - p) ** (m - nneg)
        gam = 0.5 * math.log((1 - p) / p)
        for beta in (0.4, gam, 1.7, 3.0):
            K = np.concatenate([bf_K(nv, edges, allJ[:, i:i + 1024], beta) for i in range(0, allJ.shape[1], 1024)])
            bf = dict(uH=(wts * np.abs(np.tanh(K))).sum(), pbar=(wts * (1 - np.exp(-2 * np.abs(K)))).sum(),
                      v=(wts * np.exp(-K)).sum(), vsech=(wts / np.cosh(K)).sum())
            F = HF.functionals(mpf(p), b0, k, mpf(beta), warm=True)
            d = {key: abs(float(F[key]) - bf[key]) for key in bf}
            worst = max(worst, max(d.values()))
            rows.append((b0, k, p, beta, bf, {key: float(F[key]) for key in bf}, d))
    return rows, worst


if __name__ == "__main__":
    rng = np.random.default_rng(20260926)
    n, w = check_samplewise(rng)
    print(f"(1) samplewise K_H: {n} (J, beta) cases, max |K_bruteforce - K_closed| = {w:.3e}")
    rows, w2 = check_law()
    for b0, k, p, beta, bf, cf, d in rows:
        print(f"(2) b0={b0} k={k} p={p} beta={beta:.6f}: brute u={bf['uH']:.12f} pbar={bf['pbar']:.12f} Ee^-K={bf['v']:.12f} Esech={bf['vsech']:.12f}"
              f" | closed u={cf['uH']:.12f} pbar={cf['pbar']:.12f} Ee^-K={cf['v']:.12f} | max diff {max(d.values()):.2e}")
    print(f"(2) law check max abs diff = {w2:.3e}")
    print("VALIDATION", "PASSED" if (w < 1e-9 and w2 < 1e-10) else "FAILED")
