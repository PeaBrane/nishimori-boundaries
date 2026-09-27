"""Part (a), float evidence: exhaustive spin sums on small instances of the unit U and of H4(b0,k),
compared with the law derived in model.py (which is what the rigorous checks evaluate).

Run: python bruteforce.py > logs/bruteforce.log
"""
import sys
from fractions import Fraction
from itertools import product

import numpy as np

import model as M

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")


def build_unit(b0):
    """Unit U with poles alpha=0, omega=1. Returns (n_vertices, edges, labels)."""
    V = 4  # 0 alpha, 1 omega, 2 w1, 3 w2
    edges, lab = [], []
    edges.append((0, 1)); lab.append("d")
    edges.append((0, 2)); lab.append("1")
    for c in (1, 2):
        prev = 2
        for j in range(1, b0):
            edges.append((prev, V)); lab.append(f"c{c}")
            prev = V
            V += 1
        edges.append((prev, 3)); lab.append(f"c{c}")
    edges.append((3, 1)); lab.append("2")
    return V, edges, lab


def build_h4(b0, k):
    """H4(b0,k) with poles 0 and 1."""
    V = 2
    z = []
    for _ in range(k + 1):
        z.append(V)
        V += 1
    edges = [(0, z[0])]
    for i in range(k):
        a, o = z[i], z[i + 1]
        w1, w2 = V, V + 1
        V += 2
        edges.append((a, o))
        edges.append((a, w1))
        for _c in (1, 2):
            prev = w1
            for _j in range(1, b0):
                edges.append((prev, V))
                prev = V
                V += 1
            edges.append((prev, w2))
        edges.append((w2, o))
    edges.append((z[k], 1))
    return V, edges


def K_exhaustive(V, edges, beta, Jall):
    """K_H for each row of Jall (shape [nJ, nE]) by summing over interior spins, poles 0,1."""
    nint = V - 2
    # integer arithmetic for the energies (exact, and avoids BLAS entirely)
    sig = np.array(list(product((1, -1), repeat=nint)), dtype=np.int64) if nint else np.zeros((1, 0), dtype=np.int64)
    Jint = Jall.astype(np.int64)
    out = []
    for s1 in (1, -1):
        full = np.concatenate([np.ones((sig.shape[0], 1), dtype=np.int64), s1 * np.ones((sig.shape[0], 1), dtype=np.int64), sig], axis=1)
        S = np.stack([full[:, u] * full[:, v] for (u, v) in edges], axis=1)  # [nsig, nE]
        logZ = []
        for st in range(0, Jall.shape[0], 1024):
            E = beta * (S @ Jint[st:st + 1024].T).astype(np.float64)  # [nsig, chunk]
            mx = E.max(axis=0)
            logZ.append(mx + np.log(np.exp(E - mx).sum(axis=0)))
        out.append(np.concatenate(logZ))
    return 0.5 * (out[0] - out[1])


def unit_check(b0, beta):
    V, edges, lab = build_unit(b0)
    nE = len(edges)
    Jall = np.array(list(product((1, -1), repeat=nE)), dtype=np.float64)
    Kb = K_exhaustive(V, edges, beta, Jall)
    assert np.isfinite(Kb).all()
    th = np.tanh(beta)
    kap = np.arctanh(th**b0)
    kap2 = np.arctanh(th**2 * np.tanh(2 * kap))
    worst = 0.0
    sign_ok = True
    for row, Kv in zip(Jall, Kb):
        sd = row[lab.index("d")]
        s1 = row[lab.index("1")]
        s2 = row[lab.index("2")]
        c1 = np.prod([row[i] for i, l in enumerate(lab) if l == "c1"])
        c2 = np.prod([row[i] for i, l in enumerate(lab) if l == "c2"])
        # model.py derivation: K_U = sd*beta + b*kappa2, b = s1 s2 (c1+c2)/2
        b = s1 * s2 * (c1 + c2) / 2
        Kf = sd * beta + b * kap2
        # Proposition 9.3(i) literally: beta s_d + s1 s2 atanh(theta^2 tanh((c1+c2) kappa))
        Kp = sd * beta + s1 * s2 * np.arctanh(th**2 * np.tanh((c1 + c2) * kap))
        worst = max(worst, abs(Kv - Kf), abs(Kv - Kp))
        sign_ok &= bool(np.sign(Kv) == sd)
    return worst, sign_ok, float(kap2 < beta)


def law_from_bruteforce(b0, k, p, beta):
    V, edges = build_h4(b0, k)
    nE = len(edges)
    Jall = np.array(list(product((1, -1), repeat=nE)), dtype=np.float64)
    Kb = K_exhaustive(V, edges, beta, Jall)
    assert np.isfinite(Kb).all()
    nneg = (Jall < 0).sum(axis=1)
    pr = p**nneg * (1 - p) ** (nE - nneg)
    return Kb, pr


def law_from_model(b0, k, p, beta):
    L = M.Law(Fraction(p).limit_denominator(10**9), b0=b0, k=k)
    th = np.tanh(beta)
    t = th**b0
    tau = 2 * th * th * t / (1 + t * t)
    m = (th, (th + tau) / (1 + th * tau), (th - tau) / (1 - th * tau))
    Ks, ps = [], []
    for n, (Np, Nm) in L.atoms.items():
        x = th**2 * m[0] ** n[0] * m[1] ** n[1] * m[2] ** n[2]
        Ks += [np.arctanh(x), -np.arctanh(x)]
        ps += [Np / L.D, Nm / L.D]
    return np.array(Ks), np.array(ps)


def compare(K1, p1, K2, p2, tol=1e-9):
    """Max discrepancy of the two discrete laws: compare CDF-free by merging atoms within tol."""
    allK = np.sort(np.concatenate([K1, K2]))
    reps = [allK[0]]
    for v in allK[1:]:
        if v - reps[-1] > tol:
            reps.append(v)
    reps = np.array(reps)

    def mass(K, p):
        idx = np.searchsorted(reps, K - tol)
        out = np.zeros(len(reps))
        np.add.at(out, idx, p)
        return out

    return float(np.max(np.abs(mass(K1, p1) - mass(K2, p2)))), len(reps)


def main():
    print("# bruteforce.py: exhaustive spin sums vs model.py law (float evidence)")
    print("## unit U: K_U from exhaustive interior sums vs derived formula, all 2^(2b0+3) sign configurations")
    for b0 in (2, 3, 4, 5):
        for beta in (0.37, 1.1, 2.3):
            w, sgn, k2lt = unit_check(b0, beta)
            print(f"unit b0={b0} beta={beta}: max|K_brute - K_formula| = {w:.2e}; sign K_U = s_d always: {sgn}; kappa2<beta: {bool(k2lt)}")
    print("## gadget H4(b0,k): law of K_H (signed atoms) from exhaustive sums vs model.Law DP")
    worst = 0.0
    for (b0, k) in ((2, 1), (3, 1), (2, 2), (4, 1)):
        for p in (0.1, 0.23):
            for beta in (0.4, 1.3, 2.2):
                Kb, pb = law_from_bruteforce(b0, k, p, beta)
                Km, pm = law_from_model(b0, k, p, beta)
                d, nat = compare(Kb, pb, Km, pm)
                worst = max(worst, d)
                # Nishimori identities at gamma are checked separately below
                print(f"H4({b0},{k}) p={p} beta={beta}: max atom-mass discrepancy {d:.2e} over {nat} distinct K values; total mass brute {pb.sum():.15f}")
    print(f"WORST gadget law discrepancy: {worst:.2e}")
    print("## Nishimori identities at beta = gamma(p) from exhaustive sums")
    for (b0, k) in ((2, 1), (3, 1), (2, 2)):
        for p in (0.1, 0.23):
            g = 0.5 * np.log((1 - p) / p)
            Kb, pb = law_from_bruteforce(b0, k, p, g)
            e1 = abs(np.sum(pb * np.tanh(Kb)) - np.sum(pb * np.tanh(Kb) ** 2))
            e2 = abs(np.sum(pb * np.exp(-Kb)) - np.sum(pb / np.cosh(Kb)))
            Km, pm = law_from_model(b0, k, p, g)
            e3 = abs(np.sum(pb * np.exp(-Kb)) - np.sum(pm * np.exp(-Km)))
            print(f"H4({b0},{k}) p={p}: |E tanh K - E tanh^2 K| = {e1:.1e}, |E e^-K - E sech K| = {e2:.1e}, |v_brute - v_model| = {e3:.1e}")


if __name__ == "__main__":
    sys.exit(main())
