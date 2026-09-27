"""Brute-force regression checks for the lattice argument (Section 9, Appendix C). Float evidence only.

(a) Off-line sign law + class law of K_H for small H4(b0,k) vs exhaustive enumeration (F(beta,s), several beta, p, s).
(b) Plus-boundary Peierls ingredients (Lemmas C.5 and C.6) samplewise on small boxes with random signed couplings.
(c) Newman's inequality (Lemma 9.7, lem:lat-newman) on random small graphs, and the exact law of a coupling
    that proves it.
(d) Lemma 9.10 (lem:lat-degradation; Bhattacharyya monotonicity) on a non-series-parallel gadget (Wheatstone bridge).
"""
import itertools
import math
import random
import sys
import time

import numpy as np

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

T0 = time.time()
rng = random.Random(20260927)
out = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    out.append(msg)


# ---------------------------------------------------------------- (a)
def h4_graph(b0, k):
    V = {"p0": 0, "p1": 1}
    E = []

    def v(name):
        if name not in V:
            V[name] = len(V)
        return V[name]

    E.append((v("p0"), v("z0")))
    for i in range(1, k + 1):
        zl, zr = v(f"z{i-1}"), v(f"z{i}")
        E.append((zl, zr))                      # e_d
        E.append((zl, v(f"w1_{i}")))            # e_1
        for c in (1, 2):
            prev = v(f"w1_{i}")
            for j in range(1, b0):
                nxt = v(f"path{i}_{c}_{j}")
                E.append((prev, nxt))
                prev = nxt
            E.append((prev, v(f"w2_{i}")))
        E.append((v(f"w2_{i}"), zr))            # e_2
    E.append((v(f"z{k}"), v("p1")))
    return len(V), E


def brute_F(b0, k, p, beta, s_list):
    nV, E = h4_graph(b0, k)
    nE = len(E)
    nI = nV - 2
    # spin configs: poles (+,+) and (+,-), all interior
    I = np.array(list(itertools.product([1, -1], repeat=nI)), dtype=np.int8)
    res = {}
    rows = []
    for pole1 in (1, -1):
        sig = np.empty((I.shape[0], nV), dtype=np.int8)
        sig[:, 0] = 1
        sig[:, 1] = pole1
        sig[:, 2:] = I
        prod = np.stack([sig[:, u] * sig[:, w] for (u, w) in E], axis=1).astype(np.float64)
        rows.append(prod)
    Fs = np.zeros(len(s_list))
    Js = np.array(list(itertools.product([1, -1], repeat=nE)), dtype=np.float64)
    probs = np.prod(np.where(Js < 0, p, 1 - p), axis=1)
    chunk = 512
    for st in range(0, Js.shape[0], chunk):
        Jc = Js[st:st + chunk]
        lz = []
        for prod in rows:
            en = beta * prod @ Jc.T                 # (configs, chunk)
            m = en.max(axis=0)
            lz.append(m + np.log(np.exp(en - m).sum(axis=0)))
        Kh = 0.5 * (lz[0] - lz[1])
        for i, s in enumerate(s_list):
            Fs[i] += np.sum(probs[st:st + chunk] * np.exp(-2 * s * Kh))
    return Fs


def class_F(b0, k, p, beta, s_list):
    th = 1 - 2 * p
    e = th ** b0
    q = (1 - e) / 2
    P1 = 2 * q * (1 - q)
    c = (1 - q) ** 2 + q ** 2
    P = (P1, (c + th ** 3 * e) / 2, (c - th ** 3 * e) / 2)
    N = (th * P1, (th * c + th ** 2 * e) / 2, (th * c - th ** 2 * e) / 2)
    t = math.tanh(beta)
    tau = 2 * t ** (b0 + 2) / (1 + t ** (2 * b0))
    m = (t, (t + tau) / (1 + t * tau), (t - tau) / (1 - t * tau))
    Fs = []
    for s in s_list:
        tot = 0.0
        for n1 in range(k + 1):
            for n2 in range(k + 1 - n1):
                n3 = k - n1 - n2
                M = math.comb(k, n1) * math.comb(k - n1, n2)
                a = M * P[0] ** n1 * P[1] ** n2 * P[2] ** n3
                b = M * th ** 2 * N[0] ** n1 * N[1] ** n2 * N[2] ** n3
                x = t * t * m[0] ** n1 * m[1] ** n2 * m[2] ** n3
                r = (1 - x) / (1 + x)
                tot += (a + b) / 2 * r ** s + (a - b) / 2 * r ** (-s)
        Fs.append(tot)
    return np.array(Fs)


worst = 0.0
ncases = 0
for (b0, k) in ((2, 1), (3, 1), (2, 2)):
    for p in ((0.07, 0.23, 0.41) if k == 1 else (0.07, 0.41)):
        for beta in ((0.35, 1.1, 2.3) if k == 1 else (1.1, 2.3)):
            sl = [0.25, 0.5, 0.8, 1.0]
            fb = brute_F(b0, k, p, beta, sl)
            fc = class_F(b0, k, p, beta, sl)
            worst = max(worst, float(np.max(np.abs(fb - fc) / fb)))
            ncases += len(sl)
say(f"(a) sign+class law vs brute force: H4(2,1),(3,1),(2,2); p in {{.07,.23,.41}}, beta in {{.35,1.1,2.3}} (k=1), p in {{.07,.41}}, beta in {{1.1,2.3}} (k=2); "
    f"{ncases} (beta,p,s) cases; max rel diff of F = {worst:.2e}  [{time.time()-T0:.1f}s]")


# ---------------------------------------------------------------- (b)
def box_check(Lx, Ly, x, trials):
    sites = [(i, j) for i in range(Lx) for j in range(Ly)]
    idx = {v: n for n, v in enumerate(sites)}
    inside = set(sites)
    edges = []  # (a, b) with a an index in box, b an index or None (boundary, spin +1); plus the Z2 edge key
    keys = []
    for (i, j) in sites:
        for (di, dj) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nb = (i + di, j + dj)
            key = frozenset({(i, j), nb})
            if key in keys:
                continue
            keys.append(key)
            edges.append((idx[(i, j)], idx[nb] if nb in inside else None))
    nE = len(edges)
    n = len(sites)
    S = np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.int8)
    prod = np.stack([S[:, a] * (S[:, b] if b is not None else 1) for (a, b) in edges], axis=1)
    Dmat = prod < 0

    def z2_connected(vset):
        vset = set(vset)
        if not vset:
            return True
        st = [next(iter(vset))]
        seen = {st[0]}
        while st:
            (i, j) = st.pop()
            for (di, dj) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nb = (i + di, j + dj)
                if nb in vset and nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        return len(seen) == len(vset)

    # contours: A subset of box, x in A, A connected, Z2 \ A connected (check inside a margin-2 frame)
    frame = [(i, j) for i in range(-2, Lx + 2) for j in range(-2, Ly + 2)]
    contours = []
    others = [v for v in sites if v != x]
    for r in range(len(others) + 1):
        for comb_ in itertools.combinations(others, r):
            A = set(comb_) | {x}
            if not z2_connected(A):
                continue
            if not z2_connected([v for v in frame if v not in A]):
                continue
            dA = [e for e, key in enumerate(keys) if len(key & A) == 1]
            contours.append((frozenset(A), dA))
    xi = idx[x]
    viol_union = viol_single = viol_extract = 0
    maxratio = 0.0
    # extraction check (independent of K)
    ev_any = np.zeros(S.shape[0], dtype=bool)
    evs = []
    for A, dA in contours:
        ev = Dmat[:, dA].all(axis=1)
        evs.append(ev)
        ev_any |= ev
    viol_extract = int(np.sum((S[:, xi] == -1) & ~ev_any))
    for _ in range(trials):
        Kv = np.array([rng.choice([-1, 1]) * rng.expovariate(1.0) * rng.choice([0.3, 1.0, 2.5]) for _ in range(nE)])
        en = (prod * Kv).sum(axis=1)
        w = np.exp(en - en.max())
        mu = w / w.sum()
        pm = float(mu[S[:, xi] == -1].sum())
        tot = 0.0
        for (A, dA), ev in zip(contours, evs):
            SA = float(Kv[dA].sum())
            bnd = min(1.0, math.exp(-2 * SA))
            pr = float(mu[ev].sum())
            if pr > bnd * (1 + 1e-12) + 1e-15:
                viol_single += 1
            tot += bnd
        if pm > tot * (1 + 1e-12) + 1e-15:
            viol_union += 1
        maxratio = max(maxratio, pm / tot if tot > 0 else 0)
    return len(contours), viol_extract, viol_single, viol_union, maxratio


for (Lx, Ly, x, tr) in ((3, 3, (1, 1), 150), (4, 3, (1, 1), 60), (3, 3, (0, 0), 100)):
    nc, ve, vs, vu, mr = box_check(Lx, Ly, x, tr)
    say(f"(b) plus-boundary box {Lx}x{Ly}, x={x}: {nc} contours; extraction failures {ve}; "
        f"single-contour violations {vs}; union-bound violations {vu} over {tr} random signed K; max mu(sx=-1)/sum_A min(1,e^-2S) = {mr:.3f}"
        f"  [{time.time()-T0:.1f}s]")


# ---------------------------------------------------------------- (c)
def rand_graph(nv, ne):
    while True:
        E = set()
        tree = list(range(nv))
        rng.shuffle(tree)
        for i in range(1, nv):
            E.add(tuple(sorted((tree[i], tree[rng.randrange(i)]))))
        allp = [(a, b) for a in range(nv) for b in range(a + 1, nv) if (a, b) not in E]
        rng.shuffle(allp)
        for e in allp[: max(0, ne - len(E))]:
            E.add(e)
        return sorted(E)


def gibbs(nv, E, K, pin):
    """Exact law on {+-1}^nv with pinned vertices dict pin; returns dict config->prob."""
    free = [v for v in range(nv) if v not in pin]
    law = {}
    Z = 0.0
    for vals in itertools.product([1, -1], repeat=len(free)):
        s = [0] * nv
        for v, val in pin.items():
            s[v] = val
        for v, val in zip(free, vals):
            s[v] = val
        w = math.exp(sum(k * s[a] * s[b] for (a, b), k in zip(E, K)))
        law[tuple(s)] = w
        Z += w
    return {c: w / Z for c, w in law.items()}


def conn_prob(nv, E, pe, S, D):
    tot = 0.0
    for om in itertools.product([0, 1], repeat=len(E)):
        pr = 1.0
        for o, q in zip(om, pe):
            pr *= q if o else 1 - q
        if pr == 0:
            continue
        par = list(range(nv))

        def f(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for o, (a, b) in zip(om, E):
            if o:
                par[f(a)] = f(b)
        rootsD = {f(d) for d in D}
        if any(f(s) in rootsD for s in S):
            tot += pr
    return tot


viol = 0
ntrip = 0
minslack = 1.0
for g in range(60):
    nv = rng.randint(4, 7)
    ne = rng.randint(nv - 1, min(nv * (nv - 1) // 2, 10))
    E = rand_graph(nv, ne)
    K = [rng.choice([-1, 1]) * rng.uniform(0, 2.0) for _ in E]
    pe = [1 - math.exp(-2 * abs(k)) for k in K]
    verts = list(range(nv))
    rng.shuffle(verts)
    D = verts[: rng.randint(1, 2)]
    rest = verts[len(D):]
    S = rest[: rng.randint(1, min(2, len(rest)))]
    phi = conn_prob(nv, E, pe, S, D)
    laws = {eta: gibbs(nv, E, K, dict(zip(D, eta))) for eta in itertools.product([1, -1], repeat=len(D))}
    for zeta in itertools.product([1, -1], repeat=len(S)):
        vals = {eta: sum(pr for c, pr in law.items() if all(c[s] == z for s, z in zip(S, zeta))) for eta, law in laws.items()}
        for e1, e2 in itertools.combinations(vals, 2):
            ntrip += 1
            diff = abs(vals[e1] - vals[e2])
            if diff > phi + 1e-12:
                viol += 1
            minslack = min(minslack, phi - diff)
say(f"(c1) Newman inequality (Lemma 9.7), 60 random signed graphs (4-7 vertices, <=10 edges): {ntrip} (eta,eta',f) triples, violations {viol}, min slack {minslack:.2e}")


# exact law of the coupling (shared Bernoulli bits and shared terminal uniform; independent pinning draws)
def coupling_law(nv, E, K, D, eta1, eta2):
    pe = [1 - math.exp(-2 * abs(k)) for k in K]
    joint = {}

    def mu_R(R, xi, X):
        pin = dict(zip(R, xi))
        Eact = [(e, k) for e, k, i in zip(E, K, range(len(E))) if i not in X]
        return gibbs(nv, [e for e, _ in Eact], [k for _, k in Eact], pin)

    def rec(R, X, xi1, xi2, bits, w):
        R = list(R)
        bd = [i for i, (a, b) in enumerate(E) if i not in X and ((a in R) != (b in R))]
        if not bd:
            # terminal: free Ising on V \ R, same draw for both runs
            comp = [v for v in range(nv) if v not in R]
            Ein = [(e, k) for e, k in zip(E, K) if e[0] in comp and e[1] in comp]
            law = gibbs(nv, [e for e, _ in Ein], [k for _, k in Ein], {r: 1 for r in R})
            for c, pr in law.items():
                s1 = list(c); s2 = list(c)
                for r, a1, a2 in zip(R, xi1, xi2):
                    s1[r] = a1; s2[r] = a2
                key = (tuple(s1), tuple(s2), tuple(sorted(R)))
                joint[key] = joint.get(key, 0.0) + w * pr
            return
        i = min(bd)  # spin-blind rule
        a, b = E[i]
        u, v = (a, b) if a in R else (b, a)
        if pe[i] < 1:
            rec(R, X | {i}, xi1, xi2, bits, w * (1 - pe[i]))
        if pe[i] > 0:
            rs = []
            for xi in (xi1, xi2):
                m0 = mu_R(R, xi, X)
                m1 = mu_R(R, xi, X | {i})
                r = {}
                for xx in (1, -1):
                    a0 = sum(pr for c, pr in m0.items() if c[v] == xx)
                    a1 = sum(pr for c, pr in m1.items() if c[v] == xx)
                    r[xx] = (a0 - (1 - pe[i]) * a1) / pe[i]
                rs.append(r)
            for x1 in (1, -1):
                for x2 in (1, -1):
                    ww = rs[0][x1] * rs[1][x2]
                    if ww != 0:
                        rec(R + [v], X | {i}, tuple(xi1) + (x1,), tuple(xi2) + (x2,), bits, w * pe[i] * ww)

    rec(list(D), frozenset(), tuple(eta1), tuple(eta2), (), 1.0)
    return joint, pe


maxmarg = maxoff = maxconn = 0.0
minr = 1.0
for g in range(25):
    nv = rng.randint(4, 6)
    ne = rng.randint(nv - 1, min(nv * (nv - 1) // 2, 8))
    E = rand_graph(nv, ne)
    K = [rng.choice([-1, 1]) * rng.uniform(0.05, 2.0) for _ in E]
    D = [0] if rng.random() < 0.5 else [0, 1]
    eta1 = tuple(rng.choice([1, -1]) for _ in D)
    eta2 = tuple(rng.choice([1, -1]) for _ in D)
    joint, pe = coupling_law(nv, E, K, D, eta1, eta2)
    m1 = gibbs(nv, E, K, dict(zip(D, eta1)))
    m2 = gibbs(nv, E, K, dict(zip(D, eta2)))
    a1 = {}; a2 = {}
    for (s1, s2, R), pr in joint.items():
        a1[s1] = a1.get(s1, 0) + pr
        a2[s2] = a2.get(s2, 0) + pr
        minr = min(minr, pr)
        off = [v for v in range(nv) if v not in R and s1[v] != s2[v]]
        if off:
            maxoff = max(maxoff, pr)
    maxmarg = max(maxmarg, max(abs(a1.get(c, 0) - m1[c]) for c in m1), max(abs(a2.get(c, 0) - m2[c]) for c in m2))
    for S in ([v] for v in range(nv) if v not in D):
        pRS = sum(pr for (s1, s2, R), pr in joint.items() if S[0] in R)
        maxconn = max(maxconn, abs(pRS - conn_prob(nv, E, pe, S, D)))
say(f"(c2) exact law of the Newman coupling on 25 random signed graphs: max marginal error {maxmarg:.1e}; "
    f"max mass with disagreement off R_T {maxoff:.1e}; max |P(v in R_T) - Phi_p(v<->D)| {maxconn:.1e}; min atom mass {minr:.1e}  [{time.time()-T0:.1f}s]")

# ---------------------------------------------------------------- (d)
# Wheatstone bridge (not series-parallel): vertices 0=pi0, 1=pi1, 2=a, 3=b.
Eb = [(0, 2), (0, 3), (2, 3), (2, 1), (3, 1)]


def KH(J, beta):
    Z = {}
    for s1 in (1, -1):
        tot = 0.0
        for a, b in itertools.product([1, -1], repeat=2):
            s = [1, s1, a, b]
            tot += math.exp(beta * sum(j * s[u] * s[v] for (u, v), j in zip(Eb, J)))
        Z[s1] = tot
    return 0.5 * math.log(Z[1] / Z[-1])


def v_iid(p):
    g = 0.5 * math.log((1 - p) / p)
    tot = 0.0
    for J in itertools.product([1, -1], repeat=len(Eb)):
        pr = math.prod(p if j < 0 else 1 - p for j in J)
        tot += pr * math.exp(-KH(J, g))
    return tot


def Z_channel(p):
    tot = 0.0
    for J in itertools.product([1, -1], repeat=len(Eb)):
        W = {}
        for x in (1, -1):
            acc = 0.0
            cnt = 0
            for tau in itertools.product([1, -1], repeat=4):
                if tau[0] * tau[1] != x:
                    continue
                cnt += 1
                acc += math.prod((1 - p) if (j * tau[u] * tau[v]) > 0 else p for (u, v), j in zip(Eb, J))
            W[x] = acc / cnt
        tot += math.sqrt(W[1] * W[-1])
    return tot


ps = [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.27, 0.35, 0.45]
vs = [v_iid(p) for p in ps]
zs = [Z_channel(p) for p in ps]
say(f"(d) Wheatstone bridge (non-SP): max |Z(W_p) - v_H(gamma(p))| = {max(abs(a-b) for a,b in zip(vs,zs)):.1e}; "
    f"v nondecreasing on grid {all(vs[i] <= vs[i+1] for i in range(len(vs)-1))}: {[round(x,5) for x in vs]}")
say(f"done [{time.time()-T0:.1f}s]")
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
