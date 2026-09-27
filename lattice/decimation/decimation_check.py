"""Evidence (floats/mpmath; not a certificate) for the law of Proposition 9.3 (prop:lat-law) on H4.

1. Real-graph series-parallel decimation of H4(b0,k) for random disorders vs the closed form of Proposition 9.3.
2. Exact kappa(u) for every interior vertex of H4 via the Wheatstone-bridge reduction of one unit
   (exact finite disorder mixture at gamma), vs the chain bound of Lemma C.11; exact kappa-bar.
3. v_H(gamma) from the signed law (own DP), t_U, u_H(beta), pbar_H(beta) floats, pbar(infinity).
usage: decimation_check.py b0 k p_num p_den
"""
import sys, itertools, random, math
from fractions import Fraction as Fr
from mpmath import mp, mpf, tanh, atanh, log, exp, sqrt

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

b0, k = int(sys.argv[1]), int(sys.argv[2])
pF = Fr(int(sys.argv[3]), int(sys.argv[4]))
mp.dps = 60
p = mpf(pF.numerator) / pF.denominator
g = log((1 - p) / p) / 2
vt = 1 - 2 * p
print(f"== b0={b0} k={k} p={pF} gamma={mp.nstr(g, 12)}")

# exact P's
vtF = 1 - 2 * pF
qF = (1 - vtF ** b0) / 2
aF = (1 + vtF ** 3) / 2
P1F = 2 * qF * (1 - qF); P3F = (1 - qF) ** 2 * (1 - aF) + qF * qF * aF; P2F = 1 - P1F - P3F
P1, P2, P3 = (mpf(x.numerator) / x.denominator for x in (P1F, P2F, P3F))
print(f"P1={mp.nstr(P1, 12)} P2={mp.nstr(P2, 12)} P3={mp.nstr(P3, 12)} q={float(qF):.12f}")


def closed_unit(beta):
    th = tanh(beta); kap = atanh(th ** b0); k2 = atanh(th * th * tanh(2 * kap))
    return th, kap, k2, [th, tanh(beta + k2), tanh(beta - k2)]


# ---------- 1. real-graph SP decimation ----------
def build(b0, k):
    E = []  # (a, b, role, unit)
    n = 2
    z = list(range(n, n + k + 1)); n += k + 1
    E.append((0, z[0], 'ta', -1)); E.append((z[k], 1, 'tb', -1))
    for i in range(1, k + 1):
        w1, w2 = n, n + 1; n += 2
        E.append((z[i - 1], z[i], 'd', i)); E.append((z[i - 1], w1, 'c1', i)); E.append((w2, z[i], 'c2', i))
        for c in (1, 2):
            prev = w1
            for j in range(1, b0):
                E.append((prev, n, 'p%d' % c, i)); prev = n; n += 1
            E.append((prev, w2, 'p%d' % c, i))
    return n, E


def sp_reduce(n, edges, K):
    adj = {v: {} for v in range(n)}  # v -> {eid: other}
    Kd = {}
    for eid, ((a, b, _, _), kk) in enumerate(zip(edges, K)):
        Kd[eid] = kk; adj[a][eid] = b; adj[b][eid] = a
    nxt = len(edges)
    stack = [v for v in range(2, n)]
    while stack:
        v = stack.pop()
        if v not in adj or v in (0, 1):
            continue
        if len(adj[v]) == 2:
            (e1, a), (e2, b) = adj[v].items()
            kk = atanh(tanh(Kd[e1]) * tanh(Kd[e2]))
            del adj[a][e1]; del adj[b][e2]; del adj[v]; del Kd[e1]; del Kd[e2]
            # parallel merge with existing a-b edge
            ex = [e for e, o in adj[a].items() if o == b]
            if ex:
                Kd[ex[0]] += kk
            else:
                Kd[nxt] = kk; adj[a][nxt] = b; adj[b][nxt] = a; nxt += 1
            stack.extend([a, b])
    assert len(adj[0]) == 1 and list(adj[0].values())[0] == 1, (len(adj), len(adj[0]))
    return Kd[list(adj[0].keys())[0]]


def closed_KH(edges, J, beta):
    th, kap, k2, _ = closed_unit(beta)
    tt = mpf(1)
    units = {}
    for (a, b, role, i), j in zip(edges, J):
        if role in ('ta', 'tb'):
            tt *= j * th
        else:
            units.setdefault(i, {'p1': 1, 'p2': 1})
            if role in ('p1', 'p2'):
                units[i][role] *= j
            else:
                units[i][role] = j
    for i, u in units.items():
        KU = beta * u['d'] + atanh(u['c1'] * u['c2'] * th * th * tanh((u['p1'] + u['p2']) * kap))
        tt *= tanh(KU)
    return atanh(tt)


rng = random.Random(12345)
n, edges = build(b0, k)
print(f"graph: {n} vertices, {len(edges)} edges, I={n-2}")
worst = mpf(0)
for beta in (g, 2.15 * g, 3 * g):
    for s in range(3):
        J = [(-1 if rng.random() < float(p) else 1) for _ in edges]
        if s == 0:  # force cancel events in several units: paths both negative, direct +, connectors +
            J = [1] * len(edges)
            for idx, (a, b, role, i) in enumerate(edges):
                if role in ('p1', 'p2') and i % 2 == 1 and b - a == 1 and a > 2 * k + 5:
                    pass
            # make every odd unit a cancel unit: flip the first edge of each path
            seen = set()
            for idx, (a, b, role, i) in enumerate(edges):
                if role in ('p1', 'p2') and i % 2 == 1 and (i, role) not in seen:
                    J[idx] = -1; seen.add((i, role))
        K = [beta * j for j in J]
        k_sp = sp_reduce(n, edges, K)
        k_cf = closed_KH(edges, J, beta)
        d = abs(k_sp - k_cf)
        worst = max(worst, d)
        print(f"  beta={mp.nstr(beta/g,4)}g sample {s}: K_SP={mp.nstr(k_sp, 15)} K_closed={mp.nstr(k_cf, 15)} |diff|={mp.nstr(d, 3)}")
print(f"1. worst |K_SP - K_closed| = {mp.nstr(worst, 3)}")

# ---------- 2. exact kappa via bridge reduction (float64, full j range) ----------
pf = float(p); vtf = 1 - 2 * pf; gf = 0.5 * math.log((1 - pf) / pf)


def path_law(L):
    t = vtf ** L
    return [(math.atanh(t), (1 + t) / 2), (-math.atanh(t), (1 - t) / 2)]


SPIN = {}


def e_tanh(nodes_free, target, edge_list):
    """E tanh K_eff between node 'A' (fixed +) and target, independent edge laws; returns (E t, E t^2)."""
    laws = [l for _, _, l in edge_list]
    key = (tuple(nodes_free), target, tuple((a, b) for a, b, _ in edge_list))
    if key not in SPIN:
        cfg = []
        for st in (1, -1):
            for fr in itertools.product((1, -1), repeat=len(nodes_free)):
                s = dict(zip(nodes_free, fr)); s['A'] = 1; s[target] = st
                cfg.append((st, [s[a] * s[b] for a, b, _ in edge_list]))
        SPIN[key] = cfg
    cfg = SPIN[key]
    tot = tot2 = 0.0
    for combo in itertools.product(*laws):
        w = math.prod(pr for _, pr in combo)
        ks = [kk for kk, _ in combo]
        mx = sum(abs(x) for x in ks)
        Zp = Zm = 0.0
        for st, prods in cfg:
            val = math.exp(sum(x * y for x, y in zip(ks, prods)) - mx)
            if st == 1: Zp += val
            else: Zm += val
        t = (Zp - Zm) / (Zp + Zm)
        tot += w * t; tot2 += w * t * t
    return tot, tot2


e1 = path_law(1)
tU, tU2 = e_tanh(['w1', 'w2'], 'O', [('A', 'O', e1), ('A', 'w1', e1), ('w2', 'O', e1), ('w1', 'w2', path_law(b0)), ('w1', 'w2', path_law(b0))])
print(f"2. exact t_U (bridge code, float) = {tU:.15f}  (E tanh^2 = {tU2:.15f})")
r = vtf ** 2
full = [('A', 'O', e1), ('A', 'w1', e1), ('w2', 'O', e1), ('w1', 'w2', path_law(b0)), ('w1', 'w2', path_law(b0))]


def A_of(j):
    if j == 0:
        return e_tanh(['O', 'w2'], 'w1', full)[0]
    if j == b0:
        return e_tanh(['O', 'w1'], 'w2', full)[0]
    return e_tanh(['O', 'w1', 'w2'], 'u', [('A', 'O', e1), ('A', 'w1', e1), ('w2', 'O', e1), ('w1', 'u', path_law(j)), ('u', 'w2', path_law(b0 - j)), ('w1', 'w2', path_law(b0))])[0]


def C(l1, l2):
    t1, t2 = r ** l1, r ** l2
    return (t1 + t2 - 2 * t1 * t2) / (1 - t1 * t2)


Aj = [A_of(j) for j in range(b0 + 1)]
viol = [(j, Aj[j], C(j + 1, b0 - j + 2)) for j in range(b0 + 1) if Aj[j] < C(j + 1, b0 - j + 2) - 1e-12]
print(f"   A(j)=E tanh K_U(alpha,u_j) vs chain block C(j+1,b0-j+2): violations={len(viol)} {viol[:3]}")
for j in (0, 1, b0 // 4, b0 // 2, 3 * b0 // 4, b0 - 1, b0):
    print(f"     j={j}: exact A={Aj[j]:.10f} chain C={C(j + 1, b0 - j + 2):.10f}")
I = 2 * k * b0 + k + 1
Sx = Sc = 0.0
for i in range(0, k + 1):
    Sx += r * tU ** min(i, k - i); Sc += r * tU ** min(i, k - i)
for i in range(1, k + 1):
    fw = r * tU ** (i - 1); bw = r * tU ** (k - i)
    for j in range(0, b0 + 1):
        mult = 1 if j in (0, b0) else 2
        Sx += mult * max(fw * Aj[j], bw * Aj[b0 - j])
        Sc += mult * max(fw * C(j + 1, b0 - j + 2), bw * C(b0 - j + 1, j + 2))
kbx, kbc = Sx / I, Sc / I
khx, khc = (1 + 2 * I * kbx) / (1 + 2 * I), (1 + 2 * I * kbc) / (1 + 2 * I)
print(f"   EXACT kappa-bar = {kbx:.9f} (khat {khx:.9f});  own chain-bound kappa-bar = {kbc:.9f} (khat {khc:.9f})")
kh = mpf(khc)

# ---------- 3. v_H(gamma) via signed law DP; u_H, pbar floats ----------
mp.dps = 50


def signed_unit(beta):
    th, kap, k2, _ = closed_unit(beta)
    at = {}
    pj = {1: 1 - p, -1: p}
    q = (1 - vt ** b0) / 2
    pc = {1: 1 - q, -1: q}
    for sd, s1, s2, c1, c2 in itertools.product((1, -1), repeat=5):
        w = pj[sd] * pj[s1] * pj[s2] * pc[c1] * pc[c2]
        t = tanh(beta * sd + atanh(s1 * s2 * th * th * tanh((c1 + c2) * kap)))
        key = mp.nstr(t, 40)
        at[key] = (t, at.get(key, (t, 0))[1] + w)
    return list(at.values())


def chain_law(beta):
    th = tanh(beta)
    at = signed_unit(beta)
    cur = {'1': (mpf(1), mpf(1))}
    for _ in range(k):
        nd = {}
        for t, w in cur.values():
            for ti, wi in at:
                v = t * ti; key = mp.nstr(v, 40)
                nd[key] = (v, nd.get(key, (v, 0))[1] + w * wi)
        cur = nd
    out = []
    for t, w in cur.values():
        for sa in (1, -1):
            for sb in (1, -1):
                out.append((sa * sb * th * th * t, w * (1 - p if sa == 1 else p) * (1 - p if sb == 1 else p)))
    return out


law = chain_law(g)
EeK = sum(w * sqrt((1 - t) / (1 + t)) for t, w in law)
Esech = sum(w * sqrt(1 - t * t) for t, w in law)
Et = sum(w * t for t, w in law); Et2 = sum(w * t * t for t, w in law)
print(f"3. v_H(gamma): E e^-K = {mp.nstr(EeK, 16)}  E sech K = {mp.nstr(Esech, 16)}  mass={mp.nstr(sum(w for _, w in law), 16)}; t_H: E tanh={mp.nstr(Et, 12)} E tanh^2={mp.nstr(Et2, 12)}")


def h(v):
    z = 9 * v * v
    return 4 * 9 * v ** 4 / (1 - z) ** 2 + 4 * 9 * v ** 3 / (1 - z)


print(f"   h(v_H) = {mp.nstr(h(EeK), 12)}; D_bound(chain) = khat^2 (1-h) = {mp.nstr(kh ** 2 * (1 - h(EeK)), 12)}")


def uH_pbar(beta):
    th, kap, k2, m = closed_unit(beta)
    u = th * th * (P1 * m[0] + P2 * m[1] + P3 * m[2]) ** k
    pb = mpf(0); u2 = mpf(0)
    for n1 in range(k + 1):
        for n2 in range(k + 1 - n1):
            n3 = k - n1 - n2
            w = mpf(math.factorial(k)) / (math.factorial(n1) * math.factorial(n2) * math.factorial(n3)) * P1 ** n1 * P2 ** n2 * P3 ** n3
            x = th * th * m[0] ** n1 * m[1] ** n2 * m[2] ** n3
            pb += w * 2 * x / (1 + x); u2 += w * x
    return u, pb, u2


for c in ('1', '2', '2.02', '2.1', '2.117', '2.15', '2.2', '2.22', '2.25', '2.5', '2.6', '3', '5', '10', '40'):
    mp.dps = int(60 + float(c) * float(g))
    u, pb, u2 = uH_pbar(mpf(c) * g)
    print(f"   beta={c}g: u_H={mp.nstr(u, 10)} (mixture {mp.nstr(u2, 10)}) pbar_H={mp.nstr(pb, 10)}")
mp.dps = 50
pinf = mpf(0)
for j in range(k + 1):
    w = mpf(math.comb(k, j)) * P3 ** j * (1 - P3) ** (k - j)
    x = mpf(1) / 3 ** j
    pinf += w * 2 * x / (1 + x)
print(f"   limits beta->inf: u_H -> {mp.nstr((1 - 2 * P3 / 3) ** k, 12)}, pbar_H -> {mp.nstr(pinf, 12)}")
