"""Primary rigorous certificate (Arb ball arithmetic via python-flint, exact fmpq where possible)
for the max-degree-4 gadget

    H4 = e - S_k(U) - e,   U = e || (e - (P_b0 || P_b0) - e),

iid +-J couplings, P(J=-1) = p rational, gamma(p) = (1/2) log((1-p)/p).

Certified statements (the closed form is Proposition 9.3, prop:lat-law, of the paper):
  (VP)  v_P in [vP_lo, vP_hi], rational bracket with h(vP_lo) < 1 < h(vP_hi), exact rationals.
  (W)   v_H = E e^{-K_H(gamma)} < vP_lo and h(V) < 1 for a rational V >= v_H.
  (C)   half-line claims  F(beta) < c  for all beta >= beta_0  (F = u_H or pbar_H),
        and lower-bound claims F(beta) > c for all beta >= beta_0,
        by adaptive bisection over rational beta-intervals [beta_0, BETA_BIG]
        plus one tail enclosure covering [BETA_BIG, infinity) (the variables s = e^{-2 beta},
        Z = tanh(b0 atanh s)^2 / s stay bounded, and the beta = infinity limit is included).
  (PT)  point claims F(beta_*) > c (or < c) that bracket the minimal half-line start.

Usage: python cert_arb.py out/cert_arb.json > out/cert_arb.txt
"""
import json
import sys
import time
from math import comb

from flint import arb, fmpq, ctx

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

PREC = 160
ctx.prec = PREC
BETA_BIG = fmpq(64)
MIN_WIDTH = fmpq(1, 10 ** 12)
INIT_STEP = fmpq(1, 64)


# ---------------------------------------------------------------- exact helpers
def dyadic(x):
    """Exact fmpq value of an exact arb (midpoint or radius)."""
    m, e = x.man_exp()
    return fmpq(int(m)) * (fmpq(2) ** int(e)) if e >= 0 else fmpq(int(m), 2 ** int(-e))


def bounds(x):
    """Exact rational lower/upper endpoints of the ball x."""
    mid, rad = dyadic(x.mid()), dyadic(x.rad())
    return mid - rad, mid + rad


def ball(lo, hi):
    """Arb ball certainly containing the rational interval [lo, hi]."""
    lo, hi = fmpq(lo), fmpq(hi)
    b = arb((lo + hi) / 2, (hi - lo) / 2)
    extra = fmpq(0)
    while not (b.contains(arb(lo)) and b.contains(arb(hi))):
        extra = extra * 2 + fmpq(1, 2 ** PREC)
        b = arb((lo + hi) / 2, (hi - lo) / 2 + extra)
    return b


def fl(x):
    """Float value for printing only (never used in a certified comparison)."""
    return float(arb(x).mid()) if isinstance(x, fmpq) else float(x.mid())


def fq(x):
    return f"{x.p}/{x.q}" if x.q != 1 else f"{x.p}"


# ---------------------------------------------------------------- v_P
def h_exact(v):
    d = 1 - 9 * v * v
    return 36 * v ** 4 / d ** 2 + 36 * v ** 3 / d


def vp_bracket(digits=15):
    """h is increasing on (0,1/3) (sum of products of positive increasing factors), h(0)=0,
    h -> inf at 1/3, so the root is unique; bisect on a decimal grid in exact rationals."""
    scale = 10 ** digits
    lo, hi = 0, scale // 3
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if h_exact(fmpq(mid, scale)) < 1:
            lo = mid
        else:
            hi = mid
    a, b = fmpq(lo, scale), fmpq(hi, scale)
    assert h_exact(a) < 1 < h_exact(b)
    return a, b


# ---------------------------------------------------------------- law of the unit (exact)
def unit_law(p, b0):
    tg = 1 - 2 * p
    q = (1 - tg ** b0) / 2                       # P(path of b0 edges has negative sign product)
    cp, cm = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)   # P(J_a J_b = +1), P(J_a J_b = -1)
    tpos = cp * (1 - q) ** 2 + cm * q ** 2       # P(tanh of long branch > 0)
    tzero = 2 * q * (1 - q)                      # P(the two paths disagree): long branch decouples
    tneg = cp * q ** 2 + cm * (1 - q) ** 2
    assert tpos + tzero + tneg == 1
    P1 = (1 - p) * tpos + p * tneg               # class 1: J_e agrees with branch sign, |tanh| = w1
    P2 = tzero                                   # class 2: branch decoupled, |tanh| = t
    P3 = (1 - p) * tneg + p * tpos               # class 3: J_e opposes branch sign, |tanh| = w3
    rho = ((1 - p) * tpos - p * tneg) / P1, 1 - 2 * p, ((1 - p) * tneg - p * tpos) / P3
    return dict(q=q, tpos=tpos, tzero=tzero, tneg=tneg, P=(P1, P2, P3), rho=rho)


def classes(k):
    for n1 in range(k + 1):
        for n3 in range(k + 1 - n1):
            yield n1, k - n1 - n3, n3, comb(k, n1) * comb(k - n1, n3)


# ---------------------------------------------------------------- magnitudes on balls
def w_from_sZ(s, Z):
    s2 = s * s
    r = (2 + Z * (1 + s2)) / (1 + s2 + 2 * s2 * Z)   # r = e^{-2 K_long} / s
    w1 = 2 / (1 + r * s2) - 1                        # tanh(beta + K_long)
    w3 = 1 - 2 / (r + 1)                             # tanh(beta - K_long) = (r-1)/(r+1)
    return w1, w3


def mags_interval(blo, bhi, b0):
    B = ball(blo, bhi)
    s = (-2 * B).exp()
    t = B.tanh()
    Z = (b0 * s.atanh()).tanh() ** 2 / s
    w1, w3 = w_from_sZ(s, Z)
    return t, w1, w3


def mags_tail(beta_big, b0):
    """Enclosure valid for every beta >= beta_big (and the limit beta = inf)."""
    s_hi = bounds((-2 * arb(beta_big)).exp())[1]
    assert 0 < s_hi < fmpq(1, 10)
    s = ball(0, s_hi)
    t = ball((1 - s_hi) / (1 + s_hi), 1)
    Z = ball(0, fmpq(b0) ** 2 * s_hi / (1 - s_hi ** 2) ** 2)   # 0 <= Z <= b0^2 s/(1-s^2)^2
    w1, w3 = w_from_sZ(s, Z)
    return t, w1, w3, s_hi


class Design:
    def __init__(self, name, b0, k, p):
        self.name, self.b0, self.k, self.p = name, b0, k, fmpq(p)
        self.law = unit_law(self.p, b0)
        self.P = [arb(x) for x in self.law["P"]]
        self.cls = list(classes(k))
        self.gamma = arb((1 - self.p) / self.p).log() / 2

    def functionals(self, t, w1, w3):
        P1, P2, P3 = self.P
        uH = t * t * (P1 * w1 + P2 * t + P3 * w3) ** self.k
        pw1 = [w1 ** i for i in range(self.k + 1)]
        pw3 = [w3 ** i for i in range(self.k + 1)]
        pt = [t ** i for i in range(self.k + 3)]
        pP1 = [P1 ** i for i in range(self.k + 1)]
        pP2 = [P2 ** i for i in range(self.k + 1)]
        pP3 = [P3 ** i for i in range(self.k + 1)]
        pbar = arb(0)
        for n1, n2, n3, M in self.cls:
            x = pt[2 + n2] * pw1[n1] * pw3[n3]
            pbar += M * pP1[n1] * pP2[n2] * pP3[n3] * (2 - 2 / (1 + x))
        return dict(u=uH, pbar=pbar)

    # ------------------------------------------------ warm side, exact rationals then Arb sqrt
    def warm(self, vp_lo):
        p, b0, k = self.p, self.b0, self.k
        t = 1 - 2 * p                              # tanh(gamma) = 1 - 2p exactly
        tb = t ** b0
        T2 = 2 * tb / (1 + tb * tb)                # tanh(2 kappa), kappa = atanh(t^b0)
        a = t * t * T2                             # tanh of the long branch magnitude
        w1 = (t + a) / (1 + t * a)
        w3 = (t - a) / (1 - t * a)
        r1, r2, r3 = self.law["rho"]
        # exact Nishimori checks: E[sign | class] = |tanh| of the class at gamma
        nish = (r1 == w1, r2 == t, r3 == w3)
        P1, P2, P3 = self.law["P"]
        v = arb(0)
        vsech = arb(0)
        mass = fmpq(0)
        for n1, n2, n3, M in self.cls:
            wgt = M * P1 ** n1 * P2 ** n2 * P3 ** n3
            mass += wgt
            x = t ** (2 + n2) * w1 ** n1 * w3 ** n3     # exact rational |tanh K_H|
            A = t * t * r1 ** n1 * r2 ** n2 * r3 ** n3  # E[sign K_H | class counts]; (1-2p)^2 from terminals
            g = (arb(1 - x) / arb(1 + x)).sqrt()        # e^{-|K_H|}
            v += arb(wgt) * (arb((1 + A) / 2) * g + arb((1 - A) / 2) / g)
            vsech += arb(wgt) * arb(1 - x * x).sqrt()
        assert mass == 1
        vlo, vhi = bounds(v)
        V = fmpq(int(vhi * 10 ** 12) + 1, 10 ** 12)     # rational upper bound, 12 decimals
        assert V >= vhi
        return dict(v=v, vsech=vsech, v_bounds=(vlo, vhi), V=V, hV=h_exact(V),
                    ok=bool(vhi < vp_lo and h_exact(V) < 1), nishimori_exact=nish,
                    overlap_sech=bool(v.overlaps(vsech)), w1=w1, w3=w3, a=a)

    # ------------------------------------------------ cold side
    def eval_interval(self, lo, hi):
        t, w1, w3 = mags_interval(lo, hi, self.b0)
        return self.functionals(t, w1, w3)

    def half_line(self, key, thr, beta0, upper=True):
        """Certify F(beta) < thr (upper=True) or F(beta) > thr (upper=False) for all beta >= beta0."""
        thr = fmpq(thr)
        ok_fn = (lambda F: F < arb(thr)) if upper else (lambda F: F > arb(thr))
        pieces, fails = [], []
        stack = []
        x = fmpq(beta0)
        while x < BETA_BIG:
            y = min(x + INIT_STEP, BETA_BIG)
            stack.append((x, y))
            x = y
        stack.reverse()
        nev = 0
        while stack:
            lo, hi = stack.pop()
            F = self.eval_interval(lo, hi)[key]
            nev += 1
            if ok_fn(F):
                pieces.append((lo, hi, bounds(F)))
            elif hi - lo > MIN_WIDTH:
                mid = (lo + hi) / 2
                stack.append((mid, hi))
                stack.append((lo, mid))
            else:
                fails.append((lo, hi, bounds(F)))
                break
        t, w1, w3, s_hi = mags_tail(BETA_BIG, self.b0)
        Ft = self.functionals(t, w1, w3)[key]
        tail_ok = ok_fn(Ft)
        pieces.sort()
        # contiguity check of the partition
        cover = pieces[0][0] == fmpq(beta0) and pieces[-1][1] == BETA_BIG and all(
            pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))
        ok = (not fails) and tail_ok and cover
        ext = max(pc[2][1] for pc in pieces) if upper else min(pc[2][0] for pc in pieces)
        ext = max(ext, bounds(Ft)[1]) if upper else min(ext, bounds(Ft)[0])
        return dict(ok=ok, n_pieces=len(pieces), n_eval=nev, fails=fails, tail=bounds(Ft), s_hi=s_hi,
                    extreme=ext, pieces=pieces, min_width=min(pc[1] - pc[0] for pc in pieces))

    def point(self, key, beta, thr, above=True):
        F = self.eval_interval(fmpq(beta), fmpq(beta))[key]
        ok = (F > arb(fmpq(thr))) if above else (F < arb(fmpq(thr)))
        return dict(ok=bool(ok), F=bounds(F))


def suffix_sup(pieces, tail, starts):
    out = []
    for b in starts:
        vals = [pc[2][1] for pc in pieces if pc[1] > b] + [tail[1]]
        out.append((b, max(vals)))
    return out


def run(name, b0, k, p, halflines, points, supstarts, vp):
    D = Design(name, b0, k, p)
    g_lo, g_hi = bounds(D.gamma)
    L = D.law
    print(f"=== design {name}: b0={b0} k={k} p={fq(D.p)}  edges per bond = {k*(2*b0+3)+2}  max degree 4")
    print(f"  gamma(p) in [{fl(g_lo):.15f}, {fl(g_hi):.15f}]")
    print(f"  q = P(path negative) = {fl(L['q']):.12f};  P(branch +,0,-) = {fl(L['tpos']):.12f}, {fl(L['tzero']):.12f}, {fl(L['tneg']):.12f}")
    print(f"  unit classes P1,P2,P3 = {', '.join(f'{fl(x):.12f}' for x in L['P'])}  (exact rationals; sum = 1 checked)")
    res = dict(name=name, b0=b0, k=k, p=fq(D.p), edges=k * (2 * b0 + 3) + 2, gamma=[fl(g_lo), fl(g_hi)])
    t0 = time.time()
    W = D.warm(vp[0])
    vlo, vhi = W["v_bounds"]
    print(f"  (W) v_H = E e^(-K_H(gamma)) in [{fl(vlo):.15f}, {fl(vhi):.15f}]  (E sech K_H = {W['vsech']}; overlap {W['overlap_sech']})")
    print(f"      exact Nishimori identities rho_c = |tanh| of class c at gamma: {W['nishimori_exact']}")
    print(f"      V = {fq(W['V'])} >= v_H;  h(V) = {fl(W['hV']):.12f} < 1: {W['hV'] < 1};  1 - h(V) >= {fl(1 - W['hV']):.12f}")
    print(f"      v_H < vP_lo = {fq(vp[0])}: {vhi < vp[0]};  margin vP_lo - v_H >= {fl(vp[0] - vhi):.12f} (relative {fl((vp[0]-vhi)/vp[0]):.4%})")
    print(f"      WARM OK: {W['ok']}   [{time.time()-t0:.1f}s]")
    res["warm"] = dict(v=[fl(vlo), fl(vhi)], V=fq(W["V"]), hV=fl(W["hV"]), ok=W["ok"],
                       nishimori_exact=list(W["nishimori_exact"]), margin=fl(vp[0] - vhi))
    res["halflines"] = []
    for key, thr, b0_, upper in halflines:
        t0 = time.time()
        if b0_ == "gamma":
            b0_ = fmpq(int(fl(g_lo) * 10 ** 9), 10 ** 9)
            assert b0_ <= g_lo
        HL = D.half_line(key, thr, b0_, upper)
        rel = "<" if upper else ">"
        extname = "sup" if upper else "inf"
        print(f"  (C) {key}_H(beta) {rel} {fq(fmpq(thr))} for all beta >= {fq(fmpq(b0_))} (= {fl(fmpq(b0_)) / fl(D.gamma):.7f} gamma): {HL['ok']}")
        print(f"      pieces={HL['n_pieces']} evals={HL['n_eval']} min width={fl(HL['min_width']):.3g}; tail beta >= {fq(BETA_BIG)} (s <= {fl(HL['s_hi']):.3g}): {key} in [{fl(HL['tail'][0]):.12f}, {fl(HL['tail'][1]):.12f}]")
        print(f"      certified {extname} over the half-line: {fl(HL['extreme']):.12f}  (margin to threshold {abs(fl(HL['extreme']) - fl(fmpq(thr))):.3e})   [{time.time()-t0:.1f}s]")
        if HL["fails"]:
            print(f"      FAIL at {HL['fails']}")
        entry = dict(key=key, thr=fq(fmpq(thr)), beta0=fq(fmpq(b0_)), beta0_over_gamma=fl(fmpq(b0_)) / fl(D.gamma), upper=upper,
                     ok=HL["ok"], extreme=fl(HL["extreme"]), tail=[fl(HL["tail"][0]), fl(HL["tail"][1])], pieces=HL["n_pieces"])
        if upper:
            sup = suffix_sup(HL["pieces"], HL["tail"], [fmpq(b0_)] + [fmpq(x) for x in supstarts if fmpq(x) > fmpq(b0_)])
            for b, sv in sup:
                print(f"        sup_{{beta >= {fq(b)} ({fl(b)/fl(D.gamma):.4f} gamma)}} {key}_H <= {fl(sv):.9f}")
            entry["suffix_sup"] = [(fq(b), fl(sv)) for b, sv in sup]
        res["halflines"].append(entry)
    res["points"] = []
    for key, beta, thr, above in points:
        P = D.point(key, beta, thr, above)
        rel = ">" if above else "<"
        print(f"  (PT) {key}_H({fq(fmpq(beta))}) {rel} {fq(fmpq(thr))}: {P['ok']}   value in [{fl(P['F'][0]):.12f}, {fl(P['F'][1]):.12f}]")
        res["points"].append(dict(key=key, beta=fq(fmpq(beta)), thr=fq(fmpq(thr)), above=above, ok=P["ok"], F=[fl(P["F"][0]), fl(P["F"][1])]))
    sys.stdout.flush()
    return res


if __name__ == "__main__":
    t_start = time.time()
    vp = vp_bracket(15)
    print(f"python-flint Arb, prec={PREC} bits; BETA_BIG={BETA_BIG}; MIN_WIDTH={fq(MIN_WIDTH)}")
    print(f"(VP) v_P in [{fq(vp[0])}, {fq(vp[1])}]  h(lo)={fl(h_exact(vp[0])):.17f} < 1 < h(hi)={fl(h_exact(vp[1])):.17f}")
    print(f"     exact checks: h(1/5) = {fq(h_exact(fmpq(1,5)))}, h(3/25) = {fq(h_exact(fmpq(3,25)))}")
    sup_starts = ["15/2", "31/4", "8", "9", "21/2", "14", "20"]
    results = dict(vp=[fq(vp[0]), fq(vp[1])])
    results["P"] = run("P (primary)", 1000, 10, fmpq(9, 10000),
                       [("u", fmpq(1, 3), fmpq(741995, 100000), True),
                        ("pbar", fmpq(1, 2), fmpq(698554, 100000), True),
                        ("pbar", fmpq(1, 3), "gamma", False)],
                       [("u", fmpq(741994, 100000), fmpq(1, 3), True),
                        ("pbar", fmpq(698553, 100000), fmpq(1, 2), True)],
                       sup_starts, vp)
    results["E"] = run("E (elementary)", 1500, 10, fmpq(9, 10000),
                       [("pbar", fmpq(1, 3), fmpq(790643, 100000), True),
                        ("u", fmpq(1, 3), fmpq(722772, 100000), True),
                        ("pbar", fmpq(1, 2), fmpq(706442, 100000), True)],
                       [("pbar", fmpq(790642, 100000), fmpq(1, 3), True),
                        ("u", fmpq(722771, 100000), fmpq(1, 3), True),
                        ("pbar", fmpq(706441, 100000), fmpq(1, 2), True)],
                       sup_starts, vp)
    results["B"] = run("B (b0=1500,k=8)", 1500, 8, fmpq(87373, 10 ** 8),
                       [("u", fmpq(1, 3), fmpq(783140, 100000), True),
                        ("pbar", fmpq(1, 2), fmpq(740433, 100000), True),
                        ("pbar", fmpq(1, 3), "gamma", False)],
                       [("u", fmpq(783139, 100000), fmpq(1, 3), True),
                        ("pbar", fmpq(740432, 100000), fmpq(1, 2), True)],
                       sup_starts, vp)
    allok = all(d["warm"]["ok"] and all(h["ok"] for h in d["halflines"]) and all(pt["ok"] for pt in d["points"])
                for kname, d in results.items() if kname != "vp")
    print(f"ALL CLAIMS CERTIFIED: {allok}   total {time.time()-t_start:.1f}s")
    results["all_ok"] = allok
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "cert_arb.json", "w"), indent=1)
