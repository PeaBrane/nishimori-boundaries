"""Re-certification 1: separate verification of every claim in ../out/claims.json (Appendix C.7).

Rigorous engine: fxi.py (own fixed-point intervals on Python ints, own exp/log/sqrt).
Cross-check engine: mpmath.iv at 256 bits.  Float engine (mpmath mp, 40 digits) only for reporting slacks.
The law of K_H is model.Law (own derivation by enumeration + DP), NOT the class/sign-mean formulas of
claims.json; those formulas are compared with the derivation exactly.

Usage: check_claims.py CLAIMS_JSON OUT_JSON [--cells N] [--workers W] [--no-iv]
"""
import argparse
import hashlib
import json
import multiprocessing as mp_
import sys
import time
from fractions import Fraction as Fr

import fxi
import model as M

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

FX = M.FxiEngine()
I = FX.I

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def fr(s):
    return Fr(s)


# ------------------------------------------------------------------ helpers (worker side)
_G = {}


def _init(p0):
    law = M.Law(p0)
    _G["law"] = law
    _G["lawFX"] = M.law_in_engine(FX, law)
    _G["wFX"] = M.weights_in_engine(FX, law)
    _G["iv"] = None


def _iv():
    if _G.get("iv") is None:
        E = M.IvEngine(256)
        _G["iv"] = (E, M.law_in_engine(E, _G["law"]))
    return _G["iv"]


def F_point_fx(beta, s):
    mg = M.mags_beta(FX, FX.const(beta))
    X = M.xs(FX, mg, _G["law"])
    return M.F_value(FX, _G["lawFX"], X, FX.const(s))


def F_point_iv(beta, s):
    E, lawE = _iv()
    mg = M.mags_beta(E, E.const(beta))
    X = M.xs(E, mg, _G["law"])
    return M.F_value(E, lawE, X, E.const(s))


def brackets_fx(bl, br):
    """Monotone brackets (m1, m2, m3) over beta in [bl, br], as fxi intervals."""
    thL, tauL, m2L, _ = M.mags_beta(FX, FX.const(bl))
    thR, tauR, m2R, _ = M.mags_beta(FX, FX.const(br))
    # theta, m2 >= 0 for beta >= 0, so clamping lower endpoints at 0 keeps the enclosure valid
    m1b = I(max(0, thL.lo), thR.hi)
    m2b = I(max(0, m2L.lo), m2R.hi)
    m3hi = ((thR - tauL) / (1 - thR * tauL)).hi
    m3lo = max(0, ((thL - tauR) / (1 - thL * tauR)).lo)
    return m1b, m2b, I(m3lo, m3hi)


def xs_from_brackets(m1b, m2b, m3b, law):
    k = law.k
    pw = []
    for m in (m1b, m2b, m3b):
        row = [I.frac(1)]
        for _ in range(k):
            row.append(row[-1] * m)
        pw.append(row)
    th2 = m1b * m1b
    return {n: th2 * pw[0][n[0]] * pw[1][n[1]] * pw[2][n[2]] for n in law.atoms}


def F_cell_sup_fx(bl, br, s):
    """Rigorous upper bound of sup_{beta in [bl,br]} F(beta,s), without Lemma L:
    monotone brackets give x_n in [xlo, xhi]; each atom P+ r^s + P- r^-s is convex in log r,
    so its sup over r in [r(xhi), r(xlo)] is at an endpoint."""
    law = _G["law"]
    X = xs_from_brackets(*brackets_fx(bl, br), law)
    S = FX.const(s)
    tot = Fr(0)
    for n, (Pp, Pm) in _G["lawFX"].items():
        rb = (1 - X[n]) / (1 + X[n])
        Lb = FX.log(rb)
        best = None
        for L in (I(Lb.lo, Lb.lo), I(Lb.hi, Lb.hi)):
            g = Pp * FX.exp(S * L) + Pm * FX.exp(-(S * L))
            best = g.hi if best is None else max(best, g.hi)
        tot += Fr(best, 1 << fxi.W)
    return tot


def F_cell_sup_fx_split(bl, br, s, target, depth=0, maxdepth=6):
    b = F_cell_sup_fx(bl, br, s)
    if b <= target or depth >= maxdepth:
        return b, 1
    mid = (bl + br) / 2
    b1, n1 = F_cell_sup_fx_split(bl, mid, s, target, depth + 1, maxdepth)
    b2, n2 = F_cell_sup_fx_split(mid, br, s, target, depth + 1, maxdepth)
    return max(b1, b2), n1 + n2


def F_float(beta, s):
    if _G.get("mp") is None:
        E0 = M.MpEngine(40)
        _G["mp"] = (E0, M.law_in_engine(E0, _G["law"]))
    E, lawE = _G["mp"]
    mg = M.mags_beta(E, E.const(beta))
    X = M.xs(E, mg, _G["law"])
    return M.F_value(E, lawE, X, E.const(s))


def check_cell(args):
    idx, cell, thr, use_iv = args
    bl, br, s = fr(cell["bl"]), fr(cell["br"]), fr(cell["s"])
    Fup, Wup, y = fr(cell["F_up"]), fr(cell["W_up"]), fr(cell["y"])
    ball = fr(cell["ball_up"])
    res = {"i": idx}
    res["s_ok"] = 0 < s <= 1
    res["y_ok"] = y == 6 * s * (br - bl) and 0 <= y < 2
    res["W_ok"] = Fup * (2 + y) / (2 - y) <= Wup <= thr
    res["order_ok"] = bl < br
    Ffx = F_point_fx(bl, s)
    res["F_fx_hi_le_Fup"] = Ffx.upper() <= Fup
    res["F_fx_width"] = float(Ffx.width())
    res["F_fx_slack_rel"] = float((Fup - Ffx.upper()) / Fup)
    if use_iv:
        E, _ = _iv()
        Fiv = F_point_iv(bl, s)
        res["F_iv_hi_le_Fup"] = E.hi(Fiv) <= Fup
        # the two rigorous enclosures must intersect
        res["fx_iv_overlap"] = not (E.hi(Fiv) < Ffx.lower() or Ffx.upper() < E.lo(Fiv))
    # Lemma-L-free bracket bound over the whole cell
    sup, npieces = F_cell_sup_fx_split(bl, br, s, thr)
    res["bracket_sup"] = str(sup)
    res["bracket_pieces"] = npieces
    res["bracket_ok"] = sup <= thr
    # consistency: claimed ball_up and W_up must dominate F at sample points (floats, 40 digits)
    mx = 0.0
    for b in (bl, (bl + br) / 2, br):
        mx = max(mx, float(F_float(b, s)))
    res["F_samples_max"] = mx
    res["ball_ge_samples"] = float(ball) >= mx
    res["W_ge_samples"] = float(Wup) >= mx
    res["ball_le_thr"] = ball <= thr
    return res


# ------------------------------------------------------------------ sharpness of w (lower bound of min_s F)
def w_lower_bound(beta, target, n0=64, maxdepth=14):
    """Rigorous lower bound of min_{s in [0,1]} F(beta, s) by convexity in s (tangent lines)."""
    law = _G["law"]
    mg = M.mags_beta(FX, FX.const(beta))
    X = M.xs(FX, mg, law)
    Ls = {n: FX.log((1 - x) / (1 + x)) for n, x in X.items()}
    cache = {}

    def FdF(s):
        if s in cache:
            return cache[s]
        S = FX.const(s)
        f = FX.const(0)
        d = FX.const(0)
        for n, (Pp, Pm) in _G["lawFX"].items():
            L = Ls[n]
            a = Pp * FX.exp(S * L)
            b = Pm * FX.exp(-(S * L))
            f = f + a + b
            d = d + L * (a - b)
        cache[s] = (f.lower(), d.lower(), d.upper())
        return cache[s]

    def lb(sa, sb):
        Fa, dla, _ = FdF(sa)
        Fb, _, dhb = FdF(sb)
        # l1(s) = Fa + dla (s-sa) ; l2(s) = Fb - dhb (sb - s); both are lower bounds on [sa,sb]
        cands = [sa, sb]
        den = dla - dhb
        if den != 0:
            sx = (Fb - Fa + dla * sa - dhb * sb) / den
            if sa < sx < sb:
                cands.append(sx)
        return min(max(Fa + dla * (c - sa), Fb - dhb * (sb - c)) for c in cands)

    worst = None
    stack = [(Fr(i, n0), Fr(i + 1, n0), 0) for i in range(n0)]
    evals = 0
    while stack:
        sa, sb, dep = stack.pop()
        v = lb(sa, sb)
        evals += 1
        if v < target and dep < maxdepth:
            m = (sa + sb) / 2
            stack += [(sa, m, dep + 1), (m, sb, dep + 1)]
            continue
        worst = v if worst is None else min(worst, v)
    return worst, len(cache)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claims")
    ap.add_argument("out")
    ap.add_argument("--cells", type=int, default=0, help="check only the first N cells of each window (0 = all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-iv", action="store_true")
    ap.add_argument("--skip-sharp", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    raw = open(args.claims, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    C = json.loads(raw)
    R = {"claims_sha256": sha, "mismatches": []}

    def mismatch(where, msg):
        R["mismatches"].append({"where": where, "msg": msg})
        say(f"MISMATCH [{where}] {msg}")

    say(f"claims.json sha256 {sha}")
    import mpmath

    # ---------------- design
    d = C["design"]
    b0, k, p0 = d["b0"], d["k"], fr(d["p0"])
    assert (b0, k) == (M.B0, M.K) and p0 == M.P0
    ok = d["edges_per_bond"] == k * (2 * b0 + 3) + 2
    say(f"design: H4({b0},{k}), p0={p0}, edges_per_bond {d['edges_per_bond']} == k(2b0+3)+2: {ok}")
    if not ok:
        mismatch("design.edges_per_bond", d["edges_per_bond"])
    g0 = M.gamma_float(p0, 40)
    say(f"gamma(p0) = {mpmath.nstr(g0, 20)}; claims {d['gamma_p0']}")
    if abs(float(g0) - d["gamma_p0"][0]) > 1e-12:
        mismatch("design.gamma_p0", d["gamma_p0"])

    # ---------------- law: derivation vs claims formulas (exact)
    _init(p0)
    law = _G["law"]
    fc = law.formula_classes()
    lab = {1: "disagree (m1=theta)", 2: "non-cancel (m2)", 3: "cancel (m3)"}
    for c in (1, 2, 3):
        P, rho = law.unit_class(c)
        eqP, eqr = P == fc["P"][c - 1], rho == fc["rho"][c - 1]
        say(f"law class {c} {lab[c]}: P_c={float(P):.12f} (formula equal exactly: {eqP}); rho_c={float(rho):.12f} (formula equal exactly: {eqr})")
        if not (eqP and eqr):
            mismatch(f"law_of_K_H class {c}", "enumeration != formula")
    cnt, bad = law.check_gadget_formula()
    say(f"gadget atoms: {cnt} classes x 2 signs; product formula (M(n) P^n, A_n) equals DP exactly on all: {bad == 0}")
    if bad:
        mismatch("law_of_K_H.gadget_law", f"{bad} atoms differ")
    R["law"] = {"P": [float(law.unit_class(c)[0]) for c in (1, 2, 3)], "rho": [float(law.unit_class(c)[1]) for c in (1, 2, 3)],
                "q": float(fc["q"]), "a": float(fc["a"]), "n_classes": cnt}

    # ---------------- thresholds (exact)
    ws = fr(C["thresholds"]["W_STAR"]["value"])
    wb = fr(C["thresholds"]["W_BB"]["value"])
    t1 = 9 * ws * ws < 1 and 18 * ws**4 < (1 - 9 * ws * ws) ** 2
    t1b = 2 * M.P_(ws) < 1
    t2 = 15 * wb * wb < 1
    t2b = 4 * M.P_(wb) < 1
    say(f"thresholds: w*={ws}: 9w^2<1 and 18w^4<(1-9w^2)^2: {t1}; 2P(w*)<1: {t1b}; 1-2P(w*) = {float(1 - 2 * M.P_(ws)):.10e} (claims {C['thresholds']['W_STAR']['one_minus_2P']})")
    say(f"            w**={wb}: 15w^2<1: {t2}; 4P(w**)<1: {t2b}; 1-4P(w**) = {float(1 - 4 * M.P_(wb)):.10e} (claims {C['thresholds']['W_BB']['one_minus_4P']})")
    mpmath.mp.dps = 30
    root2P = 1 / mpmath.sqrt(9 + 3 * mpmath.sqrt(2))
    say(f"            root of 2P(w)=1: {mpmath.nstr(root2P, 15)}; 1/sqrt(15) = {mpmath.nstr(1 / mpmath.sqrt(15), 15)}")
    if not (t1 and t1b and t2 and t2b):
        mismatch("thresholds", "failed")
    if abs(float(1 - 2 * M.P_(ws)) - C["thresholds"]["W_STAR"]["one_minus_2P"]) > 1e-15 or abs(float(1 - 4 * M.P_(wb)) - C["thresholds"]["W_BB"]["one_minus_4P"]) > 1e-15:
        mismatch("thresholds floats", "display value differs")

    # ---------------- W windows
    tasks = []
    for key, thr_key in (("W_plus", "W_STAR"), ("W_two_point", "W_BB")):
        Wd = C[key]
        thr = fr(Wd["threshold"])
        if thr != fr(C["thresholds"][thr_key]["value"]):
            mismatch(key, "threshold differs from thresholds block")
        cells = Wd["cells"]
        n = len(cells)
        if n != Wd["n_cells"]:
            mismatch(key, f"n_cells {Wd['n_cells']} != len(cells) {n}")
        contig = all(fr(cells[i]["br"]) == fr(cells[i + 1]["bl"]) for i in range(n - 1))
        ends = fr(cells[0]["bl"]) == fr(Wd["beta_1"]) and fr(cells[-1]["br"]) == fr(Wd["beta_2"])
        supW = max(fr(c["W_up"]) for c in cells)
        say(f"{key}: {n} cells, contiguous {contig}, endpoints = [beta_1, beta_2] {ends}, max W_up = {supW} == sup_W_up {supW == fr(Wd['sup_W_up'])}, <= threshold {supW <= thr}")
        if not (contig and ends and supW == fr(Wd["sup_W_up"]) and supW <= thr):
            mismatch(key, "structure")
        b1, b2 = fr(Wd["beta_1"]), fr(Wd["beta_2"])
        say(f"  beta_1/gamma = {(float(b1) / float(g0)):.13f} (claims {Wd['beta_1_over_gamma']}), beta_2/gamma = {(float(b2) / float(g0)):.13f} (claims {Wd['beta_2_over_gamma']})")
        if abs((float(b1) / float(g0)) - Wd["beta_1_over_gamma"]) > 1e-12 or abs((float(b2) / float(g0)) - Wd["beta_2_over_gamma"]) > 1e-12:
            mismatch(key, "beta/gamma display")
        widths = [fr(c["br"]) - fr(c["bl"]) for c in cells]
        svals = [fr(c["s"]) for c in cells]
        say(f"  cell widths in [{float(min(widths)):.3e}, {float(max(widths)):.3e}]; s in [{float(min(svals))}, {float(max(svals))}]")
        # bands
        for bd in Wd["bands"]:
            lo, hi = fr(bd["lo"]), fr(bd["hi"])
            meet = [c for c in cells if fr(c["bl"]) <= hi and fr(c["br"]) >= lo]
            sb = max(fr(c["W_up"]) for c in meet)
            val = 1 - (2 if key == "W_plus" else 4) * M.P_(sb)
            inside = b1 <= lo and hi <= b2
            okb = sb == fr(bd["sup_W_up"]) and val >= fr(bd["bound"]) and inside == bd["inside_cover"]
            say(f"  band [{lo}, {hi}] ({(float(lo) / float(g0)):.6f}..{(float(hi) / float(g0)):.6f} gamma): {len(meet)} cells meet it, sup W_up = {float(sb):.15f} (== claim {sb == fr(bd['sup_W_up'])}); 1-{2 if key == 'W_plus' else 4}P(sup) = {float(val):.9f} >= bound {bd['bound']} = {float(fr(bd['bound']))}: {val >= fr(bd['bound'])}; inside cover {inside}")
            if not okb:
                mismatch(f"{key}.band[{bd['lo']},{bd['hi']}]", f"sup {float(sb)} claim {bd['sup_W_up']} val {float(val)}")
        lim = n if args.cells <= 0 else min(n, args.cells)
        for i in range(lim):
            tasks.append((key, (i, cells[i], thr, not args.no_iv)))
    say(f"checking {len(tasks)} cells with {args.workers} worker(s) ...")
    t1_ = time.time()
    if args.workers > 1:
        ctx = mp_.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            outs = pool.map(check_cell, [t for _, t in tasks], chunksize=8)
    else:
        outs = [check_cell(t) for _, t in tasks]
    say(f"cells done in {time.time() - t1_:.1f}s")
    for key in ("W_plus", "W_two_point"):
        rs = [o for (kk, _), o in zip(tasks, outs) if kk == key]
        if not rs:
            continue
        agg = {}
        for f in ("s_ok", "y_ok", "W_ok", "order_ok", "F_fx_hi_le_Fup", "F_iv_hi_le_Fup", "fx_iv_overlap", "bracket_ok", "ball_ge_samples", "W_ge_samples", "ball_le_thr"):
            if f in rs[0]:
                bad = [r["i"] for r in rs if not r[f]]
                agg[f] = len(rs) - len(bad)
                if bad:
                    mismatch(f"{key}.{f}", f"failing cells {bad[:20]} (total {len(bad)})")
        minslack = min(rs, key=lambda r: r["F_fx_slack_rel"])
        maxw = max(r["F_fx_width"] for r in rs)
        maxpieces = max(r["bracket_pieces"] for r in rs)
        maxbr = max(fr(r["bracket_sup"]) for r in rs)
        say(f"{key}: checks passed per field (of {len(rs)}): {agg}")
        say(f"  min relative slack (F_up - F_hi)/F_up = {minslack['F_fx_slack_rel']:.3e} at cell {minslack['i']}; max fxi enclosure width {maxw:.1e}")
        say(f"  Lemma-L-free bracket bound: max over cells {float(maxbr):.15f} (threshold {float(fr(C[key]['threshold']))}); max subdivision pieces {maxpieces}")
        R[key] = {"n_checked": len(rs), "passed": agg, "min_rel_slack": minslack["F_fx_slack_rel"], "min_slack_cell": minslack["i"],
                  "max_bracket_sup": str(maxbr), "max_bracket_pieces": maxpieces,
                  "per_cell": [{"i": r["i"], "slack": r["F_fx_slack_rel"], "bracket_sup": r["bracket_sup"], "pieces": r["bracket_pieces"]} for r in rs]}

    # ---------------- sharpness_W
    if not args.skip_sharp:
        for pt in C["sharpness_W"]["points"]:
            beta = fr(pt["beta"])
            wl = fr(pt["w_lower"])
            lbv, nev = w_lower_bound(beta, wl)
            if pt["kind"] == "plus":
                fails = 2 * M.P_(wl) >= 1 or 9 * wl * wl >= 1
            else:
                fails = 15 * wl * wl >= 1
            okp = lbv >= wl and fails == pt["route_fails"]
            say(f"sharpness_W beta={pt['beta']} ({(float(beta) / float(g0)):.7f} gamma, {pt['kind']}): rigorous min_s F >= {float(lbv):.12f} ({nev} s-evaluations) >= w_lower {float(wl):.12f}: {lbv >= wl}; threshold test fails at w_lower: {fails}")
            if not okp:
                mismatch(f"sharpness_W {pt['beta']}", f"lb {float(lbv)} vs {float(wl)}")

    # ---------------- hot
    H = C["hot"]
    bh, bh3 = fr(H["beta_hot"]), fr(H["beta_hot3"])
    wFX = _G["wFX"]
    E_iv, _ = _iv()
    wIV = M.weights_in_engine(E_iv, law)
    for nm, beta in (("beta_hot", bh), ("beta_hot3", bh3), ("sharp beta", fr(H["sharpness"]["beta"])), ("sharp beta3", fr(H["sharpness"]["beta3"]))):
        mg = M.mags_beta(FX, FX.const(beta))
        X = M.xs(FX, mg, law)
        u = M.u_value(FX, wFX, X)
        pb = M.pbar_value(FX, wFX, X)
        mgi = M.mags_beta(E_iv, E_iv.const(beta))
        Xi = M.xs(E_iv, mgi, law)
        ui = M.u_value(E_iv, wIV, Xi)
        pbi = M.pbar_value(E_iv, wIV, Xi)
        say(f"hot {nm} = {beta} ({(float(beta) / float(g0)):.7f} gamma): u_H in [{float(u.lower()):.15f}, {float(u.upper()):.15f}] (iv hi {float(E_iv.hi(ui)):.15f}); pbar_H in [{float(pb.lower()):.15f}, {float(pb.upper()):.15f}] (iv hi {float(E_iv.hi(pbi)):.15f})")
        if nm == "beta_hot":
            c1 = pb.upper() <= fr(H["pbar_upper_at_beta_hot"]) < Fr(1, 2) and E_iv.hi(pbi) <= fr(H["pbar_upper_at_beta_hot"])
            c2 = u.upper() <= fr(H["u_upper_at_beta_hot"]) < Fr(1, 3) and E_iv.hi(ui) <= fr(H["u_upper_at_beta_hot"])
            say(f"   pbar <= {float(fr(H['pbar_upper_at_beta_hot']))} < 1/2: {c1};  u <= {float(fr(H['u_upper_at_beta_hot']))} < 1/3: {c2}")
            if not (c1 and c2):
                mismatch("hot.beta_hot", "point bound")
        elif nm == "beta_hot3":
            c1 = pb.upper() <= fr(H["pbar_upper_at_beta_hot3"]) < Fr(1, 3) and E_iv.hi(pbi) <= fr(H["pbar_upper_at_beta_hot3"])
            say(f"   pbar <= {float(fr(H['pbar_upper_at_beta_hot3']))} < 1/3: {c1}")
            if not c1:
                mismatch("hot.beta_hot3", "point bound")
        elif nm == "sharp beta":
            c1 = pb.lower() >= fr(H["sharpness"]["pbar_lower"]) > Fr(1, 2)
            c2 = u.lower() >= fr(H["sharpness"]["u_lower"]) > Fr(1, 3)
            say(f"   sharpness: pbar >= {float(fr(H['sharpness']['pbar_lower']))} > 1/2: {c1}; u >= {float(fr(H['sharpness']['u_lower']))} > 1/3: {c2}")
            if not (c1 and c2):
                mismatch("hot.sharpness", "lower bound")
        else:
            c1 = pb.lower() >= fr(H["sharpness"]["pbar3_lower"]) > Fr(1, 3)
            say(f"   sharpness: pbar >= {float(fr(H['sharpness']['pbar3_lower']))} > 1/3: {c1}")
            if not c1:
                mismatch("hot.sharpness3", "lower bound")
    # Lemma-H-free, one piece: for beta in (0, bh]: m1 <= th(bh), m3 = tanh(beta-kappa2) <= tanh(beta) <= th(bh),
    # m2 = tanh(beta+kappa2(beta)) <= m2(bh) (kappa2 nondecreasing): x_n <= th_h^(2+n1+n3) m2_h^n2.
    for nm, beta, tests in (("(0, beta_hot]", bh, (("pbar", Fr(1, 2)), ("u", Fr(1, 3)))), ("(0, beta_hot3]", bh3, (("pbar", Fr(1, 3)),))):
        th, _, m2, _ = M.mags_beta(FX, FX.const(beta))
        Xb = M.xs(FX, (th, None, m2, th), law)
        vals = {"u": M.u_value(FX, wFX, Xb).upper(), "pbar": M.pbar_value(FX, wFX, Xb).upper()}
        for f, tv in tests:
            say(f"hot Lemma-H-free one-piece bound on {nm}: sup {f}_H <= {float(vals[f]):.15f} < {tv}: {vals[f] < tv}")
            if not vals[f] < tv:
                mismatch(f"hot one-piece {nm} {f}", str(float(vals[f])))
    # Lemma-H-free multi-piece cover with monotone brackets (uniform in beta, 120 pieces)
    for nm, beta, f, tv, ck in (("(0, beta_hot]", bh, "pbar", Fr(1, 2), "pbar_half"), ("(0, beta_hot]", bh, "u", Fr(1, 3), "u_third"),
                                ("(0, beta_hot3]", bh3, "pbar", Fr(1, 3), "pbar_third")):
        npc = 120
        sup = Fr(0)
        for j in range(npc):
            a_, b_ = beta * j / npc, beta * (j + 1) / npc
            Xb = xs_from_brackets(*brackets_fx(a_, b_), law)
            v = (M.u_value(FX, wFX, Xb) if f == "u" else M.pbar_value(FX, wFX, Xb)).upper()
            sup = max(sup, v)
        csup = fr(H["lemma_H_free_cover"][ck]["sup"])
        say(f"hot bracket cover of {nm}, {npc} pieces: sup {f}_H <= {float(sup):.15f} < {tv}: {sup < tv}  (claims {ck}: sup {float(csup):.15f} < {tv}: {csup < tv}, ok={H['lemma_H_free_cover'][ck]['ok']})")
        if not (csup < tv and H["lemma_H_free_cover"][ck]["ok"]):
            mismatch(f"hot.lemma_H_free_cover.{ck}", "claimed sup not below threshold")
        if not sup < tv:
            mismatch(f"hot cover {nm} {f}", str(float(sup)))
    # Lemma H condition and tanh(5/2) <= theta_bar
    mono = H["monotonicity"]
    B = fr(mono["B"])
    tb = fr(mono["theta_bar"])
    lhs = 2 * (b0 + 2) * tb ** (b0 + 1)
    rhs = 1 - 4 * tb ** (2 * b0 + 4)
    condH = lhs <= rhs
    # e^5 upper bound by Taylor with remainder (exact rationals): e^5 <= sum_{j<=N} 5^j/j! + 2*5^(N+1)/(N+1)!
    N = 80
    s_ = Fr(0)
    term = Fr(1)
    for j in range(N + 1):
        s_ += term
        term = term * 5 / (j + 1)
    e5_up = s_ + 2 * term  # term = 5^(N+1)/(N+1)!; tail ratio <= 5/(N+2) < 1/2
    tanh_ok = (e5_up - 1) / (e5_up + 1) <= tb  # tanh(5/2) = (e^5-1)/(e^5+1) increasing in e^5
    e5_fx = FX.exp(FX.const(5))
    tanh_fx = ((e5_fx - 1) / (e5_fx + 1)).upper() <= tb
    say(f"Lemma H: B = {B}; tanh(5/2) <= theta_bar: exact-Taylor {tanh_ok}, fxi {tanh_fx}; (H*) 2(b0+2)tb^(b0+1) = {float(lhs):.15e} <= 1-4tb^(2b0+4) = {float(rhs):.15f}: {condH} (claims lhs {mono['lhs_float']}, rhs {mono['rhs_float']}); beta_hot, beta_hot3 <= B: {bh <= B and bh3 <= B}")
    if not (condH and tanh_ok and tanh_fx and bh <= B and bh3 <= B):
        mismatch("hot.monotonicity", "failed")
    if abs(float(lhs) - mono["lhs_float"]) > 1e-15 or abs(float(rhs) - mono["rhs_float"]) > 1e-15:
        mismatch("hot.monotonicity floats", f"{float(lhs)} {float(rhs)}")
    for nm2, bb in (("beta_hot", bh), ("beta_hot3", bh3)):
        if abs((float(bb) / float(g0)) - H[nm2 + "_over_gamma"]) > 1e-12:
            mismatch(f"hot.{nm2}_over_gamma", H[nm2 + "_over_gamma"])

    # ---------------- Nishimori line
    NL = C["nishimori_line"]
    vlo, vhi = fr(NL["vP_bracket"][0]), fr(NL["vP_bracket"][1])
    brk = M.h_(vlo) < 1 < M.h_(vhi) and 9 * vhi * vhi < 1
    say(f"v_P bracket [{float(vlo):.15f}, {float(vhi):.15f}]: h(lo) < 1 < h(hi) exactly: {brk}; h(lo) = {float(M.h_(vlo)):.3e}-1 offset {float(M.h_(vlo) - 1):.3e}, h(hi)-1 = {float(M.h_(vhi) - 1):.3e}")
    if not brk:
        mismatch("nishimori_line.vP_bracket", "failed")

    def v_enclosures(p):
        Lp = M.Law(p)
        th = FX.const(1 - 2 * p)
        mg = M.mags_theta(FX, th)
        X = M.xs(FX, mg, Lp)
        vs = M.v_signed(FX, M.law_in_engine(FX, Lp), X)
        vn = M.v_sech(FX, M.weights_in_engine(FX, Lp), X)
        Ei = E_iv
        thi = Ei.const(1 - 2 * p)
        Xi = M.xs(Ei, M.mags_theta(Ei, thi), Lp)
        vi = M.v_signed(Ei, M.law_in_engine(Ei, Lp), Xi)
        # exact Nishimori identities rho_c = m_c at theta = 1-2p
        _, _, m2e, m3e = M.mags_exact_theta(1 - 2 * p)
        rho = [Lp.unit_class(c)[1] for c in (1, 2, 3)]
        nish = [rho[0] == 1 - 2 * p, rho[1] == m2e, rho[2] == m3e]
        lo = max(vs.lower(), vn.lower(), Ei.lo(vi))
        hi = min(vs.upper(), vn.upper(), Ei.hi(vi))
        agree = not (vs.upper() < vn.lower() or vn.upper() < vs.lower())
        return lo, hi, nish, agree, (vs, vn, vi)

    for ent in NL["p1"]:
        p1 = fr(ent["p1"])
        lo, hi, nish, agree, _ = v_enclosures(p1)
        V = fr(ent["V"])
        okn = fr(ent["v_lower"]) <= lo and hi <= fr(ent["v_upper"]) <= V and V < vlo and M.h_(V) < 1 and 9 * V * V < 1
        g1 = M.gamma_float(p1, 30)
        say(f"N p1={p1}: v_H(gamma(p1)) in [{float(lo):.15f}, {float(hi):.15f}] (signed and sech forms agree: {agree}); v_lower <= v <= v_upper <= V = {float(V):.12f} < vP_lo, h(V) < 1: {okn}; h(V) = {float(M.h_(V)):.12f} (claims {ent['hV_float']}); margin vP_lo - V = {float(vlo - V):.6e} (claims {ent['margin_float']}); rho_c == m_c exactly: {nish} (claims {ent['nishimori_exact']}); gamma(p1) = {float(g1):.15f} (claims {ent['gamma_p1'][0]})")
        if not (okn and all(nish) and agree):
            mismatch(f"nishimori_line.p1 {ent['p1']}", "failed")
        if abs(float(M.h_(V)) - ent["hV_float"]) > 1e-12 or abs(float(g1) - ent["gamma_p1"][0]) > 1e-12:
            mismatch(f"nishimori_line.p1 {ent['p1']} floats", "display")
        if abs(float(vlo - V) - ent["margin_float"]) > 1e-12:
            say(f"   note: margin_float {ent['margin_float']} vs vP_lo - V = {float(vlo - V):.12e} (vP - V with vP ~ {float((vlo + vhi) / 2):.15f}: {float((vlo + vhi) / 2 - V):.12e})")
    sh = NL["sharpness"]
    lo, hi, _, _, _ = v_enclosures(fr(sh["p"]))
    oks = lo >= fr(sh["v_lower"]) > vhi
    say(f"N sharpness p={sh['p']}: v_H >= {float(lo):.15f} >= v_lower {float(fr(sh['v_lower'])):.15f} > vP_hi: {oks}")
    if not oks:
        mismatch("nishimori_line.sharpness", "failed")
    prev_hi = None
    inc = True
    for ent in NL["display"]:
        p = fr(ent["p"])
        lo, hi, nish, agree, _ = v_enclosures(p)
        okd = fr(ent["v_lower"]) <= lo and hi <= fr(ent["v_upper"])
        say(f"N display p={ent['p']}: v_H in [{float(lo):.15f}, {float(hi):.15f}] inside claimed [{float(fr(ent['v_lower'])):.15f}, {float(fr(ent['v_upper'])):.15f}]: {okd}; rho=m exact: {all(nish)}")
        if not okd:
            mismatch(f"nishimori_line.display {ent['p']}", "enclosure")
        if prev_hi is not None and not prev_hi < lo:
            inc = False
        prev_hi = hi
    say(f"N display strictly increasing (rigorous, consecutive enclosures separated): {inc} (claims {NL['display_increasing']})")
    if inc != NL["display_increasing"]:
        mismatch("nishimori_line.display_increasing", "differs")

    R["n_mismatches"] = len(R["mismatches"])
    R["elapsed_s"] = time.time() - t0
    say(f"TOTAL MISMATCHES: {len(R['mismatches'])}; elapsed {R['elapsed_s']:.1f}s")
    R["log"] = LOG
    json.dump(R, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
