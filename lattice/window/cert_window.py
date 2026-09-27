"""Certificates of the inputs (C1)-(C3), (C7), (C8) of Appendix C.7 (app:lat-cert) for design P of the
decorated lattice Z^2[H4] of Section 9 (b0 = 1000, k = 10, p0 = 9/10000).

Labels: "Lemma L" is the samplewise bound |K_H(J,b) - K_H(J,b')| <= 3|b - b'| behind the cell rule of
Lemma C.12 (lem:lat-cell); "Lemma H" is a hot-side monotonicity criterion that the paper does not use
(its input (C1) rests on the Lemma-H-free covers); "Lemma D" is Lemma 9.10 (lem:lat-degradation).

Modes
  certify CLAIMS_JSON   Rigorous. python-flint Arb (160 bits) plus exact fmpq. Imports the exact law of
                        ../cert/cert_arb.py (unit_law, classes, mags_interval, Design.warm, vp_bracket,
                        ball, bounds). Writes the machine-readable claims file; prints a report.
  verify CLAIMS_JSON    Second code path, NOT rigorous (mpmath, 60 digits). Rebuilds the law of K_H by
                        enumerating the 32 signed unit disorders (no class formulas, no rho, no s-Z
                        parametrization), recomputes every certified quantity, and re-checks every
                        rational inequality of the claims file exactly with fractions.Fraction.
  lipcheck              Float evidence for Lemma L (|dK_H/dbeta| <= 3 samplewise), Lemma H (hot-side
                        monotonicity) and Lemma D (Bhattacharyya monotone in p on the Nishimori line),
                        including brute-force partition functions on small gadgets.

Certified claims (certify):
  (W+)  cells [bl, br] covering [BETA1_PLUS, BETA2_PLUS] with rational s in (0, 1]:
          F_up >= F(bl, s) := E exp(-2 s K_H(bl))   (Arb enclosure at the rational point bl),
          y := 6 s (br - bl),  W_up >= F_up (2+y)/(2-y) >= F_up e^y,  W_up <= W_STAR,
        and 2P(W_STAR) < 1 exactly.  With Lemma L this gives E exp(-2 s K_H(beta)) <= W_STAR
        for all beta in the cell.  Independently of Lemma L, an Arb enclosure of F over the whole
        cell (beta-ball) is also checked against W_STAR.
  (W2)  the same with W_BB, 15 W_BB^2 < 1, on [BETA1_BB, BETA2_BB].
  (H)   pbar_H(BETA_HOT) < 1/2 and u_H(BETA_HOT) < 1/3 (Arb), pbar_H(BETA_HOT3) < 1/3, plus the exact
        rational condition of Lemma H at theta_bar >= tanh(BETA_MONO), which makes u_H and pbar_H
        nondecreasing on (0, BETA_MONO].  Independently of Lemma H, Arb covers of [0, BETA_HOT] and
        [0, BETA_HOT3] by beta-balls (theta-parametrization).
  (N)   v_H(gamma(p1)) < v_P for p1 in P1_LIST (exact rationals + Arb square roots, cert_arb.Design.warm).
  (S)   sharpness points (route-specific, not needed downstream): the routes fail just outside.

Usage (from this directory; certify needs python-flint 0.9.0):
  python cert_window.py certify out/claims.json > out/cert_window.txt
  python cert_window.py verify out/claims.json > out/verify.txt
  python cert_window.py lipcheck > out/lipcheck.txt
"""
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from itertools import product
from math import comb

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.normpath(os.path.join(HERE, "..", "cert"))

B0, K = 1000, 10
P0 = Fraction(9, 10000)
LIP = 3                                          # Lemma L (Lemma C.12): |K_H(J,b) - K_H(J,b')| <= 3|b - b'|
W_STAR = Fraction(274797, 10 ** 6)               # plus-state threshold: 2P(W_STAR) < 1
W_BB = Fraction(258198, 10 ** 6)                 # two-point threshold: 15 W_BB^2 < 1
BETA1_PLUS, BETA2_PLUS = Fraction(196616, 10 ** 5), Fraction(491155, 10 ** 5)
BETA1_BB, BETA2_BB = Fraction(201692, 10 ** 5), Fraction(479575, 10 ** 5)
Y_CAP = Fraction(1, 100)                         # per-cell Lipschitz inflation exponent cap (keeps W_up ~ w)
S_DEN = 1000                                     # s_j lies on the grid (1/1000) Z
DEC = 15                                         # decimal grid for stored upper/lower bounds
BETA_HOT = Fraction(154235, 10 ** 5)             # pbar_H < 1/2 and u_H < 1/3 on (0, BETA_HOT]
BETA_HOT3 = Fraction(135183, 10 ** 5)            # pbar_H < 1/3 on (0, BETA_HOT3]
BETA_MONO = Fraction(5, 2)                       # Lemma H checked on (0, 5/2]
P1_LIST = [Fraction(1, 1000), Fraction(529, 500000)]
P_SHARP = Fraction(10581, 10 ** 7)
P_DISPLAY = [Fraction(1, 2000), Fraction(7, 10000), Fraction(9, 10000), Fraction(1, 1000),
             Fraction(529, 500000), Fraction(10581, 10 ** 7), Fraction(11, 10000), Fraction(12, 10000)]
# rational beta-bands (approximately 0.60-1.35, 0.65-1.30, 0.70-1.20 gamma) for quantitative summaries
BANDS = [(Fraction(21037, 10 ** 4), Fraction(47332, 10 ** 4)),
         (Fraction(22790, 10 ** 4), Fraction(45579, 10 ** 4)),
         (Fraction(24543, 10 ** 4), Fraction(42073, 10 ** 4))]


def fs(x):
    x = Fraction(x)
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"


def P_fn(v):
    """Contour sum P(v) = 9v^4/(1-9v^2)^2 of Section 9.3 and Lemma C.7 (exact on rationals)."""
    return 9 * v ** 4 / (1 - 9 * v * v) ** 2


def Q_fn(v):
    return 9 * v ** 3 / (1 - 9 * v * v)


def h_fn(v):
    return 4 * P_fn(v) + 4 * Q_fn(v)


def up_dec(x, d=DEC):
    """Smallest element of 10^-d Z that is >= the rational x."""
    x = Fraction(x)
    return Fraction(-((-x.numerator * 10 ** d) // x.denominator), 10 ** d)


def down_dec(x, d=DEC):
    x = Fraction(x)
    return Fraction((x.numerator * 10 ** d) // x.denominator, 10 ** d)


def down_2sig(x):
    """Round a positive rational down to two significant decimal digits."""
    x = Fraction(x)
    e = math.floor(math.log10(x.numerator) - math.log10(x.denominator))
    scale = Fraction(10) ** (e - 1)
    while x / scale >= 100:
        scale *= 10
    while x / scale < 10:
        scale /= 10
    return math.floor(x / scale) * scale


def exact_threshold_checks():
    out = {}
    out["W_STAR"] = dict(value=fs(W_STAR), nine_w2_lt_1=bool(9 * W_STAR ** 2 < 1),
                         two_P_lt_1=bool(9 * W_STAR ** 2 < 1 and 18 * W_STAR ** 4 < (1 - 9 * W_STAR ** 2) ** 2),
                         one_minus_2P=float(1 - 2 * P_fn(W_STAR)))
    out["W_BB"] = dict(value=fs(W_BB), fifteen_w2_lt_1=bool(15 * W_BB ** 2 < 1),
                       one_minus_4P=float(1 - 4 * P_fn(W_BB)))
    return out


# ============================================================================== certify (Arb)
def certify(out_json):
    sys.path.insert(0, CERT_DIR)
    import cert_arb as CA
    from flint import arb, fmpq

    def Q(x):
        x = Fraction(x)
        return fmpq(x.numerator, x.denominator)

    def F_(x):
        return Fraction(int(x.p), int(x.q))

    def bnd(x):
        lo, hi = CA.bounds(x)
        return F_(lo), F_(hi)

    t_start = time.time()
    cert_arb_path = os.path.join(CERT_DIR, "cert_arb.py")
    sha_cert_arb = hashlib.sha256(open(cert_arb_path, "rb").read()).hexdigest()
    sha_self = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    import flint
    print(f"cert_window.py certify: python-flint {flint.__version__}, Arb prec {CA.PREC} bits")
    print(f"  imports {os.path.relpath(cert_arb_path, HERE)} (sha256 {sha_cert_arb})")
    print(f"  self sha256 {sha_self}")

    class Law:
        """Exact class data of cert_arb.py (cert_arb.py labeling: class 1 = non-cancel |tanh| w1,
        class 2 = paths disagree |tanh| t, class 3 = cancel |tanh| w3)."""

        def __init__(self, p):
            self.p = Q(p)
            self.L = CA.unit_law(self.p, B0)
            P1, P2, P3 = self.L["P"]
            r1, r2, r3 = self.L["rho"]
            self.cls = []
            for n1, n2, n3, M in CA.classes(K):
                W = M * P1 ** n1 * P2 ** n2 * P3 ** n3
                A = (1 - 2 * self.p) ** 2 * r1 ** n1 * r2 ** n2 * r3 ** n3
                self.cls.append((n1, n2, n3, W, A, arb(W), arb((1 + A) / 2), arb((1 - A) / 2), CA.fl(W), CA.fl(A)))
            assert sum(c[3] for c in self.cls) == 1

        def atoms_sZ(self, lo, hi):
            """log r_n = log((1-x_n)/(1+x_n)) = -2|K_H| per class, enclosed over beta in [lo, hi]."""
            t, w1, w3 = CA.mags_interval(Q(lo), Q(hi), B0)
            return self._atoms(t, w1, w3)

        def mags_theta(self, lo, hi):
            """theta-parametrization (valid down to beta = 0): tau = 2 th^(b0+2)/(1+th^(2b0)).
            Returns (t, w1, w3) in cert_arb labels, enclosed over beta in [lo, hi]."""
            th = CA.ball(Q(lo), Q(hi)).tanh()
            tb = th ** B0
            a = th * th * 2 * tb / (1 + tb * tb)
            return th, (th + a) / (1 + th * a), (th - a) / (1 - th * a)

        def _atoms(self, t, w1, w3):
            pt = [t ** i for i in range(K + 3)]
            p1 = [w1 ** i for i in range(K + 1)]
            p3 = [w3 ** i for i in range(K + 1)]
            out = []
            for n1, n2, n3, W, A, Wa, Ap, Am, Wf, Af in self.cls:
                x = pt[2 + n2] * p1[n1] * p3[n3]
                out.append(((1 - x) / (1 + x)).log())
            return out

        def F(self, lr, s):
            s = arb(Q(s))
            tot = arb(0)
            for c, l in zip(self.cls, lr):
                tot += c[5] * (c[6] * (s * l).exp() + c[7] * (-s * l).exp())
            return tot

        def dF(self, lr, s):
            s = arb(Q(s))
            tot = arb(0)
            for c, l in zip(self.cls, lr):
                tot += c[5] * l * (c[6] * (s * l).exp() - c[7] * (-s * l).exp())
            return tot

        def s_opt_float(self, lr):
            lf = [float(l.mid()) for l in lr]
            W = [(c[8], (1 + c[9]) / 2, (1 - c[9]) / 2) for c in self.cls]

            def f(s):
                return sum(w * (ap * math.exp(s * l) + am * math.exp(-s * l)) for (w, ap, am), l in zip(W, lf))
            g = (math.sqrt(5) - 1) / 2
            a, b = 0.0, 1.0
            c, d = b - g * (b - a), a + g * (b - a)
            fc, fd = f(c), f(d)
            for _ in range(80):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - g * (b - a)
                    fc = f(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + g * (b - a)
                    fd = f(d)
            return (a + b) / 2, f((a + b) / 2)

    law = Law(P0)
    gam = arb(Q((1 - P0) / P0)).log() / 2
    g_lo, g_hi = bnd(gam)
    gf = float(g_lo)
    print(f"design P: b0={B0} k={K} p0={fs(P0)}; gamma in [{float(g_lo):.15f}, {float(g_hi):.15f}]")

    # --- consistency of the class data with Proposition 9.3 (alpha_1, alpha_2, alpha_3 there) --------
    p = P0
    q = (1 - (1 - 2 * p) ** B0) / 2
    a_ = (1 + (1 - 2 * p) ** 3) / 2
    GP1 = 2 * q * (1 - q)
    GP3 = (1 - q) ** 2 * (1 - a_) + q * q * a_
    GP2 = 1 - GP1 - GP3
    CP = [F_(x) for x in law.L["P"]]
    map_ok = (CP[0] == GP2 and CP[1] == GP1 and CP[2] == GP3)
    print(f"  class map cert_arb (P1,P2,P3) = Prop. 9.3 (alpha_2,alpha_1,alpha_3) exactly: {map_ok}")
    assert map_ok
    # regression: F(gamma, 1/2) = v_H(gamma)
    lr_g = law.atoms_sZ(g_lo, g_hi)
    Fg = law.F(lr_g, Fraction(1, 2))
    print(f"  regression: F(gamma, 1/2) = E e^(-K_H(gamma)) in [{float(bnd(Fg)[0]):.15f}, {float(bnd(Fg)[1]):.15f}] (cert_arb: 0.205295045989685)")
    assert abs(float(Fg.mid()) - 0.205295045989685) < 1e-12

    thr = exact_threshold_checks()
    print(f"(T) W_STAR = {fs(W_STAR)}: 9w^2 < 1 and 2P(w) < 1 exactly: {thr['W_STAR']['two_P_lt_1']}  (1-2P = {thr['W_STAR']['one_minus_2P']:.3e})")
    print(f"    W_BB   = {fs(W_BB)}: 15 w^2 < 1 exactly: {thr['W_BB']['fifteen_w2_lt_1']}  (1-4P = {thr['W_BB']['one_minus_4P']:.3e})")
    assert thr["W_STAR"]["two_P_lt_1"] and thr["W_BB"]["fifteen_w2_lt_1"]

    # --- (W+), (W2): Lipschitz cells -------------------------------------------------------------
    def cover(beta1, beta2, wthr, tag):
        t0 = time.time()
        cells = []
        x = beta1
        n_ball_sub = 0
        while x < beta2:
            lr = law.atoms_sZ(x, x)
            s_f, _ = law.s_opt_float(lr)
            s = Fraction(min(S_DEN, max(1, round(s_f * S_DEN))), S_DEN)
            Fb = law.F(lr, s)
            Fup = up_dec(bnd(Fb)[1])
            if not Fup < wthr:
                raise RuntimeError(f"{tag}: F_up = {float(Fup)} >= threshold at beta = {fs(x)}")
            R = wthr / Fup
            ymax = min(2 * (R - 1) / (R + 1), Y_CAP)
            d = down_2sig(ymax / (LIP * 2 * s) * Fraction(999, 1000))
            while True:
                xr = min(x + d, beta2)
                y = 2 * LIP * s * (xr - x)
                Wup = up_dec(Fup * (2 + y) / (2 - y))
                if Wup <= wthr:
                    break
                d = down_2sig(d / 2)
            # Lemma-L-free cross-check: Arb enclosure over the whole cell, bisected if needed
            ball_up, nsub = ball_check(x, xr, s, wthr)
            n_ball_sub += nsub - 1
            cells.append(dict(bl=x, br=xr, s=s, F_up=Fup, y=y, W_up=Wup, ball_up=ball_up, ball_subcells=nsub,
                              F_mid=float(Fb.mid())))
            x = xr
        cover_ok = cells[0]["bl"] == beta1 and cells[-1]["br"] == beta2 and all(
            cells[i]["br"] == cells[i + 1]["bl"] for i in range(len(cells) - 1))
        ok = cover_ok and all(c["W_up"] <= wthr and 0 < c["s"] <= 1 and c["y"] < 2 for c in cells)
        ball_ok = cover_ok and all(c["ball_up"] is not None and c["ball_up"] <= wthr for c in cells)
        wmax = max(c["W_up"] for c in cells)
        print(f"({tag}) cover of [{fs(beta1)}, {fs(beta2)}] = [{float(beta1)/gf:.6f}, {float(beta2)/gf:.6f}] gamma, threshold {fs(wthr)}:")
        print(f"      {len(cells)} cells, contiguous {cover_ok}; all W_up <= threshold: {ok};  sup W_up = {float(wmax):.12f}")
        print(f"      min cell width {float(min(c['br'] - c['bl'] for c in cells)):.3g}, max {float(max(c['br'] - c['bl'] for c in cells)):.3g};"
              f" s range [{float(min(c['s'] for c in cells))}, {float(max(c['s'] for c in cells))}]")
        print(f"      Lemma-L-free Arb beta-ball enclosure over every cell <= threshold: {ball_ok} (extra bisections: {n_ball_sub})   [{time.time()-t0:.1f}s]")
        return dict(cells=cells, ok=ok, contiguous=cover_ok, ball_ok=ball_ok, sup_W_up=wmax)

    def ball_check(lo, hi, s, wthr, depth=0):
        lr = law.atoms_sZ(lo, hi)
        up = bnd(law.F(lr, s))[1]
        if up <= wthr:
            return up_dec(up), 1
        if depth >= 12:
            return None, 1
        mid = (lo + hi) / 2
        u1, n1 = ball_check(lo, mid, s, wthr, depth + 1)
        u2, n2 = ball_check(mid, hi, s, wthr, depth + 1)
        if u1 is None or u2 is None:
            return None, n1 + n2
        return max(u1, u2), n1 + n2

    def bands(cov, kind):
        rows = []
        for lo, hi in BANDS:
            ws = max(c["W_up"] for c in cov["cells"] if c["br"] > lo and c["bl"] < hi)
            inside = cov["cells"][0]["bl"] <= lo and cov["cells"][-1]["br"] >= hi
            if kind == "plus":
                val = down_dec(1 - 2 * P_fn(ws), 6)
            else:
                val = down_dec(1 - 4 * P_fn(ws), 6)
            rows.append(dict(lo=fs(lo), hi=fs(hi), lo_over_gamma=float(lo) / gf, hi_over_gamma=float(hi) / gf,
                             inside_cover=inside, sup_W_up=fs(ws), bound=fs(val), bound_float=float(val)))
            what = "1-2P(w) >=" if kind == "plus" else "1-4P(w) >="
            print(f"      band [{fs(lo)}, {fs(hi)}] = [{float(lo)/gf:.4f}, {float(hi)/gf:.4f}] gamma: sup W_up = {float(ws):.9f}, {what} {float(val):.6f}  (inside cover: {inside})")
        return rows

    def w_lower(beta, N=4000):
        """Rigorous lower bound on min_{s in [0,1]} F(beta, s) (convexity in s + tangent lines)."""
        lr = law.atoms_sZ(beta, beta)
        grid = [Fraction(i, N) for i in range(N + 1)]
        Fv = [law.F(lr, s) for s in grid]
        dv = [law.dF(lr, s) for s in grid]
        best = None
        for i in range(N):
            s0, s1 = grid[i], grid[i + 1]
            F0lo, F1lo = bnd(Fv[i])[0], bnd(Fv[i + 1])[0]
            d0lo, d0hi = bnd(dv[i])
            d1lo, d1hi = bnd(dv[i + 1])
            if d0lo >= 0:
                lb = F0lo
            elif d1hi <= 0:
                lb = F1lo
            elif d0hi < 0 and d1lo > 0:
                a0 = Fv[i] - dv[i] * arb(Q(s0))
                a1 = Fv[i + 1] - dv[i + 1] * arb(Q(s1))
                sx = (a0 - a1) / (dv[i + 1] - dv[i])
                lb = bnd(a0 + dv[i] * sx)[0]
            else:
                h = s1 - s0
                lb = max(F0lo + min(0, d0lo) * h, F1lo - max(0, d1hi) * h)
            best = lb if best is None else min(best, lb)
        return down_dec(best)

    covP = cover(BETA1_PLUS, BETA2_PLUS, W_STAR, "W+")
    bandsP = bands(covP, "plus")
    covB = cover(BETA1_BB, BETA2_BB, W_BB, "W2")
    bandsB = bands(covB, "bb")

    # sharpness: the Chernoff-optimized weight exceeds the exact threshold just outside
    t0 = time.time()
    sharp_w = []
    for beta, kind in ((BETA1_PLUS - Fraction(1, 10 ** 4), "plus"), (BETA2_PLUS + Fraction(1, 10 ** 4), "plus"),
                       (BETA1_BB - Fraction(1, 10 ** 4), "bb"), (BETA2_BB + Fraction(1, 10 ** 4), "bb")):
        wl = w_lower(beta)
        if kind == "plus":
            fails = bool(9 * wl ** 2 < 1 and 18 * wl ** 4 >= (1 - 9 * wl ** 2) ** 2)   # 2P(wl) >= 1
        else:
            fails = bool(15 * wl ** 2 >= 1)
        sharp_w.append(dict(beta=fs(beta), beta_over_gamma=float(beta) / gf, kind=kind, w_lower=fs(wl), route_fails=fails))
        print(f"(S) w(beta) = min_s F(beta,s) >= {float(wl):.12f} at beta = {fs(beta)} ({float(beta)/gf:.6f} gamma): "
              f"{'2P(w) >= 1' if kind == 'plus' else '15 w^2 >= 1'}: {fails}")
    print(f"    [{time.time()-t0:.1f}s]")

    # --- (H) hot side -----------------------------------------------------------------------------
    t0 = time.time()
    D0 = CA.Design("P", B0, K, Q(P0))

    def pt(beta):
        f = D0.eval_interval(Q(beta), Q(beta))
        return dict(u=bnd(f["u"]), pbar=bnd(f["pbar"]))
    hot = pt(BETA_HOT)
    hot3 = pt(BETA_HOT3)
    hot_s = pt(BETA_HOT + Fraction(1, 10 ** 5))
    hot3_s = pt(BETA_HOT3 + Fraction(1, 10 ** 5))
    hot_ok = hot["pbar"][1] < Fraction(1, 2) and hot["u"][1] < Fraction(1, 3)
    hot3_ok = hot3["pbar"][1] < Fraction(1, 3)
    print(f"(H) pbar_H({fs(BETA_HOT)}) <= {float(hot['pbar'][1]):.12f} < 1/2 and u_H <= {float(hot['u'][1]):.12f} < 1/3: {hot_ok}"
          f"   (beta_hot = {float(BETA_HOT)/gf:.6f} gamma)")
    print(f"    pbar_H({fs(BETA_HOT3)}) <= {float(hot3['pbar'][1]):.12f} < 1/3: {hot3_ok}   ({float(BETA_HOT3)/gf:.6f} gamma)")
    print(f"(S) pbar_H({fs(BETA_HOT + Fraction(1, 10**5))}) >= {float(hot_s['pbar'][0]):.12f} > 1/2: {hot_s['pbar'][0] > Fraction(1, 2)};"
          f"  u_H >= {float(hot_s['u'][0]):.12f} > 1/3: {hot_s['u'][0] > Fraction(1, 3)}")
    print(f"    pbar_H({fs(BETA_HOT3 + Fraction(1, 10**5))}) >= {float(hot3_s['pbar'][0]):.12f} > 1/3: {hot3_s['pbar'][0] > Fraction(1, 3)}")
    th_hi = bnd(arb(Q(BETA_MONO)).tanh())[1]
    th_bar = up_dec(th_hi, 30)
    lhs = 2 * (B0 + 2) * th_bar ** (B0 + 1)
    rhs = 1 - 4 * th_bar ** (2 * B0 + 4)
    lemmaH_ok = bool(th_bar < 1 and lhs <= rhs)
    print(f"(H) Lemma H at theta_bar = {float(th_bar):.20f} >= tanh({fs(BETA_MONO)}): 2(b0+2) th^(b0+1) = {float(lhs):.6e} <= 1 - 4 th^(2b0+4) = {float(rhs):.12f}: {lemmaH_ok}")
    print(f"    hence u_H, pbar_H nondecreasing on (0, {fs(BETA_MONO)}] = (0, {float(BETA_MONO)/gf:.4f} gamma]; {fs(BETA_HOT)} <= {fs(BETA_MONO)}: {BETA_HOT <= BETA_MONO}")

    def hot_cover(beta_end, key, thr_):
        """Lemma-H-free cross-check: Arb beta-ball cover of [0, beta_end] (theta parametrization)."""
        pieces, stack = [], []
        x = Fraction(0)
        step = Fraction(1, 64)
        while x < beta_end:
            y = min(x + step, beta_end)
            stack.append((x, y))
            x = y
        stack.reverse()
        worst = Fraction(0)
        while stack:
            lo, hi = stack.pop()
            th, w1, w3 = law.mags_theta(lo, hi)
            f = D0.functionals(th, w1, w3)[key]
            up = bnd(f)[1]
            if up < thr_:
                pieces.append((lo, hi))
                worst = max(worst, up)
            elif hi - lo > Fraction(1, 10 ** 12):
                mid = (lo + hi) / 2
                stack.append((mid, hi))
                stack.append((lo, mid))
            else:
                return dict(ok=False, fail=(fs(lo), fs(hi)))
        pieces.sort()
        cont = pieces[0][0] == 0 and pieces[-1][1] == beta_end and all(
            pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))
        return dict(ok=cont, n_pieces=len(pieces), sup=fs(up_dec(worst)), min_width=fs(min(b - a for a, b in pieces)))
    hc = hot_cover(BETA_HOT, "pbar", Fraction(1, 2))
    hcu = hot_cover(BETA_HOT, "u", Fraction(1, 3))
    hc3 = hot_cover(BETA_HOT3, "pbar", Fraction(1, 3))
    print(f"    Lemma-H-free Arb cover of [0, {fs(BETA_HOT)}]: pbar < 1/2 {hc['ok']} ({hc.get('n_pieces')} pieces, sup <= {float(Fraction(hc.get('sup', '0'))):.12f});"
          f" u < 1/3 {hcu['ok']} ({hcu.get('n_pieces')} pieces)")
    print(f"    Lemma-H-free Arb cover of [0, {fs(BETA_HOT3)}]: pbar < 1/3 {hc3['ok']} ({hc3.get('n_pieces')} pieces, sup <= {float(Fraction(hc3.get('sup', '0'))):.12f})   [{time.time()-t0:.1f}s]")

    # --- (N) Nishimori line, larger p ---------------------------------------------------------------
    t0 = time.time()
    vp = CA.vp_bracket(15)
    vp_lo, vp_hi = F_(vp[0]), F_(vp[1])
    print(f"(VP) v_P in [{fs(vp_lo)}, {fs(vp_hi)}]: h(lo) < 1 < h(hi) exactly: {h_fn(vp_lo) < 1 < h_fn(vp_hi)}")
    nish = []
    for p1 in P1_LIST:
        D = CA.Design(f"p1={fs(p1)}", B0, K, Q(p1))
        Wm = D.warm(vp[0])
        vlo, vhi = F_(Wm["v_bounds"][0]), F_(Wm["v_bounds"][1])
        V = F_(Wm["V"])
        ok = bool(Wm["ok"]) and V < vp_lo and 3 * V < 1 and h_fn(V) < 1
        g1 = CA.bounds(D.gamma)
        nish.append(dict(p1=fs(p1), gamma_p1=[float(F_(g1[0])), float(F_(g1[1]))], v_lower=fs(down_dec(vlo)), v_upper=fs(up_dec(vhi)),
                         V=fs(V), hV_float=float(h_fn(V)), V_lt_vP_lo=bool(V < vp_lo), hV_lt_1=bool(h_fn(V) < 1),
                         nishimori_exact=[bool(z) for z in Wm["nishimori_exact"]], ok=ok, margin_float=float(vp_lo - V)))
        print(f"(N) p1 = {fs(p1)}: v_H(gamma(p1)) in [{float(vlo):.15f}, {float(vhi):.15f}], V = {fs(V)}; h(V) = {float(h_fn(V)):.12f} < 1: {h_fn(V) < 1};"
              f" V < vP_lo: {V < vp_lo} (margin {float(vp_lo - V):.3e}); Nishimori identities exact: {list(Wm['nishimori_exact'])}; OK {ok}")
    Ds = CA.Design("sharp", B0, K, Q(P_SHARP))
    Ws = Ds.warm(vp[0])
    vs_lo = F_(Ws["v_bounds"][0])
    sharp_n = dict(p=fs(P_SHARP), v_lower=fs(down_dec(vs_lo)), exceeds_vP_hi=bool(vs_lo > vp_hi))
    print(f"(S) p = {fs(P_SHARP)}: v_H(gamma(p)) >= {float(vs_lo):.15f} > vP_hi: {vs_lo > vp_hi}  (the (N) route stops in ({fs(P1_LIST[-1])}, {fs(P_SHARP)}])")
    disp = []
    for pp in P_DISPLAY:
        D = CA.Design("d", B0, K, Q(pp))
        Wd = D.warm(vp[0])
        lo, hi = F_(Wd["v_bounds"][0]), F_(Wd["v_bounds"][1])
        disp.append(dict(p=fs(pp), v_lower=fs(down_dec(lo)), v_upper=fs(up_dec(hi))))
        print(f"    display: p = {fs(pp):>14s} ({float(pp):.4e}): v_H(gamma(p)) in [{float(lo):.12f}, {float(hi):.12f}]")
    incr = all(Fraction(disp[i]["v_upper"]) < Fraction(disp[i + 1]["v_lower"]) for i in range(len(disp) - 1))
    print(f"    display values strictly increasing in p (consistent with Lemma D): {incr}   [{time.time()-t0:.1f}s]")

    all_ok = (covP["ok"] and covB["ok"] and hot_ok and hot3_ok and lemmaH_ok and all(n["ok"] for n in nish)
              and thr["W_STAR"]["two_P_lt_1"] and thr["W_BB"]["fifteen_w2_lt_1"])
    cross_ok = covP["ball_ok"] and covB["ball_ok"] and hc["ok"] and hcu["ok"] and hc3["ok"]
    print(f"ALL CLAIMS CERTIFIED: {all_ok};  Lemma-L/Lemma-H-free cross-checks: {cross_ok};  total {time.time()-t_start:.1f}s")

    def cells_json(cov):
        return [dict(bl=fs(c["bl"]), br=fs(c["br"]), s=fs(c["s"]), F_up=fs(c["F_up"]), y=fs(c["y"]), W_up=fs(c["W_up"]),
                     ball_up=(fs(c["ball_up"]) if c["ball_up"] is not None else None), ball_subcells=c["ball_subcells"])
                for c in cov["cells"]]

    claims = {
        "schema": "lattice/window claims v1",
        "date": "2026-09-27",
        "status_labels": "certified = rigorous computation run by cert_window.py certify (Arb + exact rationals); "
                         "each downstream use also needs the cited lemma (Lemma L = the Lipschitz bound of Lemma C.12, "
                         "Lemma D = Lemma 9.10; Lemma H is not used by the paper)",
        "generator": dict(script="window/cert_window.py certify", sha256_script=sha_self,
                          imports="cert/cert_arb.py (unit_law, classes, mags_interval, ball, bounds, Design.warm/eval_interval/functionals, vp_bracket)",
                          sha256_cert_arb=sha_cert_arb, python_flint=flint.__version__, arb_prec_bits=CA.PREC),
        "design": dict(name="P", gadget="H4(b0,k) of Section 9.1: H4 = e_a . U^(1) ... U^(k) . e_b, U = e_d || (e_1 . (P_b0 || P_b0) . e_2)",
                       b0=B0, k=K, p0=fs(P0), edges_per_bond=K * (2 * B0 + 3) + 2, max_degree=4,
                       disorder="iid J_e in {+1,-1} on every microscopic edge, P(J=-1) = p",
                       gamma_p0=[float(g_lo), float(g_hi)]),
        "law_of_K_H": {
            "reference": "Proposition 9.3 (magnitudes y_c = m_c, class probabilities alpha_c = P_c, sign rule sgn K_U = s_d; "
                         "kappa = chi, kappa_2 = chi_2, 1-2p = vartheta there); Lemma C.3 (backbone couplings iid with this law at every beta)",
            "theta": "tanh(beta)", "kappa": "atanh(theta^b0)", "tau": "theta^2 tanh(2 kappa) = 2 theta^(b0+2)/(1+theta^(2 b0))",
            "unit_magnitudes": "m_1 = theta (paths disagree), m_2 = (theta+tau)/(1+theta tau) = tanh(beta+kappa_2) (non-cancel), "
                               "m_3 = (theta-tau)/(1-theta tau) = tanh(beta-kappa_2) (cancel), kappa_2 = atanh(tau)",
            "class_probabilities": "q = (1-(1-2p)^b0)/2, a = (1+(1-2p)^3)/2, P_1 = 2q(1-q), P_3 = (1-q)^2 (1-a) + q^2 a, P_2 = 1-P_1-P_3",
            "sign_means": "c+ = (1-p)^2+p^2, c- = 2p(1-p); rho_1 = 1-2p; "
                          "rho_2 = [(1-q)^2((1-p)c+ - p c-) + q^2((1-p)c- - p c+)]/P_2; "
                          "rho_3 = [(1-q)^2((1-p)c- - p c+) + q^2((1-p)c+ - p c-)]/P_3  (rho_c = E[s_d | class c])",
            "gadget_law": "for n = (n_1,n_2,n_3), n_1+n_2+n_3 = k: weight M(n) P_1^n_1 P_2^n_2 P_3^n_3 with M the multinomial; "
                          "|tanh K_H| = x_n = theta^2 m_1^n_1 m_2^n_2 m_3^n_3; P(K_H > 0 | n) = (1+A_n)/2, A_n = (1-2p)^2 rho_1^n_1 rho_2^n_2 rho_3^n_3 "
                          "(132 signed atoms for k = 10)",
            "labeling_note": "cert/cert_arb.py uses (class 1, class 2, class 3) = (non-cancel, disagree, cancel) = "
                             "classes (2, 1, 3) of Proposition 9.3; checked exactly by certify",
        },
        "functionals": {
            "F(beta,s)": "E exp(-2 s K_H(beta)) = sum_n M(n) P^n [ (1+A_n)/2 r_n^s + (1-A_n)/2 r_n^(-s) ],  r_n = (1-x_n)/(1+x_n)",
            "w(beta)": "min over s in [0,1] of F(beta,s)",
            "u_H(beta)": "E tanh|K_H(beta)| = sum_n M(n) P^n x_n",
            "pbar_H(beta)": "E[1 - exp(-2|K_H(beta)|)] = sum_n M(n) P^n 2 x_n/(1+x_n)",
            "v_H(gamma(p))": "E exp(-K_H(gamma(p))) = F(gamma(p), 1/2), gamma(p) = (1/2) log((1-p)/p)",
            "P(v)": "9 v^4/(1-9v^2)^2 (Section 9.3, Lemma C.7)", "h(v)": "4P(v) + 4Q(v), Q(v) = 9v^3/(1-9v^2)",
        },
        "cell_rule": "for beta in [bl, br]: E exp(-2 s K_H(beta)) <= F(bl,s) exp(6 s (beta-bl)) (Lemma L = Lemma C.12, samplewise) "
                     "<= F_up (2+y)/(2-y) <= W_up <= threshold, where y = 6 s (br-bl) and e^y <= (2+y)/(2-y) for 0 <= y < 2. "
                     "Checker obligations per cell: F(bl, s) <= F_up (high-precision evaluation at a rational point); the rest is exact rational arithmetic.",
        "thresholds": thr,
        "W_plus": dict(claim="E exp(-2 s_j K_H(beta)) <= W_STAR for every beta in [beta_1, beta_2] (cell j containing beta), and 2P(W_STAR) < 1",
                       status="certified" if covP["ok"] else "FAILED", beta_1=fs(BETA1_PLUS), beta_2=fs(BETA2_PLUS),
                       beta_1_over_gamma=float(BETA1_PLUS) / gf, beta_2_over_gamma=float(BETA2_PLUS) / gf,
                       threshold=fs(W_STAR), n_cells=len(covP["cells"]), contiguous=covP["contiguous"],
                       sup_W_up=fs(covP["sup_W_up"]), lemma_L_free_ball_check=covP["ball_ok"], bands=bandsP,
                       cells=cells_json(covP)),
        "W_two_point": dict(claim="E exp(-2 s_j K_H(beta)) <= W_BB for every beta in [beta_1, beta_2], and 15 W_BB^2 < 1",
                            status="certified" if covB["ok"] else "FAILED", beta_1=fs(BETA1_BB), beta_2=fs(BETA2_BB),
                            beta_1_over_gamma=float(BETA1_BB) / gf, beta_2_over_gamma=float(BETA2_BB) / gf,
                            threshold=fs(W_BB), n_cells=len(covB["cells"]), contiguous=covB["contiguous"],
                            sup_W_up=fs(covB["sup_W_up"]), lemma_L_free_ball_check=covB["ball_ok"], bands=bandsB,
                            cells=cells_json(covB)),
        "sharpness_W": dict(status="certified (route-specific point values; not needed downstream)",
                            meaning="w(beta) >= w_lower at the listed beta, so the same threshold test fails there",
                            points=sharp_w),
        "hot": dict(
            status="certified" if (hot_ok and hot3_ok and lemmaH_ok) else "FAILED",
            claim="pbar_H(beta) < 1/2 and u_H(beta) < 1/3 for all beta in (0, beta_hot]; pbar_H(beta) < 1/3 for all beta in (0, beta_hot3]",
            beta_hot=fs(BETA_HOT), beta_hot_over_gamma=float(BETA_HOT) / gf,
            pbar_upper_at_beta_hot=fs(up_dec(hot["pbar"][1])), u_upper_at_beta_hot=fs(up_dec(hot["u"][1])),
            beta_hot3=fs(BETA_HOT3), beta_hot3_over_gamma=float(BETA_HOT3) / gf, pbar_upper_at_beta_hot3=fs(up_dec(hot3["pbar"][1])),
            monotonicity=dict(lemma="Lemma H (not used by the paper): if 2(b0+2) th^(b0+1) <= 1 - 4 th^(2b0+4) at th = theta_bar >= tanh(B), "
                                    "then m_1, m_2, m_3 hence u_H and pbar_H are nondecreasing on (0, B]",
                              B=fs(BETA_MONO), theta_bar=fs(th_bar), lhs_float=float(lhs), rhs_float=float(rhs), ok=lemmaH_ok),
            lemma_H_free_cover=dict(pbar_half=hc, u_third=hcu, pbar_third=hc3),
            sharpness=dict(beta=fs(BETA_HOT + Fraction(1, 10 ** 5)), pbar_lower=fs(down_dec(hot_s["pbar"][0])), u_lower=fs(down_dec(hot_s["u"][0])),
                           beta3=fs(BETA_HOT3 + Fraction(1, 10 ** 5)), pbar3_lower=fs(down_dec(hot3_s["pbar"][0])))),
        "nishimori_line": dict(
            status="certified" if all(n["ok"] for n in nish) else "FAILED",
            claim="v_H(gamma(p1)) <= V < vP_lo, h(V) < 1, for each p1; with Lemma D (Lemma 9.10): v_H(gamma(p)) < v_P for every p in (0, p1]",
            vP_bracket=[fs(vp_lo), fs(vp_hi)], p1=nish, sharpness=sharp_n, display=disp, display_increasing=incr),
        "all_ok": all_ok, "cross_checks_ok": cross_ok,
    }
    json.dump(claims, open(out_json, "w"), indent=1)
    print(f"wrote {out_json}")


# ============================================================================== verify (mpmath)
def verify(claims_path, stride=1):
    from mpmath import mp, mpf
    mp.dps = 60
    C = json.load(open(claims_path))
    F = Fraction
    b0, k = C["design"]["b0"], C["design"]["k"]
    t0 = time.time()

    def mpq(x):
        if isinstance(x, type(mpf(0))):
            return x
        x = F(x)
        return mpf(x.numerator) / x.denominator

    def gamma_mp(p):
        return mp.log((1 - mpq(p)) / mpq(p)) / 2

    def law_atoms(beta, p):
        """Signed enumeration: 32 unit disorders -> signed unit atoms -> k-fold chain + two terminal edges."""
        beta, p = mpq(beta), mpq(p)
        th = mp.tanh(beta)
        kap = mp.atanh(th ** b0)
        q = (1 - (1 - 2 * p) ** b0) / 2
        pj = {1: 1 - p, -1: p}
        pc = {1: 1 - q, -1: q}
        unit = {}
        for sd, s1, s2, c1, c2 in product((1, -1), repeat=5):
            pr = pj[sd] * pj[s1] * pj[s2] * pc[c1] * pc[c2]
            KU = beta * sd + s1 * s2 * mp.atanh(th * th * mp.tanh((c1 + c2) * kap))
            sg = 1 if KU > 0 else -1
            lm = mp.log(abs(mp.tanh(KU)))
            key = (sg, mp.nstr(lm, 45))
            if key in unit:
                unit[key][1] += pr
            else:
                unit[key] = [lm, pr]
        mags = sorted({key[1] for key in unit})
        idx = {m: i for i, m in enumerate(mags)}
        lmv = [None] * len(mags)
        for (sg, m), (lm, pr) in unit.items():
            lmv[idx[m]] = lm
        cur = {(1, (0,) * len(mags)): mpf(1)}
        for _ in range(k):
            nxt = {}
            for (sg, cnt), w in cur.items():
                for (s, m), (lm, pr) in unit.items():
                    c = list(cnt)
                    c[idx[m]] += 1
                    key = (sg * s, tuple(c))
                    nxt[key] = nxt.get(key, 0) + w * pr
            cur = nxt
        pt = pj[1] ** 2 + pj[-1] ** 2                   # P(J_a J_b = +1)
        lth2 = 2 * mp.log(th)
        atoms = []
        for (sg, cnt), w in cur.items():
            x = mp.exp(lth2 + sum(ci * lmv[i] for i, ci in enumerate(cnt)))
            atoms.append((w * pt, sg, x))
            atoms.append((w * (1 - pt), -sg, x))
        return atoms

    def Fmp(atoms, s):
        s = mpq(s)
        return sum(w * ((1 - x) / (1 + x)) ** (sg * s) for w, sg, x in atoms)

    def pbar_u(atoms):
        return sum(w * 2 * x / (1 + x) for w, sg, x in atoms), sum(w * x for w, sg, x in atoms)

    def wmin(atoms):
        a, b = mpf(0), mpf(1)
        g = (mp.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = Fmp(atoms, c), Fmp(atoms, d)
        for _ in range(70):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = Fmp(atoms, c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = Fmp(atoms, d)
        return min(Fmp(atoms, (a + b) / 2), Fmp(atoms, 1))

    def P_(v):
        return 9 * v ** 4 / (1 - 9 * v * v) ** 2

    ok_all = True
    print(f"verify {claims_path} (mpmath {mp.dps} digits; signed unit enumeration; exact Fraction checks); stride {stride}")
    # thresholds
    ws, wb = F(C["thresholds"]["W_STAR"]["value"]), F(C["thresholds"]["W_BB"]["value"])
    t1 = 9 * ws ** 2 < 1 and 18 * ws ** 4 < (1 - 9 * ws ** 2) ** 2
    t2 = 15 * wb ** 2 < 1
    print(f"  thresholds: 2P({fs(ws)}) < 1: {t1};  15 ({fs(wb)})^2 < 1: {t2}")
    ok_all &= t1 and t2
    # regression at gamma(p0): F(gamma, 1/2) = v_H(gamma) = 0.205295045989685 (cert/out/cert_arb.txt)
    p0 = F(C["design"]["p0"])
    vg = Fmp(law_atoms(gamma_mp(p0), p0), F(1, 2))
    reg = abs(vg - mpf("0.205295045989685")) < mpf("1e-14")
    print(f"  regression: mp E exp(-K_H(gamma(p0))) = {mp.nstr(vg, 18)} (cert_arb 0.205295045989685): {reg}")
    ok_all &= bool(reg)
    # covers
    for key, thr in (("W_plus", ws), ("W_two_point", wb)):
        cov = C[key]
        cells = cov["cells"]
        cont = F(cells[0]["bl"]) == F(cov["beta_1"]) and F(cells[-1]["br"]) == F(cov["beta_2"]) and all(
            F(cells[i]["br"]) == F(cells[i + 1]["bl"]) for i in range(len(cells) - 1))
        exact_ok = True
        for c in cells:
            bl, br, s, Fup, y, Wup = (F(c[z]) for z in ("bl", "br", "s", "F_up", "y", "W_up"))
            exact_ok &= (0 < s <= 1 and bl < br and y == 6 * s * (br - bl) and 0 <= y < 2
                         and Fup * (2 + y) / (2 - y) <= Wup and Wup <= thr and F(cov["threshold"]) == thr)
        worst = None
        nchk = 0
        for c in cells[::stride]:
            atoms = law_atoms(c["bl"], p0)
            val = Fmp(atoms, c["s"])
            slack = (mpq(c["F_up"]) - val) / val
            nchk += 1
            if worst is None or slack < worst:
                worst = slack
        good = cont and exact_ok and worst > 0
        ok_all &= good
        print(f"  {key}: {len(cells)} cells, contiguous {cont}, exact rational chain F_up (2+y)/(2-y) <= W_up <= {fs(thr)}: {exact_ok};"
              f" mp F(bl,s) <= F_up on {nchk} cells, min relative slack {mp.nstr(worst, 5)}: {good}")
    # sharpness points
    for pt_ in C["sharpness_W"]["points"]:
        atoms = law_atoms(pt_["beta"], p0)
        wm = wmin(atoms)
        good = wm >= mpq(pt_["w_lower"])
        wl = F(pt_["w_lower"])
        fails = (9 * wl ** 2 < 1 and 18 * wl ** 4 >= (1 - 9 * wl ** 2) ** 2) if pt_["kind"] == "plus" else (15 * wl ** 2 >= 1)
        print(f"  sharpness beta={pt_['beta']}: mp w = {mp.nstr(wm, 12)} >= w_lower {float(wl):.12f}: {good}; threshold test fails exactly: {fails}")
        ok_all &= bool(good and fails == pt_["route_fails"])
    # hot
    H = C["hot"]
    atoms = law_atoms(H["beta_hot"], p0)
    pb, u = pbar_u(atoms)
    g1 = pb <= mpq(H["pbar_upper_at_beta_hot"]) and F(H["pbar_upper_at_beta_hot"]) < F(1, 2)
    g2 = u <= mpq(H["u_upper_at_beta_hot"]) and F(H["u_upper_at_beta_hot"]) < F(1, 3)
    atoms3 = law_atoms(H["beta_hot3"], p0)
    pb3, _ = pbar_u(atoms3)
    g3 = pb3 <= mpq(H["pbar_upper_at_beta_hot3"]) and F(H["pbar_upper_at_beta_hot3"]) < F(1, 3)
    M = H["monotonicity"]
    thb, B = F(M["theta_bar"]), F(M["B"])
    g4 = mp.tanh(mpq(B)) <= mpq(thb) and thb < 1 and 2 * (b0 + 2) * thb ** (b0 + 1) <= 1 - 4 * thb ** (2 * b0 + 4) and F(H["beta_hot"]) <= B and F(H["beta_hot3"]) <= B
    print(f"  hot: mp pbar_H(beta_hot) = {mp.nstr(pb, 15)} <= stored upper < 1/2: {g1}; mp u_H = {mp.nstr(u, 15)} <= stored < 1/3: {g2};"
          f" mp pbar_H(beta_hot3) = {mp.nstr(pb3, 15)} < 1/3: {g3}; Lemma H exact condition and tanh(B) <= theta_bar: {g4}")
    ok_all &= bool(g1 and g2 and g3 and g4)
    Sx = H["sharpness"]
    pbs, us = pbar_u(law_atoms(Sx["beta"], p0))
    print(f"  hot sharpness: mp pbar_H({Sx['beta']}) = {mp.nstr(pbs, 12)} >= {float(F(Sx['pbar_lower'])):.12f} > 1/2: {pbs >= mpq(Sx['pbar_lower']) and F(Sx['pbar_lower']) > F(1, 2)}")
    # Nishimori line
    N = C["nishimori_line"]
    vlo, vhi = (F(z) for z in N["vP_bracket"])
    hv = lambda v: 4 * P_(v) + 4 * 9 * v ** 3 / (1 - 9 * v * v)
    gvp = hv(vlo) < 1 < hv(vhi)
    print(f"  v_P bracket: h(lo) < 1 < h(hi) exactly: {gvp}")
    ok_all &= gvp
    for e in N["p1"]:
        pp = F(e["p1"])
        atoms = law_atoms(gamma_mp(pp), pp)
        v = Fmp(atoms, F(1, 2))
        V = F(e["V"])
        g = v <= mpq(e["v_upper"]) and F(e["v_upper"]) <= V and V < vlo and 3 * V < 1 and hv(V) < 1
        print(f"  N p1={e['p1']}: mp v_H(gamma(p1)) = {mp.nstr(v, 15)} <= V = {float(V):.12f}; V < vP_lo; h(V) < 1 exactly: {g}")
        ok_all &= bool(g)
    sp = N["sharpness"]
    v = Fmp(law_atoms(gamma_mp(F(sp["p"])), F(sp["p"])), F(1, 2))
    print(f"  N sharpness p={sp['p']}: mp v = {mp.nstr(v, 15)} >= {float(F(sp['v_lower'])):.15f} > vP_hi: {v >= mpq(sp['v_lower']) and F(sp['v_lower']) > vhi}")
    print(f"VERIFY ALL OK: {ok_all}   [{time.time()-t0:.1f}s]")


# ============================================================================== lipcheck (float evidence)
def lipcheck():
    import random
    import numpy as np
    from mpmath import mp, mpf
    mp.dps = 100          # 1 - tanh(beta)^2 ~ 4e^{-2 beta} must stay resolved up to beta = 40
    t0 = time.time()
    print("lipcheck: float evidence for Lemma L, Lemma H, Lemma D (not a certificate)")

    def kap2(beta, b0):
        th = mp.tanh(beta)
        return mp.atanh(th * th * mp.tanh(2 * mp.atanh(th ** b0)))

    def kap2_prime(beta, b0):
        th = mp.tanh(beta)
        tb = th ** b0
        tau = 2 * th ** (b0 + 2) / (1 + tb * tb)
        dtau = 2 * th ** (b0 + 1) * ((b0 + 2) + (2 - b0) * tb * tb) / (1 + tb * tb) ** 2
        return dtau * (1 - th * th) / (1 - tau * tau)

    # (1) unit level: |dK_U/dbeta| in {1, 1 + kappa_2', |1 - kappa_2'|} <= 1 + kappa_2' <= 3
    grid = [mpf("0.002") * mpf("1.004") ** i for i in range(0, 2600)]
    grid = [b for b in grid if b <= 40]
    print(f"(1) unit: kappa_2'(beta) on {len(grid)} log-spaced beta in [0.002, 40]")
    for b0 in (2, 3, 5, 10, 100, 1000, 1500):
        vals = [(kap2_prime(b, b0), b) for b in grid]
        mx, bmx = max(vals)
        # first beta where kappa_2' reaches 1 (m_3 = tanh(beta - kappa_2) stops increasing)
        first = next((b for v, b in vals if v >= 1), None)
        # formula check against numerical differentiation at a few points
        errs = [abs(mp.diff(lambda z: kap2(z, b0), b) - kap2_prime(b, b0)) for b in grid[::300]]
        print(f"    b0={b0:5d}: max kappa_2' = {mp.nstr(mx, 8)} at beta = {mp.nstr(bmx, 6)} => max |dK_U/dbeta| <= {mp.nstr(1 + mx, 8)} (Lemma L bound 3);"
              f" kappa_2' >= 1 first at beta ~ {mp.nstr(first, 6) if first is not None else 'never'}; formula vs mp.diff max err {mp.nstr(max(errs), 3)}")
    # Lemma H bound g(theta) >= kappa_2' on (0, 5/2] for b0 = 1000
    dominated = True
    mxk = mpf(0)
    mxg = mpf(0)
    for b in grid:
        if b > 2.5:
            break
        th = mp.tanh(b)
        gb = 2 * (1000 + 2) * th ** 1001 / (1 - 4 * th ** 2004)
        kp = kap2_prime(b, 1000)
        dominated &= bool(kp <= gb)
        mxk, mxg = max(mxk, kp), max(mxg, gb)
    print(f"    Lemma H (b0=1000, beta <= 5/2): kappa_2' <= 2(b0+2)th^(b0+1)/(1-4th^(2b0+4)) at every grid point: {dominated};"
          f" max kappa_2' = {mp.nstr(mxk, 5)}, max bound = {mp.nstr(mxg, 5)} (Lemma H needs <= 1)")

    # (2) chain level, design P: d|K_H|/dbeta per class via series sensitivities
    p = mpf(9) / 10000
    b0, k = 1000, 10
    print("(2) chain, design P: max over the 66 classes of |d|K_H|/dbeta| on a beta grid")
    mxc = (mpf(0), None, None)
    for b in [mpf("0.01") * mpf("1.01") ** i for i in range(0, 800) if mpf("0.01") * mpf("1.01") ** i < 40]:
        th = mp.tanh(b)
        k2 = kap2(b, b0)
        k2p = kap2_prime(b, b0)
        ms = [(th, mpf(1)), (mp.tanh(b + k2), 1 + k2p), (mp.tanh(b - k2), 1 - k2p)]   # (m_c, dk_c/dbeta) in the class labels of Proposition 9.3
        for n1 in range(k + 1):
            for n3 in range(k + 1 - n1):
                n2 = k - n1 - n3
                T = th ** 2 * ms[0][0] ** n1 * ms[1][0] ** n2 * ms[2][0] ** n3
                # sensitivity of atanh(prod t_i) to k_i is (1 - t_i^2) (T / t_i) / (1 - T^2)
                sens = lambda t: (1 - t * t) * (T / t) / (1 - T * T)
                d = 2 * sens(th) * 1 + n1 * sens(ms[0][0]) * ms[0][1] + n2 * sens(ms[1][0]) * ms[1][1] + n3 * sens(ms[2][0]) * ms[2][1]
                if abs(d) > mxc[0]:
                    mxc = (abs(d), b, (n1, n2, n3))
    print(f"    max |d|K_H|/dbeta| = {mp.nstr(mxc[0], 8)} at beta = {mp.nstr(mxc[1], 6)}, class (n1,n2,n3) = {mxc[2]} (class labels of Prop. 9.3); bound 3")

    # (3) brute force on small H4(b0,k): exhaustive interior sums
    def h4_graph(b0_, k_):
        edges = []
        nid = [0]

        def new():
            nid[0] += 1
            return nid[0] + 1          # 0, 1 are the poles
        z = [new() for _ in range(k_ + 1)]
        edges.append((0, z[0]))
        for i in range(k_):
            a_, b_ = z[i], z[i + 1]
            edges.append((a_, b_))                  # e_d
            w1, w2 = new(), new()
            edges.append((a_, w1))                  # e_1
            for _ in range(2):
                prev = w1
                for j in range(b0_ - 1):
                    v = new()
                    edges.append((prev, v))
                    prev = v
                edges.append((prev, w2))
            edges.append((w2, b_))                  # e_2
        edges.append((z[k_], 1))
        return edges, nid[0] + 2

    def energies(edges, nv, Js):
        """Energy sum_e J_e s_u s_v over all interior states, for pole configurations (+,+) and (+,-)."""
        I = nv - 2
        S = ((np.arange(2 ** I)[:, None] >> np.arange(I)[None, :]) & 1) * 2 - 1
        out = []
        for s1 in (1, -1):
            spins = np.concatenate([np.ones((2 ** I, 1)), s1 * np.ones((2 ** I, 1)), S], axis=1)
            E = np.zeros(2 ** I)
            for (u, v), J in zip(edges, Js):
                E += J * spins[:, u] * spins[:, v]
            out.append(E)
        return out

    def KH_brute(Epm, betas):
        """K_H(beta) = (1/2) log(Z(+,+)/Z(+,-)) for an array of betas (log-sum-exp)."""
        betas = np.atleast_1d(np.asarray(betas, dtype=float))
        res = []
        for E in Epm:
            X = betas[:, None] * E[None, :]
            m = X.max(axis=1)
            res.append(m + np.log(np.exp(X - m[:, None]).sum(axis=1)))
        return 0.5 * (res[0] - res[1])

    def unit_graph(b0_):
        """Bare unit U = e_d || (e_1 . (P_b0 || P_b0) . e_2) between poles 0 and 1."""
        edges = [(0, 1), (0, 2)]
        nxt = 4
        for _ in range(2):
            prev = 2
            for j in range(b0_ - 1):
                edges.append((prev, nxt))
                prev = nxt
                nxt += 1
            edges.append((prev, 3))
        edges.append((3, 1))
        return edges, nxt

    rng = random.Random(7)
    print("(3a) brute force on the bare unit U (no terminal edges): max |dK_U/dbeta| vs closed form 1 + max kappa_2'")
    fine = np.linspace(0.02, 8.0, 800)
    for b0_ in (2, 3, 4, 5, 6):
        edges, nv = unit_graph(b0_)
        E = len(edges)
        worst, where = 0.0, None
        if b0_ <= 3:
            Jlist = [[1 - 2 * ((c >> i) & 1) for i in range(E)] for c in range(2 ** E)]
        else:
            Jlist = [[1] * E] + [[rng.choice((1, -1)) for _ in range(E)] for _ in range(150)]
        for Js in Jlist:
            Epm = energies(edges, nv, Js)
            d = np.abs(KH_brute(Epm, fine + 1e-5) - KH_brute(Epm, fine - 1e-5)) / 2e-5
            i = int(d.argmax())
            if d[i] > worst:
                worst, where = float(d[i]), float(fine[i])
        cf = max(1 + kap2_prime(mpf(b), b0_) for b in fine)
        print(f"    U(b0={b0_}): |E| = {E}, {len(Jlist)} disorders{' (all)' if b0_ <= 3 else ' (all-plus + 150 random)'} x 800 betas in [0.02, 8]: brute max |dK_U/dbeta| = {worst:.6f} at beta = {where:.3f};"
              f" closed-form max 1 + kappa_2' = {mp.nstr(cf, 7)} (bound 3)")
    print("(3) brute force: K_H(beta) = (1/2) log(Z(+,+)/Z(+,-)) by exhaustive interior sums; |dK_H/dbeta| by central differences (h=1e-5)")
    betas = [0.05 * i for i in range(1, 121)]
    for b0_, k_, nJ in ((2, 1, 512), (3, 1, 400), (2, 2, 150), (4, 1, 150)):
        edges, nv = h4_graph(b0_, k_)
        E = len(edges)
        if nJ >= 2 ** E:
            Jlist = [[1 - 2 * ((c >> i) & 1) for i in range(E)] for c in range(2 ** E)]
        else:
            Jlist = [[rng.choice((1, -1)) for _ in range(E)] for _ in range(nJ)]
        worst = 0.0
        where = None
        bb = np.array(betas)
        for Js in Jlist:
            Epm = energies(edges, nv, Js)
            d = np.abs(KH_brute(Epm, bb + 1e-5) - KH_brute(Epm, bb - 1e-5)) / 2e-5
            i = int(d.argmax())
            if d[i] > worst:
                worst, where = float(d[i]), (betas[i], tuple(Js))
        print(f"    H4({b0_},{k_}): |E| = {E}, I = {nv-2}, {len(Jlist)} disorders x {len(betas)} betas in [0.05, 6]: max |dK_H/dbeta| = {worst:.6f} at beta = {where[0]:.2f} (bound 3)")

    # (4) random SP gadgets with real edge weights: |dK/dbeta| <= lambda(H)
    print("(4) random series-parallel gadgets, K_e = beta J_e with J_e in [-1,1]: max |dK/dbeta| / lambda(H)")

    def rand_tree(n):
        if n == 1:
            J = rng.choice([1, -1, rng.uniform(-1, 1)])
            return ("e", J)
        m = rng.randint(1, n - 1)
        return (rng.choice("SP"), rand_tree(m), rand_tree(n - m))

    def lam(t):
        if t[0] == "e":
            return abs(t[1])
        return max(lam(t[1]), lam(t[2])) if t[0] == "S" else lam(t[1]) + lam(t[2])

    def KdK(t, b):
        if t[0] == "e":
            return b * t[1], mpf(t[1])
        K1, d1 = KdK(t[1], b)
        K2, d2 = KdK(t[2], b)
        if t[0] == "P":
            return K1 + K2, d1 + d2
        t1, t2 = mp.tanh(K1), mp.tanh(K2)
        T = t1 * t2
        return mp.atanh(T), ((1 - t1 * t1) * t2 * d1 + (1 - t2 * t2) * t1 * d2) / (1 - T * T)

    worst = mpf(0)
    ntree = 0
    bgrid = [mpf("0.02") * mpf("1.04") ** i for i in range(0, 150)]
    for _ in range(300):
        t = rand_tree(rng.randint(2, 14))
        L = lam(t)
        if L == 0:
            continue
        ntree += 1
        for b in bgrid:
            K_, d = KdK(t, b)
            worst = max(worst, abs(d) / L)
    # adversarial: parallel bundles in series (parallel sums near saturation)
    for m in range(2, 7):
        for n in range(1, 5):
            t = ("e", 1)
            bundle = ("e", 1)
            for _ in range(m - 1):
                bundle = ("P", bundle, ("e", 1))
            t = bundle
            for _ in range(n - 1):
                t = ("S", t, bundle)
            L = lam(t)
            for b in bgrid:
                K_, d = KdK(t, b)
                worst = max(worst, abs(d) / L)
    # finite-difference check of the recursive derivative on a few trees
    fd_err = mpf(0)
    for _ in range(30):
        t = rand_tree(rng.randint(2, 10))
        for b in (mpf("0.3"), mpf("1.1"), mpf("2.7")):
            fd_err = max(fd_err, abs(mp.diff(lambda z: KdK(t, z)[0], b) - KdK(t, b)[1]))
    print(f"    {ntree} random trees (2-14 edges) + 20 parallel-bundle chains, {len(bgrid)} betas in [0.02, {mp.nstr(bgrid[-1], 3)}]: max |dK/dbeta|/lambda = {mp.nstr(worst, 10)} (must be <= 1); recursive derivative vs mp.diff max err {mp.nstr(fd_err, 3)}")
    # series sensitivity identity for two factors
    ms = 0.0
    for _ in range(20000):
        a_, b_ = rng.random(), rng.random()
        ms = max(ms, ((1 - a_ * a_) * b_ + (1 - b_ * b_) * a_) / (1 - a_ * a_ * b_ * b_) - (a_ + b_) / (1 + a_ * b_))
    print(f"    identity sum of two series sensitivities = (a+b)/(1+ab): max deviation {ms:.2e}")

    # (5) log-derivative of F(beta, s) for design P
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("(5) |d/dbeta log E exp(-2 s K_H(beta))| / (6 s) on beta in [1.9, 5.0], s in {0.25, 0.5, 0.75, 1} (must be <= 1)")
    q = (1 - (1 - 2 * p) ** b0) / 2
    cp, cm = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)
    tpos, tzero, tneg = cp * (1 - q) ** 2 + cm * q ** 2, 2 * q * (1 - q), cp * q ** 2 + cm * (1 - q) ** 2
    Pc = [(1 - p) * tpos + p * tneg, tzero, (1 - p) * tneg + p * tpos]          # cert_arb labels
    rho = [((1 - p) * tpos - p * tneg) / Pc[0], 1 - 2 * p, ((1 - p) * tneg - p * tpos) / Pc[2]]
    worst = mpf(0)
    for i in range(0, 63):
        b = mpf("1.9") + mpf("0.05") * i
        th = mp.tanh(b)
        k2 = kap2(b, b0)
        k2p = kap2_prime(b, b0)
        mm = [(mp.tanh(b + k2), 1 + k2p), (th, mpf(1)), (mp.tanh(b - k2), 1 - k2p)]
        for s in (mpf("0.25"), mpf("0.5"), mpf("0.75"), mpf(1)):
            Fv = mpf(0)
            dF = mpf(0)
            for n1 in range(k + 1):
                for n3 in range(k + 1 - n1):
                    n2 = k - n1 - n3
                    W = comb(k, n1) * comb(k - n1, n3) * Pc[0] ** n1 * Pc[1] ** n2 * Pc[2] ** n3
                    A = (1 - 2 * p) ** 2 * rho[0] ** n1 * rho[1] ** n2 * rho[2] ** n3
                    T = th ** 2 * mm[0][0] ** n1 * mm[1][0] ** n2 * mm[2][0] ** n3
                    sens = lambda t: (1 - t * t) * (T / t) / (1 - T * T)
                    kd = 2 * sens(th) + n1 * sens(mm[0][0]) * mm[0][1] + n2 * sens(mm[1][0]) * mm[1][1] + n3 * sens(mm[2][0]) * mm[2][1]
                    kk = mp.atanh(T)
                    Fv += W * ((1 + A) / 2 * mp.exp(-2 * s * kk) + (1 - A) / 2 * mp.exp(2 * s * kk))
                    dF += W * ((1 + A) / 2 * (-2 * s * kd) * mp.exp(-2 * s * kk) + (1 - A) / 2 * (2 * s * kd) * mp.exp(2 * s * kk))
            worst = max(worst, abs(dF / Fv) / (6 * s))
    print(f"    max ratio = {mp.nstr(worst, 6)}")

    # (6) Lemma H: u_H, pbar_H nondecreasing on (0, 5/2] (float grid)
    print("(6) hot side, design P: u_H and pbar_H on 1200 betas in (0, 5/2]")
    prev = (mpf(-1), mpf(-1))
    mono = True
    for i in range(1, 1201):
        b = mpf("2.5") * i / 1200
        th = mp.tanh(b)
        k2 = kap2(b, b0)
        mm = [mp.tanh(b + k2), th, mp.tanh(b - k2)]
        u = th ** 2 * (Pc[0] * mm[0] + Pc[1] * mm[1] + Pc[2] * mm[2]) ** k
        pb = mpf(0)
        for n1 in range(k + 1):
            for n3 in range(k + 1 - n1):
                n2 = k - n1 - n3
                W = comb(k, n1) * comb(k - n1, n3) * Pc[0] ** n1 * Pc[1] ** n2 * Pc[2] ** n3
                x = th ** 2 * mm[0] ** n1 * mm[1] ** n2 * mm[2] ** n3
                pb += W * 2 * x / (1 + x)
        if u < prev[0] or pb < prev[1]:
            mono = False
        prev = (u, pb)
    print(f"    nondecreasing on the grid: {mono}; values at 5/2: u = {mp.nstr(prev[0], 8)}, pbar = {mp.nstr(prev[1], 8)}")

    # (7) Lemma D: planted channel on H4(2,1), Bhattacharyya = E e^{-K_H(gamma)}, monotone in p
    print("(7) Lemma D on H4(2,1): Z(W_p) = sum_J sqrt(W(J|+) W(J|-)) vs E_iid exp(-K_H(gamma(p) J)); monotone in p")
    edges, nv = h4_graph(2, 1)
    E = len(edges)
    Jall = [[1 - 2 * ((c >> i) & 1) for i in range(E)] for c in range(2 ** E)]
    taus = [[1 - 2 * ((c >> i) & 1) for i in range(nv)] for c in range(2 ** nv)]
    rows = []
    for pp in (0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4):
        g = 0.5 * math.log((1 - pp) / pp)
        Zb = 0.0
        Ev = 0.0
        for Js in Jall:
            Wp = {1: 0.0, -1: 0.0}
            for tau in taus:
                x = tau[0] * tau[1]
                pr = 1.0
                for (u, v), J in zip(edges, Js):
                    pr *= (1 - pp) if J * tau[u] * tau[v] == 1 else pp
                Wp[x] += pr / 2 ** (nv - 1)
            Zb += math.sqrt(Wp[1] * Wp[-1])
            Piid = math.prod((1 - pp) if J == 1 else pp for J in Js)
            Ev += Piid * math.exp(-float(KH_brute(energies(edges, nv, Js), g)[0]))
        rows.append((pp, Zb, Ev))
    incr = all(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1))
    for pp, Zb, Ev in rows:
        print(f"    p={pp:.2f}: Z(W_p) = {Zb:.12f}, E exp(-K_H) = {Ev:.12f}, diff {abs(Zb-Ev):.1e}")
    print(f"    Z(W_p) increasing in p: {incr}")
    print(f"lipcheck done [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "certify":
        certify(sys.argv[2])
    elif mode == "verify":
        verify(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif mode == "lipcheck":
        lipcheck()
    else:
        sys.exit("usage: cert_window.py certify|verify CLAIMS_JSON [stride] | lipcheck")
