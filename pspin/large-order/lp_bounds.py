"""Rigorous numerical inputs of Theorem 7.15 (thm:nm-LP; "Theorem LP", "LP(a)"-"LP(e)" below) and of the
second certificate for 7 <= p <= 25 ("Theorem LP'", Appendix B.5, app:nm-cert); proofs in Appendix B.4
(app:nm-LP).

Arithmetic: mpmath.iv (directed rounding) at 40 digits.  mpmath.mp.dps is set to 60 >= iv.dps + 20,
so every conversion of an interval endpoint to mpf is exact.  Comparisons with
decimal constants are directed: a lower bound x >= c is accepted only if the lower endpoint of x is
>= the upper endpoint of the interval enclosing the decimal c (and symmetrically for upper bounds).
Displayed bounds are rounded in the safe direction (lower bounds down, upper bounds up).

Integrals are bounded by monotone cell sums (each integrand factor is monotone on each cell) plus
explicit tails; no quadrature routine is used for the local enclosures.

Part 1   c0 = int_R J, m2 = int h^2 J, m4 = int h^4 J, K = int h^2 sech^3 h: local cell enclosures on
         [0, 12] with cells of width 1/1024; tail J(h) <= (2h^2 + 2h + 1) e^{-3h}, h^2 sech^3 h <= 8h^2 e^{-3h}.
Part 1b  the Arb enclosures of c0, m2, m4 and K computed by arb-constants/lpc_arb.py (python-flint 0.9.0,
         acb.integral with rigorous error bounds; arb-constants/lpc_arb.json, "values"), read as cited certified inputs.  Arb's decimal
         output contains the computed ball, so the parsed interval is an enclosure.  Consistency with the
         local cells and with the closed forms is checked.
Part 2   the tail Lambda >= 35, used for every p >= 26 (all bounds are monotone in Lambda there;
         Appendix B.4).  Decision checks P2a-P2k and the assertions S1-S21 of every constant
         the paper states.
Part 3   Q(Lambda) < 26 on [34.6, 35] (P3), and the arithmetic of the analytic bound on (0, 0.4] (Q1).
Part 4   p = 7..25 one at a time: a point Lambda_p^- with Q(Lambda_p^-) < p, and the Part-2 bounds there.
Output: JSON on stdout; a count line on stderr; exit 1 if any check fails.
"""

import json
import os
import re
import sys

import mpmath
from mpmath import iv, mpf

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

iv.dps = 40
mpmath.mp.dps = 60
PI = iv.pi
LN2 = iv.log(2)
HERE = os.path.dirname(os.path.abspath(__file__))
out = {"decision checks": {}, "stated-constant assertions": {}, "consistency checks": {}, "remarks (analytic, not checks)": {}}
fails = []


def lo(x):
    return mpf(x.a)


def hi(x):
    return mpf(x.b)


def dec(c):
    return iv.mpf(str(c))


def ge(x, c):
    return lo(x) >= hi(dec(c))


def le(x, c):
    return hi(x) <= lo(dec(c))


def s(x, n=12):
    return f"[{mpmath.nstr(lo(x), n)}, {mpmath.nstr(hi(x), n)}]"


def down(x, nd):
    return mpmath.nstr(mpmath.floor(lo(x) * 10**nd) / 10**nd, nd + 3, strip_zeros=False)


def up(x, nd):
    return mpmath.nstr(mpmath.ceil(hi(x) * 10**nd) / 10**nd, nd + 3, strip_zeros=False)


def up_sig(x, sig):
    v = hi(x)
    e = int(mpmath.floor(mpmath.log10(v))) - sig + 1
    return mpmath.nstr(mpmath.ceil(v / mpf(10) ** e) * mpf(10) ** e, sig)


def check(group, name, cond, detail):
    out[group][name] = {"ok": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)


def cosh(x):
    return (iv.exp(x) + iv.exp(-x)) / 2


def sech(x):
    return 1 / cosh(x)


def r_(x):  # log(1 + e^{-2x})
    return iv.log1p(iv.exp(-2 * x))


def J_lo(a, b):  # lower bound of J on [a, b], 0 <= a < b (each factor monotone)
    return a * a * iv.exp(-2 * b) * sech(b) + 2 * a * iv.exp(-b) * r_(b) + cosh(a) * r_(b) ** 2


def J_hi(a, b):
    return b * b * iv.exp(-2 * a) * sech(a) + 2 * b * iv.exp(-a) * r_(a) + cosh(b) * r_(a) ** 2


def tail_poly_exp3(k, H):
    """int_H^oo h^k (2h^2 + 2h + 1) e^{-3h} dh, exactly (sum of incomplete gammas)."""
    def m(n):  # int_H^oo h^n e^{-3h} dh = e^{-3H} sum_{j=0}^n n!/j! H^j / 3^{n-j+1}
        tot = iv.mpf(0)
        fact_n = 1
        for t in range(2, n + 1):
            fact_n *= t
        fj = 1
        for j in range(0, n + 1):
            if j > 0:
                fj *= j
            tot += iv.mpf(fact_n) / fj * iv.mpf(H) ** j / iv.mpf(3) ** (n - j + 1)
        return iv.exp(-3 * iv.mpf(H)) * tot
    return 2 * m(k + 2) + 2 * m(k + 1) + m(k)


# ---------------- Part 1: local cell enclosures ----------------
N_CELL, H_MAX = 1024, 12
mom_lo = [iv.mpf(0)] * 5
mom_hi = [iv.mpf(0)] * 5
K_hi = iv.mpf(0)
for j in range(N_CELL * H_MAX):
    a = iv.mpf(j) / N_CELL
    b = iv.mpf(j + 1) / N_CELL
    jl, jh = J_lo(a, b), J_hi(a, b)
    for kk in (0, 2, 4):
        mom_lo[kk] += (b - a) * a**kk * jl
        mom_hi[kk] += (b - a) * b**kk * jh
    K_hi += (b - a) * b * b * sech(a) ** 3
for kk in (0, 2, 4):
    mom_lo[kk] = 2 * mom_lo[kk]
    mom_hi[kk] = 2 * (mom_hi[kk] + tail_poly_exp3(kk, H_MAX))
K_hi = 2 * (K_hi + 8 * (iv.exp(-3 * iv.mpf(H_MAX)) * (iv.mpf(H_MAX) ** 2 / 3 + 2 * iv.mpf(H_MAX) / 9 + iv.mpf(2) / 27)))
C0_cells = iv.mpf([lo(mom_lo[0]), hi(mom_hi[0])])
M2_cells = iv.mpf([lo(mom_lo[2]), hi(mom_hi[2])])
M4_cells = iv.mpf([lo(mom_lo[4]), hi(mom_hi[4])])
c0_closed = 4 * PI * LN2 - PI**3 / 4
K_closed = PI**3 / 8 - PI
out["c0 cell enclosure"] = s(C0_cells, 10)
out["m2 cell enclosure"] = s(M2_cells, 10)
out["m4 cell enclosure"] = s(M4_cells, 10)
out["K cell upper bound"] = mpmath.nstr(hi(K_hi), 12)
out["c0 closed form 4 pi log2 - pi^3/4 (interval)"] = s(c0_closed, 20)
check("consistency checks", "P1a closed form of c0 lies in the cell enclosure", lo(C0_cells) <= lo(c0_closed) and hi(c0_closed) <= hi(C0_cells), "consistency")
check("consistency checks", "P1c K closed form pi^3/8 - pi <= cell upper bound", hi(K_closed) <= hi(K_hi), "consistency")

# ---------------- Part 1b: Arb enclosures of arb-constants/lpc_arb.py (cited certified inputs) ----------------
arb_vals = json.load(open(os.path.join(HERE, "arb-constants", "lpc_arb.json")))["values"]


def parse_arb(txt):
    m_ = re.fullmatch(r"\[([-0-9.e+]+) \+/- ([0-9.e+-]+)\]", txt.strip())
    mid, rad = dec(m_.group(1)), dec(m_.group(2))
    return iv.mpf([lo(mid - rad), hi(mid + rad)])


C0_arb = parse_arb(arb_vals["c0"])
M2_arb = parse_arb(arb_vals["m2"])
M4_arb = parse_arb(arb_vals["m4"])
K_arb = parse_arb(arb_vals["K"])
out["c0 Arb enclosure (arb-constants)"] = s(C0_arb, 22)
out["m2 Arb enclosure (arb-constants)"] = s(M2_arb, 22)
out["m4 Arb enclosure (arb-constants)"] = s(M4_arb, 22)
check("consistency checks", "A1c Arb c0 lies inside the local cell enclosure", lo(C0_cells) <= lo(C0_arb) and hi(C0_arb) <= hi(C0_cells), "two arithmetics agree")
check("consistency checks", "A2c Arb m2 lies inside the local cell enclosure", lo(M2_cells) <= lo(M2_arb) and hi(M2_arb) <= hi(M2_cells), "two arithmetics agree")
check("consistency checks", "A3c Arb m4 lies inside the local cell enclosure", lo(M4_cells) <= lo(M4_arb) and hi(M4_arb) <= hi(M4_cells), "two arithmetics agree")
check("consistency checks", "A4c closed form 4 pi log2 - pi^3/4 meets the Arb c0 ball", lo(c0_closed) <= hi(C0_arb) and lo(C0_arb) <= hi(c0_closed), "consistency")
check("consistency checks", "A5c Arb K <= local K upper bound, and meets pi^3/8 - pi", hi(K_arb) <= hi(K_hi) and lo(K_closed) <= hi(K_arb) and lo(K_arb) <= hi(K_closed), "consistency")

# ---------------- Part 2: Lambda >= 35 ----------------
L0 = iv.mpf(35)


def eps(L):
    return iv.exp(-L / 2) / iv.sqrt(2 * PI * L)


def T1lo(L):
    return 2 * LN2 - PI * eps(L) * (L + 2)


def philo(L):
    e = eps(L)
    return 2 * LN2 - PI * e * (2 * L + 2 + PI * e)


def Rminus(L):  # pi^2 eps + (pi L + pi^2 eps)^2 eps / phi_lo
    e = eps(L)
    return PI**2 * e + (PI * L + PI**2 * e) ** 2 * e / philo(L)


def OmBup(L):  # also bounds p u and Lambda u / T1
    return PI * L * eps(L) / T1lo(L)


def Rplus(L):  # pi L q/(1 - q), q = pi L eps/T1_lo
    q = OmBup(L)
    return PI * L * q / (1 - q)


def lamMlo(L):  # 1 - (pi/2) L^2 eps/T1_lo
    return 1 - PI / 2 * L**2 * eps(L) / T1lo(L)


def lamhatlo(L):  # 1 - (pi/2) L^2 eps/(mu_lo T1_lo), mu_lo = 1 - pi eps
    return 1 - PI / 2 * L**2 * eps(L) / ((1 - PI * eps(L)) * T1lo(L))


def cMmargin(L):
    return 2 * (PI - 8 / L) - PI**2 * L**2 * eps(L) / T1lo(L)


def LB(L, c0, m2):
    return iv.mpf(lo(c0)) - iv.mpf(hi(m2)) / (2 * L) - Rminus(L)


E0 = eps(L0)
delta0 = PI * E0 * (L0 + 2)
T1_lo = T1lo(L0)
Q35 = L0 / T1_lo
lam_hat_lo = lamhatlo(L0)
lamM_lo = lamMlo(L0)
OmB_hi = OmBup(L0)
phi_lo = philo(L0)
R_minus = Rminus(L0)
R_plus = Rplus(L0)
cM_margin = cMmargin(L0)
Delta = L0 * delta0 / T1_lo  # >= p delta(Lambda_c) >= 2p ln2 - Lambda_c
norm_hi = iv.exp(Delta / 2) / iv.sqrt(1 - Delta / (52 * LN2))  # p >= 26
LB_cells = LB(L0, C0_cells, M2_cells)
LB_arb = LB(L0, C0_arb, M2_arb)
UB_arb = (iv.mpf(hi(C0_arb)) + R_plus) * norm_hi
UB_cells = (iv.mpf(hi(C0_cells)) + R_plus) * norm_hi
UB_closed = (c0_closed + R_plus) * norm_hi
for k_, v_ in [("eps(35)", E0), ("delta(35) = pi eps (Lambda+2)", delta0), ("T1_lo(35) = 2 ln2 - delta(35)", T1_lo),
               ("Q(35) <= 35/T1_lo", Q35), ("lambda_hat_lo(35)", lam_hat_lo), ("lambda_M >= (35)", lamM_lo),
               ("Omega B, p u <= (35)", OmB_hi), ("phi_lo(35)", phi_lo), ("R_-(35)", R_minus), ("R_+(35)", R_plus),
               ("c_M margin (35)", cM_margin), ("width bound Lambda delta/T1_lo at 35", Delta),
               ("normalisation upper bound", norm_hi), ("LB(35) with local cells", LB_cells),
               ("LB(35) with Arb c0, m2", LB_arb), ("upper constant with Arb c0", UB_arb),
               ("upper constant with local cells only", UB_cells), ("upper constant with the closed form of c0", UB_closed)]:
    out[k_] = s(v_, 14)

G = "decision checks"
check(G, "P2a Q(35) <= 35/T1_lo < 26", hi(Q35) < 26, s(Q35))
check(G, "P2b lambda_hat > 0 on [35, oo) (Q strictly increasing; Route A)", lo(lam_hat_lo) > 0, s(lam_hat_lo))
check(G, "P2c lambda_M >= 1 - (pi/2) Lambda^2 eps/T1_lo > 0", lo(lamM_lo) > 0, s(lamM_lo))
check(G, "P2d Omega B <= pi Lambda eps/T1_lo < 1 (also p u, and Lambda u/T1 <= 1 for the c_M step)", hi(OmB_hi) < 1, s(OmB_hi))
check(G, "P2f phi_xx >= phi_lo > 0", lo(phi_lo) > 0, s(phi_lo))
check(G, "P2g kappa/eps >= LB(35) > 0 (local cells only)", lo(LB_cells) > 0, s(LB_cells))
check(G, "P2h c_M < 2 ln2: 2(pi - 8/Lambda) - pi^2 Lambda^2 eps/T1_lo > 0", lo(cM_margin) > 0, s(cM_margin))
check(G, "P2i -m2/(2L) + m4/(8L^2) <= 0 for L >= 35: 35 >= m4_hi/(4 m2_lo) (cells)", 35 >= hi(M4_cells) / (4 * lo(M2_cells)),
      mpmath.nstr(hi(M4_cells) / (4 * lo(M2_cells)), 8))
check(G, "P2k normalisation argument: Delta/(52 ln2) < 1", hi(Delta / (52 * LN2)) < 1, s(Delta / (52 * LN2)))

S = "stated-constant assertions"
check(S, "S1 LP(a) width 2p ln2 - Lambda_c <= 4.97e-6", le(Delta, "4.97e-6"), s(Delta))
check(S, "S2 LP(b) lambda_M >= 1 - 2.4e-6", ge(lamM_lo, "0.9999976"), s(lamM_lo))
check(S, "S3 LP(b) Omega B <= 1.35e-7", le(OmB_hi, "1.35e-7"), s(OmB_hi))
check(S, "S4 LP(c) p u <= 1.35e-7 (same bound as Omega B)", le(OmB_hi, "1.35e-7"), s(OmB_hi))
check(S, "S5 LP(e) phi_xx >= 1.38629", ge(phi_lo, "1.38629"), s(phi_lo))
check(S, "S6 LP(d) lower constant 0.94844 (Arb c0, m2)", ge(LB_arb, "0.94844"), s(LB_arb))
check(S, "S7 LP(d) lower constant 0.94616 with local cells only", ge(LB_cells, "0.94616"), s(LB_cells))
check(S, "S8 LP(d) upper constant 0.95880 (Arb c0)", le(UB_arb, "0.95880"), s(UB_arb))
check(S, "S9 LP(d) upper constant 0.958793 (Arb c0)", le(UB_arb, "0.958793"), s(UB_arb))
check(S, "S10 upper constant with local cells only <= 0.96105", le(UB_cells, "0.96105"), s(UB_cells))
check(S, "S11 normalisation eps(Lambda_c) 2^p sqrt(4 pi p ln2) <= 1 + 2.6e-6", le(norm_hi, "1.0000026"), s(norm_hi))
check(S, "S12 R_-(35) <= 1.5e-5", le(R_minus, "1.5e-5"), s(R_minus))
check(S, "S13 R_+(35) <= 1.5e-5", le(R_plus, "1.5e-5"), s(R_plus))
check(S, "S14 T1 >= 1.3862941 for Lambda >= 35", ge(T1_lo, "1.3862941"), s(T1_lo))
check(S, "S15 Q(35) <= 25.2472", le(Q35, "25.2472"), s(Q35))
check(S, "S16 lambda_hat >= 0.9999976", ge(lam_hat_lo, "0.9999976"), s(lam_hat_lo))
check(S, "S17 c0 cells within [0.95652, 0.96103]", ge(C0_cells, "0.95652") and le(C0_cells, "0.96103"), s(C0_cells))
check(S, "S18 m2 cells within [0.72025, 0.72444]", ge(M2_cells, "0.72025") and le(M2_cells, "0.72444"), s(M2_cells))
check(S, "S19 m4 cells within [2.0200, 2.0319], hence m4 <= 2.032", ge(M4_cells, "2.0200") and le(M4_cells, "2.0319"), s(M4_cells))
check(S, "S20 K <= 0.735819", le(K_hi, "0.735819"), s(K_hi))
check(S, "S21 eps(35) <= 1.6933e-9 and delta(35) <= 1.969e-7", le(E0, "1.6933e-9") and le(delta0, "1.969e-7"), s(E0) + " " + s(delta0))

out["safe-direction displays (Lambda >= 35)"] = {
    "LB(35), Arb c0 and m2 (down, 5 d.p.)": down(LB_arb, 5),
    "LB(35), local cells (down, 5 d.p.)": down(LB_cells, 5),
    "upper constant, Arb c0 (up, 6 d.p.)": up(UB_arb, 6),
    "upper constant, local cells (up, 5 d.p.)": up(UB_cells, 5),
    "c0 cell lower bound (down, 7 d.p.)": down(C0_cells, 7),
    "c0 cell upper bound (up, 7 d.p.)": up(C0_cells, 7),
    "R_-(35) (up)": up_sig(R_minus, 3),
    "R_+(35) (up)": up_sig(R_plus, 3),
    "width (up)": up_sig(Delta, 3),
}

# ---------------- Part 3: Q < 26 near 35, and the (0, 0.4] arithmetic ----------------
d346 = PI * eps(iv.mpf("34.6")) * (iv.mpf("34.6") + 2)
Q_hi = iv.mpf(35) / (2 * LN2 - d346)
check(G, "P3 Q(Lambda) <= 35/(2 ln2 - delta(34.6)) < 26 on [34.6, 35]", hi(Q_hi) < 26, s(Q_hi))
L04 = iv.mpf("0.4")
qa = 1 / (iv.mpf(1) / 2 - 2 * L04 / 3 - L04**2 / 2)
check(G, "Q1 Q(Lambda) <= 1/(1/2 - 2L/3 - L^2/2) <= value at 0.4 < 26 on (0, 0.4]", hi(qa) < 26 and lo(iv.mpf(1) / 2 - 2 * L04 / 3 - L04**2 / 2) > 0, s(qa))

# ---------------- Part 4: p = 7..25 ----------------
P4 = {}
ok4 = True
for p in range(7, 26):
    Lf = mpf(2 * p) * mpf("0.6931") - mpf("0.02")  # a point just below 2p ln2
    while not (hi(iv.mpf(Lf) / T1lo(iv.mpf(Lf))) < p):
        Lf = Lf - mpf("0.005")
    L = iv.mpf(Lf)
    tb = {"Q_up": L / T1lo(L), "lam_hat_lo": lamhatlo(L), "lamM_lo": lamMlo(L), "OmB_up": OmBup(L), "phi_lo": philo(L),
          "LB_cells": LB(L, C0_cells, M2_cells), "LB_arb": LB(L, C0_arb, M2_arb), "cM_margin": cMmargin(L)}
    good = (hi(tb["Q_up"]) < p and lo(L) >= 9 and lo(tb["lam_hat_lo"]) > 0 and lo(tb["lamM_lo"]) > 0
            and hi(tb["OmB_up"]) < 1 and lo(tb["phi_lo"]) > 0 and lo(tb["LB_cells"]) > 0 and lo(tb["cM_margin"]) > 0)
    ok4 = ok4 and good
    P4[p] = {"Lambda_p^-": mpmath.nstr(Lf, 10), "Q_up(Lambda_p^-)": up(tb["Q_up"], 7),
             "kappa/eps >= (local cells; down, 4 d.p.)": down(tb["LB_cells"], 4),
             "kappa/eps >= (Arb c0, m2; down, 4 d.p.)": down(tb["LB_arb"], 4),
             "lambda_M >= (down, 6 d.p.)": down(tb["lamM_lo"], 6),
             "Omega B <= (up, 2 s.f.)": up_sig(tb["OmB_up"], 2),
             "phi_xx >= (down, 5 d.p.)": down(tb["phi_lo"], 5),
             "c_M margin (down, 3 d.p.)": down(tb["cM_margin"], 3), "all": good}
check(G, "P4 for each p = 7..25: Q(Lambda_p^-) < p, Lambda_p^- >= 9, and every bound holds at Lambda_p^- (LB with local cells > 0)", ok4, "see P4 table")
out["P4 table"] = P4

R = "remarks (analytic, not checks)"
out[R]["monotonicity"] = ("Lambda^k eps(Lambda) decreases for Lambda > 2k - 1; only k <= 2 occurs, and T1_lo, phi_lo increase. "
                          "Hence on [9, oo): LB, lambda_M bound, lambda_hat bound, phi_lo and the c_M margin increase; the Omega B bound, "
                          "R_- , R_+ and the width bound decrease (Appendix B.4).")
out[R]["counts"] = {g_: len(out[g_]) for g_ in ("decision checks", "stated-constant assertions", "consistency checks")}

json.dump(out, sys.stdout, indent=1, default=str)
print()
n = {g_: sum(1 for v in out[g_].values() if v["ok"]) for g_ in ("decision checks", "stated-constant assertions", "consistency checks")}
print(f"decision {n['decision checks']}/{len(out['decision checks'])}, stated constants {n['stated-constant assertions']}/"
      f"{len(out['stated-constant assertions'])}, consistency {n['consistency checks']}/{len(out['consistency checks'])}; "
      f"{len(fails)} fail: {fails}", file=sys.stderr)
sys.exit(1 if fails else 0)
