#!/usr/bin/env python3
"""Further Arb checks for the triple-point certificate (Appendix B.5, app:nm-cert).

Imports the unmodified ../kappa-cert/kappa_cert.py; needs python-flint 0.9.0:
    python cert_extra.py > cert_extra.json 2> cert_extra.log

W  python-flint wrapper: the analytic flag reaches the integrand; log(analytic=True) rejects a
   ball on the branch cut; the sqrt example of the python-flint docs; the certificate's log
   integrands with the flag forced off (does the flag do work?).
U  uniform-in-Lambda spot test: an integral over a Lambda-ball contains the point integrals.
N  direct Arb certification (no JSON string parsing) of the conditions (N2), (N3) of Section 7.5 (sec:nm-fm)
   and of c_M < 2 ln 2 that are not in kappa_cert's sign list: det S > 0, Btilde'' = Omega - S_qq/det S < 0, the Q_p margin
   p mu^(p-1) + (p-1) mu^p - 1 > 0, c_M = beta_c^2/2 < 2 ln 2; NL-reduced and NL-free codings.
L  point checks of the two bounds inside lemma L2 of Appendix B.5 (evidence only; L2 itself is a hand proof).
G  Lemma 6 of Zhou (2024), last step (Remark 7.4, rem:nm-chen): a full cell cover of (0,1) proving G1 > 0
   without root counting, plus the ten-cell and single-cell bounds of ../zhou-lemma6/zc_lemma6.py (I2) and
   ../ball-checks/ball_checks.py (C2), all in Arb.
"""

import json
import os
import sys
import time
from fractions import Fraction

import flint
from flint import acb, arb, ctx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kappa-cert"))
import kappa_cert as kc  # noqa: E402

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

out = {"python_flint": flint.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def s(x, n=30):
    return x.str(n, radius=True)


# ---------------------------------------------------------------------------------------------
# W: wrapper behaviour
# ---------------------------------------------------------------------------------------------
ctx.prec = 64
seen = []


def rec(z, a):
    seen.append(bool(a))
    return (-(z * z)).exp()


acb.integral(rec, -1, 1)
out["W1_analytic_flags_seen"] = {"True": seen.count(True), "False": seen.count(False)}
zc = acb(arb(-1), arb(0, 0.1))
out["W2_log_analytic_on_cut_is_nonfinite"] = not zc.log(analytic=True).is_finite()
out["W2_log_plain_on_cut_is_finite"] = bool(zc.log(analytic=False).is_finite())
ctx.prec = 96
bad = acb.integral(lambda x, _: x.sqrt(), 1, 4)
good = acb.integral(lambda x, a: x.sqrt(analytic=a), 1, 4)
exact = arb(14) / 3
out["W3_sqrt_doc_example"] = {
    "flag_ignored": s(bad.real, 20), "flag_forwarded": s(good.real, 20),
    "flag_ignored_contains_14/3": bool(bad.real.contains(exact)),
    "flag_forwarded_contains_14/3": bool(good.real.contains(exact)),
}


def growth_of(name):
    f, (a, b, c) = kc.INTEGRANDS[name]
    conv = {"ln2": kc.ln2(), "2ln2sq": 2 * kc.ln2() ** 2}
    a = conv[a] if isinstance(a, str) else kc.fr2arb(a) if isinstance(a, Fraction) else arb(a)
    return f, (a, arb(b), arb(c))


def E_custom(name, lam, noflag=False, zcut=None, tol_bits=None):
    f, g = growth_of(name)
    ff = (lambda h, a: f(h, False)) if noflag else f
    return kc.gexp(ff, g, lam, kc.ZCUT if zcut is None else zcut, kc.TOL_BITS if tol_bits is None else tol_bits)


W4 = {}
for label, prec, lam, zcut, tb in (
        ("tight_point_Lc3", 256, kc.fr2arb(Fraction(2396, 1024)), 22, 230),
        ("scan_ball_width_1/4_at_2.25", 96, kc.interval_ball(Fraction(2), Fraction(5, 2)), 12, 64),
        ("scan_ball_width_1/4_at_34.5", 96, kc.interval_ball(Fraction(69, 2), Fraction(35)), 12, 64)):
    ctx.prec = prec
    row = {}
    for name in ("logcosh", "logcosh2", "r", "ln2cosh2"):
        rr = {}
        for tag, nf in (("flag", False), ("noflag", True)):
            try:
                v = E_custom(name, lam, noflag=nf, zcut=zcut, tol_bits=tb)
                rr[tag] = s(v, 12)
                rr[tag + "_obj"] = v
            except ArithmeticError:
                rr[tag] = "non-finite"
        if "flag_obj" in rr and "noflag_obj" in rr:
            rr["overlap"] = bool(rr["flag_obj"].overlaps(rr["noflag_obj"]))
        rr.pop("flag_obj", None)
        rr.pop("noflag_obj", None)
        row[name] = rr
    W4[label] = row
out["W4_log_integrands_flag_on_vs_off"] = W4

# ---------------------------------------------------------------------------------------------
# U: uniform enclosure spot test
# ---------------------------------------------------------------------------------------------
U = {}
u_ok = True
for p, Lc in ((3, 2.339641575), (12, 16.620966), (25, 34.657353)):
    c = Fraction(round(Lc * 1024), 1024)
    for w in (Fraction(1, 1024), Fraction(1, 64), Fraction(1, 8)):
        lo, hi = c - w, c + w
        X = kc.interval_ball(lo, hi)
        for name in ("tanh", "tanh3", "logcosh", "logcosh2", "r", "ln2cosh2", "wprime", "muprime", "sech4", "q4"):
            ctx.prec = 96
            key = f"p{p}_w{w}_{name}"
            try:
                encX = E_custom(name, X, zcut=12, tol_bits=64)
            except ArithmeticError:
                U[key] = "ball integral non-finite (undecided, not a failure)"
                continue
            pts = [lo + (hi - lo) * Fraction(k, 8) for k in range(9)]
            contained = []
            for pt in pts:
                ctx.prec = 160
                v = E_custom(name, kc.fr2arb(pt), zcut=16, tol_bits=120)
                ctx.prec = 96
                contained.append(bool(encX.contains(v)))
            U[key] = {"all_points_contained": all(contained), "ball_enclosure": s(encX, 10)}
            u_ok &= all(contained)
out["U_uniform_spot_test"] = U
out["U_all_contained"] = u_ok

# ---------------------------------------------------------------------------------------------
# N: conditions (N2), (N3) and c_M < 2 ln 2 directly in Arb
# ---------------------------------------------------------------------------------------------
N = {}
n_ok = True
t0 = time.time()
for p in range(3, 26):
    L1, L2, info = kc.lambda_c(p)
    ctx.prec = kc.PREC
    f1 = kc.form1(p, L1)
    f2, raw = kc.form2(p, L2)
    mu, tau, Om = f1["mu"], f1["tau"], f1["Omega"]
    Shh, Shq, Sqq = 1 - mu, -Om * (mu - tau), f1["phi_qq"]
    detS = Shh * Sqq - Shq ** 2
    Bpp = Om - Sqq / detS
    rOm = f2["Omega"]
    rShh, rShq, rSqq = raw["E_sech2"], -rOm * raw["E_sech2tanh"], raw["phi_qq"]
    rdetS = rShh * rSqq - rShq ** 2
    rBpp = rOm - rSqq / rdetS
    n3 = p * mu ** (p - 1) + (p - 1) * mu ** p - 1
    cM = f1["beta_c"] ** 2 / 2
    checks = {
        "detS>0": bool(detS > 0), "Btilde''<0": bool(Bpp < 0),
        "raw_detS>0": bool(rdetS > 0), "raw_Btilde''<0": bool(rBpp < 0),
        "Qp_margin>0": bool(n3 > 0), "c_M<2ln2": bool(cM < 2 * kc.ln2()),
        "0<mu<1": bool(mu > 0 and mu < 1), "phi_xb>0": bool(f1["phi_xb"] > 0),
        "phi_xx>0": bool(f1["phi_xx"] > 0), "lambda_M>0": bool(f1["lambda_M"] > 0),
        "A_FM-A_SG>0": bool(f1["A_FM_minus_A_SG"] > 0), "detH<0": bool(f1["detH"] < 0),
    }
    cons = {
        "detH_vs_Om*detS*Btilde''": bool(f1["detH"].overlaps(Om * detS * Bpp)),
        "phi_mq_vs_Om*Shq": bool(f1["phi_mq"].overlaps(Om * Shq)),
        "detS_vs_raw_detS": bool(detS.overlaps(rdetS)),
        "Btilde''_vs_raw": bool(Bpp.overlaps(rBpp)),
    }
    ok = all(checks.values()) and all(cons.values()) and info["unique_zero_on_(0,inf)"]
    n_ok &= ok
    N[str(p)] = {"detS": s(detS, 20), "Btilde''": s(Bpp, 20), "raw_detS": s(rdetS, 20),
                 "Qp_margin": s(n3, 20), "c_M": s(cM, 20), "Lambda_c": s(L1, 40),
                 "unique_zero": info["unique_zero_on_(0,inf)"], "checks": checks, "consistency": cons, "ok": ok}
    print(f"N p={p} detS={detS.str(8)} B''={Bpp.str(8)} Qp={n3.str(8)} cM={cM.str(8)} ok={ok}", file=sys.stderr, flush=True)
out["N_theorem_NM_hypotheses"] = N
out["N_all_ok"] = n_ok
out["N_seconds"] = round(time.time() - t0, 1)

# ---------------------------------------------------------------------------------------------
# L: lemma L2 bounds at points (evidence)
# ---------------------------------------------------------------------------------------------
ctx.prec = 128
L = {}
for lamf in (Fraction(1, 64), Fraction(1, 16), Fraction(1, 8), Fraction(3, 16)):
    lam = kc.fr2arb(lamf)
    mu = E_custom("tanh", lam, zcut=16, tol_bits=100)
    W = E_custom("logcosh", lam, zcut=16, tol_bits=100) - lam / 2
    row = {"mu>=L-L^2-2L^3/3": bool(mu - (lam - lam ** 2 - 2 * lam ** 3 / 3) > 0),
           "W<=L^2/4": bool(lam ** 2 / 4 - W > 0)}
    for p in (3, 25):
        row[f"D_{p}<0"] = bool(W - kc.cp(p) * lam * mu < 0)
    L[str(lamf)] = row
out["L2_point_checks"] = L

# ---------------------------------------------------------------------------------------------
# G: Lemma 6 of Zhou (2024), last step, G1(t) = artanh t - 3t/den(t) > 0 on (0,1)
#     den(t) = 1 + 2(1-t^2) + 3/(1 + 2/sqrt(1-t^2)) is decreasing and >= 1 on (0,1), so
#     h(t) = 3t/den(t) is increasing and <= 3; artanh is increasing. Cell bound on [a,b]:
#     G1 >= artanh(a) - h(b).
# ---------------------------------------------------------------------------------------------
ctx.prec = 128


def q2arb(fr):
    return arb(fr.numerator) / arb(fr.denominator)


def den(t):
    return 1 + 2 * (1 - t * t) + 3 / (1 + 2 / (1 - t * t).sqrt())


def cell_lb(a, b):
    return q2arb(a).atanh() - 3 * q2arb(b) / den(q2arb(b))


G = {}
a0, a1 = Fraction(1, 1000), Fraction(999, 1000)
G["near0_den(a0)>3"] = bool(den(q2arb(a0)) > 3)          # G1(t) > t(1 - 3/den(a0)) > 0 on (0, a0]
G["near1_artanh(a1)>3"] = bool(q2arb(a1).atanh() > 3)    # G1(t) > artanh(a1) - 3 > 0 on [a1, 1)
stack, cells, minlb = [(a0, a1)], 0, None
while stack:
    a, b = stack.pop()
    lb = cell_lb(a, b)
    if lb > 0:
        cells += 1
        lo = lb.lower()
        minlb = lo if minlb is None or lo < minlb else minlb
        continue
    if b - a < Fraction(1, 10 ** 9):
        raise RuntimeError(f"G1 cover failed near {float(a)}")
    m = (a + b) / 2
    stack += [(a, m), (m, b)]
G["cover_[a0,a1]_cells"] = cells
G["cover_min_cell_lower_bound"] = s(minlb, 10)
G["cover_complete"] = bool(G["near0_den(a0)>3"] and G["near1_artanh(a1)>3"] and cells > 0)
G["I2_ten_cells_min_lb"] = s(min((cell_lb(Fraction(94000 + 100 * k, 100000), Fraction(94100 + 100 * k, 100000)).lower()
                                  for k in range(10)), key=lambda x: float(x.mid())), 10)
G["C2_single_cell_[0.65,0.66]_lb"] = s(cell_lb(Fraction(65, 100), Fraction(66, 100)).lower(), 10)
out["G_lemma6_last_step"] = G

out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
print(json.dumps(out, indent=1))
sys.exit(0 if (u_ok and n_ok and G["cover_complete"]) else 1)
