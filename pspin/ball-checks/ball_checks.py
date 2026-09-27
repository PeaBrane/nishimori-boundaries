"""Checks from the stored triple-point Arb balls and of Lemma 6 of Zhou (2024).
C1: conditions (N2) and (N3) of Section 7.5 (sec:nm-fm) from the stored ../kappa-cert/kappa_cert.json Arb
    balls (mpmath.iv, outward rounding); a third route besides ../cert-check/ and ../triple-point/.
C2: final step of Lemma 6 of Zhou (2024) (Remark 7.4, rem:nm-chen): sign structure of G1' (A>0, B<0 on [0,1]) and monotone cell bounds on
    both critical-point brackets [0.65,0.66] and [0.94,0.95].
C3: float check of G2'(x) = Cov_h(log cosh h, tanh^2 h), h ~ N(x,x).
"""
import json, os, re, sys
import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.hermite_e import hermegauss

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

iv = mp.iv
iv.dps = 90
KC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kappa-cert", "kappa_cert.json")
d = json.load(open(KC))
pat = re.compile(r"\[([-0-9.eE+]+) \+/- ([0-9.eE+-]+)\]")

def ball(s):
    m_, r_ = pat.fullmatch(s.strip()).groups()
    M = iv.mpf(m_); R = iv.mpf(r_)
    return M + iv.mpf([-R.b, R.b])

out = {"C1": {}}
allok = True
for ps, res in d["results"].items():
    p = int(ps); f = res["form1"]
    mu, tau, Om = ball(f["mu"]), ball(f["tau"]), ball(f["Omega"])
    fqq, fmq, detH = ball(f["phi_qq"]), ball(f["phi_mq"]), ball(f["detH"])
    Shh = 1 - mu; Shq = fmq / Om; Sqq = fqq
    Shq2 = -Om * (mu - tau)
    detS = Shh * Sqq - Shq2 ** 2
    red = Om - Sqq / detS
    ident = Om * detS * red - detH
    n3 = p * mu ** (p - 1) + (p - 1) * mu ** p - 1
    fl = lambda x: float(mp.mpf(x))
    r = {"detS_lo": fl(detS.a), "detS_hi": fl(detS.b),
         "reduced_curv_hi": fl(red.b), "detH_hi": fl(detH.b),
         "identity_residual_abs_max": fl(max(abs(fl(ident.a)), abs(fl(ident.b)))),
         "Shq_consistency_overlap": bool((Shq - Shq2).a <= 0 <= (Shq - Shq2).b),
         "N3_margin_lo": fl(n3.a), "mu_lo": fl(mu.a)}
    ok = detS.a > 0 and red.b < 0 and detH.b < 0 and n3.a > 0
    r["N2_and_N3_certified"] = bool(ok)
    allok &= bool(ok)
    out["C1"][ps] = r
out["C1_all"] = allok

# C2: final step of Lemma 6 of Zhou (2024)
t, s = sp.symbols("t s", positive=True)
A_s = 5*s**5 - 6*s**4 - 51*s**3 + 117*s**2 - 90*s + 27
B_s = 36*s**3 - 99*s**2 + 81*s - 27
out["C2 roots of A(s) in [0,1]"] = int(sp.Poly(A_s, s).count_roots(0, 1))
out["C2 roots of B(s) in [0,1]"] = int(sp.Poly(B_s, s).count_roots(0, 1))
out["C2 A(0),A(1),B(0),B(1)"] = [int(A_s.subs(s, 0)), int(A_s.subs(s, 1)), int(B_s.subs(s, 0)), int(B_s.subs(s, 1))]
iv.dps = 30
def cell_lb(a_, b_):
    va = iv.log((1 + a_) / (1 - a_)) / 2
    denb = 1 + 2 * (1 - b_**2) + 3 / (1 + 2 / iv.sqrt(1 - b_**2))
    return (va - 3 * b_ / denb).a
lo1 = [cell_lb(iv.mpf(6500 + k) / 10000, iv.mpf(6501 + k) / 10000) for k in range(100)]
lo2 = [cell_lb(iv.mpf(9400 + k) / 10000, iv.mpf(9401 + k) / 10000) for k in range(100)]
out["C2 min cell lb on [0.65,0.66] (100 cells)"] = float(mp.mpf(min(lo1)))
out["C2 min cell lb on [0.94,0.95] (100 cells)"] = float(mp.mpf(min(lo2)))
out["C2 single-cell lb on [0.65,0.66]"] = float(mp.mpf(cell_lb(iv.mpf("0.65"), iv.mpf("0.66"))))
out["C2 all cell lbs positive"] = bool(min(lo1) > 0 and min(lo2) > 0)

# C3: G2' = Cov(log cosh h, tanh^2 h), h ~ N(x, x)
Z, W = hermegauss(160); W = W / W.sum()
def G2(x):
    h = x + np.sqrt(x) * Z
    L = np.log(np.cosh(h))
    return x - 2 * (W @ L) + (W @ L**2) - (W @ L) ** 2
def cov(x):
    h = x + np.sqrt(x) * Z
    L = np.log(np.cosh(h)); t2 = np.tanh(h) ** 2
    return (W @ (L * t2)) - (W @ L) * (W @ t2)
rows = []
for x in (0.5, 1.0, 2.3396, 4.4789, 8.0):
    e = 1e-4
    fd = (G2(x + e) - G2(x - e)) / (2 * e)
    rows.append([x, fd, cov(x)])
out["C3 [x, FD G2', Cov]"] = rows
print(json.dumps(out, indent=1))
