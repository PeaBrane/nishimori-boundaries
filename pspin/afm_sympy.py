"""Re-derivation of the (m,h,q) Hessian computation of A_FM in the proof of Lemma 7.12 (lem:nm-closed;
Appendix B.1, app:nm-closed).

U(m;h,q) = beta j0 m^p + log2 + F(h, xi_beta'(q)) + (xi(1) - xi(q) - (1-q) xi'(q))/2 - h m,
F(h,r) = E logcosh(h + sqrt(r) g).  F is an undefined function; its derivatives are replaced by
Gaussian averages only at the end:  d_h^a d_r^b F = 2^{-b} E[logcosh^{(a+2b)}(H)], and at the
Nishimori-line point h = r = Lambda:  E t = E t^2 = mu, E t^3 = E t^4 = tau (t = tanh H).
"""

import json
import sys

import sympy as sp

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

m, h, q, be, j0 = sp.symbols("m h q beta j0", positive=True)
p = sp.symbols("p", positive=True)
Lam, mu, tau = sp.symbols("Lambda mu tau", positive=True)

t = sp.symbols("t")
lc_derivs = {1: t, 2: 1 - t**2, 3: -2 * t + 2 * t**3, 4: -2 + 8 * t**2 - 6 * t**4}
# sanity of the tanh-polynomial derivatives of log cosh
y = sp.symbols("y")
chk = [sp.simplify(sp.diff(sp.log(sp.cosh(y)), y, k) - lc_derivs[k].subs(t, sp.tanh(y))) for k in (1, 2, 3, 4)]
assert all(c == 0 for c in chk), chk


def Et(poly):
    poly = sp.expand(poly)
    vals = {0: 1, 1: mu, 2: mu, 3: tau, 4: tau}
    return sum(poly.coeff(t, k) * vals[k] for k in range(5))


# Represent U as explicit part + G(h, r(q, beta)); build derivatives through a jet of G.
rq = be**2 * p * q ** (p - 1) / 2
Uexp = be * j0 * m**p + sp.log(2) + (be**2 / 2 - be**2 * q**p / 2 - (1 - q) * rq) / 2 - h * m
# jet symbols g_{a,b} = d_h^a d_r^b G at (h, r)
gs = {(a, b): sp.Symbol(f"g_{a}_{b}") for a in range(4) for b in range(4)}


def D(expr, z):
    res = sp.diff(expr, z)
    dh = sp.diff(h, z)
    dr = sp.diff(rq, z)
    for (a, b), g in gs.items():
        c = sp.diff(expr, g)
        if c == 0:
            continue
        if (a + 1, b) in gs and dh != 0:
            res += c * gs[(a + 1, b)] * dh
        if (a, b + 1) in gs and dr != 0:
            res += c * gs[(a, b + 1)] * dr
    return res


U0 = Uexp + gs[(0, 0)]
vars_ = [m, h, q]


def at_M(expr):
    # jet values: g_{a,b} = 2^{-b} E[lc^{(a+2b)}]
    reps = {}
    for (a, b), g in gs.items():
        k = a + 2 * b
        if k == 0:
            continue  # value of G itself never needed below
        if k > 4:
            continue
        reps[g] = Et(lc_derivs[k]) / 2**b
    e = expr.subs(reps)
    # point: m = q = mu, h = Lambda, j0 = beta/2, beta^2 = 2 Lambda mu^{1-p}/p
    e = e.subs({m: mu, q: mu, h: Lam, j0: be / 2})
    e = e.subs(be, sp.sqrt(2 * Lam * mu ** (1 - p) / p))
    return sp.simplify(sp.powsimp(sp.expand_power_base(e, force=True), force=True))


results = {}
grad = [D(U0, z) for z in vars_]
Hes = sp.Matrix(3, 3, lambda i, j: D(grad[i], vars_[j]))
Ub = D(U0, be)
Ubb = D(Ub, be)
dvec = sp.Matrix([D(Ub, z) for z in vars_])

# stationarity at M (needs E tanh = mu, E tanh^2 = mu)
gradM = [at_M(g) for g in grad]
results["grad_at_M"] = [str(g) for g in gradM]

Om = (p - 1) * Lam / mu
B = 1 - 3 * mu + 2 * tau
v = mu - tau
w = 1 - 2 * mu + tau
mubar = 1 - mu
lam = 1 - Om * w
HM = Hes.applyfunc(at_M)
Hpaper = sp.Matrix([[Om, -1, 0], [-1, mubar, -Om * v], [0, -Om * v, Om / 2 * (1 - Om * (B - v))]])
results["H_minus_paper"] = str((HM - Hpaper).applyfunc(sp.simplify))
be1 = sp.sqrt(2 * Lam * mu ** (1 - p) / p)
UbbM = at_M(Ubb)
results["Ubb_minus_paper"] = str(sp.simplify(UbbM - (sp.Rational(1, 2) - mu**p / 2 - 2 * Lam**2 / be1**2 * (B - v))))
dM = dvec.applyfunc(at_M)
k = sp.Matrix([1, -2 * v, -Om * (B - v)])
results["beta1_d_minus_Lambda_k"] = str((be1 * dM - Lam * k).applyfunc(sp.simplify))
yv = -sp.Matrix([B, 1, 2 * B]) / (1 - Om * B)
results["H_y_minus_k"] = str((Hpaper * yv - k).applyfunc(sp.simplify))
kTy = (k.T * yv)[0, 0]
results["kTy_minus_paper"] = str(sp.simplify(kTy - (2 * v - B + 2 * Om * (B - v) * B) / (1 - Om * B)))
S = Hpaper[1:, 1:]
detS = sp.simplify(S.det())
results["detS_minus_paper"] = str(sp.simplify(detS - Om / 2 * (lam * B + 2 * v)))
results["detS_minus_paper_mubar_form"] = str(sp.simplify(detS - Om / 2 * (mubar - Om * w * B)))
detH3 = sp.simplify(Hpaper.det())
results["detH3_minus_paper"] = str(sp.simplify(detH3 + Om / 2 * lam * (1 - Om * B)))
results["detH3_minus_(Om detS - Sqq)"] = str(sp.simplify(detH3 - (Om * detS - S[1, 1])))
Bt2 = Om - S[1, 1] / detS
results["Bt2_minus_paper"] = str(sp.simplify(Bt2 + lam * (1 - Om * B) / (lam * B + 2 * v)))
results["Bt2_minus_detH3_over_detS"] = str(sp.simplify(Bt2 - detH3 / detS))

# A_FM from this Hessian (no use of the paper's y): 1/2 - (Ubb - d^T H^{-1} d)
phi2 = UbbM - (dM.T * HM.inv() * dM)[0, 0]
A_mine = sp.Rational(1, 2) - phi2
A_paper = mu**p / 2 + Lam**2 / be1**2 * B / (1 - Om * B)
results["A_mine_minus_paper"] = str(sp.simplify(sp.together(A_mine - A_paper)))

# certified direct form (first coding, form1, of kappa-cert/kappa_cert.py): Nishimori (m,q) 2x2 chain rule
fmm = Om * ((1 - mu) * Om - 1)
fmq = -Om**2 * (mu - tau)
fqq = Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))
fmb = Om * Lam / be1 * B
fbb = sp.Rational(1, 2) - mu**p / 2 - (Lam / be1) ** 2 * B
HN = sp.Matrix([[fmm, fmq], [fmq, fqq]])
vN = sp.Matrix([fmb, -fmb])
A_cert = sp.Rational(1, 2) - fbb + (vN.T * HN.inv() * vN)[0, 0]
A_cert_simpl = mu**p / 2 - (Lam / be1) ** 2 * Om**2 * B * lam / (2 * HN.det())
results["A_mine_minus_certified_direct"] = str(sp.simplify(sp.together(A_mine - A_cert)))
results["A_mine_minus_certified_simplified"] = str(sp.simplify(sp.together(A_mine - A_cert_simpl)))
results["detHN_minus_(-Om^2/2 lam (1-OmB))"] = str(sp.simplify(HN.det() + Om**2 / 2 * lam * (1 - Om * B)))
results["detHN_over_detH3"] = str(sp.simplify(HN.det() / detH3))

# separate re-derivation of the Nishimori-parametrization reduction: U(m; p beta j0 m^{p-1}, q)
# should reproduce Nishimori's (m,q) Hessian entries (phi_mm, phi_mq, phi_qq) and phi_mb, phi_bb
hN = p * be * j0 * m ** (p - 1)
UN = U0.subs(h, hN)


def DN(expr, z):
    res = sp.diff(expr, z)
    dh = sp.diff(hN, z)
    dr = sp.diff(rq, z)
    for (a, b), g in gs.items():
        c = sp.diff(expr, g)
        if c == 0:
            continue
        if (a + 1, b) in gs and dh != 0:
            res += c * gs[(a + 1, b)] * dh
        if (a, b + 1) in gs and dr != 0:
            res += c * gs[(a, b + 1)] * dr
    return res


gN = [DN(UN, z) for z in (m, q)]
HNm = sp.Matrix(2, 2, lambda i, j: DN(gN[i], (m, q)[j]))
UbN = DN(UN, be)


def at_MN(expr):
    return at_M(expr)


HNmM = HNm.applyfunc(at_MN)
results["NishimoriHessian_minus_cert"] = str((HNmM - HN).applyfunc(sp.simplify))
vNM = sp.Matrix([at_MN(DN(UbN, z)) for z in (m, q)])
results["Nishimori_v_minus_cert"] = str((vNM - vN).applyfunc(sp.simplify))
results["Nishimori_phibb_minus_cert"] = str(sp.simplify(at_MN(DN(UbN, be)) - fbb))

zero_forms = {"0", "Matrix([[0], [0], [0]])", "Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])", "Matrix([[0, 0], [0, 0]])", "Matrix([[0], [0]])"}
bad = [kk for kk, vv in results.items() if kk not in ("grad_at_M", "detHN_over_detH3") and vv not in zero_forms]
bad += ["grad_at_M"] if results["grad_at_M"] != ["0", "0", "0"] else []
results["NONZERO"] = bad
print(json.dumps(results, indent=1))
sys.exit(1 if bad else 0)
