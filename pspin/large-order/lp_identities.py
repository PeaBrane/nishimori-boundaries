"""Exact (sympy) identities used in Section 7 and Appendix B of the paper (Lemma 7.3 lem:nm-U,
Lemma 7.12 lem:nm-closed, Proposition 7.14 prop:nm-J).

Every check is an exact symbolic identity: a difference that simplifies to 0 (or an exact limit).
Groups A, B, C (26 identities):
A1-A7 coefficient algebra, B1-B4 calculus in Lambda (W' = mu/2 and mu' = w are inputs, lemma L1 of App. B.5),
C1-C9 elementary functions and the Catalan-constant route to c0.

D. The cell-free proof of Lemma 7.3 ("Lemma U") and the reduction used with it (Appendix B.2, app:nm-U).
   D1  tilts: cosh tanh^2 = sinh tanh; cosh sech^4 = sech^3.
   D2  k := (sech tanh + atan sinh)/2 has k' = sech^3, k(0) = 0, k(oo) = pi/4.
   D3  f := h k has f'' = sech^3 (2 - 3 h tanh h).
   D4  g := sinh tanh has g' = sinh (1 + sech^2) and g'' = sech^3 (cosh^4 - cosh^2 + 2).
   D5  Delta_l := 2 - 3 h tanh h - l (cosh^4 - cosh^2 + 2) has
       Delta_l' = -3 tanh h - 3 h sech^2 h - 2 l sinh h cosh h (2 cosh^2 h - 1).
   D6  Delta_1 = -3 h tanh h - cosh^2 h sinh^2 h.
   D7  phi_l := f - l g has phi_l(0) = phi_l'(0) = 0 and phi_l''(0) = 2(1 - l).
   D8  xi = beta^2 q^p/2, s = xi'(q), theta = q xi' - xi (symbolic real p):
       theta = (p-1) s q/p, so Phi(q) = psi(s) - (s + theta)/2 = W(s) - ((p-1)/(2p)) s q with W = psi - s/2.
   D9  beta^2 = 2 Lambda q^{1-p}/p gives xi'(q) = Lambda; and q xi''(q) = (p-1) xi'(q).
   D10 s = Lambda (q/q1)^{p-1}: log q - log q1 = (log s - log Lambda)/(p-1).
E. The Stein-form proof of Proposition 7.14 ("Proposition J") and the Catalan-free evaluation of c0
   (Appendix B.3, app:nm-J).
   E1  L := log(1 + e^{-2h}) has L' = tanh h - 1.
   E2  g = sech^2 = 1 - t^2 (t = tanh h): g'' + 2g' + g = -1 - 4t + 7t^2 + 4t^3 - 6t^4, whose expectation
       under E t = E t^2 = mu, E t^3 = E t^4 = tau is -(1 - 3 mu + 2 tau) = -B.
   E3  L(-h) = 2h + L(h).
   E4  for h > 0: (e^h L(h)^2 + e^{-h} L(-h)^2)/2 - h^2 sech h = J(h).
   E5  x = e^{-h}: e^h L(h)^2 |dh/dx| = x^{-2} log^2(1 + x^2).
   E6  (-log^2(1+x^2)/x)' = x^{-2} log^2(1+x^2) - 4 log(1+x^2)/(1+x^2); the boundary term vanishes at 0 and oo.
   E7  x = tan(theta): (1 + tan^2) cos^2 = 1 and (tan)'/(1 + tan^2) = 1, so the integrand becomes -2 log cos.
   E8  int_0^oo h^2 e^{-(2k+1)h} dh = 2/(2k+1)^3 (symbolic k), so int_0^oo h^2 sech h = 4 beta(3).
   The classical values int_0^{pi/2} log cos = -(pi/2) log 2 and beta(3) = pi^3/32 are cited, not checked here.
N. Negative control: five perturbed identities (of A1, D3, D4, E2, E4) must FAIL.
Output: JSON on stdout; exit 1 if any identity fails or any negative control passes.
"""

import json
import sys

import sympy as sp

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

out = {}
fails = []


def rec(name, expr):
    ok = sp.simplify(expr) == 0
    out[name] = "PASS (exact)" if ok else f"FAIL: {sp.simplify(expr)}"
    if not ok:
        fails.append(name)


# ---------------- A. coefficient algebra ----------------
p, L, mu, tau, V, P, be, Om = sp.symbols("p Lambda mu tau V P beta Omega", positive=True)
u = 1 - mu
v = mu - tau
w = 1 - 2 * mu + tau
B = 1 - 3 * mu + 2 * tau
lam = 1 - Om * w
phi_mm = Om * ((1 - mu) * Om - 1)
phi_mq = -Om**2 * (mu - tau)
phi_qq = Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))
phi_mb = Om * L / be * B
phi_qb = -phi_mb
phi_bb = sp.Rational(1, 2) - P / 2 - (L / be) ** 2 * B
H = sp.Matrix([[phi_mm, phi_mq], [phi_mq, phi_qq]])
vv = sp.Matrix([phi_mb, phi_qb])
A_FM = sp.Rational(1, 2) - phi_bb + (vv.T * H.inv() * vv)[0, 0]
phi_xb = be * P / 2
phi_xx = V - (p - 1) * be**2 * P / 2
A_SG = phi_xb**2 / phi_xx
kappa = sp.expand(be**2) * (A_FM - A_SG)
kappa = kappa.subs(be, sp.sqrt(2 * L * mu / (p * P)))
T1 = L * mu / p
X = V - L * mu
Y = L**2 * B
target = T1 * X / (T1 + X) + Y / (1 - Om * B)
rec("A1 kappa = T1 X/(T1+X) + Y/(1-Omega B)", sp.together(kappa - target))
rec("A2 phi_mm+2phi_mq+phi_qq = -(Omega/2) lambda", phi_mm + 2 * phi_mq + phi_qq + Om / 2 * lam)
rec("A3 det H = -(Omega^2/2) lambda (1-Omega B)", H.det() + Om**2 / 2 * lam * (1 - Om * B))
S = sp.Matrix([[1 - mu, -Om * (mu - tau)], [-Om * (mu - tau), Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))]])
detS = S.det()
rec("A4 det S = (Omega/2)(u - Omega w B)", detS - Om / 2 * (u - Om * w * B))
Bt2 = Om - S[1, 1] / detS
rec("A5 Btilde'' = -lambda(1-Omega B)/(u-Omega w B)", sp.together(Bt2 + lam * (1 - Om * B) / (u - Om * w * B)))
rec("A6 det H = Omega det S Btilde''", sp.together(H.det() - Om * detS * Bt2))
# the NL-free form used in kappa-cert (A_FM simplified) agrees with the direct form
A_FM_simpl = P / 2 - (L / be) ** 2 * Om**2 * B * lam / (2 * H.det())
rec("A7 A_FM direct = simplified (form1 of kappa_cert.py)", sp.together(A_FM - A_FM_simpl))

# ---------------- B. calculus in Lambda ----------------
Lam = sp.symbols("Lambda", positive=True)
muf = sp.Function("mu")(Lam)
Wf = sp.Function("W")(Lam)
wf = sp.Function("w")(Lam)
subsd = {sp.Derivative(Wf, Lam): muf / 2, sp.Derivative(muf, Lam): wf}
Dp = Wf - (p - 1) * Lam * muf / (2 * p)
lam_p = 1 - (p - 1) * Lam * wf / muf
rec("B1 D_p' = mu lambda_p/(2p)", sp.diff(Dp, Lam).subs(subsd) - muf * lam_p / (2 * p))
T1f = Lam * muf - 2 * Wf
rec("B2 T1' = Lambda w", sp.diff(T1f, Lam).subs(subsd) - Lam * wf)
Qf = Lam * muf / T1f
rec("B3 D_p = (T1/(2p))(Q - p)", sp.together(Dp - T1f / (2 * p) * (Qf - p)))
rec("B4 Q' = mu(1-(Q-1) Lambda w/mu)/T1", sp.together(sp.diff(Qf, Lam).subs(subsd) - muf * (1 - (Qf - 1) * Lam * wf / muf) / T1f))

# ---------------- C. elementary functions ----------------
h = sp.symbols("h", positive=True)
sech = lambda t: 1 / sp.cosh(t)
r = sp.log(1 + sp.exp(-2 * h))


def rexp(e):
    return sp.simplify(sp.expand(e.rewrite(sp.exp)))


rec("C1 (sech tanh)' = 2sech^3 - sech", rexp(sp.diff(sech(h) * sp.tanh(h), h) - (2 * sech(h) ** 3 - sech(h))))
H4 = 2 * h * sp.exp(-h) - 2 * sp.sinh(h) * r - h * sech(h)
g4 = sech(h) - 2 * h * sp.exp(-h) - 2 * sp.cosh(h) * r + h * sp.tanh(h) * sech(h)
rec("C2 H4' = g4", rexp(sp.diff(H4, h) - g4))
g3 = 2 * h * sp.cosh(h) * r + sp.cosh(h) * r**2
J = h**2 * sp.exp(-2 * h) * sech(h) + 2 * h * sp.exp(-h) * r + sp.cosh(h) * r**2
rec("C3 h H4 + g3 = J", rexp(h * H4 + g3 - J))
rec("C4 2e^-h - sech = e^-2h sech", rexp(2 * sp.exp(-h) - sech(h) - sp.exp(-2 * h) * sech(h)))
rec("C5 cosh(1 - tanh) = e^-h", rexp(sp.cosh(h) * (1 - sp.tanh(h)) - sp.exp(-h)))
rec("C6 exp(log cosh h) = exp(h - log2 + r)", rexp(sp.cosh(h) - sp.exp(h - sp.log(2) + r)))
rec("C7a arctan(sinh)' = sech", rexp(sp.diff(sp.atan(sp.sinh(h)), h) - sech(h)))
rec("C7b ((sech tanh + arctan sinh)/2)' = sech^3", rexp(sp.diff((sech(h) * sp.tanh(h) + sp.atan(sp.sinh(h))) / 2, h) - sech(h) ** 3))
rec("C7c (-h sech + arctan sinh)' = h tanh sech", rexp(sp.diff(-h * sech(h) + sp.atan(sp.sinh(h)), h) - h * sp.tanh(h) * sech(h)))
A6 = sp.sinh(h) * r + sp.exp(-h) - 2 * sp.atan(sp.exp(-h))
rec("C7d (sinh r + e^-h - 2 arctan e^-h)' = cosh r", rexp(sp.diff(A6, h) - sp.cosh(h) * r))
x = sp.symbols("x", positive=True)
rec("C7e (x log(1+x^2) - 2x + 2 arctan x)' = log(1+x^2)", sp.diff(x * sp.log(1 + x**2) - 2 * x + 2 * sp.atan(x), x) - sp.log(1 + x**2))
k = sp.symbols("k", positive=True, integer=True)
rec("C8a partial fractions 1/(k(2k+1)^2)", 1 / (k * (2 * k + 1) ** 2) - (1 / k - 2 / (2 * k + 1) - 2 / (2 * k + 1) ** 2))
rec("C8b int_0^oo h e^{-(2k+1)h} = 1/(2k+1)^2", sp.integrate(h * sp.exp(-(2 * k + 1) * h), (h, 0, sp.oo)) - 1 / (2 * k + 1) ** 2)
rec("C8c int_0^oo cosh(h) e^{-2kh} = 2k/(4k^2-1)", sp.integrate(sp.cosh(h).rewrite(sp.exp) * sp.exp(-2 * k * h), (h, 0, sp.oo)) - 2 * k / (4 * k**2 - 1))
G = sp.Catalan
ln2 = sp.log(2)
P1 = 8 - sp.pi**3 / 4                       # 2 int_0^oo h^2 (2e^-h - sech h) dh
P2 = 4 * (ln2 - 2 * (1 - sp.pi / 4) - 2 * (1 - G))  # 4 sum (-1)^{k+1}/(k(2k+1)^2)
I_log = ln2 - 2 + sp.pi / 2                  # int_0^1 log(1+x^2) dx
I_ll = sp.pi / 2 * ln2 - G                   # int_0^1 log(1+x^2)/(1+x^2) dx (classical)
Pa = ln2**2 - 4 * I_log + 4 * I_ll           # int_0^1 log^2(1+x^2) dx  (by parts)
Pb = -ln2**2 + 4 * I_ll                      # int_0^1 log^2(1+x^2)/x^2 dx (by parts)
P3 = Pa + Pb                                 # 2 int_0^oo cosh(h) log^2(1+e^-2h) dh (x = e^-h)
rec("C9 P1 + P2 + P3 = 4 pi log2 - pi^3/4", sp.expand(P1 + P2 + P3 - (4 * sp.pi * ln2 - sp.pi**3 / 4)))


# ---------------- D. Lemma 7.3 (cell-free proof) ----------------
hh = sp.symbols("h", positive=True)
ell = sp.symbols("ell", positive=True)
sh, ch, th = sp.sinh(hh), sp.cosh(hh), sp.tanh(hh)
kf = (1 / ch * th + sp.atan(sh)) / 2
rec("D1a cosh tanh^2 = sinh tanh", rexp(ch * th**2 - sh * th))
rec("D1b cosh sech^4 = sech^3", rexp(ch / ch**4 - 1 / ch**3))
rec("D2a k' = sech^3", rexp(sp.diff(kf, hh) - 1 / ch**3))
rec("D2b k(0) = 0", kf.subs(hh, 0))
rec("D2c k(oo) = pi/4", sp.limit(kf, hh, sp.oo) - sp.pi / 4)
ff = hh * kf
gg = sh * th
rec("D3 f'' = sech^3 (2 - 3 h tanh h)", rexp(sp.diff(ff, hh, 2) - (2 - 3 * hh * th) / ch**3))
rec("D4a g' = sinh (1 + sech^2)", rexp(sp.diff(gg, hh) - sh * (1 + 1 / ch**2)))
rec("D4b g'' = sech^3 (cosh^4 - cosh^2 + 2)", rexp(sp.diff(gg, hh, 2) - (ch**4 - ch**2 + 2) / ch**3))
Dl = 2 - 3 * hh * th - ell * (ch**4 - ch**2 + 2)
rec("D5 Delta_l' = -3 tanh - 3h sech^2 - 2 l sinh cosh (2cosh^2 - 1)",
    rexp(sp.diff(Dl, hh) + 3 * th + 3 * hh / ch**2 + 2 * ell * sh * ch * (2 * ch**2 - 1)))
rec("D6 Delta_1 = -3 h tanh h - cosh^2 sinh^2", rexp(Dl.subs(ell, 1) + 3 * hh * th + ch**2 * sh**2))
phil = ff - ell * gg
rec("D7a phi_l(0) = 0", sp.limit(phil, hh, 0))
rec("D7b phi_l'(0) = 0", sp.limit(sp.diff(phil, hh), hh, 0))
rec("D7c phi_l''(0) = 2(1 - l)", sp.limit(sp.diff(phil, hh, 2), hh, 0) - 2 * (1 - ell))
pr = sp.symbols("p", positive=True)
q, q1, bet, Lm = sp.symbols("q q1 beta Lambda", positive=True)
xi = bet**2 * q**pr / 2
sq = sp.diff(xi, q)
theta = q * sq - xi
Wf = sp.Function("W")
psi = Wf(sq) + sq / 2
rec("D8a theta = (p-1) s q/p", sp.simplify(theta - (pr - 1) * sq * q / pr))
rec("D8b Phi(q) = W(s) - ((p-1)/(2p)) s q", sp.simplify(psi - (sq + theta) / 2 - (Wf(sq) - (pr - 1) / (2 * pr) * sq * q)))
rec("D9a beta^2 = 2 Lambda q^{1-p}/p gives xi'(q) = Lambda", sp.simplify(sq.subs(bet, sp.sqrt(2 * Lm * q ** (1 - pr) / pr)) - Lm))
rec("D9b q xi''(q) = (p-1) xi'(q)", sp.simplify(q * sp.diff(xi, q, 2) - (pr - 1) * sq))
ss = Lm * (q / q1) ** (pr - 1)
rec("D10 log q - log q1 = (log s - log Lambda)/(p-1)", sp.simplify(sp.expand_log(sp.log(q) - sp.log(q1) - (sp.log(ss) - sp.log(Lm)) / (pr - 1), force=True)))

# ---------------- E. Proposition 7.14 (Stein form) and c0 ----------------
hr = sp.symbols("h", real=True)
Lr = sp.log(1 + sp.exp(-2 * hr))
rec("E1 L' = tanh h - 1", rexp(sp.diff(Lr, hr) - (sp.tanh(hr) - 1)))
t, mu_, tau_ = sp.symbols("t mu tau")
dd = lambda F: sp.expand(sp.diff(F, t) * (1 - t**2))  # d/dh of a polynomial in t = tanh h
gpoly = 1 - t**2
comb = sp.expand(dd(dd(gpoly)) + 2 * dd(gpoly) + gpoly)
rec("E2a g'' + 2g' + g = -1 - 4t + 7t^2 + 4t^3 - 6t^4", comb - (-1 - 4 * t + 7 * t**2 + 4 * t**3 - 6 * t**4))
mom = {0: 1, 1: mu_, 2: mu_, 3: tau_, 4: tau_}
Ecomb = sum(comb.coeff(t, kk) * mom[kk] for kk in range(5))
rec("E2b E[g'' + 2g' + g] = -B under the NL moments", sp.expand(Ecomb + (1 - 3 * mu_ + 2 * tau_)))
rec("E3 L(-h) = 2h + L(h)", rexp(sp.exp(sp.log(1 + sp.exp(2 * hr))) - sp.exp(2 * hr + sp.log(1 + sp.exp(-2 * hr)))))
rp = sp.log(1 + sp.exp(-2 * hh))
Jp = hh**2 * sp.exp(-2 * hh) / ch + 2 * hh * sp.exp(-hh) * rp + ch * rp**2
rec("E4 Even(e^h L^2) - h^2 sech h = J on h > 0", rexp((sp.exp(hh) * rp**2 + sp.exp(-hh) * (2 * hh + rp) ** 2) / 2 - hh**2 / ch - Jp))
x = sp.symbols("x", positive=True)
rec("E5 x = e^-h: e^h L(h)^2 |dh/dx| = x^-2 log^2(1+x^2)",
    sp.simplify(((sp.exp(hr) * Lr**2).subs(hr, -sp.log(x)) / x) - sp.log(1 + x**2) ** 2 / x**2))
bt = -sp.log(1 + x**2) ** 2 / x
rec("E6a (-log^2(1+x^2)/x)' = x^-2 log^2(1+x^2) - 4 log(1+x^2)/(1+x^2)",
    sp.simplify(sp.diff(bt, x) - (sp.log(1 + x**2) ** 2 / x**2 - 4 * sp.log(1 + x**2) / (1 + x**2))))
rec("E6b boundary term -> 0 at x -> 0+", sp.limit(bt, x, 0, "+"))
rec("E6c boundary term -> 0 at x -> oo", sp.limit(bt, x, sp.oo))
thv = sp.symbols("theta", positive=True)
rec("E7a (1 + tan^2) cos^2 = 1", sp.simplify((1 + sp.tan(thv) ** 2) * sp.cos(thv) ** 2 - 1))
rec("E7b tan'/(1 + tan^2) = 1", sp.simplify(sp.diff(sp.tan(thv), thv) / (1 + sp.tan(thv) ** 2) - 1))
kk_ = sp.symbols("k", positive=True, integer=True)
rec("E8 int_0^oo h^2 e^{-(2k+1)h} = 2/(2k+1)^3", sp.integrate(hh**2 * sp.exp(-(2 * kk_ + 1) * hh), (hh, 0, sp.oo)) - 2 / (2 * kk_ + 1) ** 3)

# ---------------- N. negative control ----------------
neg = {}


def negrec(name, expr):
    ok = sp.simplify(expr) == 0
    neg[name] = "FAILS as intended" if not ok else "PASSES (negative control broken)"
    if ok:
        fails.append(name)


negrec("N1 A1 with Y/(1+Omega B)", sp.together(kappa - (T1 * X / (T1 + X) + Y / (1 + Om * B))))
negrec("N2 D3 with 2 h tanh h", rexp(sp.diff(ff, hh, 2) - (2 - 2 * hh * th) / ch**3))
negrec("N3 D4b with cosh^4 + cosh^2 + 2", rexp(sp.diff(gg, hh, 2) - (ch**4 + ch**2 + 2) / ch**3))
negrec("N4 E2b with E t^3 = mu", sp.expand(sum(comb.coeff(t, kk) * {0: 1, 1: mu_, 2: mu_, 3: mu_, 4: tau_}[kk] for kk in range(5)) + (1 - 3 * mu_ + 2 * tau_)))
negrec("N5 E4 with h e^-h r in place of 2h e^-h r", rexp((sp.exp(hh) * rp**2 + sp.exp(-hh) * (2 * hh + rp) ** 2) / 2 - hh**2 / ch
                                                          - (hh**2 * sp.exp(-2 * hh) / ch + hh * sp.exp(-hh) * rp + ch * rp**2)))
out["negative control"] = neg

json.dump(out, sys.stdout, indent=1)
print()
npass = sum(1 for k_, v_ in out.items() if k_ != "negative control" and v_.startswith("PASS"))
ntot = len(out) - 1
nneg = sum(1 for v_ in neg.values() if v_.startswith("FAILS"))
print(f"{npass} of {ntot} identities pass; {nneg} of {len(neg)} negative controls fail as intended", file=sys.stderr)
sys.exit(1 if fails else 0)
