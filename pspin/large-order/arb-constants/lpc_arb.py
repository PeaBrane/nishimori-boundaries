#!/usr/bin/env python3
"""Arb recomputation of the numerical inputs of Theorem 7.15 (thm:nm-LP, "Theorem LP"), of the
second certificate for 7 <= p <= 25 ("Theorem LP'", Appendix B.5) and of Lemma 7.3 (lem:nm-U, "Lemma U").

Written separately from ../lp_*.py; it shares no code with them and does not read their output.
Arithmetic: Arb balls (python-flint). Integrals use
acb.integral (Gauss-Legendre with rigorous error bounds on complex ellipses;
integrands with branch cuts check the `analytic` flag) plus explicit analytic
tail bounds beyond h = H. Decimal thresholds are parsed into enclosing balls;
a comparison `x < y` is True only if it holds for every point of both balls.

Checks E13a, T2 and T3 test rounded constants from an earlier version of the argument (the 5-digit
cell bounds 0.95652 / 0.72444 and the reference table LP_TABLE). The paper does not use them and they
fail, so the program exits with status 1. The paper's constants come from E13b, E17c and T4, which pass.

Usage: python lpc_arb.py OUT.json SECTION[,SECTION...]
Sections: C (constants), X (z-quadrature cross-check), E (Lambda>=35 chain),
Q (Q<26 on (0,35]), T (Theorem LP' table p=7..25), U (Lemma U inputs),
D (direct kappa_p from the closed forms of Appendix B.1 at Lambda_c(p))."""
import json
import sys
import time

from flint import acb, arb, ctx

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

RES = {"checks": [], "values": {}, "tables": {}}
NFAIL = 0
T0 = time.time()
NAN = acb("nan")
H = 150


def S(x, n=20):
    return x.str(n, radius=True)


def chk(name, ok, **detail):
    global NFAIL
    ok = bool(ok)
    NFAIL += not ok
    d = {k: (S(v) if isinstance(v, arb) else v) for k, v in detail.items()}
    RES["checks"].append({"name": name, "pass": ok, **d})
    msg = "; ".join(f"{k}={v}" for k, v in d.items())
    print(f"{'PASS' if ok else 'FAIL'} {name}  {msg}", flush=True)


def val(name, x):
    RES["values"][name] = S(x) if isinstance(x, arb) else x
    print(f"  {name} = {RES['values'][name]}", flush=True)


def A(s):
    return arb(s)


def PI():
    return arb.pi()


def LN2():
    return arb(2).log()


def PM1():
    return arb(0, 1)


def eps(L):
    return (-L / 2).exp() / (2 * arb.pi() * L).sqrt()


# ---------------------------------------------------------------- integrands
def rfun(h, an):
    """r(h) = log(1 + e^{-2h}); analytic wherever Re(1 + e^{-2h}) > 0."""
    e2 = (-2 * h).exp()
    if an and not ((1 + e2).real > 0):
        return None
    return e2.log1p()


def base(kind, h, an):
    if kind == "sech":
        return h.sech()
    if kind == "sech3":
        return h.sech() ** 3
    if kind == "hexp":
        return h * (-h).exp()
    if kind == "Bk":
        s = h.sech()
        return 2 * s**3 - s
    if kind == "htanhsech":
        return h * h.tanh() * h.sech()
    if kind == "h2sech3":
        return h * h * h.sech() ** 3
    if kind == "h2sech":
        return h * h * h.sech()
    if kind == "c0a":
        return h * h * (-2 * h).exp() * h.sech()
    r = rfun(h, an)
    if r is None:
        return None
    if kind == "coshr":
        return h.cosh() * r
    if kind == "coshhr":
        return h.cosh() * h * r
    if kind == "coshr2":
        return h.cosh() * r * r
    if kind == "hexpr":
        return h * (-h).exp() * r
    J = h * h * (-2 * h).exp() * h.sech() + 2 * h * (-h).exp() * r + h.cosh() * r * r
    if kind == "J":
        return J
    if kind == "h2J":
        return h * h * J
    if kind == "h4J":
        return (h * h) ** 2 * J
    raise ValueError(kind)


def tail(kind):
    """Upper bound on int_H^inf |g(h)| dh for real h (Gaussian factor <= 1).
    Uses sech<=2e^{-h}, r<=e^{-2h}, cosh*r<=e^{-h}, cosh*r^2<=e^{-3h},
    J<=(2h^2+2h+1)e^{-3h}<=3h^2 e^{-3h} (h>=3), and
    int_H^inf h^n e^{-ch} <= H^n e^{-cH}/(c-n/H)."""
    Hh = arb(H)
    e1, e3 = (-Hh).exp(), (-3 * Hh).exp()
    return {
        "sech": 2 * e1,
        "sech3": 8 * e3 / 3,
        "hexp": (Hh + 1) * e1,
        "Bk": 18 * e1,
        "htanhsech": 2 * (Hh + 1) * e1,
        "h2sech3": 8 * Hh**2 * e3 / (3 - 2 / Hh),
        "h2sech": 2 * Hh**2 * e1 / (1 - 2 / Hh),
        "c0a": 2 * Hh**2 * e3 / (3 - 2 / Hh),
        "coshr": e1,
        "coshhr": (Hh + 1) * e1,
        "coshr2": e3 / 3,
        "hexpr": (Hh + 1) * e3,
        "J": 3 * Hh**2 * e3 / (3 - 2 / Hh),
        "h2J": 3 * Hh**4 * e3 / (3 - 4 / Hh),
        "h4J": 3 * Hh**6 * e3 / (3 - 6 / Hh),
    }[kind]


def integ(kind, L=None, half=False, tolbits=None):
    """I[g] = int_R e^{-h^2/(2L)} g(h) dh for even g (L=None: no Gaussian).
    half=True returns int_0^inf instead."""

    def f(h, an):
        v = base(kind, h, an)
        if v is None:
            return NAN
        if L is not None:
            v = v * (-(h * h) / (2 * L)).exp()
        return v

    tb = tolbits if tolbits is not None else ctx.prec - 20
    tol = arb(2) ** (-tb)
    r = acb.integral(f, 0, H, abs_tol=tol, rel_tol=tol)
    if not r.imag.contains(0):
        raise RuntimeError(f"imag part excludes 0 for {kind}: {r}")
    one = r.real + tail(kind) * PM1()
    return one if half else 2 * one


def Ez(kind, L, Z=14):
    """Direct z-quadrature E f(L + sqrt(L) z), z~N(0,1), with tail bound."""
    sL = L.sqrt()
    c = 1 / (2 * arb.pi()).sqrt()

    def f(z, an):
        x = L + sL * z
        g = (-(z * z) / 2).exp() * c
        if kind.startswith("tanh"):
            return g * x.tanh() ** int(kind[4:])
        lc = x.cosh().log(analytic=an)
        return g * lc if kind == "lc" else g * lc * lc

    tol = arb(2) ** (-(ctx.prec - 20))
    r = acb.integral(f, -Z, Z, abs_tol=tol, rel_tol=tol)
    Zb = arb(Z)
    phiZ = (-Zb * Zb / 2).exp() * c
    if kind.startswith("tanh"):
        t = 2 * phiZ / Zb
    elif kind == "lc":
        t = 2 * (L + sL) * phiZ
    else:
        t = 2 * (L + sL) ** 2 * (Zb + 1 / Zb) * phiZ
    return r.real + t * PM1()


# ------------------------------------------------------------ Gaussian moments
def mom(L, full=False, tolbits=None):
    e = eps(L)
    m = {"eps": e}
    for key, kind in (("u", "sech"), ("w", "sech3"), ("a", "hexp"), ("rho", "coshr")):
        m[key] = e * integ(kind, L, tolbits=tolbits)
    if full:
        m["Ehr"] = e * integ("coshhr", L, tolbits=tolbits)
        m["Er2"] = e * integ("coshr2", L, tolbits=tolbits)
        m["IJ"] = integ("J", L, tolbits=tolbits)
        m["IB"] = integ("Bk", L, tolbits=tolbits)
    return m


def Dp(p, L, m):
    """D_p = W - ((p-1)/(2p)) L mu with W = L/2 + a + rho - log2, mu = 1-u."""
    return L / (2 * p) + m["a"] + m["rho"] - LN2() + arb(p - 1) / (2 * p) * L * m["u"]


def T1_of(L, m):
    return 2 * LN2() - L * m["u"] - 2 * (m["a"] + m["rho"])


# ------------------------------------------------------ elementary bound chain
def T1lo(L):
    return 2 * LN2() - PI() * eps(L) * (L + 2)


def philo(L):
    e = eps(L)
    return 2 * LN2() - PI() * e * (2 * L + 2 + PI() * e)


def LBrem(L):
    e = eps(L)
    return PI() ** 2 * e + (PI() * L + PI() ** 2 * e) ** 2 * e / philo(L)


def LB(L, c0lo, m2hi):
    return c0lo - m2hi / (2 * L) - LBrem(L)


def lamM_lo(L):
    return 1 - PI() / 2 * L**2 * eps(L) / T1lo(L)


def OmB_hi(L):
    return PI() * L * eps(L) / T1lo(L)


def cM_margin(L):
    return 2 * (PI() - 8 / L) - PI() ** 2 * L**2 * eps(L) / T1lo(L)


def UBrem(L):
    x = OmB_hi(L)
    return PI() * L * x / (1 - x)


# ================================================================== sections
CONST = {}


def sec_C():
    ctx.prec = 160
    print("== C: constants", flush=True)
    c0 = integ("J")
    m2 = integ("h2J")
    m4 = integ("h4J")
    K = integ("h2sech3")
    CONST.update(c0=c0, m2=m2, m4=m4, K=K)
    for k, v in CONST.items():
        val(k, v)
    pi, l2, G = PI(), LN2(), arb.const_catalan()
    c0cf = 4 * pi * l2 - pi**3 / 4
    Kcf = pi**3 / 8 - pi
    val("c0_closed_form", c0cf)
    chk("C1 c0 encloses 4 pi log2 - pi^3/4", c0.overlaps(c0cf), c0=c0, cf=c0cf,
        rad=c0.rad())
    chk("C1neg c0 excludes closed form + 1e-40 (overlap test is discriminating)",
        not c0.overlaps(c0cf + A("1e-40")), rad=c0.rad())
    chk("C2 c0 in the rounded cell enclosure [0.95652,0.96103]", (c0 > A("0.95652")) and (c0 < A("0.96103")), c0=c0)
    chk("C3 m2 in the rounded cell enclosure [0.72025,0.72444]", (m2 > A("0.72025")) and (m2 < A("0.72444")), m2=m2)
    chk("C4 m4 in the rounded cell enclosure [2.0200,2.0319] and m4<=2.032",
        (m4 > A("2.0200")) and (m4 < A("2.0319")) and (m4 < A("2.032")), m4=m4)
    chk("C5 float m2=0.7223391..., m4=2.0259419... (digits match)",
        (abs(m2 - A("0.7223391")) < A("1e-7")) and (abs(m4 - A("2.0259419")) < A("1e-7")), m2=m2, m4=m4)
    chk("C6 U1: K <= 0.735819, K encloses pi^3/8-pi", (K < A("0.735819")) and K.overlaps(Kcf), K=K, cf=Kcf)
    # integrals of Appendix B.3
    pairs = [("sech", pi, "int sech = pi"), ("sech3", pi / 2, "int sech^3 = pi/2"),
             ("htanhsech", pi, "int h tanh sech = pi"), ("hexp", arb(2), "int |h|e^{-|h|} = 2"),
             ("coshr", pi - 2, "int cosh*r = pi-2"), ("h2sech", pi**3 / 4, "int h^2 sech = pi^3/4")]
    for kind, target, lab in pairs:
        v = integ(kind)
        chk("C7 " + lab, v.overlaps(target) and v.rad() < A("1e-30"), value=v)
    # pieces of c0 (Appendix B.3)
    v1 = 2 * integ("c0a", half=True)
    v2 = 4 * integ("hexpr", half=True)
    v3 = 2 * integ("coshr2", half=True)
    chk("C8a 2 int_0^inf h^2 e^{-2h} sech = 8 - pi^3/4", v1.overlaps(8 - pi**3 / 4), value=v1)
    chk("C8b 4 int_0^inf h e^{-h} r = 4log2-16+2pi+8G", v2.overlaps(4 * l2 - 16 + 2 * pi + 8 * G), value=v2)
    chk("C8c 2 int_0^inf cosh r^2 = 8-4log2-2pi+4pi log2-8G",
        v3.overlaps(8 - 4 * l2 - 2 * pi + 4 * pi * l2 - 8 * G), value=v3)
    chk("C8d pieces sum to c0", (v1 + v2 + v3).overlaps(c0), sum=v1 + v2 + v3)


def need_const():
    if not CONST:
        sec_C()


def sec_X():
    ctx.prec = 128
    print("== X: z-quadrature vs tilted representation", flush=True)
    for Ls in ("2.34", "10", "36"):
        L = A(Ls)
        m = mom(L, full=True)
        mu_rep = 1 - m["u"]
        tau_rep = 1 - 2 * m["u"] + m["w"]
        W_rep = L / 2 + m["a"] + m["rho"] - LN2()
        V_rep = L - 2 * L * (m["a"] + m["rho"]) + 2 * m["Ehr"] + m["Er2"] - (m["a"] + m["rho"]) ** 2
        t1, t2, t3, t4 = (Ez(f"tanh{k}", L) for k in (1, 2, 3, 4))
        lc, lc2 = Ez("lc", L), Ez("lc2", L)
        W_z = lc - L / 2
        V_z = lc2 - lc**2
        ok = (t1.overlaps(mu_rep) and t2.overlaps(mu_rep) and t3.overlaps(tau_rep) and t4.overlaps(tau_rep)
              and W_z.overlaps(W_rep) and V_z.overlaps(V_rep))
        chk(f"X1 Lambda={Ls}: E tanh, E tanh^2, E tanh^3, E tanh^4, W, V (z-quad) vs representation", ok,
            mu_z=t1, mu_rep=mu_rep, tau_z=t3, tau_rep=tau_rep, W_z=W_z, W_rep=W_rep, V_z=V_z, V_rep=V_rep)
        # Proposition J: X + Y = eps I[J] - (a+rho)^2, and B = eps I[2sech^3-sech]
        X = V_rep - L * mu_rep
        B_rep = 2 * m["w"] - m["u"]
        Y = L**2 * B_rep
        kJ = m["eps"] * m["IJ"] - (m["a"] + m["rho"]) ** 2
        chk(f"X2 Lambda={Ls}: Prop J  X+Y = eps I[J] - (a+rho)^2", (X + Y).overlaps(kJ), XplusY=X + Y, rhs=kJ)
        chk(f"X3 Lambda={Ls}: B = 1-3mu+2tau (z-quad) = eps I[2sech^3-sech] > 0",
            (1 - 3 * t1 + 2 * t3).overlaps(m["eps"] * m["IB"]) and (m["eps"] * m["IB"] > 0),
            B_z=1 - 3 * t1 + 2 * t3, B_rep=m["eps"] * m["IB"])
        # IBP form of B and the Appendix B.3 bounds on the small quantities
        IBP = m["eps"] / L * integ("htanhsech", L)
        chk(f"X4 Lambda={Ls}: eps I[2sech^3-sech] = (eps/Lambda) I[h tanh sech] <= pi eps/Lambda",
            IBP.overlaps(m["eps"] * m["IB"]) and (IBP < PI() * m["eps"] / L), lhs=m["eps"] * m["IB"], rhs=IBP)
        e = m["eps"]
        ok5 = ((m["u"] < PI() * e) and (m["w"] < PI() / 2 * e) and (m["a"] < 2 * e) and (m["rho"] < (PI() - 2) * e)
               and (m["a"] + m["rho"] > e * (PI() - 8 / L)) and (T1_of(L, m) > T1lo(L)) and (T1_of(L, m) < 2 * LN2()))
        chk(f"X5 Lambda={Ls}: u<=pi eps, w<=pi eps/2, a<=2eps, rho<=(pi-2)eps, a+rho>=eps(pi-8/L), "
            "T1 in [2log2-pi eps(L+2), 2log2)", ok5, T1=T1_of(L, m), T1lo=T1lo(L))


def sec_E():
    need_const()
    ctx.prec = 160
    print("== E: Lambda >= 35 chain (Theorem LP)", flush=True)
    L = arb(35)
    e = eps(L)
    val("eps(35)", e)
    chk("E1 eps(35) = 1.6933e-9 (rounded)", abs(e - A("1.6933e-9")) < A("5e-14"), eps35=e)
    d35 = PI() * e * (L + 2)
    chk("E2 delta(35) = pi eps(35) 37 = 1.968e-7 (rounded)", abs(d35 - A("1.968e-7")) < A("5e-11"), delta35=d35)
    t1 = T1lo(L)
    chk("E3 T1 >= 2log2 - delta(35) >= 1.3862941", t1 > A("1.3862941"), T1lo=t1)
    q35 = L / t1
    chk("E4 P2a Q(35) <= 35/T1lo <= 25.2472 < 26", (q35 < A("25.2472")), Qbound=q35)
    wid = L * d35 / t1
    chk("E5 width 35 delta(35)/T1lo <= 4.97e-6 < 5.0e-6", wid < A("4.97e-6"), width=wid)
    mulo = 1 - PI() * e
    lhat = 1 - PI() / 2 * L**2 * e / (mulo * t1)
    chk("E6 P2b lambda_hat >= 1 - (pi/2)L^2 eps/(mu T1) >= 0.9999976", lhat > A("0.9999976"), lhat=lhat)
    p3 = L / (2 * LN2() - PI() * eps(A("34.6")) * (A("34.6") + 2))
    chk("E7 P3 35/(2log2 - delta(34.6)) < 26", p3 < 26, value=p3)
    lm = lamM_lo(L)
    chk("E8 P2c lambda_M >= 1 - 2.4e-6", lm > 1 - A("2.4e-6"), lamM_lo=lm, gap=1 - lm)
    ob = OmB_hi(L)
    chk("E9 P2d/P2e Omega B <= pi L eps/T1 <= 1.35e-7 (also p u)", ob < A("1.35e-7"), OmB_hi=ob)
    pl = philo(L)
    chk("E10 P2f phi_xx >= phi_lo(35) >= 1.38629", pl > A("1.38629"), philo=pl)
    cm = cM_margin(L)
    chk("E11 P2h c_M margin 2(pi-8/L) - pi^2 L^2 eps/T1 > 0 (and x=Lu/T1<=1)",
        (cm > 0) and (ob < 1), margin=cm)
    rem = LBrem(L)
    chk("E12 LB remainder at 35 = 1.48e-5 (rounded)", abs(rem - A("1.48e-5")) < A("5e-8"), rem=rem)
    c0, m2, m4 = CONST["c0"], CONST["m2"], CONST["m4"]
    lb_stated = LB(L, A("0.95652"), A("0.72444"))
    lb_true = LB(L, c0, m2)
    val("LB(35) with stated c0_lo=0.95652, m2_hi=0.72444", lb_stated)
    val("LB(35) with Arb c0, m2", lb_true)
    chk("E13a P2g LB(35) >= 0.94616 using the 5-digit rounded cell bounds 0.95652 / 0.72444 (not used by the paper)",
        lb_stated > A("0.94616"), LB=lb_stated)
    chk("E13b LB(35) >= 0.94616 using Arb-enclosed c0, m2", lb_true > A("0.94616"), LB=lb_true)
    need = A("0.94616") + A("0.72444") / 70 + rem
    val("c0_lo needed for LB(35)>=0.94616 with m2_hi=0.72444", need)
    chk("E13c theorem-statement form LB(35) >= 0.9461 with stated bounds", lb_stated > A("0.9461"), LB=lb_stated)
    ur = UBrem(L)
    chk("E14 UB remainder pi L OmegaB/(1-OmegaB) <= 1.5e-5", ur < A("1.5e-5"), UBrem=ur)
    chk("E15 m4/(8L^2) <= m2/(2L) at L>=35 (i.e. L >= m4/(4 m2))", m4 / (4 * m2) < L, ratio=m4 / (4 * m2))
    # normalization: N = e^{D/2} (1 - D/(2p ln2))^{-1/2}, increasing in D, decreasing in p
    Dl = A("4.97e-6")
    Nmax = (Dl / 2).exp() / (1 - Dl / (2 * 26 * LN2())).sqrt()
    chk("E16 normalization 1 <= N <= 1 + 2.6e-6 (Delta<=4.97e-6, p>=26)", Nmax < 1 + A("2.6e-6"), Nmax_minus_1=Nmax - 1)
    Dl5 = A("5.0e-6")
    Nmax5 = (Dl5 / 2).exp() / (1 - Dl5 / (2 * 26 * LN2())).sqrt()
    val("N-1 at Delta=5.0e-6", Nmax5 - 1)
    up_arb = (c0 + ur) * Nmax
    up_cf = (4 * PI() * LN2() - PI() ** 3 / 4 + ur) * Nmax
    up_cell_upper = (A("0.96103") + ur) * Nmax
    chk("E17 P2i upper (c0 + UBrem) N <= 0.958793 (Arb c0)", up_arb < A("0.958793"), upper=up_arb)
    chk("E17b same with closed-form c0", up_cf < A("0.958793"), upper=up_cf)
    chk("E17c theorem statement upper <= 0.95880", up_arb < A("0.95880"), upper=up_arb)
    val("upper bound if the cell upper bound 0.96103 were used", up_cell_upper)
    chk("E18 theta_p >= -1.5e-5 (= -LBrem(35))", rem < A("1.5e-5"), rem=rem)
    chk("E19 LB(35) > 0 i.e. (N1) for p>=26", lb_stated > 0, LB=lb_stated)
    # remainder (Appendix B.4): N3 margin (2p-2)(1-pu) with pu <= OmB_hi
    chk("E20 (N3) pu <= pi L eps/T1 <= 1.35e-7", ob < A("1.35e-7"), pu_hi=ob)
    # monotonicity spot checks on a grid (supporting the analytic argument)
    prev = None
    mono_ok = True
    for Ls in ["35", "36", "38", "40", "45", "50", "60", "80", "100", "150", "200"]:
        Lg = A(Ls)
        tup = (LB(Lg, c0, m2), lamM_lo(Lg), OmB_hi(Lg), philo(Lg), cM_margin(Lg), UBrem(Lg),
               Lg * PI() * eps(Lg) * (Lg + 2) / T1lo(Lg))
        if prev is not None:
            inc = [tup[0] > prev[0], tup[1] > prev[1], tup[2] < prev[2], tup[3] > prev[3],
                   tup[4] > prev[4], tup[5] < prev[5], tup[6] < prev[6]]
            mono_ok = mono_ok and all(inc)
        prev = tup
    chk("E21 grid spot check: LB, lamM_lo, philo, cM_margin increase; OmB_hi, UBrem, width decrease on [35,200]",
        mono_ok)


def sec_Q():
    ctx.prec = 80
    print("== Q: Q(Lambda) < 26 on (0,35]", flush=True)
    # (0, 0.4]: Q <= 1/(1/2 - 2L/3 - L^2/2), increasing in L
    L0 = arb(2) / 5
    qa = 1 / (arb(1) / 2 - 2 * L0 / 3 - L0**2 / 2)
    chk("Q1 analytic bound on (0,0.4]: Q <= 1/(1/2-2L/3-L^2/2) at 0.4 < 26", qa < 26, bound=qa)
    cells = []
    edges = [arb(2) / 5 + arb(k) / 500 for k in range(0, 301)]  # 0.4..1.0 step 0.002
    edges += [1 + arb(k) / 50 for k in range(1, 201)]  # ..5.0 step 0.02
    edges += [5 + arb(k) / 10 for k in range(1, 301)]  # ..35 step 0.1
    worst = None
    allok = True
    for a_, b_ in zip(edges[:-1], edges[1:]):
        Lb = a_.union(b_)
        m = mom(Lb, tolbits=40)
        t1 = T1_of(Lb, m)
        qb = b_ / t1
        ok = (t1 > 0) and (qb < 26)
        allok = allok and ok
        cells.append([S(a_, 6), S(b_, 6), S(qb, 8), ok])
        if worst is None or float(qb.mid()) > float(worst[1].mid()):
            worst = (S(a_, 6), qb)
    RES["tables"]["Q_cells"] = cells
    chk(f"Q2 cells on [0.4,35] ({len(cells)} cells, ball-Lambda, no monotonicity): b/T1_lo < 26",
        allok, worst_cell=worst[0], worst_bound=worst[1])


# Rounded reference table from an earlier version of the argument, not used by the paper:
# p: (Lambda_p^-, kappa/eps >=, kappa-cert kappa/eps, lambda_M >=, kappa-cert lambda_M, Omega B <=).
LP_TABLE = {7: ("9.4084", "0.1197", "0.8896", "0.8777", "0.9009", "0.0259"),
           8: ("10.9196", "0.4697", "0.9097", "0.9294", "0.9406", "0.0129"),
           9: ("12.3808", "0.6678", "0.9214", "0.9593", "0.9648", "0.0066"),
           10: ("13.807", "0.7818", "0.9286", "0.9766", "0.9794", "0.0034"),
           12: ("16.6144", "0.8869", "0.9364", "0.9924", "0.9932", "9.1e-4"),
           15: ("20.773", "0.9308", "0.9417", "0.9987", "0.9988", "1.3e-4"),
           20: ("27.704", "0.9431", "0.9461", "0.99994", "0.99994", "4.6e-6"),
           25: ("34.635", "0.9461", "0.9486", "0.999997", "0.999997", "1.6e-7")}


def rule_lam_minus(p):
    """Elementary rule: x = p*T1lo(x) fixed point, minus 1e-4, floored to 4 decimals."""
    x = arb(2 * p) * LN2()
    for _ in range(60):
        x = p * T1lo(x)
    xf = float(x.mid()) - 1e-4
    return f"{int(xf * 1e4) / 1e4:.4f}"


def sec_T():
    need_const()
    ctx.prec = 128
    print("== T: Theorem LP' table p=7..25", flush=True)
    c0, m2 = CONST["c0"], CONST["m2"]
    rows = []
    allok = True
    for p in range(7, 26):
        ruled = rule_lam_minus(p)
        Ls = LP_TABLE[p][0] if p in LP_TABLE else ruled
        L = A(Ls)
        m = mom(L)
        D = Dp(p, L, m)
        qel = L / T1lo(L)
        lb_st = LB(L, A("0.95652"), A("0.72444"))
        lb_tr = LB(L, c0, m2)
        lm, ob, pl, cm = lamM_lo(L), OmB_hi(L), philo(L), cM_margin(L)
        n3 = (2 * p - 2) * (1 - ob)
        ok = ((D < 0) and (qel < p) and (lb_st > 0) and (lm > 0) and (ob < 1) and (pl > 0) and (cm > 0)
              and (n3 > 0) and (L > 9) and (c0 < PI() * L))
        allok = allok and ok
        row = {"p": p, "Lminus": Ls, "Lminus_rule": ruled, "Dp": S(D, 6), "Q_elem": S(qel, 10),
               "LB_stated": S(lb_st, 8), "LB_arb": S(lb_tr, 8), "lamM_lo": S(lm, 10),
               "OmB_hi": S(ob, 6), "philo": S(pl, 8), "cM_margin": S(cm, 6), "ok": ok}
        if p in LP_TABLE:
            _, kl, kc, ll, lc, oh = LP_TABLE[p]
            row["table_LB"] = kl
            row["table_LB_le_LBstated"] = bool(A(kl) < lb_st)
            row["table_LB_le_LBarb"] = bool(A(kl) < lb_tr)
            row["table_lamM_le_computed"] = bool(A(ll) < lm)
            row["table_OmB_ge_computed"] = bool(A(oh) > ob)
            row["table_kc"], row["table_lc"] = kc, lc
        rows.append(row)
        print("  ", row, flush=True)
    RES["tables"]["LPprime"] = rows
    chk("T1 Theorem LP' p=7..25: D_p(L^-)<0 (quadrature), L^-/T1lo<p, LB>0, lamM>0, OmB<1, philo>0, "
        "cM margin>0, N3 margin>0", allok)
    bad = [r["p"] for r in rows if "table_LB" in r and not (r["table_lamM_le_computed"] and r["table_OmB_ge_computed"])]
    chk("T2 reference table LP_TABLE: lamM>= and OmB<= columns are valid (directed) roundings (not used by the paper)", not bad, bad_rows=bad)
    bad2 = [r["p"] for r in rows if "table_LB" in r and not r["table_LB_le_LBstated"]]
    chk("T3 reference table LP_TABLE: kappa/eps>= column <= LB with the rounded constants 0.95652/0.72444 (not used by the paper)", not bad2,
        bad_rows=bad2)
    bad3 = [r["p"] for r in rows if "table_LB" in r and not r["table_LB_le_LBarb"]]
    chk("T4 reference table LP_TABLE: kappa/eps>= column <= LB computed with Arb c0, m2", not bad3, bad_rows=bad3)


def e_of(s, m):
    return s * m["w"] / (1 - m["u"])


def sec_U():
    need_const()
    ctx.prec = 80
    print("== U: Lemma U inputs", flush=True)
    Kst = A("0.735819")
    for Kv, lab in ((Kst, "stated K_hi"), (CONST["K"], "Arb K")):
        s1, s2 = A("0.82"), A("1.41")
        q1 = PI() * s1**2 - Kv * s1 - 2 * Kv
        q2 = PI() * s2**2 - (PI() + Kv) * s2 - Kv
        chk(f"U2 ({lab}): pi s^2 - K s - 2K > 0 at 0.82 (U<1/2); pi s^2-(pi+K)s-K > 0 at 1.41 (U<0)",
            (q1 > 0) and (q2 > 0) and (Kv / (2 * PI()) < s1) and ((PI() + Kv) / (2 * PI()) < s2)
            and (PI() * s1 > Kv), margin1=q1, margin2=q2)
    s = arb(1) / 5
    chk("U3 1 - 2s - 2s^2 at s=0.2 equals 0.52 > 1/2", (1 - 2 * s - 2 * s * s) > arb(1) / 2,
        value=1 - 2 * s - 2 * s * s)
    lows = []
    for k in range(242):
        sk = arb(1) / 5 + arb(k) / 200
        sk1 = arb(1) / 5 + arb(k + 1) / 200
        I1 = integ("sech", sk, tolbits=50)
        I3 = integ("sech3", sk, tolbits=50)
        e1 = eps(sk1)
        num = sk * e1 * I3
        lb1 = num / sk1
        lb2 = num / (1 - e1 * I1)
        lb = lb1 if float(lb1.mid()) >= float(lb2.mid()) else lb2
        lows.append((k, lb))
    lo82 = min((lb for k, lb in lows if k <= 123), key=lambda x: float(x.lower().mid()))
    loall = min((lb for k, lb in lows), key=lambda x: float(x.lower().mid()))
    ok82 = all(lb > arb(1) / 2 for k, lb in lows if k <= 123)
    okall = all(lb > arb(1) / 3 for k, lb in lows)
    RES["tables"]["U4"] = [[k, S(lb, 8)] for k, lb in lows]
    chk("U4a 124 cells on [0.2,0.82]: cell lower bound of e > 1/2", ok82, min_lb=lo82)
    chk("U4b 242 cells on [0.2,1.41]: cell lower bound of e > 1/3", okall, min_lb=loall)
    chk("U4c the rounded minima 0.6097 / 0.4792 do not exceed the (tighter) Arb cell minima",
        (A("0.6097") < lo82) and (A("0.4792") < loall), arb82=lo82, arball=loall)
    # crossing brackets and grid endpoints (float claims in 4.2, certified here)
    pts = {}
    for ss in ("0.1", "1", "1.3", "1.4", "6", "8", "30"):
        sv = A(ss)
        m = mom(sv)
        pts[ss] = e_of(sv, m)
        val(f"e({ss})", pts[ss])
    chk("U5 s*_3 in (1.3,1.4): e(1.3) > 1/2 > e(1.4)", (pts["1.3"] > arb(1) / 2) and (pts["1.4"] < arb(1) / 2))
    chk("U6 s*_26 in (6,8): e(6) > 1/25 > e(8)", (pts["6"] > arb(1) / 25) and (pts["8"] < arb(1) / 25))
    chk("U7 float grid endpoints e(0.1)=0.917, e(30)=1.0e-6 (rounded)",
        (abs(pts["0.1"] - A("0.917")) < A("5e-4")) and (abs(pts["30"] - A("1.0e-6")) < A("5e-8")),
        e01=pts["0.1"], e30=pts["30"])
    # s (log e)' + e = 1/2 - s/2 + s I3'/I3 at s = 1; I3' = int h^2/(2 s^2) e^{-h^2/2s} sech^3
    sv = arb(1)
    I3 = integ("sech3", sv)
    I3p = integ("h2sech3", sv) / (2 * sv * sv)
    lhs = arb(1) / 2 - sv / 2 + sv * I3p / I3
    Uv = arb(1) / 2 - sv / 2 + CONST["K"] / (PI() * sv - CONST["K"])
    chk("U8 float claim at s=1: s(log e)'+e = 0.148 <= U(1) = 0.305", (abs(lhs - A("0.148")) < A("5e-4"))
        and (abs(Uv - A("0.305")) < A("5e-4")) and (lhs < Uv), lhs=lhs, U=Uv)


LAM_START = {3: "2.339641575352", 4: "4.478889080851", 5: "6.299601448513", 6: "7.941622996196",
             7: "9.480679087097", 8: "10.95855573891", 9: "12.39955754469", 10: "13.81828132619",
             11: "15.2236128145", 12: "16.62096619263", 13: "18.01361853297", 14: "19.40353235152",
             15: "20.79186855649", 16: "22.17930533119", 17: "23.5662341595", 18: "24.95287856171",
             19: "26.33936490411", 20: "27.72576400205", 21: "29.1121152324", 22: "30.49844034025",
             23: "31.88475126071", 24: "33.27105450904"}
# kappa_p to 13 significant digits from a float evaluation of the closed forms (evidence only).
KAP_FLOAT = {3: "0.03039157247502", 4: "0.01314943454528", 5: "0.005363556846758", 6: "0.002279757450723",
           7: "0.001006925751438", 8: "0.0004574062277787", 9: "0.0002119061191444", 10: "9.952431322906e-5",
           11: "4.719476616699e-5", 12: "2.253476144849e-5", 13: "1.081471976087e-5", 14: "5.210151696676e-6",
           15: "2.517669953497e-6", 16: "1.219585505207e-6", 17: "5.919903902252e-7", 18: "2.878572254845e-7",
           19: "1.401851858264e-7", 20: "6.836176411635e-8", 21: "3.337712972285e-8", 22: "1.631392192499e-8",
           23: "7.98175617619e-9", 24: "3.908700033143e-9"}
FLOAT36 = {3: "1.2429", 5: "1.1327", 8: "0.9775", 10: "0.9511", 14: "0.9427", 20: "0.9461", 25: "0.9486",
           26: "0.9489", 30: "0.9502", 40: "0.9523", 60: "0.9545", 80: "0.9555"}


def lam_c(p, eta_digits):
    L = arb(LAM_START.get(p, str(2 * p * 0.6931471805599453)))
    for _ in range(40):
        m = mom(L)
        D = Dp(p, L, m)
        Dd = ((1 - m["u"]) - (p - 1) * L * m["w"]) / (2 * p)
        step = D / Dd
        L = (L - step).mid()
        if abs(float(step.mid())) < 10.0 ** (-eta_digits - 6):
            break
    eta = arb(10) ** (-eta_digits)
    Llo, Lhi = L - eta, L + eta
    Dlo, Dhi = Dp(p, Llo, mom(Llo)), Dp(p, Lhi, mom(Lhi))
    return Llo.union(Lhi), (Dlo < 0) and (Dhi > 0)


def direct(p, Lb):
    m = mom(Lb, full=True, tolbits=ctx.prec - 40)
    L, u, w, a, rho = Lb, m["u"], m["w"], m["a"], m["rho"]
    mu, tau, v, B = 1 - u, 1 - 2 * u + w, u - w, 2 * w - u
    W = L / 2 + a + rho - LN2()
    V = L - 2 * L * (a + rho) + 2 * m["Ehr"] + m["Er2"] - (a + rho) ** 2
    beta2 = 2 * L * mu ** (1 - p) / p
    beta = beta2.sqrt()
    Om = (p - 1) * L / mu
    lamM = 1 - Om * (1 - 2 * mu + tau)
    c4 = 1 - 4 * mu + 3 * tau
    phimm = Om * ((1 - mu) * Om - 1)
    phimq = -Om**2 * (mu - tau)
    phiqq = Om / 2 * (1 - Om * c4)
    phimb = Om * L / beta * B
    detH = Om**2 / 2 * (((1 - mu) * Om - 1) * (1 - c4 * Om) - 2 * Om**2 * (mu - tau) ** 2)
    AFM = mu**p / 2 + (L / beta) ** 2 * B + phimb**2 * (phimm + 2 * phimq + phiqq) / detH
    AFM2 = mu**p / 2 - (L / beta) ** 2 * Om**2 * B * lamM / (2 * detH)
    phixb = beta * mu**p / 2
    phixx = V - (p - 1) * beta2 * mu**p / 2
    ASG = phixb**2 / phixx
    kap = beta2 * (AFM - ASG)
    T1 = L * mu - 2 * W
    X, Y, OmB = V - L * mu, L**2 * B, Om * B
    kapA1 = T1 * X / (T1 + X) + Y / (1 - OmB)
    kapJ = m["eps"] * m["IJ"] - (a + rho) ** 2
    detS = Om / 2 * (lamM * B + 2 * v)
    Nn = arb(2) ** p * (4 * PI() * p * LN2()).sqrt()
    n3 = p * mu ** (p - 1) + (p - 1) * mu**p - 1
    return dict(L=L, mu=mu, tau=tau, V=V, lamM=lamM, B=B, OmB=OmB, detH=detH, phixx=phixx, AFM=AFM, AFM2=AFM2,
                ASG=ASG, kap=kap, kapA1=kapA1, X=X, Y=Y, kapJ=kapJ, T1=T1, T1zero=L * mu / p, detS=detS,
                kape=kap / m["eps"], nk=Nn * kap, cM=beta2 / 2, n3=n3, gap=2 * p * LN2() - L, eps=m["eps"])


def sec_D(plist):
    ctx.prec = 288
    print("== D: direct kappa_p at certified Lambda_c(p)", flush=True)
    rows = []
    for p in plist:
        t = time.time()
        Lb, okroot = lam_c(p, 60)
        d = direct(p, Lb)
        ok_id = (d["AFM"].overlaps(d["AFM2"]) and d["kap"].overlaps(d["kapA1"])
                 and (d["X"] + d["Y"]).overlaps(d["kapJ"]) and d["T1"].overlaps(d["T1zero"]))
        row = {"p": p, "root_bracket_ok": bool(okroot), "identities_ok": bool(ok_id), "Lambda_c": S(d["L"], 30),
               "kappa": S(d["kap"], 20), "kappa_over_eps": S(d["kape"], 12), "normalized": S(d["nk"], 12),
               "lambda_M": S(d["lamM"], 12), "phixx": S(d["phixx"], 12), "OmB": S(d["OmB"], 8),
               "detS_pos": bool(d["detS"] > 0), "detH": S(d["detH"], 12), "B_pos": bool(d["B"] > 0),
               "cM": S(d["cM"], 12), "cM_lt_2ln2": bool(d["cM"] < 2 * LN2()), "gap_2pln2_minus_L": S(d["gap"], 8),
               "sec": round(time.time() - t, 1)}
        if p in KAP_FLOAT:
            kd = A(KAP_FLOAT[p])
            row["rel_diff_vs_float_table"] = S((d["kap"] - kd) / kd, 4)
            row["float_table_match_13sig"] = bool(abs((d["kap"] - kd) / kd) < A("1e-11"))
        if p in LP_TABLE:
            row["table_kc"], row["table_lc"] = LP_TABLE[p][2], LP_TABLE[p][4]
            row["table_kc_match"] = bool(abs(d["kape"] - A(LP_TABLE[p][2])) < A("5.001e-5"))
            row["table_lc_match"] = bool(abs(d["lamM"] - A(LP_TABLE[p][4])) < 5 * arb(10) ** (-len(LP_TABLE[p][4]) + 1))
        if p in FLOAT36:
            row["float36"] = FLOAT36[p]
            row["float36_match"] = bool(abs(d["nk"] - A(FLOAT36[p])) < A("5.001e-5"))
        if p >= 26:
            row["LP_a"] = bool((d["gap"] > 0) and (d["gap"] < A("5.0e-6")))
            row["LP_b"] = bool((d["lamM"] > 1 - A("2.4e-6")) and (d["OmB"] > 0) and (d["OmB"] < A("1.35e-7")))
            row["LP_c"] = bool(d["n3"] > (2 * p - 2) * (1 - A("1.35e-7")))
            row["LP_d"] = bool((d["nk"] > A("0.94616")) and (d["nk"] < A("0.958793")))
            row["LP_e"] = bool((d["phixx"] > A("1.38629")) and (d["cM"] < 2 * LN2()))
            th = d["kape"] - CONST_C0_M2(d["L"])
            row["theta_p"] = S(th, 6)
            row["theta_window"] = bool((th > -A("1.5e-5")) and (th < CONST["m4"] / (8 * d["L"] ** 2) + A("1.5e-5")))
        if p <= 6:
            row["LB_at_Lambda_c"] = S(LB(d["L"], CONST["c0"], CONST["m2"]), 6)
            row["philo_at_Lambda_c"] = S(philo(d["L"]), 6)
            row["elementary_fails"] = bool((philo(d["L"]) < 0) or (LB(d["L"], CONST["c0"], CONST["m2"]) < 0))
            row["eps_Lambda_c"] = S(d["eps"], 6)
        rows.append(row)
        print("  ", row, flush=True)
        RES["tables"]["direct"] = rows
    allroot = all(r["root_bracket_ok"] for r in rows)
    allid = all(r["identities_ok"] for r in rows)
    chk("D1 Lambda_c(p) bracketed (D_p(L*-1e-60)<0<D_p(L*+1e-60)) for all p computed", allroot)
    chk("D2 A_FM two forms, identity A1, Proposition J, T1=Lambda mu/p agree at Lambda_c for all p", allid)
    der = [r for r in rows if "float_table_match_13sig" in r]
    chk("D3 kappa_p matches the float table KAP_FLOAT (13 sig. digits) p=3..24",
        all(r["float_table_match_13sig"] for r in der), n=len(der))
    t2 = [r for r in rows if "table_kc_match" in r]
    chk("D4 LP_TABLE kappa-cert columns (kappa/eps, lambda_M) match Arb values to printed digits",
        all(r["table_kc_match"] and r["table_lc_match"] for r in t2),
        bad=[r["p"] for r in t2 if not (r["table_kc_match"] and r["table_lc_match"])])
    f36 = [r for r in rows if "float36_match" in r]
    chk("D5 float table FLOAT36 of normalized kappa_p matches to printed digits",
        all(r["float36_match"] for r in f36), bad=[r["p"] for r in f36 if not r["float36_match"]])
    big = [r for r in rows if r["p"] >= 26]
    if big:
        chk("D6 Theorem LP (a)-(e) hold at every sampled p>=26 (direct Arb values)",
            all(r["LP_a"] and r["LP_b"] and r["LP_c"] and r["LP_d"] and r["LP_e"] for r in big),
            ps=[r["p"] for r in big])
        chk("D7 asymptotics: -1.5e-5 <= theta_p <= m4/(8 Lc^2) + 1.5e-5 at every sampled p>=26",
            all(r["theta_window"] for r in big))
    small = [r for r in rows if r["p"] <= 6]
    if small:
        chk("D8 elementary bound fails at Lambda_c(p) for p<=6 (phi_lo<0 or LB<0) and eps(Lambda_c) >= 2.6e-3 "
            "(Appendix B.5)",
            all(r["elementary_fails"] for r in small)
            and all(A(r["eps_Lambda_c"].split("+/-")[0].strip("[ ")) > A("2.6e-3") for r in small),
            LB=[r["LB_at_Lambda_c"] for r in small], philo=[r["philo_at_Lambda_c"] for r in small], eps=[r["eps_Lambda_c"] for r in small])


def CONST_C0_M2(L):
    return CONST["c0"] - CONST["m2"] / (2 * L)


def main():
    out = sys.argv[1]
    secs = sys.argv[2].split(",") if len(sys.argv) > 2 else list("CXEQTUD")
    plist = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else \
        list(range(3, 26)) + [26, 27, 28, 30, 35, 40, 50, 60, 80]
    RES["meta"] = {"flint": __import__("flint").__version__, "sections": secs, "H": H}
    for s in secs:
        t = time.time()
        {"C": sec_C, "X": sec_X, "E": sec_E, "Q": sec_Q, "T": sec_T, "U": sec_U}.get(s, lambda: None)()
        if s == "D":
            need_const()
            sec_D(plist)
        RES["meta"][f"time_{s}"] = round(time.time() - t, 1)
        RES["meta"]["nfail"] = NFAIL
        with open(out, "w") as fh:
            json.dump(RES, fh, indent=1)
    RES["meta"]["total_time"] = round(time.time() - T0, 1)
    with open(out, "w") as fh:
        json.dump(RES, fh, indent=1)
    print(f"done: {len(RES['checks'])} checks, {NFAIL} failed, {RES['meta']['total_time']} s", flush=True)
    sys.exit(1 if NFAIL else 0)


if __name__ == "__main__":
    main()
