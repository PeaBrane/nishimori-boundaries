#!/usr/bin/env python3
"""Interval-arithmetic (Arb, python-flint) certificate for the second-order coefficients at the
triple point M of the fully connected p-spin glass with a Curie-Weiss bias, p = 3..25.

It shares no code with the float evaluations in inputs/.  Every number printed or
stored by this script is a rigorous Arb ball: the true value lies inside it, given
  (i)  the correctness of Arb / FLINT (acb_calc_integrate, elementary functions, ball arithmetic)
       and of the python-flint bindings, and
  (ii) the mathematical definitions coded below (see README.md for the trust base).

Two independent codings are certified side by side.
  form1  the closed forms of Section 7 and Appendix B.1 of the paper, incl. the simplified A_FM (lambda_M form);
  form2  Nishimori arXiv:2608.23904 App. A3-A4 as printed ((A23),(A24),(A43),(A47)-(A49),
         (A52)-(A63), (23), (51)-(55)), with ln 2cosh coded as h + log(1 + e^{-2h});
  raw    Nishimori's chain-rule expressions *before* the Nishimori-line (NL) reductions (A51):
         every Gaussian average that the NL identity would collapse (E sech^2, E sech^2 tanh,
         E(1-4t^2+3t^4), E tanh^2, E sech^4, E[tanh - tanh^2/2]) is integrated separately.

Lambda_c (the positive root of D_p(L) = E log cosh h - L/2 - (p-1) L mu(L)/(2p), h = L + sqrt(L) z)
is certified by:
  * a global sign scan: D_p < 0 on [a0, U_lo], D_p > 0 on [U_hi, Lmax] (adaptive mean-value
    enclosures), D_p(U_lo) < 0 < D_p(U_hi) and D_p' > 0 on U = [U_lo, U_hi];
    together with the proved lemmas L2 (D_p < 0 on (0, a0]) and L3 (D_p > 0 on [2p ln2, inf))
    this gives exactly one zero on (0, inf);
  * a Krawczyk (interval Newton) test K(X) in int(X) on a tight box X in U, for form1 and form2
    separately (form2 uses its own residual and an NL-free derivative).

Usage:  python kappa_cert.py [--pmin 3] [--pmax 25] [--out kappa_cert.json]
                            [--prec 256] [--tol-bits 230] [--zcut 22]
"""

import argparse
import json
import platform
import sys
import time
from fractions import Fraction

import flint
from flint import acb, arb, ctx

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

PREC = 256          # working precision of the tight phase (bits)
TOL_BITS = 230      # abs/rel tolerance of acb.integral in the tight phase
ZCUT = 22           # z-truncation |z| <= ZCUT in the tight phase
SCAN_PREC = 96      # precision of the global sign scan
SCAN_TOL_BITS = 64
SCAN_ZCUT = 12
KRAW_RAD_EXP = -200  # Krawczyk box radius 2^-200
A0 = Fraction(3, 16)              # lemma L2 threshold
LN2_UP = Fraction(355, 512)       # dyadic upper bound of ln 2 (checked in-script)
U_HALF = Fraction(1, 64)          # half-width of the monotonicity bracket U

LOG = sys.stderr


def log(*a):
    print(*a, file=LOG, flush=True)


# ---------------------------------------------------------------------------------------------
# exact dyadic helpers
# ---------------------------------------------------------------------------------------------
def fr2arb(f):
    f = Fraction(f)
    n, d = f.numerator, f.denominator
    k = d.bit_length() - 1
    if d != 1 << k:
        raise ValueError(f"non-dyadic {f}")
    return arb(n) / arb(1 << k)  # exact


def interval_ball(lo, hi):
    """Ball containing the closed interval [lo, hi] (dyadic Fractions)."""
    return arb(fr2arb((lo + hi) / 2), fr2arb((hi - lo) / 2))


def s(x, n=64):
    return x.str(n, radius=True)


# ---------------------------------------------------------------------------------------------
# Gaussian expectations E f(L + sqrt(L) z), z ~ N(0,1), valid uniformly for all L in the ball
# ---------------------------------------------------------------------------------------------
def gexp(f, growth, lam, zcut, tol_bits):
    """Rigorous enclosure of E f(h), h = L + sqrt(L) z, for every L in the ball `lam` (lam > 0).

    growth = (a, b, c): |f(h)| <= a + b|h| + c h^2 for real h (used for the tail |z| > zcut).
    The integral over [-zcut, zcut] is Arb's rigorous Gauss-Legendre integration; the integrand
    closes over the ball lam, so each evaluation encloses f for every L in lam and the returned
    enclosure is uniform in L.
    """
    if not lam > 0:
        raise ValueError("lam must be > 0")
    lc, sc = acb(lam), acb(lam.sqrt())

    def g(z, analytic):
        return f(lc + sc * z, analytic) * (-(z * z) / 2).exp()

    tol = arb(2) ** (-tol_bits)
    I = acb.integral(g, -zcut, zcut, abs_tol=tol, rel_tol=tol)
    if not (I.real.is_finite() and I.imag.is_finite()):
        raise ArithmeticError("non-finite integral")
    sqrt2pi = (2 * arb.pi()).sqrt()
    val = I.real / sqrt2pi
    # tail: P(|z|>Z) = erfc(Z/sqrt2), E[|z|;|z|>Z] = 2 phi(Z), E[z^2;|z|>Z] = P + 2 Z phi(Z)
    a, b, c = growth
    Z = arb(zcut)
    P = (Z / arb(2).sqrt()).erfc()
    phiZ = (-(Z * Z) / 2).exp() / sqrt2pi
    L = lam.upper()
    T = a * P + b * (L * P + L.sqrt() * 2 * phiZ) + c * (2 * L * L * P + 2 * L * (P + 2 * Z * phiZ))
    Tu = T.upper()
    return val + (-Tu).union(Tu)


def ln2():
    return arb(2).log()


# integrands f(h, analytic) and tail growth constants (a, b, c)
def _t(h):
    return h.tanh()


INTEGRANDS = {
    # bounded by 1 on the real line
    "tanh": (lambda h, a: h.tanh(), (1, 0, 0)),
    "tanh2": (lambda h, a: h.tanh() ** 2, (1, 0, 0)),
    "tanh3": (lambda h, a: h.tanh() ** 3, (1, 0, 0)),
    "tanh4": (lambda h, a: h.tanh() ** 4, (1, 0, 0)),
    "sech2": (lambda h, a: h.sech() ** 2, (1, 0, 0)),
    "sech2tanh": (lambda h, a: h.sech() ** 2 * h.tanh(), (1, 0, 0)),
    "sech4": (lambda h, a: h.sech() ** 4, (1, 0, 0)),
    # 1 - 4t^2 + 3t^4 = (1 - t^2)(1 - 3t^2) in [-1/3, 1]
    "q4": (lambda h, a: (1 - _t(h) ** 2) * (1 - 3 * _t(h) ** 2), (1, 0, 0)),
    # W'-integrand (NL-free): tanh - tanh^2/2 in [-3/2, 1/2]
    "wprime": (lambda h, a: _t(h) - _t(h) ** 2 / 2, (Fraction(3, 2), 0, 0)),
    # mu'-integrand (NL-free): sech^2 (1 - tanh) in [0, 2]
    "muprime": (lambda h, a: h.sech() ** 2 * (1 - _t(h)), (2, 0, 0)),
    # 0 <= log cosh h <= |h|;  (log cosh h)^2 <= h^2
    "logcosh": (lambda h, a: h.cosh().log(analytic=a), (0, 1, 0)),
    "logcosh2": (lambda h, a: h.cosh().log(analytic=a) ** 2, (0, 0, 1)),
    # r(h) = log(1 + e^{-2h}), |r| <= ln2 + 2|h|;  (h + r)^2 = (ln 2cosh h)^2 <= 2h^2 + 2 ln^2 2
    "r": (lambda h, a: (1 + (-2 * h).exp()).log(analytic=a), ("ln2", 2, 0)),
    "ln2cosh2": (lambda h, a: (h + (1 + (-2 * h).exp()).log(analytic=a)) ** 2, ("2ln2sq", 0, 2)),
}


def E(name, lam, scan=False):
    f, (a, b, c) = INTEGRANDS[name]
    conv = {"ln2": ln2(), "2ln2sq": 2 * ln2() ** 2}
    a = conv[a] if isinstance(a, str) else arb(fr2arb(a)) if isinstance(a, Fraction) else arb(a)
    growth = (a, arb(b), arb(c))
    if scan:
        return gexp(f, growth, lam, SCAN_ZCUT, SCAN_TOL_BITS)
    return gexp(f, growth, lam, ZCUT, TOL_BITS)


# ---------------------------------------------------------------------------------------------
# the triple-point residual and its derivatives
# ---------------------------------------------------------------------------------------------
def cp(p):
    return arb(p - 1) / (2 * p)


def D1(p, lam, scan=False):
    """D_p(L) = E log cosh h - L/2 - (p-1) L mu/(2p) (Section 7.1, sec:nm-triple)."""
    return E("logcosh", lam, scan) - lam / 2 - cp(p) * lam * E("tanh", lam, scan)


def D1p(p, lam, scan=False):
    """D_p'(L) = mu/(2p) - (p-1) L (1 - 2mu + tau)/(2p)  (uses the NL identity, Lemma 7.1 lem:nm-NLid)."""
    mu, tau = E("tanh", lam, scan), E("tanh3", lam, scan)
    return mu / (2 * p) - cp(p) * lam * (1 - 2 * mu + tau)


def D2(p, lam, scan=False):
    """Nishimori (21)/(53): W(L) - (p-1) L mu/(2p), W = E ln 2cosh h - ln2 - L/2, ln 2cosh h = h + r(h)."""
    W = lam + E("r", lam, scan) - ln2() - lam / 2
    return W - cp(p) * lam * E("tanh", lam, scan)


def Dp_nlfree(p, lam, scan=False):
    """D_p' without the NL identity (Stein only): W' = E[tanh - tanh^2/2], mu' = E[sech^2 (1 - tanh)]."""
    return E("wprime", lam, scan) - cp(p) * (E("tanh", lam, scan) + lam * E("muprime", lam, scan))


# ---------------------------------------------------------------------------------------------
# Lambda_c: global scan, monotonicity bracket, Krawczyk
# ---------------------------------------------------------------------------------------------
def approx_root(p):
    """Non-rigorous: bisection on the sign of D1 at points (scan precision), then used as a centre."""
    lo, hi = A0, 2 * p * LN2_UP
    dlo = D1(p, fr2arb(lo), True)
    dhi = D1(p, fr2arb(hi), True)
    if not (dlo < 0 and dhi > 0):
        raise RuntimeError(f"p={p}: no certified sign change on [a0, Lmax]")
    for _ in range(36):
        m = (lo + hi) / 2
        dm = D1(p, fr2arb(m), True)
        if dm < 0:
            lo = m
        elif dm > 0:
            hi = m
        else:
            break
    return (lo + hi) / 2


def mv_enclosure(p, lo, hi):
    """Mean-value enclosure of {D_p(L): L in [lo, hi]}: D(m) + D'([lo,hi]) ([lo,hi] - m).
    Returns None if an integral over the ball [lo, hi] is not finite (then the caller bisects)."""
    m = (lo + hi) / 2
    X = interval_ball(lo, hi)
    try:
        return D1(p, fr2arb(m), True) + Dp_nlfree(p, X, True) * (X - fr2arb(m))
    except ArithmeticError:
        return None


def presplit(lo, hi, w=Fraction(1, 4)):
    out, a = [], lo
    while a < hi:
        b = min(a + w, hi)
        out.append((a, b))
        a = b
    return out


def sign_scan(p, lo, hi, max_pieces=20000):
    """Cover [lo, hi] by subintervals on which D_p has a certified constant sign."""
    stack = presplit(lo, hi)[::-1]
    pieces = []
    while stack:
        a, b = stack.pop()
        enc = mv_enclosure(p, a, b)
        if enc is not None and (enc < 0 or enc > 0):
            pieces.append((a, b, -1 if enc < 0 else 1, enc))
            continue
        if b - a < Fraction(1, 1 << 30) or len(pieces) > max_pieces:
            raise RuntimeError(f"p={p}: sign scan failed on [{float(a)}, {float(b)}]")
        m = (a + b) / 2
        stack.append((m, b))
        stack.append((a, m))
    pieces.sort()
    return pieces


def monotone_on(p, lo, hi, n0=8):
    """Certify D_p' > 0 on [lo, hi] (NL-free derivative, adaptive subdivision). Returns min lower bound."""
    stack = [(lo + (hi - lo) * Fraction(k, n0), lo + (hi - lo) * Fraction(k + 1, n0)) for k in range(n0)]
    lows = []
    while stack:
        a, b = stack.pop()
        try:
            d = Dp_nlfree(p, interval_ball(a, b), True)
        except ArithmeticError:
            d = None
        if d is not None and d > 0:
            lows.append(d.lower())
            continue
        if b - a < Fraction(1, 1 << 24):
            raise RuntimeError(f"p={p}: monotonicity failed near {float(a)}")
        m = (a + b) / 2
        stack += [(a, m), (m, b)]
    return min(lows, key=lambda x: float(x.mid())), len(lows)


def newton_centre(p, Dfun, Dpfun, x0, iters=12):
    """Non-rigorous Newton on exact midpoints (tight precision) to centre the Krawczyk box."""
    x = fr2arb(x0)
    for _ in range(iters):
        x = (x - Dfun(p, x) / Dpfun(p, x)).mid()
    return x


def krawczyk(p, Dfun, Dpfun, centre):
    """1-D Krawczyk: K(X) = m - y D(m) + (1 - y D'(X))(X - m); K in int X => unique zero in X."""
    r = arb(2) ** KRAW_RAD_EXP
    m = centre
    X = arb(m, r)
    Dm = Dfun(p, m)
    y = (1 / Dpfun(p, m)).mid()
    DpX = Dpfun(p, X)
    K = m - y * Dm + (1 - y * DpX) * (X - m)
    inside = bool(K.lower() > X.lower() and K.upper() < X.upper())
    return inside, K, X, Dm, DpX


def lambda_c(p):
    info = {}
    ctx.prec = SCAN_PREC
    t0 = time.time()
    # lemma L2 applicability: a0 (1 + 2 a0/3) < (p-2)/(2(p-1))  (exact rationals)
    info["lemma_L2_condition"] = bool(A0 * (1 + 2 * A0 / 3) < Fraction(p - 2, 2 * (p - 1)))
    # lemma L3 applicability: Lmax = 2p * 355/512 >= 2p ln2  (Arb)
    Lmax = 2 * p * LN2_UP
    info["lemma_L3_condition"] = bool(fr2arb(Lmax) - 2 * p * ln2() > 0)
    approx = approx_root(p)
    c = Fraction(round(approx * (1 << 20)), 1 << 20)
    Ulo, Uhi = c - U_HALF, c + U_HALF
    d_lo, d_hi = D1(p, fr2arb(Ulo), True), D1(p, fr2arb(Uhi), True)
    dmin, n_mono = monotone_on(p, Ulo, Uhi)
    info["U"] = [str(Ulo), str(Uhi)]
    info["U_float"] = [float(Ulo), float(Uhi)]
    info["D_at_U_lo"] = s(d_lo, 12)
    info["D_at_U_hi"] = s(d_hi, 12)
    info["U_sign_change"] = bool(d_lo < 0 and d_hi > 0)
    info["Dprime_on_U_min_lower_bound"] = s(dmin, 8)
    info["Dprime_on_U_positive"] = bool(dmin > 0)
    info["U_monotone_pieces"] = n_mono
    left = sign_scan(p, A0, Ulo)
    right = sign_scan(p, Uhi, Lmax) if Uhi < Lmax else []
    info["scan_left"] = {"interval": [str(A0), str(Ulo)], "pieces": len(left),
                         "all_negative": all(sg < 0 for _, _, sg, _ in left),
                         "covers": bool(left and left[0][0] == A0 and left[-1][1] == Ulo and
                                        all(left[i][1] == left[i + 1][0] for i in range(len(left) - 1))),
                         "min_margin": s(max((e.upper() for *_, e in left), key=lambda x: float(x.mid())), 6)}
    info["scan_right"] = {"interval": [str(Uhi), str(Lmax)], "pieces": len(right),
                          "all_positive": all(sg > 0 for _, _, sg, _ in right),
                          "covers": bool((not right and Uhi >= Lmax) or (right and right[0][0] == Uhi and right[-1][1] == Lmax and
                                         all(right[i][1] == right[i + 1][0] for i in range(len(right) - 1)))),
                          "min_margin": s(min((e.lower() for *_, e in right), key=lambda x: float(x.mid())), 6) if right else None}
    info["unique_zero_on_(0,inf)"] = bool(
        info["lemma_L2_condition"] and info["lemma_L3_condition"] and info["U_sign_change"]
        and info["Dprime_on_U_positive"] and info["scan_left"]["all_negative"] and info["scan_left"]["covers"]
        and info["scan_right"]["all_positive"] and info["scan_right"]["covers"])
    info["scan_seconds"] = round(time.time() - t0, 2)

    ctx.prec = PREC
    out = {}
    for tag, Dfun, Dpfun in (("form1", D1, D1p), ("form2", D2, Dp_nlfree)):
        centre = newton_centre(p, Dfun, Dpfun, c)
        ok, K, X, Dm, DpX = krawczyk(p, Dfun, Dpfun, centre)
        in_U = bool(X.lower() > fr2arb(Ulo) and X.upper() < fr2arb(Uhi))
        out[tag] = K
        info[f"krawczyk_{tag}"] = {"K_in_int_X": ok, "X_in_U": in_U, "X_radius": f"2^{KRAW_RAD_EXP}",
                                   "D(m)": s(Dm, 6), "D'(X)": s(DpX, 20), "Lambda_c": s(K, 60)}
    info["form1_form2_Lambda_c_overlap"] = bool(out["form1"].overlaps(out["form2"]))
    return out["form1"], out["form2"], info


# ---------------------------------------------------------------------------------------------
# second-order coefficients
# ---------------------------------------------------------------------------------------------
def form1(p, L):
    """Closed forms of Section 7 and Appendix B.1 (NL identity used)."""
    mu, tau, E1, E2 = E("tanh", L), E("tanh3", L), E("logcosh", L), E("logcosh2", L)
    V = E2 - E1 ** 2
    W = E1 - L / 2
    mup = mu ** p
    bc = (2 * L / (p * mu ** (p - 1))).sqrt()
    Om = arb(p * (p - 1)) * bc ** 2 * mu ** (p - 2) / 2
    lamM = 1 - Om * (1 - 2 * mu + tau)
    B = 1 - 3 * mu + 2 * tau
    fmm = Om * ((1 - mu) * Om - 1)
    fmq = -(Om ** 2) * (mu - tau)
    fqq = Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))
    detH = fmm * fqq - fmq ** 2
    detH_bracket = Om ** 2 / 2 * (((1 - mu) * Om - 1) * (1 - (1 - 4 * mu + 3 * tau) * Om) - 2 * Om ** 2 * (mu - tau) ** 2)
    fmb = Om * L * B / bc
    fbb = arb(1) / 2 - mup / 2 - (L / bc) ** 2 * B
    hsum = fmm + 2 * fmq + fqq
    vHv = fmb ** 2 * hsum / detH
    A_FM = arb(1) / 2 - fbb + vHv
    A_FM_alt = mup / 2 - (L / bc) ** 2 * Om ** 2 * B * lamM / (2 * detH)
    fxb = bc * mup / 2
    fxx = V - arb(p - 1) * bc ** 2 * mup / 2
    fxx_alt = V - 2 * W
    A_SG = fxb ** 2 / fxx
    diff = A_FM - A_SG
    kappa = bc ** 2 * diff
    Kb = diff / (2 * bc * mup)
    K = bc ** 4 * Kb
    return {
        "Lambda_c": L, "mu": mu, "tau": tau, "E1": E1, "W": W, "V": V, "beta_c": bc, "T_c": 1 / bc,
        "j0M": bc / 2, "Omega": Om, "Omega_alt": arb(p - 1) * L / mu, "lambda_M": lamM, "B": B,
        "phi_mm": fmm, "phi_mq": fmq, "phi_qq": fqq, "detH": detH, "detH_bracket": detH_bracket,
        "phi_mb": fmb, "phi_bb": fbb, "hsum": hsum, "minus_Omega_lamM_over_2": -Om * lamM / 2, "vHv": vHv,
        "A_FM": A_FM, "A_FM_alt": A_FM_alt, "phi_xb": fxb, "phi_xx": fxx, "phi_xx_alt": fxx_alt,
        "A_SG": A_SG, "A_FM_minus_A_SG": diff, "kappa": kappa, "Kb": Kb, "K": K,
        "K_alt": bc * kappa / (2 * mup), "C_SG": bc ** 2 * (arb(1) / 2 - A_SG), "C_FM": bc ** 2 * (arb(1) / 2 - A_FM),
        "G_qq": Om / 2 * lamM, "Dprime_at_Lc": D1p(p, L), "mu_lamM_over_2p": mu * lamM / (2 * p),
        "D_at_Lc": D1(p, L), "detJ": -Om / 2 * lamM * fxx, "F_delta": bc * mup, "dx_dbeta": -fxb / fxx,
        "one_minus_mu": 1 - mu, "mu_minus_tau": mu - tau, "beta_c_over_sqrt2": bc / arb(2).sqrt(),
    }


def form2(p, L):
    """Nishimori App. A3-A4 as printed, plus the NL-free ('raw') chain-rule versions."""
    mu, tau = E("tanh", L), E("tanh3", L)
    Er, Er2 = E("r", L), E("ln2cosh2", L)
    L2c = L + Er                        # E ln 2cosh h
    W = L2c - ln2() - L / 2             # (A24)
    V = Er2 - L2c ** 2                  # (43)
    mup = mu ** p
    bc2 = 2 * L / (p * mu ** (p - 1))   # (55)
    bc = bc2.sqrt()
    j0M = bc / 2
    Om = arb(p * (p - 1)) * bc2 * mu ** (p - 2) / 2          # (23)
    Om_A55 = arb(p * (p - 1)) * bc * j0M * mu ** (p - 2)     # (A55)
    lam = 1 - Om * (1 - 2 * mu + tau)                        # (23)
    B = 1 - 3 * mu + 2 * tau                                 # (44)
    fmm = Om * ((1 - mu) * Om - 1)                           # (A52)
    fmq = -(Om ** 2) * (mu - tau)                            # (A53)
    fqq = Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))         # (A54)
    detH = fmm * fqq - fmq ** 2
    dtanh = L / bc * B                                       # (A56)
    dtanh2 = 2 * L / bc * B                                  # (A57)
    fmb = Om * dtanh                                         # (A58)
    fqb = -Om / 2 * dtanh2                                   # (A58)
    fbb = arb(1) / 2 + arb(p - 1) / 2 * mup - L / bc2 * mu - L ** 2 / bc2 * B   # (A61)
    vHv = (fqq * fmb ** 2 - 2 * fmq * fmb * fqb + fmm * fqb ** 2) / detH         # (A50) explicit inverse
    A_FM_A62 = arb(1) / 2 - fbb + vHv                                           # (A62)
    A_FM_A63 = mup / 2 + L ** 2 / bc2 * B + fmb ** 2 * (fmm + 2 * fmq + fqq) / detH  # (A63)/(44)-(45)
    fxb = bc * mup / 2                                       # (A47)
    fxx = V - arb(p - 1) * bc2 * mup / 2                     # (A48)
    fxx_VW = V - 2 * W                                       # (A48), second form
    A_SG = fxb ** 2 / fxx                                    # (A49)
    diff = A_FM_A62 - A_SG
    dC = bc2 * diff                                          # (52): C_SG - C_FM
    Kb = diff / (2 * bc * mup)                               # (51)
    K = Kb * bc ** 4                                         # (51)
    K52 = bc * dC / (2 * mup)                                # (52)
    Psi_q = -Om / 2 * lam                                    # (A44)
    detJ = Psi_q * fxx                                       # (A45)
    out = {
        "Lambda_c": L, "mu": mu, "tau": tau, "E_ln2cosh": L2c, "W": W, "V": V, "beta_c": bc,
        "T_c": 1 / bc, "j0M": j0M, "Omega": Om, "Omega_A55": Om_A55, "lambda_M": lam, "B": B,
        "phi_mm": fmm, "phi_mq": fmq, "phi_qq": fqq, "detH": detH, "dtanh_dbeta": dtanh,
        "dtanh2_dbeta": dtanh2, "phi_mb": fmb, "phi_qb": fqb, "phi_bb": fbb, "vHv": vHv,
        "A_FM": A_FM_A62, "A_FM_A63": A_FM_A63, "phi_xb": fxb, "phi_xx": fxx, "phi_xx_VW": fxx_VW,
        "A_SG": A_SG, "A_FM_minus_A_SG": diff, "kappa": dC, "Kb": Kb, "K": K, "K_eq52": K52,
        "Psi_q": Psi_q, "detJ": detJ, "F_delta": bc * mup, "Dprime_at_Lc": Dp_nlfree(p, L),
        "D_at_Lc": D2(p, L), "G_qq": Om / 2 * lam,
    }
    # ---- raw (NL-free) versions: no use of (16)/(A51) ----
    S2, S2T, Q4 = E("sech2", L), E("sech2tanh", L), E("q4", L)
    T2, T4, S4, WP = E("tanh2", L), E("tanh4", L), E("sech4", L), E("wprime", L)
    Lqq = arb(p * (p - 1) * (p - 2)) * bc2 * mu ** (p - 3) / 2
    r_fmm = Om * (Om * S2 - 1)
    r_fmq = -(Om ** 2) * S2T
    r_fqq = Om / 2 * (1 - Om * Q4) + Lqq / 2 * (mu - T2)
    r_det = r_fmm * r_fqq - r_fmq ** 2
    r_fmb = Om * L / bc * (S2 - 2 * S2T)
    r_fqb = -Om / 2 * (2 * L / bc) * (S2T + Q4) + Om / bc * (mu - T2)
    r_fbb = arb(1) / 2 + arb(p - 1) / 2 * mup - L / bc2 * (1 - S2) + L ** 2 / bc2 * (S2 - 4 * S2T - 2 * Q4)
    r_vHv = (r_fqq * r_fmb ** 2 - 2 * r_fmq * r_fmb * r_fqb + r_fmm * r_fqb ** 2) / r_det
    r_AFM = arb(1) / 2 - r_fbb + r_vHv
    r_lam = 1 - Om * S4                                      # (22) literal
    r_B = -(S2 - 4 * S2T - 2 * Q4)
    r_fxb = 2 * L / bc * WP - arb(p - 1) * bc * mup / 2
    r_fxx = V - 2 * W
    r_ASG = r_fxb ** 2 / r_fxx
    r_diff = r_AFM - r_ASG
    raw = {
        "E_tanh2": T2, "E_tanh4": T4, "E_sech2": S2, "E_sech2tanh": S2T, "E_q4": Q4, "E_sech4": S4,
        "E_wprime": WP, "lambda_M": r_lam, "B": r_B, "phi_mm": r_fmm, "phi_mq": r_fmq, "phi_qq": r_fqq,
        "detH": r_det, "phi_mb": r_fmb, "phi_qb": r_fqb, "phi_bb": r_fbb, "vHv": r_vHv, "A_FM": r_AFM,
        "phi_xb": r_fxb, "phi_xx": r_fxx, "A_SG": r_ASG, "A_FM_minus_A_SG": r_diff,
        "kappa": bc2 * r_diff, "K": r_diff / (2 * bc * mup) * bc ** 4, "G_qq": Om / 2 * r_lam,
        "detJ": -Om / 2 * r_lam * r_fxx,
    }
    return out, raw


SIGN_CHECKS = [  # (key, required sign, meaning)
    ("kappa", 1, "C_SG - C_FM > 0"),
    ("K", 1, "boundary curvature K > 0"),
    ("A_FM_minus_A_SG", 1, "A_FM - A_SG > 0"),
    ("phi_xx", 1, "phi_xx > 0 (SG branch enters x<1; P_yy = phi_xx > 0)"),
    ("detH", -1, "det H < 0 (FM Hessian nondegenerate)"),
    ("lambda_M", 1, "replicon/AT eigenvalue lambda_M > 0"),
    ("G_qq", 1, "G_qq(M) = (Omega/2) lambda_M > 0"),
    ("detJ", -1, "det J_M = -(Omega/2) lambda_M phi_xx != 0 (regularized SG system)"),
    ("B", 1, "B = 1 - 3mu + 2tau > 0"),
    ("F_delta", 1, "dF/d delta = beta_c mu^p > 0"),
    ("Dprime_at_Lc", 1, "D_p'(Lambda_c) > 0 (transversal root)"),
]


def sign_of(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def certify_p(p):
    t0 = time.time()
    L1, L2, root_info = lambda_c(p)
    ctx.prec = PREC
    f1 = form1(p, L1)
    f2, raw = form2(p, L2)
    signs = {}
    for tag, d in (("form1", f1), ("form2", f2), ("raw", raw)):
        for key, want, meaning in SIGN_CHECKS:
            if key not in d:
                continue
            got = sign_of(d[key])
            signs[f"{tag}:{key}"] = {"required": want, "certified_sign": got, "ok": got == want, "meaning": meaning}
    for tag, d in (("form1", f1), ("form2", f2), ("raw", raw)):
        for key in ("phi_mm", "phi_qq", "phi_mq"):
            signs[f"{tag}:{key}(info)"] = {"certified_sign": sign_of(d[key])}
    # consistency: overlapping enclosures (no disagreement at enclosure resolution)
    pairs = []
    for k in ["Lambda_c", "mu", "tau", "W", "V", "beta_c", "T_c", "j0M", "Omega", "lambda_M", "B", "phi_mm",
              "phi_mq", "phi_qq", "detH", "phi_mb", "phi_bb", "vHv", "A_FM", "phi_xb", "phi_xx", "A_SG",
              "A_FM_minus_A_SG", "kappa", "Kb", "K", "G_qq", "detJ", "F_delta", "Dprime_at_Lc"]:
        pairs.append((f"form1.{k}", f1[k], f"form2.{k}", f2[k]))
    for k in ["lambda_M", "B", "phi_mm", "phi_mq", "phi_qq", "detH", "phi_mb", "phi_bb", "vHv", "A_FM",
              "phi_xb", "phi_xx", "A_SG", "A_FM_minus_A_SG", "kappa", "K"]:
        pairs.append((f"form2.{k}", f2[k], f"raw.{k}", raw[k]))
    pairs += [
        ("form1.A_FM", f1["A_FM"], "form1.A_FM_alt", f1["A_FM_alt"]),
        ("form1.detH", f1["detH"], "form1.detH_bracket", f1["detH_bracket"]),
        ("form1.hsum", f1["hsum"], "form1.-Omega*lamM/2", f1["minus_Omega_lamM_over_2"]),
        ("form1.phi_xx", f1["phi_xx"], "form1.V-2W", f1["phi_xx_alt"]),
        ("form1.K", f1["K"], "form1.beta_c*kappa/(2mu^p)", f1["K_alt"]),
        ("form1.Omega", f1["Omega"], "form1.(p-1)Lc/mu", f1["Omega_alt"]),
        ("form1.D'(Lc)", f1["Dprime_at_Lc"], "form1.mu*lamM/(2p)", f1["mu_lamM_over_2p"]),
        ("form1.D(Lc)", f1["D_at_Lc"], "zero", arb(0)),
        ("form2.D(Lc)", f2["D_at_Lc"], "zero", arb(0)),
        ("form2.A_FM(A62)", f2["A_FM"], "form2.A_FM(A63)", f2["A_FM_A63"]),
        ("form2.phi_xx(A48)", f2["phi_xx"], "form2.V-2W", f2["phi_xx_VW"]),
        ("form2.Omega(23)", f2["Omega"], "form2.Omega(A55)", f2["Omega_A55"]),
        ("form2.K(51)", f2["K"], "form2.K(52)", f2["K_eq52"]),
        ("form2.phi_qb", f2["phi_qb"], "-form2.phi_mb", -f2["phi_mb"]),
        ("raw.phi_qb", raw["phi_qb"], "-raw.phi_mb", -raw["phi_mb"]),
        ("form2.Psi_q", f2["Psi_q"], "-(Omega/2)lambda(raw)", -f2["Omega"] / 2 * raw["lambda_M"]),
        ("NL: E tanh^2", raw["E_tanh2"], "mu", f2["mu"]),
        ("NL: E tanh^4", raw["E_tanh4"], "tau", f2["tau"]),
        ("NL: E sech^2", raw["E_sech2"], "1-mu", 1 - f2["mu"]),
        ("NL: E sech^2 tanh", raw["E_sech2tanh"], "mu-tau", f2["mu"] - f2["tau"]),
        ("NL: E(1-4t^2+3t^4)", raw["E_q4"], "1-4mu+3tau", 1 - 4 * f2["mu"] + 3 * f2["tau"]),
        ("NL: E sech^4", raw["E_sech4"], "1-2mu+tau", 1 - 2 * f2["mu"] + f2["tau"]),
        ("NL: E[tanh-tanh^2/2]", raw["E_wprime"], "mu/2", f2["mu"] / 2),
        ("form1.E1+ln2", f1["E1"] + ln2(), "form2.E ln2cosh", f2["E_ln2cosh"]),
    ]
    consistency = []
    all_overlap = True
    for na, a, nb, b in pairs:
        ov = bool(a.overlaps(b))
        all_overlap &= ov
        consistency.append({"a": na, "b": nb, "overlap": ov,
                            "abs_diff_upper": s((a - b).abs_upper(), 3) if hasattr(a - b, "abs_upper") else None})
    all_signs = all(v["ok"] for v in signs.values() if "ok" in v)
    res = {
        "p": p,
        "Lambda_c_certificate": root_info,
        "form1": {k: s(v) for k, v in f1.items()},
        "form2": {k: s(v) for k, v in f2.items()},
        "raw": {k: s(v) for k, v in raw.items()},
        "signs": signs,
        "consistency": consistency,
        "all_required_signs_certified": all_signs,
        "all_consistency_overlaps": all_overlap,
        "root_unique_on_(0,inf)": root_info["unique_zero_on_(0,inf)"],
        "krawczyk_ok": root_info["krawczyk_form1"]["K_in_int_X"] and root_info["krawczyk_form2"]["K_in_int_X"]
                       and root_info["krawczyk_form1"]["X_in_U"] and root_info["krawczyk_form2"]["X_in_U"],
        "seconds": round(time.time() - t0, 2),
    }
    res["zhou_p3_bracket_[1.05,1.1]"] = (bool(f1["beta_c_over_sqrt2"] > arb("1.05") and f1["beta_c_over_sqrt2"] < arb("1.1"))
                                         if p == 3 else None)
    rel = lambda x: float((x.rad() / abs(x.mid())).mid()) if x.mid() != 0 else None
    log(f"p={p:2d} Lc={f1['Lambda_c'].str(20)} kappa={f1['kappa'].str(15)} (rel rad {rel(f1['kappa']):.1e}) "
        f"K={f1['K'].str(12)} detH={f1['detH'].str(10)} phi_xx={f1['phi_xx'].str(10)} lamM={f1['lambda_M'].str(10)} "
        f"B={f1['B'].str(8)} | signs {'OK' if all_signs else 'FAIL'} overlaps {'OK' if all_overlap else 'FAIL'} "
        f"root-unique {root_info['unique_zero_on_(0,inf)']} krawczyk {res['krawczyk_ok']} "
        f"scan {root_info['scan_left']['pieces']}+{root_info['scan_right']['pieces']} pieces, {res['seconds']}s")
    return res


def main():
    global PREC, TOL_BITS, ZCUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmin", type=int, default=3)
    ap.add_argument("--pmax", type=int, default=25)
    ap.add_argument("--out", default="kappa_cert.json")
    ap.add_argument("--prec", type=int, default=PREC, help="tight-phase precision (bits)")
    ap.add_argument("--tol-bits", type=int, default=TOL_BITS, help="tight-phase integration tolerance 2^-k")
    ap.add_argument("--zcut", type=int, default=ZCUT, help="tight-phase truncation |z| <= zcut")
    args = ap.parse_args()
    PREC, TOL_BITS, ZCUT = args.prec, args.tol_bits, args.zcut
    t0 = time.time()
    meta = {
        "status": "certified: Arb ball arithmetic + rigorous integration (see README.md for the trust base)",
        "python_flint": flint.__version__, "python": sys.version.split()[0], "platform": platform.system(),
        "PREC_bits": PREC, "TOL_BITS": TOL_BITS, "ZCUT": ZCUT, "SCAN_PREC_bits": SCAN_PREC,
        "SCAN_TOL_BITS": SCAN_TOL_BITS, "SCAN_ZCUT": SCAN_ZCUT, "krawczyk_radius": f"2^{KRAW_RAD_EXP}",
        "lemma_threshold_a0": str(A0), "Lmax": "2p*355/512 (>= 2p ln2, checked)", "U_half_width": str(U_HALF),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    log(f"kappa_cert: python-flint {flint.__version__}, PREC={PREC}, p={args.pmin}..{args.pmax}")
    results = {}
    for p in range(args.pmin, args.pmax + 1):
        results[str(p)] = certify_p(p)
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=1)
    summary = {
        "p_range": [args.pmin, args.pmax],
        "all_required_signs_certified": all(r["all_required_signs_certified"] for r in results.values()),
        "all_consistency_overlaps": all(r["all_consistency_overlaps"] for r in results.values()),
        "all_roots_unique_on_(0,inf)": all(r["root_unique_on_(0,inf)"] for r in results.values()),
        "all_krawczyk_ok": all(r["krawczyk_ok"] for r in results.values()),
        "total_seconds": round(time.time() - t0, 1),
    }
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "summary": summary, "results": results}, fh, indent=1)
    log("SUMMARY", json.dumps(summary))
    return 0 if all(v for k, v in summary.items() if k.startswith("all_")) else 1


if __name__ == "__main__":
    sys.exit(main())
