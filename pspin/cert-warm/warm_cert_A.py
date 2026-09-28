"""Warm predicate (W) of Theorem 6.9 (thm:reduction) -- primary implementation (Table 1, tab:cert).

Model on the Nishimori line: beta = 2 j0, xi(q) = 2 j0^2 q^p, theta(q) = (p-1) xi(q),
  Phi(q) = psi(xi'(q)) - (xi'(q) + theta(q))/2,   psi(r) = E log cosh(r + sqrt(r) g),
  I(m)   = (1+m)/2 log(1+m) + (1-m)/2 log(1-m)    (binary rate function).
Predicate (W): q0 in Q_p, m1^2 <= 1 - 2/p, and Phi(q0) > max{0, xi(m1) - I(m1)}.
Optional (identification of the warm magnetization, p = 3 only):
  Phi(1/2) + xi(1) (m1 - 1/2)^2 (m1 + 1) < Phi(q0).

Arithmetic:
  psi         -- enc.psi: numpy outward-rounded intervals, composite Simpson with interval-Taylor
                 fourth-derivative remainders, analytic Gaussian tails (libm < 2 ulp, widened 3 ulp).
  logarithms  -- exact rational arithmetic (fractions.Fraction):
                 log y = 2 sum_k u^{2k+1}/(2k+1), u = (y-1)/(y+1), with the tail bounded by
                 2|u|^{2K+1} / ((2K+1)(1-u^2)); all terms have the sign of u.
Usage: python warm_cert_A.py [n_simpson]
"""
import json
import os
import sys
from fractions import Fraction as Fr

from enc import fr_iv, psi

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
N_SIMPSON = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CASES = {
    3: dict(j0=Fr(3841, 5000), q0=Fr(814809, 10**6), m1=Fr(11, 25)),
    4: dict(j0=Fr(8107, 10000), q0=Fr(948189, 10**6), m1=Fr(16, 25)),
}
K_SERIES = 120


def log_rat(y):
    """Exact rational enclosure [lo, hi] of log y for rational y > 0."""
    y = Fr(y)
    assert y > 0
    u = (y - 1) / (y + 1)
    s = Fr(0)
    pw = u
    u2 = u * u
    for k in range(K_SERIES):
        s += pw / (2 * k + 1)
        pw *= u2
    tail = 2 * abs(pw) / ((2 * K_SERIES + 1) * (1 - u2))  # pw = u^{2K+1}
    s *= 2
    return (s, s + tail) if u >= 0 else (s - tail, s)


def rate_rat(m):
    """Exact rational enclosure of I(m), 0 <= m < 1."""
    m = Fr(m)
    a, b = log_rat(1 + m)
    c, d = log_rat(1 - m)
    wp, wm = (1 + m) / 2, (1 - m) / 2  # both > 0
    return wp * a + wm * c, wp * b + wm * d


def dec(x, digits=12, up=False):
    """Decimal string of the rational x rounded outward (floor, or ceiling if up=True) to `digits` significant digits."""
    x = Fr(x)
    if x == 0:
        return "0"
    e = 0
    ax = abs(x)
    while ax >= 10:
        ax /= 10
        e += 1
    while ax < 1:
        ax *= 10
        e -= 1
    k = max(digits - 1 - e, 0)
    v = x * Fr(10) ** k
    n = v.numerator // v.denominator  # floor
    if up and n != v:
        n += 1
    sgn = "-" if n < 0 else ""
    s = str(abs(n))
    if k == 0:
        return sgn + s
    s = s.rjust(k + 1, "0")
    return f"{sgn}{s[:-k]}.{s[-k:]}"


def fdn(x):
    return dec(x, 12, up=False)


def fup(x):
    return dec(x, 12, up=True)


def run(p, j0, q0, m1):
    c = 2 * j0 * j0  # xi(1)
    xi = lambda q: c * Fr(q) ** p
    xip = lambda q: c * p * Fr(q) ** (p - 1)
    th = lambda q: (p - 1) * c * Fr(q) ** p
    out = dict(p=p, j0=str(j0), beta_w=str(2 * j0), xi1=str(c), q0=str(q0), m1=str(m1),
               n_simpson=N_SIMPSON, series_terms=K_SERIES)

    # exact rational side conditions
    if p % 2 == 1:
        qp = p * q0 ** (p - 1) + (p - 1) * q0 ** p
        assert qp >= 1, "q0 not in Q_p"
        out["Q_p_check"] = f"p q0^(p-1) + (p-1) q0^p = {qp} >= 1"
    else:
        out["Q_p_check"] = "p even: Q_p = [0,1]"
    assert m1 * m1 <= 1 - Fr(2, p)
    out["endpoint_check"] = f"m1^2 = {m1 * m1} <= 1 - 2/p = {1 - Fr(2, p)}"

    def Phi(q):
        return psi(xip(q), N_SIMPSON) - fr_iv((xip(q) + th(q)) / 2)

    P0 = Phi(q0)
    Plo, Phi_hi = Fr(float(P0.lo)), Fr(float(P0.hi))
    out["Phi_q0"] = [fdn(Plo), fup(Phi_hi)]
    out["r0"] = str(xip(q0))

    Ilo, Ihi = rate_rat(m1)
    flo, fhi = xi(m1) - Ihi, xi(m1) - Ilo
    out["I_m1"] = [fdn(Ilo), fup(Ihi)]
    out["xi_m1"] = str(xi(m1))
    out["f_m1"] = [fdn(flo), fup(fhi)]
    out["f_m1_width"] = float(fhi - flo)

    bar = max(Fr(0), fhi)
    margin = Plo - bar
    out["margin_W_lo"] = fdn(margin)
    assert Plo > 0 and margin > 0, "(W) not certified"

    if p == 3:
        h = Fr(1, 2)
        Ph = Phi(h)
        Ph_hi = Fr(float(Ph.hi))
        B = c * (m1 - h) ** 2 * (m1 + 1)
        out["Phi_half"] = [fdn(Fr(float(Ph.lo))), fup(Ph_hi)]
        out["Bregman_half_at_m1"] = str(B)
        out["Phi_half_plus_B_hi"] = fup(Ph_hi + B)
        out["margin_half_lo"] = fdn(Plo - (Ph_hi + B))
        assert Ph_hi + B < Plo
    return out


if __name__ == "__main__":
    res = {}
    for p, kw in CASES.items():
        res[p] = run(p, **kw)
        print(json.dumps(res[p], indent=1), flush=True)
    json.dump(res, open(os.path.join(HERE, f"warm_cert_A_n{N_SIMPSON}.json"), "w"), indent=1)
