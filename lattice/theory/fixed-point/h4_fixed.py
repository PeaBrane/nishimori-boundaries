"""Second implementation of the H4 numbers of ../h4_cert.py (input (C6); Lemma C.11, lem:lat-kappa).

Methods (deliberately different from ../h4_cert.py where possible):
  * exact Fractions for P1,P2,P3, cross-checked against a 32-configuration signed enumeration;
  * fixed-point integer intervals (fx.Iv, 384 bits, directed rounding) for everything else;
  * warm v_H(gamma) from the magnitude law AND from the signed law E e^{-K} (Nishimori check);
  * cold compact part via a Lipschitz grid (lambda(H4) = 3 as in Lemma C.12) with constants 3 (u) and 6 (pbar),
    using exact-rational grid points x_j and the rigorous spacing bound
    beta_{j+1}-beta_j = (1/2) log(x_j/x_{j+1}) <= (x_j/x_{j+1}-1)/2  (not monotone brackets);
  * tail bound (beta -> infinity limit, Proposition 9.3(iii)) re-implemented from its statement.
Usage: python h4_fixed.py b0 k p_num p_den
"""
import sys
from fractions import Fraction as F
from math import comb

import mpmath as mp

from fx import Iv, fmt_down, fmt_up

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.mp.dps = 60


def classes(k):
    return [(n1, n2, k - n1 - n2) for n1 in range(k + 1) for n2 in range(k + 1 - n1)]


def unit_probs(b0, p):
    th = 1 - 2 * p
    q = (1 - th ** b0) / 2
    a = (1 + th ** 3) / 2
    P1 = 2 * q * (1 - q)
    P3 = (1 - q) ** 2 * (1 - a) + q ** 2 * a
    P2 = 1 - P1 - P3
    return th, q, a, P1, P2, P3


def signed_unit_law(b0, p):
    """Enumerate (s_d, s_1, s_2, c_1, c_2); return dict (class, sign) -> exact prob.

    class 1: c1 != c2 (|K_U| = beta); 2: c1 = c2 = eps and s_d s_1 s_2 eps = +1 (beta + kappa2);
    3: cancel (beta - kappa2). sign = sgn K_U, derived here from the formula
    K_U = beta s_d + s_1 s_2 atanh(theta^2 tanh((c1 + c2) kappa)) and 0 <= kappa2 < beta.
    """
    th = 1 - 2 * p
    q = (1 - th ** b0) / 2
    law = {}
    for sd in (1, -1):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for c1 in (1, -1):
                    for c2 in (1, -1):
                        pr = F(1)
                        for s in (sd, s1, s2):
                            pr *= (1 - p) if s == 1 else p
                        for c in (c1, c2):
                            pr *= (1 - q) if c == 1 else q
                        if c1 != c2:
                            cls, sgn = 1, sd
                        else:
                            second = s1 * s2 * c1  # sign of the kappa2 term
                            cls = 2 if second == sd else 3
                            sgn = sd  # |beta| > kappa2 so sign follows s_d
                        law[(cls, sgn)] = law.get((cls, sgn), F(0)) + pr
    return law


def unit_m_exact(th, b0):
    tb = th ** b0
    tau = 2 * tb * th * th / (1 + tb * tb)
    m1 = th
    m2 = (th + tau) / (1 + th * tau)
    m3 = (th - tau) / (1 - th * tau)
    return tau, m1, m2, m3


def unit_m_iv(x, b0):
    """x = e^{-2 beta} exact Fraction -> interval theta, tau, m1, m2, m3."""
    th = Iv.frac((1 - x) / (1 + x))
    tb = th.pow(b0)
    tau = 2 * tb * th * th / (1 + tb * tb)
    m2 = (th + tau) / (1 + th * tau)
    m3 = (th - tau) / (1 - th * tau)
    return th, tau, th, m2, m3


def weights_iv(k, Ps):
    Pi = [Iv.frac(P) for P in Ps]
    out = {}
    for n in classes(k):
        mult = comb(k, n[0]) * comb(k - n[0], n[1])
        w = Iv.int(mult)
        for c in range(3):
            w = w * Pi[c].pow(n[c])
        out[n] = w
    return out


def u_pbar_iv(x, b0, k, Ps, W):
    th, tau, m1, m2, m3 = unit_m_iv(x, b0)
    Pi = [Iv.frac(P) for P in Ps]
    mean = Pi[0] * m1 + Pi[1] * m2 + Pi[2] * m3
    u = th * th * mean.pow(k)
    pb = Iv.int(0)
    th2 = th * th
    for n, w in W.items():
        xn = th2 * m1.pow(n[0]) * m2.pow(n[1]) * m3.pow(n[2])
        pb = pb + w * (2 - 2 / (1 + xn))
    return u, pb


def h_exact(v):
    v = F(v)
    z = 9 * v * v
    Pv = 9 * v ** 4 / (1 - z) ** 2
    Qv = 9 * v ** 3 / (1 - z)
    return 4 * Pv + 4 * Qv


def x_upper(p, c, digits=40):
    """Rational X >= (p/(1-p))^c for rational c = a/b, verified exactly: X^b >= (p/(1-p))^a."""
    c = F(c)
    a, b = c.numerator, c.denominator
    base = p / (1 - p)
    val = mp.power(mp.mpf(base.numerator) / base.denominator, mp.mpf(a) / b)
    s = mp.nstr(val, digits + 5, strip_zeros=False)
    X = F(s)
    X = X * (1 + F(1, 10 ** digits))
    X = F(X.numerator, X.denominator).limit_denominator(10 ** (digits + 12))
    while not (X ** b >= base ** a):
        X *= 1 + F(1, 10 ** digits)
    return X


def x_lower(p, c, digits=40):
    c = F(c)
    a, b = c.numerator, c.denominator
    base = p / (1 - p)
    val = mp.power(mp.mpf(base.numerator) / base.denominator, mp.mpf(a) / b)
    X = F(mp.nstr(val, digits + 5, strip_zeros=False)) * (1 - F(1, 10 ** digits))
    X = X.limit_denominator(10 ** (digits + 12))
    while not (X ** b <= base ** a):
        X *= 1 - F(1, 10 ** digits)
    return X


def tail(x, b0, k, P3):
    x = F(x)
    side = 4 * x + 8 * b0 * b0 * x * x
    assert side < 2, "tail side condition fails"
    E = (4 + 8 * b0 * b0 * x) / (2 - 4 * x - 8 * b0 * b0 * x * x)
    rho = (E - 1) / (E + 1)
    P3i = Iv.frac(P3)
    U = (1 - 2 * P3i / Iv.frac(E + 1)).pow(k)
    rhoi = Iv.frac(rho)
    pb = Iv.int(0)
    for j in range(k + 1):
        term = Iv.int(comb(k, j)) * P3i.pow(j) * (1 - P3i).pow(k - j)
        rj = rhoi.pow(j)
        pb = pb + term * (2 - 2 / (1 + rj))
    return E, U, pb


def round_sig(x, sig=30, up=False):
    """Round positive Fraction to `sig` significant digits (down by default)."""
    x = F(x)
    e = len(str(x.numerator // x.denominator)) if x >= 1 else -len(str(x.denominator // x.numerator)) + 1
    scale = F(10) ** (sig - e)
    v = x * scale
    n = v.numerator // v.denominator
    if up and n * v.denominator != v.numerator:
        n += 1
    return F(n) / scale


def lipschitz_sweep(x_start, x_stop, target, lam, f, max_step=F(1, 20)):
    """Cover beta in [beta(x_start), beta(x_end)] with x_end <= x_stop.

    f(x) -> upper bound (Fraction) of the functional at exact x. Returns (sup bound, n points, x_end,
    first value). Condition per piece: (f_j + f_{j+1} + lam*Delta)/2, Delta <= (x_j/x_{j+1}-1)/2.
    """
    xs = x_start
    fs = f(xs)
    first = fs
    sup = fs
    n = 1
    while xs > x_stop:
        margin = target - fs
        assert margin > 0, (float(fs), float(target))
        step = min(max_step, F(19, 10) * margin / lam)
        while True:
            xn = round_sig(xs / (1 + 2 * step), 30)
            delta = (xs / xn - 1) / 2
            fn = f(xn)
            piece = (fs + fn + lam * delta) / 2
            if piece <= target:
                break
            step /= 2
            assert step > F(1, 10 ** 12), "step underflow"
        sup = max(sup, piece)
        xs, fs = xn, fn
        n += 1
    return sup, n, xs, first


def kappa_bar(b0, k, p, tU_lo):
    """Lower bound of Lemma C.11 (lem:lat-kappa), directed rounding down; returns (blocks, arc) as Fractions."""
    from fx import S
    th = 1 - 2 * p
    r = th * th
    ri = Iv.frac(r)
    rl = ri.lo  # lower bound scaled
    # r^l lower bounds for l = 0 .. b0 + 2
    rp = [S]
    for _ in range(b0 + 3):
        rp.append(rp[-1] * rl // S)
    tU = tU_lo  # scaled lower bound
    tUp = [S]
    for _ in range(k + 1):
        tUp.append(tUp[-1] * tU // S)

    def C(l1, l2):
        a, b = rp[l1], rp[l2]
        num = (a + b) * S - 2 * a * b
        den = S * S - a * b
        return num * S // den

    def mul(*xs):
        out = S
        for v in xs:
            out = out * v // S
        return out

    I = 2 * k * b0 + k + 1
    tot_b = 0
    tot_a = 0
    for i in range(k + 1):
        z = mul(rl, tUp[min(i, k - i)])
        tot_b += z
        tot_a += z
    for i in range(1, k + 1):
        for j in range(0, b0 + 1):
            gb = max(mul(rl, tUp[i - 1], C(j + 1, b0 - j + 2)), mul(rl, tUp[k - i], C(b0 - j + 1, j + 2)))
            ga = max(mul(rl, tUp[i - 1], rp[j + 1]), mul(rl, tUp[k - i], rp[b0 - j + 1]))
            mult = 1 if j in (0, b0) else 2
            tot_b += mult * gb
            tot_a += mult * ga
    return F(tot_b, S * I), F(tot_a, S * I), I


def main():
    b0, k, pn, pd = map(int, sys.argv[1:5])
    p = F(pn, pd)
    print(f"== H4({b0},{k}), p0 = {p} ==")
    th, q, a, P1, P2, P3 = unit_probs(b0, p)
    law = signed_unit_law(b0, p)
    Pc = {c: law.get((c, 1), 0) + law.get((c, -1), 0) for c in (1, 2, 3)}
    assert Pc[1] == P1 and Pc[2] == P2 and Pc[3] == P3, "class probabilities disagree"
    print("P1,P2,P3 exact match with signed 32-configuration enumeration: True")
    for name, v in (("q", q), ("a", a), ("P1", P1), ("P2", P2), ("P3", P3)):
        print(f"  {name} = {fmt_down(v, 12)} .. {fmt_up(v, 12)}")
    edges = k * (2 * b0 + 3) + 2
    print(f"  edges = {edges}, I = {2 * k * b0 + k + 1}")
    gam = mp.mpf(1) / 2 * mp.log(mp.mpf((1 - p).numerator * p.denominator) / (p.numerator * (1 - p).denominator))
    print(f"  gamma = {mp.nstr(gam, 12)}")

    # ---------------- warm ----------------
    tau, m1, m2, m3 = unit_m_exact(th, b0)
    th_i = Iv.frac(th)
    mi = [Iv.frac(m1), Iv.frac(m2), Iv.frac(m3)]
    W = weights_iv(k, (P1, P2, P3))
    th2 = th_i * th_i
    v = Iv.int(0)
    for n, w in W.items():
        xn = th2 * mi[0].pow(n[0]) * mi[1].pow(n[1]) * mi[2].pow(n[2])
        v = v + w * (1 - xn * xn).sqrt()
    print(f"  v_H(gamma) [magnitude law, E sech] in {v.dec(15)}")
    # signed law: E e^{-K} with e^{-K} = sqrt((1-t)/(1+t))
    Ps = {key: Iv.frac(val) for key, val in law.items()}
    states = {(0, 0, 0, 1): Iv.int(1)}
    for _ in range(k):
        new = {}
        for (n1, n2, n3, s), pr in states.items():
            for (c, sg), pc in Ps.items():
                key = (n1 + (c == 1), n2 + (c == 2), n3 + (c == 3), s * sg)
                new[key] = new.get(key, Iv.int(0)) + pr * pc
        states = new
    pi_ = Iv.frac(p)
    ends = {1: (1 - pi_) * (1 - pi_) + pi_ * pi_, -1: 2 * pi_ * (1 - pi_)}
    vs = Iv.int(0)
    tot = Iv.int(0)
    for (n1, n2, n3, s), pr in states.items():
        xn = th2 * mi[0].pow(n1) * mi[1].pow(n2) * mi[2].pow(n3)
        for se, pe in ends.items():
            sig = s * se
            t = xn if sig == 1 else -xn
            prob = pr * pe
            vs = vs + prob * ((1 - t) / (1 + t)).sqrt()
            tot = tot + prob
    print(f"  v_H(gamma) [signed law, E e^-K]    in {vs.dec(15)} (total mass {tot.dec(12)})")
    V = v.fhi()
    hV = h_exact(V)
    print(f"  h(V_hi) = {fmt_up(hV, 12)}; 1-h(V_hi) >= {fmt_down(1 - hV, 12)}; 3V<1: {3 * V < 1}")
    assert h_exact(F("0.2224074394")) < 1 < h_exact(F("0.2224074395"))
    print(f"  v_P bracket ok; margin v_P - V > {fmt_down(F('0.2224074394') - V, 6)}")
    tU = Iv.frac(P1) * mi[0] * mi[0] + Iv.frac(P2) * mi[1] * mi[1] + Iv.frac(P3) * mi[2] * mi[2]
    tU_signed = Iv.int(0)
    for (c, sg), pc in Ps.items():
        tU_signed = tU_signed + pc * (mi[c - 1] if sg == 1 else -mi[c - 1])
    print(f"  t_U = E tanh^2 K_U in {tU.dec(15)}; E tanh K_U (signed) in {tU_signed.dec(15)}; r = {fmt_down(th * th, 12)}")
    kb, ka, I = kappa_bar(b0, k, p, tU.lo)
    for nm, kk in (("blocks", kb), ("arc", ka)):
        khat = (1 + 2 * I * kk) / (1 + 2 * I)
        D = khat * khat * (1 - hV)
        print(f"  kappa_bar >= {fmt_down(kk, 9)} ({nm}); kappa_hat >= {fmt_down(khat, 9)}; liminf E<M^2> >= {fmt_down(D, 9)}")

    # ---------------- cold ----------------
    def fu(x):
        return u_pbar_iv(x, b0, k, (P1, P2, P3), W)[0].fhi()

    def fp(x):
        return u_pbar_iv(x, b0, k, (P1, P2, P3), W)[1].fhi()

    uinf = (1 - F(2, 3) * P3) ** k
    print(f"  u_H(inf) = (1-2P3/3)^k = {fmt_down(uinf, 12)}")
    for c in (F(2), F(202, 100), F(21, 10), F(215, 100), F(22, 10), F(222, 100), F(225, 100), F(23, 10), F(26, 10), F(27, 10), F(3), F(10)):
        xc = x_upper(p, c, 40)
        u_, pb_ = u_pbar_iv(xc, b0, k, (P1, P2, P3), W)
        print(f"    beta ~ {float(c)}*gamma: u_H in {u_.dec(9)}, pbar_H in {pb_.dec(9)}")
    return b0, k, p, P3, W, fu, fp


if __name__ == "__main__":
    main()
