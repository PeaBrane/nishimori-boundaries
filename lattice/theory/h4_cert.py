"""Directed-rounding rational evaluation of the H4 inputs of the paper: (C6) and the chain bound of
Lemma C.11 (lem:lat-kappa), and the class data and limits quoted in Section 9.

Single implementation, Python stdlib only (fractions + integer sqrt). Every transcendental quantity is
avoided: grid points are rational x = e^{-2 beta}, so tanh(beta) = (1-x)/(1+x) is rational, and every
other quantity is a rational function of tanh(beta) and p. Each rounding is directed (dn = floor, up =
ceil on the 10^-DIG grid) in the direction that keeps the stated inequality valid, using the
monotonicity facts listed next to each step.

usage: h4_cert.py b0 k p_num p_den c1 c2
  c1, c2: rationals like 2.15 (beta1 = c1*gamma), 2.6 (tail start beta2 = c2*gamma).
Outputs: warm V >= v_H(gamma); h(V) < 1; kappa-bar lower bounds (arc-only and cycle blocks);
         D lower bound; t_U; cold: tail bound for x <= X0 and monotone-bracket cover of [X0, XM].
"""
import sys, math
from fractions import Fraction as F

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

DIG = 60
SC = 10 ** DIG


def dn(x):
    x = F(x)
    return F((x.numerator * SC) // x.denominator, SC)


def up(x):
    x = F(x)
    return F(-((-x.numerator * SC) // x.denominator), SC)


def sqrt_up(y):
    Y = math.ceil(y * SC * SC)
    return F(math.isqrt(Y) + 1, SC)


def sqrt_dn(y):
    Y = math.floor(y * SC * SC)
    return F(math.isqrt(Y), SC)


def h_of(v):
    z = 9 * v * v
    return 4 * (9 * v ** 4) / (1 - z) ** 2 + 4 * (9 * v ** 3) / (1 - z)


def multinomials(k):
    for n1 in range(k + 1):
        for n2 in range(k + 1 - n1):
            n3 = k - n1 - n2
            yield n1, n2, n3, math.factorial(k) // (math.factorial(n1) * math.factorial(n2) * math.factorial(n3))


def main():
    b0, k = int(sys.argv[1]), int(sys.argv[2])
    p = F(int(sys.argv[3]), int(sys.argv[4]))
    c1, c2 = F(sys.argv[5]), F(sys.argv[6])
    vt = 1 - 2 * p                         # disorder bias = tanh(gamma)
    Phi = vt ** b0                         # E(product of b0 signs)
    q = (1 - Phi) / 2
    a = (1 + vt ** 3) / 2
    P1 = 2 * q * (1 - q)
    P3 = (1 - q) ** 2 * (1 - a) + q * q * a
    P2 = 1 - P1 - P3
    assert P1 > 0 and P2 > 0 and P3 > 0
    P1u, P2u, P3u = up(P1), up(P2), up(P3)
    P1d, P2d, P3d = dn(P1), dn(P2), dn(P3)
    I = k + 1 + 2 * k * b0
    nE = k * (2 * b0 + 3) + 2
    print(f"H4(b0={b0}, k={k}), p={p} = {float(p):.10g}; edges={nE}, interior vertices I={I}")
    print(f"  q=P(path sign -1)={float(q):.12f}  a=P(s_d s_1 s_2=+1)={float(a):.12f}")
    print(f"  P1={float(P1):.12f} P2={float(P2):.12f} P3={float(P3):.12f}  (exact rationals; P3 = zero-temperature cancel prob.)")

    # ---------------- warm side at gamma: all rational ----------------
    th = vt
    T = Phi
    tau2 = th * th * 2 * T / (1 + T * T)
    m = [th, (th + tau2) / (1 + th * tau2), (th - tau2) / (1 - th * tau2)]
    md = [dn(x) for x in m]
    mu = [up(x) for x in m]
    th2 = th * th
    V = F(0); Vlo = F(0)
    for n1, n2, n3, mult in multinomials(k):
        xlo = dn(th2 * md[0] ** n1 * md[1] ** n2 * md[2] ** n3)       # |tanh K_H| lower
        xhi = up(th2 * mu[0] ** n1 * mu[1] ** n2 * mu[2] ** n3)
        wu = mult * P1u ** n1 * P2u ** n2 * P3u ** n3
        wd = mult * P1d ** n1 * P2d ** n2 * P3d ** n3
        V += up(wu * sqrt_up(1 - xlo * xlo))                           # sech decreasing in |tanh|
        Vlo += dn(wd * sqrt_dn(1 - xhi * xhi))
    V = up(V); Vlo = dn(Vlo)
    hV = h_of(V)
    print(f"WARM: v_H(gamma) = E sech K_H in [{float(Vlo):.15f}, {float(V):.15f}]  (upper bound V certified by directed rounding)")
    print(f"      3V < 1: {3 * V < 1};  h(V) = {float(hV):.12f};  exact rational test h(V) < 1: {hV < 1}")
    # compare with v_P bracket: v_P in (0.2224074394, 0.2224074395)
    assert h_of(F(2224074394, 10 ** 10)) < 1 < h_of(F(2224074395, 10 ** 10))
    print(f"      v_P in (0.2224074394, 0.2224074395) (exact rational h-tests); margin v_P - V > {float(F(2224074394, 10**10) - V):.6f}")

    # t_U = E_gamma tanh K_U = E tanh^2 K_U = sum P_c m_c^2 (exact), lower rounded
    tU = P1 * m[0] ** 2 + P2 * m[1] ** 2 + P3 * m[2] ** 2
    tUd = dn(tU)
    vU = dn(P1 * sqrt_dn(1 - m[0] ** 2) + P2 * sqrt_dn(1 - m[1] ** 2) + P3 * sqrt_dn(1 - m[2] ** 2))
    r = vt * vt
    print(f"  t_U = E_gamma tanh K_U >= {float(tUd):.15f};  r = (1-2p)^2 = {float(r):.12f};  (v_U = E sech K_U ~ {float(vU):.6f})")

    # ---------------- interior-spin constant ----------------
    L = b0 + 3
    rl = [F(1)]
    for _ in range(L + 1):
        rl.append(dn(rl[-1] * r))
    tp = [F(1)]
    for _ in range(k + 1):
        tp.append(dn(tp[-1] * tUd))

    def Cyc(l1, l2):
        t1, t2 = rl[l1], rl[l2]
        return dn((t1 + t2 - 2 * t1 * t2) / (1 - t1 * t2))

    Cf = [Cyc(j + 1, b0 - j + 2) for j in range(b0 + 1)]     # toward pi_0 (via w1 / direct edge)
    Cb = [Cyc(b0 - j + 1, j + 2) for j in range(b0 + 1)]     # toward pi_1
    S_cyc = F(0); S_arc = F(0)
    for i in range(k + 1):                                   # junctions z_i
        zz = dn(r * tp[min(i, k - i)])
        S_cyc += zz; S_arc += zz
    for i in range(1, k + 1):
        fw = dn(r * tp[i - 1]); bw = dn(r * tp[k - i])
        for j in range(b0 + 1):
            mult = 1 if j in (0, b0) else 2                  # w1 (j=0), w2 (j=b0), two paths otherwise
            g_cyc = max(dn(fw * Cf[j]), dn(bw * Cb[j]))
            g_arc = max(dn(fw * rl[j + 1]), dn(bw * rl[b0 - j + 1]))
            S_cyc += mult * g_cyc; S_arc += mult * g_arc
    kb_cyc = dn(S_cyc / I); kb_arc = dn(S_arc / I)
    kh_cyc = dn((1 + 2 * I * kb_cyc) / (1 + 2 * I)); kh_arc = dn((1 + 2 * I * kb_arc) / (1 + 2 * I))
    one_minus_h = 1 - hV
    D_cyc = dn(kh_cyc ** 2 * one_minus_h); D_arc = dn(kh_arc ** 2 * one_minus_h)
    print(f"KAPPA: kbar >= {float(kb_arc):.9f} (arc chains)   kbar >= {float(kb_cyc):.9f} (two-path cycle blocks)")
    print(f"       khat = (1+2I kbar)/(1+2I) >= {float(kh_arc):.9f} / {float(kh_cyc):.9f}")
    print(f"       liminf E<M^2>_gamma >= khat^2 (1-h(V)) >= {float(D_arc):.9f} (arc) / {float(D_cyc):.9f} (cycle)   [1-h(V) >= {float(one_minus_h):.9f}]")
    # path-only averages, for the text
    path_only = dn(sum(2 * max(Cf[j], Cb[j]) for j in range(1, b0)) / (2 * (b0 - 1)))
    arc_only = dn(sum(2 * max(rl[j + 1], rl[b0 - j + 1]) for j in range(1, b0)) / (2 * (b0 - 1)))
    print(f"       (within-unit path-vertex averages: max-direction C = {float(path_only):.6f}, arcs = {float(arc_only):.6f}; unit-chain factor r t_U^(i-1) >= {float(dn(r*tp[k//2])):.6f} at mid-chain)")

    # ---------------- cold side ----------------
    xg = p / (1 - p)                        # e^{-2 gamma}
    def x_upper_for(c, sig=40):
        """rational X with X >= xg^c (c = a/b rational), verified exactly: X^b >= xg^a."""
        fl = float(xg) ** float(c)
        X = F(round(fl * 10 ** sig * (1 + 1e-12)) + 1, 10 ** sig) if fl > 0 else F(0)
        # scale by exponent to keep integer sizes moderate
        e10 = math.floor(math.log10(fl))
        X = F(math.ceil(fl * (1 + 1e-12) * 10 ** (sig - e10)), 10 ** (sig - e10))
        assert X ** c.denominator >= xg ** c.numerator
        return X

    def x_lower_for(c, sig=40):
        fl = float(xg) ** float(c)
        e10 = math.floor(math.log10(fl))
        X = F(math.floor(fl * (1 - 1e-12) * 10 ** (sig - e10)), 10 ** (sig - e10))
        assert X ** c.denominator <= xg ** c.numerator
        return X

    bb = F(b0 * b0)
    def Ebound(x):
        den = 2 - 4 * x - 8 * bb * x * x
        assert den > 0
        return (4 + 8 * bb * x) / den

    # tail: for all beta >= c2*gamma, i.e. x <= X0
    X0 = x_upper_for(c2)
    E0 = Ebound(X0)
    Ubar = up((1 - 2 * P3 / (E0 + 1)) ** k)
    rho = (E0 - 1) / (E0 + 1)
    pb_tail = F(0)
    for n3 in range(k + 1):
        w = math.comb(k, n3) * P3u ** n3 * (1 - P3d) ** (k - n3)
        x = rho ** n3
        pb_tail += up(w * 2 * x / (1 + x))
    uinf = (1 - 2 * P3 / 3) ** k
    print(f"COLD tail (beta >= {float(c2)} gamma, x <= X0={float(X0):.6e}): E(X0)={float(E0):.12f}; sup u_H <= {float(Ubar):.12f}  (<1/3: {Ubar < F(1,3)}); sup pbar_H <= {float(pb_tail):.9f} (<1/2: {pb_tail < F(1,2)})")
    print(f"      limit beta->inf: u_H -> (1-2P3/3)^k = {float(uinf):.12f}")

    # compact part: cover x in [X0, XM], XM >= xg^c1, by intervals [X_j, X_{j+1}] with ratio rho_g
    XM = x_upper_for(c1)
    assert XM < xg, "beta1 must exceed gamma"
    grid = [X0]
    ratio = F(1003, 1000)          # Delta beta = (1/2) log(1.003) ~ 1.5e-3
    while grid[-1] < XM:
        nxt = grid[-1] * ratio
        e10 = math.floor(math.log10(float(nxt)))
        nxt = F(math.ceil(nxt * 10 ** (24 - e10)), 10 ** (24 - e10))   # 25 significant digits, rounded up
        grid.append(min(nxt, XM))
    worst_u = F(0); worst_pb = F(0); worst_at = None
    cache = {}
    def pieces(x, direction):
        key = (x, direction)
        if key in cache:
            return cache[key]
        thx = (1 - x) / (1 + x)
        nu, de = thx.numerator ** b0, thx.denominator ** b0
        Tx = F(-((-nu * SC) // de), SC) if direction == 'up' else F((nu * SC) // de, SC)
        t2 = thx * thx * 2 * Tx / (1 + Tx * Tx)
        t2 = up(t2) if direction == 'up' else dn(t2)
        cache[key] = (thx, t2)
        return thx, t2
    for xl, xh in zip(grid[:-1], grid[1:]):
        th_hi, t2_hi = pieces(xl, 'up')      # beta_hi <-> smaller x; tau2 increasing in beta
        th_lo, t2_lo = pieces(xh, 'dn')      # beta_lo <-> larger x
        m1u = th_hi
        m2u = up((th_hi + t2_hi) / (1 + th_hi * t2_hi))
        assert th_hi > t2_lo
        m3u = up((th_hi - t2_lo) / (1 - th_hi * t2_lo))
        uU = up(P1 * m1u + P2 * m2u + P3 * m3u)
        uH = up(th_hi * th_hi * uU ** k)
        pb = F(0)
        m1r, m2r, m3r = up(m1u), m2u, m3u
        for n1, n2, n3, mult in multinomials(k):
            x = up(th_hi * th_hi * m1r ** n1 * m2r ** n2 * m3r ** n3)
            pb += up(mult * P1u ** n1 * P2u ** n2 * P3u ** n3 * 2 * x / (1 + x))
        if uH > worst_u:
            worst_u = uH; worst_at = (xl, xh)
        worst_pb = max(worst_pb, pb)
    bl = -0.5 * math.log(float(worst_at[1])) / (0.5 * math.log(float(1 / xg)))
    print(f"COLD compact: {len(grid)-1} intervals cover x in [X0, XM], XM={float(XM):.6e} >= e^(-2 c1 gamma), c1={float(c1)}")
    print(f"      sup u_H <= {float(worst_u):.12f} (<1/3: {worst_u < F(1,3)}), attained on interval with beta_lo = {bl:.5f} gamma;  sup pbar_H <= {float(worst_pb):.9f} (<1/2: {worst_pb < F(1,2)})")
    allU = max(worst_u, Ubar)
    print(f"RESULT: for all beta >= {float(c1)} gamma(p): u_H(beta) <= {float(allU):.12f} < 1/3 : {allU < F(1,3)};  gamma < beta1 : {XM < xg}")


if __name__ == "__main__":
    main()
