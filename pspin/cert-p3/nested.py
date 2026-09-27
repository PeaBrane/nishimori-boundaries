"""Rigorous nested Gaussian expectations  E_{Z1}[ Psi(Z1, I_1(y),...,I_r(y)) ],  y = s1*Z1,
I_j(y) = E_{Z2}[ f_j(y + s2*Z2) ].  Outer and inner composite Simpson with interval 4th-derivative
remainders (Taylor arithmetic); crude zeroth-order inner quadrature for the outer remainder jets.
Tails are bounded analytically by the caller-provided growth data."""
import math
import numpy as np
from iv import IV, T, isum, tphi, SQRT2PI, iexp, iphi
from gauss import simpson

K4 = 4


def _tail_poly_exp(a, b, c, d, L):
    """bound for int_{|z|>L} e^{a+b|z|}(c+d|z|) phi(z) dz, L>b+1"""
    t = L - b
    assert t > 1
    ph = math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    return 2 * math.exp(a + b * b / 2) * ph * ((c + d * b) / t + d) * 1.01 + 1e-300


def inner_values(fs, y, s2, n2, L2, growth):
    """Tight enclosures of I_j(y) for point nodes y (1-D array). growth=(xg, poly): |f_j(Y)| <=
    e^{xg|Y|}(1+poly|Y|). Returns list of IV arrays (len(y),)."""
    out = [[] for _ in fs]
    xg, poly = growth
    CH = 40
    for i0 in range(0, len(y), CH):
        yy = y[i0:i0 + CH][:, None]
        for j, f in enumerate(fs):
            g = lambda z, f=f: f(z * s2 + IV.pt(yy)) * tphi(z)
            core = simpson(g, L2, n2)          # shape (CH,1)? simpson sums over all -> need per-row
            out[j].append(core)
    return out


def simpson_rows(gfun, L, n, rows_shape):
    """Row-wise Simpson: gfun(z) returns jets with leading axis = rows, trailing axis = z-nodes."""
    w = 2.0 * L / n
    nodes = -L + (w / 2.0) * np.arange(2 * n + 1)
    gv = gfun(T.var(IV.pt(nodes[None, :]), 0)).c[0]
    wts = np.ones(2 * n + 1); wts[1:-1:2] = 4.0; wts[2:-1:2] = 2.0
    w6 = IV(np.nextafter(w / 6.0, -np.inf), np.nextafter(w / 6.0, np.inf))
    main = isum(gv * IV.pt(wts[None, :]), axis=1) * w6
    lo_c = -L + w * np.arange(n)
    cell = IV(lo_c[None, :], np.minimum(lo_c + w, L)[None, :])
    g4 = gfun(T.var(cell, K4)).c[4] * 24.0
    c5 = IV(np.nextafter(w ** 5 / 2880.0, -np.inf), np.nextafter(w ** 5 / 2880.0, np.inf))
    rem = isum(g4, axis=1) * c5
    return main - rem


def inner_tight(f, y, s2, n2, L2, xg, poly, positive):
    """I(y)=E f(y+s2 Z2) for an IV array y (thin intervals) and IV scalar s2;
    |f(Y)| <= 50 e^{xg|Y|}(1+poly|Y|)."""
    y = y if isinstance(y, IV) else IV.pt(y)
    s2f = float(np.max(s2.hi))
    res_lo, res_hi = [], []
    CH = 24
    N = y.lo.shape[0]
    for i0 in range(0, N, CH):
        yy = IV(y.lo[i0:i0 + CH][:, None], y.hi[i0:i0 + CH][:, None])
        g = lambda z: f(z * s2 + yy) * tphi(z)
        core = simpson_rows(g, L2, n2, None)
        ymax = float(np.max(yy.mag()))
        tb = 50 * _tail_poly_exp(xg * ymax, xg * s2f, 1 + poly * ymax, poly * s2f, L2)
        lo = core.lo - (0.0 if positive else tb)
        hi = core.hi + tb
        res_lo.append(np.nextafter(lo, -np.inf)); res_hi.append(np.nextafter(hi, np.inf))
    return IV(np.concatenate(res_lo), np.concatenate(res_hi))


def inner_jets_crude(f, zcell, s1, s2, n2c, L2, xg, poly):
    """Order-4 Taylor jets (in z1) of I(s1*z1) over z1-cells (IV arrays of shape (m,)).
    Zeroth-order inner quadrature on n2c cells of [-L2,L2] plus a tail term."""
    w = 2.0 * L2 / n2c
    lo2 = -L2 + w * np.arange(n2c)
    z2 = IV(lo2[None, :], np.minimum(lo2 + w, L2)[None, :])
    ph = iphi(z2)
    wiv = IV(np.nextafter(w, -np.inf), np.nextafter(w, np.inf))
    Z1 = T.var(IV(zcell.lo[:, None], zcell.hi[:, None]), K4)
    jet = f(Z1 * s1 + T.const(z2 * s2, K4))
    # integrate each coefficient: sum over z2 of hull(coef)*phi(cell)*w ; coef intervals may straddle 0
    out = []
    s1f, s2f = float(np.max(s1.hi)), float(np.max(s2.hi))
    ymax = float(np.max(np.maximum(np.abs(zcell.lo), np.abs(zcell.hi)))) * s1f
    tb = 50 * 24 * _tail_poly_exp(xg * ymax, xg * s2f, 1 + poly * ymax, poly * s2f, L2) * (1 + s1f) ** 4
    for c in jet.c:
        prod = c * ph * wiv
        s = isum(prod, axis=1)
        out.append(IV(np.nextafter(s.lo - tb, -np.inf), np.nextafter(s.hi + tb, np.inf)))
    return T(out)


def nested_expect(psi, fs, s1, s2, n1, L1, n2, L2, n2c, growth, outer_tail):
    """E_{Z1}[psi(Z1jet, [I_j jets])] with I_j(y) = E f_j(y+s2 Z2), y = s1 Z1.
    growth[j] = (xg, poly, positive).  outer_tail = bound on |int_{|z1|>L1} psi phi|."""
    w = 2.0 * L1 / n1
    nodes = -L1 + (w / 2.0) * np.arange(2 * n1 + 1)
    Iv = [inner_tight(f, s1 * IV.pt(nodes), s2, n2, L2, *growth[j]) for j, f in enumerate(fs)]
    Z1p = T.var(IV.pt(nodes), 0)
    gv = (psi(Z1p, [T([I]) for I in Iv]) * tphi(Z1p)).c[0]
    wts = np.ones(2 * n1 + 1); wts[1:-1:2] = 4.0; wts[2:-1:2] = 2.0
    w6 = IV(np.nextafter(w / 6.0, -np.inf), np.nextafter(w / 6.0, np.inf))
    main = isum(gv * IV.pt(wts)) * w6
    lo_c = -L1 + w * np.arange(n1)
    cell = IV(lo_c, np.minimum(lo_c + w, L1))
    jets = [inner_jets_crude(f, cell, s1, s2, n2c, L2, growth[j][0], growth[j][1]) for j, f in enumerate(fs)]
    Z1c = T.var(cell, K4)
    g4 = (psi(Z1c, jets) * tphi(Z1c)).c[4] * 24.0
    c5 = IV(np.nextafter(w ** 5 / 2880.0, -np.inf), np.nextafter(w ** 5 / 2880.0, np.inf))
    rem = isum(g4) * c5
    core = main - rem
    return IV(np.nextafter(core.lo - outer_tail, -np.inf), np.nextafter(core.hi + outer_tail, np.inf))
