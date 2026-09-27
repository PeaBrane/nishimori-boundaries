"""Exact certificate check for Lemma 5.8 (lem:accumulator) and Section 5.6 (sec:accum-cert) of the paper.

Everything is exact rational arithmetic (fractions.Fraction); no floating point enters any decision.

Input: the six coefficients of
    1024 P(l, d) = 222 - 25 l + d (617 - 343 l) + d^2 (368 l - 833).
Steps:
  1. Y(l, D) = l^4 P(l, D/l), a polynomial in (l, D).
  2. B(l, D) = 2(5-l)Y - 2Y_l(2l-D) - 6D Y_D - Y_D^2 + 2D^2(1-4l+D) + 2l^4, computed from Y
     exactly as in the paper, and compared coefficient by coefficient with the displayed 1024^2 B.
  3. Bhat(l, d) = B(l, d l) / l^2, checked to be a polynomial.
  4. (V1) Bhat > 0 on [1/4, LMAX] x [0,1];  (V2) P > 0 on [3/8, LMAX] x [0,1];
     (V3) 4/13 - P(1/4, d) > 0 on [0,1];      (H) 54 l^4 - (8l-2)^3 > 0 on [1/4, 3/8];
     each by Bernstein expansion with exact bisection (a polynomial whose Bernstein coefficients on a
     box are all positive is positive on that box).
Usage: python accum_cert.py [certificate.json]
"""
import json
import sys
from fractions import Fraction as Fr
from math import comb

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

LMAX = Fr(26181, 10000)
P_DEFAULT = {(0, 0): Fr(222, 1024), (1, 0): Fr(-25, 1024), (0, 1): Fr(617, 1024),
             (1, 1): Fr(-343, 1024), (0, 2): Fr(-833, 1024), (1, 2): Fr(368, 1024)}
B_DISPLAYED = {  # coefficients of 1024^2 B, keyed by (power of l, power of D)
    (4, 0): 733184, (5, 0): -198656, (6, 0): -329489, (7, 0): 423262, (8, 0): -117649,
    (3, 1): -3235840, (4, 1): 2695168, (5, 1): 2758308, (6, 1): -2051100, (7, 1): 504896,
    (0, 2): 2097152, (1, 2): -8388608, (2, 2): 12320768, (3, 2): -6379520, (4, 2): -3529220,
    (5, 2): 2452352, (6, 2): -541696,
    (0, 3): 2097152, (1, 3): -3411968, (2, 3): 2260992,
}


def clean(p):
    return {k: v for k, v in p.items() if v != 0}


def add(*ps):
    r = {}
    for p in ps:
        for k, v in p.items():
            r[k] = r.get(k, 0) + v
    return clean(r)


def mul(p, q):
    r = {}
    for (a, b), v in p.items():
        for (c, d), w in q.items():
            r[(a + c, b + d)] = r.get((a + c, b + d), 0) + v * w
    return clean(r)


def scal(p, s):
    return clean({k: v * s for k, v in p.items()})


def diff(p, var):
    r = {}
    for (a, b), v in p.items():
        e = (a, b)[var]
        if e:
            k = (a - 1, b) if var == 0 else (a, b - 1)
            r[k] = r.get(k, 0) + v * e
    return clean(r)


def const(c):
    return clean({(0, 0): Fr(c)})


X = {(1, 0): Fr(1)}  # first variable (l)
Z = {(0, 1): Fr(1)}  # second variable (D or d)


def build_Y(P):
    # l^4 P(l, D/l) = sum a_ij l^{i+4-j} D^j
    Y = {}
    for (i, j), v in P.items():
        assert i + 4 - j >= 0
        Y[(i + 4 - j, j)] = Y.get((i + 4 - j, j), 0) + v
    return clean(Y)


def build_B(Y):
    Yl, YD = diff(Y, 0), diff(Y, 1)
    return add(
        mul(add(const(10), scal(X, -2)), Y),                      # 2(5-l) Y
        scal(mul(Yl, add(scal(X, 2), scal(Z, -1))), -2),          # -2 Y_l (2l - D)
        scal(mul(Z, YD), -6),                                     # -6 D Y_D
        scal(mul(YD, YD), -1),                                    # -Y_D^2
        scal(mul(mul(Z, Z), add(const(1), scal(X, -4), Z)), 2),   # 2 D^2 (1 - 4l + D)
        scal(mul(mul(X, X), mul(X, X)), 2),                       # 2 l^4
    )


def build_Bhat(B):
    # B(l, d l) / l^2
    R = {}
    for (i, j), v in B.items():
        e = i + j - 2
        assert e >= 0, "B(l, d l) not divisible by l^2"
        R[(e, j)] = R.get((e, j), 0) + v
    return clean(R)


def on_box(p, a, b, c, d):
    """q(s, t) = p(a + (b-a) s, c + (d-c) t)."""
    Xs = clean({(0, 0): Fr(a), (1, 0): Fr(b) - Fr(a)})
    Ys = clean({(0, 0): Fr(c), (0, 1): Fr(d) - Fr(c)})
    ni = max((k[0] for k in p), default=0)
    nj = max((k[1] for k in p), default=0)
    xp, yp = [const(1)], [const(1)]
    for _ in range(ni):
        xp.append(mul(xp[-1], Xs))
    for _ in range(nj):
        yp.append(mul(yp[-1], Ys))
    return add(*[scal(mul(xp[i], yp[j]), v) for (i, j), v in p.items()])


def bernstein_min(q, n, m):
    best = None
    for i in range(n + 1):
        for j in range(m + 1):
            s = Fr(0)
            for k in range(i + 1):
                for l_ in range(j + 1):
                    v = q.get((k, l_))
                    if v:
                        s += Fr(comb(i, k), comb(n, k)) * Fr(comb(j, l_), comb(m, l_)) * v
            best = s if best is None or s < best else best
    return best


def certify_pos(p, a, b, c, d, maxdepth=40):
    """Prove p > 0 on [a,b] x [c,d]. Returns (ok, number of boxes examined, min certified Bernstein bound)."""
    n = max((k[0] for k in p), default=0)
    m = max((k[1] for k in p), default=0)
    stack = [(Fr(a), Fr(b), Fr(c), Fr(d), 0)]
    boxes, lo = 0, None
    while stack:
        a_, b_, c_, d_, dep = stack.pop()
        boxes += 1
        bm = bernstein_min(on_box(p, a_, b_, c_, d_), n if b_ > a_ else 0, m if d_ > c_ else 0)
        if bm > 0:
            lo = bm if lo is None or bm < lo else lo
            continue
        if dep >= maxdepth:
            return False, boxes, None
        wide_x = b_ > a_ and (d_ == c_ or (b_ - a_) / (Fr(b) - Fr(a)) >= (d_ - c_) / (Fr(d) - Fr(c)))
        if wide_x:
            h = (a_ + b_) / 2
            stack += [(a_, h, c_, d_, dep + 1), (h, b_, c_, d_, dep + 1)]
        else:
            h = (c_ + d_) / 2
            stack += [(a_, b_, c_, h, dep + 1), (a_, b_, h, d_, dep + 1)]
    return True, boxes, lo


def evalp(p, x, y):
    return sum(v * Fr(x) ** i * Fr(y) ** j for (i, j), v in p.items())


def main():
    P = dict(P_DEFAULT)
    if len(sys.argv) > 1:
        raw = json.load(open(sys.argv[1]))["P_coefficients_l^i_delta^j"]
        P = {tuple(map(int, k.split(","))): Fr(v) for k, v in raw.items()}
    print("1024 P coefficients (l^i d^j):", {k: v * 1024 for k, v in sorted(P.items())})
    Y = build_Y(P)
    B = build_B(Y)
    Bd = {k: Fr(v, 1024 ** 2) for k, v in B_DISPLAYED.items()}
    same = add(B, scal(Bd, -1)) == {}
    print("B derived from Y equals the displayed 1024^-2 polynomial:", same)
    Bhat = build_Bhat(B)
    print("Bhat = B(l, d l)/l^2: %d monomials, bidegree (%d, %d)"
          % (len(Bhat), max(k[0] for k in Bhat), max(k[1] for k in Bhat)))
    # Consistency: Bhat(l,d) l^2 == B(l, d l) at a few rational points.
    for (lv, dv) in [(Fr(1, 3), Fr(2, 7)), (Fr(5, 2), Fr(1, 1)), (Fr(7, 5), Fr(0))]:
        assert evalp(Bhat, lv, dv) * lv ** 2 == evalp(B, lv, dv * lv)
    v1 = certify_pos(Bhat, Fr(1, 4), LMAX, 0, 1)
    print("(V1) Bhat > 0 on [1/4, 26181/10000] x [0,1]: ok=%s boxes=%d certified lower bound=%s (~%.4f)"
          % (v1[0], v1[1], v1[2], float(v1[2]) if v1[2] is not None else float('nan')))
    v2 = certify_pos(P, Fr(3, 8), LMAX, 0, 1)
    print("(V2) P > 0 on [3/8, 26181/10000] x [0,1]: ok=%s boxes=%d certified lower bound=%s (~%.5f)"
          % (v2[0], v2[1], v2[2], float(v2[2])))
    P14 = {(j, 0): v for (i, j), v in on_box(P, Fr(1, 4), Fr(1, 4), 0, 1).items()}  # q(d) = P(1/4, d)
    v3 = certify_pos(add(const(Fr(4, 13)), scal(P14, -1)), 0, 1, 0, 0)
    print("(V3) 4/13 - P(1/4, d) > 0 on [0,1]: ok=%s boxes=%d" % (v3[0], v3[1]))
    H = add(scal(mul(mul(X, X), mul(X, X)), 54),
            scal(mul(mul(add(scal(X, 8), const(-2)), add(scal(X, 8), const(-2))), add(scal(X, 8), const(-2))), -1))
    v4 = certify_pos(H, Fr(1, 4), Fr(3, 8), 0, 0)
    print("(H) 54 l^4 - (8l-2)^3 > 0 on [1/4, 3/8]: ok=%s boxes=%d" % (v4[0], v4[1]))
    # Reported margins (exact evaluation on a rational grid; informative only, not part of the proof).
    N, M = 400, 200
    best = None
    for i in range(N + 1):
        lv = Fr(1, 4) + (LMAX - Fr(1, 4)) * i / N
        for j in range(M + 1):
            val = evalp(Bhat, lv, Fr(j, M))
            if best is None or val < best[0]:
                best = (val, lv, Fr(j, M))
    print("grid min of Bhat (%dx%d exact rational grid): %.5f at (l, d) = (%.4f, %.4f)"
          % (N, M, float(best[0]), float(best[1]), float(best[2])))
    print("P(l, 1) =", sum(v for (i, j), v in P.items() if i == 0),
          "+ l *", sum(v for (i, j), v in P.items() if i == 1))
    ok = same and v1[0] and v2[0] and v3[0] and v4[0]
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
