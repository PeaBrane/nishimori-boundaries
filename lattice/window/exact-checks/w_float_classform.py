"""Float evidence (mpmath, 40 digits), a separate code path:
the class form of Proposition 9.3 (classes 1/2/3 with mean signs rho_c) for H4(b0, k).
Computes w(beta) = min_{s in [0,1]} E exp(-2 s K_H(beta)), v = E e^{-K_H}, pbar = E(1 - e^{-2|K_H|}),
the beta/gamma endpoints of {w < w_plus}, {w < w_bb}, and v_H(gamma(p)) along the Nishimori line.
"""
from mpmath import mp, mpf, tanh, atanh, log, sqrt, factorial
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

mp.dps = 40
b0, k = 1000, 10
W_PLUS = 1 / sqrt(9 + 3 * sqrt(2))
W_BB = 1 / sqrt(15)
V_P = mpf('0.2224074394521915')


def gamma(p):
    return log((1 - p) / p) / 2


def law(beta, p):
    t = tanh(beta)
    kap = atanh(t ** b0)
    a = t * t * tanh(2 * kap)
    w1 = (t + a) / (1 + t * a)
    w3 = (t - a) / (1 - t * a)
    cp, cm = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)
    q = (1 - (1 - 2 * p) ** b0) / 2
    Bp = cp * (1 - q) ** 2 + cm * q ** 2
    B0 = 2 * q * (1 - q)
    Bm = cp * q ** 2 + cm * (1 - q) ** 2
    P1, P2, P3 = (1 - p) * Bp + p * Bm, B0, (1 - p) * Bm + p * Bp
    r1, r2, r3 = ((1 - p) * Bp - p * Bm) / P1, 1 - 2 * p, ((1 - p) * Bm - p * Bp) / P3
    atoms = []  # (weight, x = |tanh K|, A = E[sign | class counts])
    for n1 in range(k + 1):
        for n2 in range(k + 1 - n1):
            n3 = k - n1 - n2
            M = factorial(k) / (factorial(n1) * factorial(n2) * factorial(n3))
            wgt = M * P1 ** n1 * P2 ** n2 * P3 ** n3
            x = t ** (2 + n2) * w1 ** n1 * w3 ** n3
            A = (1 - 2 * p) ** 2 * r1 ** n1 * r2 ** n2 * r3 ** n3
            atoms.append((wgt, x, A))
    return atoms


def Es(atoms, s):
    tot = mpf(0)
    for wgt, x, A in atoms:
        g = (1 - x) / (1 + x)  # e^{-2|K|}
        tot += wgt * ((1 + A) / 2 * g ** s + (1 - A) / 2 * g ** (-s))
    return tot


def wmin(atoms):
    lo, hi = mpf(0), mpf(1)
    gr = (sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = Es(atoms, c), Es(atoms, d)
    for _ in range(90):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = Es(atoms, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = Es(atoms, d)
    s = (lo + hi) / 2
    return Es(atoms, s), s


def pbar(atoms):
    return sum(wgt * 2 * x / (1 + x) for wgt, x, A in atoms)


if __name__ == '__main__':
    p0 = mpf(9) / 10000
    g0 = gamma(p0)
    print(f"design P: b0={b0} k={k} p0=9/10000 gamma={mp.nstr(g0, 12)}  w_plus={mp.nstr(W_PLUS, 12)} w_bb={mp.nstr(W_BB, 12)}")
    for c in ('0.5', '0.55', '0.6', '0.7', '0.8', '1.0', '1.2', '1.3', '1.35', '1.4', '1.45'):
        at = law(mpf(c) * g0, p0)
        w, s = wmin(at)
        print(f"  beta/gamma={c:5s} w={mp.nstr(w, 8)} s*={mp.nstr(s, 4)} v={mp.nstr(Es(at, mpf(1)/2), 8)} "
              f"pbar={mp.nstr(pbar(at), 6)} mass-1={mp.nstr(sum(a[0] for a in at) - 1, 3)}")


    def root(fn, lo, hi, it=45):
        flo = fn(lo)
        for _ in range(it):
            mid = (lo + hi) / 2
            fm = fn(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return (lo + hi) / 2


    for name, thr in (('w_plus', W_PLUS), ('w_bb', W_BB)):
        f = lambda c: wmin(law(c * g0, p0))[0] - thr
        lo = root(f, mpf('0.5'), mpf('0.7'))
        hi = root(f, mpf('1.2'), mpf('1.5'))
        print(f"  w < {name}: beta/gamma in ({mp.nstr(lo, 6)}, {mp.nstr(hi, 6)}), beta in ({mp.nstr(lo * g0, 6)}, {mp.nstr(hi * g0, 6)})")
    f = lambda c: pbar(law(c * g0, p0)) - mpf(1) / 2
    chot = root(f, mpf('0.3'), mpf('0.5'))
    print(f"  pbar = 1/2 on the hot side at beta/gamma = {mp.nstr(chot, 6)} (beta = {mp.nstr(chot * g0, 6)})")
    print("Nishimori line: v_H(gamma(p)) = E e^{-K_H(gamma(p))} (float)")
    for pn in (8, 9, 9.5, 10, 10.5, 11, 11.5, 12):
        p = mpf(pn) / 10000
        at = law(gamma(p), p)
        print(f"  p={pn}/10000  v={mp.nstr(Es(at, mpf(1)/2), 10)}  v<v_P: {Es(at, mpf(1)/2) < V_P}")
    f = lambda p: Es(law(gamma(p), p), mpf(1) / 2) - V_P
    pstar = root(f, mpf('0.0009'), mpf('0.0015'), it=40)
    print(f"  float crossing v_H(gamma(p)) = v_P at p = {mp.nstr(pstar, 8)}")
