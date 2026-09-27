"""Model code of re-certification 1 for design P = H4(b0, k): the signed law of K_H(beta) and its functionals.

Derivation (separate from claims.json and the certificate code; cf. Proposition 9.3 of the paper):
  * A path of b0 edges with signs J_1..J_b0 has tanh K = prod tanh(beta J_i) = c theta^b0, c = prod J_i
    (series rule), so K_path = c kappa with kappa = atanh(theta^b0), theta = tanh(beta).
  * Two parallel paths: K = (c1 + c2) kappa in {2 kappa, 0, -2 kappa}.
  * Connectors e_1, e_2 in series: tanh K = s1 s2 theta^2 tanh((c1+c2) kappa), i.e.
    K = s1 s2 ((c1+c2)/2) kappa_2 with kappa_2 = atanh(theta^2 tanh(2 kappa)) (tanh, atanh odd).
  * Direct edge in parallel: K_U = s_d beta + b kappa_2, b = s1 s2 (c1+c2)/2 in {-1,0,1}.
    Since 0 <= kappa_2 < beta:  K_U = s_d (beta + s_d b kappa_2),  sign K_U = s_d,
    |K_U| = beta + (s_d b) kappa_2.   Class 1: s_d b = 0 (|tanh K_U| = theta);
    class 2: s_d b = +1 (tanh(beta+kappa_2)); class 3: s_d b = -1 (tanh(beta-kappa_2)).
  * Gadget: tanh K_H = tanh(beta J_a) tanh(beta J_b) prod_i tanh K_{U_i}  (series rule), so
    |tanh K_H| = theta^2 prod_i m_{c_i} and sign K_H = J_a J_b prod_i s_d^(i).
The joint law of (class, s_d) of one unit is obtained by enumerating the 32 configurations of
(s_d, s_1, s_2, c_1, c_2); the law of (class counts n, sign K_H) by a k-fold convolution (DP).
All probabilities are exact integers over explicit common denominators (no gcd normalisation).
"""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import comb, factorial

B0 = 1000
K = 10
P0 = Fraction(9, 10000)


class Law:
    """Exact signed law of K_H for H4(b0,k) at disorder p (rational)."""

    def __init__(self, p, b0=B0, k=K):
        p = Fraction(p)
        self.p, self.b0, self.k = p, b0, k
        a, d = p.numerator, p.denominator
        self.a, self.d = a, d
        # q = P(c = -1) for c = product of b0 iid signs = (1 - (1-2p)^b0)/2
        self.q_num = d**b0 - (d - 2 * a) ** b0
        self.q_den = 2 * d**b0
        qn, Dq = self.q_num, self.q_den
        ps = {1: d - a, -1: a}  # numerators over d
        pc = {1: Dq - qn, -1: qn}  # numerators over Dq
        self.DU = d**3 * Dq**2
        U = defaultdict(int)
        for sd, s1, s2, c1, c2 in product((1, -1), repeat=5):
            w = ps[sd] * ps[s1] * ps[s2] * pc[c1] * pc[c2]
            b = s1 * s2 * ((c1 + c2) // 2)  # coefficient of kappa_2 in K_U
            rel = sd * b
            cls = {0: 1, 1: 2, -1: 3}[rel]
            U[(cls, sd)] += w
        assert sum(U.values()) == self.DU
        self.U = dict(U)
        # k-fold convolution: state (n1,n2,n3, sign) -> integer numerator over DU^j
        st = {((0, 0, 0), 1): 1}
        for _ in range(k):
            new = defaultdict(int)
            for (n, sg), w in st.items():
                for (cls, sd), wu in self.U.items():
                    m = list(n)
                    m[cls - 1] += 1
                    new[(tuple(m), sg * sd)] += w * wu
            st = new
        # terminal edges e_a, e_b: sign J_a J_b
        jj = {1: (d - a) ** 2 + a**2, -1: 2 * a * (d - a)}
        fin = defaultdict(lambda: [0, 0])
        for (n, sg), w in st.items():
            for j, wj in jj.items():
                fin[n][0 if sg * j == 1 else 1] += w * wj
        self.D = self.DU**k * d**2
        self.atoms = {n: tuple(v) for n, v in sorted(fin.items())}  # n -> (N_plus, N_minus) over self.D
        assert sum(x + y for x, y in self.atoms.values()) == self.D

    # ---- exact class probabilities and sign means (Fractions) from the enumeration
    def unit_class(self, c):
        Pp = Fraction(self.U.get((c, 1), 0), self.DU)
        Pm = Fraction(self.U.get((c, -1), 0), self.DU)
        return Pp + Pm, (Pp - Pm) / (Pp + Pm)

    # ---- Proposition 9.3 / claims.json formulas, for comparison
    def formula_classes(self):
        p = self.p
        q = Fraction(self.q_num, self.q_den)
        th = 1 - 2 * p
        aa = (1 + th**3) / 2
        P1 = 2 * q * (1 - q)
        P3 = (1 - q) ** 2 * (1 - aa) + q**2 * aa
        P2 = 1 - P1 - P3
        cp = (1 - p) ** 2 + p**2
        cm = 2 * p * (1 - p)
        r1 = 1 - 2 * p
        r2 = ((1 - q) ** 2 * ((1 - p) * cp - p * cm) + q**2 * ((1 - p) * cm - p * cp)) / P2
        r3 = ((1 - q) ** 2 * ((1 - p) * cm - p * cp) + q**2 * ((1 - p) * cp - p * cm)) / P3
        return {"q": q, "a": aa, "P": (P1, P2, P3), "rho": (r1, r2, r3)}

    def check_gadget_formula(self):
        """Exact integer check: N_plus + N_minus = M(n) prod (U_c)^{n_c} d^2 and
        N_plus - N_minus = M(n) (d-2a)^2 prod (U_c^+ - U_c^-)^{n_c}."""
        k, a, d = self.k, self.a, self.d
        tot = {c: self.U.get((c, 1), 0) + self.U.get((c, -1), 0) for c in (1, 2, 3)}
        dif = {c: self.U.get((c, 1), 0) - self.U.get((c, -1), 0) for c in (1, 2, 3)}
        bad = 0
        count = 0
        for n1 in range(k + 1):
            for n2 in range(k + 1 - n1):
                n3 = k - n1 - n2
                n = (n1, n2, n3)
                M = factorial(k) // (factorial(n1) * factorial(n2) * factorial(n3))
                S = M * tot[1] ** n1 * tot[2] ** n2 * tot[3] ** n3 * d**2
                Dd = M * (d - 2 * a) ** 2 * dif[1] ** n1 * dif[2] ** n2 * dif[3] ** n3
                Np, Nm = self.atoms[n]
                bad += (Np + Nm != S) or (Np - Nm != Dd)
                count += 1
        return count, bad


# ---------------------------------------------------------------- magnitudes (exact rational at gamma)
def mags_exact_theta(th, b0=B0):
    th = Fraction(th)
    t = th**b0
    tau = 2 * th * th * t / (1 + t * t)
    m2 = (th + tau) / (1 + th * tau)
    m3 = (th - tau) / (1 - th * tau)
    return th, tau, m2, m3


# ---------------------------------------------------------------- engines
class FxiEngine:
    name = "fxi (own fixed-point intervals, ints only)"

    def __init__(self):
        import fxi

        self.fx = fxi
        self.I = fxi.I

    def const(self, fr):
        return self.I.frac(Fraction(fr))

    def ratio(self, N, D):
        W = self.fx.W
        return self.I((N << W) // D, self.fx.cdiv(N << W, D))

    def exp(self, x):
        return self.fx.exp(x)

    def log(self, x):
        return self.fx.log(x)

    def sqrt(self, x):
        return self.fx.sqrt(x)

    def hi(self, x):
        return x.upper()

    def lo(self, x):
        return x.lower()

    def point(self, x, which):
        v = x.lo if which == "lo" else x.hi
        return self.I(v, v)

    def nonneg(self, x):
        return self.I(max(0, x.lo), max(0, x.hi))


class IvEngine:
    name = "mpmath.iv interval arithmetic"

    def __init__(self, prec=256):
        from mpmath import iv

        iv.prec = prec
        self.iv = iv

    def const(self, fr):
        fr = Fraction(fr)
        return self.iv.mpf(fr.numerator) / self.iv.mpf(fr.denominator)

    def ratio(self, N, D):
        return self.iv.mpf(N) / self.iv.mpf(D)

    def exp(self, x):
        return self.iv.exp(x)

    def log(self, x):
        return self.iv.log(x)

    def sqrt(self, x):
        return self.iv.sqrt(x)

    def hi(self, x):
        from mpmath import libmp

        return Fraction(*libmp.to_rational(x._mpi_[1]))

    def lo(self, x):
        from mpmath import libmp

        return Fraction(*libmp.to_rational(x._mpi_[0]))

    def point(self, x, which):
        return self.const(self.lo(x) if which == "lo" else self.hi(x))

    def nonneg(self, x):
        # operate on the raw endpoint tuples (no conversion through the mp context precision)
        from mpmath import libmp

        a, b = x._mpi_
        z = libmp.fzero
        return self.iv.make_mpf((z if libmp.mpf_lt(a, z) else a, z if libmp.mpf_lt(b, z) else b))


class MpEngine:
    name = "mpmath mp floats (non-rigorous)"

    def __init__(self, dps=50):
        import mpmath

        mpmath.mp.dps = dps
        self.m = mpmath

    def const(self, fr):
        fr = Fraction(fr)
        return self.m.mpf(fr.numerator) / fr.denominator

    def ratio(self, N, D):
        sh = 4 * self.m.mp.prec
        return self.m.mpf((N << sh) // D) / self.m.mpf(2) ** sh

    def exp(self, x):
        return self.m.exp(x)

    def log(self, x):
        return self.m.log(x)

    def sqrt(self, x):
        return self.m.sqrt(x)

    def nonneg(self, x):
        return x


def law_in_engine(E, law):
    return {n: (E.ratio(Np, law.D), E.ratio(Nm, law.D)) for n, (Np, Nm) in law.atoms.items()}


def weights_in_engine(E, law):
    return {n: E.ratio(Np + Nm, law.D) for n, (Np, Nm) in law.atoms.items()}


def mags_theta(E, th, b0=B0):
    t = th**b0
    tau = 2 * th * th * t / (1 + t * t)
    m2 = (th + tau) / (1 + th * tau)
    m3 = (th - tau) / (1 - th * tau)
    return th, tau, m2, m3


def mags_beta(E, beta, b0=B0):
    """beta: engine number (point)."""
    x = E.exp(-2 * beta)
    th = E.nonneg((1 - x) / (1 + x))  # theta = tanh(beta) >= 0 for beta >= 0 (all uses have beta >= 0)
    return mags_theta(E, th, b0)


def xs(E, mags, law):
    th, _, m2, m3 = mags
    k = law.k
    pw = [[None] * (k + 1) for _ in range(3)]
    for c, m in enumerate((th, m2, m3)):
        pw[c][0] = E.const(1)
        for j in range(1, k + 1):
            pw[c][j] = pw[c][j - 1] * m
    th2 = th * th
    return {n: th2 * pw[0][n[0]] * pw[1][n[1]] * pw[2][n[2]] for n in law.atoms}


def F_value(E, lawE, xmap, s):
    """F(beta, s) = E exp(-2 s K_H) = sum_n P+_n r_n^s + P-_n r_n^{-s}, r_n = (1-x_n)/(1+x_n)."""
    tot = E.const(0)
    for n, (Pp, Pm) in lawE.items():
        x = xmap[n]
        L = E.log((1 - x) / (1 + x))
        tot = tot + Pp * E.exp(s * L) + Pm * E.exp(-(s * L))
    return tot


def u_value(E, wE, xmap):
    tot = E.const(0)
    for n, w in wE.items():
        tot = tot + w * xmap[n]
    return tot


def pbar_value(E, wE, xmap):
    # 2x/(1+x) = 2 - 2/(1+x): single occurrence of x, so interval evaluation is tight
    tot = E.const(0)
    for n, w in wE.items():
        tot = tot + w * (2 - 2 / (1 + xmap[n]))
    return tot


def v_signed(E, lawE, xmap):
    """E e^{-K} = sum P+ sqrt(r) + P- / sqrt(r)."""
    tot = E.const(0)
    for n, (Pp, Pm) in lawE.items():
        x = xmap[n]
        sr = E.sqrt((1 - x) / (1 + x))
        tot = tot + Pp * sr + Pm / sr
    return tot


def v_sech(E, wE, xmap):
    """E sech K = sum w sqrt(1 - x^2) (valid only on the Nishimori line, Lemma C.2)."""
    tot = E.const(0)
    for n, w in wE.items():
        x = xmap[n]
        tot = tot + w * E.sqrt((1 - x) * (1 + x))
    return tot


def gamma_float(p, dps=30):
    import mpmath

    mpmath.mp.dps = dps
    p = Fraction(p)
    return mpmath.log(mpmath.mpf((1 - p).numerator * p.denominator) / (p.numerator * (1 - p).denominator)) / 2


# ---------------------------------------------------------------- Peierls functions (exact)
def P_(v):
    v = Fraction(v)
    return 9 * v**4 / (1 - 9 * v * v) ** 2


def Q_(v):
    v = Fraction(v)
    return 9 * v**3 / (1 - 9 * v * v)


def h_(v):
    return 4 * P_(v) + 4 * Q_(v)
