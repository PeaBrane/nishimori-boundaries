"""Re-certification 2: interval re-check of the inputs (C1)-(C3), (C7) in ../out/claims.json (Appendix C.7).

Written from the law formulas (Proposition 9.3 + sign law derived separately here); shares no code
with cert_window.py / cert_arb.py. Uses mpmath.iv (outward-rounded interval arithmetic) and fractions.Fraction.
"""
import json
import os
import sys
import time
from fractions import Fraction as Fr
from math import comb

from mpmath import iv

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

iv.dps = 45
CLAIMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "claims.json")
B0, K = 1000, 10
P0 = Fr(9, 10000)
T0 = time.time()


def F2iv(x):
    x = Fr(x)
    return iv.mpf(x.numerator) / iv.mpf(x.denominator)


def classes():
    out = []
    for n1 in range(K + 1):
        for n2 in range(K + 1 - n1):
            n3 = K - n1 - n2
            out.append((n1, n2, n3, comb(K, n1) * comb(K - n1, n2)))
    return out


CL = classes()
assert len(CL) == 66


def unit_law(p):
    """Exact rationals: class probs P_c and signed masses N_c = E[s_d ; class c] (c = 1 disagree, 2 non-cancel, 3 cancel)."""
    th = 1 - 2 * p
    e = th ** B0              # E[eps 1{c1=c2}] = (1-q)^2 - q^2 = 1-2q
    q = (1 - e) / 2
    P1 = 2 * q * (1 - q)
    c = (1 - q) ** 2 + q ** 2
    P2 = (c + th ** 3 * e) / 2
    P3 = (c - th ** 3 * e) / 2
    assert P1 + P2 + P3 == 1
    N1 = th * P1
    N2 = (th * c + th ** 2 * e) / 2
    N3 = (th * c - th ** 2 * e) / 2
    return th, (P1, P2, P3), (N1, N2, N3)


def weights(p):
    """Per class: (w_plus, w_minus) as iv, where w_plus - w_minus = M th^2 prod N^n and w_plus + w_minus = M prod P^n."""
    th, P, N = unit_law(p)
    Pi = [F2iv(x) for x in P]
    Ni = [F2iv(x) for x in N]
    th2 = F2iv(th * th)
    ws = []
    for n1, n2, n3, M in CL:
        a = M * Pi[0] ** n1 * Pi[1] ** n2 * Pi[2] ** n3
        b = M * th2 * Ni[0] ** n1 * Ni[1] ** n2 * Ni[2] ** n3
        ws.append(((a + b) / 2, (a - b) / 2, a))
    return ws


def theta_of(beta):
    t = iv.exp(-2 * F2iv(beta))
    return 1 - 2 * t / (1 + t)


def tau_of(th):
    a = th ** (B0 + 2)
    return 2 * a / (1 + th ** (2 * B0))


def mags(th, tau_for_m2, tau_for_m3, th_m3=None):
    if th_m3 is None:
        th_m3 = th
    m1 = th
    m2 = (th + tau_for_m2) / (1 + th * tau_for_m2)
    m3 = (th_m3 - tau_for_m3) / (1 - th_m3 * tau_for_m3)
    return m1, m2, m3


def xs(th, m):
    return [th * th * m[0] ** n1 * m[1] ** n2 * m[2] ** n3 for n1, n2, n3, _ in CL]


def F_point(ws, x, s):
    """E exp(-2 s K) with |tanh K| = x_n; r = e^{-2|K|} = (1-x)/(1+x)."""
    s = F2iv(s)
    tot = iv.mpf(0)
    for (wp, wm, _), xn in zip(ws, x):
        lr = iv.log((1 - xn) / (1 + xn))
        tot += wp * iv.exp(s * lr) + wm * iv.exp(-s * lr)
    return tot


def F_ball_upper(ws, x_lo, x_hi, s):
    """Monotone bracket: plus-sign term r^s is decreasing in x (max at x_lo); minus-sign r^{-s} increasing (max at x_hi)."""
    s = F2iv(s)
    tot = iv.mpf(0)
    for (wp, wm, _), a, b in zip(ws, x_lo, x_hi):
        tot += wp * iv.exp(s * iv.log((1 - a) / (1 + a))) + wm * iv.exp(-s * iv.log((1 - b) / (1 + b)))
    return tot


def Pfun(w):
    return 9 * w ** 4 / (1 - 9 * w ** 2) ** 2


def Qfun(w):
    return 9 * w ** 3 / (1 - 9 * w ** 2)


def h(w):
    return 4 * Pfun(w) + 4 * Qfun(w)


d = json.load(open(CLAIMS))
report = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    report.append(msg)


say("iv_recheck.py: mpmath", iv.dps, "digits interval; Fraction exact")

# ---- thresholds (exact) ----
WS = Fr(d["thresholds"]["W_STAR"]["value"])
WB = Fr(d["thresholds"]["W_BB"]["value"])
say("T: 9WS^2<1", 9 * WS ** 2 < 1, " 2P(WS)<1", 2 * Pfun(WS) < 1, " 1-2P(WS)=", float(1 - 2 * Pfun(WS)),
    " 15WB^2<1", 15 * WB ** 2 < 1)
V0 = Fr(20529504599, 10 ** 11)
say("I-gamma0: 3V<1", 3 * V0 < 1, " 1-2P(V)=", float(1 - 2 * Pfun(V0)), " in [0.91700,0.91701):",
    Fr(91700, 10 ** 5) <= 1 - 2 * Pfun(V0) < Fr(91701, 10 ** 5), " 1-4P(V)=", float(1 - 4 * Pfun(V0)))

# ---- ordering (exact + interval) ----
bh = Fr(d["hot"]["beta_hot"]); b1 = Fr(d["W_plus"]["beta_1"]); b2 = Fr(d["W_plus"]["beta_2"]); bc = Fr(349277, 50000)
g0 = iv.log(F2iv(Fr(9991, 9))) / 2
say("order: bh<b1", bh < b1, " b2<bc", b2 < bc, " b1<gamma0<b2 (iv):", F2iv(b1) < g0, g0 < F2iv(b2),
    " gamma0 =", g0)

# ---- W_plus / W_two_point cells: point enclosure at bl, rational chain, Lemma-L-free monotone ball ----
ws0 = weights(P0)
for key, thr in (("W_plus", WS), ("W_two_point", WB)):
    cells = d[key]["cells"]
    ok_chain = ok_point = ok_ball = ok_s = ok_contig = True
    worst_point = None
    worst_ball = Fr(0)
    prev = Fr(d[key]["beta_1"])
    for c in cells:
        bl, br, s = Fr(c["bl"]), Fr(c["br"]), Fr(c["s"])
        Fup, y, Wup = Fr(c["F_up"]), Fr(c["y"]), Fr(c["W_up"])
        ok_contig &= (bl == prev) and (br > bl)
        prev = br
        ok_s &= (0 < s <= 1)
        ok_chain &= (y == 6 * s * (br - bl)) and (0 <= y < 2) and (Fup * (2 + y) / (2 - y) <= Wup <= thr)
        th = theta_of(bl)
        ta = tau_of(th)
        Fp = F_point(ws0, xs(th, mags(th, ta, ta)), s)
        ok_point &= (Fp.b <= F2iv(Fup).a)
        rel = (F2iv(Fup).a - Fp.b) / Fp.b
        worst_point = rel if worst_point is None or rel < worst_point else worst_point
        # Lemma-L-free: monotone bracket over [bl, br]
        thl, thr_ = theta_of(bl), theta_of(br)
        tal, tah = tau_of(thl), tau_of(thr_)
        m3lo = (thl - tah) / (1 - thl * tah)
        ok_ball &= (m3lo.a > 0)  # lower bracket of every factor must be >= 0 for the product bound
        x_lo = xs(thl, (thl, (thl + tal) / (1 + thl * tal), m3lo))
        x_hi = xs(thr_, (thr_, (thr_ + tah) / (1 + thr_ * tah), (thr_ - tal) / (1 - thr_ * tal)))
        Fb = F_ball_upper(ws0, x_lo, x_hi, s)
        ok_ball &= (Fb.b <= F2iv(thr).a)
    ok_contig &= (prev == Fr(d[key]["beta_2"]))
    say(f"{key}: {len(cells)} cells; contiguous {ok_contig}; 0<s<=1 {ok_s}; exact chain {ok_chain}; "
        f"iv F(bl,s)<=F_up {ok_point} (min rel slack {float(worst_point.a):.3e}); Lemma-L-free monotone-bracket ball <= thr {ok_ball}"
        f"  [{time.time()-T0:.1f}s]")

# ---- hot: point values and a Lemma-H-free monotone-bracket cover of (0, beta_hot] ----
th = theta_of(bh); ta = tau_of(th)
x = xs(th, mags(th, ta, ta))
pb = sum(w[2] * 2 * xn / (1 + xn) for w, xn in zip(ws0, x))
uu = sum(w[2] * xn for w, xn in zip(ws0, x))
say("hot point: pbar(beta_hot) in", pb, "<= stored", Fr(d["hot"]["pbar_upper_at_beta_hot"]) >= Fr(0) and pb.b <= F2iv(d["hot"]["pbar_upper_at_beta_hot"]).a,
    "; u(beta_hot) in", uu)
# cover: pieces in beta; sup over [a,b] of pbar <= value at bracket x_hi(a,b); pbar(0)=0.
def pbar_sup(a, b):
    tha, thb = theta_of(a), theta_of(b)
    taa, tab = tau_of(tha), tau_of(thb)
    xh = xs(thb, (thb, (thb + tab) / (1 + thb * tab), (thb - taa) / (1 - thb * taa)))
    return sum(w[2] * 2 * xn / (1 + xn) for w, xn in zip(ws0, xh))

def cover(a, b, depth=0):
    val = pbar_sup(a, b)
    if val.b < 0.5:
        return 1, val.b, True
    if depth > 14:
        return 1, val.b, False
    m = (a + b) / 2
    n1_, s1_, o1_ = cover(a, m, depth + 1)
    n2_, s2_, o2_ = cover(m, b, depth + 1)
    return n1_ + n2_, max(s1_, s2_), o1_ and o2_

n_pieces = 0
sup_b = 0
ok_cover = True
a = Fr(0)
step = Fr(1, 50)
while a < bh:
    b = min(bh, a + step)
    n_, s_, o_ = cover(a, b)
    n_pieces += n_
    sup_b = max(sup_b, s_)
    ok_cover &= o_
    a = b
sup_all = iv.mpf([0, sup_b])
say(f"hot cover (0, beta_hot] with monotone brackets (Lemma-H-free, monotone-bracket form): all pieces pbar < 1/2: {ok_cover}; "
    f"pieces {n_pieces}; sup upper {float(sup_all.b):.12f}  [{time.time()-T0:.1f}s]")

# ---- Nishimori line: v_H(gamma(p1)) both via signed F(gamma,1/2) and via sech form ----
vplo = Fr(d["nishimori_line"]["vP_bracket"][0]); vphi = Fr(d["nishimori_line"]["vP_bracket"][1])
say("vP bracket exact: h(lo)<1", h(vplo) < 1, " h(hi)>1", h(vphi) > 1)
for p1, Vs in ((Fr(1, 1000), "54076342043/250000000000"), (Fr(529, 500000), "111203052643/500000000000"), (P0, "20529504599/100000000000")):
    V = Fr(Vs)
    ws = weights(p1)
    th = F2iv(1 - 2 * p1)
    ta = tau_of(th)
    x = xs(th, mags(th, ta, ta))
    Fhalf = F_point(ws, x, Fr(1, 2))
    sech = sum(w[2] * iv.sqrt(1 - xn * xn) for w, xn in zip(ws, x))
    say(f"N p={p1}: F(gamma,1/2) in {Fhalf}; E sech in {sech}; <= V: {Fhalf.b <= F2iv(V).a}; V<vP_lo exact {V < vplo}; h(V)<1 exact {h(V) < 1}")
# sharpness point
ps = Fr(10581, 10 ** 7)
ws = weights(ps); th = F2iv(1 - 2 * ps); ta = tau_of(th)
Fs = F_point(ws, xs(th, mags(th, ta, ta)), Fr(1, 2))
say(f"N sharpness p={ps}: v in {Fs}; > vP_hi: {Fs.a > F2iv(vphi).b}")
# Lemma H theta_bar
tb = Fr(d["hot"]["monotonicity"]["theta_bar"])
say("Lemma H: tanh(5/2) <= theta_bar:", theta_of(Fr(5, 2)).b <= F2iv(tb).a, " (H*) exact:",
    2 * (B0 + 2) * tb ** (B0 + 1) <= 1 - 4 * tb ** (2 * B0 + 4))
say(f"done [{time.time()-T0:.1f}s]")
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(report) + "\n")
