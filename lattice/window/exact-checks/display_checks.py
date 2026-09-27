"""Checks behind the numbers displayed in Section 9, Appendix C.7 and Tables 3-4 of the paper.

Exact (fractions.Fraction) unless marked:
  (1) the hand ordering chain e^{2 beta_1} < e^4 < 55 < 9991/9 < 7600 < e^9 < e^{2 beta_2}, with rational
      bounds on e from its Taylor series (lower: partial sum; upper: partial sum + 2/(n+1)!);
  (2) interval (mpmath.iv, 60 digits) enclosures of beta/gamma0 for every displayed beta, printed with
      floor/ceil roundings so that displayed interval endpoints can be rounded inward;
  (3) safe-direction decimal roundings of claims.json bounds (lower bounds down, upper bounds up);
  (4) the sign-disorder floor sum_n w_n sqrt(1 - A_n^2) versus v_H(gamma0) (float evidence, 50 digits; the
      identity itself is Proposition 9.11, prop:lat-floor).
Reads ../out/claims.json (sha256 printed). Run:
  python display_checks.py > display_checks.out
"""
import hashlib
import json
import math
from fractions import Fraction as Fr
from pathlib import Path

from mpmath import iv, mp, mpf, sqrt

import w_float_classform as W
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

CLAIMS = Path(__file__).resolve().parent.parent / 'out' / 'claims.json'
raw = CLAIMS.read_bytes()
C = json.loads(raw)
print('claims.json sha256', hashlib.sha256(raw).hexdigest())


def floor_dec(x, d):
    return Fr(math.floor(x * 10 ** d), 10 ** d)


def ceil_dec(x, d):
    return Fr(math.ceil(x * 10 ** d), 10 ** d)


def dec(q, d):
    return f"{float(q):.{d}f}"


# (1) ordering chain, exact
n = 12
part = sum(Fr(1, math.factorial(j)) for j in range(n + 1))
e_lo, e_hi = part, part + Fr(2, math.factorial(n + 1))
b1 = Fr(C['W_plus']['beta_1'])
b2 = Fr(C['W_plus']['beta_2'])
bh = Fr(C['hot']['beta_hot'])
bc = Fr(349277, 50000)
print('(1) e in [', dec(e_lo, 12), ',', dec(e_hi, 12), ']')
print('    2 beta_1 = %s < 4: %s' % (2 * b1, 2 * b1 < 4))
print('    e^4 < 55: %s (e_hi^4 = %s)' % (e_hi ** 4 < 55, dec(e_hi ** 4, 6)))
print('    55 < 9991/9 < 7600: %s' % (55 < Fr(9991, 9) < 7600))
print('    7600 < e^9: %s (e_lo^9 = %s)' % (7600 < e_lo ** 9, dec(e_lo ** 9, 3)))
print('    9 < 2 beta_2 = %s: %s' % (2 * b2, 9 < 2 * b2))
print('    hence beta_1 < gamma0 = (1/2) log(9991/9) < beta_2; also beta_hot < beta_1: %s, beta_2 < beta_c: %s'
      % (bh < b1, b2 < bc))

# (2) beta / gamma0 enclosures
iv.dps = 60
g0 = iv.log(iv.mpf(9991) / 9) / 2
print('(2) gamma0 in', iv.nstr(g0, 20))


def raw_to_fr(raw):
    sign, man, exp, _ = raw
    v = Fr(man) * (Fr(2) ** exp)
    return -v if sign else v


def ratio(q):
    r = iv.mpf(q.numerator) / q.denominator / g0
    lo, hi = r._mpi_
    return raw_to_fr(lo), raw_to_fr(hi)


named = [
    ('beta_hot', bh), ('beta_hot3', Fr(C['hot']['beta_hot3'])), ('beta_1 (W+)', b1), ('beta_2 (W+)', b2),
    ("beta_1' (W2)", Fr(C['W_two_point']['beta_1'])), ("beta_2' (W2)", Fr(C['W_two_point']['beta_2'])),
    ('beta_c', bc), ('5/2 (Lemma H range)', Fr(5, 2)),
]
for bd in C['W_plus']['bands']:
    named += [('band lo ' + bd['lo'], Fr(bd['lo'])), ('band hi ' + bd['hi'], Fr(bd['hi']))]
for pt in C['sharpness_W']['points']:
    named.append(('sharp W ' + pt['kind'] + ' ' + pt['beta'], Fr(pt['beta'])))
named += [('sharp hot ' + C['hot']['sharpness']['beta'], Fr(C['hot']['sharpness']['beta'])),
          ('sharp hot3 ' + C['hot']['sharpness']['beta3'], Fr(C['hot']['sharpness']['beta3'])),
          ('cold sharp 698553/100000', Fr(698553, 100000))]
for name, q in named:
    lo, hi = ratio(q)
    print(f"    {name:28s} beta = {float(q):.6f}; beta/gamma0 in [{float(lo):.10f}, {float(hi):.10f}]; "
          f"floor4 {dec(floor_dec(lo, 4), 4)} ceil4 {dec(ceil_dec(hi, 4), 4)} | "
          f"floor6 {dec(floor_dec(lo, 6), 6)} ceil6 {dec(ceil_dec(hi, 6), 6)}")
b63 = iv.log(2 + iv.sqrt(3)) / 2
print('    (1/2) log(2+sqrt3) in', iv.nstr(b63, 15), '; / gamma0 in', iv.nstr(b63 / g0, 12))

# (3) safe-direction roundings of claims.json numbers
print('(3) lower bounds rounded down, upper bounds rounded up')
for pt in C['sharpness_W']['points']:
    wl = Fr(pt['w_lower'])
    print(f"    w_H({pt['beta']}) >= {float(wl):.13f}: display >= {dec(floor_dec(wl, 6), 6)} or {dec(floor_dec(wl, 9), 9)}")
h = C['hot']
for key, d in (('pbar_upper_at_beta_hot', 9), ('u_upper_at_beta_hot', 9), ('pbar_upper_at_beta_hot3', 9)):
    v = Fr(h[key])
    print(f"    {key} = {float(v):.15f}: display <= {dec(ceil_dec(v, d), d)} / {dec(ceil_dec(v, 12), 12)}")
for key, v in h['lemma_H_free_cover'].items():
    s = Fr(v['sup'])
    print(f"    lemma_H_free_cover {key} sup = {float(s):.13f}: display <= {dec(ceil_dec(s, 8), 8)}")
for key in ('pbar_lower', 'u_lower', 'pbar3_lower'):
    v = Fr(h['sharpness'][key])
    print(f"    hot sharpness {key} = {float(v):.15f}: display >= {dec(floor_dec(v, 9), 9)}")
N = C['nishimori_line']
vplo = Fr(N['vP_bracket'][0])
for e in N['p1']:
    V = Fr(e['V'])
    marg = vplo - V
    print(f"    p1 = {e['p1']}: V = {float(V):.12f}; vP_lo - V = {float(marg):.6e} (display >= {dec(floor_dec(marg * 10 ** 6, 2), 2)}e-6); "
          f"relative to vP_lo >= {dec(floor_dec(marg / vplo * 100, 2), 2)} %; h(V) float {e['hV_float']!r}")
vs = Fr(N['sharpness']['v_lower'])
print(f"    N sharpness v_lower = {float(vs):.15f}: display >= {dec(floor_dec(vs, 12), 12)}")
for e in N['display']:
    lo, hi = Fr(e['v_lower']), Fr(e['v_upper'])
    print(f"    display p = {e['p']}: v_H in [{float(lo):.15f}, {float(hi):.15f}]: truncated 6 digits {dec(floor_dec(lo, 6), 6)}")
ws = Fr(C['thresholds']['W_STAR']['value'])
P = lambda v: 9 * v ** 4 / (1 - 9 * v * v) ** 2
m = 1 - 2 * P(ws)
print(f"    1 - 2P(W_STAR) = {float(m):.10e}: display >= {dec(floor_dec(m * 10 ** 5, 2), 2)}e-5")
wmin = min(Fr(c['W_up']) for c in C['W_plus']['cells'])
print(f"    min W_up over W+ cells = {float(wmin):.12f} (display {dec(floor_dec(wmin, 6), 6)})")
p0 = Fr(9, 10000)
for p1 in (Fr(1, 1000), Fr(529, 500000)):
    print(f"    p1 - p0 = {p1 - p0} = {float(p1 - p0):.3e}; (p1 - p0)/p0 = {float((p1 - p0) / p0) * 100:.4f} %")

# (4) sign-disorder floor (float evidence, 50 digits)
mp.dps = 50
pm = mpf(9) / 10000
gm = W.gamma(pm)
atoms = W.law(gm, pm)
floor_ = sum(wt * sqrt(1 - A * A) for wt, x, A in atoms)
v_g = W.Es(atoms, mpf(1) / 2)
print('(4) sign-disorder floor sum_n w_n sqrt(1 - A_n^2) =', mp.nstr(floor_, 20))
print('    v_H(gamma0) = F(gamma0, 1/2)                  =', mp.nstr(v_g, 20), '; difference', mp.nstr(floor_ - v_g, 3))
print('    max_n |A_n - x_n(gamma0)| =', mp.nstr(max(abs(A - x) for wt, x, A in atoms), 3), '(per-class Nishimori relation)')
for c in ('0.3', '0.7', '0.8', '1.2', '2.0'):
    at = W.law(mpf(c) * gm, pm)
    fl = sum(wt * sqrt(1 - A * A) for wt, x, A in at)
    wv, s = W.wmin(at)
    print(f"    beta/gamma0 = {c}: floor = {mp.nstr(fl, 12)} (beta-independent); w_H = {mp.nstr(wv, 9)} at s* = {mp.nstr(s, 4)}")
q = (1 - (1 - 2 * pm) ** 12) / 2
print('    for comparison: q := P(K_H < 0) =', mp.nstr(q, 8), '; 2 sqrt(q(1-q)) =', mp.nstr(2 * sqrt(q * (1 - q)), 12),
      '(not a lower bound on w_H)')
for c in ('0.65', '0.7', '0.75', '0.8'):
    at = W.law(mpf(c) * gm, pm)
    wv, s = W.wmin(at)
    xs = [x for wt, x, A in at if wt > mpf('1e-6')]
    print(f"    beta/gamma0 = {c}: w_H = {mp.nstr(wv, 12)} at s* = {mp.nstr(s, 4)}; |tanh K_H| over atoms of weight > 1e-6 "
          f"in [{mp.nstr(min(xs), 8)}, {mp.nstr(max(xs), 8)}]")
