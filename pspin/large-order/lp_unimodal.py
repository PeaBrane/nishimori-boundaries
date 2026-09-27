"""Rigorous (mpmath.iv) inputs of the cell route to Lemma 7.3 (lem:nm-U, "Lemma U"; Appendix B.2, app:nm-U).

Lemma 7.3 also has a cell-free proof, which needs no interval input.  This
script certifies the inputs of the second, computer-assisted route, for integer p >= 3:
the elasticity e(r) = r w(r)/mu(r) crosses each level 1/(p-1) exactly once.

mpmath.mp.dps = 40 >= iv.dps + 20, so the conversions mpf(x.a), mpf(x.b)
of interval endpoints are exact; displayed minima and margins are rounded down.

Notation: h = r + sqrt(r) z, mu(r) = E tanh h, w(r) = E sech^4 h = mu'(r),
eps(r) = e^{-r/2}/sqrt(2 pi r), I_k(r) = int_R e^{-h^2/(2r)} sech^k(h) dh, so that
w = eps I_3 and 1 - mu = eps I_1 (Appendix B.3).

Checks (all are decision checks).
 U1  K := int h^2 sech^3 <= K_up (cell upper bound; the closed form is pi^3/8 - pi).
 U2a R_plus = 0.82: pi R^2 - K R - 2K > 0, so U(r) < 1/2 for r >= R_plus;
 U2b R_0 = 1.41: pi R^2 - (pi+K) R - K > 0, so U(r) < 0 for r >= R_0;
 U2c 0.82 > K/pi, so U is defined on [0.82, oo);
     here U(r) = 1/2 - r/2 + K/(pi r - K) bounds r (log e)'(r) + e(r) for r > K/pi.
 U3  small r: e(r) >= w(r) >= 1 - 2r - 2r^2 >= 0.52 on (0, 0.2].
 U4a/b cells [r_k, r_k + 1/200] covering [0.2, 1.41]:
       e(r) >= r_k eps(r_{k+1}) I3_lo(r_k) / min(r_{k+1}, 1 - eps(r_{k+1}) I1_lo(r_k)),
     with I_lo right-endpoint sums (the integrands decrease in |h|; the tail h > 10 dropped).
     Required: e > 1/2 on [0.2, R_plus] and e > 1/3 on [0.2, R_0].
 U4c the cells cover [0.2, 1.41].
Output: JSON on stdout; exit 1 if a check fails.
"""

import json
import sys

import mpmath
from mpmath import iv, mpf

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

iv.dps = 20
mpmath.mp.dps = 40
PI = iv.pi
out = {}
fails = []


def check(name, cond, detail):
    out[name] = {"ok": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)


def cosh(x):
    return (iv.exp(x) + iv.exp(-x)) / 2


def sech(x):
    return 1 / cosh(x)


def nstr(x, n=10):
    return f"[{mpmath.nstr(mpf(x.a), n)}, {mpmath.nstr(mpf(x.b), n)}]"


def down(v, nd):
    return mpmath.nstr(mpmath.floor(v * 10**nd) / 10**nd, nd + 2, strip_zeros=False)


# U1: K upper bound by cells of width 1/1024 on [0, 12] plus tail 8 h^2 e^{-3h}
K_up = iv.mpf(0)
NC = 1024
for j in range(NC * 12):
    a = iv.mpf(j) / NC
    b = iv.mpf(j + 1) / NC
    K_up += (b - a) * b * b * sech(a) ** 3
H = iv.mpf(12)
K_up = 2 * (K_up + 8 * iv.exp(-3 * H) * (H * H / 3 + 2 * H / 9 + iv.mpf(2) / 27))
K = mpf(K_up.b)
check("U1 K <= K_up, closed form pi^3/8 - pi below it", mpf((PI**3 / 8 - PI).b) <= K, f"K_up = {mpmath.nstr(K, 12)}")

# U2
Rp = iv.mpf("0.82")
R0 = iv.mpf("1.41")
Kiv = iv.mpf(K)
q1 = PI * Rp**2 - Kiv * Rp - 2 * Kiv
q0 = PI * R0**2 - (PI + Kiv) * R0 - Kiv
check("U2a U(r) < 1/2 for r >= 0.82 (pi r^2 - K r - 2K > 0 at 0.82; increasing for r > K/(2pi))", mpf(q1.a) > 0,
      f"{nstr(q1)}; rounded down {down(mpf(q1.a), 4)}")
check("U2b U(r) < 0 for r >= 1.41 (pi r^2 - (pi+K) r - K > 0 at 1.41; increasing for r > (pi+K)/(2pi))", mpf(q0.a) > 0,
      f"{nstr(q0)}; rounded down {down(mpf(q0.a), 4)}")
check("U2c 0.82 > K/pi (U defined)", mpf((Rp - Kiv / PI).a) > 0, nstr(Rp - Kiv / PI))

# U3
r02 = iv.mpf("0.2")
v3 = 1 - 2 * r02 - 2 * r02**2
check("U3 1 - 2r - 2r^2 > 1/2 at r = 0.2 (decreasing in r)", mpf(v3.a) > 0.5, nstr(v3))

# U4 cells
DH = iv.mpf(1) / 128
NH = 128 * 10
hs = [DH * j for j in range(1, NH + 1)]
s3 = [sech(h) ** 3 for h in hs]
s1 = [sech(h) for h in hs]
h2 = [h * h for h in hs]


def I_lo(r):
    t3 = iv.mpf(0)
    t1 = iv.mpf(0)
    for j in range(NH):
        g = iv.exp(-h2[j] / (2 * r))
        t3 += g * s3[j]
        t1 += g * s1[j]
    return 2 * DH * t3, 2 * DH * t1


def eps(r):
    return iv.exp(-r / 2) / iv.sqrt(2 * PI * r)


STEP = iv.mpf(1) / 200
grid = [iv.mpf("0.2") + STEP * k for k in range(0, 243)]  # 0.2 .. 1.41
Ivals = [I_lo(r) for r in grid]
evals = [eps(r) for r in grid]
cells = []
for k in range(len(grid) - 1):
    ra, rb = grid[k], grid[k + 1]
    I3a, I1a = Ivals[k]
    epb = evals[k + 1]
    w_lo = epb * I3a
    mu_hi = iv.mpf(min(mpf(rb.b), mpf((1 - epb * I1a).b)))
    e_lo = ra * w_lo / mu_hi
    cells.append((mpf(ra.a), mpf(rb.b), mpf(e_lo.a)))
min_half = min(c[2] for c in cells if c[0] < 0.82)
min_third = min(c[2] for c in cells)
check("U4a e > 1/2 on [0.2, 0.82] (cells)", min_half > 0.5, f"min lower bound, rounded down: {down(min_half, 4)}")
check("U4b e > 1/3 on [0.2, 1.41] (cells)", min_third > mpf(1) / 3, f"min lower bound, rounded down: {down(min_third, 4)}")
check("U4c cells cover [0.2, 1.41]", cells[0][0] <= mpf("0.2") and cells[-1][1] >= mpf("1.41")
      and all(cells[i][1] >= cells[i + 1][0] for i in range(len(cells) - 1)), f"{len(cells)} cells")
out["sample cell lower bounds (r_a, r_b, e_lo rounded down to 6 d.p.)"] = [
    [mpmath.nstr(c[0], 6), mpmath.nstr(c[1], 6), down(c[2], 6)] for c in cells[::30]]
json.dump(out, sys.stdout, indent=1, default=str)
print()
print(f"{len(out) - 1 - len(fails)} of {len(out) - 1} decision checks pass, {len(fails)} fail: {fails}", file=sys.stderr)
sys.exit(1 if fails else 0)
