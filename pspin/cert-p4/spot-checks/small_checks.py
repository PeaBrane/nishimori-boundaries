"""Small float and exact checks (not a certificate): e4 identity (bias and covariance), Nishimori log c0, u0 root, strip inequalities."""
import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

rng = np.random.default_rng(0)
# (1) e4(sigma) = [N^4 m^4 - 6 N^3 m^2 + 8 N^2 m^2 + 3 N^2 - 6 N] / 24 for sigma in {+-1}^N
for N in [4, 5, 7, 9]:
    for _ in range(20):
        s = rng.choice([-1, 1], size=N)
        e4 = sum(s[i] * s[j] * s[k] * s[l] for i, j, k, l in itertools.combinations(range(N), 4))
        M = int(s.sum())  # N m
        rhs = Fr(M ** 4 - 6 * N * M ** 2 + 8 * M ** 2 + 3 * N * N - 6 * N, 24)
        assert e4 == rhs, (N, s, e4, rhs)
print("(1) e4 identity OK; bias = (24 j0/N^3) e4 = j0[N m^4 - (6 - 8/N) m^2 + 3/N - 6/N^2]")
# covariance: v e4(sigma o tau) with v = 12/N^3  ->  N R^4/2 - 3R^2 + 4R^2/N + 3/(2N) - 3/N^2
N = 8
for _ in range(20):
    s, t = rng.choice([-1, 1], size=N), rng.choice([-1, 1], size=N)
    st = s * t
    e4 = sum(st[i] * st[j] * st[k] * st[l] for i, j, k, l in itertools.combinations(range(N), 4))
    R = Fr(int(st.sum()), N)
    assert Fr(12, N ** 3) * e4 == N * R ** 4 / 2 - 3 * R ** 2 + 4 * R ** 2 / N + Fr(3, 2 * N) - Fr(3, N ** 2)
print("(1b) covariance = N R^4/2 - 3R^2 + 4R^2/N + 3/(2N) - 3/N^2 (O(1) off N xi/beta^2) OK")
# (2) Nishimori: beta = mu0/v, log c0 = C(N,4) mu0^2/(2v)
j0 = Fr(8107, 10000)
for N in [10, 50, 1000]:
    mu0, v = 24 * j0 / Fr(N) ** 3, Fr(12) / Fr(N) ** 3
    assert mu0 / v == 2 * j0
    lc0 = math.comb(N, 4) * mu0 ** 2 / (2 * v)
    assert lc0 == j0 ** 2 * N * (1 - Fr(1, N)) * (1 - Fr(2, N)) * (1 - Fr(3, N))
print("(2) beta_N = 2 j0 and log c0 = j0^2 N(1-1/N)(1-2/N)(1-3/N) OK")
# (3) u0: xi'(u) = u at u^{p-2} = 2/(p b^2)
b = 10 / 3
for p in [3, 4]:
    u0 = (2 / (p * b * b)) ** (1 / (p - 2))
    print(f"(3) p={p} u0={u0:.6f}, xi'(u0)-u0={p * b * b * u0 ** (p - 1) / 2 - u0:.1e}")
print("    exact: 2 (10/3)^2 (21/100)^3 =", float(2 * Fr(10, 3) ** 2 * Fr(21, 100) ** 3), "<= 0.21")
# (4) strip inequalities on |d| <= pi/4 (sampled)
c = np.linspace(-30, 30, 3001)[:, None]
d = np.linspace(-math.pi / 4, math.pi / 4, 201)[None, :]
Y = c + 1j * d
ch = np.cosh(Y)
x = 930529 / 2000000
kap = 2 ** (-x / 2) * math.cos(x * math.pi / 4)
lcY = np.log(ch)
r1 = np.max(np.abs(ch) / np.cosh(c))
r2 = np.min(np.real(ch ** x) / (kap * np.cosh(c) ** x))
r3 = np.max(np.abs(np.tanh(Y)))
r4 = np.max(np.abs(lcY) - (np.abs(c) + 1.2))
r5 = np.max(np.abs(ch ** (x - 1)) / (2 ** ((1 - x) / 2) * np.cosh(c) ** (x - 1)))
print(f"(4) max|coshY|/cosh c={r1:.6f} (<=1); min Re cosh^x/(kappa cosh^x c)={r2:.6f} (>=1); "
      f"max|tanh|={r3:.6f} (<=1); max(|lc Y|-|c|-6/5)={r4:.4f} (<=0); max|cosh^(x-1)|/bound={r5:.6f} (<=1)")
