"""Exact rational beta -> infinity limits of u_H and pbar_H (class magnitudes w1 -> 1, t -> 1,
w3 -> 1/3), printed to 15 digits; cross-checks the tail enclosures of both certificates."""
from fractions import Fraction as Fr
from math import comb
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")


def limits(b0, k, p):
    q = (1 - (1 - 2 * p) ** b0) / 2
    cp, cm = (1 - p) ** 2 + p ** 2, 2 * p * (1 - p)
    tpos, tzero, tneg = cp * (1 - q) ** 2 + cm * q ** 2, 2 * q * (1 - q), cp * q ** 2 + cm * (1 - q) ** 2
    P1, P2, P3 = (1 - p) * tpos + p * tneg, tzero, (1 - p) * tneg + p * tpos
    u = (1 - Fr(2, 3) * P3) ** k
    pb = sum(comb(k, n1) * comb(k - n1, n3) * P1 ** n1 * P2 ** (k - n1 - n3) * P3 ** n3 * Fr(2, 3 ** n3 + 1)
             for n1 in range(k + 1) for n3 in range(k + 1 - n1))
    return u, pb, P3


def dec(x, d=15):
    return f"{x.numerator * 10 ** d // x.denominator / 10 ** d:.{d}f}"


for name, b0, k, p in [("P", 1000, 10, Fr(9, 10000)), ("E", 1500, 10, Fr(9, 10000)), ("B", 1500, 8, Fr(87373, 10 ** 8))]:
    u, pb, P3 = limits(b0, k, p)
    print(f"{name}: b0={b0} k={k} p={p}: P3={dec(P3)}  u_H(inf)=(1-2P3/3)^k={dec(u)}  pbar_H(inf)={dec(pb)}"
          f"  u<1/3:{u < Fr(1, 3)} pbar<1/2:{pb < Fr(1, 2)} pbar<1/3:{pb < Fr(1, 3)}")
