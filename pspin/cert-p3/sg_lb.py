"""Frank-Wolfe lower bound for the zero-field 3-spin free energy at beta=5/2:
F >= P(mu*) + min_u D(u), mu* = x delta_0 + (1-x) delta_q.  Formulas: Proposition 6.8 (prop:fw) and Lemma 6.9 (lem:D)."""
import sys, json, math
from fractions import Fraction as Fr
import numpy as np
from iv import *
from gauss import expect
from nested import nested_expect, _tail_poly_exp

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

beta = Fr(5, 2)
B = IV.exact_rational(5, 2)
xr, qr = Fr(554371, 1000000), Fr(929223, 1000000)
X = IV.exact_rational(xr.numerator, xr.denominator); Q = IV.exact_rational(qr.numerator, qr.denominator)
def xi(u): return B*B*u*u*u*0.5
def dxi(u): return B*B*u*u*1.5
def ddxi(u): return B*B*u*3.0
def theta(u): return B*B*u*u*u
xf, qf, bf = float(xr), float(qr), float(beta)
sq = isqrt(dxi(Q)); sqf = float(sq.hi)

def f_coshx(Y): return tpow_pos(tcosh(Y), X)
def f_coshx_tanh(Y): return tpow_pos(tcosh(Y), X) * ttanh(Y)
def f_coshx_lc(Y): return tpow_pos(tcosh(Y), X) * tlogcosh(Y)
def f_cosh_lc(Y): return tcosh(Y) * tlogcosh(Y)
def f_cosh_tanh2(Y): t = ttanh(Y); return tcosh(Y) * t * t

A = expect(lambda z: f_coshx(z*sq), 1.0, xf*sqf, n=4000)
AK = expect(lambda z: f_coshx_lc(z*sq), sqf, xf*sqf + 1, n=4000)
K = AK / A
logA = ilog(A)
P = LOG2 + logA*X.recip() + (B*B*0.5 - dxi(Q) + (1-X)*theta(Q))*0.5
C = -(logA*(X*X).recip()) + K*X.recip() - theta(Q)*0.5
D0 = (1-X)*C
Dq = -(X*C)
print("A", A, "K", K); print("P(mu*)", P); print("D(0)", D0, "D(q)", Dq)

def D_and_Gamma(u, n1=300, n2=500, n2c=300):
    U = IV.exact_rational(u.numerator, u.denominator)
    uf = float(u)
    if uf < qf:
        s1 = float(isqrt(dxi(U)).hi); s2iv = isqrt(dxi(Q) - dxi(U)); s2 = float(s2iv.hi)
        s1iv = isqrt(dxi(U))
        # use point floats for s1,s2 scale: exactness of s1,s2 is not needed? -> they ARE needed; use IV midpoint?
        L1 = xf*s1 + 13.0; L2 = xf*s2 + 13.0
        S_tail = _tail_poly_exp(math.log(2) + xf*xf*s2*s2/2, xf*s1, xf*xf*s2*s2/2 + math.log(2), xf*s1, L1)
        S = nested_expect(lambda z, I: I[0]*tlog(I[0]), [f_coshx], s1iv, s2iv, n1, L1, n2, L2, n2c,
                          [(xf, 0.0, True)], S_tail)
        G_tail = _tail_poly_exp(math.log(2) + xf*xf*s2*s2/2, xf*s1, 1.0, 0.0, L1)
        Gn = nested_expect(lambda z, I: I[1]*I[1]/I[0], [f_coshx, f_coshx_tanh], s1iv, s2iv, n1, L1, n2, L2, n2c,
                           [(xf, 0.0, True), (xf, 0.0, False)], G_tail)
        D = logA*X.recip() + (1-X)*K*X.recip() - S*(X*X*A).recip() + (theta(U) - (1-X)*theta(Q))*0.5
        return D, Gn/A
    else:
        sgiv = isqrt(dxi(U) - dxi(Q)); sg = float(sgiv.hi)
        L1 = sqf + 13.0; L2 = sg + 13.0
        T_tail = _tail_poly_exp(math.log(2) + sg*sg/2, sqf, 1 + sg*sg + sg, sqf, L1)
        T1 = nested_expect(lambda z, I: tpow_pos(tcosh(z*sq), X - 1) * I[0], [f_cosh_lc], sq, sgiv, n1, L1, n2, L2, n2c,
                           [(1.0, 1.0, True)], T_tail)
        G_tail = _tail_poly_exp(sg*sg/2, sqf, 1.0, 0.0, L1)
        G1 = nested_expect(lambda z, I: tpow_pos(tcosh(z*sq), X - 1) * I[0], [f_cosh_tanh2], sq, sgiv, n1, L1, n2, L2, n2c,
                           [(1.0, 0.0, True)], G_tail)
        e = iexp((dxi(U) - dxi(Q))*0.5)
        D = logA*X.recip() + (dxi(U) - dxi(Q))*0.5 - T1*(A*e).recip() + (X*theta(Q) + theta(U) - theta(Q))*0.5
        return D, G1/(A*e)

if __name__ == "__main__":
    import time
    for u in [Fr(1,5), Fr(3,5), Fr(9,10), Fr(97,100)]:
        t0 = time.time()
        D, G = D_and_Gamma(u)
        print(f"u={float(u)} D=[{D.lo},{D.hi}] w={D.hi-D.lo:.2e} Gamma=[{G.lo},{G.hi}] t={time.time()-t0:.1f}s", flush=True)
