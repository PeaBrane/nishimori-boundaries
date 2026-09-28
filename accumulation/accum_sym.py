"""Symbolic checks of identities of Appendix A.5 of the paper: Lemmas A.6 (lem:omega) and A.7 (lem:reduction),
the completion of the square behind eq. (28) (eq:cert-B), and the constants of Lemma A.9 (lem:accumulator)
and of (V2)-(V3).

Notation follows Section 5 and Appendix A.5, except that the atom mass lambda is written m and the
trajectory variable k is written D: V is a generic smooth function, m > 0 a constant,
a = x V'' - V', K2 = V'''^2 - 2 m V''^3,
Pi = x e^{-mV} [ int_0^x y int_0^y e^{mV} K2 - 2 int_0^x e^{mV} V''^2 ] + 2 V'^2,
Psi = e^{mV} Pi / x, Omega = int_0^x e^{mV} K2 - 2 e^{mV} (a^2 - m x V'^3) / x^3,
Q = 1/2 (x^2 V''' - 2a)^2 + a^2 (1 + 3 m x a - 4 m x^2 V'') + m^2 x^2 V'^4.
"""
import sympy as sp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

x, m = sp.symbols('x m', positive=True)
V = sp.Function('V')(x)
V1, V2, V3 = [sp.diff(V, x, k) for k in (1, 2, 3)]
E = sp.exp(m * V)
a = x * V2 - V1
K2 = V3**2 - 2 * m * V2**3
I1 = sp.Function('I1')(x)  # int_0^x y int_0^y e^{mV} K2
I2 = sp.Function('I2')(x)  # int_0^x e^{mV} V''^2
I3 = sp.Function('I3')(x)  # int_0^x e^{mV} K2
J = sp.Function('J')(x)    # int_0^y e^{mV} K2 as function of the outer variable
rules = {
    sp.Derivative(I1, x): x * I3,
    sp.Derivative(I2, x): E * V2**2,
    sp.Derivative(I3, x): E * K2,
}
Psi = I1 - 2 * I2 + 2 * E * V1**2 / x
Omega = I3 - 2 * E * (a**2 - m * x * V1**3) / x**3
Q = sp.Rational(1, 2) * (x**2 * V3 - 2 * a)**2 + a**2 * (1 + 3 * m * x * a - 4 * m * x**2 * V2) \
    + m**2 * x**2 * V1**4

dPsi = sp.diff(Psi, x).subs(rules)
print("Psi' - x Omega:", sp.simplify(dPsi - x * Omega))
dOm = sp.diff(Omega, x).subs(rules)
print("Omega' - 2 e^{mV} Q / x^4:", sp.simplify(sp.expand(dOm - 2 * E * Q / x**4)))
# Pi = x e^{-mV} Psi
Pi = x * sp.exp(-m * V) * Psi
print("Pi - [x e^{-mV}(I1 - 2 I2) + 2V'^2]:",
      sp.simplify(Pi - (x * sp.exp(-m * V) * (I1 - 2 * I2) + 2 * V1**2)))

# Dimensionless reduction.  L = m V', ell = x L, D = x (L - x L'), tau = log x, dot = x d/dx.
L = m * V1
ell = x * L
D = x * (L - x * sp.diff(L, x))
dot = lambda f: x * sp.diff(f, x)
print("ell_dot - (2 ell - D):", sp.simplify(dot(ell) - (2 * ell - D)))
Qt = sp.Rational(1, 2) * (dot(D) - 3 * D)**2 + D**2 * (1 - 4 * ell + D) + ell**4
print("m^2 x^2 Q - Qtilde:", sp.simplify(sp.expand(m**2 * x**2 * Q - Qt)))
omega = m**2 * x**5 * sp.exp(-m * V) * Omega
res = dot(omega).subs(rules) - ((5 - ell) * omega + 2 * m**2 * x**2 * Q)
print("omega_dot - ((5-ell) omega + 2 Qtilde):", sp.simplify(sp.expand(res)))
# m a x = -D
print("m x a + D:", sp.simplify(m * x * a + D))

# Small-x orders for a smooth even V: a = O(x^3), Q = O(x^6).
c0, c2, c4, c6 = sp.symbols('c0 c2 c4 c6')
Vs = c0 + c2 * x**2 + c4 * x**4 + c6 * x**6
sub = {V: Vs}
a_s = sp.expand(x * sp.diff(Vs, x, 2) - sp.diff(Vs, x))
Q_s = sp.expand(Q.subs({V3: sp.diff(Vs, x, 3), V2: sp.diff(Vs, x, 2), V1: sp.diff(Vs, x)}))
print("a lowest order:", sp.Poly(a_s, x).monoms()[-1], " Q lowest order:", sp.Poly(Q_s, x).monoms()[-1])

# Accumulator: B from completing the square.
l, Dv, Dd, dl = sp.symbols('ell D Ddot delta', real=True)
Y = sp.Function('Y')
Yf = Y(l, Dv)
Qt2 = sp.Rational(1, 2) * (Dd - 3 * Dv)**2 + Dv**2 * (1 - 4 * l + Dv) + l**4
rhs = 2 * Qt2 - 2 * (sp.diff(Yf, l) * (2 * l - Dv) + sp.diff(Yf, Dv) * Dd) + 2 * (5 - l) * Yf
# eta_dot = (5-l) eta + rhs, with eta = omega - 2Y.  Minimize rhs over Ddot.
Ds = sp.solve(sp.diff(rhs, Dd), Dd)[0]
Bgen = sp.expand(rhs.subs(Dd, Ds))
Yl, YD = sp.diff(Yf, l), sp.diff(Yf, Dv)
Bformula = 2 * (5 - l) * Yf - 2 * Yl * (2 * l - Dv) - 6 * Dv * YD - YD**2 + 2 * Dv**2 * (1 - 4 * l + Dv) + 2 * l**4
print("B(generic Y) - formula:", sp.simplify(Bgen - Bformula), "; d2/dDdot2 rhs =", sp.diff(rhs, Dd, 2))

# Primary certificate.
P = (222 - 25 * l + dl * (617 - 343 * l) + dl**2 * (368 * l - 833)) / 1024
Ypoly = sp.expand(l**4 * P.subs(dl, Dv / l))
Ydisp = l**2 * (l**2 * (222 - 25 * l) + l * Dv * (617 - 343 * l) + Dv**2 * (368 * l - 833)) / 1024
print("Y - displayed:", sp.simplify(Ypoly - Ydisp))
B = sp.expand(Bformula.subs(Yf, Ypoly).doit())
B = sp.expand(2 * (5 - l) * Ypoly - 2 * sp.diff(Ypoly, l) * (2 * l - Dv) - 6 * Dv * sp.diff(Ypoly, Dv)
              - sp.diff(Ypoly, Dv)**2 + 2 * Dv**2 * (1 - 4 * l + Dv) + 2 * l**4)
Bdisp = (733184 * l**4 - 198656 * l**5 - 329489 * l**6 + 423262 * l**7 - 117649 * l**8
         + Dv * (-3235840 * l**3 + 2695168 * l**4 + 2758308 * l**5 - 2051100 * l**6 + 504896 * l**7)
         + Dv**2 * (2097152 - 8388608 * l + 12320768 * l**2 - 6379520 * l**3 - 3529220 * l**4
                    + 2452352 * l**5 - 541696 * l**6)
         + Dv**3 * (2097152 - 3411968 * l + 2260992 * l**2)) / 1024**2
print("B - displayed 1024^-2 polynomial:", sp.simplify(B - Bdisp))
Bhat = sp.cancel(B.subs(Dv, dl * l) / l**2)
print("Bhat polynomial?", sp.Poly(Bhat, l, dl).is_polynomial if hasattr(sp.Poly(Bhat, l, dl), 'is_polynomial') else True,
      " bidegree:", sp.Poly(Bhat, l).degree(), sp.Poly(Bhat, dl).degree(),
      " #monomials:", len(sp.Poly(Bhat, l, dl).terms()))
print("P(ell,1) =", sp.simplify(P.subs(dl, 1)))
P38 = sp.expand(1024 * P.subs(l, sp.Rational(3, 8)))
print("1024 P(3/8,delta) =", P38)
LM = sp.Rational(26181, 10000)
PLM = sp.expand(1024 * P.subs(l, LM))
print("1024 P(LMAX,delta) =", PLM, "; vertex delta =", sp.solve(sp.diff(PLM, dl), dl))
P14 = sp.expand(1024 * P.subs(l, sp.Rational(1, 4)))
dstar = sp.solve(sp.diff(P14, dl), dl)[0]
mx = sp.nsimplify(P14.subs(dl, dstar) / 1024)
print("1024 P(1/4,delta) =", P14, "; argmax", dstar, "; max P(1/4,.) =", mx, "=", sp.N(mx, 12),
      "; 4/13 - max =", sp.nsimplify(sp.Rational(4, 13) - mx), "=", sp.N(sp.Rational(4, 13) - mx, 8))
print("1/416 check:", sp.Rational(2, 4**4) / sp.Rational(13, 4), " 2(1/4)^4 (4/13) =", 2 * sp.Rational(1, 4)**4 * sp.Rational(4, 13))
# Phase 1 minimum over D in [0, ell] of D^2 (1 - 4 ell + D)
f = Dv**2 * (1 - 4 * l + Dv)
crit = sp.solve(sp.diff(f, Dv), Dv)
print("phase-1 critical D:", crit, "; value:", sp.factor(f.subs(Dv, crit[1])),
      "; equals -(8l-2)^3/54:", sp.simplify(f.subs(Dv, crit[1]) + (8 * l - 2)**3 / 54) == 0)
yv = sp.symbols('y', positive=True)
print("d/dy (y+2)^4/y^3 =", sp.factor(sp.diff((yv + 2)**4 / yv**3, yv)), "; value at 1:", ((yv + 2)**4 / yv**3).subs(yv, 1),
      "; 4096/54 =", sp.N(sp.Rational(4096, 54), 8))
print("phase-3 bound at D=ell:", sp.factor(f.subs(Dv, l) + l**4), "; roots:", sp.solve(l**2 - 3 * l + 1, l),
      "; LMAX - (3+sqrt5)/2 =", sp.N(LM - (3 + sp.sqrt(5)) / 2, 6))
