"""SymPy recheck of the algebra and constants of hand proofs in Section 4 of the paper: Lemma 4.8
(lem:y) (c),(d), Corollary 4.12 (cor:envelope) and Remark 4.14 (rem:spiked).  It asserts nothing and
certifies nothing the proofs rely on; it prints each quantity (stored output: hand_checks.out)."""
import sympy as sp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

d=sp.symbols('delta',nonnegative=True)
b=1+d
Q=2*b**6-sp.Rational(32,5)*d*b**10-(2+d)
R=sp.expand(sp.cancel(Q/d))
print('R =',R)
print('R(0) =',R.subs(d,0))
Rhand=11+30*d+40*d**2+30*d**3+12*d**4+2*d**5-sp.Rational(32,5)*(1+d)**10
print('R - Rhand =',sp.expand(R-Rhand))
# hand bound: (21/20)^10 <= e^{1/2} < 1.65, 32/5*1.65 < 11
print('(21/20)^10 =',sp.N(sp.Rational(21,20)**10,12),' e^0.5=',sp.N(sp.exp(sp.Rational(1,2)),12))
print('(21/20)^10 < 33/20 :',sp.Rational(21,20)**10<sp.Rational(33,20))
print('32/5*33/20 =',sp.Rational(32,5)*sp.Rational(33,20),'< 11:',sp.Rational(32,5)*sp.Rational(33,20)<11)
# the target inequality: b^2-1-2 d b^6 + 32/5 d^2 b^10 = -d*Q
lhs=b**2-1-2*d*b**6+sp.Rational(32,5)*d**2*b**10
print('lhs + d*Q =',sp.expand(lhs+d*Q))
# X at delta=1/20
X=(sp.Rational(21,20))**2*sp.sqrt(sp.Rational(12,20))
print('X(1/20)=',sp.N(X,8),' sqrt3=',sp.N(sp.sqrt(3),8))
# factorization for delta_c
u=sp.symbols('u')
print(sp.factor(u**4-2*u**3+1))
tau=sp.nsolve(u**3-u**2-u-1,u,1.8)
print('tau=',tau,' delta_c=',sp.sqrt(2*sp.log(tau))-1)
# envelope arithmetic: delta=h^4/(192 beta^4) -> sqrt(12 delta)=h^2/(4beta^2)
h,B=sp.symbols('h beta',positive=True)
dl=h**4/(192*B**4)
print('sqrt(12 delta) =',sp.simplify(sp.sqrt(12*dl)))
lower=dl/2*(h**2/(2*B)-h**2/(4*B))
print('margin =',sp.simplify(lower))
# strict: h^2/(beta(1+delta)) >= h^2/(2beta)? needs delta<=1
# Remark 4.14 (rem:spiked): 4(2d+d^2)/(1+d)^4 <= 8d
expr=8*d-4*(2*d+d**2)/(1+d)**4
print('8d - q-bound numerator factor:',sp.factor(sp.together(expr)))
# check lambda=(1+d)^2: 4(lambda-1)/lambda^2
lam=(1+d)**2
print(sp.simplify(4*(lam-1)/lam**2-4*(2*d+d**2)/(1+d)**4))
print('sqrt(3/2)-1 =',sp.N(sp.sqrt(sp.Rational(3,2))-1,6))
