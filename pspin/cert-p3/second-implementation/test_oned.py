# nonrigorous cross-check of oned.expect against scipy quad
import time
from fractions import Fraction as Fr
import numpy as np
from scipy import integrate
import oned, rig
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

x = Fr(554371, 10**6)
for kind in oned.KINDS:
    for c in [Fr(0), Fr(468599, 10**6), Fr(-3, 2)]:
        for s in [Fr(8094, 1000), Fr(1, 10), Fr(23, 10)]:
            t = time.time()
            I, info = oned.expect(kind, x, c, s)
            dt = time.time() - t
            xf = float(x); cf = float(c); sf = float(s)
            def lc(z): z = abs(z); return z + np.log1p(np.exp(-2*z)) - np.log(2)
            f = {"coshx": lambda z: np.exp(xf*lc(z)), "logcosh": lc,
                 "coshx_logcosh": lambda z: np.exp(xf*lc(z))*lc(z),
                 "coshx_tanh2": lambda z: np.exp(xf*lc(z))*np.tanh(z)**2}[kind]
            ref = integrate.quad(lambda g: f(cf + np.sqrt(sf)*g)*np.exp(-g*g/2)/np.sqrt(2*np.pi), -40, 40, epsabs=1e-14, epsrel=1e-14, limit=500)[0]
            lo, hi = rig.flo(I), rig.fhi(I)
            print(f"{kind:14s} c={float(c):+.3f} s={float(s):.3f} [{lo:.16g},{hi:.16g}] w={hi-lo:.1e} ref={ref:.16g} in={lo-1e-13<=ref<=hi+1e-13} K={info['K']} disc={info['disc']:.1e} tail={info['tail']:.1e} {dt:.2f}s")
