#!/usr/bin/env python3
"""Exact identities showing that condition (N2) of Section 7.5 (sec:nm-fm) follows from signs that
../kappa-cert/kappa_cert.py already certifies (Lemma 7.12, lem:nm-closed). Symbols: W = Omega, mu, tau; NL-reduced entries at M.

  lambda = 1 - W(1 - 2mu + tau),   B = 1 - 3mu + 2tau
  S = [[1-mu, -W(mu-tau)], [-W(mu-tau), (W/2)(1 - W(1-4mu+3tau))]]   (Hessian of P(delta_q,h))
  H = [[W((1-mu)W-1), -W^2(mu-tau)], [-W^2(mu-tau), S_qq]]            (Nishimori (49))

I1  det H = (W^2/2) lambda (W B - 1)
I2  det S = (W/2) (lambda B + 2(mu - tau))
I3  det H = W det S (W - S_qq/det S)       (so Btilde'' = det H/(W det S))
I4  A_FM(direct) = A_FM(simplified)  given (Lambda_c/beta_c)^2 as a free symbol r2
"""

import json

import sympy as sp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

W, mu, tau, r2, mup = sp.symbols("W mu tau r2 mup")
lam = 1 - W * (1 - 2 * mu + tau)
B = 1 - 3 * mu + 2 * tau
Shh, Shq, Sqq = 1 - mu, -W * (mu - tau), W / 2 * (1 - W * (1 - 4 * mu + 3 * tau))
detS = sp.expand(Shh * Sqq - Shq**2)
fmm, fmq, fqq = W * ((1 - mu) * W - 1), -W**2 * (mu - tau), Sqq
detH = sp.expand(fmm * fqq - fmq**2)
out = {
    "I1 detH = (W^2/2) lam (W B - 1)": sp.expand(detH - W**2 / 2 * lam * (W * B - 1)) == 0,
    "I2 detS = (W/2)(lam B + 2(mu - tau))": sp.expand(detS - W / 2 * (lam * B + 2 * (mu - tau))) == 0,
    "I3 detH = W detS (W - Sqq/detS)": sp.simplify(detH - W * detS * (W - Sqq / detS)) == 0,
}
fmb = W * sp.sqrt(r2) * B            # phi_mb = W (Lambda_c/beta_c) B
A_direct = mup / 2 + r2 * B + fmb**2 * (fmm + 2 * fmq + fqq) / detH
A_simpl = mup / 2 - r2 * W**2 * B * lam / (2 * detH)
out["I4 A_FM direct = simplified"] = sp.simplify(sp.together(A_direct - A_simpl)) == 0
out["I5 hsum = -(W/2) lam"] = sp.expand(fmm + 2 * fmq + fqq + W / 2 * lam) == 0
print(json.dumps(out, indent=1))
