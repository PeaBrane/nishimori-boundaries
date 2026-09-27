"""Float evidence only (Section 7.6, sec:nm-proof): end-to-end check of the cold/warm comparison of
Theorem 7.5 (thm:nm-main) for small p.

Shares no code with the certificate programs. Trapezoid in z (exponentially accurate for these analytic integrands).
For p in P_LIST:
  1. solve (2) via D_p(Lambda)=0; get beta1, mu.
  2. SG branch at beta1+b: solve C1=D1=0 in (m,q); phiSG=P(m,q;beta).
  3. FM critical value: solve grad U=0 in (m,h,q) at (beta,j0); phiFM=U.
  4. finite-difference A_SG, A_FM, kappa_p; first derivatives of phiFM.
  5. cold gap phiSG(beta1+b(d)) - phiFM(beta1+b(d), j0M+d) vs predicted beta1 mu^p d.
  6. cold (c): NL-diagonal trial bound at the cold point vs phiSG, away from mu.
  7. warm: Phi_c(mu) > 0 at c=2 j0^2.
  8. S entries of Hess_{(h,q)} P(delta_q,h) vs the NL closed forms.
"""
import json
import sys
import numpy as np

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

Z = np.arange(-14.0, 14.0 + 1e-12, 0.004)
W = np.exp(-Z**2 / 2) / np.sqrt(2 * np.pi) * 0.004


def lc(x):
    a = np.abs(x)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2.0)


def E(f):
    return float(np.dot(W, f))


def solve_newton(F, x0, tol=1e-13, it=60, hstep=1e-7):
    x = np.array(x0, float)
    for _ in range(it):
        f = np.array(F(x))
        J = np.zeros((len(x), len(x)))
        for k in range(len(x)):
            e = np.zeros(len(x)); e[k] = hstep
            J[:, k] = (np.array(F(x + e)) - np.array(F(x - e))) / (2 * hstep)
        dx = np.linalg.solve(J, -f)
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            break
    return x, float(np.max(np.abs(np.array(F(x)))))


def run(p):
    out = {"p": p}
    # 1. triple point
    def Dp(L):
        h = L + np.sqrt(L) * Z
        return E(lc(h)) - L / 2 - (p - 1) / (2 * p) * L * E(np.tanh(h))
    lo, hi = 0.2, 2 * p * np.log(2) + 1
    # sign scan then bisection on first sign change
    grid = np.linspace(lo, hi, 400)
    vals = [Dp(L) for L in grid]
    k = next(i for i in range(len(grid) - 1) if vals[i] < 0 <= vals[i + 1])
    a, b = grid[k], grid[k + 1]
    for _ in range(80):
        c = (a + b) / 2
        if Dp(c) < 0:
            a = c
        else:
            b = c
    Lc = (a + b) / 2
    hc = Lc + np.sqrt(Lc) * Z
    mu = E(np.tanh(hc))
    beta1 = np.sqrt(2 * Lc / (p * mu ** (p - 1)))
    j0M = beta1 / 2
    out.update(Lambda_c=Lc, mu=mu, beta1=beta1, NL_check=E(np.tanh(hc)) - E(np.tanh(hc) ** 2))

    xi = lambda q, be: be**2 * q**p / 2
    xi1 = lambda q, be: p * be**2 * q ** (p - 1) / 2
    xi2 = lambda q, be: p * (p - 1) * be**2 * q ** (p - 2) / 2
    th = lambda q, be: (p - 1) * xi(q, be)

    # 2. SG two-atom functional
    def P2(m, q, be):
        y = np.sqrt(xi1(q, be)) * Z
        A = E(np.exp(m * lc(y)))
        return np.log(2) + np.log(A) / m + (xi(1, be) - xi1(q, be)) / 2 + (1 - m) * th(q, be) / 2

    def CD(x, be):
        m, q = x
        y = np.sqrt(xi1(q, be)) * Z
        w = np.exp(m * lc(y))
        A = E(w)
        K = E(w * lc(y)) / A
        T = E(w * np.tanh(y) ** 2) / A
        return [-np.log(A) / m**2 + K / m - th(q, be) / 2, T - q]

    phiPM = lambda be: np.log(2) + be**2 / 4

    def phiSG(b):
        be = beta1 + b
        if b == 0:
            return phiPM(be), (1.0, mu)
        x, res = solve_newton(lambda x: CD(x, be), [1 - 0.97 * b, mu])
        return P2(x[0], x[1], be), (x[0], x[1], res)

    # 3. FM
    def U(x, be, j0):
        m, h, q = x
        r = xi1(q, be)
        return (be * j0 * m**p + np.log(2) + E(lc(h + np.sqrt(r) * Z))
                + (xi(1, be) - xi(q, be) - (1 - q) * r) / 2 - h * m)

    def gradU(x, be, j0):
        m, h, q = x
        r = xi1(q, be)
        y = h + np.sqrt(r) * Z
        return [p * be * j0 * m ** (p - 1) - h, E(np.tanh(y)) - m,
                xi2(q, be) / 2 * (q - E(np.tanh(y) ** 2))]

    def phiFM(be, j0):
        x, res = solve_newton(lambda x: gradU(x, be, j0), [mu, Lc, mu])
        return U(x, be, j0), x, res

    f0, x0, r0 = phiFM(beta1, j0M)
    out["phiFM_minus_phiPM_at_M"] = f0 - phiPM(beta1)
    out["FM_crit_at_M_minus_(mu,Lc,mu)"] = [x0[0] - mu, x0[1] - Lc, x0[2] - mu]

    # 4. derivatives
    hb = 2e-3
    fp = phiFM(beta1 + hb, j0M)[0]; fm = phiFM(beta1 - hb, j0M)[0]
    dB = (fp - fm) / (2 * hb)
    d2B = (fp - 2 * f0 + fm) / hb**2
    hj = 1e-4
    dJ = (phiFM(beta1, j0M + hj)[0] - phiFM(beta1, j0M - hj)[0]) / (2 * hj)
    A_FM = 0.5 - d2B
    # SG: one-sided second difference via Richardson on (phiSG - phiPM)/b^2
    bs = [0.004, 0.002, 0.001]
    ratios = [(phiSG(b)[0] - phiPM(beta1 + b)) / b**2 for b in bs]
    # ratio = -A_SG/2 + c1 b + ...
    r1 = 2 * ratios[1] - ratios[0]
    r2 = 2 * ratios[2] - ratios[1]
    A_SG = -2 * r2
    out.update(dphiFM_dbeta=dB, beta1_over_2=beta1 / 2, dphiFM_dj0=dJ, beta1_mu_p=beta1 * mu**p,
               A_FM_fd=A_FM, A_SG_fd=A_SG, A_SG_richardson_pair=[-2 * r1, -2 * r2],
               kappa_fd=beta1**2 * (A_FM - A_SG))

    # 5. cold gap along b(d)
    rows = []
    for d in [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]:
        b = np.sqrt(4 * beta1 * mu**p * d / (A_FM - A_SG))
        sg, sgx = phiSG(b)
        fm_, fx, fres = phiFM(beta1 + b, j0M + d)
        gap = sg - fm_
        rows.append({"delta": d, "b": b, "b_gt_2delta": bool(b > 2 * d), "m_branch": sgx[0], "q_branch": sgx[1],
                     "gap_SG_minus_FM": gap, "pred_beta1_mu_p_delta": beta1 * mu**p * d,
                     "ratio": gap / (beta1 * mu**p * d), "FM_crit_m": fx[0], "newton_res": fres})
    out["cold_gap"] = rows

    # 6. cold (c): NL-diagonal trials (h,q)=(xi'_{beta1}(m),m) at the cold point, vs phiSG
    d = 1e-6
    b = np.sqrt(4 * beta1 * mu**p * d / (A_FM - A_SG))
    be, j0 = beta1 + b, j0M + d
    sgv = phiSG(b)[0]
    mlo = 0.5 * (beta1**2) ** (-1 / (p - 2))
    ms = np.linspace(mlo, 1.0, 801)
    marg = np.array([sgv - U([m, xi1(m, beta1), m], be, j0) for m in ms])
    for rr in [0.05, 0.1, 0.2]:
        sel = np.abs(ms - mu) >= rr
        out[f"cold_c_min_margin_r{rr}"] = float(marg[sel].min()) if sel.any() else None
    out["cold_c_margin_at_M_equals_minus_C"] = "see margins; at M margin = -C_beta1(m)"
    # 7. warm
    c = 2 * j0**2
    r = c * p * mu ** (p - 1)
    Phi = E(lc(r + np.sqrt(r) * Z)) - (r + (p - 1) * c * mu**p) / 2
    out["warm_Phi_c_mu_at_delta_1e-3"] = Phi
    out["warm_pred_beta1_mu_p_delta"] = beta1 * mu**p * d
    out["c_warm"] = c
    # 8. S entries by finite differences of P(delta_q,h) at (Lc, mu), beta1
    def Pd(h, q):
        rr_ = xi1(q, beta1)
        return np.log(2) + E(lc(h + np.sqrt(rr_) * Z)) + (xi(1, beta1) - xi(q, beta1) - (1 - q) * rr_) / 2
    e = 1e-4
    Shh = (Pd(Lc + e, mu) - 2 * Pd(Lc, mu) + Pd(Lc - e, mu)) / e**2
    Sqq = (Pd(Lc, mu + e) - 2 * Pd(Lc, mu) + Pd(Lc, mu - e)) / e**2
    Shq = (Pd(Lc + e, mu + e) - Pd(Lc + e, mu - e) - Pd(Lc - e, mu + e) + Pd(Lc - e, mu - e)) / (4 * e**2)
    Om = xi2(mu, beta1)
    tau = E(np.tanh(hc) ** 3)
    out["S_fd"] = [Shh, Shq, Sqq]
    out["S_closed"] = [1 - mu, -Om * (mu - tau), Om / 2 * (1 - Om * (1 - 4 * mu + 3 * tau))]
    detS = out["S_closed"][0] * out["S_closed"][2] - out["S_closed"][1] ** 2
    out["detS"] = detS
    out["Btilde2"] = Om - out["S_closed"][2] / detS
    return out


if __name__ == "__main__":
    res = [run(int(a)) for a in sys.argv[1:]]
    print(json.dumps(res, indent=1, default=float))
