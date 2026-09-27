"""Per-p inputs (N1)-(N3) of Section 7.5 (sec:nm-fm) for p = 3..25, read from the stored kappa-cert Arb
balls; source of Table 2 (tab:nm-cert) and of C_FM > 0 (Section 7.3, sec:nm-route).

Reads ../kappa-cert/kappa_cert.json (form1 balls) and ../ball-checks/ball_checks.json (check C1).
Directed-rounding interval arithmetic (mpmath.iv, 90 digits) on the stored balls:

  R1  (N2), (N3) recomputed with S_qq taken from (Omega, mu, tau) instead of the phi_qq ball:
      det S = (1-mu) S_qq - (Omega (mu - tau))^2 > 0,  Btilde''(mu) = Omega - S_qq/det S < 0,
      Q_p margin p mu^(p-1) + (p-1) mu^p - 1 > 0.  Cross-check against ball-checks C1.
  R2  C_FM > 0, i.e. A_FM < 1/2 (the convexity bypass fails), from the C_FM and A_FM balls.
  R3  c_M = beta_c^2/2 < 2 ln 2 (consistency with the proof that c_M < 2 ln 2 < 2).
  R4  kappa_p > 0, lambda_M > 0, phi_xx > 0 re-read from the balls (already certified in kappa-cert).

Writes nm_inputs.json and prints a markdown table. Certified only relative to the kappa-cert
trust base (FLINT/Arb, its lemmas L0-L3 and coded formulas) and the NL identities
S_hh = 1-mu, S_hq = -Omega(mu-tau), S_qq = (Omega/2)(1 - Omega(1-4mu+3tau)).
"""

import hashlib
import json
import pathlib
import re

import mpmath as mp
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = pathlib.Path(__file__).resolve().parent
KC = HERE.parent / "kappa-cert" / "kappa_cert.json"
AC = HERE.parent / "ball-checks" / "ball_checks.json"
KC_SHA = "6a9899b789f6df42b3444af893f71e589b5165c07837c995e63320be79c40117"

iv = mp.iv
iv.dps = 90
BALL = re.compile(r"\[([-+0-9.eE]+) \+/- ([0-9.eE+-]+)\]")


def ball(s):
    m_, r_ = BALL.fullmatch(s.strip()).groups()
    r = iv.mpf(r_).b
    return iv.mpf(m_) + iv.mpf([-r, r])


def lo(x):
    return float(mp.mpf(x.a))


def hi(x):
    return float(mp.mpf(x.b))


sha = hashlib.sha256(KC.read_bytes()).hexdigest()
kc = json.loads(KC.read_text())
c1 = json.loads(AC.read_text())["C1"]
two_ln2 = 2 * iv.log(iv.mpf(2))
out = {"kappa_cert_sha256": sha, "kappa_cert_sha_matches_pinned": sha == KC_SHA, "per_p": {}}
ok_all = True
rows = []
for ps in sorted(kc["results"], key=int):
    p = int(ps)
    f = kc["results"][ps]["form1"]
    mu, tau, Om = ball(f["mu"]), ball(f["tau"]), ball(f["Omega"])
    beta, AFM, CFM = ball(f["beta_c"]), ball(f["A_FM"]), ball(f["C_FM"])
    kap, lam, pxx = ball(f["kappa"]), ball(f["lambda_M"]), ball(f["phi_xx"])
    Shh = 1 - mu
    Shq = -Om * (mu - tau)
    Sqq = (Om / 2) * (1 - Om * (1 - 4 * mu + 3 * tau))
    detS = Shh * Sqq - Shq**2
    red = Om - Sqq / detS
    n3 = p * mu ** (p - 1) + (p - 1) * mu**p - 1
    cM = beta**2 / 2
    r = {
        "detS_lo": lo(detS),
        "reduced_curv_hi": hi(red),
        "N3_margin_lo": lo(n3),
        "C_FM_lo": lo(CFM),
        "half_minus_A_FM_lo": lo(iv.mpf("0.5") - AFM),
        "c_M_hi": hi(cM),
        "two_ln2_minus_c_M_lo": lo(two_ln2 - cM),
        "kappa_lo": lo(kap),
        "lambda_M_lo": lo(lam),
        "phi_xx_lo": lo(pxx),
        "agrees_with_ball_checks_C1": bool(
            abs(lo(detS) - c1[ps]["detS_lo"]) <= 1e-12 * abs(c1[ps]["detS_lo"])
            and abs(hi(red) - c1[ps]["reduced_curv_hi"]) <= 1e-12 * abs(c1[ps]["reduced_curv_hi"])
            and abs(lo(n3) - c1[ps]["N3_margin_lo"]) <= 1e-12 * abs(c1[ps]["N3_margin_lo"])
        ),
    }
    ok = (
        detS.a > 0 and red.b < 0 and n3.a > 0 and CFM.a > 0 and (iv.mpf("0.5") - AFM).a > 0
        and (two_ln2 - cM).a > 0 and kap.a > 0 and lam.a > 0 and pxx.a > 0
    )
    r["all_signs_certified"] = bool(ok)
    ok_all &= bool(ok) and r["agrees_with_ball_checks_C1"]
    out["per_p"][ps] = r
    rows.append(
        f"| {p} | {mp.nstr(mp.mpf(kap.a), 6)} | {mp.nstr(mp.mpf(lam.a), 6)} | {mp.nstr(mp.mpf(pxx.a), 6)} "
        f"| {mp.nstr(mp.mpf(detS.a), 4)} | {mp.nstr(mp.mpf(red.b), 5)} | {mp.nstr(mp.mpf(n3.a), 4)} "
        f"| {mp.nstr(mp.mpf(CFM.a), 4)} |"
    )
out["all_ok"] = bool(ok_all and out["kappa_cert_sha_matches_pinned"])
out["min_detS_lo"] = min(v["detS_lo"] for v in out["per_p"].values())
out["max_reduced_curv_hi"] = max(v["reduced_curv_hi"] for v in out["per_p"].values())
out["min_N3_margin_lo"] = min(v["N3_margin_lo"] for v in out["per_p"].values())
out["min_C_FM_lo"] = min(v["C_FM_lo"] for v in out["per_p"].values())
out["max_c_M_hi"] = max(v["c_M_hi"] for v in out["per_p"].values())
(HERE / "nm_inputs.json").write_text(json.dumps(out, indent=1))
print("| p | kappa_p >= | lambda_M >= | phi_xx >= | det S >= | Btilde''(mu) <= | Q_p margin >= | C_FM >= |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(json.dumps({k: v for k, v in out.items() if k != "per_p"}, indent=1))
