#!/usr/bin/env python3
"""Compare the certified Arb balls of kappa_cert.py with
  (a) Nishimori arXiv:2608.23904 printed values (Tables I, II and a few main-text numbers),
  (b) the float values of inputs/results.json (60-digit mpmath quadrature, p = 3..24; float evidence),
  (c) an earlier float evaluation, inputs/kappa_mp_result.json (12-significant-digit strings + float64 fields),
  (d) a second certificate run with different precision / truncation (all balls must overlap).

All comparisons are done in Arb: a decimal string s is read as an Arb ball containing s, and
"rounds correctly" means |x - s| < half an ulp of the printed last digit, certified for every x
in the certified ball.  Differences are reported as certified upper bounds.

Usage: python compare.py CERT.json [--cert2 CERT2.json] [--results results.json]
                         [--kappa-mp kappa_mp_result.json] [--out compare.json]
"""

import argparse
import json
import sys
from decimal import Decimal

from flint import arb, ctx

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

ctx.prec = 320

# Nishimori Table I (PDF p.7): p: (T_c, j0M, mu, Lambda_c, lambda(M))
TABLE_I = {
    3: ("0.651385", "0.767595", "0.813518", "2.3396", "0.3305"),
    4: ("0.616883", "0.810526", "0.948088", "4.4789", "0.5743"),
    5: ("0.606952", "0.823789", "0.981568", "6.2996", "0.7348"),
    6: ("0.603296", "0.828781", "0.992590", "7.9416", "0.8369"),
}
TABLE_I_KEYS = ("T_c", "j0M", "mu", "Lambda_c", "lambda_M")
# Nishimori Table II (PDF p.12): p: (A_SG, A_FM, C_SG - C_FM, K, lambda(M))
TABLE_II = {
    3: ("0.402977", "0.415872", "0.030392", "0.0433296", "0.3305"),
    4: ("0.469474", "0.474478", "0.013149", "0.0131910", "0.5743"),
    5: ("0.488151", "0.490126", "0.005364", "0.0048492", "0.7348"),
    6: ("0.494915", "0.495745", "0.002280", "0.0019756", "0.8369"),
    7: ("0.497698", "0.498062", "0.001007", "0.0008556", "0.9009"),
    8: ("0.498925", "0.499090", "0.000457", "0.0003849", "0.9406"),
    9: ("0.499489", "0.499565", "0.000212", "0.0001774", "0.9648"),
    10: ("0.499754", "0.499790", "0.0000995", "0.0000831", "0.9794"),
    11: ("0.499881", "0.499898", "0.0000472", "0.0000394", "0.9881"),
    12: ("0.499942", "0.499950", "0.0000225", "0.0000188", "0.9932"),
}
TABLE_II_KEYS = ("A_SG", "A_FM", "kappa", "K", "lambda_M")
# other printed numbers: (p, key, string, source)
MAIN_TEXT = [
    (3, "Lambda_c", "2.339642", "main text after (53)"),
    (3, "detH", "-4.010", "App. A4 before (A50)"),
    (3, "K", "0.043330", "Sec. III C, 'analytically'"),
    (4, "K", "0.013191", "Sec. III C, 'analytically'"),
    (5, "K", "0.00484917", "8-digit value of K at p = 5 (Table II prints 0.0048492)"),
]
RESULTS_KEYS = ["Lambda_c", "mu", "tau", "V", "beta_c", "T_c", "j0M", "Omega", "lambda_M", "B", "phi_mm",
                "phi_mq", "phi_qq", "detH", "phi_mb", "phi_bb", "phi_xb", "phi_xx", "A_SG", "A_FM",
                "A_FM_minus_A_SG", "kappa", "Kb", "K", "C_SG", "C_FM", "E1", "W", "vHv", "G_qq"]
KMP_FLOAT = {"Lc": "Lambda_c", "mu": "mu", "Tc": "T_c", "j0M": "j0M", "phi_xx": "phi_xx", "detH": "detH",
             "fmm": "phi_mm", "fqq": "phi_qq", "lambdaM": "lambda_M"}


def half_ulp_printed(sv):
    """Half a unit of the last printed digit of the decimal string sv, as an exact Decimal."""
    d = Decimal(sv)
    return Decimal(5).scaleb(d.as_tuple().exponent - 1)


def half_ulp_sig(sv, sig):
    """Half a unit in the sig-th significant digit of sv."""
    d = Decimal(sv)
    return Decimal(5).scaleb(d.adjusted() - sig)


def rounding_check(x, sv, half):
    """'ok' if |x - sv| < half certainly, 'disagree' if certainly >=, else 'undecided'."""
    diff = x - arb(sv)
    h = arb(str(half))
    if diff - h < 0 and diff + h > 0:
        return "ok"
    if diff - h >= 0 or diff + h <= 0:
        return "disagree"
    return "undecided"


def rel_upper(x, y):
    """Certified upper bound of |x - y| / |x| as a short string."""
    return ((x - y).abs_upper() / x.abs_lower()).upper().str(3, radius=False) if x.abs_lower() > 0 else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cert")
    ap.add_argument("--cert2")
    ap.add_argument("--results")
    ap.add_argument("--kappa-mp")
    ap.add_argument("--out", default="compare.json")
    a = ap.parse_args()
    cert = json.load(open(a.cert))["results"]
    X = {int(p): {k: arb(v) for k, v in r["form1"].items()} for p, r in cert.items()}
    out = {"tables": [], "main_text": [], "results_json": {}, "kappa_mp": [], "cross_run": {}}
    bad = 0

    for table, keys, name in ((TABLE_I, TABLE_I_KEYS, "Table I"), (TABLE_II, TABLE_II_KEYS, "Table II")):
        for p, vals in table.items():
            for k, sv in zip(keys, vals):
                st = rounding_check(X[p][k], sv, half_ulp_printed(sv))
                bad += st != "ok"
                out["tables"].append({"table": name, "p": p, "key": k, "printed": sv, "status": st})
    for p, k, sv, src in MAIN_TEXT:
        st = rounding_check(X[p][k], sv, half_ulp_printed(sv))
        bad += st != "ok"
        out["main_text"].append({"p": p, "key": k, "printed": sv, "source": src, "status": st})
    print(f"Nishimori printed values: {sum(e['status'] == 'ok' for e in out['tables'] + out['main_text'])}/"
          f"{len(out['tables']) + len(out['main_text'])} correctly rounded", flush=True)
    for e in out["tables"] + out["main_text"]:
        if e["status"] != "ok":
            print("  NOT OK:", e, flush=True)

    if a.results:
        rj = json.load(open(a.results))["values"]
        worst = {}
        for p, vals in rj.items():
            p = int(p)
            if p not in X:
                continue
            row = {}
            for k in RESULTS_KEYS:
                if k in vals:
                    row[k] = rel_upper(X[p][k], arb(vals[k]))
            out["results_json"][str(p)] = row
            wk = max(row, key=lambda k: float(row[k]) if row[k] != "n/a" else 0.0)
            worst[p] = (wk, row[wk])
        print("results.json (60-digit mpmath floats) vs certified: worst relative difference per p:", flush=True)
        for p in sorted(worst):
            print(f"  p={p:2d}: {worst[p][1]} ({worst[p][0]})", flush=True)
        out["results_json_worst"] = {str(p): {"key": w[0], "rel_diff_upper": w[1]} for p, w in worst.items()}

    if a.kappa_mp:
        km = json.load(open(a.kappa_mp))
        n_ok = n_all = 0
        for e in km:
            p = e["p"]
            row = {"p": p}
            for key, ck in (("dC", "kappa"), ("K", "K")):
                sv = e[key]
                st12 = rounding_check(X[p][ck], sv, half_ulp_sig(sv, 12))
                stpr = rounding_check(X[p][ck], sv, half_ulp_printed(sv))
                row[key] = {"string": sv, "rounds_at_12_sig": st12, "rounds_at_printed": stpr}
                n_all += 1
                n_ok += st12 == "ok"
            row["float64_rel_diff_upper"] = {fk: rel_upper(X[p][ck], arb(repr(float(e[fk]))))
                                             for fk, ck in KMP_FLOAT.items()}
            out["kappa_mp"].append(row)
        print(f"kappa_mp_result.json: {n_ok}/{n_all} dC/K strings are the correct 12-significant-digit roundings",
              flush=True)
        wf = max(float(v) for r in out["kappa_mp"] for v in r["float64_rel_diff_upper"].values())
        print(f"  worst float64-field relative difference: {wf:.2e}", flush=True)
        out["kappa_mp_float64_worst_rel"] = f"{wf:.3e}"

    if a.cert2:
        c2 = json.load(open(a.cert2))["results"]
        n = n_ov = 0
        fails = []
        for p, r in cert.items():
            if p not in c2:
                continue
            for sec in ("form1", "form2", "raw"):
                for k, v in r[sec].items():
                    n += 1
                    if arb(v).overlaps(arb(c2[p][sec][k])):
                        n_ov += 1
                    else:
                        fails.append(f"p={p} {sec}.{k}")
        out["cross_run"] = {"compared": n, "overlapping": n_ov, "non_overlapping": fails}
        bad += len(fails)
        print(f"cross-run ({a.cert2}): {n_ov}/{n} balls overlap", flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
