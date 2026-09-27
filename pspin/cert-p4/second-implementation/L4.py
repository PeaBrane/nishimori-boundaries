"""(L) p=4 cold landscape cover.  U(m) = beta j0 m^4 + P(tau,h) - h m, beta = 10/3, j0 = 8107/10000.
RS  tau = d_q:                P = log2 + E log cosh(h + sqrt(v(q)) g) + (v(1)-v(q))/2 - (1/2)(theta(1)-theta(q))
1R  tau = x d_0 + (1-x) d_q:  P = log2 + (1/x) log E cosh^x(h + sqrt(v(q)) g) + (v(1)-v(q))/2
                                   - (1/2)(theta(1) - (1-x) theta(q))
v(s) = 2 b^2 s^3, theta(s) = 3 b^2 s^4 / 2.  U''(m) = 12 beta j0 m^2 >= 0, so sup over [ma,mb] is at an endpoint.
Also reports the looser bin bound beta j0 mb^4 - h ma + P (valid for h >= 0).
Single process, sequential.  Trials are read from the primary certificate's frozen cover (../landscape4_cover.json) as candidate data only.
"""
import json
import os
import resource
import sys
import time
from fractions import Fraction as Fr

from mpmath import iv

import oned
import rig
from rig import ivq

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "landscape4_cover.json")
beta = Fr(10, 3)
j0 = Fr(8107, 10000)
b2 = beta * beta
v = lambda s: 2 * b2 * s ** 3
th = lambda s: Fr(3, 2) * b2 * s ** 4
M_A = Fr(2, 5)


def work(i, c):
    ma, mb = Fr(c["ma"]), Fr(c["mb"])
    t = [Fr(z) for z in c["t"]]
    if c["kind"] == "RS":
        h, q = t
        assert 0 < q <= 1 and h >= 0
        E, info = oned.expect("logcosh", None, h, v(q))
        P = rig.LOG2 + E + ivq((v(1) - v(q)) / 2 - (th(1) - th(q)) / 2)
    else:
        h, x, q = t
        assert 0 < x < 1 and 0 < q <= 1 and h >= 0
        E, info = oned.expect("coshx", x, h, v(q))
        P = rig.LOG2 + iv.log(E) / ivq(x) + ivq((v(1) - v(q)) / 2 - (th(1) - (1 - x) * th(q)) / 2)
    Us = [ivq(beta * j0 * m ** 4 - h * m) + P for m in (ma, mb)]
    Ubin = ivq(beta * j0 * mb ** 4 - h * ma) + P
    return dict(i=i, ma=c["ma"], mb=c["mb"], kind=c["kind"], P=[rig.flo(P), rig.fhi(P)],
                U=[[rig.flo(u), rig.fhi(u)] for u in Us], U_hi=max(rig.fhi(u) for u in Us),
                Ubin_hi=rig.fhi(Ubin), gen_U_hi=c["U_hi"], gen_P=c["P"],
                disc=info["disc"], tail=info["tail"], K=info["K"], width_E=info["width"])


if __name__ == "__main__":
    t0 = time.time()
    d = json.load(open(PATH))
    assert Fr(d["beta"]) == beta and Fr(d["j0"]) == j0
    cov = d["cover"]
    ivs = [(Fr(c["ma"]), Fr(c["mb"])) for c in cov]
    srt = sorted(ivs)
    tiling = dict(
        starts_at_MA=srt[0][0] == M_A,
        ends_at_1=srt[-1][1] == 1,
        nondegenerate=all(a < b for a, b in srt),
        contiguous=all(b1 == a2 for (_, b1), (a2, _) in zip(srt, srt[1:])),
        gaps=[[str(b1), str(a2)] for (_, b1), (a2, _) in zip(srt, srt[1:]) if a2 > b1],
        overlaps=[[str(b1), str(a2)] for (_, b1), (a2, _) in zip(srt, srt[1:]) if a2 < b1],
    )
    print("tiling", tiling, flush=True)
    res = []
    for i, c in enumerate(cov):
        r = work(i, c)
        res.append(r)
        print(r["i"], r["ma"], r["mb"], r["kind"],
              "U_hi=%.13f gen=%.13f dP=%.1e" % (r["U_hi"], r["gen_U_hi"],
                                                 max(abs(r["P"][0] - r["gen_P"][0]), abs(r["P"][1] - r["gen_P"][1]))),
              flush=True)
    worst = max(res, key=lambda r: r["U_hi"])
    worst_bin = max(res, key=lambda r: r["Ubin_hi"])
    summary = dict(
        n=len(res), tiling=tiling,
        tiling_ok=tiling["starts_at_MA"] and tiling["ends_at_1"] and tiling["nondegenerate"] and tiling["contiguous"],
        certified_max_U_hi=worst["U_hi"], worst_interval=[worst["ma"], worst["mb"], worst["kind"]],
        certified_max_binbound_hi=worst_bin["Ubin_hi"], worst_bin_interval=[worst_bin["ma"], worst_bin["mb"]],
        generator_worst=d["worst_U_hi"],
        max_abs_diff_P_vs_generator=max(max(abs(r["P"][0] - r["gen_P"][0]), abs(r["P"][1] - r["gen_P"][1])) for r in res),
        max_abs_diff_Uhi_vs_generator=max(abs(r["U_hi"] - r["gen_U_hi"]) for r in res),
        max_P_width=max(r["P"][1] - r["P"][0] for r in res),
        max_disc=max(r["disc"] for r in res), max_tail=max(r["tail"] for r in res),
        maxrss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        runtime_s=time.time() - t0)
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, rows=res), open(os.path.join(HERE, "L4_result.json"), "w"), indent=1)
