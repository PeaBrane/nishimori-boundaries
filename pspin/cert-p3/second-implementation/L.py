"""(L): landscape cover.  U(m) = beta j0 m^3 + P(mu,h) - h m.
RS  mu=d_q  : P = log2 + E log cosh(h+sqrt(v(q))g) + (v(1)-v(q))/2 - (1/2)(theta(1)-theta(q))
1R  mu=x d_0+(1-x) d_q: P = log2 + (1/x) log E cosh^x(h+sqrt(v(q))g) + (v(1)-v(q))/2 - (1/2)(theta(1)-(1-x)theta(q))
v(s)=3b^2 s^2/2, theta(s)=b^2 s^3.  U'' = 6 beta j0 m > 0 on m>0, so max over [ma,mb] is at an endpoint.
"""
import json, os, sys, time
from fractions import Fraction as Fr
from multiprocessing import Pool
from mpmath import iv
import rig, oned
from rig import ivq

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "landscape_cover.json")
THRESH = Fr(20456862, 10**7)
beta = Fr(5, 2); j0 = Fr(3841, 5000); b2 = beta * beta
v = lambda s: Fr(3, 2) * b2 * s * s
th = lambda s: b2 * s ** 3


def work(item):
    i, c = item
    ma, mb = Fr(c["ma"]), Fr(c["mb"])
    t = [Fr(z) for z in c["t"]]
    if c["kind"] == "RS":
        h, q = t
        assert 0 < q <= 1
        E, info = oned.expect("logcosh", None, h, v(q))
        P = rig.LOG2 + E + ivq((v(1) - v(q)) / 2 - (th(1) - th(q)) / 2)
    else:
        h, x, q = t
        assert 0 < x < 1 and 0 < q <= 1
        E, info = oned.expect("coshx", x, h, v(q))
        P = rig.LOG2 + iv.log(E) / ivq(x) + ivq((v(1) - v(q)) / 2 - (th(1) - (1 - x) * th(q)) / 2)
    Us = [ivq(beta * j0 * m ** 3 - h * m) + P for m in (ma, mb)]
    Uhi = max(rig.fhi(u) for u in Us)
    return dict(i=i, ma=c["ma"], mb=c["mb"], kind=c["kind"], P=[rig.flo(P), rig.fhi(P)],
                U=[[rig.flo(u), rig.fhi(u)] for u in Us], U_hi=Uhi, gen_U_hi=c["U_hi"], gen_P=c["P"],
                disc=info["disc"], tail=info["tail"], K=info["K"],
                convex=bool(ma > 0), passes=Uhi < float(THRESH) and rig.fhi(max(Us, key=lambda u: rig.fhi(u)) - ivq(THRESH)) < 0)


if __name__ == "__main__":
    t0 = time.time()
    d = json.load(open(PATH))
    assert Fr(d["beta"]) == beta and Fr(d["j0"]) == j0
    cov = d["cover"]
    # coverage check
    ivs = sorted((Fr(c["ma"]), Fr(c["mb"])) for c in cov)
    assert ivs[0][0] == Fr(1, 4) and ivs[-1][1] == 1
    gaps = [(a2, b1) for (a1, b1), (a2, b2_) in zip(ivs, ivs[1:]) if a2 > b1]
    assert all(a < b for a, b in ivs)
    with Pool(8) as p:
        res = p.map(work, list(enumerate(cov)))
    worst = max(res, key=lambda r: r["U_hi"])
    maxPdiff = max(max(abs(r["P"][0] - r["gen_P"][0]), abs(r["P"][1] - r["gen_P"][1])) for r in res)
    summary = dict(n=len(res), gaps=[[str(a), str(b)] for a, b in gaps], all_pass=all(r["passes"] for r in res),
                   certified_max_U_hi=worst["U_hi"], worst_interval=[worst["ma"], worst["mb"], worst["kind"]],
                   generator_worst=d["worst_U_hi"], max_abs_diff_P_vs_generator=maxPdiff,
                   max_disc=max(r["disc"] for r in res), max_tail=max(r["tail"] for r in res),
                   runtime_s=time.time() - t0)
    for r in res:
        print(r["i"], r["ma"], r["mb"], r["kind"], "U_hi=%.12f gen=%.12f P=[%.13f,%.13f]" % (r["U_hi"], r["gen_U_hi"], *r["P"]), r["passes"])
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, rows=res), open(os.path.join(HERE, "L_result.json"), "w"), indent=1)
