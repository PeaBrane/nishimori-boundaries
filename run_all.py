"""Verify the stored inputs and outputs and re-run the certificate programs.

    python run_all.py              SHA256SUMS check, then the light programs (a few minutes)
    python run_all.py --full       SHA256SUMS check, then every program (python-flint needed; ~15 min on 8 cores)
    python run_all.py --list       list the programs and their modes
    python run_all.py --only A,B   run only the named programs

The programs run in a copy of the repository under verification-output/, so the stored files are
never modified. Each re-run output is compared with its stored copy (checks/compare_outputs.py) and
then replaced by it, so every program runs on the stored inputs.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "checks"))
from compare_outputs import compare  # noqa: E402
from jobs import JOBS  # noqa: E402

SKIP_DIRS = {".git", ".venv", "venv", "verification-output", "__pycache__"}


def verify_hashes():
    listed = 0
    bad = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = ROOT / name
        listed += 1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    tracked = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()
               and not SKIP_DIRS.intersection(p.relative_to(ROOT).parts) and p.name != "SHA256SUMS"}
    listed_names = {line.split(maxsplit=1)[1].lstrip("*") for line in (ROOT / "SHA256SUMS").read_text().splitlines()}
    return listed, bad, sorted(tracked - listed_names)


def make_copy(dest, skip):
    def ignore(src, names):
        rel = Path(src).resolve().relative_to(ROOT)
        return [n for n in names if n in SKIP_DIRS or str(rel / n) in skip]
    shutil.copytree(ROOT, dest, ignore=ignore)


def run_job(job, work, logdir):
    cwd = work / job["dir"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TZ="UTC", **job.get("env", {}))
    out = cwd / job["stdout"] if job.get("stdout") else logdir / f"{job['name']}.stdout"
    err = cwd / job["stderr"] if job.get("stderr") else logdir / f"{job['name']}.stderr"
    t0 = time.time()
    with open(out, "w") as fo, open(err, "w") as fe:
        rc = subprocess.run([sys.executable] + job["cmd"], cwd=cwd, stdout=fo, stderr=fe, env=env).returncode
    secs = time.time() - t0
    for step, src, dst in job.get("post", []):
        assert step == "copy"
        shutil.copyfile(cwd / src, cwd / dst)
    problems = []
    if rc != job.get("rc", 0):
        problems.append(f"exit status {rc} (expected {job.get('rc', 0)}; see {err})")
    if job.get("failed_checks") is not None:
        failed = sorted(line.split()[1] for line in out.read_text().splitlines() if line.startswith("FAIL "))
        if failed != sorted(job["failed_checks"]):
            problems.append(f"failed checks {failed}, expected {sorted(job['failed_checks'])}")
    remarks = []
    outputs = job["outputs"] if "outputs" in job else [job["stdout"]]
    for rel in outputs:
        if not (ROOT / job["dir"] / rel).is_file() or not (cwd / rel).is_file():
            problems.append(f"{rel}: stored or re-run file missing")
            continue
        stored = (ROOT / job["dir"] / rel).read_text(encoding="utf-8")
        rerun = (cwd / rel).read_text(encoding="utf-8")
        rtol = {"exact": 0.0, "float": job.get("rtol", 1e-9)}.get(job["kind"], 1e-9)
        atol = job.get("atol", 1e-15) if job["kind"] == "float" else 0.0
        diff = compare(stored, rerun, rtol, job.get("masks", ()), atol)
        shutil.copyfile(ROOT / job["dir"] / rel, cwd / rel)
        if diff is None:
            continue
        if job["kind"] == "decision":
            remarks.append(f"{rel}: {diff} (search output, not compared)")
        else:
            problems.append(f"{rel}: {diff}")
    return secs, problems, remarks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="re-run every program, not only the light ones")
    ap.add_argument("--only", help="comma-separated job names")
    ap.add_argument("--list", action="store_true", help="list the jobs and exit")
    ap.add_argument("--out", help="working directory (default: verification-output/run-<time>)")
    a = ap.parse_args()
    if a.list:
        for j in JOBS:
            print(f"{j['name']:28s} {j['mode']:5s} {j['kind']:8s} {j['dir']}/{j['cmd'][0]}")
        return 0
    if a.only:
        names = a.only.split(",")
        unknown = set(names) - {j["name"] for j in JOBS}
        if unknown:
            raise SystemExit(f"unknown jobs: {sorted(unknown)}")
        jobs = [j for j in JOBS if j["name"] in names]
    else:
        jobs = [j for j in JOBS if a.full or j["mode"] == "light"]
    if any(j.get("flint") for j in jobs):
        try:
            import flint  # noqa: F401
        except ImportError:
            raise SystemExit("python-flint is required for the Arb programs: pip install python-flint==0.9.0")

    listed, bad, unlisted = verify_hashes()
    if bad:
        print(f"SHA256SUMS: {len(bad)} of {listed} files differ or are missing: {bad[:10]}")
        print("FAIL: stored files differ from SHA256SUMS.")
        return 1
    print(f"SHA256SUMS: {listed} files verified.", flush=True)
    if unlisted:
        print(f"info: {len(unlisted)} files not listed in SHA256SUMS (not part of the repository?): {unlisted[:5]}")

    work = Path(a.out) if a.out else ROOT / "verification-output" / time.strftime("run-%Y%m%d-%H%M%S")
    if work.exists():
        raise SystemExit(f"{work} exists; choose another --out")
    fresh = set()
    if a.full or a.only:
        fresh = {str(Path(j["dir"]) / f) for j in jobs for f in j.get("fresh", [])}
    make_copy(work, fresh)
    logdir = work / "run-logs"
    logdir.mkdir()
    print(f"Working copy: {work}", flush=True)

    failures = 0
    t_all = time.time()
    for job in jobs:
        secs, problems, remarks = run_job(job, work, logdir)
        status = "FAIL" if problems else "ok"
        print(f"{status:4s} {job['name']:28s} {secs:7.1f} s", flush=True)
        for p in problems:
            print(f"       {p}")
        for n in remarks:
            print(f"       info: {n}")
        failures += bool(problems)
    print(f"\nTotal time {time.time() - t_all:.0f} s.")
    if failures:
        print(f"FAIL: {failures} of {len(jobs)} programs did not reproduce their stored outputs.")
        return 1
    print(f"PASS: {listed} stored files verified; {len(jobs)} program runs matched their stored outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
