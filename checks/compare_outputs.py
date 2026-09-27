"""Compare a re-run output file with its stored copy.

JSON files are first re-serialized with sorted keys, so that key order does not matter.  Volatile
fields (run times, peak memory, time stamps) are then masked; a mask with a group named
"v" masks only that group.  The remaining text is split into numbers and the text between them; the text must agree exactly, and the numbers must agree exactly (rtol = atol = 0) or within
the relative tolerance rtol or the absolute tolerance atol.  Standard library only.

Usage: python checks/compare_outputs.py STORED RERUN [--rtol R] [--atol A] [--mask REGEX]...
"""

import argparse
import json
import re
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

VOLATILE = [
    # JSON fields holding run times, peak memory or time stamps
    r'"(?:started|finished|seconds|scan_seconds|total_seconds|N_seconds|elapsed_s|runtime_s|total_time|'
    r'time_[A-Z]|sec|secs|maxrss_MB|peak_rss_MB|rss_MB)": (?:"[^"]*"|[-+0-9.eE]+)',
    r"'(?:maxrss_MB|peak_rss_MB|rss_MB|runtime_s|secs|seconds)': [-+0-9.eE]+",
    r"\[\s*\d+(?:\.\d+)?\s?s\]",                                  # [0.6s]
    r"(?<=[\s(,=])\d+(?:\.\d+)?\s?s(?=[\s),;.]|$)",                # 0.66s, 218s, 2.0 s
    r"(?i)\b(?:elapsed|total|done in|in|wall|runtime)[ =:]*\d+(?:\.\d+)?\s?s\b",
    r"(?i)\b(?:max|peak )?rss[ =:]*\d+(?:\.\d+)?\s?(?:MB|kB)",
    r"(?i)\b\d+(?:\.\d+)?\s?s(?= maxrss)",
]
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _blank(m):
    if "v" not in m.re.groupindex:
        return "<masked>"
    a, b = m.start("v") - m.start(), m.end("v") - m.start()
    return m.group(0)[:a] + "<masked>" + m.group(0)[b:]


def mask(text, extra=()):
    for pat in list(VOLATILE) + list(extra):
        text = re.sub(pat, _blank, text)
    return text


def canonical(text):
    try:
        return json.dumps(json.loads(text), indent=1, sort_keys=True) + "\n"
    except ValueError:
        return text


def split(text):
    nums = [m.group(0) for m in NUM.finditer(text)]
    rest = NUM.sub("#", text)
    return nums, rest


def close(a, b, rtol, atol):
    if a == b:
        return True
    if rtol == 0 and atol == 0:
        return False
    x, y = float(a), float(b)
    return abs(x - y) <= max(rtol * max(abs(x), abs(y)), atol)


def compare(stored, rerun, rtol=0.0, extra_masks=(), atol=0.0):
    """Return None if equivalent, else a short description of the first difference."""
    a_nums, a_rest = split(mask(canonical(stored), extra_masks))
    b_nums, b_rest = split(mask(canonical(rerun), extra_masks))
    if a_rest != b_rest:
        a_lines, b_lines = a_rest.splitlines(), b_rest.splitlines()
        for i, (x, y) in enumerate(zip(a_lines, b_lines)):
            if x != y:
                return f"text differs at line {i + 1}: {x[:80]!r} vs {y[:80]!r}"
        return f"text differs: {len(a_lines)} vs {len(b_lines)} lines"
    for i, (x, y) in enumerate(zip(a_nums, b_nums)):
        if not close(x, y, rtol, atol):
            return f"number {i + 1} differs: {x} vs {y}" + (f" (rtol {rtol:g}, atol {atol:g})" if rtol or atol else "")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stored")
    ap.add_argument("rerun")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--mask", action="append", default=[])
    a = ap.parse_args()
    with open(a.stored, encoding="utf-8") as f1, open(a.rerun, encoding="utf-8") as f2:
        diff = compare(f1.read(), f2.read(), a.rtol, a.mask, a.atol)
    print("same" if diff is None else diff)
    return 0 if diff is None else 1


if __name__ == "__main__":
    sys.exit(main())
