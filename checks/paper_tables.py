"""Check that the values printed in the paper's tables follow from the stored outputs.

Reads expected/paper_tables.json. Each entry names a printed value, the stored output that implies it
(a JSON path or a regular expression with a group x) and a relation:
  printed<=value   the printed lower bound does not exceed the certified value
  printed>=value   the printed upper bound is not below the certified value
  printed==value   equal as rationals (or as strings)
  rounds_to        the printed digits are the correct rounding of every point of the Arb ball
  len==            the stored list has the printed length
  flag             the stored decision is true
Exact rational comparisons only (fractions, decimal); standard library.
"""

import json
import re
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

ROOT = Path(__file__).resolve().parent.parent
BALL = re.compile(r"\[([-+0-9.eE]+) \+/- ([0-9.eE+-]+)\]")


def lookup(entry):
    text = (ROOT / entry["source"]).read_text(encoding="utf-8")
    if "json" in entry:
        value = json.loads(text)
        for key in entry["json"]:
            value = value[key]
        return value
    m = re.search(entry["regex"], text)
    if m is None:
        raise ValueError(f"pattern not found in {entry['source']}")
    return m.group("x")


def frac(v):
    if isinstance(v, bool):
        raise ValueError("boolean where a number was expected")
    if isinstance(v, (int, float)):
        return Fraction(v)
    return Fraction(v.strip())


def rounds_to(printed, ball):
    m = BALL.fullmatch(ball.strip())
    mid, rad = Decimal(m.group(1)), Decimal(m.group(2))
    p = Decimal(printed)
    quantum = Decimal(1).scaleb(p.adjusted() - (len(p.as_tuple().digits) - 1))
    return all((x.quantize(quantum, rounding=ROUND_HALF_EVEN) == p) for x in (mid - rad, mid + rad))


def check(entry):
    printed, rel = entry["printed"], entry["relation"]
    value = lookup(entry)
    if rel == "printed<=value":
        return frac(printed) <= frac(value), value
    if rel == "printed>=value":
        return frac(printed) >= frac(value), value
    if rel == "printed==value":
        try:
            return frac(printed) == frac(value), value
        except (ValueError, ZeroDivisionError):
            return str(printed) == str(value), value
    if rel == "rounds_to":
        return rounds_to(printed, value), value
    if rel == "len==":
        return len(value) == int(printed), len(value)
    if rel == "flag":
        return value is True or value == "True", value
    raise ValueError(f"unknown relation {rel}")


def main():
    doc = json.loads((ROOT / "expected" / "paper_tables.json").read_text())
    bad = 0
    for entry in doc["entries"]:
        ok, value = check(entry)
        bad += not ok
        shown = value if not isinstance(value, str) or len(value) < 60 else value[:57] + "..."
        print(f"{'ok  ' if ok else 'FAIL'} {entry['where']}: printed {entry['printed']} ({entry['relation']}) {shown}")
    n = len(doc["entries"])
    print(f"{n - bad} of {n} printed values follow from the stored outputs.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
