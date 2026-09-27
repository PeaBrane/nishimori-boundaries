"""Digits of the (C6) chain constant from the primary program h4_cert.py.

Runs h4_cert.py (same directory) unchanged, except that one extra line prints the exact directed-rounded
lower bounds kb_cyc (kappa-bar, two-path cycle blocks) and D_cyc = khat^2 (1 - h(V)) before they are
rounded to nearest for display. h4_cert.py prints kappa-bar >= 0.491464068 (rounded to nearest, so the
last digit is not safe); the value 0.4914640678 of Table 3 (tab:lat-inputs), input (C6), is the floor of the exact bound printed here.
Usage: python h4_cert_digits.py   (design P: b0=1000, k=10, p=9/10000, c1=2.15, c2=2.6; about 6 s)
"""
import math
import os
import sys

if sys.flags.optimize:
    raise SystemExit("Verification requires assertions: run Python without -O.")

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h4_cert.py")
src = open(SRC).read()
anchor = '    print(f"KAPPA: kbar >='
assert src.count(anchor) == 1
extra = ('    print("DIGITS kappa-bar (cycle blocks) >= floor13 =", _m.floor(kb_cyc * 10**13), "e-13;  '
         'D (cycle) >= floor12 =", _m.floor(D_cyc * 10**12), "e-12")\n')
src = src.replace(anchor, extra + anchor)
sys.argv = ["h4_cert.py", "1000", "10", "9", "10000", "2.15", "2.6"]
exec(compile(src, SRC, "exec"), {"__name__": "__main__", "_m": math})
